# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block-scaled quantization enums and constants.

This module is intentionally limited to enum definitions, integer constants,
and pure lookup tables. Kernel and interface implementations may import these
values, but this file must not import backend kernel modules.

The enums in this module are private kernel vocabulary:

* ``BlockScaledFormat`` - operand precision / SF-vec-size triple
  (MXFP8_E4M3 / MXFP8_E5M2 / NVFP4 / MXFP4).
* ``ScaleFactorLayout`` - in-memory layout of the scale tensor (NATURAL,
  CUBLAS_BLOCKED). Both layouts share the same per-atom byte
  swizzle, so CUBLAS_BLOCKED feeds cuBLAS, CUTLASS, and the CuTe kernel
  directly (the latter reinterprets it as a 1D
  byte stream via ``tile_atom_to_shape_SF``).
* ``BlockScaledFormatId`` - kernel API IDs for ``BlockScaledFormat``.
* ``ScaleFactorLayoutId`` - kernel API IDs for ``ScaleFactorLayout``.
* ``AxisMask`` - kernel API bitmask for axis-mode quantization.
* ``ScaleReduction`` - kernel API mode for 1D vs 2D scale reductions.
* ``BlockScaledProducer`` - kernel API IDs for fused producer variants.

Plus the block-scaled format -> ``(operand_dtype, sf_dtype, sf_vec_size)``
lookup, shared type aliases, and host-side naming helpers.
"""

import enum

import torch


class BlockScaledFormatId(enum.IntEnum):
    """Kernel API IDs for ``BlockScaledFormat`` values; do not reorder."""

    MXFP8_E4M3 = 0
    MXFP8_E5M2 = 1
    NVFP4 = 2
    MXFP4 = 3
    MXFP6_E3M2 = 4
    MXFP6_E2M3 = 5


class ScaleFactorLayoutId(enum.IntEnum):
    """Kernel API IDs for ``ScaleFactorLayout`` values; do not reorder."""

    NATURAL = 0
    CUBLAS_BLOCKED = 1


class AxisMask(enum.IntFlag):
    """Kernel API mask for selecting quantization axes."""

    M = 1
    K = 2


class ScaleReduction(enum.IntEnum):
    """Scale-reduction mode for axis-mode quantization.

    The integer values are the kernel API IDs consumed by the Triton /
    CuTe quantize kernels - do not reorder.

    Attributes:
        ONE_D: One set of scales per axis (one reduction along each
            requested axis).
        TWO_D: A single qdata tensor is shared by both axes, with a
            2D scale tile.
    """

    ONE_D = 0
    TWO_D = 1


class BlockScaledProducer(enum.IntEnum):
    """Elementwise producer evaluated before block-scaled quantization.

    The integer values are the kernel API IDs consumed by the Triton
    fused quant + producer kernels - do not reorder.

    Attributes:
        IDENTITY: No producer; quantize the source operand directly.
        SWIGLU_FWD: SwiGLU forward (``x * sigmoid(x) * y``).
        SWIGLU_BWD_DXY: SwiGLU backward producing logical
            ``concat(dx, dy)`` for the fc13 backprop path.
        ZEROCOPY_GATHER: Per-row NVLink remote-pointer gather. The source
            tile is loaded by indexing ``gather_ptrs[row]`` into peer
            symmetric-memory buffers with no transform after load. Sentinel
            pointer ``0`` zero-fills the row (matches ``zerocopy_gather``
            padding semantics). Composes with all ``axis_mask`` /
            ``scale_reduction`` modes; only ``producer_id=IDENTITY`` is
            valid alongside ``ZEROCOPY_GATHER`` since both transform the
            load-time data.
    """

    IDENTITY = 0
    SWIGLU_FWD = 1
    SWIGLU_BWD_DXY = 2
    ZEROCOPY_GATHER = 3


# GPT-OSS clamped SwiGLU defaults. Shared by the BF16 and block-scaled
# launchers so both specialize on the same values.
SWIGLU_CLAMP_ALPHA_DEFAULT = 1.702
SWIGLU_CLAMP_LIMIT_DEFAULT = 7.0

# Block-major BF16 weights store K in blocks of 64 elements: one 128-byte TMA row.
BLOCK_MAJOR_K = 64


def canonical_swiglu_clamp(
    clamped: bool,
    alpha: float,
    limit: float,
) -> tuple[bool, float, float]:
    """Return the canonical clamped-SwiGLU kernel specialization.

    Args:
        clamped: Whether to use clamped SwiGLU.
        alpha: Sigmoid multiplier for clamped SwiGLU.
        limit: Symmetric preactivation clamp limit.

    Returns:
        Kernel specialization tuple. Plain SwiGLU uses the defaults so it has
        one compilation-cache entry regardless of ignored caller values.
    """
    if not clamped:
        return False, SWIGLU_CLAMP_ALPHA_DEFAULT, SWIGLU_CLAMP_LIMIT_DEFAULT
    return True, float(alpha), float(limit)


class BlockScaledFormat(enum.StrEnum):
    """Private operand precision / SF-vec-size kernel vocabulary.

    The choice of format fixes ``(operand_dtype, sf_dtype, sf_vec_size)``
    for the kernel. This shared kernel vocabulary is broader than any one
    public operation; each operation's configuration validates the subset it
    implements.

    Attributes:
        MXFP8_E4M3: FP8 E4M3 operands, FP8_E8M0FNU scales, vec_size=32.
            Standard MoE-FFN low-precision path.
        MXFP8_E5M2: FP8 E5M2 operands, FP8_E8M0FNU scales, vec_size=32.
            Wider exponent for activations with larger dynamic range.
        NVFP4: FP4 E2M1FN operands, FP8_E4M3FN scales, vec_size=16.
            NVIDIA NVFP4 spec.
        MXFP4: FP4 E2M1FN operands, FP8_E8M0FNU scales, vec_size=32.
            OCP MXFP4 spec.
    """

    MXFP8_E4M3 = enum.auto()
    MXFP8_E5M2 = enum.auto()
    NVFP4 = enum.auto()
    MXFP4 = enum.auto()


def _kernel_block_scaled_format(
    block_scaled_format: str,
) -> BlockScaledFormat:
    """Translate a public format value to private kernel vocabulary.

    Args:
        block_scaled_format: String-valued public block-scaled operand format.

    Returns:
        Corresponding private kernel format.
    """
    return BlockScaledFormat(block_scaled_format)


_MXFP8_DIM_MULTIPLE = 128
_NVFP4_DIM_MULTIPLE = 256
_NVFP4_HIDDEN_DIM_MAX = 16384
_NVFP4_INTERMEDIATE_DIM_MAX = 8192


class ScaleFactorLayout(enum.StrEnum):
    """SF layout in memory.

    Different backends consume scales in different layouts. The
    interface accepts scales in any of these formats and re-swizzles
    on the fly when the chosen backend wants something else.

    Attributes:
        NATURAL: Per-token scales in ``[M, K // sf_vec_size]`` shape,
            row-major. This is what ``triton_to_mxfp8_dim0`` and
            ``triton_quant_mxfp8`` produce; CuTe DSL, CUTLASS, and native
            kernels convert it to their required layout.
        CUBLAS_BLOCKED: cuBLAS FP8 block-scaling layout (the
            ``(padded_rows, n_col_blocks * 16)`` swizzled form
            produced by ``to_blocked_2d`` / ``to_blocked``). The
            per-atom byte swizzle ``(row%32)*16 + (row//32)*4 + col``
            and outer M-slow / K-fast atom ordering match the cutlass
            ``BlockScaledBasicChunk`` layout, so the same byte stream
            feeds CUTLASS and ``torch._scaled_mm`` as well as the
            CuTe DSL kernel (which reinterprets it as a 1D byte stream
            via ``tile_atom_to_shape_SF``).
    """

    NATURAL = enum.auto()
    CUBLAS_BLOCKED = enum.auto()


BLOCK_SCALED_FORMAT_IDS = {
    BlockScaledFormat.MXFP8_E4M3: BlockScaledFormatId.MXFP8_E4M3.value,
    BlockScaledFormat.MXFP8_E5M2: BlockScaledFormatId.MXFP8_E5M2.value,
    BlockScaledFormat.NVFP4: BlockScaledFormatId.NVFP4.value,
    BlockScaledFormat.MXFP4: BlockScaledFormatId.MXFP4.value,
}

SCALE_FACTOR_LAYOUT_IDS = {
    ScaleFactorLayout.NATURAL: ScaleFactorLayoutId.NATURAL.value,
    ScaleFactorLayout.CUBLAS_BLOCKED: ScaleFactorLayoutId.CUBLAS_BLOCKED.value,
}

FP8_FORMATS: tuple[BlockScaledFormat, ...] = (
    BlockScaledFormat.MXFP8_E4M3,
    BlockScaledFormat.MXFP8_E5M2,
)

FP4_FORMATS: tuple[BlockScaledFormat, ...] = (
    BlockScaledFormat.NVFP4,
    BlockScaledFormat.MXFP4,
)

ELEMENTS_PER_BYTE: dict[torch.dtype, int] = {torch.float4_e2m1fn_x2: 2}


def elements_per_byte(dtype: torch.dtype) -> int:
    """Return logical elements packed into one byte of a storage dtype.

    Args:
        dtype: Storage dtype.

    Returns:
        Logical elements represented by one storage byte.
    """
    return ELEMENTS_PER_BYTE.get(dtype, 1)


MXFP8_FORMAT_IDS: tuple[int, ...] = tuple(
    BLOCK_SCALED_FORMAT_IDS[fmt] for fmt in FP8_FORMATS
)
FP4_FORMAT_IDS: tuple[int, ...] = tuple(
    BLOCK_SCALED_FORMAT_IDS[fmt] for fmt in FP4_FORMATS
)

ALL_BLOCKSCALED_FORMATS: tuple[BlockScaledFormat, ...] = FP8_FORMATS + FP4_FORMATS

# MXFP4 uses a fixed exponent shift. Keeping the policy in the format contract
# prevents fused and composed quantization from choosing different encodings.
HALF_RANGE_SCALE_DEFAULT_FORMATS: tuple[BlockScaledFormat, ...] = (
    BlockScaledFormat.MXFP4,
)

FORMAT_TAG = {
    BlockScaledFormatId.MXFP8_E4M3.value: "mxfp8_e4m3",
    BlockScaledFormatId.MXFP8_E5M2.value: "mxfp8_e5m2",
    BlockScaledFormatId.NVFP4.value: "nvfp4",
    BlockScaledFormatId.MXFP4.value: "mxfp4",
}
LAYOUT_TAG = {
    ScaleFactorLayoutId.NATURAL.value: "natural",
    ScaleFactorLayoutId.CUBLAS_BLOCKED.value: "blocked",
}
FORMAT_SF_VEC = {
    BlockScaledFormatId.MXFP8_E4M3.value: 32,
    BlockScaledFormatId.MXFP8_E5M2.value: 32,
    BlockScaledFormatId.NVFP4.value: 16,
    BlockScaledFormatId.MXFP4.value: 32,
}
PRODUCER_TAG = {
    BlockScaledProducer.IDENTITY.value: "id",
    BlockScaledProducer.SWIGLU_FWD.value: "swiglu_fwd",
    BlockScaledProducer.SWIGLU_BWD_DXY.value: "swiglu_bwd",
    BlockScaledProducer.ZEROCOPY_GATHER.value: "zerocopy_gather",
}


def build_block_tile_tag(
    axis_mask: int | None,
    scale_reduction: int | None,
    sf_vec: int,
) -> str:
    """Format the scale-factor tile shape as ``RxC``.

    Args:
        axis_mask: Axes reduced to compute one scale.
        scale_reduction: Scale-reduction policy identifier.
        sf_vec: Scale-vector width.

    Returns:
        Canonical row-by-column tile tag.
    """
    has_m = axis_mask is not None and bool(axis_mask & AxisMask.M.value)
    has_k = axis_mask is not None and bool(axis_mask & AxisMask.K.value)
    if axis_mask is None:
        return f"1x{sf_vec}"
    if has_m and has_k:
        if scale_reduction == ScaleReduction.TWO_D.value:
            return f"{sf_vec}x{sf_vec}"
        return f"{sf_vec}x1_1x{sf_vec}"
    if has_m:
        return f"{sf_vec}x1"
    if has_k:
        return f"1x{sf_vec}"
    return f"1x{sf_vec}"


# Numeric limits of the supported block-scaled element formats. Single source
# of truth: the Triton kernels mirror these as ``tl.constexpr`` and the CuTe
# kernels as ``cutlass.Constexpr[float]`` (a plain Python global cannot be read
# from inside a ``@triton.jit`` kernel, so each backend wraps these values);
# ``_quantization.py`` uses them directly.
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
FP4_E2M1_MAX = 6.0
FP4_E2M1_MAX_RECIP = 1.0 / FP4_E2M1_MAX
# Combined E4M3 scale and E2M1 data range used by NVIDIA FP4.
NVFP4_MAX = FP8_E4M3_MAX * FP4_E2M1_MAX
# Half-range exponent-shift round boundaries (enabled by ``half_range_scale``).
FP8_E4M3_HALF_RANGE_ROUND_BOUNDARY = 232.0
FP8_E5M2_HALF_RANGE_ROUND_BOUNDARY = 30720.0
FP4_E2M1_HALF_RANGE_ROUND_BOUNDARY = 3.5
# Smallest subnormal E4M3 value (2^-9); NVFP4 scales clamp here before the FP8
# cast so very small block maxes keep fine scale resolution (the E4M3 scale
# dtype supports subnormals, so there is no reason to leave that range unused).
E4M3_MIN_SUBNORMAL = 0.001953125
# Prevent a denormal-only row from overflowing the reciprocal scale.
NVFP4_TOKEN_AMAX_FLOOR = 1.0e-8
# Per-expert weight amax floor used by every NVFP4 global-scale producer.
NVFP4_GLOBAL_SCALE_EPS = 1.0e-12
# E8M0 (MX) scale-byte exponent parameters.
E8M0_EXP_BIAS = 127.0
E8M0_MIN_UNBIASED_EXP = -127.0
E8M0_MAX_UNBIASED_EXP = 127.0

# cuBLAS-blocked scale-factor "atom" swizzle geometry (the layout cuBLAS expects
# for MX scale factors): 128 rows x 4 scale-columns per atom, each atom split
# into 32 row-lanes of 16 entries. Drives the CUBLAS_BLOCKED scale-store offset
# math in both the Triton and CuTe kernels.
CUBLAS_BLOCKED_ROWS_PER_ATOM = 128
CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM = 4
CUBLAS_BLOCKED_ROW_LANES = 32
CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE = 16
CUBLAS_BLOCKED_ENTRIES_PER_ATOM = (
    CUBLAS_BLOCKED_ROWS_PER_ATOM * CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
)

# E2M1FN LUT (sign-magnitude): values 0..7 are positive, 8..15 share
# the same magnitude as 0..7 with the sign bit set.
FP4_E2M1_TABLE: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def resolve_half_range_scale(
    format: BlockScaledFormat,
    half_range_scale: bool | None,
) -> bool:
    """Resolve an optional half-range request against the format default.

    Args:
        format: Block-scaled operand format.
        half_range_scale: Explicit policy, or ``None`` for the format default.

    Returns:
        Effective half-range scale policy.
    """
    if half_range_scale is not None:
        return half_range_scale
    return format in HALF_RANGE_SCALE_DEFAULT_FORMATS


def block_scaled_format_constants(
    format: BlockScaledFormat,
) -> tuple[torch.dtype, torch.dtype, int]:
    """Return storage constants for a block-scaled format.

    Args:
        format: Public block-scaled format.

    Returns:
        Operand dtype, scale dtype, and scale-vector width.

    Raises:
        ValueError: If ``format`` is unsupported.
    """
    match format:
        case BlockScaledFormat.MXFP8_E4M3:
            return torch.float8_e4m3fn, torch.float8_e8m0fnu, 32
        case BlockScaledFormat.MXFP8_E5M2:
            return torch.float8_e5m2, torch.float8_e8m0fnu, 32
        case BlockScaledFormat.NVFP4:
            return torch.float4_e2m1fn_x2, torch.float8_e4m3fn, 16
        case BlockScaledFormat.MXFP4:
            return torch.float4_e2m1fn_x2, torch.float8_e8m0fnu, 32
        case _:
            raise ValueError(f"Unsupported block-scaled format: {format}")
