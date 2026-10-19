# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe Blackwell grouped-GEMM pipeline config derivation."""

import math
from collections.abc import Callable

import torch

from ..formats import elements_per_byte
from ._environment import num_sms_per_device

# These neutral alignment contracts remain owned by kernels/utils. Re-export
# them here so CuTe callers need a single config import site.
from ._grouped_gemm_config import (
    BLOCKSCALED_DISPATCH_DIM_ALIGNMENT,
    BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT,
    DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    uses_paged_blockscaled_scale_rows,
)

__all__ = [
    "BLOCKSCALED_DISPATCH_DIM_ALIGNMENT",
    "blockscaled_epilogue_subtile_divisor",
    "BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT",
    "DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF",
    "SmemStageFitError",
    "STATIC_FPROP_SCHEDULER_SHAPES",
    "uses_paged_blockscaled_scale_rows",
]


class SmemStageFitError(ValueError):
    """No legal pipeline depth fits the shared-memory budget."""


BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES = 232448
SMEM_LAUNCH_SAFETY_MARGIN_BYTES = 1024
SMEM_ALIGN_BYTES = 1024
TENSORMAP_SMEM_BYTES = 128 * 3
# Block-scaled adds two extra TMA descriptors (SFA, SFB) → 5 total.
TENSORMAP_SMEM_BYTES_BLOCKSCALED = 128 * 5
# GB200 fallback for callers that do not pass the device's SM count. Other
# Blackwell parts should pass ``num_sms`` explicitly.
DEFAULT_NUM_SMS = 148

# Problem-type encoding shared by scheduler visitors and grouped GEMM kernels.
_FPROP: int = 1
_DGRAD: int = 2
_WGRAD: int = 3
# Minimum cluster waves before the recommender treats a (BM, BN) tile
# size as "saturated" enough to use the larger BLOCK_N. Below this
# threshold the recommender drops to a smaller BLOCK_N to issue more
# tiles. Empirically tuned at M=128 inference shapes on GB200: under 4
# waves, the BN=128 variant (more SMEM stages, finer N tiling) lands
# 5-15% faster on memory-bound shapes; above 4 waves the per-tile
# arithmetic-density advantage of BN=256 dominates.
MIN_CLUSTER_WAVES_THRESHOLD = 4.0
# SM100 TMEM budget — total columns available to the kernel. Acc, SFA,
# and SFB all live here in block-scaled GEMMs, so the per-config buffer
# counts must keep their combined column count under this cap.
SM100_TMEM_COLS = 512
MAX_TMEM_MMA_SLOTS = 2
MAX_COMPACT_BLOCKSCALED_TMEM_MMA_SLOTS = 3
# Cutlass blockscaled SMEM/TMEM layouts always tile 4 mma-atom K-mode
# instructions per BLOCK_K, regardless of operand precision. BLOCK_K is
# sized accordingly: 128 for MXFP8 (mma atom K-mode = 32), 256 for FP4
# (K-mode = 64). The block-scaled kernel and the SF TMEM seam math
# both read this as a constant.
MMA_INST_TILE_K = 4
_SUPPORTED_DTYPES: frozenset[torch.dtype] = frozenset(
    {
        torch.bfloat16,
        torch.float16,
        torch.float32,
        # FP8 / FP4 for block-scaled.
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e8m0fnu,
        # Packed two-FP4-per-byte storage. ``dtype.itemsize`` is the
        # storage byte count; ``elements_per_byte`` accounts
        # for the two logical values packed into that byte.
        torch.float4_e2m1fn_x2,
    }
)

_NATIVE_FPROP_BLOCK_M_GRANULARITY = 64
_NATIVE_FPROP_BLOCK_N_GRANULARITY = 32
_NATIVE_FPROP_MAX_BLOCK_N = 256
_NATIVE_FPROP_BLOCK_NS = tuple(
    range(
        _NATIVE_FPROP_BLOCK_N_GRANULARITY,
        _NATIVE_FPROP_MAX_BLOCK_N + 1,
        _NATIVE_FPROP_BLOCK_N_GRANULARITY,
    )
)
_NATIVE_FPROP_MAX_1CTA_BLOCK_M = 128
_NATIVE_FPROP_MIDDLE_BLOCK_M = 256
_NATIVE_SWAP_AB_MIN_ROWS_PER_GROUP = 64
_NATIVE_SWAP_AB_MAX_ROWS_PER_GROUP = 256
_NATIVE_MULTI_MMA_MIN_SMEM_BUFFERS = 3
_NATIVE_GROUPED_GEMM_TOPOLOGIES = (
    (1, 1, 64),
    (1, 1, 128),
    (1, 2, 256),
    (2, 1, 256),
    (2, 2, 512),
)
_NATIVE_MEGA_FEATURE_TILES = (64, 128)
_MXFP4_STAGED_COMPACT_MIN_FEATURE_WAVES = 48

# Measured tuning table: MXFP4/A8W4 inference-fprop (N, K) shapes where the
# static tile scheduler beats the dynamic one (4-7% at 4K-64K tokens/rank for
# the (12288, 3072) contraction-class fc2). This is deliberately NOT a ratio rule:
# (14336, 3584) (larger contraction fc2) has the exact same N == 4 * K proportions and
# measures 3-7% *slower* static for MXFP4 in isolated A/Bs (9,111.8 vs
# 9,753.1 us at 16K tokens/rank), as does every square shape under the former
# N >= K rule (3-10%). A K-size threshold fails too: a paired sweep over
# hypothetical N == 4 * K shapes at 16K tokens/rank measured static 7-13%
# slower at every K in {2048, 2560, 3328, 3584, 4096} - including Ks smaller
# than 3072 - leaving (12288, 3072) as a singular measured winner. The
# discriminator is empirical, not geometric; extend the table only with
# fresh isolated static-vs-dynamic measurements.
#
# TODO(shikaili): TEMPORARY tuning workaround, needs cleanup. Root-cause why
# exactly this shape prefers the static scheduler (suspected tile-count /
# SM-count interaction under CuTeDSL 4.6.1) and replace the table with a
# derived policy, or delete it once the dynamic scheduler closes the gap.
# Re-validate whenever the scheduler, CuTeDSL version, or SM count changes.
STATIC_FPROP_SCHEDULER_SHAPES: frozenset[tuple[int, int]] = frozenset(
    {
        (12288, 3072),
    }
)


def derive_blockscaled_epilogue_tile_n(
    *,
    block_m: int,
    block_n: int,
    num_ctas: int,
    c_stage_dtype: torch.dtype,
) -> int:
    """Return the reference CUTLASS block-scaled epilogue N tile."""
    cta_m = block_m // num_ctas
    warp_m, warp_n = (2, 2) if (cta_m == 64 and num_ctas == 2) else (4, 1)
    c_dtype_bits = (
        8 * _dtype_size_bytes(c_stage_dtype) // elements_per_byte(c_stage_dtype)
    )
    tile_m = min(cta_m, 32 * warp_m)
    compute_elts = 8192 if c_dtype_bits == 4 else 4096
    n_perf = compute_elts // tile_m
    while block_n % n_perf != 0:
        n_perf //= 2
    tile_n = min(block_n, max(n_perf, 8 * warp_n))
    if block_n % tile_n != 0:
        tile_n = block_n
    return tile_n


EPILOGUE_SUBTILE_AUTO: int = 0


def blockscaled_epilogue_subtile_divisor(
    *,
    epilogue_subtile: int,
    c_width: int,
    a_width: int,
    full_width: bool = False,
) -> int:
    """Divisor turning ``EPILOGUE_SUBTILE`` into a physical epilogue N tile.

    The ``bytes_scale`` factor narrows the subtile when C elements are wider
    than operand elements, keeping the C SMEM stage inside the launch budget.
    It is load-bearing: dropping it on the staged path allocates 280576 bytes
    against a 232448-byte limit.

    ``full_width`` opts a path out where the budget has room. It buys fewer,
    wider epilogue passes and — because the per-warp tile drives the atom
    choice — wider ``tcgen05.ld`` and ``stmatrix`` instructions. Callers must
    use this helper for both the kernel tile and the SMEM estimate, or the
    estimate silently stops describing the allocation.
    """
    bytes_scale = 1 if full_width else max(1, c_width // a_width)
    return epilogue_subtile * bytes_scale


def derive_blockscaled_epilogue_subtile(
    *,
    block_m: int,
    block_n: int,
    num_ctas: int,
    a_dtype: torch.dtype,
    c_stage_dtype: torch.dtype,
) -> int:
    """Encode the reference epilogue tile in the legacy split field.

    NVIDIA's example first derives an epilogue ``tile_n`` via
    ``sm100_utils.compute_epilogue_tile_shape``. Our kernel stores
    ``EPILOGUE_SUBTILE`` instead and then computes
    ``tile_n = BLOCK_N / (EPILOGUE_SUBTILE * bytes_scale)``. Return zero when
    that integer encoding cannot represent the reference tile; zero selects
    the kernel's CUTLASS auto-epilogue path directly.
    """
    a_dtype_bytes = _dtype_size_bytes(a_dtype)
    c_dtype_bytes = _dtype_size_bytes(c_stage_dtype)
    a_logical_bytes_per_elem = a_dtype_bytes / elements_per_byte(a_dtype)
    bytes_scale = max(1, int(c_dtype_bytes / a_logical_bytes_per_elem))

    tile_n = derive_blockscaled_epilogue_tile_n(
        block_m=block_m,
        block_n=block_n,
        num_ctas=num_ctas,
        c_stage_dtype=c_stage_dtype,
    )

    effective_split = block_n // tile_n
    if effective_split % bytes_scale != 0:
        return EPILOGUE_SUBTILE_AUTO
    epilogue_subtile = effective_split // bytes_scale
    if epilogue_subtile <= 0 or epilogue_subtile & (epilogue_subtile - 1):
        return EPILOGUE_SUBTILE_AUTO
    return epilogue_subtile


def _blockscaled_c_stage_bytes(
    *,
    block_m: int,
    block_n: int,
    num_ctas: int,
    epilogue_subtile: int,
    a_dtype: torch.dtype,
    c_stage_dtype: torch.dtype,
    c_stage_row_skew: int = 0,
) -> int:
    cta_m = block_m // num_ctas
    a_dtype_bytes = _dtype_size_bytes(a_dtype)
    c_dtype_bytes = _dtype_size_bytes(c_stage_dtype)
    a_logical_bytes_per_elem = a_dtype_bytes / elements_per_byte(a_dtype)
    bytes_scale = max(1, int(c_dtype_bytes / a_logical_bytes_per_elem))
    c_stage_tile_m = min(cta_m, 128)
    c_stage_tile_n = (
        derive_blockscaled_epilogue_tile_n(
            block_m=block_m,
            block_n=block_n,
            num_ctas=num_ctas,
            c_stage_dtype=c_stage_dtype,
        )
        if epilogue_subtile == 0
        else block_n // (epilogue_subtile * bytes_scale)
    )
    return _align_up(
        c_stage_tile_m * (c_stage_tile_n + c_stage_row_skew) * c_dtype_bytes,
        SMEM_ALIGN_BYTES,
    )


def _blockscaled_base_smem_overhead(
    *,
    num_smem_buffers: int,
    num_tmem_buffers: int,
    num_tile_buffers: int,
    num_ctas: int,
) -> int:
    barrier_count = (
        2 * num_smem_buffers
        + 2 * num_tmem_buffers
        + 2 * num_tile_buffers
        + (3 if num_ctas == 2 else 0)
    )
    return (
        TENSORMAP_SMEM_BYTES_BLOCKSCALED + barrier_count * 8 + num_tile_buffers * 4 + 4
    )


def _blockscaled_logical_ab_totals(
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_ctas: int,
    num_smem_buffers: int,
    sf_vec_size: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
) -> tuple[int, int, int, int]:
    cta_m = block_m // num_ctas
    return (
        _blockscaled_ab_stage_storage_bytes(
            block_mn=cta_m,
            block_k=block_k,
            dtype=a_dtype,
            other_dtype=b_dtype,
        )
        * num_smem_buffers,
        _blockscaled_ab_stage_storage_bytes(
            block_mn=block_n // num_ctas,
            block_k=block_k,
            dtype=b_dtype,
            other_dtype=a_dtype,
        )
        * num_smem_buffers,
        cta_m * (block_k // sf_vec_size) * num_smem_buffers,
        (block_n // num_ctas) * (block_k // sf_vec_size) * num_smem_buffers,
    )


def _align_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _dtype_size_bytes(dtype: torch.dtype) -> int:
    """Return supported torch dtype storage size in bytes."""
    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "grouped GEMM config derivation supports only bf16/fp16/fp32, "
            f"FP8, and packed FP4 dtypes; got {dtype!r}"
        )
    return dtype.itemsize


def _ab_stage_storage_bytes(*, block_mn: int, block_k: int, dtype: torch.dtype) -> int:
    """Storage bytes for one A or B stage given a logical-element
    ``block_k``. Halves the count for packed FP4 since two logical
    elements share one storage byte."""
    return block_mn * block_k * _dtype_size_bytes(dtype) // elements_per_byte(dtype)


def _blockscaled_ab_stage_storage_bytes(
    *,
    block_mn: int,
    block_k: int,
    dtype: torch.dtype,
    other_dtype: torch.dtype,
) -> int:
    """SMEM bytes for one block-scaled operand stage.

    Mixed-width ``mxf8f6f4`` MMA uses TMA's narrow-to-U8 unpack mode, so a
    packed FP4 operand occupies one byte per logical element in SMEM.
    """
    if dtype == torch.float4_e2m1fn_x2 and dtype != other_dtype:
        return block_mn * block_k
    return _ab_stage_storage_bytes(
        block_mn=block_mn,
        block_k=block_k,
        dtype=dtype,
    )


def _estimate_ab_stage_bytes(
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_ctas: int,
    split_m_across_ctas: bool = True,
    a_dtype: torch.dtype = torch.bfloat16,
    b_dtype: torch.dtype = torch.bfloat16,
) -> int:
    cta_m = block_m // num_ctas if split_m_across_ctas else block_m
    a_stage_bytes = cta_m * block_k * _dtype_size_bytes(a_dtype)
    b_stage_bytes = (block_n // num_ctas) * block_k * _dtype_size_bytes(b_dtype)
    return _align_up(a_stage_bytes + b_stage_bytes, SMEM_ALIGN_BYTES)


def _estimate_c_stage_bytes(
    *,
    block_m: int,
    block_n: int,
    num_ctas: int,
    epilogue_subtile: int,
    c_stage_row_skew: int,
    split_m_across_ctas: bool = True,
    c_stage_dtype: torch.dtype = torch.bfloat16,
) -> int:
    cta_m = block_m // num_ctas if split_m_across_ctas else block_m
    c_stage_tile_m = min(cta_m, 128)
    c_stage_tile_n = block_n // epilogue_subtile
    return _align_up(
        c_stage_tile_m
        * (c_stage_tile_n + c_stage_row_skew)
        * _dtype_size_bytes(c_stage_dtype),
        SMEM_ALIGN_BYTES,
    )


def estimate_grouped_gemm_smem_bytes(
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_ctas: int,
    num_smem_buffers: int,
    num_tmem_buffers: int,
    num_tile_buffers: int,
    epilogue_subtile: int,
    num_c_stages: int | None = None,
    c_stage_row_skew: int = 0,
    split_m_across_ctas: bool = True,
    a_dtype: torch.dtype = torch.bfloat16,
    b_dtype: torch.dtype = torch.bfloat16,
    c_stage_dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Conservative CuTe per-CTA SMEM estimate for config selection."""
    if num_c_stages is None:
        num_c_stages = num_tmem_buffers
    ab_stage_bytes = _estimate_ab_stage_bytes(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_ctas=num_ctas,
        split_m_across_ctas=split_m_across_ctas,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
    )
    c_stage_bytes = _estimate_c_stage_bytes(
        block_m=block_m,
        block_n=block_n,
        num_ctas=num_ctas,
        epilogue_subtile=epilogue_subtile,
        c_stage_row_skew=c_stage_row_skew,
        split_m_across_ctas=split_m_across_ctas,
        c_stage_dtype=c_stage_dtype,
    )

    barrier_count = (
        2 * num_smem_buffers
        + 2 * num_tmem_buffers
        + 2 * num_tile_buffers
        + (3 if num_ctas == 2 else 0)
    )
    return (
        num_smem_buffers * ab_stage_bytes
        + num_c_stages * c_stage_bytes
        + TENSORMAP_SMEM_BYTES
        + barrier_count * 8
        + num_tile_buffers * 4
        + 4
    )


def derive_num_tmem_buffers(*, block_n: int, num_mmas: int) -> int:
    """Use all TMEM columns available to native grouped-GEMM accumulators.

    Native kernels can use up to 16 stages for narrow tiles because no scale
    factors share TMEM. Block-scaled derivation separately caps its MMA ring at
    ``MAX_TMEM_MMA_SLOTS`` while budgeting accumulator, SFA, and SFB columns.
    """
    if block_n <= 0:
        raise ValueError(f"block_n must be positive, got {block_n}")
    if num_mmas <= 0:
        raise ValueError(f"num_mmas must be positive, got {num_mmas}")
    capacity = SM100_TMEM_COLS // (block_n * num_mmas)
    if capacity < 1:
        raise ValueError(
            "one native grouped-GEMM accumulator stage exceeds SM100 TMEM: "
            f"block_n={block_n}, num_mmas={num_mmas}"
        )
    return 1 << (capacity.bit_length() - 1)


def _valid_epilogue_subtiles(block_n: int) -> tuple[int, ...]:
    return tuple(
        split
        for split in (1, 2, 4, 8)
        if block_n % split == 0
        and block_n // split >= 8
        and (block_n // split) % 8 == 0
    )


def grouped_gemm_epilogue_tile_n(config: dict[str, int]) -> int:
    """Return the physical N width of one native C-staging subtile."""
    return config["BLOCK_SIZE_N"] // config["EPILOGUE_SUBTILE"]


def grouped_gemm_epilogue_tile_is_compatible(
    config: dict[str, int],
    multiple: int,
    *,
    swap_ab: bool = False,
) -> bool:
    if multiple <= 0:
        raise ValueError(f"epilogue_tile_n_multiple must be positive, got {multiple}")
    physical_tile_n = (
        config["BLOCK_SIZE_M"] if swap_ab else grouped_gemm_epilogue_tile_n(config)
    )
    return physical_tile_n % multiple == 0


def _filter_epilogue_tile_n_multiple(
    configs: tuple[dict[str, int], ...],
    multiple: int | None,
    *,
    swap_ab: bool = False,
) -> tuple[dict[str, int], ...]:
    if multiple is None:
        return configs
    compatible = tuple(
        config
        for config in configs
        if grouped_gemm_epilogue_tile_is_compatible(
            config,
            multiple,
            swap_ab=swap_ab,
        )
    )
    if not compatible:
        raise ValueError(
            "no grouped GEMM config satisfies the epilogue tile width multiple "
            f"{multiple}"
        )
    return compatible


def _select_grouped_gemm_pipeline_depths(
    *,
    num_ctas: int,
    num_mmas: int,
    block_m: int,
    block_n: int,
    block_k: int,
    c_stage_row_skew: int,
    num_tmem_buffers: int,
    num_tile_buffers: int,
    tma_store_epilogue: bool,
    split_m_across_ctas: bool,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_stage_dtype: torch.dtype,
) -> tuple[int, int, int]:
    budget = BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES - SMEM_LAUNCH_SAFETY_MARGIN_BYTES
    ab_stage_bytes = _estimate_ab_stage_bytes(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_ctas=num_ctas,
        split_m_across_ctas=split_m_across_ctas,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
    )
    max_smem_buffers = max(2, BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES // ab_stage_bytes)
    min_c_stages = 2 if tma_store_epilogue else 1
    viable: list[tuple[int, int, int]] = []
    for num_smem_buffers in range(max_smem_buffers, 1, -1):
        for epilogue_subtile in _valid_epilogue_subtiles(block_n):
            smem_bytes = estimate_grouped_gemm_smem_bytes(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_ctas=num_ctas,
                num_smem_buffers=num_smem_buffers,
                num_tmem_buffers=num_tmem_buffers,
                num_tile_buffers=num_tile_buffers,
                num_c_stages=min_c_stages,
                epilogue_subtile=epilogue_subtile,
                c_stage_row_skew=c_stage_row_skew,
                split_m_across_ctas=split_m_across_ctas,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                c_stage_dtype=c_stage_dtype,
            )
            if smem_bytes > budget:
                continue
            c_stage_bytes = _estimate_c_stage_bytes(
                block_m=block_m,
                block_n=block_n,
                num_ctas=num_ctas,
                epilogue_subtile=epilogue_subtile,
                c_stage_row_skew=c_stage_row_skew,
                split_m_across_ctas=split_m_across_ctas,
                c_stage_dtype=c_stage_dtype,
            )
            extra_c_stages = (budget - smem_bytes) // c_stage_bytes
            num_c_stages = (
                min(8, min_c_stages + extra_c_stages)
                if tma_store_epilogue
                else min_c_stages
            )
            viable.append((num_smem_buffers, num_c_stages, epilogue_subtile))
    if not viable:
        raise ValueError(
            "no grouped GEMM pipeline config fits the Blackwell SMEM budget for "
            f"{num_ctas=} {num_mmas=} {block_m=} {block_n=} {block_k=}"
        )
    if tma_store_epilogue and num_mmas > 1:
        balanced = tuple(
            candidate
            for candidate in viable
            if candidate[0] >= _NATIVE_MULTI_MMA_MIN_SMEM_BUFFERS
        )
        if balanced:
            num_smem_buffers, _, epilogue_subtile = min(
                balanced,
                key=lambda candidate: (candidate[2], -candidate[0]),
            )
            return num_smem_buffers, min_c_stages, epilogue_subtile
    return viable[0]


def derive_grouped_gemm_config(
    *,
    num_ctas: int,
    num_mmas: int,
    block_m: int,
    block_n: int = 256,
    block_k: int = 64,
    c_stage_row_skew: int | None = None,
    tma_store_epilogue: bool = True,
    split_m_across_ctas: bool = False,
    a_dtype: torch.dtype = torch.bfloat16,
    b_dtype: torch.dtype = torch.bfloat16,
    acc_dtype: torch.dtype = torch.float32,
    c_stage_dtype: torch.dtype | None = None,
) -> dict[str, int]:
    """Choose CuTe grouped-GEMM depths from geometry and resource limits."""
    if c_stage_row_skew is None:
        c_stage_row_skew = 8 if num_mmas == 2 else 0
    if c_stage_dtype is None:
        c_stage_dtype = a_dtype
    _ = _dtype_size_bytes(acc_dtype)
    num_tmem_buffers = derive_num_tmem_buffers(
        block_n=block_n,
        num_mmas=num_mmas,
    )
    num_tile_buffers = num_tmem_buffers + 1
    # A/B and C use independent SMEM rings. Prefer deeper A/B overlap, then
    # the fewest epilogue subtiles; spend any remaining SMEM on C stages.
    # Mega's direct-store/scatter epilogue uses one synchronous C stage.
    num_smem_buffers, num_c_stages, epilogue_subtile = (
        _select_grouped_gemm_pipeline_depths(
            num_ctas=num_ctas,
            num_mmas=num_mmas,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            c_stage_row_skew=c_stage_row_skew,
            num_tmem_buffers=num_tmem_buffers,
            num_tile_buffers=num_tile_buffers,
            tma_store_epilogue=tma_store_epilogue,
            split_m_across_ctas=split_m_across_ctas,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_stage_dtype=c_stage_dtype,
        )
    )
    return {
        "NUM_CTAS": num_ctas,
        "NUM_MMAS": num_mmas,
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "NUM_SMEM_BUFFERS": num_smem_buffers,
        "NUM_C_STAGES": num_c_stages,
        "NUM_TMEM_BUFFERS": num_tmem_buffers,
        "NUM_TILE_BUFFERS": num_tile_buffers,
        "EPILOGUE_SUBTILE": epilogue_subtile,
    }


def grouped_gemm_config_name(config: dict[str, int]) -> str:
    """Return the canonical geometry name for a complete native config.

    Operand dtype is intentionally not encoded; callers must resolve the name
    in the matching BF16/F16 or FP32 registry.
    """
    name = (
        f"{config['NUM_CTAS']}cta{config['NUM_MMAS']}mma_"
        f"bm{config['BLOCK_SIZE_M']}_bn{config['BLOCK_SIZE_N']}"
    )
    if config["BLOCK_SIZE_K"] != 64:
        name += f"_bk{config['BLOCK_SIZE_K']}"
    return name


def _register_grouped_gemm_config(
    configs: dict[str, dict[str, int]],
    config: dict[str, int],
) -> None:
    name = grouped_gemm_config_name(config)
    if name in configs:
        raise RuntimeError(
            f"duplicate native grouped-GEMM config name {name!r}: "
            f"existing={configs[name]!r}, duplicate={config!r}"
        )
    configs[name] = config


def build_grouped_gemm_configs() -> dict[str, dict[str, int]]:
    """Build every supported native CuTe grouped-GEMM launch config."""
    configs = {}
    for num_ctas, num_mmas, block_m in _NATIVE_GROUPED_GEMM_TOPOLOGIES:
        for block_n in _NATIVE_FPROP_BLOCK_NS:
            config = derive_grouped_gemm_config(
                num_ctas=num_ctas,
                num_mmas=num_mmas,
                block_m=block_m,
                block_n=block_n,
                split_m_across_ctas=num_ctas > 1,
            )
            _register_grouped_gemm_config(configs, config)
    for block_m in _NATIVE_MEGA_FEATURE_TILES:
        for block_n in _NATIVE_FPROP_BLOCK_NS:
            config = derive_grouped_gemm_config(
                num_ctas=1,
                num_mmas=1,
                block_m=block_m,
                block_n=block_n,
                block_k=128,
                tma_store_epilogue=False,
            )
            _register_grouped_gemm_config(configs, config)
    return configs


def derive_auto_grouped_gemm_fprop_config(
    *,
    GM: int,
    G: int,
    N: int | None,
    num_sms: int | None = None,
    epilogue_tile_n_multiple: int | None = None,
) -> dict[str, int]:
    """Derive a complete BF16/F16 inference FPROP launch config."""
    if GM < 0:
        raise ValueError(f"GM must be nonnegative, got {GM}")
    if G <= 0:
        raise ValueError(f"G must be positive, got {G}")
    if N is not None and N <= 0:
        raise ValueError(f"N must be positive when specified, got {N}")

    m_per_group = max(1, _ceil_div(GM, G))
    active_groups = min(G, max(1, GM))
    block_m = min(
        512,
        _align_up(m_per_group, _NATIVE_FPROP_BLOCK_M_GRANULARITY),
    )
    if block_m <= _NATIVE_FPROP_MAX_1CTA_BLOCK_M:
        # Size for ~2 standard deviations over the average rows per expert (about
        # Poisson), so busy experts don't re-read weights in a second M tile:
        # 49 -> 64 rows, 50 -> 128. Cost: a 128-row tile has fewer smem stages
        # (bm64_bn128 -> bm128_bn256 goes from 9 to 4).
        busiest_rows = m_per_group + math.ceil(2 * math.sqrt(m_per_group))
        block_m = min(
            _NATIVE_FPROP_MAX_1CTA_BLOCK_M,
            _align_up(busiest_rows, _NATIVE_FPROP_BLOCK_M_GRANULARITY),
        )
    else:
        block_m = (
            _NATIVE_FPROP_MIDDLE_BLOCK_M
            if block_m <= _NATIVE_FPROP_MIDDLE_BLOCK_M
            else 512
        )
    num_ctas = 1 if block_m <= _NATIVE_FPROP_MAX_1CTA_BLOCK_M else 2
    num_mmas = 1 if block_m <= _NATIVE_FPROP_MIDDLE_BLOCK_M else 2
    derived = _derive_auto_grouped_gemm_fprop_config(
        active_groups=active_groups,
        m_per_group=m_per_group,
        N=N,
        num_sms=num_sms,
        num_ctas=num_ctas,
        num_mmas=num_mmas,
        block_m=block_m,
        epilogue_tile_n_multiple=None,
    )
    if epilogue_tile_n_multiple is None or grouped_gemm_epilogue_tile_is_compatible(
        derived,
        epilogue_tile_n_multiple,
    ):
        return derived
    return _derive_auto_grouped_gemm_fprop_config(
        active_groups=active_groups,
        m_per_group=m_per_group,
        N=N,
        num_sms=num_sms,
        num_ctas=num_ctas,
        num_mmas=num_mmas,
        block_m=block_m,
        epilogue_tile_n_multiple=epilogue_tile_n_multiple,
    )


def derive_auto_grouped_gemm_swap_ab_fprop_config(
    *,
    rows_per_group: int,
    output_dim: int,
) -> dict[str, int] | None:
    """Derive native inference SWAP_AB geometry when it reduces weight rereads."""
    if rows_per_group <= _NATIVE_SWAP_AB_MIN_ROWS_PER_GROUP:
        return None
    if rows_per_group > _NATIVE_SWAP_AB_MAX_ROWS_PER_GROUP:
        return None
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}")
    if output_dim % _NATIVE_FPROP_MAX_1CTA_BLOCK_M != 0:
        return None

    token_tile = max(
        128,
        _align_up(rows_per_group, _NATIVE_FPROP_BLOCK_N_GRANULARITY),
    )
    return derive_grouped_gemm_config(
        num_ctas=1,
        num_mmas=1,
        block_m=_NATIVE_FPROP_MAX_1CTA_BLOCK_M,
        block_n=token_tile,
    )


def _select_mega_feature_tile(
    *,
    hidden_dim: int,
    intermediate_dim: int,
    active_experts: int,
    num_clusters: int,
) -> int:
    candidates = tuple(
        block_m
        for block_m in (64, 128)
        if hidden_dim % block_m == 0 and 2 * intermediate_dim % block_m == 0
    )
    if not candidates:
        raise ValueError(
            "native Mega dimensions require a common 64-row feature tile; "
            f"got hidden_dim={hidden_dim}, intermediate_dim={intermediate_dim}"
        )

    def tile_count(block_m: int) -> int:
        return active_experts * (
            _ceil_div(2 * intermediate_dim, block_m) + _ceil_div(hidden_dim, block_m)
        )

    largest = candidates[-1]
    if tile_count(largest) >= num_clusters:
        return largest
    return min(
        candidates,
        key=lambda block_m: (
            abs(tile_count(block_m) - num_clusters),
            -block_m,
        ),
    )


def derive_auto_grouped_gemm_mega_config(
    *,
    rows_per_active_expert: int,
    hidden_dim: int,
    intermediate_dim: int,
    active_experts: int,
    num_sms: int | None = None,
    epilogue_tile_n_multiple: int | None = None,
) -> dict[str, int]:
    """Derive a complete SWAP_AB BF16/F16 native MegaMoE config."""
    if rows_per_active_expert <= 0:
        raise ValueError(
            f"rows_per_active_expert must be positive, got {rows_per_active_expert}"
        )
    if active_experts <= 0:
        raise ValueError(f"active_experts must be positive, got {active_experts}")
    if hidden_dim <= 0 or intermediate_dim <= 0:
        raise ValueError(
            "native Mega dimensions must be positive; "
            f"got hidden_dim={hidden_dim}, intermediate_dim={intermediate_dim}"
        )

    token_tile = min(
        _NATIVE_FPROP_MAX_BLOCK_N,
        _align_up(rows_per_active_expert, _NATIVE_FPROP_BLOCK_N_GRANULARITY),
    )
    num_clusters = max(1, DEFAULT_NUM_SMS if num_sms is None else num_sms)
    feature_tile = _select_mega_feature_tile(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        active_experts=active_experts,
        num_clusters=num_clusters,
    )
    derived = derive_grouped_gemm_config(
        num_ctas=1,
        num_mmas=1,
        block_m=feature_tile,
        block_n=token_tile,
        block_k=128,
        tma_store_epilogue=False,
    )
    if epilogue_tile_n_multiple is None or grouped_gemm_epilogue_tile_is_compatible(
        derived,
        epilogue_tile_n_multiple,
        swap_ab=True,
    ):
        return derived
    candidates = _grouped_gemm_fprop_candidates(
        N=None,
        num_ctas=1,
        num_mmas=1,
        block_m=feature_tile,
        block_k=128,
        tma_store_epilogue=False,
        epilogue_tile_n_multiple=epilogue_tile_n_multiple,
        swap_ab=True,
    )

    feature_tiles = active_experts * (
        _ceil_div(2 * intermediate_dim, feature_tile)
        + _ceil_div(hidden_dim, feature_tile)
    )

    def tile_count(config: dict[str, int]) -> int:
        return feature_tiles * _ceil_div(
            rows_per_active_expert,
            config["BLOCK_SIZE_N"],
        )

    return min(
        candidates,
        key=lambda config: (
            abs(tile_count(config) - num_clusters),
            abs(config["BLOCK_SIZE_N"] - token_tile),
            -config["BLOCK_SIZE_N"],
        ),
    )


def _grouped_gemm_fprop_candidates(
    *,
    N: int | None,
    num_ctas: int,
    num_mmas: int,
    block_m: int,
    block_k: int = 64,
    tma_store_epilogue: bool = True,
    epilogue_tile_n_multiple: int | None = None,
    swap_ab: bool = False,
) -> tuple[dict[str, int], ...]:
    compatible_block_ns = tuple(
        block_n for block_n in _NATIVE_FPROP_BLOCK_NS if N is None or N % block_n == 0
    )
    preferred_block_ns = compatible_block_ns or _NATIVE_FPROP_BLOCK_NS

    def configs(block_ns: tuple[int, ...]) -> tuple[dict[str, int], ...]:
        return tuple(
            derive_grouped_gemm_config(
                num_ctas=num_ctas,
                num_mmas=num_mmas,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                tma_store_epilogue=tma_store_epilogue,
                split_m_across_ctas=num_ctas > 1,
            )
            for block_n in block_ns
        )

    preferred = tuple(
        config
        for config in configs(preferred_block_ns)
        if epilogue_tile_n_multiple is None
        or grouped_gemm_epilogue_tile_is_compatible(
            config,
            epilogue_tile_n_multiple,
            swap_ab=swap_ab,
        )
    )
    if preferred:
        return preferred
    return _filter_epilogue_tile_n_multiple(
        configs(_NATIVE_FPROP_BLOCK_NS),
        epilogue_tile_n_multiple,
        swap_ab=swap_ab,
    )


def _derive_auto_grouped_gemm_fprop_config(
    *,
    active_groups: int,
    m_per_group: int,
    N: int | None,
    num_sms: int | None,
    num_ctas: int,
    num_mmas: int,
    block_m: int,
    epilogue_tile_n_multiple: int | None,
) -> dict[str, int]:
    candidates = _grouped_gemm_fprop_candidates(
        N=N,
        num_ctas=num_ctas,
        num_mmas=num_mmas,
        block_m=block_m,
        epilogue_tile_n_multiple=epilogue_tile_n_multiple,
    )

    target_smem_stages = max(2, _ceil_div(SM100_TMEM_COLS, block_m))
    deeper_pipeline = tuple(
        config
        for config in candidates
        if config["NUM_SMEM_BUFFERS"] >= target_smem_stages
    )
    if deeper_pipeline:
        candidates = deeper_pipeline
    largest = candidates[-1]
    if N is None:
        return largest

    num_clusters = max(
        1,
        (DEFAULT_NUM_SMS if num_sms is None else num_sms) // num_ctas,
    )

    def tile_count(config: dict[str, int]) -> int:
        return (
            active_groups
            * _ceil_div(m_per_group, block_m)
            * _ceil_div(N, config["BLOCK_SIZE_N"])
        )

    if 2 * tile_count(largest) >= num_clusters:
        return largest
    return min(
        candidates,
        key=lambda config: (
            abs(tile_count(config) - num_clusters),
            -config["BLOCK_SIZE_N"],
        ),
    )


def _make_fp32_grouped_gemm_config(
    *,
    num_ctas: int,
    num_mmas: int,
    block_m: int,
    block_n: int,
) -> dict[str, int]:
    return derive_grouped_gemm_config(
        num_ctas=num_ctas,
        num_mmas=num_mmas,
        block_m=block_m,
        block_n=block_n,
        a_dtype=torch.float32,
        b_dtype=torch.float32,
        acc_dtype=torch.float32,
        c_stage_row_skew=8 if num_mmas == 2 else 0,
        split_m_across_ctas=True,
    )


GROUPED_GEMM_CONFIGS = build_grouped_gemm_configs()

_FP32_GROUPED_GEMM_CONFIG_SPECS: tuple[tuple[int, int, int, int], ...] = (
    (1, 1, 64, 128),
    (2, 1, 256, 256),
)
FP32_GROUPED_GEMM_CONFIGS = {
    grouped_gemm_config_name(config): config
    for num_ctas, num_mmas, block_m, block_n in _FP32_GROUPED_GEMM_CONFIG_SPECS
    for config in (
        _make_fp32_grouped_gemm_config(
            num_ctas=num_ctas,
            num_mmas=num_mmas,
            block_m=block_m,
            block_n=block_n,
        ),
    )
}

_LEGACY_GROUPED_GEMM_CONFIG_NAMES = {
    "1cta1mma": "1cta1mma_bm128_bn256",
    "1cta2mma": "1cta2mma_bm256_bn256",
    "2cta1mma": "2cta1mma_bm256_bn256",
    "2cta2mma": "2cta2mma_bm512_bn256",
}
GROUPED_GEMM_CONFIGS.update(
    {
        legacy_name: GROUPED_GEMM_CONFIGS[canonical_name]
        for legacy_name, canonical_name in _LEGACY_GROUPED_GEMM_CONFIG_NAMES.items()
    }
)

_TRAINING_FPROP_DGRAD_CONFIG = "training_2cta2mma_bm512_bn256"
_TRAINING_WGRAD_CONFIG = "training_2cta1mma_bm256_bn256"
_TRAINING_LARGE_BATCH_WGRAD_CONFIG = "training_wgrad_2cta2mma_bm512_bn256"
GROUPED_GEMM_CONFIGS.update(
    {
        _TRAINING_FPROP_DGRAD_CONFIG: {
            "NUM_CTAS": 2,
            "NUM_MMAS": 2,
            "BLOCK_SIZE_M": 512,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "NUM_SMEM_BUFFERS": 4,
            "NUM_TMEM_BUFFERS": 1,
            "NUM_TILE_BUFFERS": 2,
            "EPILOGUE_SUBTILE": 4,
        },
        # The single accumulator buffer leaves the epilogue exposed; a second
        # C stage overlaps its TMA stores with the next tile's first atom.
        _TRAINING_LARGE_BATCH_WGRAD_CONFIG: {
            "NUM_CTAS": 2,
            "NUM_MMAS": 2,
            "BLOCK_SIZE_M": 512,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "NUM_SMEM_BUFFERS": 4,
            "NUM_C_STAGES": 2,
            "NUM_TMEM_BUFFERS": 1,
            "NUM_TILE_BUFFERS": 2,
            "EPILOGUE_SUBTILE": 4,
        },
        _TRAINING_WGRAD_CONFIG: {
            "NUM_CTAS": 2,
            "NUM_MMAS": 1,
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "NUM_SMEM_BUFFERS": 6,
            "NUM_TMEM_BUFFERS": 2,
            "NUM_TILE_BUFFERS": 3,
            "EPILOGUE_SUBTILE": 4,
        },
    }
)

DEFAULT_GROUPED_GEMM_CONFIG = "2cta2mma_bm512_bn256"
DEFAULT_FP32_GROUPED_GEMM_CONFIG = "2cta1mma_bm256_bn256"
DEFAULT_GROUPED_GEMM_WGRAD_CONFIG = "2cta1mma_bm256_bn256"


# At or below this many routed rows per rank (tokens * topk), native training
# runs the BM256 tile for every GEMM instead of BM512. Measured end-to-end on
# GB200 EP4 fwd+bwd with 4 and 32 local experts (square and contraction shapes): BM256
# FPROP/DGRAD is 3-21% faster at <=8192 rows and 1.5-4% slower at >=16384
# rows; GB300 EP4 WGRAD follows the same split.
_TRAINING_SMALL_BATCH_MAX_ROUTED_ROWS = 8192
# FPROP/DGRAD M tiles span one local expert's rows, so below one BM512 tile of
# rows per expert every tile runs partly empty. GB300 EP4 E256 fwd+bwd at 256
# rows per expert: BM256 FPROP/DGRAD is 27-34% faster; at 512 BM512 still wins.
# WGRAD tiles span the weight dims; rows per expert only set their K loop.
_TRAINING_LARGE_BATCH_BLOCK_M = GROUPED_GEMM_CONFIGS[_TRAINING_FPROP_DGRAD_CONFIG][
    "BLOCK_SIZE_M"
]
# The FC2 combine FPROP runs BM256 with two TMEM buffers from this many routed
# rows per local expert. GB300 GALA (D=9216, I=2304) 8-layer EP8-64, both tiles
# alternating in one job: 16384 rows -1.2..-2.1%; 2048-8192 rows +5..+9%.
_COMBINE_FPROP_BM256_MIN_ROWS_PER_GROUP = 16384


def grouped_gemm_training_config(
    *,
    wgrad: bool = False,
    routed_rows: int | None = None,
    num_groups: int = 1,
    combine_fprop: bool = False,
) -> str:
    """Native training config name.

    `routed_rows` is the expected routed rows per rank (tokens * topk), spread
    over `num_groups` local experts; `None` keeps the large-batch tiles.
    `combine_fprop` selects the FC2 combine FPROP tile.
    """
    if routed_rows is not None and (
        routed_rows <= _TRAINING_SMALL_BATCH_MAX_ROUTED_ROWS
        or (not wgrad and routed_rows < num_groups * _TRAINING_LARGE_BATCH_BLOCK_M)
    ):
        return _TRAINING_WGRAD_CONFIG
    if wgrad:
        return _TRAINING_LARGE_BATCH_WGRAD_CONFIG
    if (
        combine_fprop
        and routed_rows is not None
        and routed_rows >= num_groups * _COMBINE_FPROP_BM256_MIN_ROWS_PER_GROUP
    ):
        return _TRAINING_WGRAD_CONFIG
    return _TRAINING_FPROP_DGRAD_CONFIG


def registered_grouped_gemm_config_name(config: dict[str, int]) -> str:
    name = grouped_gemm_config_name(config)
    registered = GROUPED_GEMM_CONFIGS.get(name)
    if registered != config:
        raise RuntimeError(
            "derived grouped GEMM config is not registered exactly: "
            f"name={name!r}, derived={config!r}, registered={registered!r}"
        )
    return name


def grouped_gemm_configs_for_dtype(
    dtype: torch.dtype,
) -> dict[str, dict[str, int]]:
    if dtype == torch.float32:
        return FP32_GROUPED_GEMM_CONFIGS
    return GROUPED_GEMM_CONFIGS


def resolve_grouped_gemm_config(
    *,
    requested_config: str | None,
    gm: int,
    groups: int,
    is_fprop: bool,
    n: int | None = None,
    num_sms: int | None = None,
    dtype: torch.dtype | None = None,
    epilogue_tile_n_multiple: int | None = None,
) -> str:
    if dtype == torch.float32:
        if requested_config is None:
            return DEFAULT_FP32_GROUPED_GEMM_CONFIG
        if requested_config not in FP32_GROUPED_GEMM_CONFIGS:
            raise ValueError(
                f"config {requested_config!r} has no FP32 variant; FP32 supports "
                f"{sorted(FP32_GROUPED_GEMM_CONFIGS)}"
            )
        return requested_config
    if requested_config is not None:
        return requested_config
    if not is_fprop:
        return DEFAULT_GROUPED_GEMM_CONFIG
    return registered_grouped_gemm_config_name(
        derive_auto_grouped_gemm_fprop_config(
            GM=gm,
            G=groups,
            N=n,
            num_sms=num_sms,
            epilogue_tile_n_multiple=epilogue_tile_n_multiple,
        )
    )


# ---------------------------------------------------------------------------
# Block-scaled (MXFP8 / NVFP4) grouped GEMM pipeline config derivation.
# Same shape as ``derive_grouped_gemm_config`` but accounts for the extra
# SFA/SFB SMEM staging *and* the SM100 TMEM column budget — block-scaled
# kernels co-locate the accumulator, SFA, and SFB in TMEM, so the config
# must keep their combined column count under ``SM100_TMEM_COLS``.
# ---------------------------------------------------------------------------

# Block-scaled SF atom layout — see
# ``cutlass.utils.blockscaled_layout.BlockScaledBasicChunk``: K-major MMA
# packs SF into 128-byte atoms covering 128 logical M (or N) rows × 4
# K-blocks of width ``sf_vec_size``. So the per-stage SF SMEM cost for an
# (mn × k) cluster tile is ``mn × (k // sf_vec_size)`` bytes — the SF dtype
# is always 1 byte for MXFP8/NVFP4. SF is small in absolute terms (KB) but
# we still account for it when picking pipeline depths.


def _estimate_sf_stage_bytes(*, block_mn: int, block_k: int, sf_vec_size: int) -> int:
    """Per-stage SFA *or* SFB SMEM cost for the cluster tile."""
    return block_mn * (block_k // sf_vec_size)


def _estimate_blockscaled_ab_stage_bytes(
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_ctas: int,
    sf_vec_size: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
) -> int:
    """A + B + SFA + SFB per CTA stage.

    The block-scaled 2CTA layout splits A/SFA along M and B/SFB along N for
    the per-CTA SMEM allocation. This matches the stage counts the kernel can
    actually launch; counting full B/SFB per CTA underestimates the legal AB
    pipeline depth on memory-bound BN=128 shapes.
    """
    cta_m = block_m // num_ctas
    a_stage_bytes = _blockscaled_ab_stage_storage_bytes(
        block_mn=cta_m,
        block_k=block_k,
        dtype=a_dtype,
        other_dtype=b_dtype,
    )
    b_stage_bytes = _blockscaled_ab_stage_storage_bytes(
        block_mn=block_n // num_ctas,
        block_k=block_k,
        dtype=b_dtype,
        other_dtype=a_dtype,
    )
    sfa_stage_bytes = _estimate_sf_stage_bytes(
        block_mn=cta_m, block_k=block_k, sf_vec_size=sf_vec_size
    )
    sfb_stage_bytes = _estimate_sf_stage_bytes(
        block_mn=block_n // num_ctas, block_k=block_k, sf_vec_size=sf_vec_size
    )
    return _align_up(
        a_stage_bytes + b_stage_bytes + sfa_stage_bytes + sfb_stage_bytes,
        SMEM_ALIGN_BYTES,
    )


def estimate_blockscaled_grouped_gemm_smem_bytes(
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_ctas: int,
    num_smem_buffers: int,
    num_tmem_buffers: int,
    num_tile_buffers: int,
    num_c_stages: int,
    epilogue_subtile: int,
    sf_vec_size: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_stage_dtype: torch.dtype,
    c_stage_row_skew: int = 0,
) -> int:
    """Per-CTA SMEM estimate for block-scaled grouped GEMM. Mirrors
    ``estimate_grouped_gemm_smem_bytes`` but counts SFA/SFB stages and the
    five-tensormap (A/B/SFA/SFB/C) descriptor area.

    ``A``, ``B``, ``SFA``, and ``SFB`` are each laid out as a *separate*
    1024-aligned SMEM range in the kernel's ``SharedStorage`` struct; they
    are NOT packed into one combined per-stage block. Each region's
    cosize spans all ``num_smem_buffers`` stages (stages live inside the
    aligned region). Padding therefore lands at the end of each region —
    not after every stage — which is what ``cute.cosize`` × dtype-bytes
    naturally produces.

    ``num_c_stages`` is intentionally independent of ``num_tmem_buffers``.
    The reference starts with two C SMEM stages, derives AB/SF stages from the
    remaining budget, then spends leftover SMEM on extra C stages.

    The C-stage subtile width additionally honors the kernel's
    ``bytes_scale`` factor (``c_dtype_bytes // a_dtype_bytes``, floor 1):
    when the output dtype is wider than the A operand (typical for
    ``fp8 → bf16``), the kernel reduces ``epi_tile_n`` by that factor
    so each subtile still produces the same byte volume per epilogue
    pass. Forgetting ``bytes_scale`` here over-counted ``c_stage_bytes``
    by 2× for MXFP8 → bf16 and previously caused the helper to reject
    valid configs (e.g. ``NUM_SMEM=6`` at EPI=4).
    """
    a_total_bytes, b_total_bytes, sfa_total_bytes, sfb_total_bytes = (
        _blockscaled_logical_ab_totals(
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_ctas=num_ctas,
            num_smem_buffers=num_smem_buffers,
            sf_vec_size=sf_vec_size,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
        )
    )
    c_stage_bytes = _blockscaled_c_stage_bytes(
        block_m=block_m,
        block_n=block_n,
        num_ctas=num_ctas,
        epilogue_subtile=epilogue_subtile,
        a_dtype=a_dtype,
        c_stage_dtype=c_stage_dtype,
        c_stage_row_skew=c_stage_row_skew,
    )

    return (
        _align_up(a_total_bytes, SMEM_ALIGN_BYTES)
        + _align_up(b_total_bytes, SMEM_ALIGN_BYTES)
        + _align_up(sfa_total_bytes, SMEM_ALIGN_BYTES)
        + _align_up(sfb_total_bytes, SMEM_ALIGN_BYTES)
        + c_stage_bytes * num_c_stages
        + _blockscaled_base_smem_overhead(
            num_smem_buffers=num_smem_buffers,
            num_tmem_buffers=num_tmem_buffers,
            num_tile_buffers=num_tile_buffers,
            num_ctas=num_ctas,
        )
    )


def estimate_blockscaled_tmem_cols(
    *,
    block_m: int,
    block_n: int,
    num_mmas: int,
    num_tmem_buffers: int,
    overlapping_accum: bool = False,
) -> int:
    """Estimated TMEM column footprint for block-scaled GEMM.

    Three regions co-exist in TMEM:

      * Acc — ``BLOCK_N * num_mmas`` cols per stage for FP32 accumulator on
        tcgen05 atoms (each M-axis MMA atom occupies BLOCK_N FP32 cols).
      * SFA — small constant: 4 cols per K MMA-iteration tile, 1 stage.
      * SFB — ``BLOCK_N // 32`` cols, except BN32 uses a 128-row SFB atom.

    The SM100 hardware budget is ``SM100_TMEM_COLS = 512``. If acc stages
    push the combined footprint past this, the s2t copy from the MMA warp
    writes OOB and the kernel IMAs (this is exactly the BLOCK_N=256 +
    NUM_TMEM_BUFFERS=2 failure mode tracked in
    ``blockscaled_grouped_gemm.py``).

    With ``overlapping_accum=True`` the two acc stages SHARE the last
    ``num_sf_tmem_cols`` of TMEM with the SF region — stage 0 occupies
    [0..BLOCK_N), stage 1 occupies [BLOCK_N - num_sf_tmem_cols ..
    2*BLOCK_N - num_sf_tmem_cols), and SF lives at [acc_total..
    acc_total + num_sf_tmem_cols]. Cutlass dense kernel uses this trick
    to get 2 effective acc stages at BLOCK_N=256 — see
    ``dense_blockscaled_gemm_persistent.py:368-389``.
    """
    # SFA / SFB TMEM layout (matches cutlass ``dense_blockscaled_gemm_
    # persistent`` and our kernel's ``make_tmem_layout_sf{a,b}``
    # partition):
    #   num_sfa_tmem_cols = (BM // sf_atom_mn) * mma_inst_tile_k
    #   num_sfb_tmem_cols = (sfb_tile_n // sf_atom_mn) * mma_inst_tile_k
    # ``mma_inst_tile_k`` is the cutlass invariant 4 atoms-per-BLOCK_K
    # for every block-scaled format on Blackwell — BLOCK_K is sized to
    # be 4× the mma-atom K-mode (32 for MXFP8, 64 for FP4). The kernel
    # hardcodes the same value at the SF-seam / early-release logic
    # (search for ``mma_inst_tile_k = 4``). Don't recompute as
    # ``block_k // sf_vec_size`` — that's coincidentally 4 for MXFP8
    # (128/32) but 8/16 for FP4, leading to a 2-4× over-count of SF
    # TMEM cols.
    sf_atom_mn = 32
    mma_inst_tile_k = MMA_INST_TILE_K
    sfa_cols = (block_m // sf_atom_mn) * mma_inst_tile_k
    sfb_tile_n = 128 if block_n == 32 else block_n
    sfb_cols = (sfb_tile_n // sf_atom_mn) * mma_inst_tile_k
    sf_cols = sfa_cols + sfb_cols
    if overlapping_accum:
        if num_mmas != 1:
            raise ValueError("overlapping_accum does not support multiple M atoms")
        # Stage 0 + stage 1 overlap by sf_cols at the seam:
        # acc_cols = block_n * 2 - sf_cols. Then SF lives at the end.
        if num_tmem_buffers != 2:
            raise ValueError(
                "overlapping_accum requires num_tmem_buffers=2 (the trick is "
                "specifically about double-buffering the accumulator)"
            )
        acc_cols = block_n * 2 - sf_cols
    else:
        acc_cols = block_n * num_mmas * num_tmem_buffers
    return acc_cols + sf_cols


def derive_blockscaled_num_tmem_buffers(
    *,
    block_m: int,
    block_n: int,
    num_mmas: int,
    overlapping_accum: bool = False,
    max_tmem_buffers: int | None = None,
) -> int:
    """Choose the deepest accumulator pipeline that fits SM100 TMEM."""
    if max_tmem_buffers is None:
        max_tmem_buffers = MAX_TMEM_MMA_SLOTS
        if block_n <= 64 and not overlapping_accum:
            max_tmem_buffers = MAX_COMPACT_BLOCKSCALED_TMEM_MMA_SLOTS
    if max_tmem_buffers <= 0:
        raise ValueError(f"max_tmem_buffers must be positive, got {max_tmem_buffers}")
    for candidate in range(max_tmem_buffers, 0, -1):
        cols = estimate_blockscaled_tmem_cols(
            block_m=block_m,
            block_n=block_n,
            num_mmas=num_mmas,
            num_tmem_buffers=candidate,
            overlapping_accum=overlapping_accum and candidate == 2,
        )
        if cols <= SM100_TMEM_COLS:
            return candidate
    raise AssertionError(
        f"no block-scaled TMEM stage count fits BLOCK_M={block_m}, BLOCK_N={block_n}"
    )


def _blockscaled_staged_token_tile(
    config: dict[str, int | bool],
    *,
    format_name: str,
    token_tile: int,
    hidden_dim: int | None,
    intermediate_dim: int | None,
    num_local_experts: int | None,
    num_sms: int | None,
) -> int:
    if (
        format_name != "mxfp4"
        or token_tile > 64
        or None in (hidden_dim, intermediate_dim, num_local_experts, num_sms)
    ):
        return token_tile
    assert hidden_dim is not None
    assert intermediate_dim is not None
    assert num_local_experts is not None
    assert num_sms is not None
    block_m = int(config["BLOCK_SIZE_M"])
    feature_tiles = _ceil_div(2 * intermediate_dim, block_m) + _ceil_div(
        hidden_dim, block_m
    )
    num_clusters = max(num_sms // int(config["NUM_CTAS"]), 1)
    feature_waves = _ceil_div(feature_tiles * num_local_experts, num_clusters)
    if feature_waves < _MXFP4_STAGED_COMPACT_MIN_FEATURE_WAVES:
        return int(config["BLOCK_SIZE_N"])
    return token_tile


def mixed_blockscaled_tile_prefers_deep_pipeline(
    *,
    a_width: int,
    b_width: int,
    block_n: int,
    block_k: int,
) -> bool:
    return a_width != b_width and block_n * a_width >= block_k * b_width


def _blockscaled_inference_feature_candidates(
    *,
    config: dict[str, int | bool] | None,
    token_tile: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_ctas: int,
    retune_config: Callable[[dict[str, int | bool]], dict[str, int | bool]] | None,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for block_m in (256, 512, 768):
        if (
            hidden_dim % block_m != 0
            or 2 * intermediate_dim % block_m != 0
            or estimate_blockscaled_tmem_cols(
                block_m=block_m,
                block_n=token_tile,
                num_mmas=block_m // (num_ctas * 128),
                num_tmem_buffers=1,
            )
            > SM100_TMEM_COLS
        ):
            continue
        candidate_config = dict(config or {})
        candidate_config["BLOCK_SIZE_M"] = block_m
        candidate_config["NUM_MMAS"] = block_m // (num_ctas * 128)
        if retune_config is not None:
            try:
                candidate_config = retune_config(candidate_config)
            except SmemStageFitError:
                continue
        candidates.append((block_m, int(candidate_config.get("NUM_SMEM_BUFFERS", 0))))
    return candidates


def narrow_blockscaled_inference_token_tile(
    config: dict[str, int | bool],
    *,
    weight_bits: int,
    num_tokens_per_expert: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_local_experts: int,
    num_sms: int | None,
) -> dict[str, int | bool]:
    """Trade token reuse for independent accumulators on short packed-weight grids."""
    if (
        weight_bits != 4
        or not config.get("NATIVE_MMA_TAILS", False)
        or int(config["BLOCK_SIZE_N"]) <= 128
        or num_sms is None
    ):
        return config
    feature_tiles = min(
        _ceil_div(hidden_dim, int(config["BLOCK_SIZE_M"])),
        _ceil_div(2 * intermediate_dim, int(config["BLOCK_SIZE_M"])),
    )
    token_tiles = _ceil_div(num_tokens_per_expert, int(config["BLOCK_SIZE_N"]))
    clusters = max(num_sms // int(config["NUM_CTAS"]), 1)
    if feature_tiles * token_tiles * num_local_experts >= 2 * clusters:
        return config
    return {
        **config,
        "BLOCK_SIZE_N": 128,
        "MMA_ATOM_N": 128,
        "OVERLAPPING_ACCUM": False,
        "PIPELINE_CHUNK_ROWS": 128,
    }


def _blockscaled_inference_wave_cost(
    block_m: int,
    *,
    token_tiles: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_local_experts: int,
    num_clusters: int,
) -> tuple[int, int]:
    fc13_tiles = (
        _ceil_div(2 * intermediate_dim, block_m) * token_tiles * num_local_experts
    )
    fc2_tiles = _ceil_div(hidden_dim, block_m) * token_tiles * num_local_experts
    waves = _ceil_div(fc13_tiles, num_clusters) + _ceil_div(fc2_tiles, num_clusters)
    return waves * block_m, -block_m


def _select_blockscaled_inference_feature_tile(
    candidates: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    token_tiles: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_local_experts: int,
    num_clusters: int,
    prefer_deeper_pipeline: bool,
    prioritize_pipeline_depth: bool,
) -> int:
    def wave_cost(block_m: int) -> tuple[int, int]:
        return _blockscaled_inference_wave_cost(
            block_m,
            token_tiles=token_tiles,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_local_experts=num_local_experts,
            num_clusters=num_clusters,
        )

    if not prefer_deeper_pipeline:
        return min((block_m for block_m, _ in candidates), key=wave_cost)
    best_cost = min(wave_cost(block_m)[0] for block_m, _ in candidates)
    max_near_optimal_cost = _ceil_div(best_cost * 21, 20)
    near_optimal = (
        candidate
        for candidate in candidates
        if wave_cost(candidate[0])[0] <= max_near_optimal_cost
    )
    return max(
        near_optimal,
        key=lambda candidate: (
            candidate[1] if prioritize_pipeline_depth else 0,
            candidate[0] * candidate[1],
            -candidate[0],
        ),
    )[0]


def _blockscaled_inference_feature_tile(
    *,
    staged: bool,
    token_tile: int,
    num_tokens_per_expert: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_local_experts: int,
    num_sms: int,
    num_ctas: int,
    config: dict[str, int | bool] | None = None,
    retune_config: Callable[[dict[str, int | bool]], dict[str, int | bool]]
    | None = None,
    prefer_deeper_pipeline: bool = False,
    prioritize_pipeline_depth: bool = False,
) -> int:
    candidates = _blockscaled_inference_feature_candidates(
        config=config,
        token_tile=token_tile,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_ctas=num_ctas,
        retune_config=retune_config,
    )
    if not candidates:
        raise ValueError(
            "no supported inference feature tile divides both hidden dimensions; "
            f"got hidden_dim={hidden_dim}, intermediate_dim={intermediate_dim}"
        )

    # Compact staged tiles require one feature-axis MMA atom. Mega uses
    # independent pipeline synchronization and supports wider feature tiles.
    if staged and token_tile in (32, 64):
        candidates = tuple(
            candidate for candidate in candidates if candidate[0] <= num_ctas * 128
        )
        if not candidates:
            raise ValueError(
                "compact staged tiles require a single MMA atom along "
                f"the feature axis; got num_ctas={num_ctas}"
            )
    wave_args = {
        "token_tiles": _ceil_div(num_tokens_per_expert, token_tile),
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "num_local_experts": num_local_experts,
        "num_clusters": max(num_sms // num_ctas, 1),
    }
    selection_args = dict(
        **wave_args,
        prefer_deeper_pipeline=prefer_deeper_pipeline,
        prioritize_pipeline_depth=prioritize_pipeline_depth,
    )
    selected = _select_blockscaled_inference_feature_tile(candidates, **selection_args)
    if (
        estimate_blockscaled_tmem_cols(
            block_m=selected,
            block_n=token_tile,
            num_mmas=selected // (num_ctas * 128),
            num_tmem_buffers=2,
        )
        <= SM100_TMEM_COLS
    ):
        return selected
    selected_cost = _blockscaled_inference_wave_cost(selected, **wave_args)[0]
    buffered_ties = [
        candidate
        for candidate in candidates
        if _blockscaled_inference_wave_cost(candidate[0], **wave_args)[0]
        == selected_cost
        and estimate_blockscaled_tmem_cols(
            block_m=candidate[0],
            block_n=token_tile,
            num_mmas=candidate[0] // (num_ctas * 128),
            num_tmem_buffers=2,
        )
        <= SM100_TMEM_COLS
    ]
    return (
        _select_blockscaled_inference_feature_tile(buffered_ties, **selection_args)
        if buffered_ties
        else selected
    )


def derive_blockscaled_inference_geometry(
    seed: dict[str, int | bool],
    *,
    format_name: str,
    staged: bool,
    num_tokens_per_expert: int | None,
    hidden_dim: int | None,
    intermediate_dim: int | None,
    num_local_experts: int | None,
    num_sms: int | None,
    retune_config: Callable[[dict[str, int | bool]], dict[str, int | bool]]
    | None = None,
    prefer_deeper_pipeline: bool = False,
    pipeline_depth_widths: tuple[int, int] | None = None,
    max_token_tile: int | None = None,
    routing_capacity: int | None = None,
) -> dict[str, int | bool]:
    """Apply compact inference tile geometry once."""
    config = dict(seed)
    if num_tokens_per_expert is None or num_tokens_per_expert <= 0:
        return config

    token_tile = min(256, _align_up(num_tokens_per_expert, 32))
    # Leave headroom for routing imbalance before a second tile rereads weights.
    if format_name == "nvfp4" and staged:
        token_tile = max(64, token_tile)
    if max_token_tile is not None:
        token_tile = min(token_tile, max_token_tile)
    if staged:
        token_tile = _blockscaled_staged_token_tile(
            config,
            format_name=format_name,
            token_tile=token_tile,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_local_experts=num_local_experts,
            num_sms=num_sms,
        )

    if routing_capacity is not None and routing_capacity % token_tile:
        minimum = 64 if format_name == "nvfp4" and staged else 32
        maximum = min(256, max_token_tile if max_token_tile is not None else 256)
        candidates = [
            width
            for width in range(minimum, maximum + 1, 32)
            if routing_capacity % width == 0
        ]
        if not candidates:
            raise ValueError("no supported token tile divides the routing capacity")
        token_tile = min(
            candidates,
            key=lambda width: (
                _ceil_div(num_tokens_per_expert, width),
                _ceil_div(num_tokens_per_expert, width) * width,
            ),
        )

    config["BLOCK_SIZE_N"] = token_tile
    if token_tile >= 96 and token_tile % 128 != 0:
        config["EPILOGUE_SUBTILE"] = EPILOGUE_SUBTILE_AUTO
    if None not in (hidden_dim, intermediate_dim, num_local_experts, num_sms):
        assert hidden_dim is not None
        assert intermediate_dim is not None
        assert num_local_experts is not None
        assert num_sms is not None
        num_ctas = int(config["NUM_CTAS"])
        prioritize_pipeline_depth = False
        if pipeline_depth_widths is not None:
            a_width, b_width = pipeline_depth_widths
            prioritize_pipeline_depth = mixed_blockscaled_tile_prefers_deep_pipeline(
                a_width=a_width,
                b_width=b_width,
                block_n=token_tile,
                block_k=int(config["BLOCK_SIZE_K"]),
            )
        feature_tile = _blockscaled_inference_feature_tile(
            config=config,
            staged=staged,
            token_tile=token_tile,
            num_tokens_per_expert=num_tokens_per_expert,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_local_experts=num_local_experts,
            num_sms=num_sms,
            num_ctas=num_ctas,
            retune_config=retune_config,
            prefer_deeper_pipeline=prefer_deeper_pipeline,
            prioritize_pipeline_depth=prioritize_pipeline_depth,
        )
        config["BLOCK_SIZE_M"] = feature_tile
        config["NUM_MMAS"] = feature_tile // (num_ctas * 128)
        feature_tiles = min(
            _ceil_div(hidden_dim, feature_tile),
            _ceil_div(2 * intermediate_dim, feature_tile),
        )
        token_tiles = _ceil_div(num_tokens_per_expert, token_tile)
        if (
            not staged
            and token_tile > 128
            and feature_tiles * token_tiles * num_local_experts
            < max(num_sms // num_ctas, 1)
        ):
            return derive_blockscaled_inference_geometry(
                seed,
                format_name=format_name,
                staged=staged,
                num_tokens_per_expert=num_tokens_per_expert,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_local_experts=num_local_experts,
                num_sms=num_sms,
                retune_config=retune_config,
                prefer_deeper_pipeline=prefer_deeper_pipeline,
                pipeline_depth_widths=pipeline_depth_widths,
                max_token_tile=128,
                routing_capacity=routing_capacity,
            )

    return config


def initialize_blockscaled_inference_pipeline(
    config: dict[str, int | bool],
) -> None:
    """Initialize launchable depths after compact geometry is final."""
    config["NUM_TMEM_BUFFERS"] = derive_blockscaled_num_tmem_buffers(
        block_m=int(config["BLOCK_SIZE_M"]),
        block_n=int(config["BLOCK_SIZE_N"]),
        num_mmas=int(config["NUM_MMAS"]),
        overlapping_accum=bool(config.get("OVERLAPPING_ACCUM", False)),
    )
    config["NUM_TILE_BUFFERS"] = int(config["NUM_TMEM_BUFFERS"]) + 1
    config["NUM_SMEM_BUFFERS"] = 2
    config["NUM_C_STAGES"] = 1
    if "MEGA_NUM_SMEM_BUFFERS" in config:
        config["MEGA_NUM_SMEM_BUFFERS"] = 2
    if "PIPELINE_CHUNK_ROWS" in config:
        config["PIPELINE_CHUNK_ROWS"] = math.lcm(128, int(config["BLOCK_SIZE_N"]))


def derive_blockscaled_grouped_gemm_config(
    *,
    num_ctas: int,
    block_m: int,
    block_n: int,
    num_mmas: int = 1,
    block_k: int = 128,
    sf_vec_size: int = 32,
    a_dtype: torch.dtype = torch.float8_e4m3fn,
    b_dtype: torch.dtype = torch.float8_e4m3fn,
    c_stage_dtype: torch.dtype = torch.bfloat16,
    epilogue_subtile: int | None = None,
    overlapping_accum: bool = False,
    swap_ab: bool = False,
    kloop_unroll: int = 2,
) -> dict[str, int]:
    """Choose block-scaled grouped-GEMM pipeline depths.

    Picks ``NUM_TMEM_BUFFERS`` first (constrained by the 512-col TMEM
    budget — see ``estimate_blockscaled_tmem_cols``), derives
    ``EPILOGUE_SUBTILE`` from the reference epilogue-tile heuristic unless
    explicitly overridden, then mirrors NVIDIA's stage-count structure:
    maximize AB/SF stages, start C at two stages, and spend leftover SMEM
    on extra C stages.

    ``num_mmas`` is the per-cluster atom count along M (mirrors the bf16
    grouped-GEMM helper). Blockscaled tcgen05 supports atom M=128 for 1 CTA
    and atom M in {128, 256} for 2 CTAs, so wider tiles require stacking
    atoms. The kernel sets ``atom_m = BLOCK_M // num_mmas``.
    """
    if num_mmas < 1:
        raise ValueError(f"num_mmas must be >= 1, got {num_mmas}")
    atom_m = block_m // num_mmas
    if block_m % num_mmas != 0:
        raise ValueError(f"block_m={block_m} must be divisible by num_mmas={num_mmas}")
    valid_atom_m = {128} if num_ctas == 1 else {128, 256}
    if atom_m not in valid_atom_m:
        raise ValueError(
            f"atom_m = block_m / num_mmas = {atom_m} is not in {sorted(valid_atom_m)} "
            f"(num_ctas={num_ctas}, block_m={block_m}, num_mmas={num_mmas}); "
            f"cutlass blockscaled MMA atom-M is capped per CTA group"
        )
    # Pick the largest useful acc-stage count whose acc + SFA + SFB fits TMEM.
    # Compact token tiles admit a third stage; wider tiles follow CUTLASS's
    # two-stage cap. Multiple M atoms consume independent TMEM columns.
    # When ``overlapping_accum`` is opted in, both stages share the seam region
    # with the SF block so we get 2 effective acc stages even at BN=256.
    num_tmem_buffers = derive_blockscaled_num_tmem_buffers(
        block_m=block_m,
        block_n=block_n,
        num_mmas=num_mmas,
        overlapping_accum=overlapping_accum,
    )
    if overlapping_accum and num_tmem_buffers != 2:
        raise AssertionError(
            f"overlapping_accum requires num_tmem_buffers=2, but the layout "
            f"already accommodates 2 stages without overlap "
            f"(BLOCK_M={block_m}, BLOCK_N={block_n}); use the non-overlapping "
            f"config instead"
        )
    num_tile_buffers = num_tmem_buffers + 1
    if epilogue_subtile is None:
        epilogue_subtile = derive_blockscaled_epilogue_subtile(
            block_m=block_m,
            block_n=block_n,
            num_ctas=num_ctas,
            a_dtype=a_dtype,
            c_stage_dtype=c_stage_dtype,
        )
    elif epilogue_subtile < 0 or (
        epilogue_subtile > 0 and epilogue_subtile & (epilogue_subtile - 1)
    ):
        raise ValueError(
            "epilogue_subtile must be zero or a positive power of two, "
            f"got {epilogue_subtile}"
        )
    elif epilogue_subtile > 0:
        a_logical_bytes_per_elem = _dtype_size_bytes(a_dtype) / elements_per_byte(
            a_dtype
        )
        bytes_scale = max(
            1, int(_dtype_size_bytes(c_stage_dtype) / a_logical_bytes_per_elem)
        )
        effective_split = epilogue_subtile * bytes_scale
        if block_n % effective_split != 0:
            raise ValueError(
                f"BLOCK_N={block_n} must be divisible by epilogue_subtile "
                f"({epilogue_subtile}) * bytes_scale ({bytes_scale})"
            )
    ab_stage_bytes = _estimate_blockscaled_ab_stage_bytes(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_ctas=num_ctas,
        sf_vec_size=sf_vec_size,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
    )
    max_smem_buffers = max(2, BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES // ab_stage_bytes)

    c_stage_bytes = _blockscaled_c_stage_bytes(
        block_m=block_m,
        block_n=block_n,
        num_ctas=num_ctas,
        epilogue_subtile=epilogue_subtile,
        a_dtype=a_dtype,
        c_stage_dtype=c_stage_dtype,
    )

    best: tuple[int, int] | None = None
    for num_smem_buffers in range(max_smem_buffers, 1, -1):
        base_smem_bytes = estimate_blockscaled_grouped_gemm_smem_bytes(
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_ctas=num_ctas,
            num_smem_buffers=num_smem_buffers,
            num_tmem_buffers=num_tmem_buffers,
            num_tile_buffers=num_tile_buffers,
            num_c_stages=2,
            epilogue_subtile=epilogue_subtile,
            sf_vec_size=sf_vec_size,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_stage_dtype=c_stage_dtype,
        )
        budget = BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES - SMEM_LAUNCH_SAFETY_MARGIN_BYTES
        if base_smem_bytes <= budget:
            extra_c_stages = (budget - base_smem_bytes) // c_stage_bytes
            best = (num_smem_buffers, 2 + extra_c_stages)
            break
    if best is None:
        raise SmemStageFitError(
            f"no block-scaled SMEM stage count fits BLOCK_M={block_m}, "
            f"BLOCK_N={block_n}, BLOCK_K={block_k}"
        )
    num_smem_buffers, num_c_stages = best

    return {
        "NUM_CTAS": num_ctas,
        "NUM_MMAS": num_mmas,
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "NUM_SMEM_BUFFERS": num_smem_buffers,
        "NUM_C_STAGES": num_c_stages,
        "NUM_TMEM_BUFFERS": num_tmem_buffers,
        "NUM_TILE_BUFFERS": num_tile_buffers,
        "EPILOGUE_SUBTILE": epilogue_subtile,
        "OVERLAPPING_ACCUM": overlapping_accum,
        # kSwapAB: when True the kernel re-roles its MMA-A slot to load
        # the weight tensor (so 2cta cluster multicast lands on W
        # instead of X) and its MMA-B slot to load the activation
        # tensor. ``BLOCK_SIZE_M`` then tiles W's N-axis and
        # ``BLOCK_SIZE_N`` tiles the activation M-axis. The host wrapper
        # leaves caller inputs untouched; the kernel itself rebuilds
        # TMA atoms with the swapped roles and uses a swap-aware per-
        # group size helper that flips which axis ``split_sizes[g]``
        # advances along — supports varlen activation M.
        "SWAP_AB": swap_ab,
        # K-tile producer/MMA loop unroll factor. Empirically tuned per
        # production config: u=1 wins on high-SMEM/OV paths (P0) and on
        # the no-swap NVFP4 K<N path (P2); u=2 is the default elsewhere.
        # See ``BlockScaledKnobs`` in
        # ``blockscaled_grouped_gemm.py``.
        "KLOOP_UNROLL": kloop_unroll,
    }


def derive_blockscaled_dist_combine_config(
    config: dict[str, int | bool],
    *,
    sf_vec_size: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_stage_dtype: torch.dtype,
    num_c_stages: int = 1,
    epilogue_subtile_candidates: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, int | bool]:
    """Retune a block-scaled config for fused dist COMBINE SMEM staging.

    Normal block-scaled GEMM keeps multiple C SMEM stages so the epilogue can
    pipeline TMEM->SMEM and SMEM->GMEM copies. Fused dist COMBINE scatters each
    staged subtile immediately, so it uses one C stage and spends the remaining
    SMEM on the A/B/SFA/SFB producer pipeline.

    Mathematically, for SMEM depth `s`, epilogue split `e`, and C-stage count
    `c`, the per-CTA footprint is:

      S(s, e, c) = align(A(s)) + align(B(s)) + align(SFA(s)) + align(SFB(s))
                 + c * C(e) + O(s)

    where `A/B/SFA/SFB` are linear in `s`, `C(e)` is the aligned one-subtile
    C staging tile, and `O(s)` is the tensormap/mbarrier/tile-id overhead. This
    helper sets `c = 1`, then maximizes `s` under the Blackwell dynamic SMEM
    launch budget; for a fixed `s`, it picks the smallest valid `e`.
    """
    if num_c_stages <= 0:
        raise ValueError(f"num_c_stages must be positive, got {num_c_stages}")

    cfg = dict(config)
    block_m = int(cfg["BLOCK_SIZE_M"])
    block_n = int(cfg["BLOCK_SIZE_N"])
    block_k = int(cfg["BLOCK_SIZE_K"])
    num_ctas = int(cfg["NUM_CTAS"])
    num_tmem_buffers = int(cfg["NUM_TMEM_BUFFERS"])
    num_tile_buffers = int(cfg["NUM_TILE_BUFFERS"])

    a_logical_bytes_per_elem = _dtype_size_bytes(a_dtype) / elements_per_byte(a_dtype)
    bytes_scale = max(
        1, int(_dtype_size_bytes(c_stage_dtype) / a_logical_bytes_per_elem)
    )
    if block_n >= 96 and block_n % 128 != 0:
        candidate_epilogues = [0]
    else:
        candidate_epilogues = sorted(
            {
                int(epi)
                for epi in (
                    int(cfg.get("EPILOGUE_SUBTILE", 0)),
                    *epilogue_subtile_candidates,
                )
                if int(epi) > 0 and int(epi) & (int(epi) - 1) == 0
            }
        )

    ab_stage_bytes = _estimate_blockscaled_ab_stage_bytes(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_ctas=num_ctas,
        sf_vec_size=sf_vec_size,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
    )
    max_smem_buffers = max(2, BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES // ab_stage_bytes)
    budget = BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES - SMEM_LAUNCH_SAFETY_MARGIN_BYTES

    best: tuple[int, int] | None = None
    for num_smem_buffers in range(max_smem_buffers, 1, -1):
        for epilogue_subtile in candidate_epilogues:
            if epilogue_subtile > 0:
                effective_split = epilogue_subtile * bytes_scale
                if block_n % effective_split != 0:
                    continue
            smem_bytes = estimate_blockscaled_grouped_gemm_smem_bytes(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_ctas=num_ctas,
                num_smem_buffers=num_smem_buffers,
                num_tmem_buffers=num_tmem_buffers,
                num_tile_buffers=num_tile_buffers,
                num_c_stages=num_c_stages,
                epilogue_subtile=epilogue_subtile,
                sf_vec_size=sf_vec_size,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                c_stage_dtype=c_stage_dtype,
            )
            if smem_bytes <= budget:
                best = (num_smem_buffers, epilogue_subtile)
                break
        if best is not None:
            break

    if best is None:
        raise AssertionError(
            "no block-scaled dist COMBINE SMEM config fits "
            f"BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}, "
            f"num_c_stages={num_c_stages}"
        )

    cfg["NUM_C_STAGES"] = num_c_stages
    cfg["NUM_SMEM_BUFFERS"], cfg["EPILOGUE_SUBTILE"] = best
    return cfg


# ---------------------------------------------------------------------------
# Tile-count-aware BLOCK_N recommender.
# ---------------------------------------------------------------------------
def cluster_waves(
    *,
    gm: int,
    n: int,
    block_m: int,
    block_n: int,
    num_ctas: int,
    num_sms: int = DEFAULT_NUM_SMS,
) -> float:
    """Return the cluster-wave count for the given shape and tile size.

    A "cluster" is one MMA tile = ``num_ctas`` CTAs occupying ``num_ctas``
    SMs. With ``num_clusters = ceil(GM / BM) * ceil(N / BN)`` and
    ``slots_per_wave = num_sms / num_ctas`` the wave count is
    ``num_clusters / slots_per_wave``. Values < 1 mean the workload
    can't even fill one wave (SMs are idle); values ≫ 1 mean the
    persistent kernel cycles through many waves and the tail-wave
    imbalance amortises out.
    """
    if gm <= 0 or n <= 0 or block_m <= 0 or block_n <= 0:
        return 0.0
    num_clusters = ((gm + block_m - 1) // block_m) * ((n + block_n - 1) // block_n)
    slots_per_wave = max(1, num_sms // num_ctas)
    return num_clusters / slots_per_wave


def recommend_block_n(
    *,
    gm: int,
    n: int,
    block_m: int,
    num_ctas: int,
    candidates: tuple[int, ...] = (256, 128),
    num_sms: int = DEFAULT_NUM_SMS,
    threshold_waves: float = MIN_CLUSTER_WAVES_THRESHOLD,
) -> int:
    """Recommend BLOCK_N for a given problem shape and tile-M.

    Walks ``candidates`` in descending order (largest BN first — bigger
    tiles amortise launch overhead and increase MMA arithmetic intensity)
    and returns the first one whose cluster-wave count is at least
    ``threshold_waves``. If none clears the threshold we still return the
    smallest BN in ``candidates`` (best we can do — finer tiling at least
    reduces wasted SM cycles in the tail wave).

    Use this from a kernel dispatch site that holds multiple JIT-compiled
    BLOCK_N variants in a config table (e.g. ``2cta1mma`` and
    ``2cta1mma_bn128``); compile-time constants stay constexpr, only the
    name lookup changes.
    """
    sorted_candidates = tuple(sorted(set(candidates), reverse=True))
    for bn in sorted_candidates:
        waves = cluster_waves(
            gm=gm,
            n=n,
            block_m=block_m,
            block_n=bn,
            num_ctas=num_ctas,
            num_sms=num_sms,
        )
        if waves >= threshold_waves:
            return bn
    return sorted_candidates[-1]


def auto_grouped_gemm_config(
    *,
    GM: int,
    G: int,
    problem_type: int = _FPROP,
    N: int | None = None,
    num_sms: int | None = None,
    epilogue_tile_n_multiple: int | None = None,
) -> str:
    """Pick the production tile config for a bf16/fp16 grouped GEMM shape."""
    return resolve_grouped_gemm_config(
        requested_config=None,
        gm=GM,
        groups=G,
        is_fprop=problem_type == _FPROP,
        n=N,
        num_sms=num_sms,
        epilogue_tile_n_multiple=epilogue_tile_n_multiple,
    )


def auto_grouped_gemm_mega_config(
    *,
    rows_per_active_expert: int,
    hidden_dim: int,
    intermediate_dim: int,
    active_experts: int,
    num_sms: int | None = None,
    epilogue_tile_n_multiple: int | None = None,
) -> str:
    derived = derive_auto_grouped_gemm_mega_config(
        rows_per_active_expert=rows_per_active_expert,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        active_experts=active_experts,
        num_sms=num_sms,
        epilogue_tile_n_multiple=epilogue_tile_n_multiple,
    )
    return registered_grouped_gemm_config_name(derived)


def _resolve_grouped_gemm_config(
    *,
    config: str | None,
    GM: int,
    G: int,
    problem_type: int,
    N: int | None = None,
    num_sms: int | None = None,
    dtype: torch.dtype | None = None,
    epilogue_tile_n_multiple: int | None = None,
) -> str:
    return resolve_grouped_gemm_config(
        requested_config=config,
        gm=GM,
        groups=G,
        is_fprop=problem_type == _FPROP,
        n=N,
        num_sms=num_sms,
        dtype=dtype,
        epilogue_tile_n_multiple=epilogue_tile_n_multiple,
    )


_SWAP_AB_CONFIG_SUFFIX: str = "_swap_ab"
# The selector uses total estimated rows, so this threshold is an average per
# group rather than a guarantee that every routed expert reaches this size.
_DEEP_PREFILL_PIPELINE_MIN_ROWS_PER_GROUP: int = 8192
_DEEP_PREFILL_PIPELINE_MAX_OUTPUT_DIM: int = 8192


def _dist_grouped_gemm_config(
    config_name: str,
) -> tuple[dict[str, int], bool]:
    swap_ab = config_name.endswith(_SWAP_AB_CONFIG_SUFFIX)
    base_name = (
        config_name.removesuffix(_SWAP_AB_CONFIG_SUFFIX) if swap_ab else config_name
    )
    config = GROUPED_GEMM_CONFIGS[base_name]
    if swap_ab and int(config["NUM_CTAS"]) != 1:
        raise ValueError("distributed SWAP_AB requires a 1-CTA config")
    return config, swap_ab


def _use_deep_prefill_pipeline(
    *,
    requested_config: str | None,
    swap_ab: bool,
    problem_type: int,
    config_gm: int,
    groups: int,
    output_dim: int,
) -> bool:
    return (
        requested_config is None
        and not swap_ab
        and problem_type == _FPROP
        and config_gm >= groups * _DEEP_PREFILL_PIPELINE_MIN_ROWS_PER_GROUP
        and output_dim <= _DEEP_PREFILL_PIPELINE_MAX_OUTPUT_DIM
    )


def _resolve_dist_grouped_gemm_config(
    *,
    config: str | None,
    kernel_M: int,
    G: int,
    problem_type: int,
    N: int,
    num_sms: int,
    estimate_recv_num_tokens: int | None = None,
    epilogue_tile_n_multiple: int | None = None,
    swap_ab: bool = False,
) -> str:
    requested_config = config
    config_swap_ab = bool(config and config.endswith(_SWAP_AB_CONFIG_SUFFIX))
    if config_swap_ab:
        config = config.removesuffix(_SWAP_AB_CONFIG_SUFFIX)
    swap_ab = swap_ab or config_swap_ab
    config_GM = kernel_M
    if estimate_recv_num_tokens is not None:
        if estimate_recv_num_tokens < 0:
            raise ValueError(
                "estimate_recv_num_tokens must be nonnegative, "
                f"got {estimate_recv_num_tokens}"
            )
        config_GM = max(1, estimate_recv_num_tokens)

    resolved: str | None = None
    if _use_deep_prefill_pipeline(
        requested_config=requested_config,
        swap_ab=swap_ab,
        problem_type=problem_type,
        config_gm=config_GM,
        groups=G,
        output_dim=N,
    ):
        deep_prefill_config = grouped_gemm_training_config()
        config_values, config_uses_swap_ab = _dist_grouped_gemm_config(
            deep_prefill_config
        )
        if epilogue_tile_n_multiple is None or grouped_gemm_epilogue_tile_is_compatible(
            config_values,
            epilogue_tile_n_multiple,
            swap_ab=config_uses_swap_ab,
        ):
            resolved = deep_prefill_config
    if resolved is None:
        resolved = _resolve_grouped_gemm_config(
            config=config,
            GM=config_GM,
            G=G,
            problem_type=problem_type,
            N=N,
            num_sms=num_sms,
            epilogue_tile_n_multiple=(None if swap_ab else epilogue_tile_n_multiple),
        )
    resolved_config, resolved_swap_ab = _dist_grouped_gemm_config(resolved)
    if (
        epilogue_tile_n_multiple is not None
        and not grouped_gemm_epilogue_tile_is_compatible(
            resolved_config,
            epilogue_tile_n_multiple,
            swap_ab=swap_ab or resolved_swap_ab,
        )
    ):
        raise ValueError(
            f"config {resolved!r} has an epilogue tile width incompatible with "
            f"the required multiple {epilogue_tile_n_multiple}"
        )
    return f"{resolved}{_SWAP_AB_CONFIG_SUFFIX}" if config_swap_ab else resolved


def _swap_ab_inference_config(
    config_name: str,
    *,
    requested_config: str | None,
    estimated_rows: int,
    num_groups: int,
    output_dim: int,
) -> str:
    config, swap_ab = _dist_grouped_gemm_config(config_name)
    if swap_ab:
        return config_name
    if int(config["NUM_CTAS"]) != 1:
        return config_name
    if requested_config is not None:
        return f"{config_name}{_SWAP_AB_CONFIG_SUFFIX}"

    derived = derive_auto_grouped_gemm_swap_ab_fprop_config(
        rows_per_group=max(1, math.ceil(estimated_rows / num_groups)),
        output_dim=output_dim,
    )
    if derived is None:
        return config_name
    return f"{registered_grouped_gemm_config_name(derived)}{_SWAP_AB_CONFIG_SUFFIX}"


def resolve_swizzled_inference_config(
    *,
    config: str | None,
    estimated_rows: int,
    num_groups: int,
    output_dim: int,
    num_sms: int | None,
) -> str:
    """Resolve native swizzled-inference geometry and operand orientation."""
    if num_sms is None:
        num_sms = num_sms_per_device()
    config_name = _resolve_dist_grouped_gemm_config(
        config=config,
        kernel_M=estimated_rows,
        G=num_groups,
        problem_type=_FPROP,
        N=output_dim,
        num_sms=num_sms,
        estimate_recv_num_tokens=estimated_rows,
    )
    return _swap_ab_inference_config(
        config_name,
        requested_config=config,
        estimated_rows=estimated_rows,
        num_groups=num_groups,
        output_dim=output_dim,
    )


def _activation_m_multiple(
    config_name: str,
    *,
    is_dispatch: bool,
) -> int:
    config, swap_ab = _dist_grouped_gemm_config(config_name)
    if is_dispatch and swap_ab:
        return int(config["BLOCK_SIZE_N"])
    return int(config["BLOCK_SIZE_M"])


def resolve_routing_m_multiple(
    *,
    config: str | None,
    estimated_rows: int,
    num_groups: int,
    dispatch_output_dim: int,
    combine_output_dim: int,
    num_sms: int | None,
) -> int:
    """Return the packed-row multiple shared by dispatch and combine."""
    if num_sms is None:
        num_sms = num_sms_per_device()
    dispatch_config_name = resolve_swizzled_inference_config(
        config=config,
        estimated_rows=estimated_rows,
        num_groups=num_groups,
        output_dim=dispatch_output_dim,
        num_sms=num_sms,
    )
    combine_config_name = _resolve_dist_grouped_gemm_config(
        config=config,
        kernel_M=estimated_rows,
        G=num_groups,
        problem_type=_FPROP,
        N=combine_output_dim,
        num_sms=num_sms,
        estimate_recv_num_tokens=estimated_rows,
    )
    dispatch_multiple = _activation_m_multiple(
        dispatch_config_name,
        is_dispatch=True,
    )
    combine_multiple = _activation_m_multiple(
        combine_config_name,
        is_dispatch=False,
    )
    return math.lcm(dispatch_multiple, combine_multiple)
