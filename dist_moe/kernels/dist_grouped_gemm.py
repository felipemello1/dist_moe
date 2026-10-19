# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTeDSL distributed grouped GEMM for dist_moe.

Persistent, warp-specialized SM100 kernel for FPROP/DGRAD dispatch and combine.
The deterministic numerical contract is validated by runtime parity.

CuTe grouped-GEMM launchers never synchronize expert-parallel peers. The
autograd orchestration layer owns the barriers that order dispatch publication
before remote gathers and remote combine scatters before local consumption.

Several @cute.kernel / @cute.jit bodies trip C901 (cyclomatic
complexity). They are tightly coupled per-tile state machines
(per-group accumulators, per-tile fences, cluster handshakes);
decomposing them forces re-threading dozens of @cute.jit arguments
and breaks JIT signature binding. We mark the affected defs with a
per-function ``# noqa: C901`` because flake8 (the lint backend used
in this repo) does not honor file-level per-rule disables.
"""

import cutlass
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

from .._activation_buffer_planner_kernel import (
    MEMORY_ALIGNMENT as _ACTIVATION_BUFFER_ALIGNMENT,
)
from ..formats import (
    BLOCK_MAJOR_K,
    canonical_swiglu_clamp,
    SWIGLU_CLAMP_ALPHA_DEFAULT,
    SWIGLU_CLAMP_LIMIT_DEFAULT,
)
from ._environment import l2_cache_size, num_sms_per_device
from ._grouped_gemm_config import (
    NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE,
    NATIVE_SWIGLU_EPILOGUE_TILE_N_MULTIPLE,
)
from .config import (  # noqa: F401
    _activation_m_multiple,
    _DEEP_PREFILL_PIPELINE_MAX_OUTPUT_DIM,
    _DEEP_PREFILL_PIPELINE_MIN_ROWS_PER_GROUP,
    _dist_grouped_gemm_config,
    _resolve_dist_grouped_gemm_config,
    _resolve_grouped_gemm_config,
    _SWAP_AB_CONFIG_SUFFIX,
    _swap_ab_inference_config,
    _use_deep_prefill_pipeline,
    DEFAULT_GROUPED_GEMM_WGRAD_CONFIG,
    derive_auto_grouped_gemm_swap_ab_fprop_config,
    GROUPED_GEMM_CONFIGS,
    grouped_gemm_epilogue_tile_is_compatible,
    grouped_gemm_epilogue_tile_n,
    grouped_gemm_training_config,
    registered_grouped_gemm_config_name,
    resolve_routing_m_multiple,
    resolve_swizzled_inference_config,
)

# Two distinct alignment contracts in this file. Naming them up front so
# every `from_dlpack(..., assumed_align=N)` callsite can refer to the
# right constant instead of bare numbers.
#
# * `_ACTIVATION_BUFFER_ALIGNMENT` — sub-allocations inside the shared
#   activation buffer are aligned to this by the routing planner
#   (`triton/dist_moe/activation_buffer_planner_kernel.MEMORY_ALIGNMENT`). Per-
#   group pointer arithmetic `base + accumulated_M * stride * elem_size`
#   preserves this alignment as long as `stride * elem_size` is a
#   multiple of it (true for typical hidden dims: bf16 × 64 = 128, etc.).
#   We import the planner constant to stay in lock-step.
#
# * `_PYTORCH_ALLOCATOR_MIN_ALIGNMENT` — conservative lower bound for
#   pointers returned by the PyTorch caching allocator. Empirically
#   fresh CUDA allocations are 8192-byte aligned, but ``slice/view`` at
#   a non-zero byte offset preserves only `gcd(base_align,
#   offset_bytes)` — so 16 is the smallest alignment we can rely on
#   without inspecting the slice math. Used for tensor descriptors
#   that are NOT served from `activation_buffer`.
from .dispatch_quant import _gather_k_segments
from .dist_grouped_gemm_kernel import (
    DistGroupedGemmKernel,
)

# Reuse the base kernel — class + module-level helpers + constants.
from .grouped_gemm import (
    _allocate_workspace,
    _compile_or_get,
    _DGRAD,
    _FPROP,
    _torch_layout_signature,
    _validate_activation_buffer_args,
    _validate_inputs,
    _WGRAD,
)
from .native_combine_swiglu import (
    native_swiglu_bwd_col_multiple,
)
from .pdl import pdl_enabled

_PYTORCH_ALLOCATOR_MIN_ALIGNMENT: int = 16


DIST_IDLE_WARP_IDS: tuple[int, int] = (6, 7)
# Fraction of L2 dedicated to B working set for N-clustering decision.
_L2_B_BUDGET_FRACTION: float = 0.5

_PADDING_SOURCE_CACHE: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}
_PADDING_SINK_CACHE: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}


def _padding_row(
    cache: dict[tuple[torch.device, torch.dtype, int], torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    numel: int,
    zero: bool,
) -> torch.Tensor:
    """Return a cached padding source/sink without retaining capture storage."""
    key = (device, dtype, numel)
    row = cache.get(key)
    if row is None:
        factory = torch.zeros if zero else torch.empty
        row = factory(numel, dtype=dtype, device=device)
        if not torch.cuda.is_current_stream_capturing():
            cache[key] = row
    return row


def _get_num_n_clusters(n: int, k: int, dtype_bytes: int) -> int:
    """Compute NUM_N_CLUSTERS from B working set vs L2 budget.

    N-clustering only pays off when B does not fit in L2; smaller shapes
    leave clustering at 1 to preserve pipeline overlap.
    """
    l2_budget = int(l2_cache_size() * _L2_B_BUDGET_FRACTION)
    total_b_bytes = n * k * dtype_bytes
    return max((total_b_bytes + l2_budget - 1) // l2_budget, 1)


def _native_swiglu_fwd_config_is_compatible(config: str, intermediate_dim: int) -> bool:
    """Return whether a native config can emit natural h1 and SwiGLU h2.

    The retained producer requires 32 gate/up columns per half-subtile and
    128 rows per CTA MMA atom. This only classifies the source geometry;
    it does not select a replacement config or collect clip counters.
    """
    cfg, swap_ab = _dist_grouped_gemm_config(config)
    return (
        intermediate_dim > 0
        and intermediate_dim % NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE == 0
        and not swap_ab
        and grouped_gemm_epilogue_tile_n(cfg) == 2 * NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE
        and cfg["BLOCK_SIZE_M"] == 128 * cfg["NUM_CTAS"] * cfg["NUM_MMAS"]
    )


# Plain COMBINE FPROP sends each output element to a peer after 2*K FLOPs.
# Up to K=3072 that NVLink stream rivals the MMA, and N-fastest order spends a
# wave on a couple of M tiles, i.e. on few destinations' rows; 4-tile N bands
# spread it over clusters/4 M tiles (-8..-16% at K 1792/3072). At K=4096 the
# MMA dominates and bands only lose locality (+1..+4% at 4096x4096). Above
# EP8, K=3072 bands lose end to end (contraction inference: 1-2% at 8K/32K
# tokens/rank, 14% at EP64 64K).
_COMBINE_N_BAND_MAX_K: int = 3072
_COMBINE_N_BAND_MAX_K_ABOVE_EP8: int = 2048
_COMBINE_N_BAND_TILES: int = 4


def _get_combine_fprop_num_n_clusters(
    n: int, k: int, block_n: int, world_size: int
) -> int:
    """Choose the retained output-bound COMBINE FPROP N-band schedule."""
    num_n_tiles = (n + block_n - 1) // block_n
    if world_size <= 8:
        max_k = _COMBINE_N_BAND_MAX_K
    else:
        max_k = _COMBINE_N_BAND_MAX_K_ABOVE_EP8
    if k > max_k:
        return 1
    # Ragged bands would pad the tile grid with idle slots.
    band = next(
        (
            b
            for b in range(_COMBINE_N_BAND_TILES, num_n_tiles + 1)
            if num_n_tiles % b == 0
        ),
        num_n_tiles,
    )
    return num_n_tiles // band


# Cross-CTA memory ops semantics (issued inline at callsites):
#
# `scope="gpu"`: critical for 2-CTA CGAs — default-scope atomics get merged
# across CTAs by ptxas, breaking per-CTA independent signals. `.gpu` scope
# prevents the merge while still providing release/acquire ordering.
#
# Peer-row LDGs use `cop="cg"` (L1-bypass → SASS `LDG.E.STRONG.GPU`). The
# signal load with sem="acquire", scope="gpu" establishes happens-before;
# subsequent cg data loads see post-release data via L2. `cop="cv"` would
# force system-scope strong (`LDG.E.STRONG.SYS`, ~3× slower) and is
# unnecessary on Blackwell NVLink-attached peers where L2 is invalidated
# by remote writes. STGs use `cop="cg"` so peer writes land at L2 before
# the destination CTA's TMA producer reads them.


# Host-side launchers: validate inputs, allocate per-launch workspace
# (atomic counters + per-M-tile gather counter), and dispatch the cached
# CuTe kernel.

_DIST_KERNEL_CACHE: dict[tuple, object] = {}


def _get_dist_kernel(
    config_name: str,
    problem_type: int,
    mode: int,
    force_n_major: bool,
    num_n_clusters: int,
    world_size: int,
    static_scheduler: bool,
    swiglu_fast_math: bool,
    host_a_tensormap: bool,
    has_padding_sentinels: bool,
    swap_ab: bool,
    gather_k_segments: int,
    schedule_rotation: bool = True,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    swiglu_emit_h2: bool = False,
    swiglu_emit_h1: bool = False,
    use_pdl: bool = False,
    block_major_weights: bool = False,
    single_a_tensormap: bool = False,
    has_bias: bool = False,
) -> DistGroupedGemmKernel:
    """Return a cached kernel with the exact scheduling and SwiGLU controls."""
    swiglu_clamped, swiglu_alpha, swiglu_limit = canonical_swiglu_clamp(
        swiglu_clamped, swiglu_alpha, swiglu_limit
    )
    key = (
        config_name,
        problem_type,
        mode,
        force_n_major,
        num_n_clusters,
        world_size,
        static_scheduler,
        swiglu_fast_math,
        host_a_tensormap,
        has_padding_sentinels,
        swap_ab,
        gather_k_segments,
        schedule_rotation,
        swiglu_clamped,
        swiglu_alpha,
        swiglu_limit,
        swiglu_emit_h2,
        swiglu_emit_h1,
        use_pdl,
        block_major_weights,
        single_a_tensormap,
        has_bias,
    )
    obj = _DIST_KERNEL_CACHE.get(key)
    if obj is None:
        obj = DistGroupedGemmKernel(
            config=_dist_grouped_gemm_config(config_name)[0],
            problem_type=problem_type,
            mode=mode,
            force_n_major=force_n_major,
            num_n_clusters=num_n_clusters,
            world_size=world_size,
            static_scheduler=static_scheduler,
            swiglu_fast_math=swiglu_fast_math,
            host_a_tensormap=host_a_tensormap,
            has_padding_sentinels=has_padding_sentinels,
            swap_ab=swap_ab,
            schedule_rotation=schedule_rotation,
            swiglu_clamped=swiglu_clamped,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_emit_h2=swiglu_emit_h2,
            swiglu_emit_h1=swiglu_emit_h1,
            gather_k_segments=gather_k_segments,
            use_pdl=use_pdl,
            block_major_weights=block_major_weights,
            single_a_tensormap=single_a_tensormap,
            has_bias=has_bias,
        )
        _DIST_KERNEL_CACHE[key] = obj
    return obj


def _counter_workspace_splits(
    *, kernel_m: int, block_m: int, num_groups: int
) -> tuple[int, int, int]:
    """Return counter lengths for the maximum grouped M-tile grid."""
    max_m_tiles = ((kernel_m + block_m - 1) // block_m) + num_groups
    return 1, 1, max_m_tiles


def allocate_decode_counter_storage(
    *, rows: int, num_groups: int, device: torch.device
) -> torch.Tensor:
    """Allocate counters that routing must clear before a BF16 decode launch."""
    return torch.empty(rows + num_groups + 2, dtype=torch.int32, device=device)


# Smallest token tile of an activation-buffer FPROP launch: BF16 presets and
# auto configs have BM >= 64; swapped inference GEMMs use BN >= 32.
_MIN_ACTIVATION_BUFFER_TOKEN_TILE = 32


def _activation_buffer_kernel_m(
    *, buffer_bytes: int, row_elements: int, itemsize: int
) -> int:
    """Return the kernel-M bound of an activation-buffer launch.

    The launcher and ``allocate_activation_buffer_counter_storage`` must agree on
    it. A dispatch row holds the gathered input and the output; a combine row only
    the input.
    """
    return max(1, buffer_bytes // (row_elements * itemsize))


def allocate_activation_buffer_counter_storage(
    *, activation_buffer: torch.Tensor, w: torch.Tensor, is_dispatch: bool
) -> torch.Tensor:
    """Allocate the counters of one activation-buffer FPROP launch; routing zeroes them.

    Sized for the smallest token tile, so any config fits:
        kernel_M 1000, 32-row tiles, 64 experts -> 1 + 1 + (32 + 64) = 98 int32

    Args:
        activation_buffer: Byte buffer the launch reads and writes.
        w: Expert weights ``[G, N, K]`` of the launch.
        is_dispatch: Whether this is the W13 dispatch launch (else W2 combine).

    Returns:
        Uninitialized int32 counter storage for ``precleared_counter_storage``.
    """
    num_groups, out_dim, in_dim = w.shape
    kernel_m = _activation_buffer_kernel_m(
        buffer_bytes=activation_buffer.numel(),
        row_elements=in_dim + out_dim if is_dispatch else in_dim,
        itemsize=w.element_size(),
    )
    splits = _counter_workspace_splits(
        kernel_m=kernel_m,
        block_m=_MIN_ACTIVATION_BUFFER_TOKEN_TILE,
        num_groups=num_groups,
    )
    return torch.empty(sum(splits), dtype=torch.int32, device=activation_buffer.device)


def _native_mega_counter_storage_splits(
    *,
    rows: int,
    num_groups: int,
    intermediate_dim: int,
    config: str,
) -> tuple[int, int]:
    """Return primary and secondary counter lengths for native Mega."""
    cfg, _ = _dist_grouped_gemm_config(config)
    primary_splits = _counter_workspace_splits(
        kernel_m=rows,
        block_m=int(cfg["BLOCK_SIZE_N"]),
        num_groups=num_groups,
    )
    primary_count = sum(primary_splits)
    token_tiles = primary_splits[-1]
    feature_tiles = intermediate_dim // int(cfg["BLOCK_SIZE_K"])
    return primary_count, token_tiles * feature_tiles


def allocate_native_mega_counter_storage(
    *,
    rows: int,
    num_groups: int,
    intermediate_dim: int,
    config: str,
    device: torch.device,
) -> torch.Tensor:
    """Allocate counters cleared by routing before a native Mega launch."""
    splits = _native_mega_counter_storage_splits(
        rows=rows,
        num_groups=num_groups,
        intermediate_dim=intermediate_dim,
        config=config,
    )
    return torch.empty(sum(splits), dtype=torch.int32, device=device)


def _allocate_counter_workspace(
    *,
    kernel_m: int,
    block_m: int,
    num_groups: int,
    device: torch.device,
    precleared_counter_storage: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build counters or validate routing-owned precleared counter storage."""
    splits = _counter_workspace_splits(
        kernel_m=kernel_m,
        block_m=block_m,
        num_groups=num_groups,
    )
    required_count = sum(splits)
    if precleared_counter_storage is None:
        storage = torch.zeros(required_count, dtype=torch.int32, device=device)
    else:
        if (
            precleared_counter_storage.dtype != torch.int32
            or precleared_counter_storage.device != device
            or not precleared_counter_storage.is_contiguous()
            or precleared_counter_storage.numel() < required_count
        ):
            raise ValueError(
                "precleared_counter_storage must be a contiguous int32 tensor on "
                f"the launch device with at least {required_count} elements"
            )
        storage = precleared_counter_storage[:required_count]
    counter, gather_counter, a_buff_counter = torch.split(storage, splits)
    return counter, gather_counter, a_buff_counter


def _fused_dist_grouped_gemm_impl(  # noqa: C901
    x: torch.Tensor,
    w: torch.Tensor,
    num_tokens_per_local_expert: torch.Tensor,
    peer_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer,
    *,
    problem_type: int,
    mode: int,
    num_sms: int | None,
    y: torch.Tensor | None,
    x_gathered: torch.Tensor | None,
    config: str | None,
    activation_buffer: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    estimate_recv_num_tokens: int | None = None,
    static_scheduler: bool = False,
    name_prefix: str | None = None,
    # Real [M, N/2] output, or int64[1] activation-buffer byte offset.
    swiglu_output: torch.Tensor | None = None,
    swiglu_fast_math: bool = False,
    precleared_counter_storage: torch.Tensor | None = None,
    m_multiple_of: int | None = None,
    has_padding_sentinels: bool = False,
    kernel_override: DistGroupedGemmKernel | None = None,
    scatter_ptrs: torch.Tensor | None = None,
    secondary_weight: torch.Tensor | None = None,
    secondary_ready_counter: torch.Tensor | None = None,
    secondary_ready_feature_tiles: int | None = None,
    pre_shuffled_rows: bool = False,
    # COMBINE_SWIGLU_BWD: `x` is the grad_h1 output the kernel fills from
    # grad_h2 (GM, I) and h1 (GM, 2I) before its TMA loads read it;
    # `swiglu_emit_h2` then overwrites grad_h2 with swiglu_fwd(h1).
    swiglu_dz: torch.Tensor | None = None,
    swiglu_h1: torch.Tensor | None = None,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    swiglu_emit_h2: bool = False,
    # DISPATCH_SWIGLU_FWD from natural (unswizzled) W13: also write the
    # natural h1 into `y` and return `(y, x_gathered)`.
    swiglu_emit_h1: bool = False,
    # FPROP: weights rewritten by `relayout_to_block_major`.
    block_major_weights: bool = False,
    # FPROP: read A through one descriptor when the launch allows it (see
    # GroupedGemmKernel.SINGLE_A_TENSORMAP); otherwise per-group descriptors.
    single_a_tensormap: bool = False,
    # COMBINE FPROP: contiguous BF16 [G, N] bias added to each scattered row.
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused-kernel impl: launches DistGroupedGemmKernel directly.

    DISPATCH (gather + GEMM): input `x` is local (M_local, in_dim);
    peer_ptrs are gather pointers; output is (GM, out_dim).

    COMBINE (GEMM + scatter): input `x` is already-gathered
    (M_local = GM); peer_ptrs are scatter pointers; the kernel writes
    locally then scatters per-row to peer combine buffers via the EPILOG
    WG. `num_out_tokens` must equal M_local.

    `local_rank` / `world_size` are derived from `symm_mem_buffer.hdl`.
    """
    if problem_type not in (_FPROP, _DGRAD):
        raise ValueError(
            f"unsupported problem_type={problem_type} (only _FPROP/_DGRAD)"
        )
    local_rank = symm_mem_buffer.hdl.rank
    world_size = symm_mem_buffer.hdl.world_size
    if mode not in (
        DistGroupedGemmKernel.DISPATCH,
        DistGroupedGemmKernel.COMBINE,
        DistGroupedGemmKernel.DISPATCH_SWIGLU_FWD,
        DistGroupedGemmKernel.COMBINE_SWIGLU_BWD,
    ):
        raise ValueError(f"unsupported mode={mode}")
    use_fused_swiglu_bwd = mode == DistGroupedGemmKernel.COMBINE_SWIGLU_BWD
    if use_fused_swiglu_bwd != (swiglu_dz is not None and swiglu_h1 is not None):
        raise ValueError("COMBINE_SWIGLU_BWD requires exactly swiglu_dz and swiglu_h1")
    if use_fused_swiglu_bwd and kernel_override is not None:
        raise ValueError("COMBINE_SWIGLU_BWD does not support a kernel override")
    use_fused_swiglu_dispatch = mode == DistGroupedGemmKernel.DISPATCH_SWIGLU_FWD
    if swiglu_emit_h1 and (
        not use_fused_swiglu_dispatch or kernel_override is not None
    ):
        raise ValueError(
            "swiglu_emit_h1 requires DISPATCH_SWIGLU_FWD without a secondary GEMM"
        )
    is_dispatch = mode in (
        DistGroupedGemmKernel.DISPATCH,
        DistGroupedGemmKernel.DISPATCH_SWIGLU_FWD,
    )
    use_activation_buffer = activation_buffer is not None
    use_secondary_gemm = kernel_override is not None
    secondary_values = (
        scatter_ptrs,
        secondary_weight,
        secondary_ready_counter,
        secondary_ready_feature_tiles,
    )
    if use_secondary_gemm != all(value is not None for value in secondary_values):
        raise ValueError(
            "kernel_override and all secondary GEMM launch arguments must be "
            "provided together"
        )
    if use_secondary_gemm and use_activation_buffer:
        raise ValueError("secondary grouped GEMM does not support activation_buffer")
    G, N, K = w.shape
    in_dim = K if problem_type == _FPROP else N
    out_dim = N if problem_type == _FPROP else K
    # CONTRACT: dim must be multiple of 8 (b16 vectorized 16-byte
    # LDG/STG, 8 elements per op). The fall-through partial-chunk arm
    # was removed because it bloated SASS 2.16x on COMBINE and masked
    # the vectorized path; it never executed for production hidden
    # dims, so the contract is enforced on the host instead.
    if not is_dispatch and out_dim % 8 != 0:
        raise ValueError(
            f"COMBINE requires out_dim ({out_dim}) to be a multiple of 8 "
            "(b16 vectorized scatter packs 8 elements per st.global.cg.v4.b32)"
        )
    if is_dispatch and in_dim % 8 != 0:
        raise ValueError(
            f"DISPATCH requires in_dim ({in_dim}) to be a multiple of 8 "
            "(b16 vectorized gather packs 8 elements per ld.global.cv.v4.b32)"
        )
    # In FPROP DISPATCH and COMBINE the activation-buffer path takes
    # `x` as an int64[1] byte offset; in DGRAD DISPATCH `dy` remains a
    # real tensor (launcher still stages to symm-mem). `_validate_inputs`
    # checks `x.dtype == w.dtype`; point it at a same-dtype buffer view
    # for offset-tensor cases.
    x_is_offset = use_activation_buffer and (not is_dispatch or problem_type == _FPROP)
    validation_x = activation_buffer.view(w.dtype) if x_is_offset else x
    split_sizes = _validate_inputs(validation_x, w, num_tokens_per_local_expert)
    if use_fused_swiglu_dispatch:
        if N % 32 != 0:
            raise ValueError(
                f"fused SwiGLU dispatch requires W13 output dim ({N}) divisible by 32"
            )
        if swiglu_output is None:
            raise ValueError("fused SwiGLU dispatch requires swiglu_output")
        if use_activation_buffer:
            if (
                swiglu_output.dtype != torch.int64
                or swiglu_output.shape != (1,)
                or swiglu_output.device != x.device
                or not swiglu_output.is_contiguous()
            ):
                raise ValueError(
                    "swiglu_output must be a contiguous int64[1] tensor on x.device"
                )
        elif (
            swiglu_output.shape != (num_out_tokens, N // 2)
            or swiglu_output.dtype != w.dtype
            or swiglu_output.device != w.device
            or not swiglu_output.is_contiguous()
        ):
            raise ValueError(
                "swiglu_output must be contiguous with shape "
                f"({num_out_tokens}, {N // 2}) and match w dtype/device"
            )

    # Real-tensor `x`: validate shape. Offset-tensor `x`: kernel_M comes
    # from buffer capacity below; symm-mem is pre-populated by the caller.
    if not x_is_offset and x.shape[-1] != in_dim:
        raise ValueError(f"x.shape[-1]={x.shape[-1]} != expected {in_dim}")

    if use_activation_buffer:
        # Capture the offset tensors (returned to caller for the dispatch
        # API contract) before we rebuild y / x_gathered as buffer views.
        ab_dtype = w.dtype
        ab_itemsize = ab_dtype.itemsize

    if is_dispatch:
        if use_activation_buffer:
            # `y` / `x_gathered` arrived as int64[1] byte-offset tensors;
            # save them and rebuild as max-shape buffer views so the host
            # TMA-atom build sees correct shape/strides. Kernel patches
            # the descriptor base ptr at runtime.
            assert y is not None and y.dtype == torch.int64 and y.shape == (1,), (
                "y must be int64[1] offset tensor in activation_buffer mode"
            )
            assert (
                x_gathered is not None
                and x_gathered.dtype == torch.int64
                and x_gathered.shape == (1,)
            ), "x_gathered must be int64[1] offset tensor in activation_buffer mode"
            if swiglu_emit_h1:
                _validate_activation_buffer_args(activation_buffer, y, x_gathered)
                if (
                    activation_buffer.dtype != torch.uint8
                    or activation_buffer.ndim != 1
                    or not activation_buffer.is_contiguous()
                    or activation_buffer.data_ptr() % _ACTIVATION_BUFFER_ALIGNMENT != 0
                    or activation_buffer.device != w.device
                ):
                    raise ValueError(
                        "fused native forward requires aligned contiguous uint8 "
                        "activation storage on w.device"
                    )
                if not y.is_contiguous() or not x_gathered.is_contiguous():
                    raise ValueError("fused native forward requires contiguous offsets")
            output_offset_t = y
            gathered_offset_t = x_gathered
            # Kernel-M upper bound from buffer capacity; split_sizes
            # drives actual writes.
            max_GM = _activation_buffer_kernel_m(
                buffer_bytes=activation_buffer.numel(),
                row_elements=in_dim + out_dim,
                itemsize=ab_itemsize,
            )
            kernel_M = max_GM
            # Typed views at offset 0; pointer is ignored at runtime
            # (descriptor is patched), shape/strides are honored.
            x_gathered = activation_buffer.view(ab_dtype)[: max_GM * in_dim].view(
                max_GM, in_dim
            )
            y = activation_buffer.view(ab_dtype)[: max_GM * out_dim].view(
                max_GM, out_dim
            )
            a_input = x_gathered
        else:
            if num_out_tokens is None:
                raise ValueError(
                    "num_out_tokens is required when activation_buffer is not provided"
                )
            GM = num_out_tokens
            if x_gathered is None:
                x_gathered = torch.empty((GM, in_dim), dtype=x.dtype, device=x.device)
            a_input = x_gathered
            kernel_M = int(GM)
            output_offset_t = None
            gathered_offset_t = None
    else:  # COMBINE
        if use_activation_buffer:
            # `x` is an int64[1] byte-offset; actual A data lives at
            # `activation_buffer + offset`. Build a max-shape view;
            # kernel patches the descriptor base ptr at runtime.
            assert x.dtype == torch.int64 and x.shape == (1,), (
                "x must be int64[1] offset tensor in activation_buffer mode"
            )
            input_offset_t = x
            max_M = _activation_buffer_kernel_m(
                buffer_bytes=activation_buffer.numel(),
                row_elements=in_dim,
                itemsize=ab_itemsize,
            )
            kernel_M = max_M
            a_input = activation_buffer.view(ab_dtype)[: max_M * in_dim].view(
                max_M, in_dim
            )
            x_gathered = a_input
            # Combine writes via `scatter_ptrs` to peer GPUs — local `y`
            # is never written. It exists only as a TMA-descriptor placeholder
            # for `from_dlpack(...).mark_layout_dynamic(leading_dim=1)` below,
            # which fixes out_dim and makes M dynamic (patched per-group at
            # runtime). The kernel_M extent is unused by the TMA atom, so we
            # use a 1-row view into `activation_buffer` instead of falling
            # through to `torch.empty((kernel_M, out_dim))` which allocates
            # ~`buffer_bytes` of fresh memory every call (e.g. 93 GiB for a
            # GB300 production-shape activation buffer).
            y = activation_buffer.view(ab_dtype)[:out_dim].view(1, out_dim)
        else:
            # In combine the kernel A operand is the input x (already
            # gathered).
            M_local = x.shape[0]
            a_input = x
            x_gathered = a_input
            kernel_M = int(M_local)
            input_offset_t = None
            if num_out_tokens != M_local:
                raise ValueError(
                    f"COMBINE: num_out_tokens ({num_out_tokens}) must "
                    f"equal M_local ({M_local}); the kernel writes M_local "
                    f"rows locally then scatters them via peer pointers."
                )

    if y is None:
        y = torch.empty((kernel_M, out_dim), dtype=a_input.dtype, device=a_input.device)
    elif swiglu_emit_h1 and (
        y.shape != (kernel_M, out_dim)
        or not y.is_contiguous()
        or y.data_ptr() % 32 != 0
    ):
        raise ValueError(
            f"swiglu_emit_h1 requires a contiguous, 32-byte aligned y of shape "
            f"({kernel_M}, {out_dim})"
        )
    if swiglu_emit_h1 and out_dim % (2 * NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE) != 0:
        raise ValueError(
            f"swiglu_emit_h1 requires N divisible by "
            f"{2 * NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE}; got {out_dim}"
        )

    if num_sms is None:
        num_sms = num_sms_per_device()
    override_swap_ab = bool(kernel_override and kernel_override.SWAP_AB)
    if swiglu_emit_h1 and config is None:
        config = grouped_gemm_training_config(
            routed_rows=estimate_recv_num_tokens, num_groups=G
        )
    config = _resolve_dist_grouped_gemm_config(
        config=config,
        kernel_M=kernel_M,
        G=G,
        problem_type=problem_type,
        N=out_dim,
        num_sms=num_sms,
        estimate_recv_num_tokens=estimate_recv_num_tokens,
        epilogue_tile_n_multiple=(
            NATIVE_SWIGLU_EPILOGUE_TILE_N_MULTIPLE
            if use_fused_swiglu_dispatch and not swiglu_emit_h1
            else None
        ),
        swap_ab=override_swap_ab,
    )
    cfg, config_swap_ab = _dist_grouped_gemm_config(config)
    swap_ab = (
        bool(kernel_override.SWAP_AB)
        if kernel_override is not None
        else problem_type == _FPROP and config_swap_ab
    )
    if swap_ab and use_activation_buffer and not single_a_tensormap:
        raise ValueError(
            "distributed SWAP_AB with activation_buffer requires single_a_tensormap"
        )
    # The kernel rejects the other unsupported modes when it is built.
    if block_major_weights and (
        kernel_override is not None
        or not w.is_contiguous()
        or w.shape[2] % BLOCK_MAJOR_K != 0
    ):
        raise ValueError(
            "block_major_weights needs no kernel override and contiguous weights "
            "whose K is a multiple of 64"
        )
    if swiglu_emit_h1 and not _native_swiglu_fwd_config_is_compatible(
        config, out_dim // 2
    ):
        raise ValueError(
            "swiglu_emit_h1 requires a non-swapped config with 128 rows per CTA "
            f"MMA atom and {2 * NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE}-column epilogue "
            f"subtiles; got {config!r}"
        )
    token_tile = int(cfg["BLOCK_SIZE_N"])
    if (
        swap_ab
        and use_fused_swiglu_dispatch
        and (not 32 <= token_tile <= 256 or token_tile % 32 != 0)
    ):
        raise ValueError(
            "fused distributed SWAP_AB requires a 32..256-row token tile "
            "divisible by 32"
        )
    token_tile = int(cfg["BLOCK_SIZE_N"] if swap_ab else cfg["BLOCK_SIZE_M"])
    has_padded_activation = m_multiple_of is not None and not use_activation_buffer
    host_a_tensormap = has_padded_activation and not swap_ab
    # Other launches keep a per-group A descriptor (the kernel rejects these).
    single_a_tensormap = (
        single_a_tensormap
        and problem_type == _FPROP
        and not (host_a_tensormap or cfg["NUM_CTAS"] != 1)
    )
    has_padding_sentinels = has_padding_sentinels or has_padded_activation
    if has_padded_activation and m_multiple_of % token_tile != 0:
        raise ValueError(
            f"m_multiple_of={m_multiple_of} must be divisible by the activation "
            f"tile multiple {token_tile} for config={config!r}"
        )
    if has_padding_sentinels:
        padding_row = _padding_row(
            _PADDING_SOURCE_CACHE if is_dispatch else _PADDING_SINK_CACHE,
            device=a_input.device,
            dtype=a_input.dtype,
            numel=in_dim if is_dispatch else out_dim,
            zero=is_dispatch,
        )
    else:
        padding_row = a_input
    NUM_CTAS = cfg["NUM_CTAS"]
    num_clusters = max(1, num_sms // NUM_CTAS)

    # Caller-provided storage is cleared by decode routing before launch.
    # Regular launches allocate zero-initialized counters here instead.
    counter, gather_counter, a_buff_counter = _allocate_counter_workspace(
        kernel_m=kernel_M,
        block_m=token_tile,
        num_groups=G,
        device=x.device,
        precleared_counter_storage=precleared_counter_storage,
    )
    _, tensormaps = _allocate_workspace(
        num_clusters,
        NUM_CTAS,
        x.device,
        counter=counter,
    )

    # Fused SwiGLU-bwd COMBINE runs N-major so the first wave waits on fewer
    # producer M-tiles (-7% at D12288 with 8K rows/expert). Plain COMBINE keeps
    # the per-group shape heuristic, with N bands for output-bound FPROP.
    force_n_major = is_dispatch or use_fused_swiglu_bwd
    if force_n_major:
        num_n_clusters = _get_num_n_clusters(
            int(out_dim), int(in_dim), a_input.element_size()
        )
    elif problem_type == _FPROP and not swap_ab:
        num_n_clusters = _get_combine_fprop_num_n_clusters(
            int(out_dim), int(in_dim), int(cfg["BLOCK_SIZE_N"]), world_size
        )
    else:
        num_n_clusters = 1
    if kernel_override is None:
        kernel = _get_dist_kernel(
            config_name=config,
            problem_type=problem_type,
            mode=mode,
            force_n_major=force_n_major,
            num_n_clusters=num_n_clusters,
            world_size=world_size,
            static_scheduler=static_scheduler,
            swiglu_fast_math=swiglu_fast_math,
            host_a_tensormap=host_a_tensormap,
            has_padding_sentinels=has_padding_sentinels,
            swap_ab=swap_ab,
            gather_k_segments=_gather_k_segments(int(out_dim)) if is_dispatch else 1,
            schedule_rotation=not pre_shuffled_rows,
            swiglu_clamped=swiglu_clamped,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_emit_h2=swiglu_emit_h2,
            swiglu_emit_h1=swiglu_emit_h1,
            use_pdl=pdl_enabled(),
            block_major_weights=block_major_weights,
            single_a_tensormap=single_a_tensormap,
            has_bias=bias is not None,
        )
    else:
        kernel = kernel_override

    # Placeholders set up the TMA atom layout per problem_type.
    # FPROP : A=x_gathered (M, K), B=w[g] (N, K), C=y (M, N)         — K-major
    # DGRAD : A=dy_gathered (M, N), B=w[g].t() (K, N), C=dx (M, K)   — B is MN-major
    # Strides for the per-group B descriptor differ accordingly.
    #
    # Alignment hints:
    #   * In `activation_buffer` mode A and C live inside a buffer whose
    #     sub-allocations are aligned to `_ACTIVATION_BUFFER_ALIGNMENT`
    #     by the routing planner. Per-group offsets compute as
    #     `base + accumulated_M * stride * itemsize`; this preserves the
    #     buffer alignment as long as `stride * itemsize` divides
    #     `_ACTIVATION_BUFFER_ALIGNMENT` — true for our typical hidden
    #     dims (bf16 × 64 = 128, etc.). Caller is responsible for not
    #     violating that contract; otherwise the TMA descriptor patch
    #     would assume more alignment than the address actually has.
    #   * Outside buffer mode A / C are caller-provided torch tensors;
    #     we only assume `_PYTORCH_ALLOCATOR_MIN_ALIGNMENT` (the
    #     conservative slice/view bound). Fresh allocations are 8192-
    #     aligned in practice but slices into them may not be.
    placeholder_align = (
        _ACTIVATION_BUFFER_ALIGNMENT
        if use_activation_buffer
        else _PYTORCH_ALLOCATOR_MIN_ALIGNMENT
    )
    a_placeholder = a_input.detach().unsqueeze(-1)
    c_placeholder = y.detach().unsqueeze(-1)
    a_cute = from_dlpack(
        a_placeholder, assumed_align=placeholder_align
    ).mark_layout_dynamic(leading_dim=1)
    c_cute = from_dlpack(
        c_placeholder, assumed_align=placeholder_align
    ).mark_layout_dynamic(leading_dim=1)
    if problem_type == _FPROP:
        # B = w view (N, K, G), K-major (stride[1]=1). Caller-provided
        # weight tensor — fresh torch allocation, only assume the
        # conservative caching-allocator minimum.
        b_placeholder = w.detach().permute(1, 2, 0)
        b_cute = from_dlpack(
            b_placeholder, assumed_align=_PYTORCH_ALLOCATOR_MIN_ALIGNMENT
        ).mark_layout_dynamic(leading_dim=1)
        b_s0_val = int(w.stride(1))
        b_s1_val = int(w.stride(2))
    else:  # _DGRAD: B is w viewed as MN-major (K_in, N, G) — stride[0]=1.
        b_placeholder = w.detach().permute(2, 1, 0)
        b_cute = from_dlpack(
            b_placeholder, assumed_align=_PYTORCH_ALLOCATOR_MIN_ALIGNMENT
        ).mark_layout_dynamic(leading_dim=0)
        b_s0_val = 1
        b_s1_val = int(K)  # K_in (= w.shape[2])
    # Tensors below are dtype-natural alignment for cute.arch.load: int32
    # → 4-byte, int64 → 8-byte. These are not subject to the buffer
    # alignment contract because the kernel reads them as plain scalars,
    # not as TMA-vectorized tiles.
    split_cute = from_dlpack(split_sizes.detach(), assumed_align=4)
    counter_cute = from_dlpack(counter, assumed_align=4)
    gather_counter_cute = from_dlpack(gather_counter, assumed_align=4)
    a_buff_counter_cute = from_dlpack(a_buff_counter, assumed_align=4)
    if use_fused_swiglu_dispatch:
        assert swiglu_output is not None
        if use_activation_buffer:
            max_swiglu_rows = max(
                1,
                activation_buffer.numel() // ((N // 2) * w.dtype.itemsize),
            )
            h2_words = (
                activation_buffer.view(w.dtype)[: max_swiglu_rows * (N // 2)]
                .view(max_swiglu_rows, N // 2)
                .view(torch.int32)
            )
        else:
            h2_words = swiglu_output.detach().view(torch.int32)
    else:
        # This branch is compile-time dead, but CuTe still requires a rank-2
        # int32 tensor with the same compile/runtime layout.
        h2_words = counter.view(1, 1)
    # Int64 row stride: the h2 store's Int32 row * (N // 4) can pass 2**31.
    h2_words_cute = from_dlpack(h2_words, assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    peer_ptrs_cute = from_dlpack(peer_ptrs.detach(), assumed_align=8)
    scatter_ptrs_cute = from_dlpack(
        (scatter_ptrs if scatter_ptrs is not None else peer_ptrs).detach(),
        assumed_align=8,
    )
    tensormaps_cute = from_dlpack(tensormaps, assumed_align=8)

    secondary_compile_args: tuple = ()
    secondary_runtime_args: tuple = ()
    secondary_layout_key: tuple = ()
    if use_secondary_gemm:
        assert secondary_weight is not None
        assert secondary_ready_counter is not None
        assert secondary_ready_feature_tiles is not None
        assert swiglu_output is not None
        ready_cute = from_dlpack(secondary_ready_counter, assumed_align=4)
        secondary_compile_args = (
            cutlass.Int64(secondary_weight.data_ptr()),
            cutlass.Int64(swiglu_output.data_ptr()),
            cutlass.Int32(secondary_weight.stride(1)),
            cutlass.Int32(secondary_weight.stride(2)),
            cutlass.Int32(swiglu_output.stride(0)),
            cutlass.Int32(swiglu_output.stride(1)),
            ready_cute,
            cutlass.Int32(secondary_ready_feature_tiles),
        )
        secondary_runtime_args = secondary_compile_args
        secondary_layout_key = (
            _torch_layout_signature(secondary_weight),
            _torch_layout_signature(swiglu_output),
        )

    # Inactive offset arguments are compile-time dead.
    dummy_offset = peer_ptrs[:1]
    if use_activation_buffer:
        if is_dispatch:
            output_offset_for_kernel = output_offset_t
            gathered_offset_for_kernel = gathered_offset_t
            input_offset_for_kernel = dummy_offset
        else:
            output_offset_for_kernel = dummy_offset
            gathered_offset_for_kernel = dummy_offset
            input_offset_for_kernel = input_offset_t
        swiglu_output_offset_for_kernel = (
            swiglu_output if use_fused_swiglu_dispatch else dummy_offset
        )
        activation_buffer_base_ptr_i64 = activation_buffer.data_ptr()
        activation_buffer_size_bytes = activation_buffer.nbytes
    else:
        output_offset_for_kernel = dummy_offset
        gathered_offset_for_kernel = dummy_offset
        input_offset_for_kernel = dummy_offset
        swiglu_output_offset_for_kernel = dummy_offset
        activation_buffer_base_ptr_i64 = 0
        activation_buffer_size_bytes = 0
    # ALIGNMENT CONTRACT: the offset value (not the tensor) must come
    # from the activation-buffer planner so that
    # `activation_buffer.data_ptr() + offsets_ptr[0]` is
    # `_ACTIVATION_BUFFER_ALIGNMENT`-aligned for downstream TMA. The
    # offset tensor itself only carries 8-byte alignment.
    output_offsets_cute = from_dlpack(output_offset_for_kernel, assumed_align=8)
    gathered_offsets_cute = from_dlpack(gathered_offset_for_kernel, assumed_align=8)
    input_offsets_cute = from_dlpack(input_offset_for_kernel, assumed_align=8)
    swiglu_output_offsets_cute = from_dlpack(
        swiglu_output_offset_for_kernel, assumed_align=8
    )

    # Conditional execution: caller threads a device int32[1] flag;
    # kernel returns early if the flag is 0. Caller passes None to
    # disable (constexpr no-op + dummy placeholder).
    use_conditional_execution = conditional_execution is not None
    if use_conditional_execution:
        if conditional_execution.shape != (1,):
            raise ValueError(
                "conditional_execution must be a 1-element device tensor; "
                f"got shape={tuple(conditional_execution.shape)}"
            )
        if not conditional_execution.is_cuda:
            raise ValueError("conditional_execution must reside on CUDA device")
        if conditional_execution.device != a_input.device:
            raise ValueError(
                f"conditional_execution must share x.device={a_input.device}; "
                f"got {conditional_execution.device}"
            )
        # Kernel reads as cute.Int32; widen non-int32 dtypes (e.g.
        # torch.bool from `forward_plan.need_recompute`).
        if conditional_execution.dtype != torch.int32:
            cond_tensor = conditional_execution.to(torch.int32)
        else:
            cond_tensor = conditional_execution
    else:
        cond_tensor = counter
    cond_cute = from_dlpack(cond_tensor, assumed_align=4)

    if use_fused_swiglu_bwd:
        assert swiglu_dz is not None and swiglu_h1 is not None
        swiglu_args = (
            cutlass.Int64(0 if use_activation_buffer else swiglu_dz.data_ptr()),
            cutlass.Int64(0 if use_activation_buffer else swiglu_h1.data_ptr()),
            cutlass.Int32(
                in_dim // 2 if use_activation_buffer else swiglu_dz.stride(0)
            ),
            cutlass.Int32(in_dim if use_activation_buffer else swiglu_h1.stride(0)),
        )
    else:
        swiglu_args = (
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
        )

    swiglu_dz_offsets_cute = from_dlpack(
        swiglu_dz if use_fused_swiglu_bwd and use_activation_buffer else dummy_offset,
        assumed_align=8,
    )
    swiglu_h1_offsets_cute = from_dlpack(
        swiglu_h1 if use_fused_swiglu_bwd and use_activation_buffer else dummy_offset,
        assumed_align=8,
    )

    # CORRECTNESS: with activation_buffer, `x` may be an int64[1] offset
    # tensor; reading elem from it would give 8 instead of the actual A
    # dtype size and break TMA stride math. Read from `a_input` instead.
    elem = a_input.element_size() if use_activation_buffer else x.element_size()
    stream = cutlass_torch.current_stream()
    cache_key = (
        config,
        problem_type,
        mode,
        force_n_major,
        num_n_clusters,
        world_size,
        static_scheduler,
        _torch_layout_signature(a_placeholder),
        _torch_layout_signature(b_placeholder),
        _torch_layout_signature(c_placeholder),
        G,
        elem,
        use_activation_buffer,
        use_conditional_execution,
        swiglu_fast_math,
        host_a_tensormap,
        has_padding_sentinels,
        swap_ab,
        type(kernel).__name__,
        # The chunked native Mega bakes its cross-group lead window into the
        # traced schedule; launches differing only in the window must not
        # share a compiled binary.
        kernel.GLOBAL_LEAD_CHUNKS,
        secondary_layout_key,
        _torch_layout_signature(h2_words),
        "fused",
        pre_shuffled_rows,
        kernel.swiglu_clamped,
        kernel.swiglu_alpha,
        kernel.swiglu_limit,
        kernel.swiglu_emit_h2,
        kernel.swiglu_emit_h1,
        kernel.GATHER_K_SEGMENTS,
        kernel.use_pdl,
        kernel.block_major_weights,
        kernel.SINGLE_A_TENSORMAP,
        kernel.has_bias,
    )
    # Kernel-logical axes depend on problem_type:
    #   FPROP: out=N_w, in=K_w    → kernel (kernel_M, N_w, K_w)
    #   DGRAD: out=K_w, in=N_w    → kernel (kernel_M, K_w, N_w)
    if problem_type == _FPROP:
        kernel_N = int(N)
        kernel_K = int(K)
    else:  # _DGRAD
        kernel_N = int(K)
        kernel_K = int(N)
    bias_ptr = 0 if bias is None else bias.data_ptr()
    compile_args = (
        a_cute,
        b_cute,
        c_cute,
        split_cute,
        counter_cute,
        gather_counter_cute,
        a_buff_counter_cute,
        h2_words_cute,
        peer_ptrs_cute,
        scatter_ptrs_cute,
        tensormaps_cute,
        padding_row.data_ptr(),
        bias_ptr,
        a_input.data_ptr(),
        w.data_ptr(),
        y.data_ptr(),
        a_input.stride(0),
        a_input.stride(1),
        b_s0_val,
        b_s1_val,
        y.stride(0),
        y.stride(1),
        elem,
        elem,
        elem,
        G,
        kernel_M,
        kernel_N,
        kernel_K,
        local_rank,
        activation_buffer_base_ptr_i64,
        cutlass.Int64(activation_buffer_size_bytes),
        output_offsets_cute,
        gathered_offsets_cute,
        input_offsets_cute,
        swiglu_output_offsets_cute,
        use_activation_buffer,
        cond_cute,
        use_conditional_execution,
        *swiglu_args,
        swiglu_dz_offsets_cute,
        swiglu_h1_offsets_cute,
        num_clusters,
        stream,
        *secondary_compile_args,
    )
    # Runtime args: drop constexpr (elem_size_bytes_a/b/c, G,
    # use_activation_buffer, use_conditional_execution) — baked into the
    # compiled kernel.
    runtime_args = (
        a_cute,
        b_cute,
        c_cute,
        split_cute,
        counter_cute,
        gather_counter_cute,
        a_buff_counter_cute,
        h2_words_cute,
        peer_ptrs_cute,
        scatter_ptrs_cute,
        tensormaps_cute,
        padding_row.data_ptr(),
        bias_ptr,
        a_input.data_ptr(),
        w.data_ptr(),
        y.data_ptr(),
        a_input.stride(0),
        a_input.stride(1),
        b_s0_val,
        b_s1_val,
        y.stride(0),
        y.stride(1),
        kernel_M,
        kernel_N,
        kernel_K,
        local_rank,
        activation_buffer_base_ptr_i64,
        cutlass.Int64(activation_buffer_size_bytes),
        output_offsets_cute,
        gathered_offsets_cute,
        input_offsets_cute,
        swiglu_output_offsets_cute,
        cond_cute,
        *swiglu_args,
        swiglu_dz_offsets_cute,
        swiglu_h1_offsets_cute,
        num_clusters,
        stream,
        *secondary_runtime_args,
    )
    prefix_target = (
        DistGroupedGemmKernel.dispatch_kernel
        if mode
        in (
            DistGroupedGemmKernel.DISPATCH,
            DistGroupedGemmKernel.DISPATCH_SWIGLU_FWD,
        )
        else DistGroupedGemmKernel.combine_kernel
    )
    compiled = _compile_or_get(
        cache_key,
        kernel,
        *compile_args,
        name_prefix=name_prefix,
        prefix_target=prefix_target,
    )
    compiled(*runtime_args)
    if use_activation_buffer and is_dispatch:
        # Public-API contract: when activation_buffer is provided in
        # DISPATCH, return the original offset tensors (caller dereferences
        # them post-kernel). The internal y / x_gathered are buffer views
        # that don't survive the launcher boundary.
        if use_fused_swiglu_dispatch and not swiglu_emit_h1:
            assert swiglu_output is not None
            return swiglu_output, gathered_offset_t
        return output_offset_t, gathered_offset_t
    if use_fused_swiglu_dispatch and not swiglu_emit_h1:
        assert swiglu_output is not None
        return swiglu_output, x_gathered
    return y, x_gathered


def dist_grouped_gemm_fprop_dispatch(
    x: torch.Tensor,  # (M_local, K)
    w: torch.Tensor,  # (G, N, K)
    num_tokens_per_local_expert: torch.Tensor,
    gather_ptrs: torch.Tensor,  # int64[GM] - peer-rank A pointers
    num_out_tokens: int | None,
    symm_mem_buffer,  # SymmMemBuffer (lazy import to avoid torch.distributed dep)
    *,
    topk: int | None = None,  # accepted for signature parity; ptrs encode topk
    num_sms: int | None = None,
    y: torch.Tensor | None = None,
    x_gathered: torch.Tensor | None = None,
    config: str | None = None,
    activation_buffer: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    estimate_recv_num_tokens: int | None = None,
    static_scheduler: bool = False,
    precleared_counter_storage: torch.Tensor | None = None,
    m_multiple_of: int | None = None,
    pre_shuffled_rows: bool = False,
    swiglu_h2: torch.Tensor | None = None,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    block_major_weights: bool = False,
    single_a_tensormap: bool = False,
    interleaved_gate_up: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed grouped GEMM fprop dispatch: gather A from peers, GEMM.

    `local_rank` and `world_size` are derived from `symm_mem_buffer.hdl`.

    `swiglu_h2` ([num_out_tokens, N/2]) additionally receives `swiglu_fwd(y)`
    from the retained epilogue with the same activation flags. W13 and y
    remain in natural `[gate; up]` order. Fusion requires N/2 divisible by
    32, a non-swapped config, 64-column epilogue subtiles and 128 rows per
    CTA MMA atom. No clip counters or Python callbacks are run here.

    When `activation_buffer` is provided, `y` and `x_gathered` must be
    int64[1] device tensors holding byte offsets into the buffer (kernel
    patches TMA descriptors at runtime). `single_a_tensormap=True` patches
    the activation descriptor (A, or B when swapped) once per CTA instead of
    once per expert when the launch allows it (unpadded, 1-CTA); otherwise it
    is ignored. The caller must pre-stage `x` to symm-mem and barrier the EP
    group before this launch. Addresses must be 128-byte aligned.

    In that mode `swiglu_h2` is also an int64[1] device byte offset. Existing
    context storage supplies all three disjoint packed destinations: y has
    width N, swiglu_h2 has width N/2, and x_gathered has width K. Only the
    rows described by num_tokens_per_local_expert are written. The caller
    keeps standalone SwiGLU when clip counters are required; this launcher
    never silently substitutes another variant. Weight and expert-output
    processing remain outside this boundary.

    `precleared_counter_storage`, when given, holds int32 counters the caller
    zeroed before this launch (`allocate_activation_buffer_counter_storage`);
    the launch then skips its own memset.

    `block_major_weights` reads weights rewritten by `relayout_to_block_major`
    (`Config.block_major_weights`). With `interleaved_gate_up` (W13 rows in
    gate/up groups of 16), `swiglu_h2` is written and y is not.

    Returns: (y_out, x_out), or the original (y_offset, x_gathered_offset)
    tensors when activation_buffer is provided.
    """
    emit_swiglu = swiglu_h2 is not None
    return _fused_dist_grouped_gemm_impl(
        x=x,
        w=w,
        num_tokens_per_local_expert=num_tokens_per_local_expert,
        peer_ptrs=gather_ptrs,
        num_out_tokens=num_out_tokens,
        symm_mem_buffer=symm_mem_buffer,
        problem_type=_FPROP,
        mode=(
            DistGroupedGemmKernel.DISPATCH_SWIGLU_FWD
            if emit_swiglu
            else DistGroupedGemmKernel.DISPATCH
        ),
        num_sms=num_sms,
        y=y,
        x_gathered=x_gathered,
        config=config,
        activation_buffer=activation_buffer,
        conditional_execution=conditional_execution,
        estimate_recv_num_tokens=estimate_recv_num_tokens,
        static_scheduler=static_scheduler,
        name_prefix=(
            "_cute_dist_grouped_gemm_dispatch_fprop_emit_swiglu_fwd"
            if emit_swiglu
            else "_cute_dist_grouped_gemm_dispatch_fprop"
        ),
        pre_shuffled_rows=pre_shuffled_rows,
        precleared_counter_storage=precleared_counter_storage,
        m_multiple_of=m_multiple_of,
        swiglu_output=swiglu_h2,
        swiglu_emit_h1=emit_swiglu and not interleaved_gate_up,
        swiglu_fast_math=emit_swiglu and swiglu_fast_math,
        swiglu_clamped=emit_swiglu and swiglu_clamped,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        block_major_weights=block_major_weights,
        single_a_tensormap=single_a_tensormap,
    )


def dist_grouped_gemm_fprop_swiglu_fwd_dispatch(
    x: torch.Tensor,
    w13_swizzled: torch.Tensor,
    num_tokens_per_local_expert: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer,
    *,
    h1_scratch: torch.Tensor | None = None,
    # Real [num_out_tokens, N/2] output, or int64[1] activation-buffer offset.
    h2: torch.Tensor | None = None,
    x_gathered: torch.Tensor | None = None,
    num_sms: int | None = None,
    config: str | None = None,
    activation_buffer: torch.Tensor | None = None,
    estimate_recv_num_tokens: int | None = None,
    precleared_counter_storage: torch.Tensor | None = None,
    swiglu_fast_math: bool = False,
    m_multiple_of: int | None = None,
    pre_shuffled_rows: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather, run swizzled FC13, and write compact SwiGLU output."""
    return _fused_dist_grouped_gemm_impl(
        x=x,
        w=w13_swizzled,
        num_tokens_per_local_expert=num_tokens_per_local_expert,
        peer_ptrs=gather_ptrs,
        num_out_tokens=num_out_tokens,
        symm_mem_buffer=symm_mem_buffer,
        problem_type=_FPROP,
        mode=DistGroupedGemmKernel.DISPATCH_SWIGLU_FWD,
        num_sms=num_sms,
        y=h1_scratch,
        x_gathered=x_gathered,
        config=config,
        activation_buffer=activation_buffer,
        estimate_recv_num_tokens=estimate_recv_num_tokens,
        name_prefix="_cute_dist_grouped_gemm_dispatch_fprop_swiglu_fwd",
        pre_shuffled_rows=pre_shuffled_rows,
        swiglu_output=h2,
        swiglu_fast_math=swiglu_fast_math,
        precleared_counter_storage=precleared_counter_storage,
        m_multiple_of=m_multiple_of,
    )


def dist_grouped_gemm_wgrad_dispatch(
    dy: torch.Tensor,
    x: torch.Tensor,
    split_sizes: torch.Tensor,
    gather_ptrs: torch.Tensor,
    symm_mem_buffer,
    *,
    wgrad: torch.Tensor,
    hidden_dim_dy: int,
    hidden_dim_x: int,
    dtype: torch.dtype,
    output_accum: bool = False,
    num_sms: int | None = None,
    config: str = DEFAULT_GROUPED_GEMM_WGRAD_CONFIG,
    activation_buffer: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch BF16 input columns into transient scratch while computing WGRAD."""
    assert dtype == torch.bfloat16
    assert hidden_dim_dy % 64 == 0 and hidden_dim_x % 64 == 0
    use_ab = activation_buffer is not None
    dummy_offset = gather_ptrs[:1]
    dy_offset, x_offset = (dy, x) if use_ab else (dummy_offset, dummy_offset)
    if activation_buffer is not None:
        _validate_activation_buffer_args(activation_buffer, dy, x)
        dy = activation_buffer.view(dtype)[:hidden_dim_dy].view(1, hidden_dim_dy)
        x = activation_buffer.view(dtype)[:hidden_dim_x].view(1, hidden_dim_x)
    split_sizes = _validate_inputs(x, dy, split_sizes)
    assert dy.stride(1) == x.stride(1) == 1
    assert dy.stride(0) * dtype.itemsize % _ACTIVATION_BUFFER_ALIGNMENT == 0
    assert x.stride(0) * dtype.itemsize % _ACTIVATION_BUFFER_ALIGNMENT == 0
    groups = split_sizes.numel()
    cfg = GROUPED_GEMM_CONFIGS[config]
    num_sms = num_sms if num_sms is not None else num_sms_per_device()
    clusters = max(1, num_sms // cfg["NUM_CTAS"])
    num_n_clusters = (hidden_dim_x + cfg["BLOCK_SIZE_N"] - 1) // cfg["BLOCK_SIZE_N"]
    kernel = DistGroupedGemmKernel(
        cfg,
        _WGRAD,
        DistGroupedGemmKernel.DISPATCH,
        world_size=symm_mem_buffer.hdl.world_size,
        num_n_clusters=num_n_clusters,
        has_padding_sentinels=True,
        output_accum=output_accum,
    )
    counter, tensormaps = _allocate_workspace(clusters, cfg["NUM_CTAS"], dy.device)
    gather_counter = torch.zeros_like(counter)
    ready = torch.zeros(
        groups
        * ((hidden_dim_x + kernel.AG_BLOCK_SIZE_K - 1) // kernel.AG_BLOCK_SIZE_K),
        dtype=torch.int32,
        device=dy.device,
    )
    padding = _padding_row(
        _PADDING_SOURCE_CACHE,
        device=dy.device,
        dtype=dtype,
        numel=hidden_dim_x,
        zero=True,
    )
    if dy.shape[0] == 0:
        dy = torch.empty((1, hidden_dim_dy), dtype=dtype, device=dy.device)
        x = torch.empty((1, hidden_dim_x), dtype=dtype, device=x.device)
    a = dy.t().detach().unsqueeze(-1)
    b = x.t().detach().unsqueeze(-1)
    c = wgrad[0].detach().unsqueeze(-1)
    tensors = (
        from_dlpack(a, assumed_align=16).mark_layout_dynamic(leading_dim=0),
        from_dlpack(b, assumed_align=16).mark_layout_dynamic(leading_dim=0),
        from_dlpack(c, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(split_sizes.detach(), assumed_align=4),
        from_dlpack(counter, assumed_align=4),
        from_dlpack(gather_counter, assumed_align=4),
        from_dlpack(ready, assumed_align=4),
        from_dlpack(counter.view(1, 1), assumed_align=4),
        from_dlpack(gather_ptrs.detach(), assumed_align=8),
        from_dlpack(gather_ptrs.detach(), assumed_align=8),
        from_dlpack(tensormaps, assumed_align=8),
    )
    ptrs = (
        padding.data_ptr(),
        dy.data_ptr(),
        x.data_ptr(),
        wgrad.data_ptr(),
        dy.stride(1),
        dy.stride(0),
        x.stride(1),
        x.stride(0),
        wgrad.stride(1),
        wgrad.stride(2),
    )
    dims = (hidden_dim_dy, hidden_dim_x, 0, symm_mem_buffer.hdl.rank)
    offsets = (
        activation_buffer.data_ptr() if activation_buffer is not None else 0,
        cutlass.Int64(activation_buffer.nbytes if activation_buffer is not None else 0),
        from_dlpack(dy_offset.detach(), assumed_align=8),
        from_dlpack(x_offset.detach(), assumed_align=8),
        from_dlpack(dummy_offset, assumed_align=8),
        from_dlpack(dummy_offset, assumed_align=8),
    )
    cond = from_dlpack(
        conditional_execution if conditional_execution is not None else counter,
        assumed_align=4,
    )
    tail = (
        from_dlpack(dummy_offset, assumed_align=8),
        cutlass.Int64(0),
        None,  # WGRAD does not use borrowed DGRAD's SR metadata.
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        clusters,
        cutlass_torch.current_stream(),
    )
    compile_args = (
        *tensors,
        *ptrs,
        dtype.itemsize,
        dtype.itemsize,
        wgrad.element_size(),
        groups,
        *dims,
        *offsets,
        use_ab,
        cond,
        conditional_execution is not None,
        *tail,
        *([None] * 13),
    )
    runtime_args = (*tensors, *ptrs, *dims, *offsets, cond, *tail)
    cache_key = (
        "wgrad_dispatch",
        conditional_execution.dtype if conditional_execution is not None else None,
        config,
        groups,
        symm_mem_buffer.hdl.world_size,
        use_ab,
        output_accum,
        num_n_clusters,
        _torch_layout_signature(a),
        _torch_layout_signature(b),
        _torch_layout_signature(c),
    )
    compiled = _compile_or_get(
        cache_key,
        kernel,
        *compile_args,
        name_prefix="_cute_dist_grouped_gemm_wgrad_dispatch",
        prefix_target=DistGroupedGemmKernel.dispatch_kernel,
    )
    compiled(*runtime_args)
    return wgrad


def dist_grouped_gemm_dgrad_dispatch(
    dy: torch.Tensor,  # (M_local, N) — local dy (will be staged to symm-mem)
    w: torch.Tensor,  # (G, N, K)
    num_tokens_per_local_expert: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer,
    *,
    num_sms: int | None = None,
    dx: torch.Tensor | None = None,
    dy_gathered: torch.Tensor | None = None,
    config: str | None = None,
    activation_buffer: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    static_scheduler: bool = False,
    has_padding_sentinels: bool = False,
    pre_shuffled_rows: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed grouped GEMM dgrad dispatch: gather dy, GEMM → dx.

    `dy` may be 2-D (M_local, N) or 3-D (M_local, topk, N); only the last
    axis must match w.N. See `dist_grouped_gemm_fprop_dispatch` for
    `activation_buffer` semantics; `dx` / `dy_gathered` play the role of
    `y` / `x_gathered`.

    """
    G, N, K = w.shape
    # Flatten leading axes; topk expansion is already encoded in gather_ptrs.
    dy_2d = dy.reshape(-1, N)
    return _fused_dist_grouped_gemm_impl(
        x=dy_2d,
        w=w,
        num_tokens_per_local_expert=num_tokens_per_local_expert,
        peer_ptrs=gather_ptrs,
        num_out_tokens=num_out_tokens,
        symm_mem_buffer=symm_mem_buffer,
        problem_type=_DGRAD,
        mode=DistGroupedGemmKernel.DISPATCH,
        num_sms=num_sms,
        y=dx,
        x_gathered=dy_gathered,
        config=config,
        activation_buffer=activation_buffer,
        conditional_execution=conditional_execution,
        static_scheduler=static_scheduler,
        name_prefix="_cute_dist_grouped_gemm_dispatch_dgrad",
        pre_shuffled_rows=pre_shuffled_rows,
        has_padding_sentinels=has_padding_sentinels,
    )


# Combine launchers: fused GEMM + per-row scatter. Scatter is fused
# into the EPILOG WG (no GATHER WG, 256 threads/CTA): each subtile is
# staged through SMEM and a 128-thread per-row scatter STGs to the peer
# buffer resolved from `scatter_ptrs`.


def dist_grouped_gemm_fprop_combine(
    x: torch.Tensor,  # (GM, K) — pre-gathered A on local rank, OR int64[1] offset
    w: torch.Tensor,  # (G, N, K)
    num_tokens_per_local_expert: torch.Tensor,
    scatter_ptrs: torch.Tensor,  # int64[GM] — per-row peer-rank C pointers
    symm_mem_buffer,
    *,
    num_sms: int | None = None,
    config: str | None = None,
    activation_buffer: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    estimate_recv_num_tokens: int | None = None,
    static_scheduler: bool = False,
    precleared_counter_storage: torch.Tensor | None = None,
    m_multiple_of: int | None = None,
    pre_shuffled_rows: bool = False,
    block_major_weights: bool = False,
    single_a_tensormap: bool = False,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Distributed grouped GEMM forward combine: fused GEMM + scatter.

    Returns the local view of `symm_mem_buffer` (peers scatter their
    GEMM outputs in via `scatter_ptrs`).

    When `activation_buffer` is provided, `x` is an int64[1] byte-offset
    tensor; kernel patches its A-tensor TMA descriptor at runtime, once per
    CTA with `single_a_tensormap` (same rule as the dispatch launch).
    Addresses must be 128-byte aligned.

    `precleared_counter_storage` and `block_major_weights` follow the dispatch
    launch's contract.

    `bias` is an optional contiguous ``[G, N]`` tensor in `w`'s dtype; row
    ``g`` is added to expert ``g``'s outputs before the scatter.

    The caller owns all peer synchronization.
    """
    if bias is not None and (
        bias.shape != w.shape[:2]
        or bias.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
        or not bias.is_contiguous()
        or bias.device != w.device
    ):
        raise ValueError("bias must be a contiguous BF16 [G, N] tensor matching w")
    if scatter_ptrs.dtype != torch.int64:
        raise TypeError("scatter_ptrs must be int64")
    if not scatter_ptrs.is_cuda or scatter_ptrs.device != x.device:
        raise ValueError("scatter_ptrs must be on x.device")
    if activation_buffer is None:
        num_out_tokens = x.shape[0]
    else:
        num_out_tokens = None  # max-bound derived from buffer in impl
    _fused_dist_grouped_gemm_impl(
        x=x,
        w=w,
        num_tokens_per_local_expert=num_tokens_per_local_expert,
        peer_ptrs=scatter_ptrs,
        num_out_tokens=num_out_tokens,
        symm_mem_buffer=symm_mem_buffer,
        problem_type=_FPROP,
        mode=DistGroupedGemmKernel.COMBINE,
        num_sms=num_sms,
        y=None,
        x_gathered=None,
        config=config,
        activation_buffer=activation_buffer,
        conditional_execution=conditional_execution,
        estimate_recv_num_tokens=estimate_recv_num_tokens,
        static_scheduler=static_scheduler,
        name_prefix="_cute_dist_grouped_gemm_combine_fprop",
        pre_shuffled_rows=pre_shuffled_rows,
        precleared_counter_storage=precleared_counter_storage,
        m_multiple_of=m_multiple_of,
        block_major_weights=block_major_weights,
        single_a_tensormap=single_a_tensormap,
        bias=bias,
    )
    return symm_mem_buffer.local()


def dist_grouped_gemm_dgrad_combine(
    dy: torch.Tensor,  # (GM, N) — pre-gathered dy, OR int64[1] offset
    w: torch.Tensor,  # (G, N, K)
    num_tokens_per_local_expert: torch.Tensor,
    scatter_ptrs: torch.Tensor,  # int64[GM] — per-row peer-rank dx pointers
    symm_mem_buffer,
    *,
    num_sms: int | None = None,
    config: str | None = None,
    activation_buffer: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    static_scheduler: bool = False,
    has_padding_sentinels: bool = False,
    pre_shuffled_rows: bool = False,
) -> torch.Tensor:
    """Distributed grouped GEMM dgrad combine: fused dgrad GEMM + scatter.

    See `dist_grouped_gemm_fprop_combine` for `activation_buffer` and
    synchronization semantics; here `dy` plays the role of `x`.

    """
    if scatter_ptrs.dtype != torch.int64:
        raise TypeError("scatter_ptrs must be int64")
    if not scatter_ptrs.is_cuda or scatter_ptrs.device != dy.device:
        raise ValueError("scatter_ptrs must be on dy.device")
    if activation_buffer is None:
        num_out_tokens = dy.shape[0]
    else:
        num_out_tokens = None
    _fused_dist_grouped_gemm_impl(
        x=dy,
        w=w,
        num_tokens_per_local_expert=num_tokens_per_local_expert,
        peer_ptrs=scatter_ptrs,
        num_out_tokens=num_out_tokens,
        symm_mem_buffer=symm_mem_buffer,
        problem_type=_DGRAD,
        mode=DistGroupedGemmKernel.COMBINE,
        num_sms=num_sms,
        y=None,
        x_gathered=None,
        config=config,
        activation_buffer=activation_buffer,
        conditional_execution=conditional_execution,
        static_scheduler=static_scheduler,
        name_prefix="_cute_dist_grouped_gemm_combine_dgrad",
        pre_shuffled_rows=pre_shuffled_rows,
        has_padding_sentinels=has_padding_sentinels,
    )
    return symm_mem_buffer.local()


def dist_grouped_gemm_dgrad_swiglu_bwd_combine(  # noqa: C901 -- source fused launch and storage validation
    grad_h2: torch.Tensor,  # (GM, I)
    h1: torch.Tensor,  # (GM, 2I) = [x | y]
    w: torch.Tensor,  # (G, 2I, K)
    num_tokens_per_local_expert: torch.Tensor,
    scatter_ptrs: torch.Tensor,  # int64[GM] — per-row peer-rank dx pointers
    symm_mem_buffer,
    *,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    num_sms: int | None = None,
    config: str | None = None,
    static_scheduler: bool = False,
    has_padding_sentinels: bool = False,
    pre_shuffled_rows: bool = False,
    emit_h2: bool = False,
    grad_h1: torch.Tensor | None = None,
    activation_buffer: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    estimate_recv_num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`dist_grouped_gemm_dgrad_combine` of ``grad_h1 = swiglu_bwd(grad_h2,
    h1)``, with the SwiGLU backward computed in-kernel per activation tile.

    Only the first ``sum(num_tokens_per_local_expert)`` rows of the returned
    grad_h1 are written. Returns ``(combine_buffer, grad_h1)``. ``emit_h2``
    overwrites those rows of grad_h2 with ``swiglu_fwd(h1)``.

    With ``activation_buffer``, grad_h2, h1 and grad_h1 must be contiguous
    device int64[1] byte offsets into its aligned, context-owned uint8 storage.
    The returned grad_h1 is that original offset tensor, not a dense view.
    Rows are packed in expert order with widths I, 2I and 2I respectively;
    the kernel reads the row count from num_tokens_per_local_expert. The
    planner owns disjoint input/output ranges and their lifetimes. In
    particular, W2 WGRAD must finish before an h2/grad_h1 alias is overwritten.
    No output allocation or host offset read occurs in this mode.
    """
    if scatter_ptrs.dtype != torch.int64:
        raise TypeError("scatter_ptrs must be int64")
    if not scatter_ptrs.is_cuda or scatter_ptrs.device != grad_h2.device:
        raise ValueError("scatter_ptrs must be on grad_h2.device")
    intermediate = w.shape[1] // 2
    if w.shape[1] != 2 * intermediate:
        raise ValueError("w must have an even SwiGLU output dimension")
    col_multiple = native_swiglu_bwd_col_multiple()
    if intermediate % col_multiple != 0:
        raise ValueError(
            f"fused SwiGLU-bwd combine requires I ({intermediate}) divisible "
            f"by {col_multiple}"
        )
    if activation_buffer is not None:
        if grad_h1 is None:
            raise ValueError("activation_buffer requires a planned grad_h1 offset")
        _validate_activation_buffer_args(activation_buffer, grad_h2, h1)
        _validate_activation_buffer_args(activation_buffer, grad_h1, h1)
        if (
            activation_buffer.dtype != torch.uint8
            or activation_buffer.ndim != 1
            or not activation_buffer.is_contiguous()
            or activation_buffer.data_ptr() % _ACTIVATION_BUFFER_ALIGNMENT != 0
        ):
            raise ValueError(
                "activation_buffer must be aligned contiguous uint8 storage"
            )
        for name, offset in (("grad_h2", grad_h2), ("h1", h1), ("grad_h1", grad_h1)):
            if offset.shape != (1,) or not offset.is_contiguous():
                raise ValueError(f"{name} must be a contiguous int64[1] byte offset")
        if activation_buffer.device != w.device:
            raise ValueError("activation_buffer and w must share a device")
        GM = None
    else:
        GM = grad_h2.shape[0]
        if grad_h2.shape != (GM, intermediate) or h1.shape != (GM, 2 * intermediate):
            raise ValueError(
                f"shape mismatch: grad_h2={tuple(grad_h2.shape)} "
                f"h1={tuple(h1.shape)} w={tuple(w.shape)}"
            )
        if grad_h1 is None:
            grad_h1 = torch.empty(
                (GM, 2 * intermediate), dtype=w.dtype, device=w.device
            )
        if grad_h1.shape != (GM, 2 * intermediate):
            raise ValueError("grad_h1 must match the h1 shape")
        for name, t in (("grad_h2", grad_h2), ("h1", h1), ("grad_h1", grad_h1)):
            if t.dtype != w.dtype or t.device != w.device:
                raise ValueError(f"{name} must share w dtype/device")
            if (
                t.stride(1) != 1
                or (t.stride(0) * t.element_size()) % 16 != 0
                or t.data_ptr() % 16 != 0
            ):
                raise ValueError(f"{name} must be row-major with 16-byte aligned rows")
    _fused_dist_grouped_gemm_impl(
        x=grad_h1,
        w=w,
        num_tokens_per_local_expert=num_tokens_per_local_expert,
        peer_ptrs=scatter_ptrs,
        num_out_tokens=GM,
        symm_mem_buffer=symm_mem_buffer,
        problem_type=_DGRAD,
        mode=DistGroupedGemmKernel.COMBINE_SWIGLU_BWD,
        num_sms=num_sms,
        y=None,
        x_gathered=None,
        config=config,
        static_scheduler=static_scheduler,
        name_prefix="_cute_dist_grouped_gemm_combine_dgrad_swiglu_bwd",
        activation_buffer=activation_buffer,
        conditional_execution=conditional_execution,
        estimate_recv_num_tokens=estimate_recv_num_tokens,
        swiglu_fast_math=swiglu_fast_math,
        pre_shuffled_rows=pre_shuffled_rows,
        has_padding_sentinels=has_padding_sentinels,
        swiglu_dz=grad_h2,
        swiglu_h1=h1,
        swiglu_clamped=swiglu_clamped,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        swiglu_emit_h2=emit_h2,
    )
    return symm_mem_buffer.local(), grad_h1
