# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import cutlass
import cutlass.cute as cute

from ..formats import SWIGLU_CLAMP_ALPHA_DEFAULT, SWIGLU_CLAMP_LIMIT_DEFAULT
from ._grouped_gemm_config import (
    BLOCKSCALED_SWIZZLE_GROUP_SIZE,
)
from ._quant_conversion import (
    _cvt_f16x2_to_f32x2,
    _cvt_f32x2_to_f16x2_rn,
)
from ._quant_packing import (
    BLOCK_SCALED_FORMAT_MXFP4,
    BLOCK_SCALED_FORMAT_MXFP8_E4M3,
)
from ._swiglu_quant import (
    _quantize_swiglu_fp8_row_tile_values,
    _swiglu_fp8_row_scale_offset_base,
)
from .blockscaled_grouped_gemm import (
    NVFP4,
)
from .blockscaled_quantization_common import (
    _copy_store_u32_words,
    _load_tensor_row_as_b16x2,
    _mul_f32x2,
    _sigmoid_f32,
    _swiglu_clamped_fwd_f32,
)
from .grouped_gemm_kernel import (
    _BAR_EPILOG_SYNC,
    _transpose_first_two_modes,
)
from .params import (
    DispatchQuantSync,
)
from .swiglu_fragment import _SIGMOID_BATCH, _swiglu_words
from .tile_scheduler import blockscaled_scale_row_start

# Staged uses two lanes per MX scale block; Mega keeps the lower-pressure fragment.
MEGA_BLOCKSCALED_SWIGLU_VALUES_PER_THREAD = BLOCKSCALED_SWIZZLE_GROUP_SIZE
STAGED_BLOCKSCALED_SWIGLU_VALUES_PER_THREAD = 2 * BLOCKSCALED_SWIZZLE_GROUP_SIZE


@cute.jit
def _interleaved_swiglu_tile_coordinates(
    group_linear: cutlass.Int32,
    groups_per_row: cutlass.Constexpr[int],
    epi_m: cutlass.Constexpr[int],
    epi_n: cutlass.Constexpr[int],
    tile_m_idx: cutlass.Int32,
    tile_n_idx: cutlass.Int32,
    mma_m_idx: cutlass.Constexpr[int],
    mma_n_idx: cutlass.Constexpr[int],
    subtile_idx: cutlass.Constexpr[int],
    block_size_m: cutlass.Constexpr[int],
    block_size_n: cutlass.Constexpr[int],
    num_mma_atoms_m: cutlass.Constexpr[int],
    num_mma_atoms_n: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    cluster_cta_rank: cutlass.Int32,
    cluster_split_m: cutlass.Constexpr[bool],
    subtile_axis_m: cutlass.Constexpr[bool],
    values_per_thread: cutlass.Constexpr[int],
) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32]:
    local_row = group_linear // cutlass.Int32(groups_per_row)
    local_group = group_linear - local_row * cutlass.Int32(groups_per_row)
    atom_m: cutlass.Constexpr[int] = block_size_m // num_mma_atoms_m
    atom_cta_m: cutlass.Constexpr[int] = atom_m // num_ctas
    atom_n: cutlass.Constexpr[int] = block_size_n // num_mma_atoms_n
    row_in_group = (
        tile_m_idx * cutlass.Int32(block_size_m)
        + cutlass.Int32(mma_m_idx * atom_m)
        + local_row
    )
    if cutlass.const_expr(cluster_split_m):
        row_in_group += cluster_cta_rank * cutlass.Int32(atom_cta_m)
    if cutlass.const_expr(subtile_axis_m):
        row_in_group += cutlass.Int32(subtile_idx * epi_m)
    input_col = (
        tile_n_idx * cutlass.Int32(block_size_n)
        + cutlass.Int32(mma_n_idx * atom_n)
        + local_group * cutlass.Int32(2 * values_per_thread)
    )
    if cutlass.const_expr(not subtile_axis_m):
        input_col += cutlass.Int32(subtile_idx * epi_n)
    if cutlass.const_expr(not cluster_split_m):
        input_col += cluster_cta_rank * cutlass.Int32(atom_n // num_ctas)
    return local_row, local_group, row_in_group, input_col


@cute.jit
def _load_interleaved_swiglu_values(
    sC_stage: cute.Tensor,
    local_row: cutlass.Int32,
    local_group: cutlass.Int32,
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    swizzle_group_size: cutlass.Constexpr[int],
    values_per_thread: cutlass.Constexpr[int],
) -> cute.Tensor:
    packed_words: cutlass.Constexpr[int] = swizzle_group_size // 2
    groups_per_thread: cutlass.Constexpr[int] = values_per_thread // swizzle_group_size
    interleaved_group_size: cutlass.Constexpr[int] = 2 * swizzle_group_size
    work_group_base = local_group * cutlass.Int32(2 * values_per_thread)
    vals = cute.make_rmem_tensor((1, values_per_thread), cutlass.Float32)
    for layout_group in cutlass.range_constexpr(groups_per_thread):
        gate_packed = cute.make_rmem_tensor(packed_words, cutlass.Int32)
        up_packed = cute.make_rmem_tensor(packed_words, cutlass.Int32)
        layout_group_base = work_group_base + cutlass.Int32(
            layout_group * interleaved_group_size
        )
        _load_tensor_row_as_b16x2(
            sC_stage,
            local_row,
            layout_group_base,
            gate_packed,
            swizzle_group_size,
            sC_stage.element_type,
        )
        _load_tensor_row_as_b16x2(
            sC_stage,
            local_row,
            layout_group_base + cutlass.Int32(swizzle_group_size),
            up_packed,
            swizzle_group_size,
            sC_stage.element_type,
        )
        for pair in cutlass.range_constexpr(packed_words):
            gate = _cvt_f16x2_to_f32x2(gate_packed[pair], sC_stage.element_type)
            up = _cvt_f16x2_to_f32x2(up_packed[pair], sC_stage.element_type)
            if cutlass.const_expr(clamped):
                result = (
                    _swiglu_clamped_fwd_f32(gate[0], up[0], alpha, limit, fast_math),
                    _swiglu_clamped_fwd_f32(gate[1], up[1], alpha, limit, fast_math),
                )
            else:
                sigmoid = (
                    _sigmoid_f32(gate[0], fast_math),
                    _sigmoid_f32(gate[1], fast_math),
                )
                result = _mul_f32x2(_mul_f32x2(gate, sigmoid), up)
            output_base: cutlass.Constexpr[int] = (
                layout_group * swizzle_group_size + 2 * pair
            )
            vals[0, output_base] = result[0]
            vals[0, output_base + 1] = result[1]
    return vals


@cute.jit
def store_interleaved_swiglu_smem_tile(
    tidx: cutlass.Int32,
    sC_stage: cute.Tensor,
    m_output_words: cute.Tensor,
    tile_m_idx: cutlass.Int32,
    tile_n_idx: cutlass.Int32,
    mma_m_idx: cutlass.Constexpr[int],
    mma_n_idx: cutlass.Constexpr[int],
    subtile_idx: cutlass.Constexpr[int],
    num_mma_atoms_m: cutlass.Constexpr[int],
    num_mma_atoms_n: cutlass.Constexpr[int],
    cm_start: cutlass.Int32,
    m_size: cutlass.Int32,
    n_size: cutlass.Int32,
    block_size_m: cutlass.Constexpr[int],
    block_size_n: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    cluster_cta_rank: cutlass.Int32,
    cluster_split_m: cutlass.Constexpr[bool],
    subtile_axis_m: cutlass.Constexpr[bool],
    epilogue_threads: cutlass.Constexpr[int],
    fast_math: cutlass.Constexpr[bool],
    swizzle_group_size: cutlass.Constexpr[int],
    clamped: cutlass.Constexpr[bool] = False,
    alpha: cutlass.Constexpr[float] = SWIGLU_CLAMP_ALPHA_DEFAULT,
    limit: cutlass.Constexpr[float] = SWIGLU_CLAMP_LIMIT_DEFAULT,
) -> None:
    packed_words: cutlass.Constexpr[int] = swizzle_group_size // 2
    epi_m: cutlass.Constexpr[int] = cute.size(sC_stage.shape, mode=[0])
    epi_n: cutlass.Constexpr[int] = cute.size(sC_stage.shape, mode=[1])
    if cutlass.const_expr(epi_n % (2 * swizzle_group_size) != 0):
        raise ValueError(
            "swizzled SwiGLU requires the epilogue tile width to be divisible "
            f"by {2 * swizzle_group_size}; got {epi_n}"
        )
    groups_per_row: cutlass.Constexpr[int] = epi_n // (2 * swizzle_group_size)
    total_groups: cutlass.Constexpr[int] = epi_m * groups_per_row
    # Each group of 16 gate/up values goes to 2 threads, 8 values each, so a
    # thread runs one batched sigmoid of 8 (default variant) instead of 16
    # sigmoids in sequence. It pays most at 32 rows per expert.
    # Block-scaled callers (groups of 8) keep 1 thread per group. The batched
    # sigmoid votes across the warp, so it needs whole warps.
    items_per_group: cutlass.Constexpr[int] = (
        2
        if swizzle_group_size == 2 * _SIGMOID_BATCH
        and (2 * total_groups) % 32 == 0
        and epilogue_threads % 32 == 0
        else 1
    )
    total_items: cutlass.Constexpr[int] = total_groups * items_per_group
    chunk_words: cutlass.Constexpr[int] = _SIGMOID_BATCH // 2
    passes: cutlass.Constexpr[int] = (
        total_items + epilogue_threads - 1
    ) // epilogue_threads
    for work_pass in cutlass.range_constexpr(passes):
        item = tidx + cutlass.Int32(work_pass * epilogue_threads)
        if item < cutlass.Int32(total_items):
            group_linear = item
            if cutlass.const_expr(items_per_group == 2):
                group_linear = item // cutlass.Int32(2)
            local_row, local_group, row_in_group, input_col = (
                _interleaved_swiglu_tile_coordinates(
                    group_linear,
                    groups_per_row,
                    epi_m,
                    epi_n,
                    tile_m_idx,
                    tile_n_idx,
                    mma_m_idx,
                    mma_n_idx,
                    subtile_idx,
                    block_size_m,
                    block_size_n,
                    num_mma_atoms_m,
                    num_mma_atoms_n,
                    num_ctas,
                    cluster_cta_rank,
                    cluster_split_m,
                    subtile_axis_m,
                    swizzle_group_size,
                )
            )
            stored = (row_in_group < m_size) and (
                input_col + cutlass.Int32(2 * swizzle_group_size) <= n_size
            )
            if cutlass.const_expr(items_per_group == 2):
                # This thread's 8 gate and 8 up values as `[gate | up]` b16x2
                # words; rows not stored compute on zeros (no slow sigmoid path).
                chunk_idx = item % cutlass.Int32(2)
                gate_col = local_group * cutlass.Int32(
                    2 * swizzle_group_size
                ) + chunk_idx * cutlass.Int32(_SIGMOID_BATCH)
                words = cute.make_rmem_tensor((chunk_words, 2), cutlass.Int32)
                words.fill(0)
                if stored:
                    for half in cutlass.range_constexpr(2):
                        _load_tensor_row_as_b16x2(
                            sC_stage,
                            local_row,
                            gate_col + cutlass.Int32(half * swizzle_group_size),
                            words[None, half],
                            _SIGMOID_BATCH,
                            sC_stage.element_type,
                        )
                batch_words = cute.make_rmem_tensor(chunk_words, cutlass.Uint32)
                _swiglu_words(
                    words,
                    batch_words,
                    fast_math,
                    clamped,
                    alpha,
                    limit,
                    sC_stage.element_type,
                )
                if stored:
                    _copy_store_u32_words(
                        m_output_words,
                        cutlass.Int64(cm_start + row_in_group),
                        input_col // cutlass.Int32(4)
                        + chunk_idx * cutlass.Int32(chunk_words),
                        batch_words,
                        chunk_words,
                    )
            elif stored:
                vals = _load_interleaved_swiglu_values(
                    sC_stage,
                    local_row,
                    local_group,
                    fast_math,
                    clamped,
                    alpha,
                    limit,
                    swizzle_group_size,
                    swizzle_group_size,
                )
                output_words = cute.make_rmem_tensor(packed_words, cutlass.Uint32)
                for pair in cutlass.range_constexpr(packed_words):
                    output_words[pair] = _cvt_f32x2_to_f16x2_rn(
                        vals[0, 2 * pair], vals[0, 2 * pair + 1], sC_stage.element_type
                    )
                # Int64: m_output_words may have a static row stride.
                _copy_store_u32_words(
                    m_output_words,
                    cutlass.Int64(cm_start + row_in_group),
                    input_col // cutlass.Int32(4),
                    output_words,
                    packed_words,
                )


@cute.jit
def store_interleaved_swiglu_epilogue(
    epilogue_tidx: cutlass.Int32,
    sC_stage: cute.Tensor,
    swapped_sC_stage: cute.Tensor,
    m_output_words: cute.Tensor,
    tile_m_idx: cutlass.Int32,
    tile_n_idx: cutlass.Int32,
    swapped_tile_n_idx: cutlass.Int32,
    swapped_block_size_n: cutlass.Constexpr[int],
    swapped_cluster_cta_rank: cutlass.Int32,
    mma_m_idx: cutlass.Constexpr[int],
    mma_n_idx: cutlass.Constexpr[int],
    subtile_idx: cutlass.Constexpr[int],
    num_mma_atoms_m: cutlass.Constexpr[int],
    num_mma_atoms_n: cutlass.Constexpr[int],
    cm_start: cutlass.Int32,
    m_size: cutlass.Int32,
    n_size: cutlass.Int32,
    block_size_m: cutlass.Constexpr[int],
    block_size_n: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    cluster_cta_rank: cutlass.Int32,
    epilogue_threads: cutlass.Constexpr[int],
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    swap_ab: cutlass.Constexpr[bool],
) -> None:
    # Under SWAP_AB the feature axis becomes the output column axis, and the
    # staged and mega kernels tile it differently: staged indexes whole
    # BLOCK_SIZE_M blocks and splits each MMA atom across the cluster, while
    # mega indexes per-CTA chunks with the rank already folded into
    # swapped_tile_n_idx. The caller therefore supplies the column block size
    # and the rank that offsets it.
    if cutlass.const_expr(swap_ab):
        store_interleaved_swiglu_smem_tile(
            tidx=epilogue_tidx,
            sC_stage=swapped_sC_stage,
            m_output_words=m_output_words,
            tile_m_idx=tile_n_idx,
            tile_n_idx=swapped_tile_n_idx,
            mma_m_idx=mma_n_idx,
            mma_n_idx=mma_m_idx,
            subtile_idx=subtile_idx,
            num_mma_atoms_m=num_mma_atoms_n,
            num_mma_atoms_n=num_mma_atoms_m,
            cm_start=cm_start,
            m_size=m_size,
            n_size=n_size,
            block_size_m=block_size_n,
            block_size_n=swapped_block_size_n,
            num_ctas=num_ctas,
            cluster_cta_rank=swapped_cluster_cta_rank,
            cluster_split_m=False,
            subtile_axis_m=True,
            epilogue_threads=epilogue_threads,
            fast_math=fast_math,
            clamped=clamped,
            alpha=alpha,
            limit=limit,
            swizzle_group_size=BLOCKSCALED_SWIZZLE_GROUP_SIZE,
        )
    else:
        store_interleaved_swiglu_smem_tile(
            tidx=epilogue_tidx,
            sC_stage=sC_stage,
            m_output_words=m_output_words,
            tile_m_idx=tile_m_idx,
            tile_n_idx=tile_n_idx,
            mma_m_idx=mma_m_idx,
            mma_n_idx=mma_n_idx,
            subtile_idx=subtile_idx,
            num_mma_atoms_m=num_mma_atoms_m,
            num_mma_atoms_n=num_mma_atoms_n,
            cm_start=cm_start,
            m_size=m_size,
            n_size=n_size,
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            num_ctas=num_ctas,
            cluster_cta_rank=cluster_cta_rank,
            cluster_split_m=True,
            subtile_axis_m=False,
            epilogue_threads=epilogue_threads,
            fast_math=fast_math,
            clamped=clamped,
            alpha=alpha,
            limit=limit,
            swizzle_group_size=BLOCKSCALED_SWIZZLE_GROUP_SIZE,
        )


@cute.jit
def quantize_interleaved_swiglu_mx_smem_tile(
    tidx: cutlass.Int32,
    sC_stage: cute.Tensor,
    m_row_q_words: cute.Tensor,
    m_row_scale: cute.Tensor,
    tile_m_idx: cutlass.Int32,
    tile_n_idx: cutlass.Int32,
    mma_m_idx: cutlass.Constexpr[int],
    mma_n_idx: cutlass.Constexpr[int],
    subtile_idx: cutlass.Constexpr[int],
    num_mma_atoms_m: cutlass.Constexpr[int],
    num_mma_atoms_n: cutlass.Constexpr[int],
    cm_start: cutlass.Int32,
    scale_cm_start: cutlass.Int32,
    m_size: cutlass.Int32,
    n_size: cutlass.Int32,
    block_size_m: cutlass.Constexpr[int],
    block_size_n: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    cluster_cta_rank: cutlass.Int32,
    cluster_split_m: cutlass.Constexpr[bool],
    subtile_axis_m: cutlass.Constexpr[bool],
    epilogue_threads: cutlass.Constexpr[int],
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    format_id: cutlass.Constexpr[int],
    qdata_elems_per_word: cutlass.Constexpr[int],
    values_per_thread: cutlass.Constexpr[int],
    store_cache_modifier: cutlass.Constexpr = "cs",
) -> None:
    if cutlass.const_expr(
        format_id not in (BLOCK_SCALED_FORMAT_MXFP8_E4M3, BLOCK_SCALED_FORMAT_MXFP4)
    ):
        raise ValueError("interleaved SwiGLU quantization requires an MX format")
    epi_m: cutlass.Constexpr[int] = cute.size(sC_stage.shape, mode=[0])
    epi_n: cutlass.Constexpr[int] = cute.size(sC_stage.shape, mode=[1])
    if cutlass.const_expr(
        values_per_thread
        not in (
            BLOCKSCALED_SWIZZLE_GROUP_SIZE,
            2 * BLOCKSCALED_SWIZZLE_GROUP_SIZE,
        )
    ):
        raise ValueError(
            "MX epilogue values per thread must be 8 or 16 for the G8 swizzled layout"
        )
    interleaved_group_size: cutlass.Constexpr[int] = 2 * values_per_thread
    groups_per_row: cutlass.Constexpr[int] = epi_n // interleaved_group_size
    if cutlass.const_expr(epi_n not in (64, 128)):
        raise ValueError("MX epilogue quantization requires 64 or 128 columns")
    total_groups: cutlass.Constexpr[int] = epi_m * groups_per_row
    passes: cutlass.Constexpr[int] = (
        total_groups + epilogue_threads - 1
    ) // epilogue_threads
    row_lanes: cutlass.Constexpr[int] = 32 // groups_per_row
    groups_per_scale: cutlass.Constexpr[int] = 32 // values_per_thread
    row_reduce_stages: cutlass.Constexpr[int] = groups_per_scale.bit_length() - 1

    for work_pass in cutlass.range_constexpr(passes):
        group_linear = tidx + cutlass.Int32(work_pass * epilogue_threads)
        if group_linear < cutlass.Int32(total_groups):
            local_row, local_group, row_in_group, input_col = (
                _interleaved_swiglu_tile_coordinates(
                    group_linear,
                    groups_per_row,
                    epi_m,
                    epi_n,
                    tile_m_idx,
                    tile_n_idx,
                    mma_m_idx,
                    mma_n_idx,
                    subtile_idx,
                    block_size_m,
                    block_size_n,
                    num_mma_atoms_m,
                    num_mma_atoms_n,
                    num_ctas,
                    cluster_cta_rank,
                    cluster_split_m,
                    subtile_axis_m,
                    values_per_thread,
                )
            )
            if (row_in_group < m_size) and (
                input_col + cutlass.Int32(interleaved_group_size) <= n_size
            ):
                vals = _load_interleaved_swiglu_values(
                    sC_stage,
                    local_row,
                    local_group,
                    fast_math,
                    clamped,
                    alpha,
                    limit,
                    BLOCKSCALED_SWIZZLE_GROUP_SIZE,
                    values_per_thread,
                )

                h2_col = input_col // cutlass.Int32(2)
                scale_col = h2_col // cutlass.Int32(32)
                subgroup_lane = local_group % cutlass.Int32(groups_per_scale)
                lane_in_warp = tidx % cutlass.Int32(32)
                row_lane = lane_in_warp // cutlass.Int32(groups_per_row)
                data_warp_row_start = cm_start + row_in_group - row_lane
                scale_group_row = row_in_group - row_lane
                scale_warp_row_start = blockscaled_scale_row_start(
                    scale_cm_start,
                    scale_group_row,
                    True,
                    block_size_m,
                )
                scale_offset, scale_store_lane = _swiglu_fp8_row_scale_offset_base(
                    scale_warp_row_start,
                    row_lane,
                    scale_col,
                    cutlass.Int32(0),
                    subgroup_lane,
                    groups_per_scale,
                    (n_size // cutlass.Int32(2)) // cutlass.Int32(32),
                    cutlass.Int32(0),
                )
                scale_recips = cute.make_rmem_tensor(1, cutlass.Float32)
                q_vals = cute.make_rmem_tensor(values_per_thread, cutlass.Float32)
                q_f32x2_packed = cute.make_rmem_tensor(
                    values_per_thread // 2, cutlass.Uint64
                )
                _quantize_swiglu_fp8_row_tile_values(
                    vals=vals,
                    mRowQWords=m_row_q_words,
                    mRowScale=m_row_scale,
                    scale_recips=scale_recips,
                    q_vals=q_vals,
                    q_f32x2_packed=q_f32x2_packed,
                    row_start=data_warp_row_start,
                    row_lane=row_lane,
                    q_word=h2_col // cutlass.Int32(qdata_elems_per_word),
                    scale_offset_base=scale_offset,
                    scale_store_lane=scale_store_lane,
                    row_lanes_cfg=row_lanes,
                    row_reps_cfg=1,
                    num_elems=values_per_thread,
                    lane_pairs=values_per_thread // 2,
                    row_reduce_stages=row_reduce_stages,
                    format_id=format_id,
                    packed_f32x2=False,
                    precomputed_amax=False,
                    paired_scale_compute=False,
                    qdata_elems_per_word=qdata_elems_per_word,
                    cache_modifier=store_cache_modifier,
                )


@cute.jit
def quantize_interleaved_swiglu_mx_epilogue(
    epilogue_tidx: cutlass.Int32,
    sC_stage: cute.Tensor,
    swapped_sC_stage: cute.Tensor,
    m_row_q_words: cute.Tensor,
    m_row_scale: cute.Tensor,
    tile_m_idx: cutlass.Int32,
    tile_n_idx: cutlass.Int32,
    swapped_tile_n_idx: cutlass.Int32,
    swapped_block_size_n: cutlass.Constexpr[int],
    swapped_cluster_cta_rank: cutlass.Int32,
    mma_m_idx: cutlass.Constexpr[int],
    mma_n_idx: cutlass.Constexpr[int],
    subtile_idx: cutlass.Constexpr[int],
    num_mma_atoms_m: cutlass.Constexpr[int],
    num_mma_atoms_n: cutlass.Constexpr[int],
    cm_start: cutlass.Int32,
    scale_cm_start: cutlass.Int32,
    m_size: cutlass.Int32,
    n_size: cutlass.Int32,
    block_size_m: cutlass.Constexpr[int],
    block_size_n: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    cluster_cta_rank: cutlass.Int32,
    epilogue_threads: cutlass.Constexpr[int],
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    format_id: cutlass.Constexpr[int],
    qdata_elems_per_word: cutlass.Constexpr[int],
    swap_ab: cutlass.Constexpr[bool],
    values_per_thread: cutlass.Constexpr[int],
    store_cache_modifier: cutlass.Constexpr = "cs",
) -> None:
    # Under SWAP_AB the feature axis becomes the output column axis, and the
    # staged and mega kernels tile it differently: staged indexes whole
    # BLOCK_SIZE_M blocks and splits each MMA atom across the cluster, while
    # mega indexes per-CTA chunks with the rank already folded into
    # swapped_tile_n_idx. The caller therefore supplies the column block size
    # and the rank that offsets it.
    if cutlass.const_expr(swap_ab):
        quantize_interleaved_swiglu_mx_smem_tile(
            tidx=epilogue_tidx,
            sC_stage=swapped_sC_stage,
            m_row_q_words=m_row_q_words,
            m_row_scale=m_row_scale,
            tile_m_idx=tile_n_idx,
            tile_n_idx=swapped_tile_n_idx,
            mma_m_idx=mma_n_idx,
            mma_n_idx=mma_m_idx,
            subtile_idx=subtile_idx,
            num_mma_atoms_m=num_mma_atoms_n,
            num_mma_atoms_n=num_mma_atoms_m,
            cm_start=cm_start,
            scale_cm_start=scale_cm_start,
            m_size=m_size,
            n_size=n_size,
            block_size_m=block_size_n,
            block_size_n=swapped_block_size_n,
            num_ctas=num_ctas,
            cluster_cta_rank=swapped_cluster_cta_rank,
            cluster_split_m=False,
            subtile_axis_m=True,
            epilogue_threads=epilogue_threads,
            fast_math=fast_math,
            clamped=clamped,
            alpha=alpha,
            limit=limit,
            format_id=format_id,
            qdata_elems_per_word=qdata_elems_per_word,
            values_per_thread=values_per_thread,
            store_cache_modifier=store_cache_modifier,
        )
    else:
        quantize_interleaved_swiglu_mx_smem_tile(
            tidx=epilogue_tidx,
            sC_stage=sC_stage,
            m_row_q_words=m_row_q_words,
            m_row_scale=m_row_scale,
            tile_m_idx=tile_m_idx,
            tile_n_idx=tile_n_idx,
            mma_m_idx=mma_m_idx,
            mma_n_idx=mma_n_idx,
            subtile_idx=subtile_idx,
            num_mma_atoms_m=num_mma_atoms_m,
            num_mma_atoms_n=num_mma_atoms_n,
            cm_start=cm_start,
            scale_cm_start=scale_cm_start,
            m_size=m_size,
            n_size=n_size,
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            num_ctas=num_ctas,
            cluster_cta_rank=cluster_cta_rank,
            cluster_split_m=True,
            subtile_axis_m=False,
            epilogue_threads=epilogue_threads,
            fast_math=fast_math,
            clamped=clamped,
            alpha=alpha,
            limit=limit,
            format_id=format_id,
            qdata_elems_per_word=qdata_elems_per_word,
            values_per_thread=values_per_thread,
            store_cache_modifier=store_cache_modifier,
        )


@cute.jit
def _wait_pending_epilog_store(
    EPILOG_WARP_IDS: cutlass.Constexpr,
    EPILOG_WG_THREADS: cutlass.Constexpr[int],
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == EPILOG_WARP_IDS[0]:
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    cute.arch.barrier(
        barrier_id=_BAR_EPILOG_SYNC,
        number_of_threads=EPILOG_WG_THREADS,
    )


@cute.jit
def _signal_fc13_epilogue(
    activation_tile_idx,
    act_tile_start,
    is_leader,
    activation_quant: DispatchQuantSync,
    INTERLEAVED_FC13: cutlass.Constexpr[bool],
    format: cutlass.Constexpr,
    EPILOG_WG_THREADS: cutlass.Constexpr[int],
) -> None:
    done_counter_offset = activation_quant.done_counter_offsets[1]
    if cutlass.const_expr(INTERLEAVED_FC13 and format is not NVFP4):
        # Forward-quant warps exit in this mode, leaving slot 0 single-writer.
        done_counter_offset = activation_quant.done_counter_offsets[0]
    _signal_output_tile(
        activation_quant.done_counter,
        done_counter_offset + act_tile_start,
        activation_tile_idx,
        is_leader,
        EPILOG_WG_THREADS=EPILOG_WG_THREADS,
    )


@cute.jit
def _signal_output_tile(
    counter: cute.Tensor,
    counter_offset,
    tile_idx,
    is_leader,
    EPILOG_WG_THREADS: cutlass.Constexpr[int],
) -> None:
    # The counter publishes stores issued by the entire epilogue warpgroup:
    # bar.sync orders them before the elected thread's cumulative gpu-scope
    # release, so only that thread waits for the stores to drain.
    cute.arch.barrier(
        barrier_id=_BAR_EPILOG_SYNC,
        number_of_threads=EPILOG_WG_THREADS,
    )
    if is_leader:
        with cute.arch.elect_one():
            ptr = cute.recast_ptr(
                counter.iterator + counter_offset + tile_idx,
                dtype=cutlass.Uint32,
            )
            cute.arch.atomic_add(
                ptr,
                cutlass.Uint32(1),
                sem="release",
                scope="gpu",
            )


@cute.jit
def blockscaled_epilog_postprocess_store(
    epilogue_tidx: cutlass.Int32,
    sC_stage: cute.Tensor,
    output_words: cute.Tensor,
    output_scale: cute.Tensor,
    tile_m_idx: cutlass.Int32,
    tile_n_idx: cutlass.Int32,
    swapped_tile_n_idx: cutlass.Int32,
    mma_m_idx: cutlass.Constexpr[int],
    mma_n_idx: cutlass.Constexpr[int],
    subtile_idx: cutlass.Constexpr[int],
    num_mma_atoms_m: cutlass.Constexpr[int],
    num_mma_atoms_n: cutlass.Constexpr[int],
    cm_start: cutlass.Int32,
    scale_cm_start: cutlass.Int32,
    m_size: cutlass.Int32,
    n_size: cutlass.Int32,
    cluster_cta_rank: cutlass.Int32,
    swapped_cluster_cta_rank: cutlass.Int32,
    swapped_block_size_n: cutlass.Constexpr[int],
    values_per_thread: cutlass.Constexpr[int],
    is_nvfp4: cutlass.Constexpr[bool],
    block_size_m: cutlass.Constexpr[int],
    block_size_n: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    epilogue_threads: cutlass.Constexpr[int],
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    swap_ab: cutlass.Constexpr[bool],
    format_id: cutlass.Constexpr[int],
    qdata_elems_per_word: cutlass.Constexpr[int],
    store_cache_modifier: cutlass.Constexpr = "cs",
) -> None:
    """Post-process one C stage through the SwiGLU store/quant epilogue.

    Shared body of the staged (dist) and chunked Mega overrides of
    ``_blockscaled_epilog_postprocess_store``; the swapped-axis arguments
    carry the per-kernel tiling difference. ``store_cache_modifier``
    defaults to evict-first streaming; the activation ring keeps h2 lines
    cacheable so FC2's reads can hit L2.
    """
    swapped_sC_stage = cute.make_tensor(
        sC_stage.iterator,
        _transpose_first_two_modes(sC_stage.layout),
    )
    if cutlass.const_expr(is_nvfp4):
        store_interleaved_swiglu_epilogue(
            epilogue_tidx=epilogue_tidx,
            sC_stage=sC_stage,
            swapped_sC_stage=swapped_sC_stage,
            m_output_words=output_words,
            tile_m_idx=tile_m_idx,
            tile_n_idx=tile_n_idx,
            swapped_tile_n_idx=swapped_tile_n_idx,
            swapped_block_size_n=swapped_block_size_n,
            swapped_cluster_cta_rank=swapped_cluster_cta_rank,
            mma_m_idx=mma_m_idx,
            mma_n_idx=mma_n_idx,
            subtile_idx=subtile_idx,
            num_mma_atoms_m=num_mma_atoms_m,
            num_mma_atoms_n=num_mma_atoms_n,
            cm_start=cm_start,
            m_size=m_size,
            n_size=n_size,
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            num_ctas=num_ctas,
            cluster_cta_rank=cluster_cta_rank,
            epilogue_threads=epilogue_threads,
            fast_math=fast_math,
            clamped=clamped,
            alpha=alpha,
            limit=limit,
            swap_ab=swap_ab,
        )
        return
    quantize_interleaved_swiglu_mx_epilogue(
        epilogue_tidx=epilogue_tidx,
        sC_stage=sC_stage,
        swapped_sC_stage=swapped_sC_stage,
        m_row_q_words=output_words,
        m_row_scale=output_scale,
        store_cache_modifier=store_cache_modifier,
        tile_m_idx=tile_m_idx,
        tile_n_idx=tile_n_idx,
        swapped_tile_n_idx=swapped_tile_n_idx,
        swapped_block_size_n=swapped_block_size_n,
        swapped_cluster_cta_rank=swapped_cluster_cta_rank,
        mma_m_idx=mma_m_idx,
        mma_n_idx=mma_n_idx,
        subtile_idx=subtile_idx,
        num_mma_atoms_m=num_mma_atoms_m,
        num_mma_atoms_n=num_mma_atoms_n,
        cm_start=cm_start,
        scale_cm_start=scale_cm_start,
        m_size=m_size,
        n_size=n_size,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        num_ctas=num_ctas,
        cluster_cta_rank=cluster_cta_rank,
        epilogue_threads=epilogue_threads,
        fast_math=fast_math,
        clamped=clamped,
        alpha=alpha,
        limit=limit,
        format_id=format_id,
        qdata_elems_per_word=qdata_elems_per_word,
        swap_ab=swap_ab,
        values_per_thread=values_per_thread,
    )
