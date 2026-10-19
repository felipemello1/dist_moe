# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Warp-role bodies for the fused block-scaled grouped GEMMs.

The TMA-producer and MMA-consumer role bodies (plus their tile gating) shared
by the Mega and Chunked-Mega kernels. The per-tile load/consume primitives
live in ``blockscaled_gemm_tiles``; epilogue signaling lives in
``swiglu_epilogue``; quant-progress waits live in ``dispatch_quant`` and
``combine_swiglu_quant``.
"""

import cutlass
import cutlass.cute as cute

from . import sm103_blockscaled_helpers as sm103
from .blockscaled_gemm_tiles import (
    _blockscaled_mma_consumer_tile,
    _blockscaled_tma_load_tile,
    _sm103_mma_consumer_tile,
)
from .blockscaled_grouped_gemm import (
    NVFP4,
)
from .combine_swiglu_quant import (
    _mega_combine_wait_quant_ready,
    _mega_forward_h2_tile_wait,
)
from .dispatch_quant import (
    _mega_activation_wgrad_tile_wait,
    _mega_forward_wait_h1_tile,
    _mega_wait_quant_ready,
)
from .grouped_gemm import (
    _FPROP,
)
from .grouped_gemm_kernel import _weight_l2_policy
from .params import (
    ChunkedGemmWarpParams,
    DispatchQuantSync as _MegaDispatchQuantSync,
    GroupedGemmMmaPipeline as _MegaMmaPipeline,
    GroupedGemmPipelineSync,
    GroupedGemmPipelineSync as _MegaPipelineSync,
    GroupedGemmProblem,
    GroupedGemmProblem as _MegaProblem,
    GroupedGemmTmaPipeline as _MegaTmaPipeline,
    MegaPipelineParams,
)
from .tensormap_fold import wait_folded_tensormaps
from .tile_scheduler import (
    _get_bufidx_phase,
    _subtile_coords,
    _subtile_is_valid,
    _subtile_valid_n,
    CHUNKED_MEGA_WORK_INFO_ROLE_ALL,
    CHUNKED_MEGA_WORK_INFO_ROLE_MMA,
    CHUNKED_MEGA_WORK_INFO_ROLE_TMA_B,
    ChunkedMegaProblemVisitor,
    ChunkedMegaWorkTileInfo,
    load_chunked_mega_work_info,
    load_mega_work_info,
    MegaDynamicScheduler,
    MegaProblemVisitor,
    MegaStaticScheduler,
    publish_chunked_mega_work_info,
    publish_mega_work_info,
)
from .weight_borrow import _weight_borrow_gate


@cute.jit
def _mega_problem_specs(
    M,
    N,
    K,
    MEGA_DGRAD_PROBLEM_TYPE: cutlass.Constexpr[int],
    MEGA_FORWARD_MODE: cutlass.Constexpr[int],
    MEGA_WGRAD_PROBLEM_TYPE: cutlass.Constexpr[int],
    mode: cutlass.Constexpr[int],
):
    if cutlass.const_expr(mode == MEGA_FORWARD_MODE):
        return ((M, 2 * N, K), (M, K, N)), (_FPROP, _FPROP)
    return ((M, N, K), (K, N, M)), (
        MEGA_DGRAD_PROBLEM_TYPE,
        MEGA_WGRAD_PROBLEM_TYPE,
    )


@cute.jit
def _mega_tma_producer_body(  # noqa: C901
    params: MegaPipelineParams,
    dgrad: _MegaTmaPipeline,
    wgrad: _MegaTmaPipeline,
    problem: _MegaProblem,
    sync: _MegaPipelineSync,
    dispatch_quant: _MegaDispatchQuantSync,
    activation_quant: _MegaDispatchQuantSync,
    tensormap_manager,
    tensormaps: cute.Tensor,
    combine: cutlass.Constexpr[bool],
    activation_dispatch: cutlass.Constexpr[bool],
    use_device_tensormaps: cutlass.Constexpr[bool],
    work_info_smem_ptr: cute.Pointer,
    format: cutlass.Constexpr,
) -> None:
    M, N, K = problem.mnk
    row_done_counter_offset, col_done_counter_offset = (
        dispatch_quant.done_counter_offsets
    )
    accum_cnt_smem = cutlass.Int32(0)
    accum_cnt_out = cutlass.Int32(0)

    if cutlass.const_expr(params.STATIC_SCHEDULER):
        scheduler = MegaStaticScheduler.create_producer(params.NUM_CTAS)
    else:
        scheduler = MegaDynamicScheduler.create_producer(
            sync.counter_ptr,
            sync.tile_cta_bar_mbar,
            sync.tile_id_smem_ptr,
            sync.tile_consumer_mbar,
            sync.tile_producer_mbar,
            sync.cluster_cta_rank,
            params.NUM_CTAS,
            params.NUM_TILE_BUFFERS,
            params.NUM_TILE_CTA_BARS,
        )
    tile_idx = scheduler.initial_work_tile_info()

    tensormap_base_pair = (
        params.MEGA_DGRAD_TENSORMAP_BASE,
        params.MEGA_WGRAD_TENSORMAP_BASE,
    )
    gemm_mnk_pair, gemm_problem_type_pair = _mega_problem_specs(
        M,
        N,
        K,
        MEGA_DGRAD_PROBLEM_TYPE=params.MEGA_DGRAD_PROBLEM_TYPE,
        MEGA_FORWARD_MODE=params.MEGA_FORWARD_MODE,
        MEGA_WGRAD_PROBLEM_TYPE=params.MEGA_WGRAD_PROBLEM_TYPE,
        mode=params.mode,
    )
    group_tile_size: cutlass.Constexpr[int] = (
        params.BLOCK_SIZE_N if params.SWAP_AB else params.BLOCK_SIZE_M
    )
    visitor = MegaProblemVisitor.create(
        problem.split_sizes,
        gemm_mnk_pair,
        problem.groups,
        gemm_problem_type_pair,
        params.BLOCK_SIZE_M,
        params.BLOCK_SIZE_N,
        params.BLOCK_SIZE_K,
        params.force_n_major,
        params.num_n_clusters,
        problem.local_rank,
        params.schedule_world_size,
        params.SWAP_AB,
        group_tile_size,
    )
    work = visitor.get_work(tile_idx)

    while work.is_valid_tile:
        g = work.group_idx
        dgrad_act_off_tiles = work.act_tile_prefix_0
        wgrad_act_off_tiles = work.act_tile_prefix_1
        for w in cutlass.range_constexpr(2):
            pipeline = (dgrad, wgrad)[w]
            act_off_tiles = (dgrad_act_off_tiles, wgrad_act_off_tiles)[w]
            m_size = work.m_size
            n_size = work.n_size
            num_k_tiles = work.num_k_tiles
            base = tensormap_base_pair[w]

            if work.problem_idx == cutlass.Int32(w):
                if cutlass.const_expr(use_device_tensormaps):
                    tensormap_a_g_ptr = tensormap_manager.get_tensormap_ptr(
                        tensormaps[(g, base, None)].iterator
                    )
                    tensormap_b_g_ptr = tensormap_manager.get_tensormap_ptr(
                        tensormaps[(g, base + 1, None)].iterator
                    )
                    tensormap_sfa_g_ptr = tensormap_manager.get_tensormap_ptr(
                        tensormaps[(g, base + 2, None)].iterator
                    )
                    tensormap_sfb_g_ptr = tensormap_manager.get_tensormap_ptr(
                        tensormaps[(g, base + 3, None)].iterator
                    )
                    tma_desc_ptrs = (
                        tensormap_manager.get_tensormap_ptr(
                            tensormap_a_g_ptr,
                            cute.AddressSpace.generic,
                        ),
                        tensormap_manager.get_tensormap_ptr(
                            tensormap_b_g_ptr,
                            cute.AddressSpace.generic,
                        ),
                        tensormap_manager.get_tensormap_ptr(
                            tensormap_sfa_g_ptr,
                            cute.AddressSpace.generic,
                        ),
                        tensormap_manager.get_tensormap_ptr(
                            tensormap_sfb_g_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_a_g_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_b_g_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_sfa_g_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_sfb_g_ptr)
                else:
                    tma_desc_ptrs = ()
                while (
                    work.is_valid_tile
                    and work.group_idx == g
                    and work.problem_idx == cutlass.Int32(w)
                ):
                    if cutlass.const_expr(not params.STATIC_SCHEDULER):
                        publish_mega_work_info(
                            work,
                            work_info_smem_ptr,
                            accum_cnt_out % params.NUM_TILE_BUFFERS,
                        )
                    scheduler.producer_publish_tile(accum_cnt_out)
                    tile_m_idx = work.tile_m_idx
                    tile_n_idx = work.tile_n_idx
                    if cutlass.const_expr(params.mode == params.MEGA_FORWARD_MODE):
                        if cutlass.const_expr(w == 0):
                            _mega_wait_quant_ready(
                                wait_axis=0,
                                tile_m_idx=tile_m_idx,
                                tile_n_idx=tile_n_idx,
                                dgrad_act_off_tiles=dgrad_act_off_tiles,
                                g=g,
                                split_sizes=problem.split_sizes,
                                m_size=m_size,
                                n_size=n_size,
                                dispatch_dim=K,
                                mDoneCounter=dispatch_quant.done_counter,
                                row_done_counter_offset=row_done_counter_offset,
                                col_done_counter_offset=col_done_counter_offset,
                                BLOCK_SIZE_M=params.BLOCK_SIZE_M,
                                BLOCK_SIZE_N=params.BLOCK_SIZE_N,
                                DISPATCH_QUANT_SCALE_COLS_PER_TILE=params.DISPATCH_QUANT_SCALE_COLS_PER_TILE,
                                SWAP_AB=params.SWAP_AB,
                                sf_vec_size=params.sf_vec_size,
                            )
                        else:
                            if cutlass.const_expr(params.SWAP_AB):
                                h2_act_tile_idx = tile_n_idx
                                H2_ACT_BLOCK_SIZE: cutlass.Constexpr[int] = (
                                    params.BLOCK_SIZE_N
                                )
                                h2_act_rows = n_size
                            else:
                                h2_act_tile_idx = tile_m_idx
                                H2_ACT_BLOCK_SIZE: cutlass.Constexpr[int] = (
                                    params.BLOCK_SIZE_M
                                )
                                h2_act_rows = m_size
                            _mega_forward_h2_tile_wait(
                                mDoneCounter=activation_quant.done_counter,
                                row_done_counter_offset=(
                                    activation_quant.done_counter_offsets[0]
                                ),
                                act_tile_idx=h2_act_tile_idx,
                                act_tile_start_per_group=dgrad_act_off_tiles,
                                act_rows=h2_act_rows,
                                h2_dim=N,
                                ACT_BLOCK_SIZE=H2_ACT_BLOCK_SIZE,
                                COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE=params.COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE,
                                format=format,
                                sf_vec_size=params.sf_vec_size,
                            )
                    elif cutlass.const_expr(combine):
                        _mega_combine_wait_quant_ready(
                            wait_axis=w,
                            tile_m_idx=tile_m_idx,
                            tile_n_idx=tile_n_idx,
                            dgrad_act_off_tiles=dgrad_act_off_tiles,
                            g=g,
                            split_sizes=problem.split_sizes,
                            m_size=m_size,
                            n_size=n_size,
                            dxy_dim=K,
                            mDoneCounter=dispatch_quant.done_counter,
                            col_done_counter_offset=col_done_counter_offset,
                            BLOCK_SIZE_M=params.BLOCK_SIZE_M,
                            BLOCK_SIZE_N=params.BLOCK_SIZE_N,
                            SWAP_AB=params.SWAP_AB,
                            sf_vec_size=params.sf_vec_size,
                            COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE=params.COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE,
                        )
                        if cutlass.const_expr(activation_dispatch and w == 1):
                            if cutlass.const_expr(params.SWAP_AB):
                                activation_feature_tile_idx = tile_m_idx
                                ACTIVATION_FEATURE_TILE_ELEMS: cutlass.Constexpr[
                                    int
                                ] = params.BLOCK_SIZE_M
                                activation_dim = m_size
                            else:
                                activation_feature_tile_idx = tile_n_idx
                                ACTIVATION_FEATURE_TILE_ELEMS = params.BLOCK_SIZE_N
                                activation_dim = n_size
                            _mega_activation_wgrad_tile_wait(
                                mDoneCounter=activation_quant.done_counter,
                                col_done_counter_offset=(
                                    activation_quant.done_counter_offsets[1]
                                ),
                                g=g,
                                feature_tile_idx=activation_feature_tile_idx,
                                feature_tile_elems=ACTIVATION_FEATURE_TILE_ELEMS,
                                activation_dim=activation_dim,
                                group_row_blocks=(
                                    cutlass.Int32(problem.split_sizes[g])
                                    // cutlass.Int32(params.sf_vec_size)
                                ),
                                DISPATCH_QUANT_SCALE_COLS_PER_TILE=params.DISPATCH_QUANT_SCALE_COLS_PER_TILE,
                                sf_vec_size=params.sf_vec_size,
                            )
                    else:
                        _mega_wait_quant_ready(
                            wait_axis=w,
                            tile_m_idx=tile_m_idx,
                            tile_n_idx=tile_n_idx,
                            dgrad_act_off_tiles=dgrad_act_off_tiles,
                            g=g,
                            split_sizes=problem.split_sizes,
                            m_size=m_size,
                            n_size=n_size,
                            dispatch_dim=K,
                            mDoneCounter=dispatch_quant.done_counter,
                            row_done_counter_offset=row_done_counter_offset,
                            col_done_counter_offset=col_done_counter_offset,
                            BLOCK_SIZE_M=params.BLOCK_SIZE_M,
                            BLOCK_SIZE_N=params.BLOCK_SIZE_N,
                            DISPATCH_QUANT_SCALE_COLS_PER_TILE=params.DISPATCH_QUANT_SCALE_COLS_PER_TILE,
                            SWAP_AB=params.SWAP_AB,
                            sf_vec_size=params.sf_vec_size,
                        )
                    if cutlass.const_expr(use_device_tensormaps):
                        tile_axes = (
                            tile_m_idx,
                            tile_n_idx,
                            tile_m_idx,
                            tile_n_idx,
                        )
                        group_axes = (0, 0, 0, 0)
                    elif cutlass.const_expr(params.SWAP_AB):
                        tile_axes = (
                            tile_m_idx,
                            act_off_tiles + tile_n_idx,
                            tile_m_idx,
                            act_off_tiles + tile_n_idx,
                        )
                        group_axes = (g, 0, g, 0)
                    else:
                        tile_axes = (
                            act_off_tiles + tile_m_idx,
                            tile_n_idx,
                            act_off_tiles + tile_m_idx,
                            tile_n_idx,
                        )
                        group_axes = (0, g, 0, g)
                    accum_cnt_smem = _blockscaled_tma_load_tile(
                        pipeline,
                        sync,
                        num_k_tiles,
                        accum_cnt_smem,
                        tile_axes,
                        group_axes,
                        tma_desc_ptrs,
                        use_device_tensormaps,
                        KLOOP_UNROLL=params.KLOOP_UNROLL,
                        NUM_CTAS=params.NUM_CTAS,
                        NUM_SMEM_BUFFERS=params.NUM_SMEM_BUFFERS,
                        weights_ahead=cutlass.Int32(0),
                    )

                    accum_cnt_out += cutlass.Int32(1)
                    scheduler.producer_wait_tile_released(accum_cnt_out)
                    tile_idx = scheduler.advance_producer(accum_cnt_out)
                    work = visitor.get_work(tile_idx)

    scheduler.producer_publish_termination(accum_cnt_out)


@cute.jit
def _mega_mma_consumer_body(  # noqa: C901
    params: MegaPipelineParams,
    dgrad: _MegaMmaPipeline,
    wgrad: _MegaMmaPipeline,
    problem: _MegaProblem,
    sync: _MegaPipelineSync,
    work_info_smem_ptr: cute.Pointer,
    cta_group: cutlass.Constexpr,
    sf_dtype: cutlass.Constexpr[type[cutlass.Numeric]],
) -> None:
    M, N, K = problem.mnk
    accum_cnt_tile = cutlass.Int32(0)
    accum_cnt_smem = cutlass.Int32(0)

    if cutlass.const_expr(params.STATIC_SCHEDULER):
        scheduler = MegaStaticScheduler.create_consumer(params.NUM_CTAS)
    else:
        scheduler = MegaDynamicScheduler.create_consumer(
            sync.tile_consumer_mbar,
            sync.tile_id_smem_ptr,
            params.NUM_CTAS,
            params.NUM_TILE_BUFFERS,
        )
    tile_idx = scheduler.initial_work_tile_info()

    if cutlass.const_expr(params.STATIC_SCHEDULER):
        gemm_mnk_pair, gemm_problem_type_pair = _mega_problem_specs(
            M,
            N,
            K,
            MEGA_DGRAD_PROBLEM_TYPE=params.MEGA_DGRAD_PROBLEM_TYPE,
            MEGA_FORWARD_MODE=params.MEGA_FORWARD_MODE,
            MEGA_WGRAD_PROBLEM_TYPE=params.MEGA_WGRAD_PROBLEM_TYPE,
            mode=params.mode,
        )
        group_tile_size: cutlass.Constexpr[int] = (
            params.BLOCK_SIZE_N if params.SWAP_AB else params.BLOCK_SIZE_M
        )
        visitor = MegaProblemVisitor.create(
            problem.split_sizes,
            gemm_mnk_pair,
            problem.groups,
            gemm_problem_type_pair,
            params.BLOCK_SIZE_M,
            params.BLOCK_SIZE_N,
            params.BLOCK_SIZE_K,
            params.force_n_major,
            params.num_n_clusters,
            problem.local_rank,
            params.schedule_world_size,
            params.SWAP_AB,
            group_tile_size,
        )
        work = visitor.get_work(tile_idx)
    else:
        work = load_mega_work_info(
            tile_idx,
            work_info_smem_ptr,
            cutlass.Int32(0),
        )

    while work.is_valid_tile:
        if work.problem_idx == cutlass.Int32(0):
            accum_cnt_smem = _blockscaled_mma_consumer_tile(
                dgrad,
                sync,
                work.num_k_tiles,
                accum_cnt_tile,
                accum_cnt_smem,
                KLOOP_UNROLL=params.KLOOP_UNROLL,
                NUM_CTAS=params.NUM_CTAS,
                NUM_SMEM_BUFFERS=params.NUM_SMEM_BUFFERS,
                NUM_TMEM_BUFFERS=params.NUM_TMEM_BUFFERS,
                OVERLAPPING_ACCUM=params.OVERLAPPING_ACCUM,
                cta_group=cta_group,
                sf_dtype=sf_dtype,
            )
        else:
            accum_cnt_smem = _blockscaled_mma_consumer_tile(
                wgrad,
                sync,
                work.num_k_tiles,
                accum_cnt_tile,
                accum_cnt_smem,
                KLOOP_UNROLL=params.KLOOP_UNROLL,
                NUM_CTAS=params.NUM_CTAS,
                NUM_SMEM_BUFFERS=params.NUM_SMEM_BUFFERS,
                NUM_TMEM_BUFFERS=params.NUM_TMEM_BUFFERS,
                OVERLAPPING_ACCUM=params.OVERLAPPING_ACCUM,
                cta_group=cta_group,
                sf_dtype=sf_dtype,
            )
        accum_cnt_tile += cutlass.Int32(1)
        tile_idx = scheduler.advance_consumer(accum_cnt_tile)
        if cutlass.const_expr(params.STATIC_SCHEDULER):
            work = visitor.get_work(tile_idx)
        else:
            work = load_mega_work_info(
                tile_idx,
                work_info_smem_ptr,
                accum_cnt_tile % params.NUM_TILE_BUFFERS,
            )


@cute.jit
def _wait_before_tma_load(
    gemm_idx: cutlass.Constexpr[int],
    tile_m_idx,
    tile_n_idx,
    act_tile_start,
    group_idx,
    group_rows,
    full_m_size,
    full_n_size,
    hidden_dim,
    intermediate_dim,
    split_sizes,
    dispatch_quant: _MegaDispatchQuantSync,
    activation_quant: _MegaDispatchQuantSync,
    BLOCK_SIZE_M: cutlass.Constexpr[int],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
    INTERLEAVED_FC13: cutlass.Constexpr[bool],
    SWAP_AB: cutlass.Constexpr[bool],
    format: cutlass.Constexpr,
    DISPATCH_QUANT_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
    COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    NUM_CTAS: cutlass.Constexpr[int],
) -> None:
    if cutlass.const_expr(gemm_idx == 0):
        _mega_wait_quant_ready(
            wait_axis=0,
            tile_m_idx=tile_m_idx,
            tile_n_idx=tile_n_idx,
            dgrad_act_off_tiles=act_tile_start,
            g=group_idx,
            split_sizes=split_sizes,
            m_size=full_m_size,
            n_size=full_n_size,
            dispatch_dim=hidden_dim,
            mDoneCounter=dispatch_quant.done_counter,
            row_done_counter_offset=dispatch_quant.done_counter_offsets[0],
            col_done_counter_offset=dispatch_quant.done_counter_offsets[1],
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            DISPATCH_QUANT_SCALE_COLS_PER_TILE=DISPATCH_QUANT_SCALE_COLS_PER_TILE,
            SWAP_AB=SWAP_AB,
            sf_vec_size=sf_vec_size,
        )
        return
    act_block: cutlass.Constexpr[int] = BLOCK_SIZE_N if SWAP_AB else BLOCK_SIZE_M
    activation_tile_idx = tile_n_idx if SWAP_AB else tile_m_idx
    if cutlass.const_expr(INTERLEAVED_FC13 and format is not NVFP4):
        _mega_forward_wait_h1_tile(
            activation_quant.done_counter,
            activation_quant.done_counter_offsets[0],
            act_tile_start + activation_tile_idx,
            intermediate_dim,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            NUM_CTAS=NUM_CTAS,
            SWAP_AB=SWAP_AB,
        )
    else:
        _mega_forward_h2_tile_wait(
            mDoneCounter=activation_quant.done_counter,
            row_done_counter_offset=activation_quant.done_counter_offsets[0],
            act_tile_idx=activation_tile_idx,
            act_tile_start_per_group=act_tile_start,
            act_rows=group_rows,
            h2_dim=intermediate_dim,
            ACT_BLOCK_SIZE=act_block,
            COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE=COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE,
            format=format,
            sf_vec_size=sf_vec_size,
        )


@cute.jit
def _sm103_tma_load_pair_segments(
    sync: GroupedGemmPipelineSync,
    data_atom,
    scale_atom,
    g_data,
    g_scale,
    s_data,
    s_scale,
    data_axis,
    scale_axis,
    group_axis,
    data_desc,
    scale_desc,
    data_mask,
    scale_mask,
    num_k_tiles,
    accum_cnt_smem,
    ab_seg_tx_bytes: cutlass.Constexpr[int],
    sf_seg_tx_bytes: cutlass.Constexpr[int],
    use_device_tensormaps: cutlass.Constexpr[bool],
    KLOOP_UNROLL: cutlass.Constexpr[int],
    NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
    SF_RING_TILES: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
    cache_policy=None,
):
    """SM103 ultra loads for one operand pair: per K=768 tile, three K=256
    data segments ride the A/B smem ring (``accum * 3 + seg``) and the scale
    segments (four K=192 at SF vec 16, two K=384 at vec 32) ride the
    dedicated SF ring of ``SF_RING_TILES`` K-tiles whose phase toggles once
    per ring revolution. Both pair-loader warps arrive-and-expect on the
    shared mbarriers with their own byte halves (init count 2)."""
    SF_SEGMENTS: cutlass.Constexpr[int] = sm103.sf_segments(sf_vec_size)
    LOAD_STEPS: cutlass.Constexpr[int] = sm103.sf_load_steps(sf_vec_size)
    for kk in cutlass.range(num_k_tiles, unroll=KLOOP_UNROLL):
        sf_slot_base = (accum_cnt_smem % cutlass.Int32(SF_RING_TILES)) * cutlass.Int32(
            SF_SEGMENTS
        )
        sf_phase = (accum_cnt_smem // cutlass.Int32(SF_RING_TILES)) & cutlass.Int32(1)
        for sf_stage in cutlass.range_constexpr(LOAD_STEPS):
            if cutlass.const_expr(sf_stage < sm103.SM103_AB_SEGMENTS):
                ab_count = accum_cnt_smem * cutlass.Int32(
                    sm103.SM103_AB_SEGMENTS
                ) + cutlass.Int32(sf_stage)
                ab_buf = ab_count % cutlass.Int32(NUM_SMEM_BUFFERS)
                ab_phase = (
                    ab_count // cutlass.Int32(NUM_SMEM_BUFFERS)
                ) & cutlass.Int32(1)
                cute.arch.mbarrier_wait(sync.ab_empty_mbar + ab_buf, ab_phase ^ 1)
                if sync.cluster_cta_rank == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            sync.ab_full_mbar + ab_buf,
                            ab_seg_tx_bytes,
                        )
                g_seg = cute.group_modes(
                    g_data[(None, None, sf_stage, data_axis, kk, group_axis)], 0, 2
                )
                if cutlass.const_expr(use_device_tensormaps):
                    cute.copy(
                        data_atom,
                        g_seg,
                        s_data[(None, ab_buf)],
                        tma_bar_ptr=sync.ab_full_mbar + ab_buf,
                        mcast_mask=data_mask,
                        tma_desc_ptr=data_desc,
                        cache_policy=cache_policy,
                    )
                else:
                    cute.copy(
                        data_atom,
                        g_seg,
                        s_data[(None, ab_buf)],
                        tma_bar_ptr=sync.ab_full_mbar + ab_buf,
                        mcast_mask=data_mask,
                        cache_policy=cache_policy,
                    )
            if cutlass.const_expr(sf_stage < SF_SEGMENTS):
                sf_k = kk * cutlass.Int32(SF_SEGMENTS) + cutlass.Int32(sf_stage)
                sf_slot = sf_slot_base + cutlass.Int32(sf_stage)
                cute.arch.mbarrier_wait(sync.sf_empty_mbar + sf_slot, sf_phase ^ 1)
                if sync.cluster_cta_rank == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            sync.sf_full_mbar + sf_slot,
                            sf_seg_tx_bytes,
                        )
                g_sf_seg = cute.filter_zeros(
                    g_scale[(None, scale_axis, sf_k, group_axis)]
                )
                if cutlass.const_expr(use_device_tensormaps):
                    cute.copy(
                        scale_atom,
                        g_sf_seg,
                        s_scale[(None, sf_slot)],
                        tma_bar_ptr=sync.sf_full_mbar + sf_slot,
                        mcast_mask=scale_mask,
                        tma_desc_ptr=scale_desc,
                        cache_policy=cache_policy,
                    )
                else:
                    cute.copy(
                        scale_atom,
                        g_sf_seg,
                        s_scale[(None, sf_slot)],
                        tma_bar_ptr=sync.sf_full_mbar + sf_slot,
                        mcast_mask=scale_mask,
                        cache_policy=cache_policy,
                    )
        accum_cnt_smem += cutlass.Int32(1)
    return accum_cnt_smem


@cute.jit
def _blockscaled_tma_load_pair_tile(
    sync: GroupedGemmPipelineSync,
    data_atom,
    scale_atom,
    g_data,
    g_scale,
    s_data,
    s_scale,
    data_axis,
    scale_axis,
    group_axis,
    data_desc,
    scale_desc,
    data_mask,
    scale_mask,
    num_k_tiles,
    accum_cnt_smem,
    expected_tx_bytes: int,
    use_device_tensormaps: cutlass.Constexpr[bool],
    KLOOP_UNROLL: cutlass.Constexpr[int],
    NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
    cache_policy=None,
):
    buf, phase = _get_bufidx_phase(accum_cnt_smem, NUM_SMEM_BUFFERS)
    for kk in cutlass.range(num_k_tiles, unroll=KLOOP_UNROLL):
        cute.arch.mbarrier_wait(sync.ab_empty_mbar + buf, phase ^ 1)
        if sync.cluster_cta_rank == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    sync.ab_full_mbar + buf,
                    expected_tx_bytes,
                )
        if cutlass.const_expr(use_device_tensormaps):
            cute.copy(
                data_atom,
                g_data[(None, data_axis, kk, 0)],
                s_data[(None, buf)],
                tma_bar_ptr=sync.ab_full_mbar + buf,
                mcast_mask=data_mask,
                tma_desc_ptr=data_desc,
                cache_policy=cache_policy,
            )
            cute.copy(
                scale_atom,
                g_scale[(None, scale_axis, kk, 0)],
                s_scale[(None, buf)],
                tma_bar_ptr=sync.ab_full_mbar + buf,
                mcast_mask=scale_mask,
                tma_desc_ptr=scale_desc,
                cache_policy=cache_policy,
            )
        else:
            cute.copy(
                data_atom,
                g_data[(None, data_axis, kk, group_axis)],
                s_data[(None, buf)],
                tma_bar_ptr=sync.ab_full_mbar + buf,
                mcast_mask=data_mask,
                cache_policy=cache_policy,
            )
            cute.copy(
                scale_atom,
                g_scale[(None, scale_axis, kk, group_axis)],
                s_scale[(None, buf)],
                tma_bar_ptr=sync.ab_full_mbar + buf,
                mcast_mask=scale_mask,
                cache_policy=cache_policy,
            )
        buf += cutlass.Int32(1)
        wrapped = buf == cutlass.Int32(NUM_SMEM_BUFFERS)
        buf = cutlass.Int32(0) if wrapped else buf
        phase ^= cutlass.Int32(wrapped)
        accum_cnt_smem += cutlass.Int32(1)
    return accum_cnt_smem


@cute.jit
def _make_chunked_mega_visitor(
    params: ChunkedGemmWarpParams,
    problem: GroupedGemmProblem,
) -> ChunkedMegaProblemVisitor:
    _, N, K = problem.mnk
    return ChunkedMegaProblemVisitor.create(
        problem.split_sizes,
        N,
        K,
        problem.groups,
        params.BLOCK_SIZE_M,
        params.BLOCK_SIZE_N,
        params.BLOCK_SIZE_K,
        params.force_n_major,
        params.num_n_clusters,
        problem.local_rank,
        params.schedule_world_size,
        params.SWAP_AB,
        params.PIPELINE_CHUNK_ROWS,
        params.PIPELINE_LEAD_CHUNKS,
        params.FC2_TILES_PER_SCHEDULE_ITEM,
        params.STAGE_ALL_FC13,
        params.GLOBAL_LEAD_CHUNKS,
    )


@cute.jit
def _make_chunked_mega_producer(
    params: ChunkedGemmWarpParams,
    sync: GroupedGemmPipelineSync,
) -> MegaDynamicScheduler:
    return MegaDynamicScheduler.create_producer(
        sync.counter_ptr,
        sync.tile_cta_bar_mbar,
        sync.tile_id_smem_ptr,
        sync.tile_consumer_mbar,
        sync.tile_producer_mbar,
        sync.cluster_cta_rank,
        params.NUM_CTAS,
        params.NUM_TILE_BUFFERS,
        params.NUM_TILE_CTA_BARS,
    )


@cute.jit
def _publish_chunked_mega_terminal_work(
    params: ChunkedGemmWarpParams,
    scheduler: MegaDynamicScheduler,
    work: ChunkedMegaWorkTileInfo,
    work_info_smem_ptr: cute.Pointer,
    accum_cnt_out: cutlass.Int32,
) -> None:
    scheduler.producer_wait_tile_released(accum_cnt_out)
    publish_chunked_mega_work_info(
        work,
        work_info_smem_ptr,
        accum_cnt_out % params.NUM_TILE_BUFFERS,
        native_mma_tails=params.NATIVE_MMA_TAILS,
    )
    scheduler.producer_publish_tile(accum_cnt_out)


@cute.jit
def _gemm_scheduler_warp(
    params: ChunkedGemmWarpParams,
    problem: GroupedGemmProblem,
    sync: GroupedGemmPipelineSync,
    work_info_smem_ptr: cute.Pointer,
    dispatch_gather_ptrs=None,
) -> None:
    scheduler = _make_chunked_mega_producer(params, sync)
    visitor = _make_chunked_mega_visitor(params, problem)
    tile_idx = scheduler.initial_work_tile_info()
    work = visitor.get_work(tile_idx)
    accum_cnt_out = cutlass.Int32(0)
    while work.is_valid_tile:
        publish_chunked_mega_work_info(
            work,
            work_info_smem_ptr,
            accum_cnt_out % params.NUM_TILE_BUFFERS,
            native_mma_tails=params.NATIVE_MMA_TAILS,
            gather_ptrs=dispatch_gather_ptrs,
            block_n=params.BLOCK_SIZE_N,
        )
        scheduler.producer_publish_tile(accum_cnt_out)
        accum_cnt_out += cutlass.Int32(1)
        scheduler.producer_wait_tile_released(accum_cnt_out)
        tile_idx = scheduler.advance_producer(accum_cnt_out)
        work = visitor.get_work(tile_idx)
    _publish_chunked_mega_terminal_work(
        params, scheduler, work, work_info_smem_ptr, accum_cnt_out
    )


@cute.jit
def _gemm_tma_warp(  # noqa: C901
    params: ChunkedGemmWarpParams,
    fc13: _MegaTmaPipeline,
    fc2: _MegaTmaPipeline,
    fc13_tx_bytes: int,
    fc2_tx_bytes: int,
    load_a: cutlass.Constexpr[bool],
    problem: GroupedGemmProblem,
    sync: GroupedGemmPipelineSync,
    dispatch_quant: _MegaDispatchQuantSync,
    activation_quant: _MegaDispatchQuantSync,
    tensormap_manager,
    tensormaps: cute.Tensor,
    use_device_tensormaps: cutlass.Constexpr[bool],
    work_info_smem_ptr: cute.Pointer,
    format: cutlass.Constexpr,
    KLOOP_UNROLL: cutlass.Constexpr[int],
    NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
    INTERLEAVED_FC13: cutlass.Constexpr[bool],
    DISPATCH_QUANT_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
    COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    fc13_seg_tx: cutlass.Constexpr[tuple] = (0, 0),
    fc2_seg_tx: cutlass.Constexpr[tuple] = (0, 0),
    weight_borrow_args=None,
    b_layouts: tuple = (),
    dispatch_gather_ptrs=None,
) -> None:
    M, N, K = problem.mnk
    is_producer: cutlass.Constexpr[bool] = load_a and not params.SCHEDULER_RELAY
    work_info_role: cutlass.Constexpr[int] = (
        CHUNKED_MEGA_WORK_INFO_ROLE_ALL
        if params.SCHEDULER_RELAY
        else CHUNKED_MEGA_WORK_INFO_ROLE_TMA_B
    )
    if cutlass.const_expr(is_producer):
        scheduler = _make_chunked_mega_producer(params, sync)
    else:
        scheduler = MegaDynamicScheduler.create_record_consumer(
            sync.tile_consumer_mbar,
            sync.tile_id_smem_ptr,
            params.NUM_CTAS,
            params.NUM_TILE_BUFFERS,
        )
    tile_idx = scheduler.initial_work_tile_info()
    accum_cnt_smem = cutlass.Int32(0)
    accum_cnt_out = cutlass.Int32(0)
    tensormap_base_pair = (
        params.MEGA_DGRAD_TENSORMAP_BASE,
        params.MEGA_WGRAD_TENSORMAP_BASE,
    )
    if cutlass.const_expr(is_producer):
        visitor = _make_chunked_mega_visitor(params, problem)
        work = visitor.get_work(tile_idx)
    else:
        work = load_chunked_mega_work_info(
            tile_idx,
            work_info_smem_ptr,
            cutlass.Int32(0),
            format is not NVFP4,
            work_info_role,
            native_mma_tails=params.NATIVE_MMA_TAILS,
        )

    if cutlass.const_expr(use_device_tensormaps and params.FOLD_TENSORMAP_PREP):
        wait_folded_tensormaps(sync.counter_ptr, problem.groups)
    while work.is_valid_tile:
        g = work.group_idx
        desc_ptrs_pair: list = []
        for gemm_idx in cutlass.range_constexpr(2):
            desc_base = tensormap_base_pair[gemm_idx]
            desc_ptrs: list = []
            if cutlass.const_expr(use_device_tensormaps):
                for desc_delta in cutlass.range_constexpr(4):
                    desc_g_ptr = tensormap_manager.get_tensormap_ptr(
                        tensormaps[(g, desc_base + desc_delta, None)].iterator
                    )
                    desc_ptrs.append(
                        tensormap_manager.get_tensormap_ptr(
                            desc_g_ptr,
                            cute.AddressSpace.generic,
                        )
                    )
                    if cutlass.const_expr((desc_delta % 2 == 0) == load_a):
                        tensormap_manager.fence_tensormap_update(desc_g_ptr)
            desc_ptrs_pair.append(desc_ptrs)

        # Borrowed trailing groups' weights stream in from a peer while local
        # groups compute; the weight loader alone gates on their arrival, and
        # it also publishes the schedule, so no consumer can outrun it. Local
        # groups keep the ungated loads.
        if cutlass.const_expr(load_a and params.WEIGHT_BORROW_SLOTS > 0):
            num_local_groups: cutlass.Constexpr[int] = (
                problem.groups - params.WEIGHT_BORROW_SLOTS
            )
            if g >= cutlass.Int32(num_local_groups):
                _weight_borrow_gate(
                    weight_borrow_args,
                    activation_quant.done_counter,
                    cutlass.Int32(params.WEIGHT_BORROW_DONE_OFFSET),
                    g,
                    cutlass.Int32(num_local_groups),
                    CHUNK_BYTES=params.WEIGHT_BORROW_CHUNK_BYTES,
                )

        while work.is_valid_tile and work.group_idx == g:
            for gemm_idx in cutlass.range_constexpr(2):
                if work.problem_idx == cutlass.Int32(gemm_idx):
                    pipeline = (fc13, fc2)[gemm_idx]
                    desc_ptrs = desc_ptrs_pair[gemm_idx]
                    full_m_size = (2 * N, K)[gemm_idx]
                    # Under SWAP_AB, the visitor stores the expert row count
                    # in full_n_size; full_m_size is the static output width.
                    if cutlass.const_expr(is_producer):
                        publish_chunked_mega_work_info(
                            work,
                            work_info_smem_ptr,
                            accum_cnt_out % params.NUM_TILE_BUFFERS,
                            native_mma_tails=params.NATIVE_MMA_TAILS,
                            gather_ptrs=dispatch_gather_ptrs,
                            block_n=params.BLOCK_SIZE_N,
                        )
                        scheduler.producer_publish_tile(accum_cnt_out)
                    tile_m_idx = work.tile_m_idx
                    tile_n_idx = work.tile_n_idx
                    if cutlass.const_expr(not load_a):
                        _wait_before_tma_load(
                            gemm_idx,
                            tile_m_idx,
                            tile_n_idx,
                            work.group_tile_prefix,
                            g,
                            work.full_n_size,
                            full_m_size,
                            work.full_n_size,
                            K,
                            N,
                            problem.split_sizes,
                            dispatch_quant,
                            activation_quant,
                            BLOCK_SIZE_M=params.BLOCK_SIZE_M,
                            BLOCK_SIZE_N=params.BLOCK_SIZE_N,
                            INTERLEAVED_FC13=INTERLEAVED_FC13,
                            SWAP_AB=params.SWAP_AB,
                            format=format,
                            DISPATCH_QUANT_SCALE_COLS_PER_TILE=DISPATCH_QUANT_SCALE_COLS_PER_TILE,
                            sf_vec_size=sf_vec_size,
                            COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE=COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE,
                            NUM_CTAS=params.NUM_CTAS,
                        )
                    expected_tx_bytes = (fc13_tx_bytes, fc2_tx_bytes)[gemm_idx]
                    a_mask, b_mask, sfa_mask, sfb_mask = sync.tma_mcast_masks
                    tiles_per_item: cutlass.Constexpr[int] = (
                        1,
                        params.FC2_TILES_PER_SCHEDULE_ITEM,
                    )[gemm_idx]
                    for subtile_idx in cutlass.range_constexpr(tiles_per_item):
                        subtile_m_idx, subtile_n_idx = _subtile_coords(
                            tile_m_idx, tile_n_idx, subtile_idx, SWAP_AB=params.SWAP_AB
                        )
                        if _subtile_is_valid(
                            subtile_m_idx,
                            subtile_n_idx,
                            work.num_output_tiles,
                            SWAP_AB=params.SWAP_AB,
                        ):
                            if cutlass.const_expr(load_a):
                                data_axis = subtile_m_idx
                                if cutlass.const_expr(gemm_idx == 0):
                                    data_axis += (
                                        sync.cluster_cta_rank * work.num_m_tiles
                                    )
                                if cutlass.const_expr(use_device_tensormaps):
                                    group_axis = cutlass.Int32(0)
                                    data_desc = desc_ptrs[0]
                                    scale_desc = desc_ptrs[2]
                                elif cutlass.const_expr(params.SWAP_AB):
                                    group_axis = g
                                    data_desc = None
                                    scale_desc = None
                                else:
                                    data_axis += work.group_tile_prefix
                                    group_axis = cutlass.Int32(0)
                                    data_desc = None
                                    scale_desc = None
                                weight_l2_policy = _weight_l2_policy(
                                    work.full_n_size, params.BLOCK_SIZE_N
                                )
                                if cutlass.const_expr(params.USE_SM103_ULTRA):
                                    seg_tx = (fc13_seg_tx, fc2_seg_tx)[gemm_idx]
                                    if cutlass.const_expr(gemm_idx == 0):
                                        # FC13 A rides the paired 128-row view:
                                        # data_axis is already the CTA's own
                                        # gate/up 128-row block, which is the
                                        # SFA tile index as-is.
                                        sfa_axis = data_axis
                                    else:
                                        sfa_axis = (
                                            data_axis * cutlass.Int32(params.NUM_CTAS)
                                            + sync.cluster_cta_rank
                                        )
                                    accum_cnt_smem = _sm103_tma_load_pair_segments(
                                        sync,
                                        pipeline.tma_atom_a,
                                        pipeline.tma_atom_sfa,
                                        pipeline.gA,
                                        pipeline.gSFA,
                                        pipeline.sA,
                                        pipeline.sSFA,
                                        data_axis,
                                        sfa_axis,
                                        group_axis,
                                        data_desc,
                                        scale_desc,
                                        a_mask,
                                        sfa_mask,
                                        work.num_k_tiles,
                                        accum_cnt_smem,
                                        ab_seg_tx_bytes=seg_tx[0],
                                        sf_seg_tx_bytes=seg_tx[1],
                                        use_device_tensormaps=use_device_tensormaps,
                                        KLOOP_UNROLL=KLOOP_UNROLL,
                                        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                                        SF_RING_TILES=params.SM103_SF_RING_TILES,
                                        sf_vec_size=format.sf_vec_size,
                                        cache_policy=weight_l2_policy,
                                    )
                                else:
                                    accum_cnt_smem = _blockscaled_tma_load_pair_tile(
                                        sync,
                                        pipeline.tma_atom_a,
                                        pipeline.tma_atom_sfa,
                                        pipeline.gA,
                                        pipeline.gSFA,
                                        pipeline.sA,
                                        pipeline.sSFA,
                                        data_axis,
                                        data_axis,
                                        group_axis,
                                        data_desc,
                                        scale_desc,
                                        a_mask,
                                        sfa_mask,
                                        work.num_k_tiles,
                                        accum_cnt_smem,
                                        expected_tx_bytes,
                                        use_device_tensormaps,
                                        KLOOP_UNROLL=KLOOP_UNROLL,
                                        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                                        cache_policy=weight_l2_policy,
                                    )
                            else:
                                data_axis = subtile_n_idx
                                if cutlass.const_expr(
                                    params.ACTIVATION_RING_CHUNKS > 0
                                ):
                                    # FC13 reads x and FC2 reads h2 from their
                                    # activation rings: remap the group-relative
                                    # row tile to the schedule-slot ring
                                    # position. Ring mode requires SWAP_AB +
                                    # device tensormaps (validated on the host).
                                    ring_row_tiles: cutlass.Constexpr[int] = (
                                        params.PIPELINE_CHUNK_ROWS
                                        // params.BLOCK_SIZE_N
                                    )
                                    data_axis = (
                                        subtile_n_idx
                                        - work.chunk_start // params.BLOCK_SIZE_N
                                        + (
                                            work.sched_chunk_idx
                                            % params.ACTIVATION_RING_CHUNKS
                                        )
                                        * cutlass.Int32(ring_row_tiles)
                                    )
                                if cutlass.const_expr(use_device_tensormaps):
                                    group_axis = cutlass.Int32(0)
                                    data_desc = desc_ptrs[1]
                                    scale_desc = desc_ptrs[3]
                                elif cutlass.const_expr(params.SWAP_AB):
                                    data_axis += work.group_tile_prefix
                                    group_axis = cutlass.Int32(0)
                                    data_desc = None
                                    scale_desc = None
                                else:
                                    group_axis = g
                                    data_desc = None
                                    scale_desc = None
                                scale_axis = data_axis
                                if cutlass.const_expr(params.USE_SM103_ULTRA):
                                    seg_tx = (fc13_seg_tx, fc2_seg_tx)[gemm_idx]
                                    accum_cnt_smem = _sm103_tma_load_pair_segments(
                                        sync,
                                        pipeline.tma_atom_b,
                                        pipeline.tma_atom_sfb,
                                        pipeline.gB,
                                        pipeline.gSFB,
                                        pipeline.sB,
                                        pipeline.sSFB,
                                        data_axis,
                                        scale_axis,
                                        group_axis,
                                        data_desc,
                                        scale_desc,
                                        b_mask,
                                        sfb_mask,
                                        work.num_k_tiles,
                                        accum_cnt_smem,
                                        ab_seg_tx_bytes=seg_tx[0],
                                        sf_seg_tx_bytes=seg_tx[1],
                                        use_device_tensormaps=use_device_tensormaps,
                                        KLOOP_UNROLL=KLOOP_UNROLL,
                                        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                                        SF_RING_TILES=params.SM103_SF_RING_TILES,
                                        sf_vec_size=format.sf_vec_size,
                                    )
                                else:
                                    gB = pipeline.gB
                                    if cutlass.const_expr(
                                        params.COMPACT_DECODE_ROWS
                                        or params.NATIVE_MMA_TAILS
                                    ):
                                        valid_n = _subtile_valid_n(
                                            work.full_n_size,
                                            subtile_n_idx,
                                            params.BLOCK_SIZE_N,
                                        )
                                        if cutlass.const_expr(params.NATIVE_MMA_TAILS):
                                            valid_n = work.mma_n
                                        delta = sync.cluster_cta_rank * (
                                            (
                                                valid_n
                                                - cutlass.Int32(params.BLOCK_SIZE_N)
                                            )
                                            // cutlass.Int32(2)
                                        )
                                        origin = cute.make_tensor(
                                            gB.iterator, b_layouts[gemm_idx]
                                        )
                                        shifted = cute.domain_offset(
                                            (delta, 0, 0), origin
                                        )
                                        gB = cute.make_tensor(
                                            shifted.iterator, gB.layout
                                        )
                                    accum_cnt_smem = _blockscaled_tma_load_pair_tile(
                                        sync,
                                        pipeline.tma_atom_b,
                                        pipeline.tma_atom_sfb,
                                        gB,
                                        pipeline.gSFB,
                                        pipeline.sB,
                                        pipeline.sSFB,
                                        data_axis,
                                        scale_axis,
                                        group_axis,
                                        data_desc,
                                        scale_desc,
                                        b_mask,
                                        sfb_mask,
                                        work.num_k_tiles,
                                        accum_cnt_smem,
                                        expected_tx_bytes,
                                        use_device_tensormaps,
                                        KLOOP_UNROLL=KLOOP_UNROLL,
                                        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                                    )
            accum_cnt_out += cutlass.Int32(1)
            if cutlass.const_expr(is_producer):
                scheduler.producer_wait_tile_released(accum_cnt_out)
                tile_idx = scheduler.advance_producer(accum_cnt_out)
                work = visitor.get_work(tile_idx)
            else:
                tile_idx = scheduler.advance_record_consumer(accum_cnt_out)
                work = load_chunked_mega_work_info(
                    tile_idx,
                    work_info_smem_ptr,
                    accum_cnt_out % params.NUM_TILE_BUFFERS,
                    format is not NVFP4,
                    work_info_role,
                    native_mma_tails=params.NATIVE_MMA_TAILS,
                )
    if cutlass.const_expr(is_producer):
        _publish_chunked_mega_terminal_work(
            params, scheduler, work, work_info_smem_ptr, accum_cnt_out
        )


@cute.jit
def _gemm_mma_warp(
    params: ChunkedGemmWarpParams,
    fc13: _MegaMmaPipeline,
    fc2: _MegaMmaPipeline,
    problem: GroupedGemmProblem,
    sync: GroupedGemmPipelineSync,
    work_info_smem_ptr: cute.Pointer,
    format: cutlass.Constexpr,
    cta_group: cutlass.Constexpr,
    sf_dtype: cutlass.Constexpr[type[cutlass.Numeric]],
    dispatch_gather_ptrs=None,
) -> None:
    M, N, K = problem.mnk
    work_info_role: cutlass.Constexpr[int] = (
        CHUNKED_MEGA_WORK_INFO_ROLE_ALL
        if params.COMPACT_DECODE_ROWS and not params.NATIVE_MMA_TAILS
        else CHUNKED_MEGA_WORK_INFO_ROLE_MMA
    )
    scheduler = MegaDynamicScheduler.create_record_consumer(
        sync.tile_consumer_mbar,
        sync.tile_id_smem_ptr,
        params.NUM_CTAS,
        params.NUM_TILE_BUFFERS,
    )
    tile_idx = scheduler.initial_work_tile_info()
    accum_cnt_tile = cutlass.Int32(0)
    accum_cnt_smem = cutlass.Int32(0)
    accum_cnt_out = cutlass.Int32(0)
    work = load_chunked_mega_work_info(
        tile_idx,
        work_info_smem_ptr,
        cutlass.Int32(0),
        format is not NVFP4,
        work_info_role,
        native_mma_tails=params.NATIVE_MMA_TAILS,
    )

    while work.is_valid_tile:
        for gemm_idx in cutlass.range_constexpr(2):
            if work.problem_idx == cutlass.Int32(gemm_idx):
                pipeline = (fc13, fc2)[gemm_idx]
                tiles_per_item: cutlass.Constexpr[int] = (
                    1,
                    params.FC2_TILES_PER_SCHEDULE_ITEM,
                )[gemm_idx]
                for subtile_idx in cutlass.range_constexpr(tiles_per_item):
                    subtile_m_idx, subtile_n_idx = _subtile_coords(
                        work.tile_m_idx,
                        work.tile_n_idx,
                        subtile_idx,
                        SWAP_AB=params.SWAP_AB,
                    )
                    if _subtile_is_valid(
                        subtile_m_idx,
                        subtile_n_idx,
                        work.num_output_tiles,
                        SWAP_AB=params.SWAP_AB,
                    ):
                        if cutlass.const_expr(params.USE_SM103_ULTRA):
                            accum_cnt_smem = _sm103_mma_consumer_tile(
                                pipeline,
                                sync,
                                work.num_k_tiles,
                                accum_cnt_tile,
                                accum_cnt_smem,
                                KLOOP_UNROLL=params.KLOOP_UNROLL,
                                NUM_CTAS=params.NUM_CTAS,
                                NUM_SMEM_BUFFERS=params.NUM_SMEM_BUFFERS,
                                NUM_TMEM_BUFFERS=params.NUM_TMEM_BUFFERS,
                                OVERLAPPING_ACCUM=params.OVERLAPPING_ACCUM,
                                SF_RING_TILES=params.SM103_SF_RING_TILES,
                                cta_group=cta_group,
                                sf_dtype=sf_dtype,
                                sf_vec_size=format.sf_vec_size,
                                FRESH_TILED_MMA=True,
                            )
                        else:
                            accum_cnt_smem = _blockscaled_mma_consumer_tile(
                                pipeline,
                                sync,
                                work.num_k_tiles,
                                accum_cnt_tile,
                                accum_cnt_smem,
                                KLOOP_UNROLL=params.KLOOP_UNROLL,
                                NUM_CTAS=params.NUM_CTAS,
                                NUM_SMEM_BUFFERS=params.NUM_SMEM_BUFFERS,
                                NUM_TMEM_BUFFERS=params.NUM_TMEM_BUFFERS,
                                OVERLAPPING_ACCUM=params.OVERLAPPING_ACCUM,
                                cta_group=cta_group,
                                sf_dtype=sf_dtype,
                                dynamic_n=(
                                    work.mma_n
                                    if params.NATIVE_MMA_TAILS
                                    else _subtile_valid_n(
                                        work.full_n_size,
                                        subtile_n_idx,
                                        params.BLOCK_SIZE_N,
                                    )
                                    if params.COMPACT_DECODE_ROWS
                                    or params.NATIVE_MMA_TAILS
                                    else None
                                ),
                            )
                        accum_cnt_tile += cutlass.Int32(1)
        accum_cnt_out += cutlass.Int32(1)
        tile_idx = scheduler.advance_record_consumer(accum_cnt_out)
        work = load_chunked_mega_work_info(
            tile_idx,
            work_info_smem_ptr,
            accum_cnt_out % params.NUM_TILE_BUFFERS,
            format is not NVFP4,
            work_info_role,
            native_mma_tails=params.NATIVE_MMA_TAILS,
        )
