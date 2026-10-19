# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-side kernel class and CuTeDSL helpers for the distributed grouped GEMM."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync

from ..formats import (
    BLOCK_MAJOR_K,
    SWIGLU_CLAMP_ALPHA_DEFAULT,
    SWIGLU_CLAMP_LIMIT_DEFAULT,
)
from ._grouped_gemm_config import (
    NATIVE_SWIZZLE_GROUP_SIZE,
    WGRAD_DISPATCH_TILE_M,
)
from .combine_scatter import CombineScatterMixin
from .dispatch_quant import (
    _dispatch_gather_producer_body,
    _gather_k_segment_tma_wait,
    _GATHER_K_SEGMENTS,
    _mega_dispatch_quant_wgrad_feature_tile_wait,
)
from .grouped_gemm_kernel import (
    _BAR_EPILOG_SYNC,
    _BAR_FULL_CTA_SYNC,
    _clear_tma_oob_prefetch_bit,
    _packed_group_rows_byte_extent,
    _require_valid_activation_buffer_range,
    _swap_if,
    _TMEM_LD_MAX_REGS_PER_THREAD,
    _transpose_c_if_swap,
    GroupedGemmKernel,
)
from .grouped_gemm_tiles import release_acc
from .native_combine_swiglu import (
    NATIVE_SWIGLU_BWD_PASSES_PER_BATCH,
    native_swiglu_bwd_producer_body,
    NATIVE_SWIGLU_BWD_ROWS,
)
from .params import GroupedGemmSecondaryOperands
from .swiglu_epilogue import (
    store_interleaved_swiglu_smem_tile,
)
from .swiglu_fragment import (
    h1_epi_smem_layout,
    h1_swiglu_epilogue,
    interleaved_fc13_view,
)
from .tile_scheduler import (
    _DGRAD,
    _FPROP,
    _get_bufidx_phase,
    _WGRAD,
    DynamicTileScheduler,
    GroupedProblemVisitor,
    stage_expert_metadata,
    StaticTileScheduler,
)

# Default gather sub-tile M (must be < BLOCK_SIZE_M for finer dispatch).
_AG_BLOCK_SIZE_M: int = 16


# Gather K chunk size: 256 b16 = 512 B = 4× LDG.E.128 per thread.
_AG_BLOCK_SIZE_K: int = 256


# Target gather sub-tiles per parent GEMM M-tile. 2CTA needs finer
# granularity since CuTe 2CTA splits one M tile across CTAs.
_TARGET_NUM_SUB_PER_GEMM_M_1CTA: int = 8


_TARGET_NUM_SUB_PER_GEMM_M_2CTA: int = 16


# M tiles per block of the rotated schedule's source-segment interleave
# (`_interleave_m_tile_idx`). Measured on GB300 EP4 at farm FC2 shapes:
# uniform COMBINE p50/p90 1208/1522 -> 1101/1139 us (1 block: 1087/1123,
# 3: 1159/1223), and the 2x-skew hot rank's DGRAD dispatch went from 9% to
# 27% faster than tlx (1 block ties). FPROP dispatch measured 0-3% slower
# on uniform routing and keeps the plain rotation.
_M_TILE_INTERLEAVE: int = 2


# Split-atom COMBINE tiles cast an atom's last this-many subtiles into
# registers before staging them, so the atom's TMEM goes back to the MMA
# while they are still being scattered. Uniform hot-rank combine FPROP vs
# tlx on GB300 EP4: 0 held +3.1%, 1 +0.6%, 2 -3.0%; 3 and 4 no better.
_COMBINE_HELD_SUBTILES: int = 2


# Warp layout (12 warps = 384 threads/CTA in DISPATCH; 8 warps = 256 in COMBINE):
#   Warps 0-3 : EPILOG (consumer; fuses scatter in COMBINE)
#   Warp  4   : MMA consumer
#   Warp  5   : TMA producer
#   Warps 6-7 : IDLE padding
#   Warps 8-11: GATHER producer (DISPATCH only; not launched in COMBINE)
DIST_EPILOG_WARP_IDS: tuple[int, int, int, int] = (0, 1, 2, 3)


DIST_MMA_WARP_ID: int = 4


DIST_TMA_WARP_ID: int = 5


GATHER_WARP_IDS: tuple[int, int, int, int] = (8, 9, 10, 11)


DIST_TOTAL_WARPS_DISPATCH: int = 12


DIST_TOTAL_WARPS_COMBINE: int = 8


DIST_THREADS_PER_CTA_DISPATCH: int = 32 * DIST_TOTAL_WARPS_DISPATCH  # 384


DIST_THREADS_PER_CTA_COMBINE: int = 32 * DIST_TOTAL_WARPS_COMBINE  # 256


# Named CTA-scope `bar.sync` ID allocation for the dist kernel:
#
#   * IDs 0/1/2 are imported from the base grouped-GEMM kernel
#     (`_BAR_FULL_CTA_SYNC`, `_BAR_EPILOG_SYNC`, `_BAR_TMEM_PTR_SYNC`)
#     and keep their base meanings.
#
#   * Dist-only barriers extend that contiguous allocation. Do not inline
#     barrier IDs at use sites; add new IDs here with owner + participants.
#
#   * `_BAR_TMA_WAIT_INTERNAL` (id 3): TMA-warp-internal sync around the
#     dispatch gather wait (32 threads, TMA producer warp).
#
#   * `_BAR_GATHER_WG_INTERNAL` (id 4): warp-group-internal sync among
#     the 4 gather warps (128 threads). Used between leader-elected
#     atomic-add of the per-tile counter and the broadcast of the
#     resulting tile_idx via SMEM scratch.
#
# Cross-WG gather→TMA synchronization is handled per-tile via the
# `a_buff_counter` per-M-tile counter (see `_tma_per_tile_wait`); no
# explicit cross-WG barrier is needed.
_BAR_TMA_WAIT_INTERNAL: int = 3


_BAR_GATHER_WG_INTERNAL: int = 4


_DIST_GATHER_WG_THREADS: int = 32 * len(GATHER_WARP_IDS)  # 128


_DIST_EPILOG_WG_THREADS: int = 32 * len(DIST_EPILOG_WARP_IDS)  # 128


def _get_ag_block_size_m(block_size_m: int, target_num_sub_per_gemm_m: int) -> int:
    """Choose gather sub-tile M, scaling with BLOCK_M but no smaller than
    `_AG_BLOCK_SIZE_M`. Caller picks target based on 1CTA vs 2CTA."""
    return max(_AG_BLOCK_SIZE_M, block_size_m // target_num_sub_per_gemm_m)


class DistGroupedGemmKernel(CombineScatterMixin, GroupedGemmKernel):
    """Distributed grouped GEMM with gather A or scatter C.

    Mode is fixed at construction (`DISPATCH` for gather-A, `COMBINE` for
    scatter-C). Wgrad dispatch gathers B and synchronizes feature columns.
    """

    # Bound each TMEM load's destinations within the fixed 168-register
    # epilogue allocation; the logical fragment and its lifetime stay unchanged.
    TMEM_LOAD_MAX_REGS_PER_THREAD: int = _TMEM_LD_MAX_REGS_PER_THREAD
    DISPATCH: int = 1
    COMBINE: int = 2
    DISPATCH_SWIGLU_FWD: int = 3
    # DGRAD COMBINE whose A operand (grad_h1) is produced in-kernel by the
    # warps 8-11 SwiGLU-backward producer.
    COMBINE_SWIGLU_BWD: int = 4

    # Cross-group FC13->FC2 lead window (chunked native Mega only). Part of the
    # compile cache key, so every dist kernel carries it.
    GLOBAL_LEAD_CHUNKS: int = 0
    GATHER_WG_THREADS: int = _DIST_GATHER_WG_THREADS
    EPILOG_WG_THREADS: int = _DIST_EPILOG_WG_THREADS
    # Per-K-segment gather signals; off for TMA producers that wait for
    # whole tiles.
    GATHER_SEGMENT_SIGNALS: bool = True
    # Peer loads in flight per gather lane on narrow (<= 4 rows per warp)
    # sub-tiles.
    GATHER_LOADS_PER_LANE: int = 32
    _tma_k_segment_wait = _gather_k_segment_tma_wait

    def __init__(
        self,
        config: dict,
        problem_type: int,
        mode: int,  # DISPATCH / COMBINE
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        force_n_major: bool = True,  # dist kernel always uses N-major scheduling
        num_n_clusters: int = 1,
        world_size: int = 1,
        static_scheduler: bool = False,
        swiglu_fast_math: bool = False,
        host_a_tensormap: bool = False,
        has_padding_sentinels: bool = False,
        swap_ab: bool = False,
        output_accum: bool = False,
        schedule_rotation: bool = True,
        swiglu_clamped: bool = False,
        swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
        swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
        swiglu_emit_h2: bool = False,
        swiglu_emit_h1: bool = False,
        gather_k_segments: int = _GATHER_K_SEGMENTS,
        use_pdl: bool = False,
        block_major_weights: bool = False,
        single_a_tensormap: bool = False,
    ) -> None:
        if problem_type not in (_FPROP, _DGRAD, _WGRAD):
            raise ValueError(
                f"DistGroupedGemmKernel: unsupported problem_type={problem_type} "
                f"(only _FPROP / _DGRAD / _WGRAD supported)"
            )
        if problem_type == _WGRAD and mode != self.DISPATCH:
            raise ValueError("WGRAD supports DISPATCH only")
        if mode not in (
            self.DISPATCH,
            self.COMBINE,
            self.DISPATCH_SWIGLU_FWD,
            self.COMBINE_SWIGLU_BWD,
        ):
            raise ValueError(f"DistGroupedGemmKernel: unsupported mode={mode}")
        if mode == self.DISPATCH_SWIGLU_FWD and problem_type != _FPROP:
            raise ValueError("DISPATCH_SWIGLU_FWD supports FPROP only")
        if mode == self.COMBINE_SWIGLU_BWD and problem_type != _DGRAD:
            raise ValueError("COMBINE_SWIGLU_BWD supports RN DGRAD only")
        if swiglu_emit_h2 and mode != self.COMBINE_SWIGLU_BWD:
            raise ValueError("swiglu_emit_h2 requires COMBINE_SWIGLU_BWD")
        # Only the host-built (non-swapped) B tensormap takes the interleaved view.
        if swiglu_emit_h1 and (mode != self.DISPATCH_SWIGLU_FWD or swap_ab):
            raise ValueError("swiglu_emit_h1 requires non-swapped DISPATCH_SWIGLU_FWD")
        if swap_ab and (
            problem_type != _FPROP
            or mode not in (self.DISPATCH, self.DISPATCH_SWIGLU_FWD, self.COMBINE)
        ):
            raise ValueError("distributed SWAP_AB supports FPROP only")
        # Swapped COMBINE scatters a tile's tokens through one pointer table.
        if (
            swap_ab
            and mode == self.COMBINE
            and (config["NUM_CTAS"], config["NUM_MMAS"]) != (1, 1)
        ):
            raise ValueError("swapped COMBINE supports 1-CTA, 1-MMA tiles only")
        if single_a_tensormap and (
            problem_type != _FPROP or host_a_tensormap or config["NUM_CTAS"] != 1
        ):
            raise ValueError("single_a_tensormap supports unpadded, 1-CTA FPROP only")
        # Swapped weights need the host-built A descriptor: per-group descriptor
        # updates write row-major strides.
        if block_major_weights and (
            problem_type != _FPROP
            or swiglu_emit_h1
            or (swap_ab and not single_a_tensormap)
        ):
            raise ValueError(
                "block_major_weights supports FPROP without emit_h1, and swapped "
                "FPROP only with single_a_tensormap"
            )
        super().__init__(
            config=config,
            problem_type=problem_type,
            acc_dtype=acc_dtype,
            force_n_major=force_n_major,
            num_n_clusters=num_n_clusters,
            world_size=world_size,
            static_scheduler=static_scheduler,
            host_a_tensormap=host_a_tensormap,
            swap_ab=swap_ab,
            schedule_rotation=schedule_rotation,
        )
        self.mode = mode
        self.swiglu_fast_math = swiglu_fast_math
        self.swiglu_clamped = swiglu_clamped
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.swiglu_emit_h2 = swiglu_emit_h2
        self.swiglu_emit_h1 = swiglu_emit_h1
        self.has_padding_sentinels = has_padding_sentinels
        self.use_pdl = use_pdl
        self.block_major_weights = block_major_weights
        self.SINGLE_A_TENSORMAP = single_a_tensormap
        self.output_accum = output_accum
        # 2CTA needs finer gather granularity because CuTe splits one M
        # tile across CTAs.
        target_num_sub_per_gemm_m = (
            _TARGET_NUM_SUB_PER_GEMM_M_2CTA
            if self.NUM_CTAS == 2
            else _TARGET_NUM_SUB_PER_GEMM_M_1CTA
        )
        self.AG_BLOCK_SIZE_M: int = (
            WGRAD_DISPATCH_TILE_M
            if problem_type == _WGRAD
            else _get_ag_block_size_m(
                self.BLOCK_SIZE_N if self.SWAP_AB else self.BLOCK_SIZE_M,
                target_num_sub_per_gemm_m,
            )
        )
        self.AG_BLOCK_SIZE_K: int = _AG_BLOCK_SIZE_K
        self.GATHER_K_SEGMENTS: int = 1 if problem_type == _WGRAD else gather_k_segments
        self.M_TILE_INTERLEAVE: int = (
            _M_TILE_INTERLEAVE
            if mode == self.COMBINE
            or (mode == self.DISPATCH and problem_type == _DGRAD)
            else 0
        )
        # Plain COMBINE FPROP N bands: walking every M tile in one band pins a
        # wave's scatter to one column band of the destination rows.
        self.DIAG_N_BANDS: bool = not force_n_major
        self.dispatch_scatter: bool = False

    @cute.jit
    def _tma_per_col_tile_wait(
        self,
        counter_addr: cutlass.Int64,
        group: cutlass.Int32,
        tile_n: cutlass.Int32,
        n_size: cutlass.Int32,
        k_size: cutlass.Int32,
    ):
        col_tiles = cute.ceil_div(n_size, self.AG_BLOCK_SIZE_K)
        counters = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, counter_addr, cute.AddressSpace.gmem),
            cute.make_layout((group + 1) * col_tiles),
        )
        _mega_dispatch_quant_wgrad_feature_tile_wait(
            mDoneCounter=counters,
            col_done_counter_offset=cutlass.Int32(0),
            g=group,
            feature_tile_idx=tile_n,
            feature_tile_elems=self.BLOCK_SIZE_N,
            feature_size=n_size,
            group_row_blocks=cute.ceil_div(k_size, self.AG_BLOCK_SIZE_M),
            col_work_tiles=col_tiles,
            DISPATCH_QUANT_SCALE_COLS_PER_TILE=self.AG_BLOCK_SIZE_K,
            sf_vec_size=1,
        )

    def _setup_attributes(self):
        """Set up base GEMM layouts plus COMBINE's row-major scatter SMEM."""
        super()._setup_attributes()
        if self.swiglu_emit_h1:
            self.epi_smem_layout_staged = h1_epi_smem_layout(
                self.c_dtype, self.epi_tile, self.NUM_C_STAGES
            )
        self.c_smem_layout_staged = self.epi_smem_layout_staged
        if not self.swiglu_emit_h1 and self.mode in (
            self.COMBINE,
            self.DISPATCH_SWIGLU_FWD,
            self.COMBINE_SWIGLU_BWD,
        ):
            # COMBINE does not TMA-store C; use a row-major scratch tile so
            # adjacent-N values stay physically adjacent for 16-byte
            # LDS->STG. The scatter epilogue only uses stage 0, so one
            # stage is allocated. An 8-element row skew avoids the 32-way
            # shared-bank conflicts of an unpadded 64-col stride; chunked
            # mega's 1-stage 2-TMEM tiles are at the dynamic-SMEM launch
            # limit and stay unpadded.
            epi_m = cute.size(self.epi_tile[0])
            epi_n = cute.size(self.epi_tile[1])
            row_skew = 8 if self.NUM_TMEM_BUFFERS == 1 or self.NUM_C_STAGES > 1 else 0
            if self.SWAP_AB:
                if self.mode == self.COMBINE and epi_n != self.BLOCK_SIZE_N:
                    raise ValueError(
                        "swapped COMBINE needs epilogue subtiles spanning the token tile"
                    )
                col_stride = epi_m + row_skew
                self.c_smem_layout_staged = cute.make_layout(
                    (epi_m, epi_n, 1), stride=(1, col_stride, epi_n * col_stride)
                )
            else:
                row_stride = epi_n + row_skew
                self.c_smem_layout_staged = cute.make_layout(
                    (epi_m, epi_n, 1), stride=(row_stride, 1, epi_m * row_stride)
                )

    def _make_shared_storage(self, a_dtype, b_dtype, c_dtype, G: int):  # noqa: ANN001
        NUM_SMEM = self.NUM_SMEM_BUFFERS
        NUM_TILE = self.NUM_TILE_BUFFERS
        NUM_CTAS = self.NUM_CTAS

        a_smem_elems = cute.cosize(self.a_smem_layout_staged.outer)
        b_smem_elems = cute.cosize(self.b_smem_layout_staged.outer)
        if isinstance(self.c_smem_layout_staged, cute.ComposedLayout):
            c_smem_elems = cute.cosize(self.c_smem_layout_staged.outer)
        else:
            c_smem_elems = cute.cosize(self.c_smem_layout_staged)
        scatter_ptr_smem_elems = (
            cute.size(self.epi_tile[1] if self.SWAP_AB else self.epi_tile[0])
            if self.mode in (self.COMBINE, self.COMBINE_SWIGLU_BWD)
            or self.dispatch_scatter
            else 0
        )
        if self.SPLIT_ATOM_ACCUM:
            scatter_ptr_smem_elems *= self.NUM_MMAS

        n_smem_empty = NUM_SMEM
        n_smem_full = NUM_SMEM
        n_tmem_full = self.NUM_ACC_BARRIERS
        n_tmem_empty = self.NUM_ACC_BARRIERS
        n_tile_consumer = NUM_TILE
        n_tile_producer = NUM_TILE
        n_tile_cta_bar = 2 if NUM_CTAS == 2 else 0
        n_tmem_dealloc = 1 if NUM_CTAS == 2 else 0

        @cute.struct
        class DistSharedStorage:
            sC: cute.struct.Align[cute.struct.MemRange[c_dtype, c_smem_elems], 1024]
            sA: cute.struct.Align[cute.struct.MemRange[a_dtype, a_smem_elems], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[b_dtype, b_smem_elems], 1024]
            tile_id_smem: cute.struct.MemRange[cutlass.Int32, NUM_TILE]
            split_sizes_smem: cute.struct.MemRange[cutlass.Int32, G]
            # Broadcast slot: gather WG leader's atomic-counter result.
            gather_tile_idx_smem: cute.struct.MemRange[cutlass.Int32, 1]
            # COMBINE-only: per-row peer base pointers cached in CTA SMEM
            # to avoid a global pointer-table load per 16-byte store chunk.
            scatter_ptr_smem: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, scatter_ptr_smem_elems], 128
            ]
            tensormap_buffer: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, 16 * 3], 128
            ]
            smem_empty_mbar: cute.struct.MemRange[cutlass.Int64, n_smem_empty]
            smem_full_mbar: cute.struct.MemRange[cutlass.Int64, n_smem_full]
            tmem_full_mbar: cute.struct.MemRange[cutlass.Int64, n_tmem_full]
            tmem_empty_mbar: cute.struct.MemRange[cutlass.Int64, n_tmem_empty]
            tile_id_consumer_mbar: cute.struct.MemRange[cutlass.Int64, n_tile_consumer]
            tile_id_producer_mbar: cute.struct.MemRange[cutlass.Int64, n_tile_producer]
            tile_cta_bar_mbar: cute.struct.MemRange[cutlass.Int64, n_tile_cta_bar]
            tmem_dealloc_mbar: cute.struct.MemRange[cutlass.Int64, n_tmem_dealloc]
            tmem_holding_buf: cutlass.Int32

        return DistSharedStorage

    @cute.jit
    def _tma_per_tile_wait(  # type: ignore[override]
        self,
        tile_m_idx: cutlass.Int32,
        tile_m_start_per_group: cutlass.Int32,
        a_buff_counter_addr_i64: cutlass.Int64,
        NUM_SUB_PER_GEMM_M: cutlass.Constexpr[int],
    ):
        """DISPATCH per-tile gather wait: before issuing TMA loads for
        tile (group g, gemm_m_idx), busy-wait until the gather WG has
        filled this M-tile's slots in `a_buff_counter`."""
        ready_count = NUM_SUB_PER_GEMM_M
        if cutlass.const_expr(self.GATHER_K_SEGMENTS > 1):
            # One byte per K segment, each filled by every gather warp of
            # every sub-tile (see `_gather_k_segment_tma_wait`); no byte can
            # exceed its final value, so the sum is reached only when all are.
            ready_count = (
                NUM_SUB_PER_GEMM_M
                * (self.GATHER_WG_THREADS // 32)
                * (((1 << (8 * self.GATHER_K_SEGMENTS)) - 1) // 255)
            )
        counter_slot_addr = a_buff_counter_addr_i64 + cutlass.Int64(
            tile_m_start_per_group + tile_m_idx
        ) * cutlass.Int64(4)
        ptr = cute.make_ptr(
            cutlass.Uint32,
            counter_slot_addr,
            cute.AddressSpace.gmem,
        )
        cute.arch.barrier(
            barrier_id=_BAR_TMA_WAIT_INTERNAL,
            number_of_threads=32,
        )
        with cute.arch.elect_one():
            ready = False
            while not ready:
                v = cute.arch.load(ptr, cutlass.Uint32, sem="acquire", scope="gpu")
                ready = v >= cutlass.Uint32(ready_count)
        cute.arch.barrier(
            barrier_id=_BAR_TMA_WAIT_INTERNAL,
            number_of_threads=32,
        )

    @cute.jit
    def _combine_epilog_consumer_body(  # noqa: C901
        self,
        tidx: cutlass.Int32,
        tCtAcc_base: cute.Tensor,
        tCgC: cute.Tensor,
        sC: cute.Tensor,
        tTR_rAcc: cute.Tensor,
        tiled_copy_r2s,
        tRS_rC: cute.Tensor,
        tRS_sC: cute.Tensor,
        epi_tile: cute.Tile,
        tmem_full_mbar: cute.Pointer,
        tmem_empty_mbar: cute.Pointer,
        tile_consumer_mbar: cute.Pointer,
        tile_producer_mbar: cute.Pointer,
        tile_id_smem_ptr: cute.Pointer,
        # COMBINE-only scatter destination.
        scatter_ptrs_addr_i64: cutlass.Int64,
        padding_row_addr_i64: cutlass.Int64,
        scatter_ptr_smem_ptr: cute.Pointer,
        mSwigluOutputWords: cute.Tensor,
        fuse_swiglu_fwd: cutlass.Constexpr[bool],
        elem_size_bytes_c: cutlass.Constexpr[int],
        split_sizes: cute.Tensor,
        cluster_cta_rank: cutlass.Int32,
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        problem_type: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        WORLD_SIZE: cutlass.Constexpr[int],
        NUM_TMEM_BUFFERS: cutlass.Constexpr[int],
        NUM_TILE_BUFFERS: cutlass.Constexpr[int],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        NUM_CTAS: cutlass.Constexpr[int],
        secondary_operands=None,
    ):
        """Scatter epilogue consumer (COMBINE mode): TMEM -> registers
        -> epilogue SMEM -> per-row peer scatter.

        """
        accum_cnt_tile = cutlass.Int32(0)

        if cutlass.const_expr(self.STATIC_SCHEDULER):
            scheduler = StaticTileScheduler.create_consumer(NUM_CTAS)
        else:
            scheduler = DynamicTileScheduler.create_consumer(
                tile_consumer_mbar,
                tile_id_smem_ptr,
                NUM_CTAS,
                NUM_TILE_BUFFERS,
            )
        tile_idx = scheduler.initial_work_tile_info()
        visitor = GroupedProblemVisitor.create(
            split_sizes,
            M,
            N,
            K,
            G,
            problem_type,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            FORCE_N_MAJOR,
            NUM_N_CLUSTERS,
            local_rank,
            WORLD_SIZE,
            self.SWAP_AB,
            M_INTERLEAVE=self.M_TILE_INTERLEAVE,
            DIAG_N_BANDS=self.DIAG_N_BANDS,
        )
        work = visitor.get_work(tile_idx)
        # Token rows of contiguous features, swapped or not.
        row_major_sC = _transpose_c_if_swap(sC, self.SWAP_AB)

        while work.is_valid_tile:
            m_size = work.m_size
            n_size = work.n_size
            num_k_tiles = work.num_k_tiles
            if cutlass.const_expr(problem_type == _WGRAD):
                # WGRAD uses the fixed problem M dimension for every group.
                cm_start = work.group_idx * M
            else:
                cm_start = work.split_prefix

            if work.is_valid_tile:
                g = work.group_idx
                while work.is_valid_tile and work.group_idx == g:
                    tmem_buf, tmem_phase = _get_bufidx_phase(
                        accum_cnt_tile, NUM_TMEM_BUFFERS
                    )
                    tile_buf, _ = _get_bufidx_phase(accum_cnt_tile, NUM_TILE_BUFFERS)

                    tile_m_idx = work.tile_m_idx
                    tile_n_idx = work.tile_n_idx
                    # The scatter and the SwiGLU store walk token rows; swapped
                    # tiles hold tokens on N.
                    token_tile_idx, feature_tile_idx = _swap_if(
                        self.SWAP_AB, tile_m_idx, tile_n_idx
                    )
                    token_size, feature_size = _swap_if(self.SWAP_AB, m_size, n_size)
                    token_block, feature_block = _swap_if(
                        self.SWAP_AB, BLOCK_SIZE_M, BLOCK_SIZE_N
                    )

                    # Wait MMA done for this tmem buffer.
                    if cutlass.const_expr(not self.SPLIT_ATOM_ACCUM):
                        cute.arch.mbarrier_wait(tmem_full_mbar + tmem_buf, tmem_phase)

                    NUM_MMA_ATOMS_M = cute.size(tCtAcc_base.shape, mode=[1])
                    NUM_MMA_ATOMS_N = cute.size(tCtAcc_base.shape, mode=[2])

                    # Split-atom tiles load every atom's pointer table up
                    # front: a reload after atom 0's scatter queues behind its
                    # in-flight remote stores while the MMA waits on atom 1.
                    PTR_ATOMS = NUM_MMA_ATOMS_M if self.SPLIT_ATOM_ACCUM else 1
                    EPILOGUE_M = cute.size(sC.shape, mode=[0])

                    ATOM_TRIPS = 1
                    for _atom_trip in cutlass.range(ATOM_TRIPS, unroll=1):
                        for atom_unrolled in cutlass.range_constexpr(
                            NUM_MMA_ATOMS_M // ATOM_TRIPS
                        ):
                            mma_m_idx = atom_unrolled
                            ptr_tab = (
                                scatter_ptr_smem_ptr
                                + (mma_m_idx % PTR_ATOMS) * EPILOGUE_M
                            )
                            if not fuse_swiglu_fwd and mma_m_idx % PTR_ATOMS == 0:
                                for ld_m_idx in cutlass.range_constexpr(PTR_ATOMS):
                                    self._combine_epilog_load_scatter_ptrs(
                                        tidx=tidx,
                                        sC=row_major_sC,
                                        scatter_ptr_smem_ptr=scatter_ptr_smem_ptr
                                        + ld_m_idx * EPILOGUE_M,
                                        tile_m_idx=token_tile_idx,
                                        mma_m_idx=mma_m_idx + ld_m_idx,
                                        cm_start=cm_start,
                                        m_size=token_size,
                                        scatter_ptrs_addr_i64=scatter_ptrs_addr_i64,
                                        padding_row_addr_i64=padding_row_addr_i64,
                                        BLOCK_SIZE_M=token_block,
                                        NUM_MMA_ATOMS_M=NUM_MMA_ATOMS_M,
                                        NUM_CTAS=NUM_CTAS,
                                        cluster_cta_rank=cluster_cta_rank,
                                    )
                                # Publish the CTA-SMEM pointer table before scatter.
                                cute.arch.barrier(
                                    barrier_id=_BAR_EPILOG_SYNC,
                                    number_of_threads=_DIST_EPILOG_WG_THREADS,
                                )
                            if cutlass.const_expr(self.SPLIT_ATOM_ACCUM):
                                cute.arch.mbarrier_wait(
                                    tmem_full_mbar + mma_m_idx, tmem_phase
                                )
                            for mma_n_idx in cutlass.range_constexpr(NUM_MMA_ATOMS_N):
                                (
                                    _tiled_copy_t2r_atom,
                                    tTR_tAcc_base,
                                    _tTR_rAcc_atom,
                                ) = self._epilog_tmem_copy_and_partition(
                                    tidx=tidx,
                                    tAcc=tCtAcc_base,
                                    gC_mnl=tCgC,
                                    epi_tile=epi_tile,
                                    use_2cta_instrs=NUM_CTAS == 2,
                                    mma_m_idx=mma_m_idx,
                                    mma_n_idx=mma_n_idx,
                                )
                                tTR_tAcc = tTR_tAcc_base[
                                    (None, None, None, None, None, tmem_buf)
                                ]
                                tTR_tAcc = cute.group_modes(
                                    tTR_tAcc, 3, cute.rank(tTR_tAcc)
                                )
                                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                                # Subtiles from `first_held` on are all read from
                                # TMEM at `first_held`, which then releases the atom.
                                first_held = subtile_cnt
                                if cutlass.const_expr(
                                    self.SPLIT_ATOM_ACCUM
                                    and mma_n_idx == NUM_MMA_ATOMS_N - 1
                                ):
                                    first_held = subtile_cnt - min(
                                        _COMBINE_HELD_SUBTILES, subtile_cnt
                                    )
                                subtile_rC = [tRS_rC] * (first_held + 1)
                                for _ in cutlass.range_constexpr(
                                    first_held + 1, subtile_cnt
                                ):
                                    subtile_rC.append(
                                        cute.make_rmem_tensor_like(tRS_rC)
                                    )

                                for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                                    ld_lo = subtile_idx
                                    ld_hi = subtile_idx + 1
                                    if cutlass.const_expr(subtile_idx == first_held):
                                        ld_hi = subtile_cnt
                                    elif cutlass.const_expr(subtile_idx > first_held):
                                        ld_hi = ld_lo
                                    for ld_idx in cutlass.range_constexpr(ld_lo, ld_hi):
                                        ld_rC = subtile_rC[ld_idx]
                                        cute.copy(
                                            _tiled_copy_t2r_atom,
                                            tTR_tAcc[(None, None, None, ld_idx)],
                                            tTR_rAcc,
                                        )

                                        # Cast to c_dtype and stage through SMEM;
                                        # scatter below reloads raw b16 payloads.
                                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                                        if num_k_tiles == 0:
                                            acc_vec = cute.zeros_like(acc_vec)
                                        ld_rC.store(acc_vec.to(self.c_dtype))
                                    cute.copy(
                                        tiled_copy_r2s,
                                        subtile_rC[subtile_idx],
                                        tRS_sC[(None, None, None, 0)],
                                    )

                                    cute.arch.fence_proxy(
                                        "async.shared",
                                        space="cta",
                                    )
                                    if cutlass.const_expr(subtile_idx == first_held):
                                        # Doubles as the C-staging barrier.
                                        release_acc(
                                            tmem_empty_mbar + mma_m_idx,
                                            DIST_EPILOG_WARP_IDS[0],
                                            _DIST_EPILOG_WG_THREADS,
                                            _BAR_EPILOG_SYNC,
                                            NUM_CTAS,
                                        )
                                    else:
                                        cute.arch.barrier(
                                            barrier_id=_BAR_EPILOG_SYNC,
                                            number_of_threads=_DIST_EPILOG_WG_THREADS,
                                        )

                                    if cutlass.const_expr(fuse_swiglu_fwd):
                                        swiglu_mma_m_idx, swiglu_mma_n_idx = _swap_if(
                                            self.SWAP_AB, mma_m_idx, mma_n_idx
                                        )
                                        (
                                            swiglu_num_mma_atoms_m,
                                            swiglu_num_mma_atoms_n,
                                        ) = _swap_if(
                                            self.SWAP_AB,
                                            NUM_MMA_ATOMS_M,
                                            NUM_MMA_ATOMS_N,
                                        )
                                        store_interleaved_swiglu_smem_tile(
                                            tidx=tidx,
                                            sC_stage=row_major_sC[(None, None, 0)],
                                            m_output_words=mSwigluOutputWords,
                                            tile_m_idx=token_tile_idx,
                                            tile_n_idx=feature_tile_idx,
                                            mma_m_idx=swiglu_mma_m_idx,
                                            mma_n_idx=swiglu_mma_n_idx,
                                            subtile_idx=subtile_idx,
                                            num_mma_atoms_m=swiglu_num_mma_atoms_m,
                                            num_mma_atoms_n=swiglu_num_mma_atoms_n,
                                            cm_start=cm_start,
                                            m_size=token_size,
                                            n_size=feature_size,
                                            block_size_m=token_block,
                                            block_size_n=feature_block,
                                            num_ctas=NUM_CTAS,
                                            cluster_cta_rank=cluster_cta_rank,
                                            cluster_split_m=not self.SWAP_AB,
                                            subtile_axis_m=self.SWAP_AB,
                                            epilogue_threads=_DIST_EPILOG_WG_THREADS,
                                            fast_math=self.swiglu_fast_math,
                                            swizzle_group_size=NATIVE_SWIZZLE_GROUP_SIZE,
                                        )
                                    else:
                                        self._combine_epilog_scatter_smem_tile(
                                            tidx=tidx,
                                            sC=row_major_sC,
                                            scatter_ptr_smem_ptr=ptr_tab,
                                            tile_m_idx=token_tile_idx,
                                            tile_n_idx=feature_tile_idx,
                                            mma_m_idx=mma_m_idx,
                                            mma_n_idx=mma_n_idx,
                                            subtile_idx=subtile_idx,
                                            NUM_MMA_ATOMS_M=NUM_MMA_ATOMS_M,
                                            NUM_MMA_ATOMS_N=NUM_MMA_ATOMS_N,
                                            cm_start=cm_start,
                                            m_size=token_size,
                                            n_size=feature_size,
                                            elem_size_bytes_c=elem_size_bytes_c,
                                            BLOCK_SIZE_M=token_block,
                                            BLOCK_SIZE_N=feature_block,
                                            NUM_CTAS=NUM_CTAS,
                                            cluster_cta_rank=cluster_cta_rank,
                                        )
                                    cute.arch.barrier(
                                        barrier_id=_BAR_EPILOG_SYNC,
                                        number_of_threads=_DIST_EPILOG_WG_THREADS,
                                    )

                    # Signal tile_done so producer can reuse slot.
                    warp_idx_local = cute.arch.make_warp_uniform(cute.arch.warp_idx())
                    scheduler.consumer_release_tile(
                        tile_producer_mbar,
                        tile_buf,
                        cluster_cta_rank,
                        warp_idx_local == DIST_EPILOG_WARP_IDS[0],
                    )

                    if cutlass.const_expr(not self.SPLIT_ATOM_ACCUM):
                        release_acc(
                            tmem_empty_mbar + tmem_buf,
                            DIST_EPILOG_WARP_IDS[0],
                            _DIST_EPILOG_WG_THREADS,
                            _BAR_EPILOG_SYNC,
                            NUM_CTAS,
                        )

                    accum_cnt_tile += cutlass.Int32(1)

                    tile_idx = scheduler.advance_consumer(accum_cnt_tile)
                    work = visitor.get_work(tile_idx)

    # Device kernel entries: dispatch_kernel (3 WGs, 384 threads) and
    # combine_kernel (2 WGs, 256 threads). Layout per WG:
    #   WG0 warps 0-3  (alloc 168 regs): EPILOG (with fused scatter in COMBINE)
    #   WG1 warps 4-7  (dealloc 40 regs): MMA (4) + TMA (5) + IDLE (6,7)
    #   WG2 warps 8-11 (alloc 168 regs): GATHER (DISPATCH only)

    @cute.kernel
    def dispatch_kernel(  # type: ignore[override]  # noqa: C901
        self,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        tma_atom_c: cute.CopyAtom,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.Layout | cute.ComposedLayout,
        epi_tile: cute.Tile,
        cluster_layout_vmnk: cute.Layout,
        tiled_mma: cute.TiledMma,
        split_sizes: cute.Tensor,
        counter_ptr: cute.Pointer,
        # Dist-specific gather scheduler (DISPATCH-only):
        gather_counter_ptr: cute.Pointer,
        a_buff_counter_ptr: cute.Pointer,
        gather_a_ptrs_ptr: cute.Pointer,
        scatter_ptrs_ptr: cute.Pointer,
        mSwigluOutputWords: cute.Tensor,
        padding_row_ptr: cutlass.Int64,
        tensormaps: cute.Tensor,
        a_base_ptr: cutlass.Int64,
        b_base_ptr: cutlass.Int64,
        c_base_ptr: cutlass.Int64,
        a_s0: cutlass.Int32,
        a_s1: cutlass.Int32,
        b_s0: cutlass.Int32,
        b_s1: cutlass.Int32,
        c_s0: cutlass.Int32,
        c_s1: cutlass.Int32,
        elem_size_bytes_a: cutlass.Constexpr[int],
        elem_size_bytes_b: cutlass.Constexpr[int],
        elem_size_bytes_c: cutlass.Constexpr[int],
        secondary_operands: GroupedGemmSecondaryOperands | None,
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        problem_type: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        WORLD_SIZE: cutlass.Constexpr[int],
        AG_BLOCK_SIZE_M: cutlass.Constexpr[int],
        AG_BLOCK_SIZE_K: cutlass.Constexpr[int],
        NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
        NUM_TMEM_BUFFERS: cutlass.Constexpr[int],
        NUM_C_STAGES: cutlass.Constexpr[int],
        NUM_TILE_BUFFERS: cutlass.Constexpr[int],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        NUM_CTAS: cutlass.Constexpr[int],
        # Activation-buffer mode: when USE_ACTIVATION_BUFFER is True, the
        # kernel patches `tma_atom_c` (y_out) and `tma_atom_a` (x_gathered
        # read) to point at `activation_buffer_base_ptr + offset`. Each
        # offset is loaded from a 1-element int64 device tensor at kernel
        # entry. Addresses are assumed 128-byte aligned. The GATHER WG
        # also uses `activation_buffer_base_ptr + gathered_offset` for
        # raw STG to the gathered local A buffer.
        activation_buffer_base_ptr: cutlass.Int64,
        activation_buffer_size_bytes: cutlass.Int64,
        output_offsets_ptr: cute.Pointer,
        gathered_offsets_ptr: cute.Pointer,
        swiglu_output_offsets_ptr: cute.Pointer,
        USE_ACTIVATION_BUFFER: cutlass.Constexpr[bool],
        # Device-side skip flag: when USE_CONDITIONAL_EXECUTION is True
        # the kernel returns early if conditional_execution_tensor[0]
        # is zero. Lets cudagraph replays skip recompute without a host
        # sync. A placeholder int32[1] is passed when disabled.
        conditional_execution_tensor: cute.Tensor,
        USE_CONDITIONAL_EXECUTION: cutlass.Constexpr[bool],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        cluster_cta_rank = cutlass.Int32(0)
        pred_cta0 = True
        if cutlass.const_expr(NUM_CTAS == 2):
            cluster_cta_rank = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            pred_cta0 = cluster_cta_rank == 0

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_full_mbar = storage.smem_full_mbar.data_ptr()
        ab_empty_mbar = storage.smem_empty_mbar.data_ptr()
        tmem_full_mbar = storage.tmem_full_mbar.data_ptr()
        tmem_empty_mbar = storage.tmem_empty_mbar.data_ptr()
        tile_consumer_mbar = storage.tile_id_consumer_mbar.data_ptr()
        tile_producer_mbar = storage.tile_id_producer_mbar.data_ptr()
        tile_cta_bar_mbar = (
            storage.tile_cta_bar_mbar.data_ptr() if NUM_CTAS == 2 else None
        )
        tmem_dealloc_mbar = (
            storage.tmem_dealloc_mbar.data_ptr() if NUM_CTAS == 2 else None
        )
        tile_id_smem_ptr = storage.tile_id_smem.data_ptr()
        gather_tile_idx_smem_ptr = storage.gather_tile_idx_smem.data_ptr()
        scatter_ptr_smem_ptr = storage.scatter_ptr_smem.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf

        if warp_idx == DIST_EPILOG_WARP_IDS[0]:
            with cute.arch.elect_one():
                for i in range(NUM_SMEM_BUFFERS):
                    cute.arch.mbarrier_init(ab_empty_mbar + i, 1)
                for i in range(NUM_SMEM_BUFFERS):
                    cute.arch.mbarrier_init(ab_full_mbar + i, 1)
                for i in range(self.NUM_ACC_BARRIERS):
                    cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_empty_mbar + i, NUM_CTAS)
                for i in range(NUM_TILE_BUFFERS):
                    cute.arch.mbarrier_init(tile_consumer_mbar + i, 1)
                    cute.arch.mbarrier_init(tile_producer_mbar + i, NUM_CTAS)
                if cutlass.const_expr(NUM_CTAS == 2):
                    for i in range(2):
                        cute.arch.mbarrier_init(tile_cta_bar_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_dealloc_mbar, 32)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(NUM_CTAS == 2):
            # Cluster-scope async-shared proxy fence: required between
            # DSMEM writes and the cluster mbarrier arrive that consumes
            # them; CTA-scope (`fence_view_async_shared`) is insufficient.
            cute.arch.fence_proxy(kind="async.shared", space="cluster")
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(
                barrier_id=_BAR_FULL_CTA_SYNC,
                number_of_threads=DIST_THREADS_PER_CTA_DISPATCH,
            )

        # PDL: the barrier setup above touches no global memory, so it overlaps
        # the previous kernel. Wait before the first global read (the flag below).
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        # Conditional execution: gate body calls on a device flag.
        # CuTeDSL forbids early returns from `@cute.kernel`, so we
        # materialize a runtime bool and gate each body individually.
        # TMEM alloc + dealloc still run so resources release cleanly.
        # Placed AFTER cluster sync so all CTAs decide identically.
        should_run = cutlass.Boolean(True)
        if cutlass.const_expr(USE_CONDITIONAL_EXECUTION):
            should_run = conditional_execution_tensor[0] != cutlass.Int32(0)

        if warp_idx == DIST_EPILOG_WARP_IDS[0]:
            cute.arch.alloc_tmem(
                self.num_tmem_alloc_cols,
                tmem_holding_buf,
                is_two_cta=(NUM_CTAS == 2),
            )

        split_sizes = stage_expert_metadata(
            split_sizes,
            storage.split_sizes_smem.data_ptr(),
            G,
            synchronize=False,
        )
        # CTA-wide sync so MMA can read the TMEM ptr written by EPILOG
        # warp 0; also covers init + alloc visibility in 1-CTA mode.
        cute.arch.barrier(
            barrier_id=0,
            number_of_threads=DIST_THREADS_PER_CTA_DISPATCH,
        )

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        if cutlass.const_expr(isinstance(c_smem_layout_staged, cute.ComposedLayout)):
            sC = storage.sC.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )
        else:
            sC = storage.sC.get_tensor(c_smem_layout_staged)

        gA = cute.local_tile(
            mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB = cute.local_tile(
            mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC = cute.local_tile(
            mC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )

        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cluster_cta_rank
        )

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((BLOCK_SIZE_M, BLOCK_SIZE_N))
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, NUM_TMEM_BUFFERS)
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        ab_empty_mcast_mask = None
        acc_full_mcast_mask = None
        if cutlass.const_expr(NUM_CTAS == 2):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            block_in_cluster_coord_vmnk_peer = (
                block_in_cluster_coord_vmnk[0] ^ 1,
                *block_in_cluster_coord_vmnk[1:],
            )
            a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=2
            )
            b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=1
            )
            ab_empty_mcast_mask = (
                a_full_mcast_mask
                | b_full_mcast_mask
                | a_full_mcast_mask_peer
                | b_full_mcast_mask_peer
            )
            acc_full_mcast_mask = cute.make_layout_image_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mode=0
            )

        a_copy_bytes = cute.size_in_bytes(
            sA.element_type,
            cute.slice_(a_smem_layout_staged, (None, None, None, 0)),
        )
        b_copy_bytes = cute.size_in_bytes(
            sB.element_type,
            cute.slice_(b_smem_layout_staged, (None, None, None, 0)),
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        num_tma_load_bytes = (a_copy_bytes + b_copy_bytes) * atom_thr_size

        # Tensormap manager + per-CTA workspace.
        grid_dim = cute.arch.grid_dim()
        tm_workspace_idx = (
            bid[2] * grid_dim[1] * grid_dim[0] + bid[1] * grid_dim[0] + bid[0]
        )
        tensormap_manager = utils.TensorMapManager(utils.TensorMapUpdateMode.SMEM, 128)
        tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tm_workspace_idx, 0, None)].iterator
        )
        tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tm_workspace_idx, 1, None)].iterator
        )
        tensormap_c_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tm_workspace_idx, 2, None)].iterator
        )
        tensormap_buffer_ptr = storage.tensormap_buffer.data_ptr()
        tensormap_a_smem_ptr = tensormap_buffer_ptr
        tensormap_b_smem_ptr = tensormap_buffer_ptr + 16
        tensormap_c_smem_ptr = tensormap_buffer_ptr + 32

        if warp_idx == DIST_TMA_WARP_ID:
            if cutlass.const_expr(not self.host_a_tensormap):
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, tensormap_a_smem_ptr, 5
                )
                _clear_tma_oob_prefetch_bit(tensormap_a_smem_ptr)
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_b, tensormap_b_smem_ptr, 5
            )
            _clear_tma_oob_prefetch_bit(tensormap_b_smem_ptr)
        if warp_idx == DIST_EPILOG_WARP_IDS[0]:
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_c, tensormap_c_smem_ptr, 0
            )
            _clear_tma_oob_prefetch_bit(tensormap_c_smem_ptr)
        tensormap_manager.fence_tensormap_initialization()

        # Activation-buffer mode: rewrite base pointers; per-group
        # tensormap update inherits from the base machinery. Addresses
        # are 128-byte aligned by host construction.
        effective_a_base_ptr = a_base_ptr
        effective_b_base_ptr = b_base_ptr
        effective_c_base_ptr = c_base_ptr
        effective_mSwigluOutputWords = mSwigluOutputWords
        if cutlass.const_expr(USE_ACTIVATION_BUFFER):
            y_off = cutlass.Int64(
                cute.make_tensor(output_offsets_ptr, cute.make_layout(1))[0]
            )
            x_g_off = cutlass.Int64(
                cute.make_tensor(gathered_offsets_ptr, cute.make_layout(1))[0]
            )
            if (cute.arch.block_idx()[0] == 0) & (tidx == 0):
                # Swapped dispatch transposes C and moves the tokens to B.
                output_row_stride = c_s1 if self.SWAP_AB else c_s0
                gathered_row_stride = b_s0 if self.SWAP_AB else a_s0
                gathered_elem_bytes = (
                    elem_size_bytes_b if self.SWAP_AB else elem_size_bytes_a
                )
                output_byte_extent = _packed_group_rows_byte_extent(
                    split_sizes,
                    a_s1 if problem_type == _WGRAD else output_row_stride,
                    elem_size_bytes_a if problem_type == _WGRAD else elem_size_bytes_c,
                    G,
                )
                gathered_byte_extent = _packed_group_rows_byte_extent(
                    split_sizes,
                    b_s1 if problem_type == _WGRAD else gathered_row_stride,
                    elem_size_bytes_b
                    if problem_type == _WGRAD
                    else gathered_elem_bytes,
                    G,
                )
                _require_valid_activation_buffer_range(
                    byte_offset=y_off,
                    byte_extent=output_byte_extent,
                    activation_buffer_size_bytes=activation_buffer_size_bytes,
                    warp_scoped_diagnostic=False,
                )
                _require_valid_activation_buffer_range(
                    byte_offset=x_g_off,
                    byte_extent=gathered_byte_extent,
                    activation_buffer_size_bytes=activation_buffer_size_bytes,
                    warp_scoped_diagnostic=False,
                )
            if cutlass.const_expr(problem_type == _WGRAD):
                effective_a_base_ptr = activation_buffer_base_ptr + y_off
                effective_b_base_ptr = activation_buffer_base_ptr + x_g_off
            elif cutlass.const_expr(self.SWAP_AB):
                # Swapped dispatch: the gathered tokens are the B operand.
                effective_b_base_ptr = activation_buffer_base_ptr + x_g_off
                effective_c_base_ptr = activation_buffer_base_ptr + y_off
            else:
                effective_a_base_ptr = activation_buffer_base_ptr + x_g_off
                effective_c_base_ptr = activation_buffer_base_ptr + y_off
            if cutlass.const_expr(self.mode == self.DISPATCH_SWIGLU_FWD):
                h2_off = cutlass.Int64(
                    cute.make_tensor(swiglu_output_offsets_ptr, cute.make_layout(1))[0]
                )
                if (cute.arch.block_idx()[0] == 0) & (tidx == 0):
                    _require_valid_activation_buffer_range(
                        byte_offset=h2_off,
                        byte_extent=_packed_group_rows_byte_extent(
                            split_sizes,
                            N // cutlass.Int32(2),
                            elem_size_bytes_c,
                            G,
                        ),
                        activation_buffer_size_bytes=activation_buffer_size_bytes,
                        warp_scoped_diagnostic=False,
                    )
                effective_mSwigluOutputWords = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Uint32,
                        activation_buffer_base_ptr + h2_off,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    mSwigluOutputWords.layout,
                )

        # Per-WG reg hints. RF budget DISPATCH: 4*32*(168+40+168) = 48128
        # < 64K SM100 cap. COMBINE: 4*32*(168+40) = 26624.
        if warp_idx < len(DIST_EPILOG_WARP_IDS):
            cute.arch.setmaxregister_increase(168)  # WG0: epilog
        elif warp_idx < GATHER_WARP_IDS[0]:
            cute.arch.setmaxregister_decrease(40)  # WG1: mma+tma+idle
        else:
            cute.arch.setmaxregister_increase(168)  # WG2: gather (DISPATCH only)

        if warp_idx >= GATHER_WARP_IDS[0]:  # GATHER WG (warps 8-11)
            if should_run:
                gather_base_ptr = effective_a_base_ptr
                gather_stride = a_s0
                gather_elem_size = elem_size_bytes_a
                gather_block_m: cutlass.Constexpr[int] = BLOCK_SIZE_M
                gather_features = K
                if cutlass.const_expr(problem_type == _WGRAD):
                    gather_base_ptr = effective_b_base_ptr
                    gather_stride = b_s1
                    gather_elem_size = elem_size_bytes_b
                    gather_block_m = AG_BLOCK_SIZE_M
                    gather_features = N
                if cutlass.const_expr(self.SWAP_AB):
                    gather_base_ptr = effective_b_base_ptr
                    gather_stride = b_s0
                    gather_elem_size = elem_size_bytes_b
                    gather_block_m = BLOCK_SIZE_N
                _dispatch_gather_producer_body(
                    gather_counter_addr_i64=cutlass.Int64(gather_counter_ptr.toint()),
                    a_buff_counter_addr_i64=cutlass.Int64(a_buff_counter_ptr.toint()),
                    gather_a_ptrs_addr_i64=cutlass.Int64(gather_a_ptrs_ptr.toint()),
                    padding_row_addr_i64=padding_row_ptr,
                    a_local_addr_i64=gather_base_ptr,
                    stride_am_elems=gather_stride,
                    gather_tile_idx_smem_ptr=gather_tile_idx_smem_ptr,
                    split_sizes=split_sizes,
                    elem_size_bytes_a=gather_elem_size,
                    G=G,
                    K=gather_features,
                    local_rank=local_rank,
                    WORLD_SIZE=WORLD_SIZE,
                    AG_BLOCK_SIZE_M=AG_BLOCK_SIZE_M,
                    AG_BLOCK_SIZE_K=AG_BLOCK_SIZE_K,
                    BLOCK_SIZE_M=gather_block_m,
                    has_padding_sentinels=self.has_padding_sentinels,
                    _BAR_GATHER_WG_INTERNAL=_BAR_GATHER_WG_INTERNAL,
                    _DIST_GATHER_WG_THREADS=_DIST_GATHER_WG_THREADS,
                    SEGMENT_SIGNALS=self.GATHER_SEGMENT_SIGNALS,
                    LOADS_PER_LANE=self.GATHER_LOADS_PER_LANE,
                    gather_columns=problem_type == _WGRAD,
                    K_SEGMENTS=self.GATHER_K_SEGMENTS,
                    M_INTERLEAVE=self.M_TILE_INTERLEAVE,
                )
        elif warp_idx == DIST_TMA_WARP_ID:  # TMA producer
            # No cross-WG handoff barrier: TMA busy-waits per (group,
            # gemm_m_idx) on `a_buff_counter` via `_tma_per_tile_wait`.
            # Gather WG counter increments ARE the gather→TMA signal.
            NUM_SUB_PER_GEMM_M_VAL: cutlass.Constexpr[int] = (
                BLOCK_SIZE_N if self.SWAP_AB else BLOCK_SIZE_M
            ) // AG_BLOCK_SIZE_M
            if should_run:
                self._tma_producer_body(
                    tma_atom_a=tma_atom_a,
                    tma_atom_b=tma_atom_b,
                    tAgA=tAgA,
                    tBgB=tBgB,
                    tAsA=tAsA,
                    tBsB=tBsB,
                    a_full_mcast_mask=a_full_mcast_mask,
                    b_full_mcast_mask=b_full_mcast_mask,
                    num_tma_load_bytes=num_tma_load_bytes,
                    tensormap_manager=tensormap_manager,
                    tensormap_a_ptr=tensormap_a_ptr,
                    tensormap_b_ptr=tensormap_b_ptr,
                    tensormap_a_smem_ptr=tensormap_a_smem_ptr,
                    tensormap_b_smem_ptr=tensormap_b_smem_ptr,
                    ab_full_mbar=ab_full_mbar,
                    ab_empty_mbar=ab_empty_mbar,
                    tile_producer_mbar=tile_producer_mbar,
                    tile_consumer_mbar=tile_consumer_mbar,
                    tile_cta_bar_mbar=tile_cta_bar_mbar,
                    tile_id_smem_ptr=tile_id_smem_ptr,
                    counter_ptr=counter_ptr,
                    split_sizes=split_sizes,
                    cluster_cta_rank=cluster_cta_rank,
                    a_base_ptr=effective_a_base_ptr,
                    b_base_ptr=effective_b_base_ptr,
                    a_s0=a_s0,
                    a_s1=a_s1,
                    b_s0=b_s0,
                    b_s1=b_s1,
                    elem_size_bytes_a=elem_size_bytes_a,
                    elem_size_bytes_b=elem_size_bytes_b,
                    use_activation_buffer=False,
                    activation_buffer_size_bytes=activation_buffer_size_bytes,
                    a_offset_tensor=split_sizes,  # unused dummy
                    b_offset_tensor=split_sizes,
                    a_buff_counter_addr_i64=cutlass.Int64(a_buff_counter_ptr.toint()),
                    wait_per_m_tile=True,
                    NUM_SUB_PER_GEMM_M=NUM_SUB_PER_GEMM_M_VAL,
                    G=G,
                    M=M,
                    N=N,
                    K=K,
                    local_rank=local_rank,
                    problem_type=problem_type,
                    FORCE_N_MAJOR=FORCE_N_MAJOR,
                    NUM_N_CLUSTERS=NUM_N_CLUSTERS,
                    WORLD_SIZE=WORLD_SIZE,
                    NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                    NUM_TILE_BUFFERS=NUM_TILE_BUFFERS,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    NUM_CTAS=NUM_CTAS,
                    secondary_operands=secondary_operands,
                    a_coord_layout=mA.layout,
                    b_coord_layout=mB.layout,
                )
        elif warp_idx == DIST_MMA_WARP_ID:  # MMA consumer
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            if should_run:
                self._mma_consumer_body(
                    tiled_mma=tiled_mma,
                    tCrA=tCrA,
                    tCrB=tCrB,
                    tCtAcc_base=tCtAcc_base,
                    ab_full_mbar=ab_full_mbar,
                    ab_empty_mbar=ab_empty_mbar,
                    tmem_full_mbar=tmem_full_mbar,
                    tmem_empty_mbar=tmem_empty_mbar,
                    tile_consumer_mbar=tile_consumer_mbar,
                    tile_id_smem_ptr=tile_id_smem_ptr,
                    ab_empty_mcast_mask=ab_empty_mcast_mask,
                    acc_full_mcast_mask=acc_full_mcast_mask,
                    split_sizes=split_sizes,
                    pred_cta0=pred_cta0,
                    G=G,
                    M=M,
                    N=N,
                    K=K,
                    problem_type=problem_type,
                    NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                    NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                    NUM_TILE_BUFFERS=NUM_TILE_BUFFERS,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    NUM_CTAS=NUM_CTAS,
                    FORCE_N_MAJOR=FORCE_N_MAJOR,
                    NUM_N_CLUSTERS=NUM_N_CLUSTERS,
                    secondary_operands=secondary_operands,
                )
        elif warp_idx < len(DIST_EPILOG_WARP_IDS):  # EPILOG (warps 0-3)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            (
                tiled_copy_t2r,
                tTR_tAcc_base_00,
                tTR_rAcc,
            ) = self._epilog_tmem_copy_and_partition(
                tidx=tidx,
                tAcc=tCtAcc_base,
                gC_mnl=tCgC,
                epi_tile=epi_tile,
                use_2cta_instrs=NUM_CTAS == 2,
                mma_m_idx=0,
                mma_n_idx=0,
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, sC.element_type)
            (
                tiled_copy_r2s,
                tRS_rC,
                tRS_sC,
            ) = self._epilog_smem_copy_and_partition(
                tiled_copy_t2r=tiled_copy_t2r,
                tTR_rC=tTR_rC,
                tidx=tidx,
                sC=sC,
            )
            if cutlass.const_expr(
                self.mode == self.DISPATCH_SWIGLU_FWD and not self.swiglu_emit_h1
            ):
                if should_run:
                    self._combine_epilog_consumer_body(
                        tidx=tidx,
                        tCtAcc_base=tCtAcc_base,
                        tCgC=tCgC,
                        sC=sC,
                        tTR_rAcc=tTR_rAcc,
                        tiled_copy_r2s=tiled_copy_r2s,
                        tRS_rC=tRS_rC,
                        tRS_sC=tRS_sC,
                        epi_tile=epi_tile,
                        tmem_full_mbar=tmem_full_mbar,
                        tmem_empty_mbar=tmem_empty_mbar,
                        tile_consumer_mbar=tile_consumer_mbar,
                        tile_producer_mbar=tile_producer_mbar,
                        tile_id_smem_ptr=tile_id_smem_ptr,
                        scatter_ptrs_addr_i64=cutlass.Int64(scatter_ptrs_ptr.toint()),
                        padding_row_addr_i64=padding_row_ptr,
                        scatter_ptr_smem_ptr=scatter_ptr_smem_ptr,
                        mSwigluOutputWords=effective_mSwigluOutputWords,
                        fuse_swiglu_fwd=True,
                        elem_size_bytes_c=elem_size_bytes_c,
                        split_sizes=split_sizes,
                        cluster_cta_rank=cluster_cta_rank,
                        G=G,
                        M=M,
                        N=N,
                        K=K,
                        local_rank=local_rank,
                        problem_type=problem_type,
                        FORCE_N_MAJOR=FORCE_N_MAJOR,
                        NUM_N_CLUSTERS=NUM_N_CLUSTERS,
                        WORLD_SIZE=WORLD_SIZE,
                        NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                        NUM_TILE_BUFFERS=NUM_TILE_BUFFERS,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        BLOCK_SIZE_K=BLOCK_SIZE_K,
                        NUM_CTAS=NUM_CTAS,
                        secondary_operands=secondary_operands,
                    )
            elif should_run:
                self._epilog_consumer_body(
                    tidx=tidx,
                    tma_atom_c=tma_atom_c,
                    tCtAcc_base=tCtAcc_base,
                    tCgC=tCgC,
                    sC=sC,
                    tTR_rAcc=tTR_rAcc,
                    tiled_copy_r2s=tiled_copy_r2s,
                    tRS_rC=tRS_rC,
                    tRS_sC=tRS_sC,
                    epi_tile=epi_tile,
                    tensormap_manager=tensormap_manager,
                    tensormap_c_ptr=tensormap_c_ptr,
                    tensormap_c_smem_ptr=tensormap_c_smem_ptr,
                    tmem_full_mbar=tmem_full_mbar,
                    tmem_empty_mbar=tmem_empty_mbar,
                    tile_consumer_mbar=tile_consumer_mbar,
                    tile_producer_mbar=tile_producer_mbar,
                    tile_id_smem_ptr=tile_id_smem_ptr,
                    c_base_ptr=effective_c_base_ptr,
                    c_s0=c_s0,
                    c_s1=c_s1,
                    elem_size_bytes_c=elem_size_bytes_c,
                    split_sizes=split_sizes,
                    cluster_cta_rank=cluster_cta_rank,
                    G=G,
                    M=M,
                    N=N,
                    K=K,
                    local_rank=local_rank,
                    problem_type=problem_type,
                    FORCE_N_MAJOR=FORCE_N_MAJOR,
                    NUM_N_CLUSTERS=NUM_N_CLUSTERS,
                    WORLD_SIZE=WORLD_SIZE,
                    NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                    NUM_C_STAGES=NUM_C_STAGES,
                    NUM_TILE_BUFFERS=NUM_TILE_BUFFERS,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    NUM_CTAS=NUM_CTAS,
                    swiglu_h2=h1_swiglu_epilogue(
                        self,
                        effective_mSwigluOutputWords,
                        effective_c_base_ptr,
                        M,
                        c_s0,
                    ),
                )
            # TMEM dealloc handshake (epilog warps).
            if warp_idx == DIST_EPILOG_WARP_IDS[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=(NUM_CTAS == 2))
            cute.arch.barrier(
                barrier_id=_BAR_EPILOG_SYNC,
                number_of_threads=_DIST_EPILOG_WG_THREADS,
            )
            if warp_idx == DIST_EPILOG_WARP_IDS[0]:
                if cutlass.const_expr(NUM_CTAS == 2):
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar,
                        peer_cta_rank_in_cluster=cluster_cta_rank ^ 1,
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar, 0)
                cute.arch.dealloc_tmem(
                    tmem_ptr,
                    self.num_tmem_alloc_cols,
                    is_two_cta=(NUM_CTAS == 2),
                )
        else:  # IDLE warps (6, 7)
            self._idle_body()

        if cutlass.const_expr(NUM_CTAS == 2):
            # Keep every active thread alive until peer DSMEM accesses complete.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    @cute.kernel
    def combine_kernel(  # type: ignore[override]  # noqa: C901
        self,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        tma_atom_c: cute.CopyAtom,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.Layout | cute.ComposedLayout,
        epi_tile: cute.Tile,
        cluster_layout_vmnk: cute.Layout,
        tiled_mma: cute.TiledMma,
        split_sizes: cute.Tensor,
        counter_ptr: cute.Pointer,
        # COMBINE_SWIGLU_BWD producer work / per-M-tile done counters.
        gather_counter: cute.Tensor,
        a_buff_counter: cute.Tensor,
        # Combine-mode scatter scheduler:
        scatter_ptrs_ptr: cute.Pointer,
        padding_row_ptr: cutlass.Int64,
        tensormaps: cute.Tensor,
        a_base_ptr: cutlass.Int64,
        b_base_ptr: cutlass.Int64,
        a_s0: cutlass.Int32,
        a_s1: cutlass.Int32,
        b_s0: cutlass.Int32,
        b_s1: cutlass.Int32,
        elem_size_bytes_a: cutlass.Constexpr[int],
        elem_size_bytes_b: cutlass.Constexpr[int],
        elem_size_bytes_c: cutlass.Constexpr[int],
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        problem_type: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        WORLD_SIZE: cutlass.Constexpr[int],
        NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
        NUM_TMEM_BUFFERS: cutlass.Constexpr[int],
        NUM_TILE_BUFFERS: cutlass.Constexpr[int],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        NUM_CTAS: cutlass.Constexpr[int],
        # Activation-buffer mode: when USE_ACTIVATION_BUFFER is True, the
        # kernel patches `tma_atom_a` (x read) to point at
        # `activation_buffer_base_ptr + input_offsets_ptr[0]`. Address
        # assumed 128-byte aligned. Combine has no GMEM C output (scatter
        # writes directly to peer combine buffers via scatter_ptrs).
        activation_buffer_base_ptr: cutlass.Int64,
        activation_buffer_size_bytes: cutlass.Int64,
        input_offsets_ptr: cute.Pointer,
        USE_ACTIVATION_BUFFER: cutlass.Constexpr[bool],
        # Device-side skip flag: when USE_CONDITIONAL_EXECUTION is True
        # the kernel returns early if conditional_execution_tensor[0]
        # is zero. Lets cudagraph replays skip recompute without a host
        # sync. A placeholder int32[1] is passed when disabled.
        conditional_execution_tensor: cute.Tensor,
        USE_CONDITIONAL_EXECUTION: cutlass.Constexpr[bool],
        # COMBINE_SWIGLU_BWD inputs: grad_h2 (GM, I) and h1 (GM, 2I).
        swiglu_dz_base_ptr: cutlass.Int64,
        swiglu_h1_base_ptr: cutlass.Int64,
        swiglu_dz_s0: cutlass.Int32,
        swiglu_h1_s0: cutlass.Int32,
        swiglu_dz_offsets_ptr: cute.Pointer,
        swiglu_h1_offsets_ptr: cute.Pointer,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        fuse_swiglu_bwd = self.mode == self.COMBINE_SWIGLU_BWD
        cta_threads = (
            DIST_THREADS_PER_CTA_DISPATCH
            if fuse_swiglu_bwd
            else DIST_THREADS_PER_CTA_COMBINE
        )

        cluster_cta_rank = cutlass.Int32(0)
        pred_cta0 = True
        if cutlass.const_expr(NUM_CTAS == 2):
            cluster_cta_rank = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            pred_cta0 = cluster_cta_rank == 0

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_full_mbar = storage.smem_full_mbar.data_ptr()
        ab_empty_mbar = storage.smem_empty_mbar.data_ptr()
        tmem_full_mbar = storage.tmem_full_mbar.data_ptr()
        tmem_empty_mbar = storage.tmem_empty_mbar.data_ptr()
        tile_consumer_mbar = storage.tile_id_consumer_mbar.data_ptr()
        tile_producer_mbar = storage.tile_id_producer_mbar.data_ptr()
        tile_cta_bar_mbar = (
            storage.tile_cta_bar_mbar.data_ptr() if NUM_CTAS == 2 else None
        )
        tmem_dealloc_mbar = (
            storage.tmem_dealloc_mbar.data_ptr() if NUM_CTAS == 2 else None
        )
        tile_id_smem_ptr = storage.tile_id_smem.data_ptr()
        scatter_ptr_smem_ptr = storage.scatter_ptr_smem.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf

        if warp_idx == DIST_EPILOG_WARP_IDS[0]:
            with cute.arch.elect_one():
                for i in range(NUM_SMEM_BUFFERS):
                    cute.arch.mbarrier_init(ab_empty_mbar + i, 1)
                for i in range(NUM_SMEM_BUFFERS):
                    cute.arch.mbarrier_init(ab_full_mbar + i, 1)
                for i in range(self.NUM_ACC_BARRIERS):
                    cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_empty_mbar + i, NUM_CTAS)
                for i in range(NUM_TILE_BUFFERS):
                    cute.arch.mbarrier_init(tile_consumer_mbar + i, 1)
                    cute.arch.mbarrier_init(tile_producer_mbar + i, NUM_CTAS)
                if cutlass.const_expr(NUM_CTAS == 2):
                    for i in range(2):
                        cute.arch.mbarrier_init(tile_cta_bar_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_dealloc_mbar, 32)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(NUM_CTAS == 2):
            # Cluster-scope async-shared proxy fence: required between
            # DSMEM writes and the cluster mbarrier arrive that consumes
            # them; CTA-scope (`fence_view_async_shared`) is insufficient.
            cute.arch.fence_proxy(kind="async.shared", space="cluster")
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(
                barrier_id=_BAR_FULL_CTA_SYNC,
                number_of_threads=cta_threads,
            )

        # PDL: wait before the first global read, as in the dispatch kernel.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        # Conditional execution: gate body calls on a device flag.
        # CuTeDSL forbids early returns from `@cute.kernel`; TMEM
        # alloc + dealloc still run so resources release cleanly.
        should_run = cutlass.Boolean(True)
        if cutlass.const_expr(USE_CONDITIONAL_EXECUTION):
            should_run = conditional_execution_tensor[0] != cutlass.Int32(0)

        if warp_idx == DIST_EPILOG_WARP_IDS[0]:
            cute.arch.alloc_tmem(
                self.num_tmem_alloc_cols,
                tmem_holding_buf,
                is_two_cta=(NUM_CTAS == 2),
            )

        split_sizes = stage_expert_metadata(
            split_sizes,
            storage.split_sizes_smem.data_ptr(),
            G,
            synchronize=False,
        )
        # CTA-wide sync so MMA can read the TMEM ptr written by EPILOG
        # warp 0; also covers init + alloc visibility in 1-CTA mode.
        cute.arch.barrier(
            barrier_id=0,
            number_of_threads=cta_threads,
        )

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        if cutlass.const_expr(isinstance(c_smem_layout_staged, cute.ComposedLayout)):
            sC = storage.sC.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )
        else:
            sC = storage.sC.get_tensor(c_smem_layout_staged)

        gA = cute.local_tile(
            mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB = cute.local_tile(
            mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC = cute.local_tile(
            mC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )

        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cluster_cta_rank
        )

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((BLOCK_SIZE_M, BLOCK_SIZE_N))
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, NUM_TMEM_BUFFERS)
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        ab_empty_mcast_mask = None
        acc_full_mcast_mask = None
        if cutlass.const_expr(NUM_CTAS == 2):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            block_in_cluster_coord_vmnk_peer = (
                block_in_cluster_coord_vmnk[0] ^ 1,
                *block_in_cluster_coord_vmnk[1:],
            )
            a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=2
            )
            b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=1
            )
            ab_empty_mcast_mask = (
                a_full_mcast_mask
                | b_full_mcast_mask
                | a_full_mcast_mask_peer
                | b_full_mcast_mask_peer
            )
            acc_full_mcast_mask = cute.make_layout_image_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mode=0
            )

        a_copy_bytes = cute.size_in_bytes(
            sA.element_type,
            cute.slice_(a_smem_layout_staged, (None, None, None, 0)),
        )
        b_copy_bytes = cute.size_in_bytes(
            sB.element_type,
            cute.slice_(b_smem_layout_staged, (None, None, None, 0)),
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        num_tma_load_bytes = (a_copy_bytes + b_copy_bytes) * atom_thr_size

        # Tensormap manager + per-CTA workspace.
        grid_dim = cute.arch.grid_dim()
        tm_workspace_idx = (
            bid[2] * grid_dim[1] * grid_dim[0] + bid[1] * grid_dim[0] + bid[0]
        )
        tensormap_manager = utils.TensorMapManager(utils.TensorMapUpdateMode.SMEM, 128)
        tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tm_workspace_idx, 0, None)].iterator
        )
        tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tm_workspace_idx, 1, None)].iterator
        )
        # COMBINE has no per-group C-tensormap update (no TMA C-store;
        # the EPILOG scatters via NVLink). The C SMEM-side init still
        # runs to keep the SMEM layout symmetric with DISPATCH.
        tensormap_buffer_ptr = storage.tensormap_buffer.data_ptr()
        tensormap_a_smem_ptr = tensormap_buffer_ptr
        tensormap_b_smem_ptr = tensormap_buffer_ptr + 16
        tensormap_c_smem_ptr = tensormap_buffer_ptr + 32

        if warp_idx == DIST_TMA_WARP_ID:
            if cutlass.const_expr(not self.host_a_tensormap):
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, tensormap_a_smem_ptr, 5
                )
                _clear_tma_oob_prefetch_bit(tensormap_a_smem_ptr)
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_b, tensormap_b_smem_ptr, 5
            )
            _clear_tma_oob_prefetch_bit(tensormap_b_smem_ptr)
        if warp_idx == DIST_EPILOG_WARP_IDS[0]:
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_c, tensormap_c_smem_ptr, 0
            )
            _clear_tma_oob_prefetch_bit(tensormap_c_smem_ptr)
        tensormap_manager.fence_tensormap_initialization()

        # Activation-buffer mode: rewrite base pointer; per-group
        # tensormap update inherits from the base machinery. Addresses
        # are 128-byte aligned by host construction.
        effective_a_base_ptr = a_base_ptr
        effective_b_base_ptr = b_base_ptr
        if cutlass.const_expr(USE_ACTIVATION_BUFFER):
            x_off = cutlass.Int64(
                cute.make_tensor(input_offsets_ptr, cute.make_layout(1))[0]
            )
            if (cute.arch.block_idx()[0] == 0) & (tidx == 0):
                # Swapped combine moves the input rows to B.
                _require_valid_activation_buffer_range(
                    byte_offset=x_off,
                    byte_extent=_packed_group_rows_byte_extent(
                        split_sizes,
                        b_s0 if self.SWAP_AB else a_s0,
                        elem_size_bytes_b if self.SWAP_AB else elem_size_bytes_a,
                        G,
                    ),
                    activation_buffer_size_bytes=activation_buffer_size_bytes,
                    warp_scoped_diagnostic=False,
                )
            if cutlass.const_expr(self.SWAP_AB):
                effective_b_base_ptr = activation_buffer_base_ptr + x_off
            else:
                effective_a_base_ptr = activation_buffer_base_ptr + x_off

        if cutlass.const_expr(USE_ACTIVATION_BUFFER and fuse_swiglu_bwd):
            dz_off = cutlass.Int64(
                cute.make_tensor(swiglu_dz_offsets_ptr, cute.make_layout(1))[0]
            )
            h1_off = cutlass.Int64(
                cute.make_tensor(swiglu_h1_offsets_ptr, cute.make_layout(1))[0]
            )
            _require_valid_activation_buffer_range(
                byte_offset=dz_off,
                byte_extent=_packed_group_rows_byte_extent(
                    split_sizes, swiglu_dz_s0, elem_size_bytes_a, G
                ),
                activation_buffer_size_bytes=activation_buffer_size_bytes,
                warp_scoped_diagnostic=False,
            )
            _require_valid_activation_buffer_range(
                byte_offset=h1_off,
                byte_extent=_packed_group_rows_byte_extent(
                    split_sizes, swiglu_h1_s0, elem_size_bytes_a, G
                ),
                activation_buffer_size_bytes=activation_buffer_size_bytes,
                warp_scoped_diagnostic=False,
            )
            swiglu_dz_base_ptr = activation_buffer_base_ptr + dz_off
            swiglu_h1_base_ptr = activation_buffer_base_ptr + h1_off

        if warp_idx < len(DIST_EPILOG_WARP_IDS):
            cute.arch.setmaxregister_increase(168)  # WG0: epilog
        elif warp_idx < GATHER_WARP_IDS[0]:
            cute.arch.setmaxregister_decrease(40)  # WG1: mma+tma+idle
        elif cutlass.const_expr(fuse_swiglu_bwd):
            cute.arch.setmaxregister_increase(168)  # WG2: SwiGLU-bwd producer

        # Plain COMBINE launches no warps 8-11.
        if warp_idx == DIST_TMA_WARP_ID:  # TMA producer
            if should_run:
                self._tma_producer_body(
                    tma_atom_a=tma_atom_a,
                    tma_atom_b=tma_atom_b,
                    tAgA=tAgA,
                    tBgB=tBgB,
                    tAsA=tAsA,
                    tBsB=tBsB,
                    a_full_mcast_mask=a_full_mcast_mask,
                    b_full_mcast_mask=b_full_mcast_mask,
                    num_tma_load_bytes=num_tma_load_bytes,
                    tensormap_manager=tensormap_manager,
                    tensormap_a_ptr=tensormap_a_ptr,
                    tensormap_b_ptr=tensormap_b_ptr,
                    tensormap_a_smem_ptr=tensormap_a_smem_ptr,
                    tensormap_b_smem_ptr=tensormap_b_smem_ptr,
                    ab_full_mbar=ab_full_mbar,
                    ab_empty_mbar=ab_empty_mbar,
                    tile_producer_mbar=tile_producer_mbar,
                    tile_consumer_mbar=tile_consumer_mbar,
                    tile_cta_bar_mbar=tile_cta_bar_mbar,
                    tile_id_smem_ptr=tile_id_smem_ptr,
                    counter_ptr=counter_ptr,
                    split_sizes=split_sizes,
                    cluster_cta_rank=cluster_cta_rank,
                    a_base_ptr=effective_a_base_ptr,
                    b_base_ptr=effective_b_base_ptr,
                    a_s0=a_s0,
                    a_s1=a_s1,
                    b_s0=b_s0,
                    b_s1=b_s1,
                    elem_size_bytes_a=elem_size_bytes_a,
                    elem_size_bytes_b=elem_size_bytes_b,
                    use_activation_buffer=False,
                    activation_buffer_size_bytes=activation_buffer_size_bytes,
                    a_offset_tensor=split_sizes,  # unused dummy
                    b_offset_tensor=split_sizes,
                    a_buff_counter_addr_i64=cutlass.Int64(
                        a_buff_counter.iterator.toint()
                    ),
                    wait_per_m_tile=fuse_swiglu_bwd,
                    NUM_SUB_PER_GEMM_M=BLOCK_SIZE_M // NATIVE_SWIGLU_BWD_ROWS,
                    G=G,
                    M=M,
                    N=N,
                    K=K,
                    local_rank=local_rank,
                    problem_type=problem_type,
                    FORCE_N_MAJOR=FORCE_N_MAJOR,
                    NUM_N_CLUSTERS=NUM_N_CLUSTERS,
                    WORLD_SIZE=WORLD_SIZE,
                    NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                    NUM_TILE_BUFFERS=NUM_TILE_BUFFERS,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    NUM_CTAS=NUM_CTAS,
                    a_coord_layout=mA.layout,
                    b_coord_layout=mB.layout,
                )
        elif warp_idx == DIST_MMA_WARP_ID:  # MMA consumer
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            if should_run:
                self._mma_consumer_body(
                    tiled_mma=tiled_mma,
                    tCrA=tCrA,
                    tCrB=tCrB,
                    tCtAcc_base=tCtAcc_base,
                    ab_full_mbar=ab_full_mbar,
                    ab_empty_mbar=ab_empty_mbar,
                    tmem_full_mbar=tmem_full_mbar,
                    tmem_empty_mbar=tmem_empty_mbar,
                    tile_consumer_mbar=tile_consumer_mbar,
                    tile_id_smem_ptr=tile_id_smem_ptr,
                    ab_empty_mcast_mask=ab_empty_mcast_mask,
                    acc_full_mcast_mask=acc_full_mcast_mask,
                    split_sizes=split_sizes,
                    pred_cta0=pred_cta0,
                    G=G,
                    M=M,
                    N=N,
                    K=K,
                    problem_type=problem_type,
                    NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                    NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                    NUM_TILE_BUFFERS=NUM_TILE_BUFFERS,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    NUM_CTAS=NUM_CTAS,
                    FORCE_N_MAJOR=FORCE_N_MAJOR,
                    NUM_N_CLUSTERS=NUM_N_CLUSTERS,
                )
        elif warp_idx < len(DIST_EPILOG_WARP_IDS):  # EPILOG-with-scatter
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            (
                tiled_copy_t2r,
                _tTR_tAcc_base_00,
                tTR_rAcc,
            ) = self._epilog_tmem_copy_and_partition(
                tidx=tidx,
                tAcc=tCtAcc_base,
                gC_mnl=tCgC,
                epi_tile=epi_tile,
                use_2cta_instrs=NUM_CTAS == 2,
                mma_m_idx=0,
                mma_n_idx=0,
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, sC.element_type)
            (
                tiled_copy_r2s,
                tRS_rC,
                tRS_sC,
            ) = self._epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rC, tidx, sC)
            if should_run:
                self._combine_epilog_consumer_body(
                    tidx=tidx,
                    tCtAcc_base=tCtAcc_base,
                    tCgC=tCgC,
                    sC=sC,
                    tTR_rAcc=tTR_rAcc,
                    tiled_copy_r2s=tiled_copy_r2s,
                    tRS_rC=tRS_rC,
                    tRS_sC=tRS_sC,
                    epi_tile=epi_tile,
                    tmem_full_mbar=tmem_full_mbar,
                    tmem_empty_mbar=tmem_empty_mbar,
                    tile_consumer_mbar=tile_consumer_mbar,
                    tile_producer_mbar=tile_producer_mbar,
                    tile_id_smem_ptr=tile_id_smem_ptr,
                    scatter_ptrs_addr_i64=cutlass.Int64(scatter_ptrs_ptr.toint()),
                    padding_row_addr_i64=padding_row_ptr,
                    scatter_ptr_smem_ptr=scatter_ptr_smem_ptr,
                    mSwigluOutputWords=mC,
                    fuse_swiglu_fwd=False,
                    elem_size_bytes_c=elem_size_bytes_c,
                    split_sizes=split_sizes,
                    cluster_cta_rank=cluster_cta_rank,
                    G=G,
                    M=M,
                    N=N,
                    K=K,
                    local_rank=local_rank,
                    problem_type=problem_type,
                    FORCE_N_MAJOR=FORCE_N_MAJOR,
                    NUM_N_CLUSTERS=NUM_N_CLUSTERS,
                    WORLD_SIZE=WORLD_SIZE,
                    NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                    NUM_TILE_BUFFERS=NUM_TILE_BUFFERS,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    NUM_CTAS=NUM_CTAS,
                )
            # TMEM dealloc handshake (epilog warps).
            if warp_idx == DIST_EPILOG_WARP_IDS[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=(NUM_CTAS == 2))
            cute.arch.barrier(
                barrier_id=_BAR_EPILOG_SYNC,
                number_of_threads=_DIST_EPILOG_WG_THREADS,
            )
            if warp_idx == DIST_EPILOG_WARP_IDS[0]:
                if cutlass.const_expr(NUM_CTAS == 2):
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar,
                        peer_cta_rank_in_cluster=cluster_cta_rank ^ 1,
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar, 0)
                cute.arch.dealloc_tmem(
                    tmem_ptr,
                    self.num_tmem_alloc_cols,
                    is_two_cta=(NUM_CTAS == 2),
                )
        elif cutlass.const_expr(fuse_swiglu_bwd) and warp_idx >= GATHER_WARP_IDS[0]:
            if should_run:
                native_swiglu_bwd_producer_body(
                    mWorkCounter=gather_counter,
                    mDoneCounter=a_buff_counter,
                    split_sizes=split_sizes,
                    dz_base_ptr=swiglu_dz_base_ptr,
                    dz_s0=swiglu_dz_s0,
                    h1_base_ptr=swiglu_h1_base_ptr,
                    h1_s0=swiglu_h1_s0,
                    dxy_base_ptr=effective_a_base_ptr,
                    dxy_s0=a_s0,
                    K=K,
                    local_rank=local_rank,
                    producer_tidx=tidx - cutlass.Int32(GATHER_WARP_IDS[0] * 32),
                    dtype=self.a_dtype,
                    G=G,
                    WORLD_SIZE=WORLD_SIZE,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    ROWS=NATIVE_SWIGLU_BWD_ROWS,
                    PASSES_PER_BATCH=NATIVE_SWIGLU_BWD_PASSES_PER_BATCH,
                    fast_math=self.swiglu_fast_math,
                    clamped=self.swiglu_clamped,
                    alpha=self.swiglu_alpha,
                    limit=self.swiglu_limit,
                    emit_h2=self.swiglu_emit_h2,
                )
        else:  # IDLE warps (6, 7)
            self._idle_body()

        if cutlass.const_expr(NUM_CTAS == 2):
            # Keep every active thread alive until peer DSMEM accesses complete.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    @cute.jit
    def __call__(  # type: ignore[override]
        self,
        tensor_a: cute.Tensor,
        tensor_b: cute.Tensor,
        tensor_c: cute.Tensor,
        split_sizes: cute.Tensor,
        counter: cute.Tensor,
        gather_counter: cute.Tensor,
        a_buff_counter: cute.Tensor,
        h2_words: cute.Tensor,
        gather_a_ptrs: cute.Tensor,
        scatter_ptrs: cute.Tensor,
        tensormaps: cute.Tensor,
        padding_row_ptr: cutlass.Int64,
        a_base_ptr: cutlass.Int64,
        b_base_ptr: cutlass.Int64,
        c_base_ptr: cutlass.Int64,
        a_s0: cutlass.Int32,
        a_s1: cutlass.Int32,
        b_s0: cutlass.Int32,
        b_s1: cutlass.Int32,
        c_s0: cutlass.Int32,
        c_s1: cutlass.Int32,
        elem_size_bytes_a: cutlass.Constexpr[int],
        elem_size_bytes_b: cutlass.Constexpr[int],
        elem_size_bytes_c: cutlass.Constexpr[int],
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        # Activation-buffer mode (see kernel docstring).
        activation_buffer_base_ptr: cutlass.Int64,
        activation_buffer_size_bytes: cutlass.Int64,
        output_offsets: cute.Tensor,
        gathered_offsets: cute.Tensor,
        input_offsets: cute.Tensor,
        swiglu_output_offsets: cute.Tensor,
        use_activation_buffer: cutlass.Constexpr[bool],
        # Device-side skip flag (see kernel comment on
        # USE_CONDITIONAL_EXECUTION). Always present in the signature; a
        # placeholder int32[1] tensor is passed when the caller did not
        # supply one and `use_conditional_execution=False` makes the
        # check a constexpr no-op.
        conditional_execution_tensor: cute.Tensor,
        use_conditional_execution: cutlass.Constexpr[bool],
        # COMBINE_SWIGLU_BWD grad_h2 / h1 operands; 0 for every other mode.
        swiglu_dz_base_ptr: cutlass.Int64,
        swiglu_h1_base_ptr: cutlass.Int64,
        swiglu_dz_s0: cutlass.Int32,
        swiglu_h1_s0: cutlass.Int32,
        swiglu_dz_offsets: cute.Tensor,
        swiglu_h1_offsets: cute.Tensor,
        num_clusters: int,
        stream: cuda.CUstream,
        secondary_a_base_ptr: cutlass.Int64 | None = None,
        secondary_b_base_ptr: cutlass.Int64 | None = None,
        secondary_a_s0: cutlass.Int32 | None = None,
        secondary_a_s1: cutlass.Int32 | None = None,
        secondary_b_s0: cutlass.Int32 | None = None,
        secondary_b_s1: cutlass.Int32 | None = None,
        secondary_ready_counter: cute.Tensor | None = None,
        secondary_ready_feature_tiles: cutlass.Int32 | None = None,
    ):
        """Build TMA atoms + SharedStorage, launch fused dist_kernel
        with 384 threads/CTA."""
        import cutlass.utils.blackwell_helpers as sm100_utils

        tensor_a, tensor_b = _swap_if(self.SWAP_AB, tensor_a, tensor_b)
        a_base_ptr, b_base_ptr = _swap_if(self.SWAP_AB, a_base_ptr, b_base_ptr)
        (a_s0, a_s1), (b_s0, b_s1) = _swap_if(
            self.SWAP_AB,
            (a_s0, a_s1),
            (b_s0, b_s1),
        )
        elem_size_bytes_a, elem_size_bytes_b = _swap_if(
            self.SWAP_AB,
            elem_size_bytes_a,
            elem_size_bytes_b,
        )
        tensor_c = _transpose_c_if_swap(tensor_c, self.SWAP_AB)
        if cutlass.const_expr(self.SWAP_AB):
            c_s0, c_s1 = c_s1, c_s0

        self.a_dtype = tensor_a.element_type
        self.b_dtype = tensor_b.element_type
        self.c_dtype = tensor_c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(tensor_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(tensor_b).mma_major_mode()
        if cutlass.const_expr(self.block_major_weights):
            # Each group's [N, K] weight is stored as [K / 64, N, 64]: view the
            # (N, K, G) weight operand (A when swapped, else B) with a two-level
            # K mode so each 64-wide K block of a tile's rows is one contiguous
            # TMA box.
            weights = tensor_a if self.SWAP_AB else tensor_b
            n_w, k_w, g_w = weights.shape
            weights = cute.make_tensor(
                weights.iterator,
                cute.make_layout(
                    (n_w, (BLOCK_MAJOR_K, k_w // BLOCK_MAJOR_K), g_w),
                    stride=(BLOCK_MAJOR_K, (1, BLOCK_MAJOR_K * n_w), n_w * k_w),
                ),
            )
            if cutlass.const_expr(self.SWAP_AB):
                tensor_a = weights
            else:
                tensor_b = weights
        self.c_layout = utils.LayoutEnum.from_tensor(tensor_c)

        from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode

        if cutlass.const_expr(self.problem_type == _DGRAD):
            self.b_major_mode = OperandMajorMode.MN
        elif cutlass.const_expr(self.problem_type == _WGRAD):
            self.a_major_mode = OperandMajorMode.MN
            self.b_major_mode = OperandMajorMode.MN

        self._setup_attributes()

        # TMA atoms.
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            tensor_a,
            a_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        if cutlass.const_expr(self.swiglu_emit_h1):
            tensor_b = interleaved_fc13_view(tensor_b, 0)
            tensor_c = interleaved_fc13_view(tensor_c, 1)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            tensor_b,
            b_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        c_cta_v_layout = cute.composition(
            cute.make_identity_layout(tensor_c.shape), self.epi_tile
        )
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyReduceBulkTensorTileS2GOp()
            if self.output_accum
            else cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            c_cta_v_layout,
        )

        self.shared_storage = self._make_shared_storage(
            self.a_dtype, self.b_dtype, self.c_dtype, G
        )

        secondary_operands = None
        if cutlass.const_expr(secondary_a_base_ptr is not None):
            secondary_operands = GroupedGemmSecondaryOperands(
                a_base_ptr=secondary_a_base_ptr,
                b_base_ptr=secondary_b_base_ptr,
                a_strides=(secondary_a_s0, secondary_a_s1),
                b_strides=(secondary_b_s0, secondary_b_s1),
                elem_sizes=(elem_size_bytes_a, elem_size_bytes_b),
                ready_counter=secondary_ready_counter,
                ready_feature_tiles=secondary_ready_feature_tiles,
                schedule_rank=local_rank,
            )

        grid = (num_clusters * self.NUM_CTAS, 1, 1)
        if cutlass.const_expr(
            self.mode == DistGroupedGemmKernel.DISPATCH
            or self.mode == DistGroupedGemmKernel.DISPATCH_SWIGLU_FWD
        ):
            self.dispatch_kernel(
                tma_atom_a=tma_atom_a,
                tma_atom_b=tma_atom_b,
                tma_atom_c=tma_atom_c,
                mA=tma_tensor_a,
                mB=tma_tensor_b,
                mC=tma_tensor_c,
                a_smem_layout_staged=self.a_smem_layout_staged,
                b_smem_layout_staged=self.b_smem_layout_staged,
                epi_smem_layout_staged=self.epi_smem_layout_staged,
                c_smem_layout_staged=self.c_smem_layout_staged,
                epi_tile=self.epi_tile,
                cluster_layout_vmnk=self.cluster_layout_vmnk,
                tiled_mma=self.tiled_mma,
                split_sizes=split_sizes,
                counter_ptr=counter.iterator,
                gather_counter_ptr=gather_counter.iterator,
                a_buff_counter_ptr=a_buff_counter.iterator,
                gather_a_ptrs_ptr=gather_a_ptrs.iterator,
                scatter_ptrs_ptr=scatter_ptrs.iterator,
                mSwigluOutputWords=h2_words,
                padding_row_ptr=padding_row_ptr,
                tensormaps=tensormaps,
                a_base_ptr=a_base_ptr,
                b_base_ptr=b_base_ptr,
                c_base_ptr=c_base_ptr,
                a_s0=a_s0,
                a_s1=a_s1,
                b_s0=b_s0,
                b_s1=b_s1,
                c_s0=c_s0,
                c_s1=c_s1,
                elem_size_bytes_a=elem_size_bytes_a,
                elem_size_bytes_b=elem_size_bytes_b,
                elem_size_bytes_c=elem_size_bytes_c,
                secondary_operands=secondary_operands,
                G=G,
                M=M,
                N=N,
                K=K,
                local_rank=local_rank,
                problem_type=self.problem_type,
                FORCE_N_MAJOR=self.force_n_major,
                NUM_N_CLUSTERS=self.num_n_clusters,
                WORLD_SIZE=self.schedule_world_size,
                AG_BLOCK_SIZE_M=self.AG_BLOCK_SIZE_M,
                AG_BLOCK_SIZE_K=self.AG_BLOCK_SIZE_K,
                NUM_SMEM_BUFFERS=self.NUM_SMEM_BUFFERS,
                NUM_TMEM_BUFFERS=self.NUM_TMEM_BUFFERS,
                NUM_C_STAGES=self.NUM_C_STAGES,
                NUM_TILE_BUFFERS=self.NUM_TILE_BUFFERS,
                BLOCK_SIZE_M=self.BLOCK_SIZE_M,
                BLOCK_SIZE_N=self.BLOCK_SIZE_N,
                BLOCK_SIZE_K=self.BLOCK_SIZE_K,
                NUM_CTAS=self.NUM_CTAS,
                activation_buffer_base_ptr=activation_buffer_base_ptr,
                activation_buffer_size_bytes=activation_buffer_size_bytes,
                output_offsets_ptr=output_offsets.iterator,
                gathered_offsets_ptr=gathered_offsets.iterator,
                swiglu_output_offsets_ptr=swiglu_output_offsets.iterator,
                USE_ACTIVATION_BUFFER=use_activation_buffer,
                conditional_execution_tensor=conditional_execution_tensor,
                USE_CONDITIONAL_EXECUTION=use_conditional_execution,
            ).launch(
                grid=grid,
                block=(DIST_THREADS_PER_CTA_DISPATCH, 1, 1),
                cluster=(*self.cluster_shape_mn, 1),
                stream=stream,
                use_pdl=self.use_pdl,
            )
        else:  # COMBINE
            self.combine_kernel(
                tma_atom_a=tma_atom_a,
                tma_atom_b=tma_atom_b,
                tma_atom_c=tma_atom_c,
                mA=tma_tensor_a,
                mB=tma_tensor_b,
                mC=tma_tensor_c,
                a_smem_layout_staged=self.a_smem_layout_staged,
                b_smem_layout_staged=self.b_smem_layout_staged,
                epi_smem_layout_staged=self.epi_smem_layout_staged,
                c_smem_layout_staged=self.c_smem_layout_staged,
                epi_tile=self.epi_tile,
                cluster_layout_vmnk=self.cluster_layout_vmnk,
                tiled_mma=self.tiled_mma,
                split_sizes=split_sizes,
                counter_ptr=counter.iterator,
                gather_counter=gather_counter,
                a_buff_counter=a_buff_counter,
                scatter_ptrs_ptr=scatter_ptrs.iterator,
                padding_row_ptr=padding_row_ptr,
                tensormaps=tensormaps,
                a_base_ptr=a_base_ptr,
                b_base_ptr=b_base_ptr,
                a_s0=a_s0,
                a_s1=a_s1,
                b_s0=b_s0,
                b_s1=b_s1,
                elem_size_bytes_a=elem_size_bytes_a,
                elem_size_bytes_b=elem_size_bytes_b,
                elem_size_bytes_c=elem_size_bytes_c,
                G=G,
                M=M,
                N=N,
                K=K,
                local_rank=local_rank,
                problem_type=self.problem_type,
                FORCE_N_MAJOR=self.force_n_major,
                NUM_N_CLUSTERS=self.num_n_clusters,
                WORLD_SIZE=self.schedule_world_size,
                NUM_SMEM_BUFFERS=self.NUM_SMEM_BUFFERS,
                NUM_TMEM_BUFFERS=self.NUM_TMEM_BUFFERS,
                NUM_TILE_BUFFERS=self.NUM_TILE_BUFFERS,
                BLOCK_SIZE_M=self.BLOCK_SIZE_M,
                BLOCK_SIZE_N=self.BLOCK_SIZE_N,
                BLOCK_SIZE_K=self.BLOCK_SIZE_K,
                NUM_CTAS=self.NUM_CTAS,
                activation_buffer_base_ptr=activation_buffer_base_ptr,
                activation_buffer_size_bytes=activation_buffer_size_bytes,
                input_offsets_ptr=input_offsets.iterator,
                USE_ACTIVATION_BUFFER=use_activation_buffer,
                conditional_execution_tensor=conditional_execution_tensor,
                USE_CONDITIONAL_EXECUTION=use_conditional_execution,
                swiglu_dz_base_ptr=swiglu_dz_base_ptr,
                swiglu_h1_base_ptr=swiglu_h1_base_ptr,
                swiglu_dz_s0=swiglu_dz_s0,
                swiglu_h1_s0=swiglu_h1_s0,
                swiglu_dz_offsets_ptr=swiglu_dz_offsets.iterator,
                swiglu_h1_offsets_ptr=swiglu_h1_offsets.iterator,
            ).launch(
                grid=grid,
                block=(
                    DIST_THREADS_PER_CTA_DISPATCH
                    if self.mode == DistGroupedGemmKernel.COMBINE_SWIGLU_BWD
                    else DIST_THREADS_PER_CTA_COMBINE,
                    1,
                    1,
                ),
                cluster=(*self.cluster_shape_mn, 1),
                stream=stream,
                use_pdl=self.use_pdl,
            )
