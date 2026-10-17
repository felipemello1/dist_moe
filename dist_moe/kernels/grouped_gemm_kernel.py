# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GroupedGemmKernel: the bf16 grouped-GEMM kernel class and its device closure.

Split out of grouped_gemm.py, which keeps the host-side entry points,
config/workspace helpers, and the launch plumbing."""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op

from ._dsl_compat import thread_exit
from .blockscaled_quantization_common import device_trap
from .config import blockscaled_epilogue_subtile_divisor
from .grouped_gemm_tiles import (
    native_mma_tile,
    release_acc,
    split_atom_accum,
    split_atom_mma_tile,
)
from .swiglu_fragment import (
    interleaved_fc13_view,
    store_swiglu_h2_fragment,
    store_swiglu_h2_from_h1,
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


@functools.lru_cache(maxsize=None)
def _tmem_ld_wide_fragment_nvvm_broken() -> bool:
    """CuTeDSL 4.6.x ships a CUDA r12.9 NVVM backend that fails with
    "NVVM backend compilation failed" on Blackwell (`sm_100a` and `sm_103a`)
    when a single `tcgen05.ld` in this kernel's epilogue moves too wide a
    per-thread fragment. The exact breaking width varies with the enclosing
    kernel's complexity: a 64-register form (`32x32b.x64`, `16x256b.x16`)
    compiles in the plain grouped GEMM but fails in the fused dist kernels,
    while a 128-register form (`16x256b.x32`) fails even in the plain one.
    32-register forms compile everywhere, so `_cap_tmem_ld_repetition` tiles
    every wider load into 32-register instructions. Verified on
    nvidia-cutlass-dsl 4.6.1; 4.4.2 compiles all variants, so the workaround
    is gated on both the DSL version and the device."""
    version = str(getattr(cutlass, "__version__", ""))
    if not version.startswith("4.6"):
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


# Registers per thread moved by one repetition of each `tcgen05.ld` shape:
# datapaths x bits / (32 lanes x 32-bit registers). `Ld16x32bx2` covers two
# 16x32b halves per repetition.
_TMEM_LD_REGS_PER_REPETITION: dict[type, int] = {
    tcgen05.Ld32x32bOp: 1,
    tcgen05.Ld16x64bOp: 1,
    tcgen05.Ld16x32bx2Op: 1,
    tcgen05.Ld16x128bOp: 2,
    tcgen05.Ld16x256bOp: 4,
}

_TMEM_LD_MAX_REGS_PER_THREAD: int = 32


def _capped_tmem_ld_repetition(
    op: object,
    *,
    max_regs_per_thread: int = _TMEM_LD_MAX_REGS_PER_THREAD,
) -> tcgen05.Repetition | None:
    """The repetition to substitute for ``op``'s, or ``None`` to keep it.

    Caps one load instruction at 32 registers per thread (see
    ``_tmem_ld_wide_fragment_nvvm_broken``). Anything at or under the cap must
    be returned unchanged: an epilogue tile whose per-warp N is narrower than
    the atom — a ``(16,2)`` split tile has 16 columns — cannot host a wider
    atom at all.
    """
    repeat = getattr(op, "repeat", None)
    regs_per_repetition = _TMEM_LD_REGS_PER_REPETITION.get(type(op))
    if repeat is None or regs_per_repetition is None:
        return None
    max_repetition = max_regs_per_thread // regs_per_repetition
    if repeat.value <= max_repetition:
        return None
    return tcgen05.Repetition(max_repetition)


def _cap_tmem_ld_repetition(
    copy_atom: cute.CopyAtom,
    acc_dtype: type[cutlass.Numeric],
    *,
    max_regs_per_thread: int | None = None,
) -> cute.CopyAtom:
    """Cap affected TMEM loads at 32 registers per thread per instruction.

    CuTeDSL 4.6.x cannot lower wider loads on Blackwell. A caller may also
    supply its fixed fragment budget: fused native epilogues retain values
    across stores and SwiGLU and must fit their warpgroup register allocation.
    ``make_tmem_copy`` tiles the narrowed atom across the same logical epilogue
    tile, without adding epilogue passes or changing numerical operations.
    """
    if max_regs_per_thread is None:
        if not _tmem_ld_wide_fragment_nvvm_broken():
            return copy_atom
        max_regs_per_thread = _TMEM_LD_MAX_REGS_PER_THREAD
    op = copy_atom.op
    capped = _capped_tmem_ld_repetition(op, max_regs_per_thread=max_regs_per_thread)
    if capped is None:
        return copy_atom
    return cute.make_copy_atom(type(op)(capped, pack=op.pack), acc_dtype)


def _explicit_epilogue_tile_shape(
    *,
    cta_m: int,
    cta_n: int,
    num_ctas: int,
    epilogue_subtile: int,
    c_width: int,
    a_width: int,
) -> tuple[int, int, int, int]:
    """Return a tile whose C stage fits when C elements are wider than A."""
    warp_m, warp_n = (2, 2) if (cta_m == 64 and num_ctas == 2) else (4, 1)
    epilogue_divisor = blockscaled_epilogue_subtile_divisor(
        epilogue_subtile=epilogue_subtile,
        c_width=c_width,
        a_width=a_width,
    )
    tile_m = min(cta_m, 32 * warp_m)
    tile_n = cta_n // epilogue_divisor
    return tile_m, tile_n, warp_m, warp_n


@cute.jit
def _require_valid_activation_buffer_range(
    byte_offset: cutlass.Int64,
    byte_extent: cutlass.Int64,
    activation_buffer_size_bytes: cutlass.Int64,
    warp_scoped_diagnostic: cutlass.Constexpr[bool],
) -> None:
    if (
        (byte_offset < cutlass.Int64(0))
        | (byte_extent < cutlass.Int64(0))
        | (
            (byte_extent > cutlass.Int64(0))
            & (
                (byte_extent > activation_buffer_size_bytes)
                | (byte_offset > activation_buffer_size_bytes - byte_extent)
            )
        )
    ):
        if cutlass.const_expr(warp_scoped_diagnostic):
            diagnostic_thread = cute.arch.lane_idx() == 0
        else:
            diagnostic_thread = cute.arch.thread_idx()[0] == 0
        if diagnostic_thread & (cute.arch.block_idx()[0] == 0):
            cute.printf(
                "invalid activation-buffer range: offset=%lld, extent=%lld, size=%lld\n",
                byte_offset,
                byte_extent,
                activation_buffer_size_bytes,
            )
        device_trap()


@cute.jit
def _activation_buffer_rows(
    split_sizes: cute.Tensor,
    G: cutlass.Constexpr[int],
) -> cutlass.Int32:
    rows = cutlass.Int32(0)
    for group_idx in cutlass.range_constexpr(G):
        rows += cutlass.Int32(split_sizes[group_idx])
    return rows


@cute.jit
def _packed_group_rows_byte_extent(
    split_sizes: cute.Tensor,
    row_stride: cutlass.Int32,
    element_size_bytes: cutlass.Constexpr[int],
    group_count: cutlass.Constexpr[int],
) -> cutlass.Int64:
    total_rows = cutlass.Int64(0)
    for group_idx in cutlass.range(group_count, unroll=1):
        total_rows += cutlass.Int64(split_sizes[group_idx])
    return total_rows * cutlass.Int64(row_stride) * cutlass.Int64(element_size_bytes)


# Warp-group layout: 4 epilogue + 1 MMA + 1 TMA producer + 2 idle = 8 warps.
EPILOG_WARP_IDS: tuple[int, int, int, int] = (0, 1, 2, 3)


MMA_WARP_ID: int = 4


TMA_WARP_ID: int = 5


TOTAL_WARPS: int = 8


THREADS_PER_CTA: int = 32 * TOTAL_WARPS  # 256


# Named CTA-scope `bar.sync` ID allocation.
#
# Keep IDs contiguous and owned in one place. ID 0 is reserved for whole-CTA
# one-shot sync points; narrower participant barriers use IDs 1+.
_BAR_FULL_CTA_SYNC: int = 0
_BAR_EPILOG_SYNC: int = 1
# Used by the dist variant for an MMA + EPILOG warp-subset sync after
# its alloc_tmem (dist's warp layout differs so it needs a narrower
# barrier than the base kernel's CTA-wide post-alloc bar).
_BAR_TMEM_PTR_SYNC: int = 2


@cute.jit
def _get_group_sizes(
    split_sizes: cute.Tensor,
    g: cutlass.Int32,
    M: cutlass.Int32,
    N: cutlass.Int32,
    K: cutlass.Int32,
    problem_type: cutlass.Constexpr[int],
):
    """Per-group (m, n, k). FPROP/DGRAD: split_sizes is per-group M;
    WGRAD: split_sizes is per-group K."""
    m_size = cutlass.Int32(0)
    n_size = cutlass.Int32(0)
    k_size = cutlass.Int32(0)
    if cutlass.const_expr(problem_type == _FPROP or problem_type == _DGRAD):
        m_size = cutlass.Int32(split_sizes[g])
        n_size = N
        k_size = K
    else:  # _WGRAD
        m_size = M
        n_size = N
        k_size = cutlass.Int32(split_sizes[g])
    return m_size, n_size, k_size


def _transpose_first_two_modes(layout: cute.Layout) -> cute.Layout:
    """Transpose the logical C view without moving its backing storage."""
    shape = layout.shape
    stride = layout.stride
    return cute.make_layout(
        (shape[1], shape[0], *shape[2:]),
        stride=(stride[1], stride[0], *stride[2:]),
    )


def _swap_if(swap_ab: cutlass.Constexpr[bool], a, b):
    return (b, a) if cutlass.const_expr(swap_ab) else (a, b)


def _transpose_c_if_swap(tensor_c: cute.Tensor, swap_ab: cutlass.Constexpr[bool]):
    if cutlass.const_expr(swap_ab):
        return cute.make_tensor(
            tensor_c.iterator,
            _transpose_first_two_modes(tensor_c.layout),
        )
    return tensor_c


def _ceil_div(x, y):
    return (x + y - 1) // y


@cute.jit
def _get_tile_grid(
    m_size: cutlass.Int32,
    n_size: cutlass.Int32,
    k_size: cutlass.Int32,
    BLOCK_SIZE_M: cutlass.Constexpr[int],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
    BLOCK_SIZE_K: cutlass.Constexpr[int],
    FORCE_N_MAJOR: cutlass.Constexpr[bool],
    NUM_N_CLUSTERS: cutlass.Constexpr[int],
    SWAP_MN: cutlass.Constexpr[bool] = False,
):
    """Tile grid for one group.

    ``_ceil_div(0, BLOCK_M)`` is 0, so an empty group (m_size=0 in
    FPROP/DGRAD) naturally yields num_tiles=0 without a special-case
    branch. WGRAD's m_size=N is always > 0; the epilogue writes zeros
    when k_size == 0. 2-CTA cooperates on the same tile, so num_m_tiles
    is the cluster-tile count (no pairing pad).

    When N-clustering is active, the num_tiles calculation needs to round-up
    to a multiple of NUM_N_CLUSTERS to account for invalid, residual N tiles
    at the end of a N cluster. Otherwise, valid tiles may be skipped.
    """
    num_m_tiles = _ceil_div(m_size, BLOCK_SIZE_M)
    num_n_tiles = _ceil_div(n_size, BLOCK_SIZE_N)
    num_k_tiles = _ceil_div(k_size, BLOCK_SIZE_K)
    logical_num_m_tiles = num_m_tiles
    logical_num_n_tiles = num_n_tiles
    logical_m_size = m_size
    logical_n_size = n_size
    if cutlass.const_expr(SWAP_MN):
        logical_num_m_tiles = num_n_tiles
        logical_num_n_tiles = num_m_tiles
        logical_m_size = n_size
        logical_n_size = m_size

    num_tiles = num_m_tiles * num_n_tiles
    if cutlass.const_expr(NUM_N_CLUSTERS > 1):
        if cutlass.const_expr(FORCE_N_MAJOR) or not (logical_m_size < logical_n_size):
            n_tiles_per_cluster = _ceil_div(logical_num_n_tiles, NUM_N_CLUSTERS)
            num_tiles = (
                logical_num_m_tiles
                * n_tiles_per_cluster
                * cutlass.Int32(NUM_N_CLUSTERS)
            )
    return num_m_tiles, num_n_tiles, num_k_tiles, num_tiles


@cute.jit
def _remap_m_tile_idx(
    tile_m_idx: cutlass.Int32,
    num_m_tiles: cutlass.Int32,
    local_rank: cutlass.Int32,
    WORLD_SIZE: cutlass.Constexpr[int],
):
    """Per-rank circular rotation to spread NVLink reads across ranks."""
    if cutlass.const_expr(WORLD_SIZE > 1):
        m_offset = (local_rank * _ceil_div(num_m_tiles, WORLD_SIZE)) % num_m_tiles
        tile_m_idx = (tile_m_idx + m_offset) % num_m_tiles
    return tile_m_idx


@cute.jit
def _get_tile_coords(
    cur_tile_idx: cutlass.Int32,
    num_m_tiles: cutlass.Int32,
    num_n_tiles: cutlass.Int32,
    m_size: cutlass.Int32,
    n_size: cutlass.Int32,
    FORCE_N_MAJOR: cutlass.Constexpr[bool],
    NUM_N_CLUSTERS: cutlass.Constexpr[int],
    local_rank: cutlass.Int32,
    WORLD_SIZE: cutlass.Constexpr[int],
    SWAP_MN: cutlass.Constexpr[bool] = False,
):
    """N-major (M varies fastest) for tall/skinny shapes, or N-cluster
    swizzled to keep B tiles resident in L2. ``SWAP_MN`` runs this logic in
    logical M/N space, then swaps back to kernel M/N coordinates."""
    # Pre-declare so CuTeDSL flow analysis is happy.
    logical_tile_m_idx = cutlass.Int32(0)
    logical_tile_n_idx = cutlass.Int32(0)
    tile_m_idx = cutlass.Int32(0)
    tile_n_idx = cutlass.Int32(0)

    logical_num_m_tiles = num_m_tiles
    logical_num_n_tiles = num_n_tiles
    logical_m_size = m_size
    logical_n_size = n_size
    if cutlass.const_expr(SWAP_MN):
        logical_num_m_tiles = num_n_tiles
        logical_num_n_tiles = num_m_tiles
        logical_m_size = n_size
        logical_n_size = m_size

    if cutlass.const_expr(FORCE_N_MAJOR) or not (logical_m_size < logical_n_size):
        n_tiles_per_cluster = _ceil_div(logical_num_n_tiles, NUM_N_CLUSTERS)
        cluster_tiles = logical_num_m_tiles * n_tiles_per_cluster
        cluster_idx = cur_tile_idx // cluster_tiles
        within_cluster = cur_tile_idx % cluster_tiles
        n_offset = cluster_idx * n_tiles_per_cluster

        # Single (m, n) per cluster pair regardless of NUM_CTAS — both CTAs
        # cooperate on the same tile via cta_group=TWO MMA.
        logical_tile_m_idx = within_cluster // n_tiles_per_cluster
        logical_tile_n_idx = n_offset + within_cluster % n_tiles_per_cluster
    else:
        logical_tile_m_idx = cur_tile_idx % logical_num_m_tiles
        logical_tile_n_idx = cur_tile_idx // logical_num_m_tiles

    logical_tile_m_idx = _remap_m_tile_idx(
        logical_tile_m_idx, logical_num_m_tiles, local_rank, WORLD_SIZE
    )
    if cutlass.const_expr(SWAP_MN):
        tile_m_idx = logical_tile_n_idx
        tile_n_idx = logical_tile_m_idx
    else:
        tile_m_idx = logical_tile_m_idx
        tile_n_idx = logical_tile_n_idx
    return tile_m_idx, tile_n_idx


# createpolicy.fractional encodings for the TMA cache-policy operand.
_L2_EVICT_NORMAL = 0x1000000000000000
_L2_EVICT_FIRST = 0x12F0000000000000


@cute.jit
def _weight_l2_policy(group_rows, token_tile: cutlass.Constexpr[int]):
    """L2 policy for a group's weight loads. When the group's routed rows fit
    one token tile, each weight tile is read exactly once, so it streams
    evict-first instead of displacing the reused activation tiles and the
    combine writes. Larger groups re-read every weight tile per token tile
    and keep the normal policy."""
    # A select, not an `if`: a policy yielded by a dynamic `if` fails to
    # compile in the fused-SwiGLU dispatch kernels.
    larger = cutlass.Int64(group_rows > cutlass.Int32(token_tile))
    return cutlass.Int64(_L2_EVICT_FIRST) + larger * cutlass.Int64(
        _L2_EVICT_NORMAL - _L2_EVICT_FIRST
    )


@cute.jit
def _epilog_wait_pending_tma_store():
    """Drain the previous SMEM->GMEM store before reusing the C SMEM tile."""
    warp_idx_local = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx_local == EPILOG_WARP_IDS[0]:
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    cute.arch.barrier(
        barrier_id=_BAR_EPILOG_SYNC,
        number_of_threads=32 * len(EPILOG_WARP_IDS),
    )


@dsl_user_op
def _clear_tma_oob_prefetch_bit(
    smem_desc_ptr: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> None:
    """Clear bit 21 of `desc_u64[1]` (= bit 21 of the 64-bit flags word at
    offset 8 of the TMA descriptor staged in SMEM).

    Workaround for the Blackwell TMA OOB-past-allocation IMA: this bit
    enables a HW prefetch path that issues real GMEM loads to OOB box rows
    when the patched per-group `globalDim` is small. With it cleared, TMA
    falls back to the documented `tile`-mode OOB-zero-fill semantics —
    addresses past the live tensor are not actually read.

    Mirrors the NVIDIA driver-side workaround
        if (max_byte_index + 1 < 128*1024) desc_u64[1] &= ~(1ull << 21);
    we always clear unconditionally instead of doing the per-group
    `< 128 KB` check, since `update_tma_descriptor` only patches
    address/dim/stride and never touches this flags word — a single
    init-time clear in SMEM propagates to GMEM via every subsequent
    `cp_fence_tma_desc_release`.

    Bit 21 lives in the low 32 bits of `u64[1]` (= byte offset 8). The
    descriptor SMEM is allocated as `MemRange[Int64, 16*3]`, so the
    incoming pointer is Int64-typed and `+1` advances by 8 bytes; we
    recast to Uint32 to access the low 32 bits.
    """
    ptr_u32 = cute.recast_ptr(smem_desc_ptr + 1, dtype=cutlass.Uint32)
    with cute.arch.elect_one():
        val = cute.arch.load(ptr_u32, cutlass.Uint32, ss="cta")
        cute.arch.store(ptr_u32, val & cutlass.Uint32(0xFFDFFFFF))
    cute.arch.sync_warp()


class GroupedGemmKernel:
    """Persistent SM100 grouped GEMM (FPROP/DGRAD/WGRAD). __call__ takes
    torch tensors + a config dict and launches the device kernel."""

    TMEM_LOAD_MAX_REGS_PER_THREAD: int | None = None
    # K segments the dist DISPATCH gather signals separately per M-tile.
    GATHER_K_SEGMENTS: int = 1
    # Block size of the rotated dist schedule's source-segment interleave
    # (`_interleave_m_tile_idx`); 0 keeps the plain rotation.
    M_TILE_INTERLEAVE: int = 0
    # Rotate each M tile's N-band order (`_tile_coords` DIAG_N_BANDS). Every
    # warp role must decode tiles with the same value.
    DIAG_N_BANDS: bool = False
    # FPROP only (set by DistGroupedGemmKernel): read every group's A rows
    # through one descriptor at a row offset instead of patching, and draining
    # before, a descriptor per group. A tile's tail then loads the next group's
    # rows (possibly still being gathered). Safe: an output row depends only on
    # its own A row and every epilogue clips to m_size (per-group C descriptor,
    # fused SwiGLU stores, combine scatter `row_in_group < m_size`).
    SINGLE_A_TENSORMAP: bool = False

    def __init__(
        self,
        config: dict,
        problem_type: int,  # _FPROP / _DGRAD / _WGRAD
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        force_n_major: bool = False,
        num_n_clusters: int = 1,
        world_size: int = 1,
        static_scheduler: bool = False,
        host_a_tensormap: bool = False,
        swap_ab: bool = False,
        schedule_rotation: bool = True,
    ):
        if swap_ab and problem_type != _FPROP:
            raise ValueError("SWAP_AB is currently supported for FPROP only")
        self.config = dict(config)  # shallow copy so we can mutate locally
        self.problem_type = problem_type
        self.acc_dtype = acc_dtype
        self.force_n_major = force_n_major
        self.num_n_clusters = num_n_clusters
        self.world_size = world_size
        # Constexpr world size seeding the schedule rotation. 1 when the
        # routing layer pre-shuffled the rows (rotation compiles out) or
        # on a single rank.
        self.schedule_world_size = world_size if schedule_rotation else 1
        self.STATIC_SCHEDULER = static_scheduler
        self.SWAP_AB = swap_ab
        self.host_a_tensormap = host_a_tensormap and not swap_ab
        self.host_b_tensormap: bool = problem_type in (_FPROP, _DGRAD) and not swap_ab

        c = self.config
        self.NUM_MMAS: int = c["NUM_MMAS"]
        self.BLOCK_SIZE_M: int = c["BLOCK_SIZE_M"]
        self.BLOCK_SIZE_N: int = c["BLOCK_SIZE_N"]
        self.BLOCK_SIZE_K: int = c["BLOCK_SIZE_K"]
        self.NUM_SMEM_BUFFERS: int = c["NUM_SMEM_BUFFERS"]
        self.NUM_TMEM_BUFFERS: int = c["NUM_TMEM_BUFFERS"]
        self.NUM_C_STAGES: int = c.get("NUM_C_STAGES", self.NUM_TMEM_BUFFERS)
        if not 1 <= self.NUM_C_STAGES <= 8:
            raise ValueError(f"NUM_C_STAGES must be in [1, 8], got {self.NUM_C_STAGES}")
        self.NUM_TILE_BUFFERS: int = c["NUM_TILE_BUFFERS"]
        self.NUM_CTAS: int = c["NUM_CTAS"]
        self.NUM_TILE_CTA_BARS: int = 2 if self.NUM_CTAS == 2 else 0
        self.EPILOGUE_SUBTILE: int = c.get("EPILOGUE_SUBTILE", 0)  # 0 = auto
        self.SPLIT_ATOM_ACCUM: bool = split_atom_accum(
            self.NUM_TMEM_BUFFERS, self.NUM_MMAS
        )
        # One tmem_full / tmem_empty pair per accumulator hand-off unit.
        self.NUM_ACC_BARRIERS: int = (
            self.NUM_MMAS if self.SPLIT_ATOM_ACCUM else self.NUM_TMEM_BUFFERS
        )

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.NUM_CTAS == 2 else tcgen05.CtaGroup.ONE
        )
        # 2-CTA pairs along M, single along N.
        self.cluster_shape_mn: tuple[int, int] = (
            (2, 1) if self.NUM_CTAS == 2 else (1, 1)
        )
        self.mma_tiler_mn: tuple[int, int] = (
            self.BLOCK_SIZE_M,
            self.BLOCK_SIZE_N,
        )

    # 2-CTA: both CTAs cooperate on the full (M, N, K) tile via tiled_mma
    # with cta_group=TWO; we do NOT halve any axis manually. _make_shared_storage
    # must be called after _setup_attributes.
    def _make_shared_storage(self, a_dtype, b_dtype, c_dtype, G: int):
        NUM_SMEM = self.NUM_SMEM_BUFFERS
        NUM_TILE = self.NUM_TILE_BUFFERS
        NUM_CTAS = self.NUM_CTAS

        a_smem_elems = cute.cosize(self.a_smem_layout_staged.outer)
        b_smem_elems = cute.cosize(self.b_smem_layout_staged.outer)
        c_smem_elems = cute.cosize(self.epi_smem_layout_staged.outer)

        # mbarrier counts (one Int64 per mbarrier). Single empty-mbarrier per
        # SMEM buffer: the MMA consumer issues one combined tcgen05.commit
        # per kk (atom-along-M lives inside one cute.gemm call).
        n_smem_empty = NUM_SMEM
        n_smem_full = NUM_SMEM
        n_tmem_full = self.NUM_ACC_BARRIERS
        n_tmem_empty = self.NUM_ACC_BARRIERS
        n_tile_consumer = NUM_TILE
        n_tile_producer = NUM_TILE
        n_tile_cta_bar = 2 if NUM_CTAS == 2 else 0  # 2-CTA only
        n_tmem_dealloc = 1 if NUM_CTAS == 2 else 0  # 2-CTA only

        @cute.struct
        class SharedStorage:
            # SMEM staging for the epilogue (tmem -> reg -> sC -> gmem).
            sC: cute.struct.Align[cute.struct.MemRange[c_dtype, c_smem_elems], 1024]
            # Full-tile SMEM for A/B (2-CTA: each CTA holds its share).
            sA: cute.struct.Align[cute.struct.MemRange[a_dtype, a_smem_elems], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[b_dtype, b_smem_elems], 1024]
            # tile_id slots — one per pipeline stage of the tile-info protocol.
            tile_id_smem: cute.struct.MemRange[cutlass.Int32, NUM_TILE]
            split_sizes_smem: cute.struct.MemRange[cutlass.Int32, G]
            # SMEM-staged broadcast of warp-leader's atomic_add result, needed
            # because elect_one() values cannot escape its scf.if region.
            # SMEM staging buffer for tensormap descriptors — 3 × 128B.
            # In SMEM mode, the descriptor is patched in SMEM and atomically
            # copied to GMEM via `tensormap.cp_fenceproxy.global.shared::cta`.
            # Mirrors the cute/gemm reference layout (single 384B buffer,
            # three pointers offset by 16 Int64 each).
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

        return SharedStorage

    # @cute.kernel device entry — four warp-group branches dispatched to
    # helper bodies so the entry reads as a high-level orchestration.
    @cute.kernel
    def kernel(  # noqa: C901
        self,
        # ---- TMA atoms / global tensors -----------------------------------
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        tma_atom_c: cute.CopyAtom,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        # ---- Layouts ------------------------------------------------------
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        cluster_layout_vmnk: cute.Layout,
        tiled_mma: cute.TiledMma,
        # ---- Group + scheduler inputs -------------------------------------
        split_sizes: cute.Tensor,
        counter_ptr: cute.Pointer,  # global int32 counter
        # ---- Tensormap workspace ------------------------------------------
        tensormaps: cute.Tensor,
        # ---- Per-tensor base ptrs + strides as scalars (no per-launch
        #      device tensors). Per-group offsets computed inside producer
        #      and epilog bodies via TLX-style accumulators.
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
        # ---- Activation-buffer offset (TLX-style; no D2H sync) ------------
        # When `use_activation_buffer` is True, the kernel loads two int64
        # byte offsets from device tensors `a_offset_tensor` /
        # `b_offset_tensor` and adds them to a_base_ptr / b_base_ptr at
        # kernel entry. Use case (wgrad): A and B share a single
        # `activation_buffer` device allocation; the per-tensor offset
        # within it is computed on the device by an upstream op and fed
        # to this kernel without going through the host.
        use_activation_buffer: cutlass.Constexpr[bool],
        activation_buffer_size_bytes: cutlass.Int64,
        a_offset_tensor: cute.Tensor,
        b_offset_tensor: cute.Tensor,
        # ---- Problem dimensions -------------------------------------------
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        # ---- Compile-time problem flags -----------------------------------
        problem_type: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        WORLD_SIZE: cutlass.Constexpr[int],
        # ---- Pipeline depths (constexpr) ----------------------------------
        NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
        NUM_TMEM_BUFFERS: cutlass.Constexpr[int],
        NUM_C_STAGES: cutlass.Constexpr[int],
        NUM_TILE_BUFFERS: cutlass.Constexpr[int],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        NUM_CTAS: cutlass.Constexpr[int],
        condition: cute.Tensor,
        USE_CONDITIONAL_EXECUTION: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(USE_CONDITIONAL_EXECUTION):
            if condition[0] == 0:
                thread_exit()
        # ----- Warp / cluster identity (TLX prologue) ---------------------
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        cluster_cta_rank = cutlass.Int32(0)
        pred_cta0 = True
        if cutlass.const_expr(NUM_CTAS == 2):
            cluster_cta_rank = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            pred_cta0 = cluster_cta_rank == 0

        # ----- SMEM / mbarrier allocation ---------------------------------
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
        tmem_holding_buf = storage.tmem_holding_buf

        # mbarrier arrive_counts (one elected lane of one warp per CTA
        # performs the inits). Each count = arrives expected per cycle:
        #   ab_full / ab_empty:               1   (leader broadcasts via mcast)
        #   tmem_full:                        1   (MMA warp's tcgen05.commit)
        #   tmem_empty:                       NUM_CTAS — in 2-CTA both peers'
        #       epilogs arrive remotely on leader's bar so leader's next MMA
        #       waits for BOTH CTAs to finish reading tmem; otherwise peer
        #       can read tmem after leader writes the next tile, corrupting
        #       the gmem slab. (Ref: grouped_gemm acc_empty_mbar.)
        #   tile_consumer:                    1            (single TMA producer)
        #   tile_producer:                    NUM_CTAS     (each CTA's epilog)
        #   tile_cta_bar     (2-CTA only):    1
        #   tmem_dealloc     (2-CTA only):    32  (peer's epilog warp = 32 lanes)
        if warp_idx == EPILOG_WARP_IDS[0]:
            with cute.arch.elect_one():
                for i in range(NUM_SMEM_BUFFERS):
                    cute.arch.mbarrier_init(ab_empty_mbar + i, 1)
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

        # Init visibility: `mbarrier_init_fence` orders the inits CTA-locally.
        # In 2-CTA mode the cluster-scope proxy fence + cluster_arrive/wait
        # additionally publish them to peer CTA. In 1-CTA mode no extra
        # sync is needed here — the post-alloc CTA-wide barrier below
        # makes both the inits and the alloc visible to all warps.
        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(NUM_CTAS == 2):
            cute.arch.fence_proxy(kind="async.shared", space="cluster")
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # ----- TMEM allocation (epilogue warps own it) ---------------------
        if warp_idx == EPILOG_WARP_IDS[0]:
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
        # warp 0. Also doubles as the post-init sync in 1-CTA mode.
        cute.arch.barrier(barrier_id=0, number_of_threads=THREADS_PER_CTA)

        # =================================================================
        # Common prologue: TMA partitioning, tensormap manager,
        # MMA fragments, multicast masks, epilogue copy partitioning.
        # Mirrors the cute/gemm reference but our group metadata comes from
        # `split_sizes / strides_abc / ptrs_abc` instead of a pre-built
        # per-group problem_sizes_mnkl.
        # =================================================================
        # SMEM tensors for A, B, C — using the FULL mma_tiler layouts.
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # Tile partitioning. Tensors are 3D with trivial batch L=1 —
        # matches the cute/gemm reference which works on the realistic
        # 8-group failing shape.
        gA = cute.local_tile(
            mA,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        gB = cute.local_tile(
            mB,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        gC = cute.local_tile(
            mC,
            cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None, None),
        )

        # Map block_in_cluster_coord_vmnk for cluster mcast masks.
        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cluster_cta_rank
        )

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)

        # TMA partitioning.
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

        # MMA fragments. partition_shape_C takes the
        # FULL (M, N) — tiled_mma already accounts for thr_id splitting and
        # atom-along-M tiling.
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((BLOCK_SIZE_M, BLOCK_SIZE_N))
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, NUM_TMEM_BUFFERS)
        )

        # Compute multicast masks.
        # `ab_empty_mcast_mask`: leader's `tcgen05.commit` for smem_empty
        # broadcasts to both CTAs via local | peer mcast.
        # `acc_full_mcast_mask`: leader's `tcgen05.commit` for tmem_full
        # broadcasts to both CTAs via cluster-image mask (mode=0 over v).
        # Without these, peer CTA's bars never receive arrives → hangs.
        # ----- Multicast masks for 2-CTA cluster --------------------------
        # `cluster_layout_vmnk` has 4 axes: V (cluster CTA, 2 for 2-CTA),
        # M, N, K. `create_tma_multicast_mask(layout, coord, mcast_mode=k)`
        # builds a bitmask of CTA ranks that share the SAME slice along
        # mode `k` of the layout, given `coord` along the other modes.
        # The mask is consumed by TMA `multicast::mask` for the load that
        # broadcasts to all CTAs in that slice, AND by mbarrier-arrive to
        # signal completion to all CTAs in that slice.
        #
        # For the standard 2-CTA grouped GEMM:
        #   * `mcast_mode=2` (M-mode): A is broadcast along M — both CTAs
        #     of the cluster share the same N-tile and need the same A.
        #     The mask covers the A cluster-cohort = both CTAs (current +
        #     peer with V flipped). `a_full_mcast_mask` from the local
        #     coord covers the local-side arrives; `a_full_mcast_mask_peer`
        #     covers the peer-side arrives — combined for ab_empty.
        #   * `mcast_mode=1` (N-mode): B is broadcast along N. Same logic
        #     yields `b_full_mcast_mask` and `b_full_mcast_mask_peer`.
        #   * `ab_empty_mcast_mask = a_full | b_full | a_peer | b_peer`:
        #     the union covers all CTAs that need to be arrived-on for
        #     the empty bar (after a peer's TMA load consumed an A or B
        #     SMEM slot, both CTAs must be told the slot is now empty).
        #   * `acc_full_mcast_mask = make_layout_image_mask(layout, coord,
        #     mode=0)` — image of `coord` along the V (cluster CTA) mode:
        #     the set of CTAs that share the same M, N, K with `coord` —
        #     i.e. the peer CTA. Used by `tcgen05.commit + acc_full_mcast`
        #     to broadcast the per-tile MMA-done signal to BOTH CTAs of
        #     the cluster from a single arrive.
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
            # Peer-position coord: flip the v (cluster) bit so the peer-side
            # mask covers the OTHER CTA of the same M / N / K slice.
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

        # TMA load bytes per tile.
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

        # Tensormap manager + per-CTA workspace pointers.
        grid_dim = cute.arch.grid_dim()
        tm_workspace_idx = (
            bid[2] * grid_dim[1] * grid_dim[0] + bid[1] * grid_dim[0] + bid[0]
        )
        # SMEM mode: matches the cute/gemm reference + TLX/Triton behavior.
        # Per-group descriptor patches happen in SMEM, then
        # `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.
        # release.gpu.sync.aligned` atomically copies SMEM→GMEM and fences.
        # This dodges the rank-5 partial-patch IMA on Blackwell when
        # globalDim[1] (m_size) is patched < 8 elements for the LAST group.
        tensormap_manager = utils.TensorMapManager(
            utils.TensorMapUpdateMode.SMEM,
            128,  # bytes_per_tensormap
        )
        tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tm_workspace_idx, 0, None)].iterator
        )
        tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tm_workspace_idx, 1, None)].iterator
        )
        tensormap_c_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tm_workspace_idx, 2, None)].iterator
        )
        # SMEM staging: 3 × 128B contiguous buffer, A/B/C at offsets 0/16/32
        # Int64 (matches the cute/gemm reference layout).
        tensormap_buffer_ptr = storage.tensormap_buffer.data_ptr()
        tensormap_a_smem_ptr = tensormap_buffer_ptr
        tensormap_b_smem_ptr = tensormap_buffer_ptr + 16  # +128 bytes
        tensormap_c_smem_ptr = tensormap_buffer_ptr + 32  # +256 bytes
        # In SMEM mode, init the SMEM staging from the atom (not GMEM).
        # The `update_tensormap` SMEM path patches SMEM and `cp_fence_tma_
        # desc_release` publishes SMEM→GMEM atomically, so GMEM never needs
        # a separate init.
        tensormap_a_init_ptr = tensormap_a_smem_ptr
        tensormap_b_init_ptr = tensormap_b_smem_ptr
        tensormap_c_init_ptr = tensormap_c_smem_ptr

        # Initialize tensormaps from atoms (TMA warp does A/B; epilog does C).
        # In SMEM mode, init the SMEM staging (tensormap_*_init_ptr =
        # SMEM ptr); the GMEM tensormap is populated by the first
        # `cp_fence_tma_desc_release` from update_tensormap.
        #
        # After init, clear bit 21 of `desc_u64[1]` in SMEM to disable the
        # Blackwell TMA OOB-past-allocation prefetch path that otherwise
        # IMAs when the LAST group's patched `globalDim` is small. The
        # flags word at offset 8 is not rewritten by `update_tma_descriptor`
        # so a single init-time clear is sufficient; subsequent
        # `cp_fence_tma_desc_release` publishes the cleared bit to GMEM.
        if warp_idx == TMA_WARP_ID:
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_a, tensormap_a_init_ptr, TMA_WARP_ID
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_b, tensormap_b_init_ptr, TMA_WARP_ID
            )
            _clear_tma_oob_prefetch_bit(tensormap_a_smem_ptr)
            _clear_tma_oob_prefetch_bit(tensormap_b_smem_ptr)
        if warp_idx == EPILOG_WARP_IDS[0]:
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_c, tensormap_c_init_ptr, EPILOG_WARP_IDS[0]
            )
            _clear_tma_oob_prefetch_bit(tensormap_c_smem_ptr)
        tensormap_manager.fence_tensormap_initialization()

        # ----- Per-warpgroup register redistribution (mirrors TLX) ---------
        # TLX uses `num_regs=48` on producer/MMA tasks via `tlx.async_task`,
        # which compiles to PTX `setmaxnreg.dealloc`. CuTeDSL has the same
        # primitive (`setmaxregister_increase/decrease`) but it must be issued
        # at warpgroup granularity — all 128 threads of a warpgroup hit
        # the SAME `setmaxnreg.aligned` instruction with the SAME imm
        # count. Layout: WG0 (warps 0-3) = EPILOG (heavy register use,
        # holds accumulator + reg-staged store pipeline); WG1 (warps 4-7)
        # = MMA + TMA + 2 IDLE (light use). Values mirror TLX's
        # producer/MMA hint of 48 regs/thread.
        if warp_idx < 4:
            cute.arch.setmaxregister_increase(216)
        else:
            cute.arch.setmaxregister_decrease(40)

        # =================================================================
        # Warp-group dispatch (TLX `with tlx.async_task(...)` blocks).
        # =================================================================
        if warp_idx == TMA_WARP_ID:
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
                a_base_ptr=a_base_ptr,
                b_base_ptr=b_base_ptr,
                a_s0=a_s0,
                a_s1=a_s1,
                b_s0=b_s0,
                b_s1=b_s1,
                elem_size_bytes_a=elem_size_bytes_a,
                elem_size_bytes_b=elem_size_bytes_b,
                use_activation_buffer=use_activation_buffer,
                activation_buffer_size_bytes=activation_buffer_size_bytes,
                a_offset_tensor=a_offset_tensor,
                b_offset_tensor=b_offset_tensor,
                # Standalone callers don't have a gather WG; the
                # per-M-tile wait is disabled and the address/count are
                # ignored.
                a_buff_counter_addr_i64=cutlass.Int64(0),
                wait_per_m_tile=False,
                NUM_SUB_PER_GEMM_M=0,
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
            )
        elif warp_idx == MMA_WARP_ID:
            # Retrieve TMEM ptr and build accumulator tensor.
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
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
        elif warp_idx in EPILOG_WARP_IDS:
            # Build the TMEM->reg, reg->SMEM, SMEM->GMEM partitioned tensors.
            # Mirrors reference epilog_tmem_copy_and_partition.
            #
            # NUM_MMA_ATOMS_M: number of MMA atoms along M within one tile.
            # >1 only for 1cta2mma (BLOCK_M=256, atom_M=128 → MMA_M=2).
            # Other configs have MMA_M=1.
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            # Reference partitions (atom 0,0) used for tiled_copy / shapes.
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
                c_base_ptr=c_base_ptr,
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
            )
            # ---- TMEM dealloc handshake -----------------------------------
            # Required to avoid CUDA_ERROR_TENSOR_MEMORY_LEAK (721). Only
            # epilog warp 0 calls relinquish + dealloc; all 4 epilog warps
            # sync via barrier_id=_BAR_EPILOG_SYNC so the dealloc happens
            # after every epilog warp finished using TMEM.
            #
            # 2-CTA: each CTA's epilog warp 0 issues a remote arrive on the
            # peer CTA's `tmem_dealloc_mbar` and waits for the peer to ack
            # before calling `dealloc_tmem(is_two_cta=True)`. arrive_count=2.
            if warp_idx == EPILOG_WARP_IDS[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=(NUM_CTAS == 2))
            cute.arch.barrier(
                barrier_id=_BAR_EPILOG_SYNC,
                number_of_threads=32 * len(EPILOG_WARP_IDS),
            )
            if warp_idx == EPILOG_WARP_IDS[0]:
                if cutlass.const_expr(NUM_CTAS == 2):
                    # All 32 lanes of epilog warp 0 issue a remote arrive on
                    # the peer CTA's `tmem_dealloc_mbar` (arrive_count=32).
                    # Then wait local. Standard 2-CTA handshake from the
                    # cute/gemm reference.
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
        else:  # idle warp group (warp_id ∈ IDLE_WARP_IDS)
            self._idle_body()

        if cutlass.const_expr(NUM_CTAS == 2):
            # Keep every active thread alive until peer DSMEM accesses complete.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    @cute.jit
    def _tma_per_tile_wait(
        self,
        tile_m_idx: cutlass.Int32,
        tile_m_start_per_group: cutlass.Int32,
        a_buff_counter_addr_i64: cutlass.Int64,
        NUM_SUB_PER_GEMM_M: cutlass.Constexpr[int],
    ):
        """Pre-TMA per-tile hook. No-op default — subclasses override
        when they need a cross-CTA dependency (e.g. the dist DISPATCH
        gather→TMA pipeline). Called only when the caller passes
        `wait_per_m_tile=True` to `_tma_producer_body`; the base
        `_tma_producer_body` constexpr-elides the call otherwise.
        """
        pass

    @cute.jit
    def _tma_per_col_tile_wait(
        self,
        counter_addr: cutlass.Int64,
        group: cutlass.Int32,
        tile_n: cutlass.Int32,
        n_size: cutlass.Int32,
        k_size: cutlass.Int32,
    ):
        """Pre-TMA hook for subclasses with WGRAD column dependencies."""
        pass

    @cute.jit
    def _tma_k_segment_wait(
        self,
        tile_m_idx: cutlass.Int32,
        tile_m_start_per_group: cutlass.Int32,
        a_buff_counter_addr_i64: cutlass.Int64,
        kk: cutlass.Int32,
        num_k_tiles: cutlass.Int32,
        k_size: cutlass.Int32,
        NUM_SUB_PER_GEMM_M: cutlass.Constexpr[int],
    ) -> cutlass.Int32:
        """`_tma_per_tile_wait` for `GATHER_K_SEGMENTS > 1`: waits until
        k-tile `kk` is ready and returns the count of leading ready
        k-tiles."""
        return num_k_tiles

    @cute.jit
    def _tma_producer_body(  # noqa: C901
        self,
        # ---- TMA copy infrastructure -----------------------------------------
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        tAgA: cute.Tensor,  # partitioned A: ((atom_v, rest_v), RestM, RestK, RestL)
        tBgB: cute.Tensor,  # partitioned B: ((atom_v, rest_v), RestN, RestK, RestL)
        tAsA: cute.Tensor,  # partitioned A in SMEM: ((atom_v, rest_v), STAGE)
        tBsB: cute.Tensor,  # partitioned B in SMEM: ((atom_v, rest_v), STAGE)
        a_full_mcast_mask,  # cluster mcast mask for A (None for 1-CTA non-mcast)
        b_full_mcast_mask,
        num_tma_load_bytes: cutlass.Constexpr[int],
        # ---- Tensormap update infrastructure ---------------------------------
        # `tensormap_manager` is a `utils.TensorMapManager`; `tensormap_a/b_ptr`
        # are the per-CTA workspace pointers; `tensormap_a/b_smem_ptr` are SMEM
        # staging buffers (None when GMEM update mode is used).
        tensormap_manager,
        tensormap_a_ptr,
        tensormap_b_ptr,
        tensormap_a_smem_ptr,
        tensormap_b_smem_ptr,
        # ---- Sync state ------------------------------------------------------
        ab_full_mbar: cute.Pointer,
        ab_empty_mbar: cute.Pointer,
        tile_producer_mbar: cute.Pointer,
        tile_consumer_mbar: cute.Pointer,
        tile_cta_bar_mbar,  # may be None in 1-CTA mode
        tile_id_smem_ptr: cute.Pointer,
        # ---- Scheduler -------------------------------------------------------
        counter_ptr: cute.Pointer,
        split_sizes: cute.Tensor,
        cluster_cta_rank: cutlass.Int32,
        # ---- Per-tensor scalar metadata (replaces strides_abc/ptrs_abc) -----
        # Base pointers (int64), and (s0, s1) strides matching the (m, k) /
        # (n, k) view used by `_make_tensor_for_tensormap_update`. Per-group
        # base ptr is computed inside the kernel via TLX-style accumulators
        # (start_am, start_bn, etc.) — no CPU sync, no per-launch device
        # tensor.
        a_base_ptr: cutlass.Int64,
        b_base_ptr: cutlass.Int64,
        a_s0: cutlass.Int32,
        a_s1: cutlass.Int32,
        b_s0: cutlass.Int32,
        b_s1: cutlass.Int32,
        elem_size_bytes_a: cutlass.Constexpr[int],
        elem_size_bytes_b: cutlass.Constexpr[int],
        # ---- Activation-buffer (TLX-style; see kernel docstring) ------------
        use_activation_buffer: cutlass.Constexpr[bool],
        activation_buffer_size_bytes: cutlass.Int64,
        a_offset_tensor: cute.Tensor,
        b_offset_tensor: cute.Tensor,
        # ---- Per-M-tile gather→TMA wait (dist DISPATCH; mirrors TLX) -------
        # When `wait_per_m_tile` is True, busy-wait per (group, tile_m_idx)
        # before issuing the k-loop's TMA loads on `a_buff_counter[
        # tile_m_start_per_group + tile_m_idx] >= NUM_SUB_PER_GEMM_M`.
        # Standalone callers pass `wait_per_m_tile=False` and the address
        # / NUM_SUB are unused. Mirrors TLX line 938-963.
        a_buff_counter_addr_i64: cutlass.Int64,
        wait_per_m_tile: cutlass.Constexpr[bool],
        NUM_SUB_PER_GEMM_M: cutlass.Constexpr[int],
        # ---- Problem dimensions ---------------------------------------------
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        # ---- Constexpr flags ------------------------------------------------
        problem_type: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        WORLD_SIZE: cutlass.Constexpr[int],
        NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
        NUM_TILE_BUFFERS: cutlass.Constexpr[int],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        NUM_CTAS: cutlass.Constexpr[int],
        secondary_operands=None,
        a_coord_layout=None,
    ):
        """TMA producer task. Ports the TLX producer body in control flow,
        with intentional 2-CTA cooperative-tile semantics (see deltas below).

        Mirrors TLX:
            * Persistent atomic-counter scheduler (CTA 0 fetches; CTA 1
              receives via DSMEM).
            * Per-group: tensormap update once at group entry (or once per
              CTA with `SINGLE_A_TENSORMAP`), then tile-by-tile k-loop until
              tile_idx leaves the group's range.
            * Flow control: `producer_wait_tile_released` placed AFTER TMA
              issue (not before) to avoid a circular deadlock.
            * SENTINEL drain at end so MMA/epilogue consumers terminate.

        Intentional 2-CTA delta vs TLX (cooperative-tile semantics):
          * TLX pairs separate tiles: each CTA owns half of N
            (`BLOCK_SIZE_N // NUM_CTAS`) and loads its own B half via
            per-CTA TMA.
          * CuTeDSL cooperates on a single cluster-wide tile: both CTAs
            share the FULL tile. The `tiled_mma` with `cta_group=TWO` and
            TMA atoms with multicast handle the 2-CTA distribution
            automatically. We do NOT halve N here.

        The only structural delta from TLX is that `_make_tma_descriptors`
        becomes `tensormap_manager.update_tensormap` — algorithmically the
        same per-group descriptor refresh, expressed in CuTeDSL idiom.
        """
        # ---- Loop-carried state -----------------------------------------
        # Pre-declared because CuTeDSL's @cute.jit scoping requires names
        # read in a conditional region (or carried across iterations) to
        # be defined in the enclosing scope. Per-iteration temporaries
        # (m_size/n_size/k_size, num_*_tiles, tile_end, tile_m_idx/...,
        # cur_tile_idx, tile_buf, prev_accum_cnt_smem) are assigned before
        # use within each iteration and don't need pre-init.
        accum_cnt_smem = cutlass.Int32(0)
        accum_cnt_out = cutlass.Int32(0)

        # Activation-buffer offset (TLX-style).
        # When enabled, the kernel loads two int64 byte offsets from device
        # tensors and adjusts the A/B base pointers — entirely on device, no
        # CPU sync. Used in FP8 / quant pipelines where activations live at
        # offset locations within a shared `activation_buffer`.
        a_base_eff = a_base_ptr
        b_base_eff = b_base_ptr
        a_offset = cutlass.Int64(0)
        b_offset = cutlass.Int64(0)
        if cutlass.const_expr(use_activation_buffer):
            a_offset = cutlass.Int64(a_offset_tensor[0])
            b_offset = cutlass.Int64(b_offset_tensor[0])
            a_base_eff = a_base_ptr + a_offset
            b_base_eff = b_base_ptr + b_offset
            if cutlass.const_expr(problem_type == _WGRAD):
                if (cute.arch.block_idx()[0] == 0) & (cute.arch.lane_idx() == 0):
                    _require_valid_activation_buffer_range(
                        byte_offset=a_offset,
                        byte_extent=_packed_group_rows_byte_extent(
                            split_sizes,
                            a_s1,
                            elem_size_bytes_a,
                            G,
                        ),
                        activation_buffer_size_bytes=activation_buffer_size_bytes,
                        warp_scoped_diagnostic=True,
                    )
                    _require_valid_activation_buffer_range(
                        byte_offset=b_offset,
                        byte_extent=_packed_group_rows_byte_extent(
                            split_sizes,
                            b_s1,
                            elem_size_bytes_b,
                            G,
                        ),
                        activation_buffer_size_bytes=activation_buffer_size_bytes,
                        warp_scoped_diagnostic=True,
                    )

        if cutlass.const_expr(self.STATIC_SCHEDULER):
            scheduler = StaticTileScheduler.create_producer(NUM_CTAS)
        else:
            scheduler = DynamicTileScheduler.create_producer(
                counter_ptr,
                tile_cta_bar_mbar,
                tile_id_smem_ptr,
                tile_consumer_mbar,
                tile_producer_mbar,
                cluster_cta_rank,
                NUM_CTAS,
                NUM_TILE_BUFFERS,
                self.NUM_TILE_CTA_BARS,
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
            SKIP_EMPTY_K=problem_type == _WGRAD and self.output_accum,
        )
        work = visitor.get_work(tile_idx)

        # Per-group A descriptor patch (and the drain before it) unless one
        # descriptor covers every group.
        patch_a_per_group = not (self.host_a_tensormap or self.SINGLE_A_TENSORMAP)
        if cutlass.const_expr(self.SINGLE_A_TENSORMAP):
            # One A descriptor over every group's rows, from the activation
            # offset (host_a_tensormap needs padded rows and no activation
            # buffer). Its extent is the exact row total, so TMA zero-fills past
            # the last group and never reads past the buffer; at least 1 row so
            # an empty rank never encodes a zero-size descriptor. The warp sums
            # the group sizes lane-parallel, loads issued back to back.
            total_rows = cutlass.Int32(0)
            lane = cute.arch.lane_idx()
            for chunk in cutlass.range_constexpr((G + 31) // 32):
                if lane + chunk * 32 < G:
                    total_rows += cutlass.Int32(split_sizes[lane + chunk * 32])
            for shift in cutlass.range_constexpr(5):
                total_rows += cute.arch.shuffle_sync_bfly(total_rows, offset=1 << shift)
            tensormap_manager.update_tensormap(
                (
                    _make_tensor_for_tensormap_update(
                        a_base_eff,
                        self.a_dtype,
                        (cutlass.max(total_rows, cutlass.Int32(1)), N, K),
                        a_s0,
                        a_s1,
                        tensor_index=0,
                        problem_type=problem_type,
                    ),
                ),
                (tma_atom_a,),
                (tensormap_a_ptr,),
                TMA_WARP_ID,
                (tensormap_a_smem_ptr,),
            )
            tensormap_manager.fence_tensormap_update(tensormap_a_ptr)

        while work.is_valid_tile:
            g = work.group_idx
            m_size = work.m_size
            n_size = work.n_size
            k_size = work.k_size
            num_k_tiles = work.num_k_tiles
            prev_accum_cnt_smem = accum_cnt_smem
            group_tAgA = tAgA
            if cutlass.const_expr(self.SINGLE_A_TENSORMAP):
                # Shift the A coordinates to this group's first row. domain_offset
                # needs the (M, K, L) coordinate profile, which the partitioned
                # tAgA.layout has lost (same idiom as gemm_warp_roles.py).
                a_coords = cute.make_tensor(tAgA.iterator, a_coord_layout)
                group_a_coords = cute.domain_offset((work.split_prefix, 0, 0), a_coords)
                group_tAgA = cute.make_tensor(group_a_coords.iterator, tAgA.layout)

            if work.is_valid_tile:
                # ---- Per-group tensormap update --------------------------
                if cutlass.const_expr(problem_type == _FPROP):
                    if cutlass.const_expr(self.SWAP_AB):
                        a_off_bytes = (
                            cutlass.Int64(g)
                            * cutlass.Int64(m_size)
                            * cutlass.Int64(a_s0)
                            * cutlass.Int64(elem_size_bytes_a)
                        )
                        b_off_bytes = (
                            cutlass.Int64(work.split_prefix)
                            * cutlass.Int64(b_s0)
                            * cutlass.Int64(elem_size_bytes_b)
                        )
                    else:
                        a_off_bytes = (
                            cutlass.Int64(work.split_prefix)
                            * cutlass.Int64(a_s0)
                            * cutlass.Int64(elem_size_bytes_a)
                        )
                        b_off_bytes = (
                            cutlass.Int64(g)
                            * cutlass.Int64(n_size)
                            * cutlass.Int64(b_s0)
                            * cutlass.Int64(elem_size_bytes_b)
                        )
                elif cutlass.const_expr(problem_type == _DGRAD):
                    a_off_bytes = (
                        cutlass.Int64(work.split_prefix)
                        * cutlass.Int64(a_s0)
                        * cutlass.Int64(elem_size_bytes_a)
                    )
                    b_off_bytes = (
                        cutlass.Int64(g)
                        * cutlass.Int64(k_size)
                        * cutlass.Int64(b_s1)
                        * cutlass.Int64(elem_size_bytes_b)
                    )
                else:
                    a_off_bytes = (
                        cutlass.Int64(work.split_prefix)
                        * cutlass.Int64(a_s1)
                        * cutlass.Int64(elem_size_bytes_a)
                    )
                    b_off_bytes = (
                        cutlass.Int64(work.split_prefix)
                        * cutlass.Int64(b_s1)
                        * cutlass.Int64(elem_size_bytes_b)
                    )
                if cutlass.const_expr(use_activation_buffer and problem_type != _WGRAD):
                    if (cute.arch.block_idx()[0] == 0) & (cute.arch.lane_idx() == 0):
                        _require_valid_activation_buffer_range(
                            byte_offset=a_offset + a_off_bytes,
                            byte_extent=cutlass.Int64(k_size)
                            * cutlass.Int64(a_s1)
                            * cutlass.Int64(elem_size_bytes_a),
                            activation_buffer_size_bytes=activation_buffer_size_bytes,
                            warp_scoped_diagnostic=True,
                        )
                        _require_valid_activation_buffer_range(
                            byte_offset=b_offset + b_off_bytes,
                            byte_extent=cutlass.Int64(k_size)
                            * cutlass.Int64(b_s1)
                            * cutlass.Int64(elem_size_bytes_b),
                            activation_buffer_size_bytes=activation_buffer_size_bytes,
                            warp_scoped_diagnostic=True,
                        )
                if cutlass.const_expr(patch_a_per_group):
                    a_base_g = a_base_eff + a_off_bytes
                    real_tensor_a = _make_tensor_for_tensormap_update(
                        a_base_g,
                        self.a_dtype,
                        (m_size, n_size, k_size),
                        a_s0,
                        a_s1,
                        tensor_index=0,
                        problem_type=problem_type,
                    )
                if cutlass.const_expr(patch_a_per_group and self.host_b_tensormap):
                    tensormap_manager.update_tensormap(
                        (real_tensor_a,),
                        (tma_atom_a,),
                        (tensormap_a_ptr,),
                        TMA_WARP_ID,
                        (tensormap_a_smem_ptr,),
                    )
                elif cutlass.const_expr(not self.host_b_tensormap):
                    b_base_g = b_base_eff + b_off_bytes
                    real_tensor_b = _make_tensor_for_tensormap_update(
                        b_base_g,
                        self.b_dtype,
                        (m_size, n_size, k_size),
                        b_s0,
                        b_s1,
                        tensor_index=1,
                        problem_type=problem_type,
                    )
                    if cutlass.const_expr(self.host_a_tensormap):
                        tensormap_manager.update_tensormap(
                            (real_tensor_b,),
                            (tma_atom_b,),
                            (tensormap_b_ptr,),
                            TMA_WARP_ID,
                            (tensormap_b_smem_ptr,),
                        )
                    else:
                        tensormap_manager.update_tensormap(
                            (real_tensor_a, real_tensor_b),
                            (tma_atom_a, tma_atom_b),
                            (tensormap_a_ptr, tensormap_b_ptr),
                            TMA_WARP_ID,
                            (tensormap_a_smem_ptr, tensormap_b_smem_ptr),
                        )

                # Acquire fence — published descriptor must be visible to the
                # TMA proxy before any TMA load below uses it. ONCE per group
                # (the descriptor doesn't change between tiles within a group),
                # not per tile — the per-tile placement was wasteful.
                if cutlass.const_expr(patch_a_per_group):
                    tensormap_manager.fence_tensormap_update(tensormap_a_ptr)
                if cutlass.const_expr(not self.host_b_tensormap):
                    tensormap_manager.fence_tensormap_update(tensormap_b_ptr)

                # The weights are the B operand only in a non-swapped forward; it always
                # loads B via the host tensormap, so the other copies stay as is.
                b_policy = None
                if cutlass.const_expr(problem_type == _FPROP and not self.SWAP_AB):
                    b_policy = _weight_l2_policy(m_size, self.BLOCK_SIZE_M)

                # Loop while current tile_idx falls within this group.
                while work.is_valid_tile and work.group_idx == g:
                    scheduler.producer_publish_tile(accum_cnt_out)

                    # ---- TMA loads for this tile ------------------------
                    tile_m_idx = work.tile_m_idx
                    tile_n_idx = work.tile_n_idx

                    # Optional pre-TMA hook for subclass-specific
                    # cross-CTA dependencies (e.g. dist DISPATCH gather→
                    # TMA per-M-tile wait). Default impl on this base
                    # class is a no-op; `DistGroupedGemmKernel` overrides
                    # it to busy-wait on the per-M-tile gather counter
                    # `a_buff_counter[tile_m_start_per_group + tile_m_idx]
                    # >= NUM_SUB_PER_GEMM_M`. Standalone callers pass
                    # `wait_per_m_tile=False`; the constexpr eliminates
                    # the call entirely.
                    # DISPATCH (1-CTA): the B (weight) operand depends only on
                    # the group tensormap, not on gathered tokens, so issue up
                    # to a ring's worth of B loads BEFORE the per-M-tile
                    # gather wait; the wait then hides under weight streaming
                    # and only the A copies are back-filled after it. One
                    # expect_tx per stage as usual - a stage completes when
                    # both its copies land, so consumer ordering is unchanged.
                    # (SWAP_AB inverts operand roles - the gathered tokens
                    # become B - so the prologue only applies when B really
                    # is the gather-independent weight operand.)
                    weights_ahead = cutlass.Int32(0)
                    segmented_gather = (
                        wait_per_m_tile
                        and problem_type != _WGRAD
                        and self.GATHER_K_SEGMENTS > 1
                    )
                    ready_k = cutlass.Int32(0)
                    wait_tile = tile_n_idx if self.SWAP_AB else tile_m_idx
                    wait_prefix = (
                        work.n_tile_prefix if self.SWAP_AB else work.m_tile_prefix
                    )
                    # Prefill configurations use larger M tiles; even at this
                    # tile size, the full-tile boundary regresses when armed
                    # ahead, so restrict the overlap to strict decode shapes.
                    if cutlass.const_expr(
                        wait_per_m_tile
                        and problem_type != _WGRAD
                        and NUM_CTAS == 1
                        and not self.SWAP_AB
                        and self.BLOCK_SIZE_M <= 64
                    ):
                        if m_size < cutlass.Int32(self.BLOCK_SIZE_M):
                            weights_ahead = cutlass.min(
                                cutlass.Int32(NUM_SMEM_BUFFERS), num_k_tiles
                            )
                            for kk in cutlass.range(weights_ahead, unroll=1):
                                buf, phase = _get_bufidx_phase(
                                    accum_cnt_smem + kk, NUM_SMEM_BUFFERS
                                )
                                cute.arch.mbarrier_wait(
                                    ab_empty_mbar + buf,
                                    phase ^ 1,
                                )
                                with cute.arch.elect_one():
                                    cute.arch.mbarrier_arrive_and_expect_tx(
                                        ab_full_mbar + buf, num_tma_load_bytes
                                    )
                                if cutlass.const_expr(self.host_b_tensormap):
                                    cute.copy(
                                        tma_atom_b,
                                        tBgB[(None, tile_n_idx, kk, g)],
                                        tBsB[(None, buf)],
                                        tma_bar_ptr=ab_full_mbar + buf,
                                        mcast_mask=b_full_mcast_mask,
                                        cache_policy=b_policy,
                                    )
                                else:
                                    b_desc_ptr = tensormap_manager.get_tensormap_ptr(
                                        tensormap_b_ptr,
                                        cute.AddressSpace.generic,
                                    )
                                    cute.copy(
                                        tma_atom_b,
                                        tBgB[(None, tile_n_idx, kk, 0)],
                                        tBsB[(None, buf)],
                                        tma_bar_ptr=ab_full_mbar + buf,
                                        mcast_mask=b_full_mcast_mask,
                                        tma_desc_ptr=b_desc_ptr,
                                    )

                    if cutlass.const_expr(wait_per_m_tile):
                        if cutlass.const_expr(problem_type == _WGRAD):
                            self._tma_per_col_tile_wait(
                                a_buff_counter_addr_i64, g, tile_n_idx, n_size, k_size
                            )
                        elif cutlass.const_expr(segmented_gather):
                            ready_k = self._tma_k_segment_wait(
                                wait_tile,
                                wait_prefix,
                                a_buff_counter_addr_i64,
                                weights_ahead - cutlass.Int32(1),
                                num_k_tiles,
                                k_size,
                                NUM_SUB_PER_GEMM_M,
                            )
                        else:
                            self._tma_per_tile_wait(
                                wait_tile,
                                wait_prefix,
                                a_buff_counter_addr_i64,
                                NUM_SUB_PER_GEMM_M,
                            )

                    # Back-fill activations for stages pre-armed above.
                    for kk in cutlass.range(weights_ahead, unroll=1):
                        buf, _ = _get_bufidx_phase(accum_cnt_smem, NUM_SMEM_BUFFERS)
                        if cutlass.const_expr(self.host_a_tensormap):
                            cute.copy(
                                tma_atom_a,
                                tAgA[(None, work.m_tile_prefix + tile_m_idx, kk, 0)],
                                tAsA[(None, buf)],
                                tma_bar_ptr=ab_full_mbar + buf,
                                mcast_mask=a_full_mcast_mask,
                            )
                        else:
                            a_desc_ptr = tensormap_manager.get_tensormap_ptr(
                                tensormap_a_ptr, cute.AddressSpace.generic
                            )
                            cute.copy(
                                tma_atom_a,
                                group_tAgA[(None, tile_m_idx, kk, 0)],
                                tAsA[(None, buf)],
                                tma_bar_ptr=ab_full_mbar + buf,
                                mcast_mask=a_full_mcast_mask,
                                tma_desc_ptr=a_desc_ptr,
                            )
                        accum_cnt_smem += cutlass.Int32(1)

                    # Issue the remaining activation and weight tiles.
                    for kk in cutlass.range(weights_ahead, num_k_tiles, unroll=1):
                        if cutlass.const_expr(segmented_gather):
                            if kk >= ready_k:
                                ready_k = self._tma_k_segment_wait(
                                    wait_tile,
                                    wait_prefix,
                                    a_buff_counter_addr_i64,
                                    kk,
                                    num_k_tiles,
                                    k_size,
                                    NUM_SUB_PER_GEMM_M,
                                )
                        buf, phase = _get_bufidx_phase(accum_cnt_smem, NUM_SMEM_BUFFERS)
                        cute.arch.mbarrier_wait(
                            ab_empty_mbar + buf,
                            phase ^ 1,
                        )
                        if cutlass.const_expr(NUM_CTAS == 2):
                            if cluster_cta_rank == 0:
                                with cute.arch.elect_one():
                                    cute.arch.mbarrier_arrive_and_expect_tx(
                                        ab_full_mbar + buf, num_tma_load_bytes
                                    )
                        else:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    ab_full_mbar + buf, num_tma_load_bytes
                                )

                        if cutlass.const_expr(self.host_a_tensormap):
                            cute.copy(
                                tma_atom_a,
                                tAgA[(None, work.m_tile_prefix + tile_m_idx, kk, 0)],
                                tAsA[(None, buf)],
                                tma_bar_ptr=ab_full_mbar + buf,
                                mcast_mask=a_full_mcast_mask,
                            )
                        else:
                            a_desc_ptr = tensormap_manager.get_tensormap_ptr(
                                tensormap_a_ptr, cute.AddressSpace.generic
                            )
                            cute.copy(
                                tma_atom_a,
                                group_tAgA[(None, tile_m_idx, kk, 0)],
                                tAsA[(None, buf)],
                                tma_bar_ptr=ab_full_mbar + buf,
                                mcast_mask=a_full_mcast_mask,
                                tma_desc_ptr=a_desc_ptr,
                            )
                        if cutlass.const_expr(self.host_b_tensormap):
                            cute.copy(
                                tma_atom_b,
                                tBgB[(None, tile_n_idx, kk, g)],
                                tBsB[(None, buf)],
                                tma_bar_ptr=ab_full_mbar + buf,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_policy,
                            )
                        else:
                            b_desc_ptr = tensormap_manager.get_tensormap_ptr(
                                tensormap_b_ptr, cute.AddressSpace.generic
                            )
                            cute.copy(
                                tma_atom_b,
                                tBgB[(None, tile_n_idx, kk, 0)],
                                tBsB[(None, buf)],
                                tma_bar_ptr=ab_full_mbar + buf,
                                mcast_mask=b_full_mcast_mask,
                                tma_desc_ptr=b_desc_ptr,
                            )
                        accum_cnt_smem += cutlass.Int32(1)

                    accum_cnt_out += cutlass.Int32(1)

                    # ---- Flow control AFTER TMA ------------------------
                    # Wait for epilogue to free the next tile_info slot.
                    # Placed after issue to avoid circular deadlock where CTA 0
                    # would block before issuing TMA, starving MMA/epilogue.
                    scheduler.producer_wait_tile_released(accum_cnt_out)
                    tile_idx = scheduler.advance_producer(accum_cnt_out)
                    work = visitor.get_work(tile_idx)

            # ---- Drain last TMA load ----------------------------------------
            # We must observe the final TMA load complete before the next
            # group patches the tensormap; otherwise the in-flight load
            # could pick up the new descriptor.
            # 2-CTA: leader is the one who armed the bar via
            # mbarrier_arrive_and_expect_tx, so only leader waits. Peer CTA
            # has no equivalent local arrive on this bar.
            # Skipped for a single A descriptor: nothing is patched between
            # groups (B uses the host descriptor).
            if cutlass.const_expr(not self.SINGLE_A_TENSORMAP):
                if accum_cnt_smem > prev_accum_cnt_smem:
                    if (NUM_CTAS == 1) or (cluster_cta_rank == 0):
                        buf, phase = _get_bufidx_phase(
                            accum_cnt_smem - cutlass.Int32(1), NUM_SMEM_BUFFERS
                        )
                        cute.arch.mbarrier_wait(ab_full_mbar + buf, phase)

        scheduler.producer_publish_termination(accum_cnt_out)

    @cute.jit
    def _mma_consumer_body(  # noqa: C901
        self,
        # ---- MMA infrastructure ----------------------------------------------
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,  # (MMA, MMA_M, MMA_K, STAGE)
        tCrB: cute.Tensor,  # (MMA, MMA_N, MMA_K, STAGE)
        tCtAcc_base: cute.Tensor,  # (MMA, MMA_M, MMA_N, ACC_STAGE) — TMEM accum
        # ---- Sync state ------------------------------------------------------
        ab_full_mbar: cute.Pointer,
        ab_empty_mbar: cute.Pointer,
        tmem_full_mbar: cute.Pointer,
        tmem_empty_mbar: cute.Pointer,
        tile_consumer_mbar: cute.Pointer,
        tile_id_smem_ptr: cute.Pointer,
        # ---- Multicast masks for tcgen05.commit ------------------------------
        ab_empty_mcast_mask,
        acc_full_mcast_mask,
        # ---- Scheduler / problem ---------------------------------------------
        split_sizes: cute.Tensor,
        pred_cta0: cutlass.Boolean,
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        # ---- Constexpr -------------------------------------------------------
        problem_type: cutlass.Constexpr[int],
        NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
        NUM_TMEM_BUFFERS: cutlass.Constexpr[int],
        NUM_TILE_BUFFERS: cutlass.Constexpr[int],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        NUM_CTAS: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        secondary_operands=None,
    ):
        """MMA consumer task. Ports the TLX MMA path + ``_mma_tile_inner``
        in control flow, with the same cooperative-tile delta as the TMA
        producer (the cluster cooperates on one tile rather than pairing
        two TLX-style tiles).

        Loops over groups via the tile-info consumer barrier. For each tile:
          * Wait tmem_empty for the chosen tmem buffer.
          * Per k-tile: wait smem_full; in 2-CTA mode, do the cta_bars
            handshake; issue MMA fragments of `cute.gemm` (== async_dot)
            sliced along M.
          * After all k-tiles for the tile: tcgen05.commit on tmem_full to
            signal epilogue (or, if num_k_tiles==0 in 2-CTA mode, manually
            arrive on tmem_full because no async_dot was issued).
        """
        # Pre-read first tile_idx from the producer.
        accum_cnt_tile = cutlass.Int32(0)
        accum_cnt_smem = cutlass.Int32(0)

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
            cutlass.Int32(0),
            1,
            self.SWAP_AB,
            DIAG_N_BANDS=self.DIAG_N_BANDS,
            SKIP_EMPTY_K=problem_type == _WGRAD and self.output_accum,
        )
        work = visitor.get_mma_work(tile_idx)

        while work.is_valid_tile:
            num_k_tiles = work.num_k_tiles
            tmem_buf, tmem_phase = _get_bufidx_phase(accum_cnt_tile, NUM_TMEM_BUFFERS)

            if cutlass.const_expr(self.SPLIT_ATOM_ACCUM):
                accum_cnt_smem = split_atom_mma_tile(
                    tiled_mma,
                    tCrA,
                    tCrB,
                    tCtAcc_base[(None, None, None, 0)],
                    ab_full_mbar,
                    ab_empty_mbar,
                    tmem_full_mbar,
                    tmem_empty_mbar,
                    ab_empty_mcast_mask,
                    acc_full_mcast_mask,
                    pred_cta0,
                    num_k_tiles,
                    tmem_phase,
                    accum_cnt_smem,
                    NUM_SMEM_BUFFERS,
                    NUM_CTAS,
                    self.cta_group,
                )
            else:
                accum_cnt_smem = native_mma_tile(
                    tiled_mma,
                    tCrA,
                    tCrB,
                    tCtAcc_base,
                    ab_full_mbar,
                    ab_empty_mbar,
                    tmem_full_mbar,
                    tmem_empty_mbar,
                    ab_empty_mcast_mask,
                    acc_full_mcast_mask,
                    pred_cta0,
                    num_k_tiles,
                    tmem_buf,
                    tmem_phase,
                    accum_cnt_smem,
                    NUM_SMEM_BUFFERS,
                    NUM_CTAS,
                    self.cta_group,
                )

            accum_cnt_tile += cutlass.Int32(1)
            tile_idx = scheduler.advance_consumer(accum_cnt_tile)
            work = visitor.get_mma_work(tile_idx)

    @cute.jit
    def _epilog_consumer_body(  # noqa: C901
        self,
        # ---- Thread index for re-partitioning per atom ----------------------
        tidx: cutlass.Int32,
        # ---- TMA-store / TMEM->reg copy infrastructure -----------------------
        tma_atom_c: cute.CopyAtom,
        tCtAcc_base: cute.Tensor,  # (MMA, MMA_M, MMA_N, ACC_STAGE)
        tCgC: cute.Tensor,  # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        sC: cute.Tensor,  # SMEM staging for C
        tTR_rAcc: cute.Tensor,
        tiled_copy_r2s,  # reg -> smem
        tRS_rC: cute.Tensor,
        tRS_sC: cute.Tensor,
        epi_tile: cute.Tile,
        # ---- Tensormap manager for per-group C descriptor refresh -----------
        tensormap_manager,
        tensormap_c_ptr,
        tensormap_c_smem_ptr,
        # ---- Sync state ------------------------------------------------------
        tmem_full_mbar: cute.Pointer,
        tmem_empty_mbar: cute.Pointer,
        tile_consumer_mbar: cute.Pointer,
        tile_producer_mbar: cute.Pointer,
        tile_id_smem_ptr: cute.Pointer,
        # ---- Per-tensor scalar base ptr + strides (replaces strides_abc/ptrs_abc) ---
        c_base_ptr: cutlass.Int64,
        c_s0: cutlass.Int32,
        c_s1: cutlass.Int32,
        elem_size_bytes_c: cutlass.Constexpr[int],
        # ---- Scheduler / problem ---------------------------------------------
        split_sizes: cute.Tensor,
        cluster_cta_rank: cutlass.Int32,
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        # ---- Constexpr -------------------------------------------------------
        problem_type: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        WORLD_SIZE: cutlass.Constexpr[int],
        NUM_TMEM_BUFFERS: cutlass.Constexpr[int],
        NUM_C_STAGES: cutlass.Constexpr[int],
        NUM_TILE_BUFFERS: cutlass.Constexpr[int],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        NUM_CTAS: cutlass.Constexpr[int],
        secondary_operands=None,
        swiglu_h2=None,
    ):
        """Epilogue consumer task. Ports the TLX epilogue + ``_epi_tile_inner``
        in control flow; the same cooperative-tile delta as the producer/MMA
        bodies applies (one cluster-wide tile instead of TLX's paired
        half-tiles).

        Per-tile flow:
          * preread tile_idx via tile_consumer_mbar.
          * If tile_idx is in current group: per-group, update tensormap C;
            then loop tiles in group:
              - wait tmem_full
              - tmem -> reg -> smem -> TMA store, in EPILOGUE_SUBTILE pieces
                (the previous TMA store is waited only when the C SMEM stage
                is about to be reused)
              - signal tmem_empty after the last TMEM load is fenced, before
                waiting for the final TMA store
              - signal tile_done to producer (so producer can reuse the
                tile-info slot)
          * On _TILE_SENTINEL: exit.
        Final: relinquish_tmem_alloc_permit, dealloc_tmem (with peer
        handshake in 2-CTA mode).

        """
        # State.
        accum_cnt_tile = cutlass.Int32(0)
        # SwiGLU in the TMEM drain is hidden behind a second TMEM buffer; with
        # one it stalls the next tile's MMA, so recompute h2 from the stored h1
        # after the accumulator release instead.
        swiglu_inline = swiglu_h2 is not None and NUM_TMEM_BUFFERS > 1
        swiglu_deferred = swiglu_h2 is not None and NUM_TMEM_BUFFERS == 1

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
            SKIP_EMPTY_K=problem_type == _WGRAD and self.output_accum,
        )
        work = visitor.get_work(tile_idx)

        while work.is_valid_tile:
            g = work.group_idx
            m_size = work.m_size
            n_size = work.n_size
            k_size = work.k_size
            num_k_tiles = work.num_k_tiles

            if work.is_valid_tile:
                # If a previous group's final TMA store is still reading the
                # shared C SMEM tile, drain it before patching/reusing the C
                # descriptor for this group.
                _epilog_wait_pending_tma_store()

                # ---- Per-group tensormap update for C -----------------------
                if cutlass.const_expr(problem_type == _WGRAD):
                    c_row_prefix = cutlass.Int64(g) * cutlass.Int64(m_size)
                    c_row_stride = c_s0
                else:
                    c_row_prefix = cutlass.Int64(work.split_prefix)
                    c_row_stride = c_s1 if self.SWAP_AB else c_s0
                c_base_g = c_base_ptr + (
                    c_row_prefix
                    * cutlass.Int64(c_row_stride)
                    * cutlass.Int64(elem_size_bytes_c)
                )
                real_tensor_c = _make_tensor_for_tensormap_update(
                    c_base_g,
                    sC.element_type,
                    (m_size, n_size, k_size),
                    c_s0,
                    c_s1,
                    tensor_index=2,
                    problem_type=problem_type,
                )
                if cutlass.const_expr(swiglu_h2 is not None):
                    real_tensor_c = interleaved_fc13_view(real_tensor_c, 1)
                tensormap_manager.update_tensormap(
                    (real_tensor_c,),
                    (tma_atom_c,),
                    (tensormap_c_ptr,),
                    EPILOG_WARP_IDS[0],
                    (tensormap_c_smem_ptr,),
                )
                tensormap_manager.fence_tensormap_update(tensormap_c_ptr)

                while work.is_valid_tile and work.group_idx == g:
                    tmem_buf, tmem_phase = _get_bufidx_phase(
                        accum_cnt_tile, NUM_TMEM_BUFFERS
                    )
                    tile_buf, _ = _get_bufidx_phase(accum_cnt_tile, NUM_TILE_BUFFERS)

                    tile_m_idx = work.tile_m_idx
                    tile_n_idx = work.tile_n_idx

                    # ---- Wait MMA done for this tmem buffer ----------------
                    if cutlass.const_expr(not self.SPLIT_ATOM_ACCUM):
                        cute.arch.mbarrier_wait(tmem_full_mbar + tmem_buf, tmem_phase)

                    # ---- TMEM -> reg -> SMEM -> GMEM (per-MMA, per-subtile) -
                    c_desc_ptr = tensormap_manager.get_tensormap_ptr(
                        tensormap_c_ptr, cute.AddressSpace.generic
                    )

                    # NUM_MMA_ATOMS_M / N: number of MMA atoms along M / N
                    # for this tile. >1 only for 1cta2mma where BLOCK_M=256
                    # but atom_M=128 → MMA_M=2. We must iterate all atoms
                    # so the full tile gets written to gmem. The partitions
                    # for each atom address a different TMEM column range
                    # and a different gmem (m, n) sub-block.
                    NUM_MMA_ATOMS_M = cute.size(tCtAcc_base.shape, mode=[1])
                    NUM_MMA_ATOMS_N = cute.size(tCtAcc_base.shape, mode=[2])
                    for mma_m_idx in cutlass.range_constexpr(NUM_MMA_ATOMS_M):
                        if cutlass.const_expr(self.SPLIT_ATOM_ACCUM):
                            cute.arch.mbarrier_wait(
                                tmem_full_mbar + mma_m_idx, tmem_phase
                            )
                        for mma_n_idx in cutlass.range_constexpr(NUM_MMA_ATOMS_N):
                            # Re-build per-atom partitions. mma_m_idx and
                            # mma_n_idx are constexpr so this is unrolled at
                            # compile time.
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
                            (
                                _,
                                bSG_sC,
                                bSG_gC_partitioned,
                            ) = self._epilog_gmem_copy_and_partition(
                                tma_atom_c,
                                tCgC,
                                epi_tile,
                                sC,
                                mma_m_idx=mma_m_idx,
                                mma_n_idx=mma_n_idx,
                            )

                            # Slice gC for this tile coord (m, n in cta units).
                            bSG_gC = bSG_gC_partitioned[
                                (None, None, None, tile_m_idx, tile_n_idx, 0)
                            ]
                            # Collapse trailing modes so we can index with a
                            # single subtile_idx (matches the cute/gemm reference).
                            bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                            tTR_tAcc = tTR_tAcc_base[
                                (None, None, None, None, None, tmem_buf)
                            ]
                            # Collapse EPI_M / EPI_N into one indexable mode
                            # (matches the cute/gemm reference).
                            tTR_tAcc = cute.group_modes(
                                tTR_tAcc, 3, cute.rank(tTR_tAcc)
                            )
                            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                            atom_idx = mma_m_idx * NUM_MMA_ATOMS_N + mma_n_idx
                            subtiles_per_tile = (
                                NUM_MMA_ATOMS_M * NUM_MMA_ATOMS_N * subtile_cnt
                            )
                            num_prev_subtiles = (
                                accum_cnt_tile * subtiles_per_tile
                                + atom_idx * subtile_cnt
                            )
                            # Unrolled, the inline SwiGLU body thrashes the
                            # icache.
                            for subtile_idx in cutlass.range(
                                subtile_cnt,
                                unroll=1,
                                unroll_full=not swiglu_inline,
                            ):
                                if cutlass.const_expr(NUM_C_STAGES == 1):
                                    _epilog_wait_pending_tma_store()
                                c_buffer = (
                                    num_prev_subtiles + subtile_idx
                                ) % NUM_C_STAGES

                                # Skip the TMEM->reg copy on the empty-
                                # contraction path (mirrors TLX, which
                                # never reads the uninitialized
                                # accumulator). The retile().load() that
                                # follows is a register-only no-op used
                                # solely to expose the layout/dtype to
                                # `cute.zeros_like`; CuTeDSL requires the
                                # SSA producing `acc_vec` to live outside
                                # any dynamic branch, so we leave the
                                # load unconditional.
                                if num_k_tiles != 0:
                                    cute.copy(
                                        _tiled_copy_t2r_atom,
                                        tTR_tAcc[(None, None, None, subtile_idx)],
                                        tTR_rAcc,
                                    )

                                acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                                if num_k_tiles == 0:
                                    acc_vec = cute.zeros_like(acc_vec)
                                tRS_rC.store(acc_vec.to(sC.element_type))

                                # reg -> smem
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rC,
                                    tRS_sC[(None, None, None, c_buffer)],
                                )

                                # fence + barrier so smem store is visible to TMA
                                cute.arch.fence_proxy(
                                    "async.shared",
                                    space="cta",
                                )
                                cute.arch.barrier(
                                    barrier_id=_BAR_EPILOG_SYNC,
                                    number_of_threads=32 * len(EPILOG_WARP_IDS),
                                )

                                # smem -> gmem TMA store (warp 0 only)
                                warp_idx_local = cute.arch.make_warp_uniform(
                                    cute.arch.warp_idx()
                                )
                                if warp_idx_local == EPILOG_WARP_IDS[0]:
                                    cute.copy(
                                        tma_atom_c,
                                        bSG_sC[(None, c_buffer)],
                                        bSG_gC[(None, subtile_idx)],
                                        tma_desc_ptr=c_desc_ptr,
                                    )
                                    cute.arch.cp_async_bulk_commit_group()
                                    if cutlass.const_expr(NUM_C_STAGES > 1):
                                        cute.arch.cp_async_bulk_wait_group(
                                            NUM_C_STAGES - 1,
                                            read=True,
                                        )
                                if cutlass.const_expr(swiglu_inline):
                                    store_swiglu_h2_fragment(
                                        swiglu_h2,
                                        tRS_rC,
                                        tidx,
                                        work.split_prefix,
                                        m_size,
                                        n_size,
                                        tile_m_idx,
                                        tile_n_idx,
                                        mma_m_idx,
                                        mma_n_idx,
                                        subtile_idx,
                                        NUM_MMA_ATOMS_M,
                                        NUM_MMA_ATOMS_N,
                                        BLOCK_SIZE_M,
                                        BLOCK_SIZE_N,
                                        NUM_CTAS,
                                        cluster_cta_rank,
                                    )
                                if cutlass.const_expr(NUM_C_STAGES > 1):
                                    cute.arch.barrier(
                                        barrier_id=_BAR_EPILOG_SYNC,
                                        number_of_threads=32 * len(EPILOG_WARP_IDS),
                                    )
                        if cutlass.const_expr(
                            self.SPLIT_ATOM_ACCUM and mma_m_idx < NUM_MMA_ATOMS_M - 1
                        ):
                            release_acc(
                                tmem_empty_mbar + mma_m_idx,
                                EPILOG_WARP_IDS[0],
                                32 * len(EPILOG_WARP_IDS),
                                _BAR_EPILOG_SYNC,
                                NUM_CTAS,
                            )

                    acc_slot = tmem_buf
                    if cutlass.const_expr(self.SPLIT_ATOM_ACCUM):
                        acc_slot = NUM_MMA_ATOMS_M - 1
                    release_acc(
                        tmem_empty_mbar + acc_slot,
                        EPILOG_WARP_IDS[0],
                        32 * len(EPILOG_WARP_IDS),
                        _BAR_EPILOG_SYNC,
                        NUM_CTAS,
                    )
                    warp_idx_local = cute.arch.make_warp_uniform(cute.arch.warp_idx())
                    if cutlass.const_expr(swiglu_deferred):
                        store_swiglu_h2_from_h1(
                            swiglu_h2,
                            warp_idx_local == EPILOG_WARP_IDS[0],
                            _BAR_EPILOG_SYNC,
                            32 * len(EPILOG_WARP_IDS),
                            tidx,
                            work.split_prefix,
                            m_size,
                            n_size,
                            tile_m_idx,
                            tile_n_idx,
                            NUM_MMA_ATOMS_M,
                            NUM_MMA_ATOMS_N,
                            subtile_cnt,
                            BLOCK_SIZE_M,
                            BLOCK_SIZE_N,
                            NUM_CTAS,
                            cluster_cta_rank,
                            sC.element_type,
                        )

                    # ---- Signal tile_done so producer can reuse slot --------
                    # Gate by EPILOG_WARP_IDS[0] so exactly ONE warp issues
                    # the signal — otherwise all 4 epilog warps each elect a
                    # lane and we get 4 arrives per CTA (overflows
                    # tile_id_producer_bars whose arrive_count is NUM_CTAS).
                    scheduler.consumer_release_tile(
                        tile_producer_mbar,
                        tile_buf,
                        cluster_cta_rank,
                        warp_idx_local == EPILOG_WARP_IDS[0],
                    )

                    accum_cnt_tile += cutlass.Int32(1)

                    tile_idx = scheduler.advance_consumer(accum_cnt_tile)
                    work = visitor.get_work(tile_idx)

        # Drain the final async store before the epilogue exits and the
        # kernel deallocates TMEM / returns to the caller.
        _epilog_wait_pending_tma_store()

        # TMEM relinquish + dealloc handshake is handled by the caller in
        # `kernel()` immediately after `_epilog_consumer_body` returns
        # (relinquish_tmem_alloc_permit + 2-CTA dealloc-bar handshake +
        # dealloc_tmem).

    # -------------------------------------------------------------------
    # Epilogue copy / partition helpers — direct adaptations of the
    # cute/gemm reference. They build the tiled copies for tmem→reg,
    # reg→smem, smem→gmem and partition the source/destination tensors for
    # each thread.
    # -------------------------------------------------------------------
    def _epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs,
        mma_m_idx: int = 0,
        mma_n_idx: int = 0,
    ) -> tuple:
        """tmem -> reg copy partition.

        For multi-MMA configurations (1cta2mma: atom_M=128 but BLOCK_M=256),
        the accumulator has shape ((MMA_atom), MMA_M, MMA_N, STAGE) with
        MMA_M > 1. We pick a specific (mma_m_idx, mma_n_idx) atom to
        partition. The consumer body iterates over all atoms.
        """
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,  # set in __call__
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        copy_atom_t2r = _cap_tmem_ld_repetition(
            copy_atom_t2r,
            self.acc_dtype,
            max_regs_per_thread=self.TMEM_LOAD_MAX_REGS_PER_THREAD,
        )
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), mma_m_idx, mma_n_idx, None)], epi_tile
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), mma_m_idx, mma_n_idx, None, None, None)],
            epi_tile,
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def _epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> tuple:
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def _epilog_gmem_copy_and_partition(
        self,
        tma_atom_c,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
        mma_m_idx: int = 0,
        mma_n_idx: int = 0,
    ) -> tuple:
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.flat_divide(
            gC_mnl[((None, None), mma_m_idx, mma_n_idx, None, None, None)],
            epi_tile,
        )
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            cute.group_modes(gC_for_tma_partition, 0, 2),
        )
        return tma_atom_c, bSG_sC, bSG_gC

    # -------------------------------------------------------------------
    # Host-side @cute.jit __call__. Builds CUTLASS metadata (tiled MMA,
    # SMEM layouts, TMA atoms), allocates per-CTA tensormap workspace
    # and the global tile counter, packs strides_abc/ptrs_abc, then
    # launches the kernel. Mirrors the cute/gemm reference but with our
    # 4-warp-group, atomic-counter scheduler structure.
    # -------------------------------------------------------------------
    def _setup_attributes(self):
        """Compute layouts that depend on input dtypes/majorness.

        Mirrors the cute/gemm reference's _setup_attributes.

        IMPORTANT: `mma_tiler` is the FULL cluster tile (not per-CTA).
        `cta_tile_shape_mnk` is the per-CTA share, where M is divided by
        `tiled_mma.thr_id.shape` (= NUM_CTAS for cta_group=TWO MMA atom).
        SMEM layouts use mma_tiler (full), and `tma_partition` /
        `partition_A/B/C` automatically deliver the per-CTA slice.

        MMA atom sizing rules (for cta_group=ONE the atom M-mode must be
        64 or 128; for cta_group=TWO it must be 128 or 256). When BLOCK_M
        exceeds the atom's M capacity (only happens in 1cta2mma:
        BLOCK_M=256 with cta_group=ONE), we stack atoms along M via
        `atom_layout_mnk=(NUM_MMAS_ATOM, 1, 1)`. For 2-CTA configs a
        single atom suffices (atom_M=BLOCK_M) since cta_group=TWO already
        handles the per-CTA distribution.
        """
        # Atom M-mode rules (cta_group=ONE: M ∈ {64,128};
        #                    cta_group=TWO: M ∈ {128,256}).
        # `NUM_MMAS` controls how many `tcgen05.mma` atoms are stacked
        # along M per cluster tile. With NUM_MMAS=1 a single atom covers
        # the full cluster M; with NUM_MMAS=2 the kernel emits two stacked
        # MMAs per k-iter (matching TLX's `for mma in static_range(NUM_MMAS)`
        # pattern). `cute.gemm(tiled_mma, ...)` automatically iterates the
        # MMA-M partitions of the operand fragments. Configs respect the
        # atom-M limits by choosing atom_m in the supported set above.
        atom_m = self.BLOCK_SIZE_M // self.NUM_MMAS
        atom_mma_tiler_mn = (atom_m, self.BLOCK_SIZE_N)

        self.tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            atom_mma_tiler_mn,
        )
        # Full cluster tile (M, N, K).
        self.mma_tiler = (*self.mma_tiler_mn, self.BLOCK_SIZE_K)
        # Per-CTA tile (M is split via thr_id; N and K are full).
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(self.tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (self.tiled_mma.thr_id.shape,),
        )

        if self.EPILOGUE_SUBTILE > 0:
            # Explicit subtile count overrides the auto heuristic, which
            # picks tile_n based on a fixed compute_elts/tile_m budget and
            # often ends up smaller than ideal — yielding more subtile
            # passes (TMEM->reg->SMEM->TMA-store) than necessary. TLX uses
            # EPILOGUE_SUBTILE=4 (BLOCK_N // 4 per subtile); mirror that
            # so we get the same number of epilog passes per tile.
            cta_n = self.cta_tile_shape_mnk[1]
            tile_m, tile_n, warp_m, warp_n = _explicit_epilogue_tile_shape(
                cta_m=self.cta_tile_shape_mnk[0],
                cta_n=cta_n,
                num_ctas=self.NUM_CTAS,
                epilogue_subtile=self.EPILOGUE_SUBTILE,
                c_width=self.c_dtype.width,
                a_width=self.a_dtype.width,
            )
            tile_m_layout = cute.make_layout(tile_m)
            tile_n_layout = cute.make_layout(
                (tile_n // warp_n, warp_n), stride=(1, cta_n // warp_n)
            )
            self.epi_tile = (tile_m_layout, cute.coalesce(tile_n_layout))
        else:
            self.epi_tile = utils.compute_epilogue_tile_shape(
                self.cta_tile_shape_mnk,
                self.NUM_CTAS == 2,
                self.c_layout,
                self.c_dtype,
            )

        # SMEM layouts (staged) — use the FULL mma_tiler.
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.NUM_SMEM_BUFFERS,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.NUM_SMEM_BUFFERS,
        )
        self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.NUM_C_STAGES,
        )

        # TMEM allocation columns. Build a fake accumulator fragment whose
        # shape matches what the kernel will actually allocate, then ask
        # CUTLASS for the column count. With `atom_layout_mnk=(NUM_MMAS_ATOM,
        # 1, 1)` the tiled_mma's M already covers BLOCK_SIZE_M, so we no
        # longer multiply by NUM_MMAS — TLX's per-MMA accumulator slicing
        # is now folded into a single MMA-M dim of the fragment.
        acc_shape = self.tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = self.tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.NUM_TMEM_BUFFERS)
        )
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

    @cute.jit
    def __call__(
        self,
        tensor_a: cute.Tensor,
        tensor_b: cute.Tensor,
        tensor_c: cute.Tensor,
        split_sizes: cute.Tensor,
        counter: cute.Tensor,  # int32[1] — global tile counter
        tensormaps: cute.Tensor,  # workspace for descriptor patching
        # Per-tensor base ptrs + strides as scalars (no per-launch tensor):
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
        # Activation-buffer offset support (TLX wgrad-style):
        use_activation_buffer: cutlass.Constexpr[bool],
        activation_buffer_size_bytes: cutlass.Int64,
        a_offset_tensor: cute.Tensor,  # int64[1] (or any int64 tensor; only [0] is read)
        b_offset_tensor: cute.Tensor,
        # Output accumulation: when True the C TMA atom uses
        # CopyReduceBulkTensorTileS2GOp (reduce-add) so writes ACCUMULATE
        # into the existing C tensor instead of overwriting. Used for wgrad
        # in distributed training; for fprop/dgrad always pass False.
        output_accum: cutlass.Constexpr[bool],
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        num_clusters: int,
        stream: cuda.CUstream,
        condition: cute.Tensor,
        USE_CONDITIONAL_EXECUTION: cutlass.Constexpr[bool],
    ):
        tensor_a, tensor_b = _swap_if(self.SWAP_AB, tensor_a, tensor_b)
        a_base_ptr, b_base_ptr = _swap_if(self.SWAP_AB, a_base_ptr, b_base_ptr)
        a_offset_tensor, b_offset_tensor = _swap_if(
            self.SWAP_AB,
            a_offset_tensor,
            b_offset_tensor,
        )
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

        # Capture dtypes / layouts.
        self.output_accum = output_accum
        self.a_dtype = tensor_a.element_type
        self.b_dtype = tensor_b.element_type
        self.c_dtype = tensor_c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(tensor_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(tensor_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(tensor_c)

        # Problem-type majorness overrides:
        # FPROP : A=(m, k=K_in) row-major → K-major (k inner). B=(n=N, k=K_in)
        #         row-major → K-major (k inner). Auto-detect is correct.
        # DGRAD : A=grad_y=(m, k=N) row-major → K-major. B=w=(k=N, n=K_in)
        #         where the placeholder is row-major (N, K_in): n inner →
        #         B is MN-major (k outer). Auto-detect returns K-major,
        #         which is WRONG. Override to MN.
        # WGRAD : A=x=(m=K_in, k=GM) needs MN-major. B=grad_y=(k=GM, n=N)
        #         needs MN-major.
        from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode

        if cutlass.const_expr(self.problem_type == _DGRAD):
            self.b_major_mode = OperandMajorMode.MN
        elif cutlass.const_expr(self.problem_type == _WGRAD):
            self.a_major_mode = OperandMajorMode.MN
            self.b_major_mode = OperandMajorMode.MN

        self._setup_attributes()

        # Build TMA atoms.
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
        # When `output_accum` is True we use the reduce-add bulk TMA store
        # (`CopyReduceBulkTensorTileS2GOp`) so the kernel ACCUMULATES into C
        # rather than overwriting. Required for wgrad in distributed
        # training (gradients accumulated across micro-batches / passes).
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyReduceBulkTensorTileS2GOp()
            if output_accum
            else cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            c_cta_v_layout,
        )

        # Build SharedStorage class — must be done after _setup_attributes
        # so cosize values are computed from the staged SMEM layouts.
        self.shared_storage = self._make_shared_storage(
            self.a_dtype, self.b_dtype, self.c_dtype, G
        )

        # Grid: persistent kernel — num_clusters clusters, NUM_CTAS CTAs each.
        grid = (num_clusters * self.NUM_CTAS, 1, 1)

        # Launch.
        self.kernel(
            tma_atom_a=tma_atom_a,
            tma_atom_b=tma_atom_b,
            tma_atom_c=tma_atom_c,
            mA=tma_tensor_a,
            mB=tma_tensor_b,
            mC=tma_tensor_c,
            a_smem_layout_staged=self.a_smem_layout_staged,
            b_smem_layout_staged=self.b_smem_layout_staged,
            epi_smem_layout_staged=self.epi_smem_layout_staged,
            epi_tile=self.epi_tile,
            cluster_layout_vmnk=self.cluster_layout_vmnk,
            tiled_mma=self.tiled_mma,
            split_sizes=split_sizes,
            counter_ptr=counter.iterator,
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
            use_activation_buffer=use_activation_buffer,
            activation_buffer_size_bytes=activation_buffer_size_bytes,
            a_offset_tensor=a_offset_tensor,
            b_offset_tensor=b_offset_tensor,
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
            NUM_C_STAGES=self.NUM_C_STAGES,
            NUM_TILE_BUFFERS=self.NUM_TILE_BUFFERS,
            BLOCK_SIZE_M=self.BLOCK_SIZE_M,
            BLOCK_SIZE_N=self.BLOCK_SIZE_N,
            BLOCK_SIZE_K=self.BLOCK_SIZE_K,
            NUM_CTAS=self.NUM_CTAS,
            condition=condition,
            USE_CONDITIONAL_EXECUTION=USE_CONDITIONAL_EXECUTION,
        ).launch(
            grid=grid,
            block=(THREADS_PER_CTA, 1, 1),
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.jit
    def _idle_body(self):
        """Idle padding warps (warps 6-7). They participate in the cluster
        rendezvous in the kernel prologue and tail, but do no pipeline work.

        TLX equivalent: `with tlx.async_task(num_warps=2, num_regs=48):
        if NUM_CTAS == 2: tlx.cluster_barrier()`.
        """
        return


@cute.jit
def _make_tensor_for_tensormap_update(
    per_group_base_ptr: cutlass.Int64,
    dtype: type[cutlass.Numeric],
    problem_shape_mnk: tuple,
    s0: cutlass.Int32,  # outer-axis stride for this tensor
    s1: cutlass.Int32,  # inner-axis stride for this tensor
    *,
    tensor_index: cutlass.Constexpr[int],  # 0=A, 1=B, 2=C
    problem_type: cutlass.Constexpr[int],
    use_sm103_ultra: cutlass.Constexpr[bool] = False,
):
    """Build a per-group GMEM tensor for the tensormap update call.

    Inputs are SCALARS (no per-launch device tensor). The producer /
    epilog bodies maintain `start_*` accumulators (TLX-style) and
    compute `per_group_base_ptr = base + start * stride * elem_size`
    plus the per-tensor strides. This eliminates the CPU-side prefix
    sum and the strides_abc/ptrs_abc device tensors entirely.

    Layout is uniform: A=(m, k, 1), B=(n, k, 1), C=(m, n, 1) across
    FPROP/DGRAD/WGRAD; only the per-group base ptr + strides differ.
    """
    # `assumed_align=16` — match the cute/gemm reference. The per-group
    # base ptr alignment is dictated by torch's caching allocator
    # (256B) and stride multiples; we conservatively assume only 16B
    # to mirror the cute/gemm reference exactly (which PASSES the
    # failing shape).
    gmem_ptr = cute.make_ptr(
        dtype, per_group_base_ptr, cute.AddressSpace.gmem, assumed_align=16
    )
    c0 = cutlass.Int32(0)
    c1 = cutlass.Int32(1)
    m = problem_shape_mnk[0]
    n = problem_shape_mnk[1]
    k = problem_shape_mnk[2]
    if cutlass.const_expr(use_sm103_ultra and tensor_index != 2):
        # SM103 ultra TMA atoms address A/B as Uint8 (two FP4 values per
        # byte): halve K and the outer stride to byte units.
        u8_ptr = cute.make_ptr(
            cutlass.Uint8,
            per_group_base_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        k_desc_bytes = (k + cutlass.Int32(1)) // cutlass.Int32(2)
        mn = m if cutlass.const_expr(tensor_index == 0) else n
        return cute.make_tensor(
            u8_ptr,
            cute.make_layout(
                (mn, k_desc_bytes, c1),
                stride=(s0 // cutlass.Int32(2), s1, c0),
            ),
        )
    # 3D layout with trivial batch L=1 — matches the cute/gemm reference
    # which uses (m, k, 1)/(n, k, 1)/(m, n, 1) and PASSES the realistic
    # 8-group failing shape.
    if cutlass.const_expr(tensor_index == 0):  # A
        k_desc = (
            cutlass.max(k, cutlass.Int32(1))
            if cutlass.const_expr(problem_type == _WGRAD)
            else k
        )
        return cute.make_tensor(
            gmem_ptr,
            cute.make_layout((m, k_desc, c1), stride=(s0, s1, c0)),
        )
    elif cutlass.const_expr(tensor_index == 1):  # B
        k_desc = (
            cutlass.max(k, cutlass.Int32(1))
            if cutlass.const_expr(problem_type == _WGRAD)
            else k
        )
        return cute.make_tensor(
            gmem_ptr,
            cute.make_layout((n, k_desc, c1), stride=(s0, s1, c0)),
        )
    else:  # C
        return cute.make_tensor(
            gmem_ptr,
            cute.make_layout((m, n, c1), stride=(s0, s1, c0)),
        )
