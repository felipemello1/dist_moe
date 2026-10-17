# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton kernels for DistMoE dispatch routing.

Device side of ``dist_dispatch_routing.py``: the ``@triton.jit`` kernels for
low-latency decode routing (state clear, fused single-CTA routing), metadata
gather, and the split/prefix/combine training routing pipeline. The host-side
entry points, buffer bookkeeping, and validation live in
``dist_dispatch_routing.py``.
"""

import triton
import triton.language as tl

from ..pdl import pdl_wait_then_release
from .comm_utils import (
    get_flat_tid,
    sync_threads,
)
from .device_trap import device_trap_if


@triton.jit
def _trap_if_peer_token_counts_differ(
    routing_buffer_ptrs,
    num_tokens,
    world_size: tl.constexpr,
    BLOCK_WORLD_SIZE: tl.constexpr,
):
    """Trap before routing when EP ranks publish different physical shapes."""
    ranks = tl.arange(0, BLOCK_WORLD_SIZE)
    rank_mask = ranks < world_size
    routing_buffer_ptrs = routing_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    routing_buffer_addrs = tl.load(
        routing_buffer_ptrs + ranks,
        mask=rank_mask,
        other=0,
    )
    rank_zero_addr = tl.load(routing_buffer_ptrs)
    rank_zero_num_tokens = tl.load(rank_zero_addr.to(tl.pointer_type(tl.int32)))
    peer_num_tokens = tl.load(
        routing_buffer_addrs.to(tl.pointer_type(tl.int32)),
        mask=rank_mask,
        other=num_tokens,
    )
    mismatch = (rank_zero_num_tokens != num_tokens) | (
        tl.sum(
            tl.where(rank_mask, peer_num_tokens != rank_zero_num_tokens, False).to(
                tl.int32
            )
        )
        > 0
    )
    device_trap_if(mismatch & (get_flat_tid() == 0))


@triton.jit
def _triton_clear_decode_routing_state(
    expert_ids_ptr,
    routing_output_ptr,
    expert_counters_ptr,
    completed_ctas_ptr,
    num_tokens_per_local_experts_ptr,
    num_tokens_per_rank_ptr,
    fwd_gather_ptrs_ptr,
    scatter_ptrs_ptr,
    counter_workspace_0_ptr,
    counter_workspace_1_ptr,
    num_routing_slots,
    expert_ids_stride_0,
    expert_ids_stride_1,
    topk: tl.constexpr,
    num_experts: tl.constexpr,
    num_local_experts: tl.constexpr,
    world_size: tl.constexpr,
    ptrs_size,
    counter_workspace_0_size,
    counter_workspace_1_size,
    COPY_EXPERT_IDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PDL: tl.constexpr,
):
    pdl_wait_then_release(PDL)
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if COPY_EXPERT_IDS:
        token_offsets = offsets // topk
        topk_offsets = offsets % topk
        expert_ids = tl.load(
            expert_ids_ptr
            + token_offsets * expert_ids_stride_0
            + topk_offsets * expert_ids_stride_1,
            mask=offsets < num_routing_slots,
        )
        tl.store(
            routing_output_ptr + offsets,
            expert_ids.to(tl.int16),
            mask=offsets < num_routing_slots,
        )
    tl.store(fwd_gather_ptrs_ptr + offsets, 0, mask=offsets < ptrs_size)
    tl.store(scatter_ptrs_ptr + offsets, 0, mask=offsets < ptrs_size)
    tl.store(expert_counters_ptr + offsets, 0, mask=offsets < num_experts)
    tl.store(
        num_tokens_per_local_experts_ptr + offsets,
        0,
        mask=offsets < num_local_experts,
    )
    tl.store(num_tokens_per_rank_ptr + offsets, 0, mask=offsets < world_size)
    tl.store(completed_ctas_ptr + offsets, 0, mask=offsets == 0)
    tl.store(
        counter_workspace_0_ptr + offsets,
        0,
        mask=offsets < counter_workspace_0_size,
    )
    tl.store(
        counter_workspace_1_ptr + offsets,
        0,
        mask=offsets < counter_workspace_1_size,
    )


@triton.jit
def _decode_routing_arrive_and_wait(completed_ctas_ptr, world_size):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 old_count;
            .reg .u32 observed_count;
            .reg .pred waiting;

            atom.global.release.gpu.add.u32 old_count, [$1], 1;
        wait_for_decode_routing_ctas:
            ld.global.acquire.gpu.u32 observed_count, [$1];
            setp.lt.u32 waiting, observed_count, $2;
            @waiting bra wait_for_decode_routing_ctas;
            mov.u32 $0, 0;
        }
        """,
        "=r,l,r",
        [completed_ctas_ptr, world_size],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit(do_not_specialize=["local_rank", "num_tokens"])
def _triton_decode_dispatch_routing(
    routing_buffer_ptrs,
    gather_buffer_ptrs,
    scatter_buffer_ptrs,
    expert_counters_ptr,
    completed_ctas_ptr,
    num_tokens_per_local_experts_ptr,
    num_tokens_per_rank_ptr,
    fwd_gather_ptrs_ptr,
    scatter_ptrs_ptr,
    local_rank,
    num_tokens,
    gather_stride_bytes,
    scatter_stride_bytes,
    world_size: tl.constexpr,
    num_experts: tl.constexpr,
    num_local_experts: tl.constexpr,
    topk: tl.constexpr,
    m_multiple_of: tl.constexpr,
    routing_header_size_bytes: tl.constexpr,
    max_num_recv_tokens,
    has_max_num_recv_tokens: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    BLOCK_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_WORLD_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    PDL: tl.constexpr,
):
    pdl_wait_then_release(PDL)
    # One CTA per (source rank, chunk of BLOCK_SLOTS route slots).
    pid = tl.program_id(0)
    source_rank = pid // NUM_CHUNKS
    chunk = pid % NUM_CHUNKS
    slot_offsets = chunk * BLOCK_SLOTS + tl.arange(0, BLOCK_SLOTS)
    valid_slots = slot_offsets < num_tokens * topk

    # Load the peer's IDs before the token-count check so the two peer reads overlap;
    # the trap still precedes any use. This rank's count bounds the load and the
    # routing buffers are symmetric, so a mismatched peer cannot fault it.
    routing_buffer_addr = (
        tl.load(routing_buffer_ptrs.to(tl.pointer_type(tl.uint64)) + source_rank)
        + routing_header_size_bytes
    )
    routing_buffer_ptr = routing_buffer_addr.to(tl.pointer_type(tl.int16))
    expert_ids = tl.load(
        routing_buffer_ptr + slot_offsets,
        mask=valid_slots,
        other=-1,
    ).to(tl.int32)
    _trap_if_peer_token_counts_differ(
        routing_buffer_ptrs,
        num_tokens,
        world_size,
        BLOCK_WORLD_SIZE,
    )
    valid_experts = valid_slots & (expert_ids >= 0) & (expert_ids < num_experts)
    expert_slots = tl.atomic_add(
        expert_counters_ptr + expert_ids,
        1,
        mask=valid_experts,
        sem="relaxed",
        scope="gpu",
    )

    sync_threads()
    if get_flat_tid() == 0:
        _decode_routing_arrive_and_wait(completed_ctas_ptr, world_size * NUM_CHUNKS)
    sync_threads()

    all_expert_ids = tl.arange(0, BLOCK_EXPERTS)
    expert_counts = tl.load(
        expert_counters_ptr + all_expert_ids,
        mask=all_expert_ids < num_experts,
        other=0,
    )
    if m_multiple_of > 0:
        padded_expert_counts = tl.cdiv(expert_counts, m_multiple_of) * m_multiple_of
    else:
        padded_expert_counts = expert_counts

    local_expert_start = local_rank * num_local_experts
    local_expert_ids = tl.arange(0, BLOCK_LOCAL_EXPERTS)
    local_expert_counts = tl.load(
        expert_counters_ptr + local_expert_start + local_expert_ids,
        mask=local_expert_ids < num_local_experts,
        other=0,
    )
    if m_multiple_of > 0:
        padded_local_expert_counts = (
            tl.cdiv(local_expert_counts, m_multiple_of) * m_multiple_of
        )
    else:
        padded_local_expert_counts = local_expert_counts

    if pid == 0:
        if has_max_num_recv_tokens:
            total_stored_tokens = tl.sum(
                tl.where(
                    local_expert_ids < num_local_experts,
                    padded_local_expert_counts,
                    0,
                ).to(tl.int64)
            )
            _device_trap_if(
                total_stored_tokens > max_num_recv_tokens,
                total_stored_tokens,
                max_num_recv_tokens,
            )
        tl.store(
            num_tokens_per_local_experts_ptr + local_expert_ids,
            padded_local_expert_counts,
            mask=local_expert_ids < num_local_experts,
        )
        # Sum counts per owning rank; unlike a reshape, this works when
        # num_local_experts is not a power of two.
        rank_ids = tl.arange(0, BLOCK_WORLD_SIZE)
        expert_ranks = all_expert_ids // num_local_experts
        rank_counts = tl.sum(
            tl.where(
                expert_ranks[None, :] == rank_ids[:, None],
                padded_expert_counts[None, :],
                0,
            ),
            axis=1,
        )
        tl.store(
            num_tokens_per_rank_ptr + rank_ids, rank_counts, mask=rank_ids < world_size
        )

    local_expert_indices = expert_ids - local_expert_start
    routes_to_local_expert = (
        valid_experts
        & (local_expert_indices >= 0)
        & (local_expert_indices < num_local_experts)
    )
    # Each local expert's first row (exclusive prefix), then one lookup per slot.
    # Non-local slots look up entry 0; the stores below mask them out.
    local_expert_starts = (
        tl.cumsum(padded_local_expert_counts, axis=0) - padded_local_expert_counts
    )
    expert_prefixes = tl.gather(
        local_expert_starts,
        tl.where(routes_to_local_expert, local_expert_indices, 0),
        axis=0,
    )
    output_rows = expert_prefixes.to(tl.int64) + expert_slots.to(tl.int64)

    gather_buffer_ptrs = gather_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    scatter_buffer_ptrs = scatter_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    gather_buffer_addr = tl.load(gather_buffer_ptrs + source_rank)
    scatter_buffer_addr = tl.load(scatter_buffer_ptrs + source_rank)
    token_indices = (slot_offsets // topk).to(tl.int64)
    fwd_gather_addrs = gather_buffer_addr + token_indices * gather_stride_bytes
    scatter_addrs = (
        scatter_buffer_addr + slot_offsets.to(tl.int64) * scatter_stride_bytes
    )
    tl.store(
        fwd_gather_ptrs_ptr + output_rows,
        fwd_gather_addrs,
        mask=routes_to_local_expert,
    )
    tl.store(
        scatter_ptrs_ptr + output_rows,
        scatter_addrs,
        mask=routes_to_local_expert,
    )


@triton.jit(do_not_specialize=["num_tokens"])
def _triton_dist_gather_metadata(
    routing_buffer_ptrs,
    gathered_expert_ids_ptr,
    per_rank_token_cnts_ptr,
    num_tokens_per_rank_ptr,
    world_size: tl.constexpr,
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    block_size_t: tl.constexpr,
    num_gather_workers: tl.constexpr,
    split_num_workers: tl.constexpr,
    index_ty: tl.constexpr,
    routing_header_size_bytes: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_WORLD_SIZE: tl.constexpr,
):
    """Pass 1: Gather expert IDs from NVLink to HBM using large tiles.

    This kernel performs simple memory copies from remote routing buffers
    to local HBM. It uses large tiles (1024 tokens) to maximize NVLink
    bandwidth and requires minimal shared memory. Peer routing buffers must
    already be published and fenced by the caller.

    Args:
        routing_buffer_ptrs: Pointers to each rank's expert_ids buffer
        gathered_expert_ids_ptr: Output HBM buffer [EP, T, K]
        per_rank_token_cnts_ptr: Split counters to initialize [EP, EL, NW]
        num_tokens_per_rank_ptr: Expert counters used to produce per-rank totals [E]
        world_size: Total number of ranks
        num_tokens: Number of tokens per rank
        num_experts: Total number of experts
        topk: Number of experts selected per token
        block_size_t: Block size for token dimension (large, e.g., 1024)
        num_gather_workers: Number of workers in this gather kernel
        split_num_workers: Number of workers in the following split kernel
        index_ty: Triton dtype for indices
        routing_header_size_bytes: Byte offset from the routing header to the
            expert-ID payload.
        BLOCK_K: Power-of-two top-k tile.
        BLOCK_WORLD_SIZE: Power-of-two expert-parallel rank tile.
    """
    remote_rank = tl.program_id(0)
    worker_id = tl.program_id(1)

    _trap_if_peer_token_counts_differ(
        routing_buffer_ptrs,
        num_tokens,
        world_size,
        BLOCK_WORLD_SIZE,
    )

    num_local_experts: tl.constexpr = num_experts // world_size
    zero_block_size: tl.constexpr = 128
    zero_block = tl.arange(0, zero_block_size)
    split_counters_per_rank: tl.constexpr = num_local_experts * split_num_workers
    for zero_start in range(
        worker_id * zero_block_size,
        split_counters_per_rank,
        num_gather_workers * zero_block_size,
    ):
        zero_offsets = zero_start + zero_block
        tl.store(
            per_rank_token_cnts_ptr
            + remote_rank * split_counters_per_rank
            + zero_offsets,
            0,
            mask=zero_offsets < split_counters_per_rank,
        )
    expert_offsets = worker_id * zero_block_size + zero_block
    for expert_start in range(0, num_experts, num_gather_workers * zero_block_size):
        expert_offsets_i = expert_start + expert_offsets
        tl.store(
            num_tokens_per_rank_ptr + expert_offsets_i,
            0,
            mask=(remote_rank == 0) & (expert_offsets_i < num_experts),
        )

    # Get remote routing buffer pointer
    routing_buffer_ptrs = routing_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    routing_buffer_addr = (
        tl.load(routing_buffer_ptrs + remote_rank) + routing_header_size_bytes
    )
    routing_buffer_ptr = routing_buffer_addr.to(tl.pointer_type(index_ty))

    # Calculate work range for this worker
    num_tokens_per_worker = tl.cdiv(num_tokens, num_gather_workers)
    start_t = worker_id * num_tokens_per_worker
    end_t = tl.minimum(start_t + num_tokens_per_worker, num_tokens)

    block_t = tl.arange(0, block_size_t)
    # BLOCK_K is host-computed (= next_power_of_2(topk)); newer Triton's jit
    # rejects calling triton.next_power_of_2 in-kernel.
    block_k = tl.arange(0, BLOCK_K)
    mask_k = block_k < topk

    # Copy expert IDs from NVLink to HBM with large tiles
    for offset_t in range(start_t, end_t, block_size_t):
        mask_t = (offset_t + block_t) < end_t

        # Load from remote routing buffer via NVLink
        remote_experts = tl.load(
            routing_buffer_ptr
            + (offset_t + block_t)[:, None] * topk
            + block_k[None, :],
            mask=mask_t[:, None] & mask_k[None, :],
            other=-1,
        )  # [bT, K]

        # Store to local HBM buffer (keep in L2 for split kernel to read)
        tl.store(
            gathered_expert_ids_ptr
            + remote_rank.to(tl.int64) * num_tokens * topk
            + (offset_t + block_t)[:, None].to(tl.int64) * topk
            + block_k[None, :],
            remote_experts,
            mask=mask_t[:, None] & mask_k[None, :],
            eviction_policy="evict_last",
        )


@triton.jit(do_not_specialize=["expert_id_offset", "dest_rank_base", "num_tokens"])
def _triton_dispatch_routing_split(
    gathered_expert_ids_ptr,
    gather_buffer_ptrs,
    scatter_buffer_ptrs,
    per_rank_token_cnts_ptr,
    per_rank_bwd_gather_ptrs_ptr,
    gather_stride_bytes,
    bwd_gather_stride_bytes,
    per_rank_fwd_gather_ptrs_ptr,
    per_rank_scatter_ptrs_ptr,
    scatter_stride_bytes,
    num_tokens_per_rank_ptr,
    expert_id_offset,
    dest_rank_base,
    world_size: tl.constexpr,
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    block_size_t: tl.constexpr,
    BLOCK_EL: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Pass 2: Process gathered expert IDs with smaller tiles.

    This kernel reads expert IDs from local HBM (gathered in Pass 1) and
    computes gather/scatter indices. It uses smaller tiles (128 tokens)
    to reduce shared memory usage, avoiding the overflow issue.

    The 3D tensor allocation is now smaller:
    - local_mask is [BLOCK_EL, block_size_t, K] instead of [BLOCK_EL, 256, K]
    - This cuts shared memory usage in half compared to the original implementation

    Args:
        gathered_expert_ids_ptr: Input HBM buffer [EP, T, K] from Pass 1
        gather_buffer_ptrs: Pointers to each rank's input token buffer
        scatter_buffer_ptrs: Pointers to each rank's output token buffer
        per_rank_token_cnts_ptr: Output token counts [EP, EL, NW]
        per_rank_bwd_gather_ptrs_ptr: Output bwd gather indices [EP, EL, NW, T_per_worker]
        gather_stride_bytes: Per-token stride of the gather buffer in bytes
        bwd_gather_stride_bytes: Per-topk-token stride of the backward gather
            buffer in bytes
        per_rank_fwd_gather_ptrs_ptr: Output fwd gather indices [EP, EL, NW, T_per_worker]
        per_rank_scatter_ptrs_ptr: Output scatter indices [EP, EL, NW, T_per_worker]
        scatter_stride_bytes: Per-token stride of the scatter buffer in bytes
        expert_id_offset: Starting global expert ID owned by this rank
        dest_rank_base: expert_id_offset // num_local_experts, subtracted from
            dest_rank computation to map global expert IDs to group-local ranks
        world_size: Total number of ranks
        num_tokens: Number of tokens per rank
        num_experts: Total number of experts
        topk: Number of experts selected per token
        block_size_t: Block size for token dimension (small, e.g., 128)
    """
    tl.static_assert(num_experts % world_size == 0)
    num_local_experts: tl.constexpr = num_experts // world_size
    # BLOCK_EL / BLOCK_K are host-computed (= next_power_of_2 of num_local_experts
    # / topk); newer Triton's jit rejects calling triton.next_power_of_2 in-kernel.

    remote_rank = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    # Get buffer pointers for gather/scatter
    gather_buffer_ptrs = gather_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    gather_buffer_addr = tl.load(gather_buffer_ptrs + remote_rank)

    scatter_buffer_ptrs = scatter_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    scatter_buffer_addr = tl.load(scatter_buffer_ptrs + remote_rank)

    # Identify local experts.  Padded entries (beyond num_local_experts) are
    # set to -1 so they can never match a valid expert ID — prevents phantom
    # matches that would cause OOB writes into per-rank buffers.
    block_el = tl.arange(0, BLOCK_EL)
    el_mask = block_el < num_local_experts
    local_experts = tl.where(el_mask, expert_id_offset + block_el, -1)  # [BLOCK_EL]

    block_k = tl.arange(0, BLOCK_K)
    mask_k = block_k < topk
    block_t = tl.arange(0, block_size_t)

    BLOCK_E: tl.constexpr = triton.next_power_of_2(num_experts)
    block_experts = tl.arange(0, BLOCK_E)
    expert_counts = tl.zeros((BLOCK_E,), dtype=tl.int32)

    # Setup output pointers
    num_tokens_per_worker = tl.cdiv(num_tokens, num_workers)
    if per_rank_bwd_gather_ptrs_ptr is not None:
        per_rank_bwd_gather_ptrs_ptr += (
            remote_rank * num_local_experts * num_workers * num_tokens_per_worker
        )
    per_rank_fwd_gather_ptrs_ptr += (
        remote_rank * num_local_experts * num_workers * num_tokens_per_worker
    )
    per_rank_scatter_ptrs_ptr += (
        remote_rank * num_local_experts * num_workers * num_tokens_per_worker
    )
    per_rank_token_cnts_ptr += remote_rank * num_local_experts * num_workers

    base_offsets = (
        block_el * num_workers * num_tokens_per_worker
        + worker_id * num_tokens_per_worker
    )[:, None]
    base_offsets = tl.broadcast_to(
        base_offsets, (BLOCK_EL, block_size_t)
    )  # [BLOCK_EL, bT]
    running_token_counts = tl.zeros((BLOCK_EL,), dtype=tl.int32)  # [BLOCK_EL]

    start_t = worker_id * num_tokens_per_worker
    end_t = tl.minimum(start_t + num_tokens_per_worker, num_tokens)

    for offset_t in range(start_t, end_t, block_size_t):
        mask_t = (offset_t + block_t) < end_t

        # Load expert IDs from local HBM (gathered in Pass 1, discard from cache after use)
        remote_experts = tl.load(
            gathered_expert_ids_ptr
            + remote_rank.to(tl.int64) * num_tokens * topk
            + (offset_t + block_t)[:, None].to(tl.int64) * topk
            + block_k[None, :],
            mask=mask_t[:, None] & mask_k[None, :],
            other=-1,
            eviction_policy="evict_first",
        )  # [bT, K]
        tl.static_assert(remote_experts.shape == (block_size_t, BLOCK_K))

        # Calculate which tokens match local experts.
        # Guard against -1 padding/sentinel lanes phantom-matching: padded
        # EXPERT lanes carry local_experts=-1 (tl.where(el_mask, ..., -1)
        # above) and padded TOKEN lanes load remote_experts=-1 (other=-1
        # above). A bare `==` makes -1 == -1 a phantom match → inflated
        # counts/offsets → out-of-bounds gather/scatter pointer stores (and a
        # downstream illegal address when the grouped-GEMM dispatch
        # dereferences those pointers). Exclude any -1 lane from matching.
        # Exposed by hot experts: L+G is typically non-power-of-2, so
        # BLOCK_EL > num_local_experts creates padded expert lanes.
        local_mask = (
            (remote_experts[None, :, :] == local_experts[:, None, None])
            & (local_experts[:, None, None] >= 0)
            & (remote_experts[None, :, :] >= 0)
            & mask_k[None, None, :]
        )  # [BLOCK_EL, bT, K]
        local_index = tl.where(
            local_mask, block_k[None, None, :], 0
        )  # [BLOCK_EL, bT, K]
        tl.static_assert(local_mask.shape == (BLOCK_EL, block_size_t, BLOCK_K))
        tl.static_assert(local_index.shape == (BLOCK_EL, block_size_t, BLOCK_K))

        local_mask = tl.sum(local_mask.to(tl.int32), axis=2)  # [BLOCK_EL, bT]
        local_index = tl.sum(local_index, axis=2)  # [BLOCK_EL, bT]

        tl.static_assert(local_mask.shape == (BLOCK_EL, block_size_t))
        tl.static_assert(local_index.shape == (BLOCK_EL, block_size_t))

        # Compute output offsets (same as original)
        local_offsets = (
            base_offsets
            + running_token_counts[:, None]
            + tl.cumsum(local_mask.to(tl.int32), axis=1)
            - 1
        )  # [BLOCK_EL, bT]
        local_mask_boolean = local_mask > 0
        # Defense-in-depth: mask the pointer stores with the same validity
        # invariant the token counts use — el_mask for padded EXPERT lanes,
        # mask_t for padded TOKEN lanes — so a padded/sentinel lane can never
        # store into the gather/scatter pointer buffers even if a fill value
        # changes later. The guarded equality above already prevents the -1
        # phantom match; this encodes the invariant directly at the stores
        # (the original OOB was specifically unguarded pointer-buffer stores).
        store_mask = local_mask_boolean & el_mask[:, None] & mask_t[None, :]

        # Update gather indices
        token_idx_i64 = (offset_t.to(tl.int64) + block_t.to(tl.int64))[
            None, :
        ]  # [1, bT]
        # Topk-expanded element index (used for both bwd gather and scatter, [T, K, D] layout)
        topk_element_index = token_idx_i64 * topk + local_index.to(
            tl.int64
        )  # [BLOCK_EL, bT]
        # Fwd gather index (dispatch buffer as [T, D], non-topk-expanded)
        fwd_gather_element_index = token_idx_i64  # [1, bT]
        gather_stride_i64 = gather_stride_bytes.to(tl.int64)
        # Backward gather pointers (topk-expanded) are only needed by the
        # backward pass; inference fprop passes a None buffer to skip them.
        if per_rank_bwd_gather_ptrs_ptr is not None:
            bwd_gather_stride_i64 = bwd_gather_stride_bytes.to(tl.int64)
            bwd_gather_byte_offset = (
                topk_element_index * bwd_gather_stride_i64
            )  # [BLOCK_EL, bT] in bytes
            bwd_gather_addrs = (
                gather_buffer_addr + bwd_gather_byte_offset
            )  # [BLOCK_EL, bT]
            tl.store(
                per_rank_bwd_gather_ptrs_ptr + local_offsets,
                value=bwd_gather_addrs,
                mask=store_mask,
                eviction_policy="evict_last",
            )

        # Update fwd gather indices (non-topk-expanded, for forward dispatch gather)
        fwd_bwd_gather_byte_offset = (
            fwd_gather_element_index * gather_stride_i64
        )  # [1, bT] in bytes, broadcast over BLOCK_EL at store
        fwd_bwd_gather_addrs = (
            gather_buffer_addr + fwd_bwd_gather_byte_offset
        )  # [1, bT], broadcast over BLOCK_EL at store
        tl.store(
            per_rank_fwd_gather_ptrs_ptr + local_offsets,
            value=fwd_bwd_gather_addrs,
            mask=store_mask,
            eviction_policy="evict_last",
        )

        # Update scatter indices
        scatter_stride_i64 = scatter_stride_bytes.to(tl.int64)
        scatter_byte_offset = (
            topk_element_index * scatter_stride_i64
        )  # [BLOCK_EL, bT] in bytes
        scatter_addrs = scatter_buffer_addr + scatter_byte_offset  # [BLOCK_EL, bT]
        tl.store(
            per_rank_scatter_ptrs_ptr + local_offsets,
            value=scatter_addrs,
            mask=store_mask,
            eviction_policy="evict_last",
        )

        # Update token counts
        local_mask_for_counting = tl.where(mask_t[None, :], local_mask, 0)
        token_counts = tl.sum(
            local_mask_for_counting.to(tl.int32), axis=1
        )  # [BLOCK_EL]
        tl.atomic_add(
            per_rank_token_cnts_ptr + block_el * num_workers + worker_id,
            token_counts,
            mask=el_mask,
            sem="relaxed",
        )
        running_token_counts += token_counts  # [BLOCK_EL]

        group_expert_ids = (remote_experts - dest_rank_base * num_local_experts).to(
            tl.int32
        )
        valid_expert = (
            mask_t[:, None]
            & mask_k[None, :]
            & (group_expert_ids >= 0)
            & (group_expert_ids < num_experts)
        )
        expert_counts += tl.histogram(
            tl.reshape(group_expert_ids, (block_size_t * BLOCK_K,)),
            BLOCK_E,
            mask=tl.reshape(valid_expert, (block_size_t * BLOCK_K,)),
        )

    tl.atomic_add(
        num_tokens_per_rank_ptr + block_experts,
        expert_counts,
        mask=block_experts < num_experts,
    )


@triton.jit
def _device_trap_if(condition, actual, capacity):
    """Report scratch-capacity overflow and trap without Triton debug mode."""
    should_trap = condition & (get_flat_tid() == 0)
    if should_trap:
        tl.device_print("DistMoE scratch capacity exceeded")
        tl.device_print("required_receive_rows: ", actual)
        tl.device_print("total_scratch_capacity_rows: ", capacity)
    device_trap_if(should_trap)


@triton.jit
def _triton_dispatch_routing_prefix(
    per_rank_token_cnts_ptr,
    token_cnts_ptr,
    num_tokens_per_rank_ptr,
    compact_offsets_ptr,
    world_size: tl.constexpr,
    num_experts: tl.constexpr,
    num_workers: tl.constexpr,
    compact_ctas_per_pair: tl.constexpr,
    workers_per_compact_cta: tl.constexpr,
    BLOCK_EL: tl.constexpr,
    m_multiple_of: tl.constexpr = 0,
    max_num_recv_tokens=0,
    has_max_num_recv_tokens: tl.constexpr = False,
):
    """Compute compacted output offsets and token counts in one CTA.

    The combine CTAs consume one scalar offset each instead of independently
    reducing the full ``[rank, local_expert, worker]`` count tensor.
    """
    tl.static_assert(num_experts % world_size == 0)
    tl.static_assert(num_workers % compact_ctas_per_pair == 0)
    num_local_experts: tl.constexpr = num_experts // world_size

    block_ranks = tl.arange(0, world_size)
    block_local_experts = tl.arange(0, BLOCK_EL)
    block_workers = tl.arange(0, num_workers)
    el_mask = block_local_experts < num_local_experts
    counts = tl.load(
        per_rank_token_cnts_ptr
        + block_ranks[:, None, None] * num_local_experts * num_workers
        + block_local_experts[None, :, None] * num_workers
        + block_workers[None, None, :],
        mask=el_mask[None, :, None],
        other=0,
        eviction_policy="evict_first",
    )
    chunk_counts = tl.sum(
        tl.reshape(
            counts,
            (
                world_size,
                BLOCK_EL,
                compact_ctas_per_pair,
                workers_per_compact_cta,
            ),
        ),
        axis=3,
    )
    pair_counts = tl.sum(chunk_counts, axis=2)
    expert_counts = tl.sum(pair_counts, axis=0)
    if m_multiple_of > 0:
        stored_counts = tl.cdiv(expert_counts, m_multiple_of) * m_multiple_of
    else:
        stored_counts = expert_counts

    if has_max_num_recv_tokens:
        total_stored_tokens = tl.sum(tl.where(el_mask, stored_counts, 0).to(tl.int64))
        _device_trap_if(
            total_stored_tokens > max_num_recv_tokens,
            total_stored_tokens,
            max_num_recv_tokens,
        )
    tl.store(token_cnts_ptr + block_local_experts, stored_counts, mask=el_mask)

    rank_expert_counts = tl.load(
        num_tokens_per_rank_ptr
        + block_ranks[:, None] * num_local_experts
        + block_local_experts[None, :],
        mask=el_mask[None, :],
        other=0,
    )
    if m_multiple_of > 0:
        rank_expert_counts = tl.cdiv(rank_expert_counts, m_multiple_of) * m_multiple_of
    tl.store(
        num_tokens_per_rank_ptr + block_ranks,
        tl.sum(rank_expert_counts, axis=1),
    )

    expert_prefix = tl.cumsum(stored_counts, axis=0) - stored_counts
    rank_prefix = tl.cumsum(pair_counts, axis=0) - pair_counts
    chunk_prefix = tl.cumsum(chunk_counts, axis=2) - chunk_counts
    compact_offsets = (
        expert_prefix[None, :, None] + rank_prefix[:, :, None] + chunk_prefix
    ).to(tl.int64)
    block_compact_ctas = tl.arange(0, compact_ctas_per_pair)
    tl.store(
        compact_offsets_ptr
        + block_ranks[:, None, None] * num_local_experts * compact_ctas_per_pair
        + block_local_experts[None, :, None] * compact_ctas_per_pair
        + block_compact_ctas[None, None, :],
        compact_offsets,
        mask=el_mask[None, :, None],
    )


@triton.jit(do_not_specialize=["num_tokens_per_worker"])
def _triton_dispatch_routing_combine(  # noqa: C901
    per_rank_token_cnts_ptr,
    per_rank_bwd_gather_ptrs_ptr,
    per_rank_fwd_gather_ptrs_ptr,
    per_rank_scatter_ptrs_ptr,
    token_cnts_ptr,
    compact_offsets_ptr,
    bwd_gather_ptrs_ptr,
    fwd_gather_ptrs_ptr,
    scatter_ptrs_ptr,
    world_size: tl.constexpr,
    num_experts: tl.constexpr,
    block_size_t: tl.constexpr,
    num_workers: tl.constexpr,
    num_tokens_per_worker,
    compact_ctas_per_pair: tl.constexpr,
    workers_per_compact_cta: tl.constexpr,
    m_multiple_of: tl.constexpr = 0,
):
    """Global compaction kernel - compacts gather/scatter indices from all peers.

    This kernel compacts the sparse per-rank indices into dense output arrays.
    It runs one or more CTAs for each (rank, expert) pair. A preceding prefix
    kernel computes the destination offset consumed by each CTA.

    For each peer rank, indices are compacted in order by:
    1. Local expert ID (0 to num_local_experts-1)
    2. Remote rank ID (0 to world_size-1)

    This ensures tokens are grouped by expert for efficient grouped GEMM execution.

    Args:
        per_rank_token_cnts_ptr: Input token counts per peer [EP, EL, NW]
        per_rank_bwd_gather_ptrs_ptr: Input bwd gather indices per peer [EP, EL, NW, T_per_worker]
        per_rank_fwd_gather_ptrs_ptr: Input fwd gather indices per peer [EP, EL, NW, T_per_worker]
        per_rank_scatter_ptrs_ptr: Input scatter indices per peer [EP, EL, NW, T_per_worker]
        token_cnts_ptr: Output total token counts per local expert [EL]
        compact_offsets_ptr: Destination offsets per compact CTA [EP, EL, NC]
        bwd_gather_ptrs_ptr: Output compacted bwd gather indices [T']
        fwd_gather_ptrs_ptr: Output compacted fwd gather indices [T']
        scatter_ptrs_ptr: Output compacted scatter indices [T']
        world_size: Total number of ranks
        num_experts: Total number of experts
        block_size_t: Block size for token dimension
        num_workers: Number of workers used in split phase
        num_tokens_per_worker: Tokens per worker
        compact_ctas_per_pair: Number of CTAs compacting each (rank, expert) pair
        workers_per_compact_cta: Consecutive split workers handled by each CTA
    """
    tl.static_assert(num_experts % world_size == 0)
    tl.static_assert(num_workers % compact_ctas_per_pair == 0)
    num_local_experts: tl.constexpr = num_experts // world_size
    pid = tl.program_id(0)
    compact_cta_id = pid % compact_ctas_per_pair
    pair_id = pid // compact_ctas_per_pair
    remote_rank = pair_id // num_local_experts
    local_expert_id = pair_id % num_local_experts
    worker_start = compact_cta_id * workers_per_compact_cta

    # 2.0 Gather / scatter index buffers use logical [EP, EL, NW, T_per_worker] layout.
    # Advance pointers to this remote_rank's slice - use int64 to avoid overflow
    if per_rank_bwd_gather_ptrs_ptr is not None:
        per_rank_bwd_gather_ptrs_ptr += (
            remote_rank.to(tl.int64)
            * num_local_experts
            * num_workers
            * num_tokens_per_worker
        )
    per_rank_fwd_gather_ptrs_ptr += (
        remote_rank.to(tl.int64)
        * num_local_experts
        * num_workers
        * num_tokens_per_worker
    )
    per_rank_scatter_ptrs_ptr += (
        remote_rank.to(tl.int64)
        * num_local_experts
        * num_workers
        * num_tokens_per_worker
    )

    # 2.1 Load the prefix computed once for this rank/expert/worker group.
    compact_offset = (
        remote_rank * num_local_experts * compact_ctas_per_pair
        + local_expert_id * compact_ctas_per_pair
        + compact_cta_id
    )
    copy_dst_start_t = tl.load(compact_offsets_ptr + compact_offset)

    block_t = tl.arange(0, block_size_t)
    block_w = tl.arange(0, workers_per_compact_cta)

    # 2.2 Compact every worker of this CTA at once: the worker counts give each
    # worker's destination by an exclusive prefix, so the dependent per-worker
    # count loads and copies no longer serialize.
    copy_token_counts = tl.load(
        per_rank_token_cnts_ptr
        + remote_rank * num_local_experts * num_workers
        + local_expert_id * num_workers
        + worker_start
        + block_w,
        eviction_policy="evict_first",
    )  # [W]
    worker_dst = copy_dst_start_t + (
        tl.cumsum(copy_token_counts, axis=0) - copy_token_counts
    ).to(tl.int64)  # [W]
    # Offset into the logical [EL, NW, T_per_worker] layout.
    worker_src = (
        local_expert_id * num_workers * num_tokens_per_worker
        + (worker_start + block_w) * num_tokens_per_worker
    ).to(tl.int64)  # [W]
    for offset_t in range(0, tl.max(copy_token_counts, axis=0), block_size_t):
        mask_t = (offset_t + block_t)[None, :] < copy_token_counts[:, None]
        token_offsets = (offset_t + block_t).to(tl.int64)[None, :]
        src_offsets = worker_src[:, None] + token_offsets  # [W, bT]
        dst_offsets = worker_dst[:, None] + token_offsets  # [W, bT]

        # Backward gather pointers are backward-only; a None buffer (from
        # inference fprop) skips their compaction.
        if per_rank_bwd_gather_ptrs_ptr is not None:
            bwd_gather_vals = tl.load(
                per_rank_bwd_gather_ptrs_ptr + src_offsets,
                mask=mask_t,
                other=0,
                eviction_policy="evict_first",
            )
            tl.store(bwd_gather_ptrs_ptr + dst_offsets, bwd_gather_vals, mask=mask_t)

        fwd_gather_vals = tl.load(
            per_rank_fwd_gather_ptrs_ptr + src_offsets,
            mask=mask_t,
            other=0,
            eviction_policy="evict_first",
        )
        tl.store(fwd_gather_ptrs_ptr + dst_offsets, fwd_gather_vals, mask=mask_t)

        scatter_ptrs = tl.load(
            per_rank_scatter_ptrs_ptr + src_offsets,
            mask=mask_t,
            other=0,
            eviction_policy="evict_first",
        )
        tl.store(scatter_ptrs_ptr + dst_offsets, scatter_ptrs, mask=mask_t)

    copy_dst_start_t += tl.sum(copy_token_counts, axis=0).to(tl.int64)

    # Zero-fill padding slots at the tail of this expert's region. Only the
    # last source rank's CTA does this — by construction it writes after all
    # other source ranks for the same expert. The outer `if` is a constexpr
    # branch, so when `m_multiple_of == 0` the whole block is DCE'd.
    if m_multiple_of > 0:
        if (remote_rank == world_size - 1) & (
            compact_cta_id == compact_ctas_per_pair - 1
        ):
            expert_base = tl.load(
                compact_offsets_ptr + local_expert_id * compact_ctas_per_pair
            )
            padded_end = expert_base + tl.load(token_cnts_ptr + local_expert_id)
            pad_offset = copy_dst_start_t
            while pad_offset < padded_end:
                pad_block = pad_offset + block_t.to(tl.int64)
                pad_mask = pad_block < padded_end
                if bwd_gather_ptrs_ptr is not None:
                    tl.store(bwd_gather_ptrs_ptr + pad_block, 0, mask=pad_mask)
                tl.store(fwd_gather_ptrs_ptr + pad_block, 0, mask=pad_mask)
                tl.store(scatter_ptrs_ptr + pad_block, 0, mask=pad_mask)
                pad_offset += block_size_t
