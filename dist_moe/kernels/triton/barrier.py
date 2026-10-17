# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton symmetric-memory barrier helpers.

This barrier intentionally does not use the backend-managed `signal_pads_dev_`
array from `hdl.barrier()`. Instead it launches from Python with the
`signal_pad_ptrs_tensor` copied from `hdl.signal_pad_ptrs`, which gives us a
clean way to A/B the backend barrier against the exact same signal-pad protocol.
"""

import torch
import triton
import triton.language as tl

from ..pdl import pdl_launch_kwargs, pdl_wait_then_release
from .annotations import annotate_barrier
from .comm_utils import (
    get_flat_tid,
    multimem_red_add1_release_u32,
    red_sub_relaxed_sys_u32,
    send_signal,
    sync_threads,
    wait_signal,
    wait_wrap_ge_u32,
)


@triton.jit(do_not_specialize=["local_rank"])
def _triton_symm_mem_barrier(
    signal_pad_ptrs,
    local_rank,
    conditional_execution_ptr,
    world_size: tl.constexpr,
    channel: tl.constexpr,
):
    # Check conditional execution first — skip barrier if condition is 0
    if conditional_execution_ptr is not None:
        cond_val = tl.load(conditional_execution_ptr, eviction_policy="evict_last")
        if cond_val == 0:
            return

    tid = get_flat_tid()

    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    local_signal_pad_ptr = tl.load(signal_pad_ptrs + local_rank).to(
        tl.pointer_type(tl.uint32)
    )

    sync_threads()
    if tid < world_size and tid != local_rank:
        peer_signal_pad_ptr = tl.load(signal_pad_ptrs + tid).to(
            tl.pointer_type(tl.uint32)
        )
        send_signal_ptr = peer_signal_pad_ptr + channel * world_size + local_rank
        wait_signal_ptr = local_signal_pad_ptr + channel * world_size + tid
        send_signal(send_signal_ptr, sem="acq_rel")
        wait_signal(wait_signal_ptr, sem="acq_rel")
    sync_threads()


def triton_symm_mem_barrier_ptrs(
    signal_pad_ptrs: torch.Tensor,
    *,
    rank: int,
    n_ranks: int,
    pg_name: str = "",
    channel: int = 1,
    conditional_execution=None,
) -> None:
    """Run a symmetric-memory barrier via Triton from raw pointers (no
    ``SymmMemBuffer`` required), on the current CUDA stream.

    Args:
        signal_pad_ptrs: Device int64 tensor of per-rank signal-pad base pointers
            (one per rank, including self). Each pad must hold at least
            ``(channel + 1) * n_ranks`` zero-initialized uint32 slots and be
            IPC-visible to every peer.
        rank: This rank's index among the ``n_ranks`` sharing the signal pads.
        n_ranks: Number of ranks sharing the signal pads (NOT the global PG size).
        pg_name: Process-group name; used only for the profiler annotation.
        channel: Signal pad channel for the barrier.
        conditional_execution: Optional device tensor with one bool or int32 element.
            If provided and zero, the barrier is skipped on all ranks.
            All ranks must pass the same logical condition to avoid deadlocks.
    """
    block_size = max(32, triton.next_power_of_2(n_ranks))
    num_warps = max(1, block_size // 32)
    with annotate_barrier(pg_name=pg_name):
        _triton_symm_mem_barrier[(1,)](
            signal_pad_ptrs=signal_pad_ptrs,
            local_rank=rank,
            conditional_execution_ptr=conditional_execution,
            world_size=n_ranks,
            channel=channel,
            num_warps=num_warps,
        )


def triton_symm_mem_barrier(
    symm_mem_buf,
    channel: int = 1,
    conditional_execution=None,
) -> None:
    """``SymmMemBuffer`` adapter over :func:`triton_symm_mem_barrier_ptrs`.

    Args:
        symm_mem_buf: Symmetric memory buffer handle.
        channel: Signal pad channel for the barrier.
        conditional_execution: Optional device tensor with a single int32 element.
            If provided and value is 0, the barrier is skipped on all ranks.
            All ranks must pass the same logical condition to avoid deadlocks.
    """
    triton_symm_mem_barrier_ptrs(
        symm_mem_buf.signal_pad_ptrs_tensor,
        rank=symm_mem_buf.hdl.rank,
        n_ranks=symm_mem_buf.hdl.world_size,
        pg_name=symm_mem_buf.pg_name,
        channel=channel,
        conditional_execution=conditional_execution,
    )


@triton.jit(do_not_specialize=["mc_flag_ptr", "local_flag_ptr", "channel"])
def _triton_symm_mem_barrier_multimem(
    mc_flag_ptr,
    local_flag_ptr,
    channel,
    conditional_execution_ptr,
    world_size: tl.constexpr,
    PDL: tl.constexpr,
):
    """Multicast (NVLS) symmetric-memory barrier.

    One ``multimem.red`` arrival increments every rank's u32 arrive count at
    once — O(1) instructions per rank instead of the O(world_size) per-peer
    CAS exchange in ``_triton_symm_mem_barrier``. Each rank then
    acquire-polls its own arrive count until all ``world_size`` arrivals of
    the current round have landed, and resets it by atomically subtracting
    ``world_size``.

    The MC/UC address split is what makes the reset safe and the protocol
    stateless across invocations: increments go through the MULTICAST VA, so
    every rank's arrive count sees every arrival, while the reset decrements
    through the rank's own UNICAST VA — the one arrive count only it reads.
    A frontrunner's freshly reset arrive count stays below ``world_size``
    until the last laggard issues its multicast arrival, so no rank runs more
    than one round ahead; and a reset never touches a peer's arrive count, so
    it cannot erase a round a laggard has yet to observe. No wait-target
    state to maintain, no monotonically-increasing counter, no wrap handling.

    Concretely, with frontrunner F and laggard S at ``world_size=2``, round r:

    1. Both arrive: F's and S's arrive counts each reach 2.
    2. F's poll sees 2; F resets its count to 0, races into round r+1 and
       arrives again: F's count is 1, S's count is 3 (S has not reset yet).
    3. S's poll needs >= 2 (not ==): 3 passes — all round-r arrivals are
       provably in, F's early round-r+1 arrival just rides on top. S
       subtracts 2, leaving exactly that early arrival (1) banked for r+1.
    4. F polls round r+1 at 1 < 2 and blocks until S's arrival lands — no
       rank runs more than one round ahead, bounding every arrive count to
       ``[0, 2 * world_size)``.

    Concurrent barriers on the same channel from different streams are not
    allowed (two interleaved rounds satisfy each other's counts).
    """
    pdl_wait_then_release(PDL)
    if conditional_execution_ptr is not None:
        cond_val = tl.load(conditional_execution_ptr, eviction_policy="evict_last")
        if cond_val == 0:
            return

    tid = get_flat_tid()
    if tid == 0:
        multimem_red_add1_release_u32(mc_flag_ptr + channel * 4)
        wait_wrap_ge_u32(local_flag_ptr + channel * 4, world_size)
        red_sub_relaxed_sys_u32(local_flag_ptr + channel * 4, world_size)


def triton_symm_mem_barrier_multimem_ptrs(
    mc_flag_ptr: int,
    local_flag_ptr: int,
    *,
    n_ranks: int,
    pg_name: str = "",
    channel: int = 1,
    conditional_execution=None,
) -> None:
    """Run a multicast symmetric-memory barrier via Triton from raw pointers,
    on the current CUDA stream.

    Args:
        mc_flag_ptr: Multicast VA of the flag array — u32 slots (int32 bit
            pattern), one per channel, zero-initialized on all ranks before
            first use. The protocol is self-resetting: every slot returns to
            zero once all ranks' rounds on it have been reset.
        local_flag_ptr: This rank's unicast VA of the same flag array.
        n_ranks: Number of ranks subscribed to the multicast group.
        pg_name: Process-group name; used only for the profiler annotation.
        channel: Flag slot for the barrier; must be within the flag array.
        conditional_execution: Optional device tensor with one bool or int32
            element. If provided and zero, the barrier is skipped on all ranks.
            All ranks must pass the same logical condition to avoid deadlocks.
    """
    with annotate_barrier(pg_name=pg_name):
        _triton_symm_mem_barrier_multimem[(1,)](
            mc_flag_ptr=mc_flag_ptr,
            local_flag_ptr=local_flag_ptr,
            channel=channel,
            conditional_execution_ptr=conditional_execution,
            world_size=n_ranks,
            num_warps=1,
            **pdl_launch_kwargs(),
        )
