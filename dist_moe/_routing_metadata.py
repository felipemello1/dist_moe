# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-side routing metadata generation for distributed MoE."""

import dataclasses

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from ._activation_buffer import MEMORY_ALIGNMENT
from ._buffers import (
    _CommunicationBuffers,
    _ROUTING_HEADER_SIZE_BYTES,
    _routing_ids_view,
    SymmetricMemoryBuffer,
)
from .kernels.config import num_sms_per_device
from .kernels.pdl import pdl_launch_kwargs
from .kernels.triton.dist_dispatch_routing_kernel import (
    _triton_clear_decode_routing_state,
    _triton_decode_dispatch_routing,
    _triton_dispatch_routing_combine,
    _triton_dispatch_routing_prefix,
    _triton_dispatch_routing_split,
    _triton_dist_gather_metadata as _gather_routing_metadata_kernel,
)

# Fused decode routing gives each source rank one CTA per
# _DECODE_ROUTING_SLOTS_PER_CTA route slots, at 16 warps (two slots per thread);
# larger per-CTA tiers spill heavily. _DECODE_ROUTING_MAX_SLOTS (1024 tokens at
# top-8) covers decode; larger prefill calls take the general routing pipeline.
# _DECODE_ROUTING_MAX_EXPERTS bounds the per-CTA expert arrays (no spills at 512).
_DECODE_ROUTING_SLOTS_PER_CTA = 1024
_DECODE_ROUTING_MAX_SLOTS = 8192
_DECODE_ROUTING_MAX_WARPS = 16
_DECODE_ROUTING_MAX_EXPERTS = 512
_DECODE_ROUTING_CLEAR_BLOCK_SIZE = 256
_DECODE_COUNTER_WORKSPACE_COUNT = 2
# The 128-thread, 32-register compactor can keep 16 CTAs resident per SM.
_COMPACT_CTAS_PER_SM = 16
_DISPATCH_VECTOR_ALIGNMENT_BYTES = 16


@dataclasses.dataclass(frozen=True)
class DistDispatchRoutingOutput:
    """Output of the distributed dispatch routing kernel.

    This dataclass contains all information needed for distributed grouped GEMM,
    including routing indices and symmetric memory buffers for communication.

    When ``dist_dispatch_routing`` is called with ``m_multiple_of`` set,
    per-expert token counts in ``num_tokens_per_local_experts`` are rounded
    up to a multiple of that value, and the corresponding extra entries in
    ``bwd_gather_ptrs`` / ``fwd_gather_ptrs`` / ``scatter_ptrs`` hold the
    sentinel value 0 ("skip" — gathers produce zero rows and scatters no-op).
    Entries beyond that initialized prefix are undefined unless the caller
    explicitly clears a larger static-capacity tail.
    """

    # [E // EP] - Number of tokens assigned to each local expert (padded when
    # `m_multiple_of` is set)
    num_tokens_per_local_experts: torch.Tensor
    # [EP] - Number of rows routed to each destination rank, including optional
    # per-expert padding
    num_tokens_per_rank: torch.Tensor
    # [T'] - Gather pointers for backward dispatch gather (topk-expanded, [T, K, D] layout)
    bwd_gather_ptrs: torch.Tensor
    # [T'] - Gather pointers for forward dispatch gather (non-topk-expanded, [T, D] layout)
    fwd_gather_ptrs: torch.Tensor
    # [T'] - Scatter pointers for writing output tokens back to peer GPUs
    scatter_ptrs: torch.Tensor


@dataclasses.dataclass(frozen=True)
class PreparedDecodeRouting:
    """Own decode routing state prepared before peer publication completes.

    ``output`` and the counter tensors are rank-local GPU metadata. The three
    buffer handles keep the symmetric allocations alive while their pointer
    tables may name local or peer HBM. ``local_rank``/``world_size`` and the
    static shape fields describe the launch that may consume this state after
    the caller-issued barrier; no field requires a host read of routing data.
    """

    output: DistDispatchRoutingOutput
    routing_buffer: object
    dispatch_buffer: object
    combine_buffer: object
    expert_counters: torch.Tensor
    completed_ctas: torch.Tensor
    local_rank: int
    world_size: int
    num_tokens: int
    topk: int
    num_experts: int
    num_local_experts: int
    dispatch_stride_bytes: int
    scatter_stride_bytes: int
    m_multiple_of: int
    max_num_recv_tokens: int | None = None


def to_tl_dtype(dtype: torch.dtype) -> tl.dtype:
    """Map a supported PyTorch dtype to its Triton scalar dtype.

    Args:
        dtype: PyTorch scalar dtype.

    Returns:
        Corresponding Triton dtype.

    Raises:
        NotImplementedError: If ``dtype`` is unsupported.
    """
    match dtype:
        case torch.int16:
            return tl.int16
        case torch.int32:
            return tl.int32
        case torch.int64:
            return tl.int64
        case torch.uint16:
            return tl.uint16
        case torch.float32:
            return tl.float32
        case torch.bfloat16:
            return tl.bfloat16
        case torch.float16:
            return tl.float16
        case _:
            raise NotImplementedError(f"No support for {dtype=}!")


def _routing_ptrs_size(
    *,
    world_size: int,
    num_tokens: int,
    topk: int,
    num_local_experts: int,
    m_multiple_of: int,
) -> int:
    """Return the maximum pointer count implied by routing topology.

    Args:
        world_size: Expert-parallel group size.
        num_tokens: Local token count.
        topk: Routes per token.
        num_local_experts: Experts owned by this rank.
        m_multiple_of: Per-expert row padding multiple.

    Returns:
        Required routing-pointer capacity.
    """
    if m_multiple_of <= 0:
        return world_size * num_local_experts * num_tokens

    max_unpadded = world_size * num_tokens * min(topk, num_local_experts)
    max_padded = max_unpadded + num_local_experts * (m_multiple_of - 1)
    return ((max_padded + m_multiple_of - 1) // m_multiple_of) * m_multiple_of


def _routing_ptrs_capacity(
    *,
    world_size: int,
    num_tokens: int,
    topk: int,
    num_local_experts: int,
    m_multiple_of: int,
    max_num_recv_tokens: int | None,
    pointer_capacity_multiple_of: int = 1,
) -> int:
    """Return pointer storage covering topology and an optional row ceiling.

    Args:
        world_size: Expert-parallel group size.
        num_tokens: Local token count.
        topk: Routes per token.
        num_local_experts: Experts owned by this rank.
        m_multiple_of: Per-expert row padding multiple.
        max_num_recv_tokens: Optional configured receive-row ceiling.
        pointer_capacity_multiple_of: Physical allocation row alignment. It
            does not change the logical overflow guard.

    Returns:
        Required routing-pointer capacity.
    """
    required = _routing_ptrs_size(
        world_size=world_size,
        num_tokens=num_tokens,
        topk=topk,
        num_local_experts=num_local_experts,
        m_multiple_of=m_multiple_of,
    )
    if pointer_capacity_multiple_of <= 0:
        raise ValueError("pointer_capacity_multiple_of must be positive")
    capacity = max(required, max_num_recv_tokens or 0)
    return (
        (capacity + pointer_capacity_multiple_of - 1)
        // pointer_capacity_multiple_of
        * pointer_capacity_multiple_of
    )


def _is_power_of_two(value: int) -> bool:
    """Return whether a positive integer is a power of two.

    Args:
        value: Integer to inspect.

    Returns:
        Whether ``value`` is a positive power of two.
    """
    return value > 0 and value & (value - 1) == 0


def _normalize_m_multiple_of(m_multiple_of: int | None) -> int:
    """Normalize an optional expert-row padding multiple.

    Args:
        m_multiple_of: Requested row multiple, or ``None`` for no padding.

    Returns:
        Zero or a positive power-of-two multiple.

    Raises:
        ValueError: If the requested multiple is invalid.
    """
    if m_multiple_of is None:
        return 0
    if not _is_power_of_two(m_multiple_of):
        raise ValueError(
            f"m_multiple_of must be a positive power of two, got {m_multiple_of}"
        )
    return m_multiple_of


def _validate_routing_strides(
    *,
    dispatch_stride_bytes: int,
    scatter_stride_bytes: int,
) -> None:
    """Validate vector alignment for communication rows.

    Args:
        dispatch_stride_bytes: Bytes in one dispatched row.
        scatter_stride_bytes: Bytes in one combined row.

    Raises:
        ValueError: If either stride violates kernel alignment.
    """
    if scatter_stride_bytes % MEMORY_ALIGNMENT != 0:
        raise ValueError(
            f"scatter_stride_bytes={scatter_stride_bytes} must be "
            f"{MEMORY_ALIGNMENT}-byte aligned"
        )
    if dispatch_stride_bytes % _DISPATCH_VECTOR_ALIGNMENT_BYTES != 0:
        raise ValueError(
            f"dispatch_stride_bytes={dispatch_stride_bytes} must be "
            f"{_DISPATCH_VECTOR_ALIGNMENT_BYTES}-byte aligned"
        )


def _can_use_decode_routing(
    *,
    use_low_latency: bool,
    expert_id_offset: int | None,
    generate_bwd_gather_ptrs: bool,
    world_size: int,
    num_tokens: int,
    topk: int,
    num_experts: int,
) -> bool:
    """Return whether the bounded direct decode routing path is valid.

    Args:
        use_low_latency: Whether the caller selected decode routing.
        expert_id_offset: Optional contiguous expert-ID range offset.
        generate_bwd_gather_ptrs: Whether backward gather pointers are needed.
        world_size: Expert-parallel group size.
        num_tokens: Local token count.
        topk: Routes per token.
        num_experts: Global expert count.

    Returns:
        Whether direct decode routing supports the request.
    """
    return (
        use_low_latency
        and expert_id_offset is None
        and not generate_bwd_gather_ptrs
        and num_tokens * topk <= _DECODE_ROUTING_MAX_SLOTS
        and num_experts <= _DECODE_ROUTING_MAX_EXPERTS
        and _is_power_of_two(world_size)
        # The cooperative launch needs every CTA resident at once.
        and world_size * _decode_routing_num_chunks(num_tokens * topk)
        <= num_sms_per_device()
    )


def _decode_routing_num_chunks(num_slots: int) -> int:
    """Return the CTAs per source rank of a decode routing launch."""
    return max(1, triton.cdiv(num_slots, _DECODE_ROUTING_SLOTS_PER_CTA))


def low_latency_decode_routing_capacity(
    *,
    world_size: int,
    num_tokens: int,
    topk: int,
    num_experts: int,
    expert_id_offset: int | None,
    m_multiple_of: int | None,
) -> int | None:
    """Return the direct decode pointer capacity when supported.

    Args:
        world_size: Expert-parallel group size.
        num_tokens: Local token count.
        topk: Routes per token.
        num_experts: Global expert count.
        expert_id_offset: Optional contiguous expert-ID range offset.
        m_multiple_of: Optional per-expert row padding multiple.

    Returns:
        Pointer capacity, or ``None`` when direct routing is unsupported.
    """
    if num_experts % world_size != 0:
        return None
    normalized_m_multiple_of = _normalize_m_multiple_of(m_multiple_of)
    if not _can_use_decode_routing(
        use_low_latency=True,
        expert_id_offset=expert_id_offset,
        generate_bwd_gather_ptrs=False,
        world_size=world_size,
        num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
    ):
        return None
    num_local_experts = num_experts // world_size
    return _routing_ptrs_size(
        world_size=world_size,
        num_tokens=num_tokens,
        topk=topk,
        num_local_experts=num_local_experts,
        m_multiple_of=normalized_m_multiple_of,
    )


def _decode_dispatch_routing(
    *,
    routing_buffer: SymmetricMemoryBuffer,
    dispatch_buffer: SymmetricMemoryBuffer,
    combine_buffer: SymmetricMemoryBuffer,
    device: torch.device,
    local_rank: int,
    world_size: int,
    num_tokens: int,
    topk: int,
    num_experts: int,
    dispatch_stride_bytes: int,
    scatter_stride_bytes: int,
    m_multiple_of: int,
    max_num_recv_tokens: int | None,
    pointer_capacity_multiple_of: int,
    counter_workspaces: tuple[torch.Tensor, ...],
) -> DistDispatchRoutingOutput:
    """Allocate, clear, and execute one direct decode routing launch.

    Args:
        routing_buffer: Symmetric expert-ID publication buffer.
        dispatch_buffer: Symmetric activation dispatch buffer.
        combine_buffer: Symmetric activation combine buffer.
        device: CUDA device for routing outputs.
        local_rank: Rank within the expert-parallel group.
        world_size: Expert-parallel group size.
        num_tokens: Local token count.
        topk: Routes per token.
        num_experts: Global expert count.
        dispatch_stride_bytes: Bytes in one dispatched row.
        scatter_stride_bytes: Bytes in one combined row.
        m_multiple_of: Per-expert row padding multiple.
        max_num_recv_tokens: Optional configured receive-row ceiling.
        pointer_capacity_multiple_of: Physical pointer-storage row alignment.
        counter_workspaces: Grouped-GEMM counter tensors to zero in the clear launch.

    Returns:
        Routing counts and gather/scatter pointer tables.
    """
    prepared = _allocate_prepared_decode_routing(
        routing_buffer=routing_buffer,
        dispatch_buffer=dispatch_buffer,
        combine_buffer=combine_buffer,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
        num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        dispatch_stride_bytes=dispatch_stride_bytes,
        scatter_stride_bytes=scatter_stride_bytes,
        m_multiple_of=m_multiple_of,
        max_num_recv_tokens=max_num_recv_tokens,
        pointer_capacity_multiple_of=pointer_capacity_multiple_of,
    )
    _clear_prepared_decode_routing(prepared, counter_workspaces=counter_workspaces)
    return launch_prepared_decode_routing(prepared)


def _allocate_prepared_decode_routing(
    *,
    routing_buffer: SymmetricMemoryBuffer,
    dispatch_buffer: SymmetricMemoryBuffer,
    combine_buffer: SymmetricMemoryBuffer,
    device: torch.device,
    local_rank: int,
    world_size: int,
    num_tokens: int,
    topk: int,
    num_experts: int,
    dispatch_stride_bytes: int,
    scatter_stride_bytes: int,
    m_multiple_of: int,
    max_num_recv_tokens: int | None,
    pointer_capacity_multiple_of: int = 1,
) -> PreparedDecodeRouting:
    """Allocate fixed-capacity outputs and counters for decode routing.

    Args:
        routing_buffer: Symmetric expert-ID publication buffer.
        dispatch_buffer: Symmetric activation dispatch buffer.
        combine_buffer: Symmetric activation combine buffer.
        device: CUDA device for routing outputs.
        local_rank: Rank within the expert-parallel group.
        world_size: Expert-parallel group size.
        num_tokens: Local token count.
        topk: Routes per token.
        num_experts: Global expert count.
        dispatch_stride_bytes: Bytes in one dispatched row.
        scatter_stride_bytes: Bytes in one combined row.
        m_multiple_of: Per-expert row padding multiple.
        max_num_recv_tokens: Optional configured receive-row ceiling.
        pointer_capacity_multiple_of: Physical pointer-storage row alignment.

    Returns:
        Reusable decode-routing state.
    """
    num_local_experts = num_experts // world_size
    ptrs_size = _routing_ptrs_capacity(
        world_size=world_size,
        num_tokens=num_tokens,
        topk=topk,
        num_local_experts=num_local_experts,
        m_multiple_of=m_multiple_of,
        max_num_recv_tokens=max_num_recv_tokens,
        pointer_capacity_multiple_of=pointer_capacity_multiple_of,
    )
    expert_counters = torch.empty((num_experts,), dtype=torch.int32, device=device)
    completed_ctas = torch.empty((1,), dtype=torch.int32, device=device)
    num_tokens_per_local_experts = torch.empty(
        (num_local_experts,), dtype=torch.int32, device=device
    )
    num_tokens_per_rank = torch.empty((world_size,), dtype=torch.int32, device=device)
    fwd_gather_ptrs = torch.empty((ptrs_size,), dtype=torch.int64, device=device)
    scatter_ptrs = torch.empty((ptrs_size,), dtype=torch.int64, device=device)

    output = DistDispatchRoutingOutput(
        num_tokens_per_local_experts=num_tokens_per_local_experts,
        num_tokens_per_rank=num_tokens_per_rank,
        bwd_gather_ptrs=torch.empty((0,), dtype=torch.int64, device=device),
        fwd_gather_ptrs=fwd_gather_ptrs,
        scatter_ptrs=scatter_ptrs,
    )
    return PreparedDecodeRouting(
        output=output,
        routing_buffer=routing_buffer,
        dispatch_buffer=dispatch_buffer,
        combine_buffer=combine_buffer,
        expert_counters=expert_counters,
        completed_ctas=completed_ctas,
        local_rank=local_rank,
        world_size=world_size,
        num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        dispatch_stride_bytes=dispatch_stride_bytes,
        scatter_stride_bytes=scatter_stride_bytes,
        m_multiple_of=m_multiple_of,
        max_num_recv_tokens=max_num_recv_tokens,
    )


def _clear_prepared_decode_routing(
    prepared: PreparedDecodeRouting,
    *,
    expert_ids: torch.Tensor | None = None,
    local_routing_buffer: torch.Tensor | None = None,
    counter_workspaces: tuple[torch.Tensor, ...] = (),
) -> None:
    """Clear reusable decode outputs and optional fused counter workspaces.

    Args:
        prepared: Reusable decode-routing state.
        expert_ids: Optional source IDs to publish while clearing.
        local_routing_buffer: Optional local publication destination.
        counter_workspaces: Optional fused-kernel counter tensors.

    Raises:
        ValueError: If publication arguments or counter workspaces are invalid.
    """
    if len(counter_workspaces) > _DECODE_COUNTER_WORKSPACE_COUNT:
        raise ValueError(
            "decode routing can clear at most "
            f"{_DECODE_COUNTER_WORKSPACE_COUNT} counter workspaces"
        )
    copy_expert_ids = expert_ids is not None
    if copy_expert_ids != (local_routing_buffer is not None):
        raise ValueError(
            "expert_ids and local_routing_buffer must be provided together"
        )
    for workspace in counter_workspaces:
        if (
            workspace.dtype != torch.int32
            or workspace.device != prepared.expert_counters.device
            or not workspace.is_contiguous()
        ):
            raise ValueError(
                "decode counter workspaces must be contiguous int32 tensors "
                "on the routing device"
            )
    workspace_0 = (
        counter_workspaces[0]
        if len(counter_workspaces) > 0
        else prepared.expert_counters
    )
    workspace_1 = (
        counter_workspaces[1]
        if len(counter_workspaces) > 1
        else prepared.expert_counters
    )
    workspace_0_size = counter_workspaces[0].numel() if counter_workspaces else 0
    workspace_1_size = (
        counter_workspaces[1].numel() if len(counter_workspaces) > 1 else 0
    )
    ptrs_size = prepared.output.fwd_gather_ptrs.numel()
    clear_size = max(
        ptrs_size,
        prepared.num_experts,
        prepared.num_local_experts,
        prepared.world_size,
        workspace_0_size,
        workspace_1_size,
    )
    expert_ids_ptr = expert_ids if expert_ids is not None else prepared.expert_counters
    routing_output_ptr = (
        local_routing_buffer
        if local_routing_buffer is not None
        else prepared.expert_counters
    )
    clear_grid = (triton.cdiv(clear_size, _DECODE_ROUTING_CLEAR_BLOCK_SIZE),)
    _triton_clear_decode_routing_state[clear_grid](
        expert_ids_ptr=expert_ids_ptr,
        routing_output_ptr=routing_output_ptr,
        expert_counters_ptr=prepared.expert_counters,
        completed_ctas_ptr=prepared.completed_ctas,
        num_tokens_per_local_experts_ptr=(prepared.output.num_tokens_per_local_experts),
        num_tokens_per_rank_ptr=prepared.output.num_tokens_per_rank,
        fwd_gather_ptrs_ptr=prepared.output.fwd_gather_ptrs,
        scatter_ptrs_ptr=prepared.output.scatter_ptrs,
        counter_workspace_0_ptr=workspace_0,
        counter_workspace_1_ptr=workspace_1,
        num_routing_slots=prepared.num_tokens * prepared.topk,
        expert_ids_stride_0=0 if expert_ids is None else expert_ids.stride(0),
        expert_ids_stride_1=0 if expert_ids is None else expert_ids.stride(1),
        topk=prepared.topk,
        num_experts=prepared.num_experts,
        num_local_experts=prepared.num_local_experts,
        world_size=prepared.world_size,
        ptrs_size=ptrs_size,
        counter_workspace_0_size=workspace_0_size,
        counter_workspace_1_size=workspace_1_size,
        COPY_EXPERT_IDS=copy_expert_ids,
        BLOCK_SIZE=_DECODE_ROUTING_CLEAR_BLOCK_SIZE,
        **pdl_launch_kwargs(),
    )


def launch_prepared_decode_routing(
    prepared: PreparedDecodeRouting,
) -> DistDispatchRoutingOutput:
    """Launch direct routing after peer input publication is complete.

    Args:
        prepared: Cleared, reusable decode-routing state.

    Returns:
        Routing counts and gather/scatter pointer tables.
    """
    num_slots = prepared.num_tokens * prepared.topk
    num_chunks = _decode_routing_num_chunks(num_slots)
    block_slots = min(triton.next_power_of_2(num_slots), _DECODE_ROUTING_SLOTS_PER_CTA)
    _triton_decode_dispatch_routing[(prepared.world_size * num_chunks,)](
        routing_buffer_ptrs=prepared.routing_buffer.buffer_ptrs_tensor,
        gather_buffer_ptrs=prepared.dispatch_buffer.buffer_ptrs_tensor,
        scatter_buffer_ptrs=prepared.combine_buffer.buffer_ptrs_tensor,
        expert_counters_ptr=prepared.expert_counters,
        completed_ctas_ptr=prepared.completed_ctas,
        num_tokens_per_local_experts_ptr=(prepared.output.num_tokens_per_local_experts),
        num_tokens_per_rank_ptr=prepared.output.num_tokens_per_rank,
        fwd_gather_ptrs_ptr=prepared.output.fwd_gather_ptrs,
        scatter_ptrs_ptr=prepared.output.scatter_ptrs,
        local_rank=prepared.local_rank,
        num_tokens=prepared.num_tokens,
        gather_stride_bytes=prepared.dispatch_stride_bytes,
        scatter_stride_bytes=prepared.scatter_stride_bytes,
        world_size=prepared.world_size,
        num_experts=prepared.num_experts,
        num_local_experts=prepared.num_local_experts,
        topk=prepared.topk,
        m_multiple_of=prepared.m_multiple_of,
        routing_header_size_bytes=_ROUTING_HEADER_SIZE_BYTES,
        max_num_recv_tokens=(
            0 if prepared.max_num_recv_tokens is None else prepared.max_num_recv_tokens
        ),
        has_max_num_recv_tokens=prepared.max_num_recv_tokens is not None,
        BLOCK_SLOTS=block_slots,
        BLOCK_EXPERTS=triton.next_power_of_2(prepared.num_experts),
        BLOCK_LOCAL_EXPERTS=triton.next_power_of_2(prepared.num_local_experts),
        BLOCK_WORLD_SIZE=triton.next_power_of_2(prepared.world_size),
        NUM_CHUNKS=num_chunks,
        num_warps=max(1, min(block_slots // 32, _DECODE_ROUTING_MAX_WARPS)),
        launch_cooperative_grid=True,
        **pdl_launch_kwargs(),
    )
    return prepared.output


def prepare_decode_routing(
    *,
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
    group: dist.ProcessGroup,
    comm_buffer: _CommunicationBuffers,
    expert_id_offset: int | None,
    m_multiple_of: int | None,
    dispatch_stride_bytes: int,
    counter_workspaces: tuple[torch.Tensor, ...],
    max_num_recv_tokens: int | None = None,
) -> PreparedDecodeRouting | None:
    """Publish expert IDs and clear decode state before the peer barrier.

    Args:
        tokens: Local activation rows with shape ``[T, D]``.
        expert_ids: Global expert IDs with shape ``[T, K]``.
        num_experts: Global expert count.
        group: Expert-parallel process group.
        comm_buffer: DistMoE communication-buffer owner.
        expert_id_offset: Optional contiguous expert-ID range offset.
        m_multiple_of: Optional per-expert row padding multiple.
        dispatch_stride_bytes: Bytes in one dispatched row.
        counter_workspaces: Optional fused-kernel counter tensors.
        max_num_recv_tokens: Optional configured receive-row ceiling.

    Returns:
        Prepared direct-routing state, or ``None`` when unsupported.

    Raises:
        ValueError: If token and routing metadata are inconsistent.
    """
    local_rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    num_tokens, dim = tokens.shape
    num_tokens_, topk = expert_ids.shape
    if num_tokens != num_tokens_:
        raise ValueError("tokens and expert_ids must have the same token count")
    if expert_ids.device != tokens.device or not expert_ids.is_cuda:
        raise ValueError("expert_ids must be a CUDA tensor on the token device")
    if (
        low_latency_decode_routing_capacity(
            world_size=world_size,
            num_tokens=num_tokens,
            topk=topk,
            num_experts=num_experts,
            expert_id_offset=expert_id_offset,
            m_multiple_of=m_multiple_of,
        )
        is None
    ):
        return None
    scatter_stride_bytes = dim * tokens.element_size()
    _validate_routing_strides(
        dispatch_stride_bytes=dispatch_stride_bytes,
        scatter_stride_bytes=scatter_stride_bytes,
    )
    m_multiple_of_constexpr = _normalize_m_multiple_of(m_multiple_of)

    routing_buffer = comm_buffer.routing
    prepared = _allocate_prepared_decode_routing(
        routing_buffer=routing_buffer,
        dispatch_buffer=comm_buffer.dispatch,
        combine_buffer=comm_buffer.combine,
        device=expert_ids.device,
        local_rank=local_rank,
        world_size=world_size,
        num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        dispatch_stride_bytes=dispatch_stride_bytes,
        scatter_stride_bytes=scatter_stride_bytes,
        m_multiple_of=m_multiple_of_constexpr,
        max_num_recv_tokens=max_num_recv_tokens,
    )
    local_routing_buffer = _routing_ids_view(
        routing_buffer, local_rank, (expert_ids.shape[0], expert_ids.shape[1])
    )
    _clear_prepared_decode_routing(
        prepared,
        expert_ids=expert_ids,
        local_routing_buffer=local_routing_buffer,
        counter_workspaces=counter_workspaces,
    )
    return prepared


def dist_dispatch_routing(
    tokens: torch.Tensor,  # [T, D]
    expert_ids: torch.Tensor,  # [T, K]
    num_experts: int,
    group: dist.ProcessGroup,
    comm_buffer: _CommunicationBuffers,
    *,
    expert_id_offset: int | None = None,
    m_multiple_of: int | None = None,
    max_num_recv_tokens: int | None = None,
    dispatch_stride_bytes: int | None = None,
    bwd_dispatch_stride_bytes: int | None = None,
    generate_bwd_gather_ptrs: bool = True,
    use_low_latency: bool = False,
    pointer_capacity_multiple_of: int = 1,
    counter_workspaces: tuple[torch.Tensor, ...] = (),
) -> DistDispatchRoutingOutput:
    """Compute routing indices for distributed grouped GEMM with expert parallelism.

    This function performs distributed routing for Mixture of Experts (MoE) layers where
    experts are partitioned across multiple GPUs. It computes gather and scatter indices
    needed to route tokens to the correct experts and aggregate results.

    Every rank must publish its expert IDs into ``comm_buffer.routing`` and
    complete a group-wide GPU barrier before calling this function.

    Implementation uses a 3-phase approach to avoid shared memory overflow:

    Phase 1 (Gather): Copy expert IDs from NVLink to HBM
        - Uses large tiles (1024 tokens) for efficient bandwidth utilization
        - Minimal shared memory usage (simple memcpy operations)
        - Each rank gathers expert IDs from all other ranks via NVLink
        - Output: gathered_expert_ids buffer in HBM [world_size, T, K]

    Phase 2 (Routing): Compute gather/scatter indices from HBM
        - Uses smaller tiles (128 tokens) to reduce shared memory pressure
        - Reads expert IDs from local HBM (gathered in Phase 1)
        - Computes which tokens should be routed to local experts
        - Generates per-rank gather/scatter pointers
        - Shared memory: ~131 KB (vs 262 KB in original single-pass)

    Phase 3 (Combine): Compact sparse indices into dense arrays
        - Aggregates token counts across all ranks
        - Compacts per-rank indices into contiguous output arrays
        - Ensures tokens are grouped by expert for efficient GEMM execution

    Args:
        tokens: Input tokens [T, D] where T is number of tokens, D is hidden dimension
        expert_ids: TopK expert assignments [T, K] where K is number of experts per token
        num_experts: Total number of experts across all ranks
        group: Process group for distributed communication
        expert_id_offset: Rank-specific starting global expert ID for the current rank.
              When provided, this rank claims experts
              [expert_id_offset, expert_id_offset + num_local_experts) instead of
              [local_rank * num_local_experts, (local_rank + 1) * num_local_experts).
              The public flat expert-parallel path leaves this unset and derives
              ownership from the current rank.
        m_multiple_of: Optional alignment hint that rounds each local expert's
              token count up to a multiple of this value. Must be a positive
              power of 2 (typically matching ``blockscaled_grouped_gemm``'s
              ``m_multiple_of``). Padding slots in the returned gather/scatter
              pointer arrays hold the sentinel value 0; consumer kernels must
              treat addr==0 as "skip"
              (gather → fill output row with zeros, scatter → no-op).
              Defaults to ``None`` (no padding).
        max_num_recv_tokens: Optional static capacity for the padded received-token
              count. The routing combine kernel traps on device if the actual
              count exceeds this value.
        dispatch_stride_bytes: Per-token byte stride of the forward dispatch
              payload. Defaults to the dense token stride.
        bwd_dispatch_stride_bytes: Per-topk-token byte stride of the backward
              gradient payload. Defaults to the dense token stride. This may
              differ from ``dispatch_stride_bytes`` when forward uses a packed
              block-scaled payload and backward writes native gradients.
        generate_bwd_gather_ptrs: When False (inference forward), the
              topk-expanded backward gather pointers are not produced — the
              split/combine kernels skip those stores and ``bwd_gather_ptrs`` is
              returned as an empty tensor. Backward needs them, so keep the
              default ``True`` for training.
        use_low_latency: Select the two-launch decode path when its bounded
              shape and synchronization requirements are satisfied. Unsupported
              configurations fall back to the general three-phase path.
        pointer_capacity_multiple_of: Physical pointer-storage row alignment
              for a capacity-shaped consumer. The configured receive-row
              overflow guard is unchanged.
        counter_workspaces: Up to two int32 grouped-GEMM counter tensors this
              call zeroes, so the GEMM launches need no memset of their own.

    Returns:
        DistDispatchRoutingOutput containing:
            - num_tokens_per_local_experts: Token count per local expert [num_local_experts]
            - bwd_gather_ptrs: Indices for backward gather (topk-expanded) [T']
            - fwd_gather_ptrs: Indices for forward gather (non-topk-expanded) [T']
            - scatter_ptrs: Indices for scattering output tokens [T']

    Note:
        - Experts are evenly distributed across ranks (num_experts % world_size == 0)
        - Uses symmetric memory for low-latency cross-GPU communication
        - A token never selects the same expert multiple times in its topk choices
        - The 3-phase design trades HBM bandwidth for reduced SMEM usage
    """
    local_rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    # Routing buffer is int16 to halve NVLink traffic. Bound num_experts so
    # raw expert IDs and the offset-shifted IDs in the kernel both fit.
    assert num_experts <= torch.iinfo(torch.int16).max, (
        f"num_experts={num_experts} exceeds int16 max "
        f"({torch.iinfo(torch.int16).max}); routing buffer cannot represent "
        f"all expert IDs."
    )

    routing_buffer = comm_buffer.routing
    dispatch_buffer = comm_buffer.dispatch
    combine_buffer = comm_buffer.combine

    device = expert_ids.device

    num_tokens, dim = tokens.shape
    num_tokens_, topk = expert_ids.shape
    assert num_tokens == num_tokens_

    scatter_stride_bytes = dim * tokens.element_size()
    dispatch_stride_bytes = (
        scatter_stride_bytes if dispatch_stride_bytes is None else dispatch_stride_bytes
    )
    bwd_dispatch_stride_bytes = (
        scatter_stride_bytes
        if bwd_dispatch_stride_bytes is None
        else bwd_dispatch_stride_bytes
    )
    _validate_routing_strides(
        dispatch_stride_bytes=dispatch_stride_bytes,
        scatter_stride_bytes=scatter_stride_bytes,
    )
    _validate_routing_strides(
        dispatch_stride_bytes=bwd_dispatch_stride_bytes,
        scatter_stride_bytes=scatter_stride_bytes,
    )

    assert num_experts % world_size == 0
    num_local_experts = num_experts // world_size

    m_multiple_of_constexpr = _normalize_m_multiple_of(m_multiple_of)

    if _can_use_decode_routing(
        use_low_latency=use_low_latency,
        expert_id_offset=expert_id_offset,
        generate_bwd_gather_ptrs=generate_bwd_gather_ptrs,
        world_size=world_size,
        num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
    ):
        return _decode_dispatch_routing(
            routing_buffer=routing_buffer,
            dispatch_buffer=dispatch_buffer,
            combine_buffer=combine_buffer,
            device=device,
            local_rank=local_rank,
            world_size=world_size,
            num_tokens=num_tokens,
            topk=topk,
            num_experts=num_experts,
            dispatch_stride_bytes=dispatch_stride_bytes,
            scatter_stride_bytes=scatter_stride_bytes,
            m_multiple_of=m_multiple_of_constexpr,
            max_num_recv_tokens=max_num_recv_tokens,
            pointer_capacity_multiple_of=pointer_capacity_multiple_of,
            counter_workspaces=counter_workspaces,
        )
    # The general path does not clear them in a launch of its own; zero them here
    # so the GEMM launches can still skip their memsets.
    for workspace in counter_workspaces:
        workspace.zero_()

    # Plan the split grid before the gather launch so the gather CTAs can also
    # initialize the counters consumed by the split kernel.
    block_size_tk = 256
    block_size_k = triton.next_power_of_2(topk)
    block_size_t = max(1, block_size_tk // block_size_k)

    max_ctas = 512 if generate_bwd_gather_ptrs else 256
    assert max_ctas % world_size == 0
    max_workers = max_ctas // world_size
    num_workers = max(
        1, min(max_workers, (num_tokens + block_size_t - 1) // block_size_t)
    )
    # Keep the worker count power-of-two for the combine kernel's tl.arange.
    num_workers = min(max_workers, triton.next_power_of_2(num_workers))
    num_tokens_per_worker = (num_tokens + num_workers - 1) // num_workers

    per_rank_token_counts = torch.empty(
        (world_size, num_local_experts, num_workers), dtype=torch.int32, device=device
    )
    num_tokens_per_rank = torch.empty((num_experts,), dtype=torch.int32, device=device)

    # ========================================================================
    # PHASE 1: Gather expert IDs from NVLink to HBM
    # ========================================================================
    # Use large tiles (1024 tokens) for efficient NVLink bandwidth utilization
    # This phase has minimal shared memory requirements (simple memcpy)

    gather_block_size_t = 1024  # Large tiles for Phase 1
    gather_num_workers = max(
        1, (num_tokens + gather_block_size_t - 1) // gather_block_size_t
    )

    # Allocate HBM buffer to store gathered expert IDs from all ranks. Use
    # int16 to match the routing buffer dtype: the gather kernel is a pure
    # memcpy, no widening needed, and downstream `_triton_dispatch_routing_split`
    # handles int16 since it does only equality / divmod against `num_experts`.
    gathered_expert_ids = torch.empty(
        (world_size, num_tokens, topk), dtype=torch.int16, device=device
    )

    # Launch Phase 1: Gather metadata from NVLink to HBM
    grid_gather = (world_size, gather_num_workers, 1)
    _gather_routing_metadata_kernel[grid_gather](
        routing_buffer_ptrs=routing_buffer.buffer_ptrs_tensor,
        gathered_expert_ids_ptr=gathered_expert_ids,
        per_rank_token_cnts_ptr=per_rank_token_counts,
        num_tokens_per_rank_ptr=num_tokens_per_rank,
        world_size=world_size,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        block_size_t=gather_block_size_t,
        num_gather_workers=gather_num_workers,
        split_num_workers=num_workers,
        index_ty=tl.int16,
        routing_header_size_bytes=_ROUTING_HEADER_SIZE_BYTES,
        BLOCK_K=triton.next_power_of_2(topk),
        BLOCK_WORLD_SIZE=triton.next_power_of_2(world_size),
    )

    # ========================================================================
    # PHASE 2: Compute routing indices from HBM
    # ========================================================================
    # Use smaller tiles (128 tokens) to reduce shared memory pressure
    # Reads expert IDs from local HBM and computes gather/scatter pointers

    # Intermediate pointer tensors are only written at matched token positions.
    # Backward gather pointers are skipped entirely for inference forward: a
    # None buffer flows through the split/combine kernels' ``is not None`` guards.
    per_rank_bwd_gather_ptrs = (
        torch.empty(
            (world_size, num_local_experts, num_workers, num_tokens_per_worker),
            dtype=torch.int64,
            device=device,
        )
        if generate_bwd_gather_ptrs
        else None
    )
    per_rank_fwd_gather_ptrs = torch.empty(
        (world_size, num_local_experts, num_workers, num_tokens_per_worker),
        dtype=torch.int64,
        device=device,
    )
    per_rank_scatter_ptrs = torch.empty(
        (world_size, num_local_experts, num_workers, num_tokens_per_worker),
        dtype=torch.int64,
        device=device,
    )

    # Output tensors - initialized to zero as they are only partially written
    num_tokens_per_local_experts = torch.empty(
        (num_local_experts,), dtype=torch.int32, device=device
    )
    # Each token can select at most one copy of each local expert, so the tight
    # worst-case receive bound uses min(topk, num_local_experts). Round the padded
    # bound itself so its length is a valid static blockscaled GEMM capacity.
    ptrs_size = _routing_ptrs_capacity(
        world_size=world_size,
        num_tokens=num_tokens,
        topk=topk,
        num_local_experts=num_local_experts,
        m_multiple_of=m_multiple_of_constexpr,
        max_num_recv_tokens=max_num_recv_tokens,
        pointer_capacity_multiple_of=pointer_capacity_multiple_of,
    )
    bwd_gather_ptrs = (
        torch.empty((ptrs_size,), dtype=torch.int64, device=device)
        if generate_bwd_gather_ptrs
        else None
    )
    fwd_gather_ptrs = torch.empty((ptrs_size,), dtype=torch.int64, device=device)
    scatter_ptrs = torch.empty((ptrs_size,), dtype=torch.int64, device=device)

    # Launch Phase 2: Process gathered expert IDs with smaller tiles
    grid = (world_size, num_workers, 1)
    effective_expert_offset = (
        local_rank * num_local_experts if expert_id_offset is None else expert_id_offset
    )
    # dest_rank_base: subtracted from expert_id // num_local_experts to get
    # the group-local destination rank. When expert_id_offset is rank-specific,
    # subtract the current local_rank to recover the group base.
    dest_rank_base = effective_expert_offset // num_local_experts - local_rank
    _triton_dispatch_routing_split[grid](
        gathered_expert_ids_ptr=gathered_expert_ids,
        gather_buffer_ptrs=dispatch_buffer.buffer_ptrs_tensor,
        scatter_buffer_ptrs=combine_buffer.buffer_ptrs_tensor,
        per_rank_token_cnts_ptr=per_rank_token_counts,
        per_rank_bwd_gather_ptrs_ptr=per_rank_bwd_gather_ptrs,
        gather_stride_bytes=dispatch_stride_bytes,
        bwd_gather_stride_bytes=bwd_dispatch_stride_bytes,
        per_rank_fwd_gather_ptrs_ptr=per_rank_fwd_gather_ptrs,
        per_rank_scatter_ptrs_ptr=per_rank_scatter_ptrs,
        scatter_stride_bytes=scatter_stride_bytes,
        num_tokens_per_rank_ptr=num_tokens_per_rank,
        expert_id_offset=effective_expert_offset,
        dest_rank_base=dest_rank_base,
        world_size=world_size,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        block_size_t=block_size_t,
        BLOCK_EL=triton.next_power_of_2(num_local_experts),
        BLOCK_K=triton.next_power_of_2(topk),
    )

    # ========================================================================
    # PHASE 3: Combine/compact sparse indices into dense arrays
    # ========================================================================
    # Aggregate token counts and compact per-rank indices into final output

    max_compact_ctas = (
        _COMPACT_CTAS_PER_SM
        * torch.cuda.get_device_properties(device).multi_processor_count
    )
    compact_ctas_per_pair_limit = max(1, max_compact_ctas // num_experts)
    compact_ctas_per_pair = min(
        num_workers,
        1 << (compact_ctas_per_pair_limit.bit_length() - 1),
    )
    assert num_workers % compact_ctas_per_pair == 0
    workers_per_compact_cta = num_workers // compact_ctas_per_pair
    grid = (
        world_size * num_local_experts * compact_ctas_per_pair,
        1,
        1,
    )
    compact_offsets = torch.empty(
        (world_size, num_local_experts, compact_ctas_per_pair),
        dtype=torch.int64,
        device=device,
    )
    _triton_dispatch_routing_prefix[(1, 1, 1)](
        per_rank_token_cnts_ptr=per_rank_token_counts,
        token_cnts_ptr=num_tokens_per_local_experts,
        num_tokens_per_rank_ptr=num_tokens_per_rank,
        compact_offsets_ptr=compact_offsets,
        world_size=world_size,
        num_experts=num_experts,
        num_workers=num_workers,
        compact_ctas_per_pair=compact_ctas_per_pair,
        workers_per_compact_cta=workers_per_compact_cta,
        BLOCK_EL=triton.next_power_of_2(num_local_experts),
        m_multiple_of=m_multiple_of_constexpr,
        max_num_recv_tokens=(0 if max_num_recv_tokens is None else max_num_recv_tokens),
        has_max_num_recv_tokens=max_num_recv_tokens is not None,
    )
    _triton_dispatch_routing_combine[grid](
        per_rank_token_cnts_ptr=per_rank_token_counts,
        per_rank_bwd_gather_ptrs_ptr=per_rank_bwd_gather_ptrs,
        per_rank_fwd_gather_ptrs_ptr=per_rank_fwd_gather_ptrs,
        per_rank_scatter_ptrs_ptr=per_rank_scatter_ptrs,
        token_cnts_ptr=num_tokens_per_local_experts,
        compact_offsets_ptr=compact_offsets,
        bwd_gather_ptrs_ptr=bwd_gather_ptrs,
        fwd_gather_ptrs_ptr=fwd_gather_ptrs,
        scatter_ptrs_ptr=scatter_ptrs,
        world_size=world_size,
        num_experts=num_experts,
        block_size_t=block_size_t,
        num_workers=num_workers,
        num_tokens_per_worker=num_tokens_per_worker,
        compact_ctas_per_pair=compact_ctas_per_pair,
        workers_per_compact_cta=workers_per_compact_cta,
        m_multiple_of=m_multiple_of_constexpr,
    )

    return DistDispatchRoutingOutput(
        num_tokens_per_local_experts=num_tokens_per_local_experts,
        num_tokens_per_rank=num_tokens_per_rank[:world_size],
        bwd_gather_ptrs=(
            bwd_gather_ptrs
            if bwd_gather_ptrs is not None
            else torch.empty((0,), dtype=torch.int64, device=device)
        ),
        fwd_gather_ptrs=fwd_gather_ptrs,
        scatter_ptrs=scatter_ptrs,
    )
