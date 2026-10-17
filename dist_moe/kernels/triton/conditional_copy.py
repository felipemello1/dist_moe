# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Conditional copy kernel for MoE activation saving/loading.

This kernel conditionally copies tensors based on a device-side condition tensor:
- Forward save: copy x or h3 to buffer based on recompute decision
- Backward load: copy from buffer to output tensor when not recomputing

The ternary API supports:
- condition=True: execute lhs path (copy lhs tensor to/from buffer at lhs_offset)
- condition=False: execute rhs path (copy rhs tensor to/from buffer at rhs_offset)
- Either lhs or rhs can be None to skip that path
"""

import torch
import triton
import triton.language as tl

try:
    from torch._native.const_tensor_wrapper import (
        ConstTensorWrapper as _ReadOnlyTritonTensor,
    )
except ImportError:

    class _ReadOnlyTritonTensor:
        """Backport PyTorch's read-only Triton tensor protocol."""

        __slots__ = ("_tensor",)

        def __init__(self, tensor: torch.Tensor) -> None:
            self._tensor = tensor

        @property
        def dtype(self) -> torch.dtype:
            return self._tensor.dtype

        def data_ptr(self) -> int:
            return torch._C._data_address(self._tensor) + (
                self._tensor.storage_offset() * self._tensor.element_size()
            )


from ..._activation_buffer_planner_kernel import (
    MEMORY_ALIGNMENT_TL_CONSTEXPR,
)
from .._environment import num_sms_per_device
from ..activation_buffer import (
    validate_conditional_execution,
)
from ..pdl import pdl_launch_kwargs, pdl_wait_then_release
from .comm_utils import get_flat_tid
from .device_trap import device_trap_if

_COPY_BLOCK_SIZE = 1024
_COPY_CTAS_PER_SM = 32
_COPY_NUM_WARPS = 4
_COPY_NUM_STAGES = 1


def _copy_launch_config() -> tuple[int, int]:
    """Return the retained persistent copy CTA count and warp count."""
    num_ctas = num_sms_per_device() * _COPY_CTAS_PER_SM
    return num_ctas, _COPY_NUM_WARPS


def _validate_copy_tensor(tensor: torch.Tensor, *, name: str) -> None:
    """Validate the dtype and contiguous layout of an activation copy tensor."""
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.uint8):
        raise TypeError(f"{name} must have dtype float16, bfloat16, or uint8")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _triton_copy_dtype(dtype: torch.dtype):
    """Map a supported copy dtype to its Triton scalar dtype."""
    if dtype is torch.float16:
        return tl.float16
    if dtype is torch.bfloat16:
        return tl.bfloat16
    if dtype is torch.uint8:
        return tl.uint8
    raise TypeError(f"unsupported copy dtype {dtype}")


def _validate_activation_destination(
    *,
    activation_buffer: torch.Tensor,
    activation_offset: torch.Tensor,
    device: torch.device,
) -> None:
    """Validate buffer and device-offset metadata without reading its value."""
    if activation_buffer.dtype is not torch.uint8:
        raise TypeError("activation_buffer must have dtype uint8")
    if activation_buffer.device != device:
        raise ValueError("activation_buffer must be on the input device")
    if activation_offset.dtype is not torch.int64 or activation_offset.numel() != 1:
        raise TypeError("activation_offset must contain one int64 element")
    if activation_offset.device != device:
        raise ValueError("activation_offset must be on the input device")


def _validate_routing_publication(
    *,
    expert_ids: torch.Tensor,
    routing: torch.Tensor,
    routing_token_count_1: torch.Tensor,
    device: torch.device,
) -> None:
    """Validate views used to publish one rank's routing metadata."""
    if expert_ids.dtype not in (torch.int16, torch.int32, torch.int64):
        raise TypeError("expert_ids must have an integer dtype")
    if expert_ids.ndim != 2:
        raise ValueError("expert_ids must be two-dimensional")
    if routing.shape != expert_ids.shape or routing.dtype != torch.int16:
        raise ValueError("routing must match expert_ids shape and have dtype int16")
    if not routing.is_contiguous():
        raise ValueError("routing must be contiguous")
    if (
        routing_token_count_1.shape != (1,)
        or routing_token_count_1.dtype != torch.int32
    ):
        raise ValueError("routing_token_count_1 must be an int32 tensor with shape [1]")
    if not all(
        tensor.device == device
        for tensor in (expert_ids, routing, routing_token_count_1)
    ):
        raise ValueError("routing publication tensors must be on the input device")


def copy_routing(
    *,
    expert_ids: torch.Tensor,
    routing: torch.Tensor,
    routing_token_count_1: torch.Tensor,
    num_experts: int,
    validate_expert_ids: bool,
) -> None:
    """Publish inference routing metadata with native stream semantics."""
    _validate_routing_publication(
        expert_ids=expert_ids,
        routing=routing,
        routing_token_count_1=routing_token_count_1,
        device=expert_ids.device,
    )
    if validate_expert_ids:
        valid = (expert_ids >= -1) & (expert_ids < num_experts)
        torch._assert_async(
            torch.all(valid),
            "DistMoE expert ID is outside [-1, num_experts)",
        )
    routing.copy_(expert_ids)
    routing_token_count_1.fill_(expert_ids.shape[0])


def copy_routing_and_dispatch(
    *,
    x: torch.Tensor,
    dispatch: torch.Tensor,
    expert_ids: torch.Tensor,
    routing: torch.Tensor,
    routing_token_count_1: torch.Tensor,
    num_experts: int,
    validate_expert_ids: bool,
) -> None:
    """Publish actual T, routing IDs, and dense dispatch input in one launch."""
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("x must have dtype float16, bfloat16, or float32")
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional")
    if not dispatch.is_contiguous():
        raise ValueError("dispatch must be contiguous")
    if dispatch.shape != x.shape or dispatch.dtype != x.dtype:
        raise ValueError("dispatch must match x shape and dtype")
    _validate_routing_publication(
        expert_ids=expert_ids,
        routing=routing,
        routing_token_count_1=routing_token_count_1,
        device=x.device,
    )
    if dispatch.device != x.device:
        raise ValueError("dispatch must be on the input device")
    num_elements = max(x.numel(), expert_ids.numel())
    # The token-count header is published even for an empty payload.
    grid = (max(1, triton.cdiv(num_elements, _COPY_BLOCK_SIZE)),)
    _triton_copy_routing_and_dispatch[grid](
        # Triton otherwise calls mutable data_ptr(), which materializes COW
        # inputs and changes their address during CUDA-graph capture.
        x_ptr=_ReadOnlyTritonTensor(x),
        dispatch_ptr=dispatch,
        expert_ids_ptr=_ReadOnlyTritonTensor(expert_ids),
        routing_ptr=routing,
        routing_token_count_ptr=routing_token_count_1,
        x_num_elements=x.numel(),
        x_num_cols=x.shape[1],
        x_stride_0=x.stride(0),
        x_stride_1=x.stride(1),
        routing_num_elements=expert_ids.numel(),
        routing_num_rows=expert_ids.shape[0],
        routing_num_cols=expert_ids.shape[1],
        routing_stride_0=expert_ids.stride(0),
        routing_stride_1=expert_ids.stride(1),
        BLOCK_SIZE=_COPY_BLOCK_SIZE,
        X_CONTIGUOUS=x.is_contiguous(),
        ROUTING_CONTIGUOUS=expert_ids.is_contiguous(),
        num_warps=_COPY_NUM_WARPS,
        num_stages=_COPY_NUM_STAGES,
        NUM_EXPERTS=num_experts,
        VALIDATE_EXPERT_IDS=validate_expert_ids,
        **pdl_launch_kwargs(),
    )


def zero_out_padded_rows_(
    values_MD: torch.Tensor,
    expert_ids_TK: torch.Tensor,
) -> None:
    """Zero route rows whose corresponding expert ID is negative in place."""
    if values_MD.ndim != 2 or not values_MD.is_contiguous():
        raise ValueError("values_MD must be a contiguous two-dimensional tensor")
    if expert_ids_TK.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_ids_TK must use int32 or int64")
    if not expert_ids_TK.is_contiguous() or expert_ids_TK.device != values_MD.device:
        raise ValueError("expert_ids_TK must be contiguous on the values device")
    if values_MD.shape[0] != expert_ids_TK.numel():
        raise ValueError("values_MD rows must match the flattened expert-ID count")
    _triton_zero_out_padded_rows[(values_MD.shape[0],)](
        values_MD,
        expert_ids_TK,
        values_MD.shape[1],
        BLOCK_D=1024,
        num_warps=_COPY_NUM_WARPS,
        num_stages=_COPY_NUM_STAGES,
    )


def copy_dispatch_to_activation(
    *,
    dispatch: torch.Tensor,
    activation_buffer: torch.Tensor,
    activation_offset: torch.Tensor,
    condition: torch.Tensor | None,
) -> None:
    """Save the published dispatch payload to the activation buffer when selected."""
    _validate_copy_tensor(dispatch, name="dispatch")
    _validate_activation_destination(
        activation_buffer=activation_buffer,
        activation_offset=activation_offset,
        device=dispatch.device,
    )
    if condition is not None:
        validate_conditional_execution(condition, device=dispatch.device)

    num_ctas, num_warps = _copy_launch_config()
    _triton_copy_dispatch_to_activation[(num_ctas,)](
        dispatch_ptr=dispatch,
        activation_buffer_ptr=activation_buffer,
        activation_offset_ptr=activation_offset,
        condition_ptr=condition,
        num_elements=dispatch.numel(),
        BLOCK_SIZE=_COPY_BLOCK_SIZE,
        NUM_CTA=num_ctas,
        DTYPE=_triton_copy_dtype(dispatch.dtype),
        num_warps=num_warps,
        num_stages=_COPY_NUM_STAGES,
    )


def copy_activation_to_dispatch(
    *,
    dispatch: torch.Tensor,
    activation_buffer: torch.Tensor,
    activation_offset: torch.Tensor,
    condition: torch.Tensor | None,
) -> None:
    """Publish a saved activation, unconditionally when condition is None."""
    _validate_copy_tensor(dispatch, name="dispatch")
    _validate_activation_destination(
        activation_buffer=activation_buffer,
        activation_offset=activation_offset,
        device=dispatch.device,
    )
    if condition is not None:
        validate_conditional_execution(condition, device=dispatch.device)

    num_ctas, num_warps = _copy_launch_config()
    _triton_copy_activation_to_dispatch[(num_ctas,)](
        dispatch_ptr=dispatch,
        activation_buffer_ptr=activation_buffer,
        activation_offset_ptr=activation_offset,
        condition_ptr=condition,
        num_elements=dispatch.numel(),
        BLOCK_SIZE=_COPY_BLOCK_SIZE,
        NUM_CTA=num_ctas,
        DTYPE=_triton_copy_dtype(dispatch.dtype),
        num_warps=num_warps,
        num_stages=_COPY_NUM_STAGES,
    )


@triton.jit
def _triton_copy_routing_and_dispatch(
    x_ptr,
    dispatch_ptr,
    expert_ids_ptr,
    routing_ptr,
    routing_token_count_ptr,
    x_num_elements,
    x_num_cols,
    x_stride_0,
    x_stride_1,
    routing_num_elements,
    routing_num_rows,
    routing_num_cols,
    routing_stride_0,
    routing_stride_1,
    BLOCK_SIZE: tl.constexpr,
    X_CONTIGUOUS: tl.constexpr,
    ROUTING_CONTIGUOUS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    VALIDATE_EXPERT_IDS: tl.constexpr,
    PDL: tl.constexpr,
):
    pdl_wait_then_release(PDL)
    # int64: pid * BLOCK_SIZE wraps int32 once the published payload exceeds 2**31 elements.
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x_mask = offsets < x_num_elements
    if X_CONTIGUOUS:
        x_offsets = offsets
    else:
        x_rows = offsets // x_num_cols
        x_cols = offsets - x_rows * x_num_cols
        x_offsets = x_rows * x_stride_0 + x_cols * x_stride_1
    x_values = tl.load(x_ptr + x_offsets, mask=x_mask)
    tl.store(dispatch_ptr + offsets, x_values, mask=x_mask)

    if block_start < routing_num_elements:
        routing_mask = offsets < routing_num_elements
        if ROUTING_CONTIGUOUS:
            routing_offsets = offsets
        else:
            routing_rows = offsets // routing_num_cols
            routing_cols = offsets - routing_rows * routing_num_cols
            routing_offsets = (
                routing_rows * routing_stride_0 + routing_cols * routing_stride_1
            )
        expert_ids = tl.load(expert_ids_ptr + routing_offsets, mask=routing_mask)
        if VALIDATE_EXPERT_IDS:
            invalid = routing_mask & ((expert_ids < -1) | (expert_ids >= NUM_EXPERTS))
            has_invalid = tl.sum(invalid.to(tl.int32)) > 0
            should_trap = has_invalid & (get_flat_tid() == 0)
            if should_trap:
                tl.device_print("DistMoE expert ID is outside [-1, num_experts)")
            device_trap_if(should_trap)
        tl.store(routing_ptr + offsets, expert_ids, mask=routing_mask)
    tl.store(
        routing_token_count_ptr + offsets,
        routing_num_rows,
        mask=offsets == 0,
    )


@triton.jit
def _triton_zero_out_padded_rows(
    values_ptr,
    expert_ids_ptr,
    feature_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    route = tl.program_id(0).to(tl.int64)
    if tl.load(expert_ids_ptr + route) < 0:
        offsets = tl.arange(0, BLOCK_D)
        for start_d in tl.range(0, feature_dim, BLOCK_D):
            cols = start_d + offsets
            tl.store(
                values_ptr + route * feature_dim + cols,
                0.0,
                mask=cols < feature_dim,
            )


def conditional_copy_activations(
    condition: torch.Tensor,  # scalar bool/int32 tensor - nonzero=lhs, zero=rhs
    lhs: torch.Tensor | None,  # lhs tensor (None = skip lhs path)
    lhs_offset: torch.Tensor | None,  # buffer offset for lhs
    rhs: torch.Tensor | None,  # rhs tensor (None = skip rhs path)
    rhs_offset: torch.Tensor | None,  # buffer offset for rhs
    activation_buffer: torch.Tensor,  # uint8 buffer
    copy_to_buffer: bool = True,  # True = tensor->buffer, False = buffer->tensor
) -> None:
    """Conditionally copy between tensors and buffer based on device condition.

    This function is a ternary operation that executes one of two copy paths
    based on the condition tensor, with all decisions made on GPU:
    - If condition=True: execute lhs path (lhs <-> buffer[lhs_offset])
    - If condition=False: execute rhs path (rhs <-> buffer[rhs_offset])
    - If the selected tensor is None, no copy is performed (no-op)

    Direction is controlled by copy_to_buffer:
    - copy_to_buffer=True: tensor -> buffer (for forward save)
    - copy_to_buffer=False: buffer -> tensor (for backward load)

    Args:
        condition: Bool or int32 tensor selecting lhs when nonzero and rhs when zero
        lhs: Left-hand side tensor (or None to skip when condition=True)
        lhs_offset: Byte offset in buffer for lhs (required if lhs is not None)
        rhs: Right-hand side tensor (or None to skip when condition=False)
        rhs_offset: Byte offset in buffer for rhs (required if rhs is not None)
        activation_buffer: Activation memory buffer (uint8)
        copy_to_buffer: If True, copy tensor->buffer; if False, copy buffer->tensor
    """
    # Validate that at least one path is defined
    assert lhs is not None or rhs is not None, (
        "At least one of lhs or rhs must be provided"
    )

    # Determine dtype from the non-None tensor
    tensor = lhs if lhs is not None else rhs
    assert tensor is not None
    dtype = tensor.dtype
    assert dtype in (torch.float16, torch.bfloat16), "Only float16/bfloat16 supported"
    device = tensor.device
    validate_conditional_execution(condition, device=device)

    # Validate tensor properties
    if lhs is not None:
        assert lhs.is_contiguous(), "lhs must be contiguous"
        assert lhs_offset is not None, "lhs_offset required when lhs is provided"
    if rhs is not None:
        assert rhs.is_contiguous(), "rhs must be contiguous"
        assert rhs_offset is not None, "rhs_offset required when rhs is provided"
    if lhs is not None and rhs is not None:
        assert lhs.dtype == rhs.dtype, "lhs and rhs must have same dtype"

    # Get dimensions (use 0 for None tensors)
    lhs_num_elements = lhs.numel() if lhs is not None else 0
    rhs_num_elements = rhs.numel() if rhs is not None else 0

    num_ctas, num_warps = _copy_launch_config()

    # Get dtype for triton
    tl_dtype = tl.float16 if dtype == torch.float16 else tl.bfloat16

    # Handle None tensors by using the other tensor as dummy pointer (same type required)
    # Kernel will skip via num_elements=0
    if lhs is None and rhs is not None:
        lhs_ptr = rhs  # Use rhs as dummy (same dtype)
        lhs_offset_ptr = rhs_offset  # dummy
        rhs_ptr = rhs
        rhs_offset_ptr = rhs_offset
    elif rhs is None and lhs is not None:
        lhs_ptr = lhs
        lhs_offset_ptr = lhs_offset
        rhs_ptr = lhs  # Use lhs as dummy (same dtype)
        rhs_offset_ptr = lhs_offset  # dummy
    else:
        lhs_ptr = lhs
        rhs_ptr = rhs
        lhs_offset_ptr = lhs_offset
        rhs_offset_ptr = rhs_offset

    # Launch kernel
    _triton_conditional_copy[(num_ctas,)](
        lhs_ptr=lhs_ptr,
        rhs_ptr=rhs_ptr,
        buffer_ptr=activation_buffer,
        lhs_offset_ptr=lhs_offset_ptr,
        rhs_offset_ptr=rhs_offset_ptr,
        condition_ptr=condition,
        lhs_num_elements=lhs_num_elements,
        rhs_num_elements=rhs_num_elements,
        BLOCK_SIZE=_COPY_BLOCK_SIZE,
        NUM_CTA=num_ctas,
        DTYPE=tl_dtype,
        COPY_TO_BUFFER=copy_to_buffer,
        num_warps=num_warps,
        num_stages=_COPY_NUM_STAGES,
    )


@triton.jit
def _triton_copy_dispatch_to_activation(
    dispatch_ptr,
    activation_buffer_ptr,
    activation_offset_ptr,
    condition_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTA: tl.constexpr,
    DTYPE: tl.constexpr,
):
    if condition_ptr is not None:
        if tl.load(condition_ptr, eviction_policy="evict_last") == 0:
            return

    pid = tl.program_id(0)
    byte_offset = tl.load(activation_offset_ptr, eviction_policy="evict_last").to(
        tl.int64
    )
    activation_ptr_raw = (activation_buffer_ptr + byte_offset).to(
        tl.pointer_type(DTYPE)
    )
    align: tl.constexpr = MEMORY_ALIGNMENT_TL_CONSTEXPR // (
        DTYPE.primitive_bitwidth // 8
    )
    activation_ptr = tl.multiple_of(activation_ptr_raw, align)

    # Not tl.cdiv: n + BLOCK_SIZE - 1 wraps int32 just below 2**31 elements; n - 1 cannot.
    num_blocks = (num_elements - 1) // BLOCK_SIZE + 1
    num_blocks_per_cta = tl.cdiv(num_blocks, NUM_CTA)
    block_start = pid * num_blocks_per_cta
    block_end = min(block_start + num_blocks_per_cta, num_blocks)
    for block_idx in tl.range(block_start, block_end):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        values = tl.load(dispatch_ptr + offsets, mask=mask)
        tl.store(activation_ptr + offsets, values, mask=mask)


@triton.jit
def _triton_copy_activation_to_dispatch(
    dispatch_ptr,
    activation_buffer_ptr,
    activation_offset_ptr,
    condition_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTA: tl.constexpr,
    DTYPE: tl.constexpr,
):
    if condition_ptr is not None:
        if tl.load(condition_ptr, eviction_policy="evict_last") == 0:
            return

    pid = tl.program_id(0)
    byte_offset = tl.load(activation_offset_ptr, eviction_policy="evict_last").to(
        tl.int64
    )
    activation_ptr_raw = (activation_buffer_ptr + byte_offset).to(
        tl.pointer_type(DTYPE)
    )
    align: tl.constexpr = MEMORY_ALIGNMENT_TL_CONSTEXPR // (
        DTYPE.primitive_bitwidth // 8
    )
    activation_ptr = tl.multiple_of(activation_ptr_raw, align)

    # Not tl.cdiv: n + BLOCK_SIZE - 1 wraps int32 just below 2**31 elements; n - 1 cannot.
    num_blocks = (num_elements - 1) // BLOCK_SIZE + 1
    num_blocks_per_cta = tl.cdiv(num_blocks, NUM_CTA)
    block_start = pid * num_blocks_per_cta
    block_end = min(block_start + num_blocks_per_cta, num_blocks)
    for block_idx in tl.range(block_start, block_end):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        values = tl.load(activation_ptr + offsets, mask=mask)
        tl.store(dispatch_ptr + offsets, values, mask=mask)


@triton.jit
def _triton_conditional_copy(
    lhs_ptr,  # Pointer to lhs tensor
    rhs_ptr,  # Pointer to rhs tensor
    buffer_ptr,  # Pointer to shared buffer (uint8)
    lhs_offset_ptr,  # Pointer to lhs byte offset
    rhs_offset_ptr,  # Pointer to rhs byte offset
    condition_ptr,  # Pointer to condition (True=lhs, False=rhs)
    lhs_num_elements,  # Number of elements in lhs (0 if None)
    rhs_num_elements,  # Number of elements in rhs (0 if None)
    BLOCK_SIZE: tl.constexpr,
    NUM_CTA: tl.constexpr,
    DTYPE: tl.constexpr,
    COPY_TO_BUFFER: tl.constexpr,  # True = tensor->buffer, False = buffer->tensor
):
    """Triton kernel for conditional copy between tensor and buffer.

    Uses persistent kernel design where each CTA processes contiguous blocks.
    All threads check the condition once and perform the appropriate copy.

    When COPY_TO_BUFFER=True: copies from tensor to buffer[offset]
    When COPY_TO_BUFFER=False: copies from buffer[offset] to tensor
    """
    pid = tl.program_id(0)

    # Load the condition once (same for all iterations)
    condition = tl.load(condition_ptr, eviction_policy="evict_last").to(tl.int1)

    # Select tensor pointer, num_elements, and offset based on condition
    tensor_ptr = tl.where(condition, lhs_ptr, rhs_ptr)
    num_elements = tl.where(condition, lhs_num_elements, rhs_num_elements)
    byte_offset = tl.where(
        condition,
        tl.load(lhs_offset_ptr, eviction_policy="evict_last"),
        tl.load(rhs_offset_ptr, eviction_policy="evict_last"),
    )

    # Early exit if num_elements is 0 (selected path is None)
    if num_elements == 0:
        return

    # Precompute typed buffer base pointer ONCE before the loop.
    # This eliminates per-iteration int64 multiply (offsets * element_size),
    # int64 add (buffer_element_offset + offsets), and pointer cast.
    typed_buffer_base_raw = (buffer_ptr + byte_offset).to(tl.pointer_type(DTYPE))
    # Apply tl.multiple_of to attach MEMORY_ALIGNMENT-byte alignment metadata
    # This enables vectorized memory operations
    align: tl.constexpr = MEMORY_ALIGNMENT_TL_CONSTEXPR // (
        DTYPE.primitive_bitwidth // 8
    )
    typed_buffer_base = tl.multiple_of(typed_buffer_base_raw, align)

    # Persistent kernel: each CTA processes contiguous blocks
    # Not tl.cdiv: n + BLOCK_SIZE - 1 wraps int32 just below 2**31 elements; n - 1 cannot.
    num_blocks = (num_elements - 1) // BLOCK_SIZE + 1
    num_blocks_per_cta = tl.cdiv(num_blocks, NUM_CTA)
    block_start = pid * num_blocks_per_cta
    block_end = min(block_start + num_blocks_per_cta, num_blocks)
    for block_idx in tl.range(block_start, block_end):
        element_start = block_idx * BLOCK_SIZE
        offsets = element_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements

        if COPY_TO_BUFFER:
            # tensor -> buffer: load from tensor, store to buffer
            vals = tl.load(tensor_ptr + offsets, mask=mask)
            tl.store(typed_buffer_base + offsets, vals, mask=mask)
        else:
            # buffer -> tensor: load from buffer, store to tensor
            vals = tl.load(typed_buffer_base + offsets, mask=mask)
            tl.store(tensor_ptr + offsets, vals, mask=mask)
