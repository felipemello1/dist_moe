# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Activation buffer planner for distributed MoE with dynamic recompute decision.

This module provides utilities for planning activation buffer usage in the forward
and backward pass of distributed MoE layers. It calculates memory requirements
and determines whether recompute is needed based on available buffer capacity.

MoE dataflow and tensor glossary
--------------------------------
A distributed-MoE layer routes each of this rank's ``num_tokens`` tokens to
``topk`` experts, dispatches them (all-to-all) to the ranks owning those experts,
runs a two-matmul SwiGLU FFN per expert, then combines (all-to-all back +
weighted sum) the results. ``num_recv_tokens`` is how many token rows this rank
receives for its local experts; its padded worst case is ``max_recv_tokens``.

Forward (per layer):
    x           --dispatch-->  x_gathered
    x_gathered  --FC13----->   h1 = [gate | up]
    h1          --SwiGLU--->   h2 = silu(gate) * up
    h2          --FC2------>   h3
    h3          --combine-->   layer output

Backward reverses this, producing a gradient for each activation:
    grad_h3 --FC2 DGRAD-->  grad_h2       (FC2 WGRAD consumes grad_h3, h2)
    grad_h2 --SwiGLU bwd->  grad_h1 (dxy)
    grad_h1 --FC13 DGRAD->  grad_x        (FC13 WGRAD consumes grad_h1, x_gathered)

Tensors (R = received-token rows; the planner sizes bundles at max_recv_tokens):
    x                 [num_tokens, hidden_dim]         layer input tokens (pre-dispatch)
    x_gathered        [R, hidden_dim]                  tokens received for local experts (FC13 input)
    h1                [R, 2 * intermediate_dim]        FC13 output; packed SwiGLU input [gate | up]
    h2                [R, intermediate_dim]            SwiGLU output; FC2 input
    h3                [num_tokens * topk, hidden_dim]  FC2 output staged for combine (saved as h3)
    grad_h3           [R, hidden_dim]                  gradient w.r.t. h3 (a.k.a. grad_h3_gathered)
    grad_h2           [R, intermediate_dim]            gradient w.r.t. h2
    grad_h1 / dxy     [R, 2 * intermediate_dim]        gradient w.r.t. h1 (both SwiGLU branches; "dxy")

For dense BF16/FP16 activations each tensor is a single allocation. For the async
block-scaled (MXFP8/NVFP4) kernels, tensors are stored as quantized bundles --
see BlockscaledStorageConfig for the row/column bundle geometry.
"""

from __future__ import annotations

import dataclasses
import functools
from logging import getLogger

import torch

from ._activation_buffer_planner_kernel import (
    _triton_get_backward_plan,
    _triton_get_blockscaled_backward_plan,
    _triton_get_blockscaled_forward_plan,
    _triton_get_forward_plan,
    BLOCKSCALED_DISPATCH_ROW_ALIGNMENT_BYTES,
    MEMORY_ALIGNMENT,
)
from .kernels.activation_buffer import (
    ACTIVATION_OFFSET_COUNT,
    ACTIVATION_ROW_SCALE_STORAGE_MULTIPLE,
    BACKWARD_ACTIVATION_OFFSET_COUNT,
    BACKWARD_FC2_DGRAD_OFFSET_BASE,
    BACKWARD_FC2_WGRAD_OFFSET_BASE,
    BACKWARD_FC13_DGRAD_OFFSET_BASE,
    BACKWARD_FC13_WGRAD_OFFSET_BASE,
    BACKWARD_RECOMPUTE_COMBINE_OFFSET_BASE,
    BACKWARD_RECOMPUTE_DISPATCH_OFFSET_BASE,
    FORWARD_ACTIVATION_OFFSET_COUNT,
    FORWARD_COMBINE_OFFSET_BASE,
    FORWARD_DISPATCH_OFFSET_BASE,
    GEMM_OPERAND_OFFSET_COUNT,
    MEGA_ACTIVATION_OFFSET_COUNT,
)
from .kernels.pdl import pdl_launch_kwargs

logger = getLogger()

_BF16_FORWARD_PLAN_OFFSET_COUNT = 5
_BF16_FORWARD_STATE_COUNT = _BF16_FORWARD_PLAN_OFFSET_COUNT + 1
_BLOCKSCALED_FORWARD_PLAN_OFFSET_COUNT = 9
_BLOCKSCALED_BACKWARD_PLAN_OFFSET_COUNT = 16


def _align_bytes(size_bytes: int) -> int:
    """Align one host-side byte count to the planner's kernel alignment.

    Args:
        size_bytes: Unaligned byte count.

    Returns:
        Byte count rounded up to ``MEMORY_ALIGNMENT``.
    """
    return (size_bytes + MEMORY_ALIGNMENT - 1) // MEMORY_ALIGNMENT * MEMORY_ALIGNMENT


def _normalize_aligned_buffer(
    buffer: torch.Tensor,
    *,
    required_size: int | None = None,
) -> torch.Tensor:
    """Return an aligned byte view of an activation allocation.

    Args:
        buffer: One-dimensional byte allocation.
        required_size: Optional exact aligned view size in bytes.

    Returns:
        A 128-byte-aligned view backed by ``buffer``.

    Raises:
        ValueError: If the requested aligned view does not fit.
    """
    assert buffer.dtype == torch.uint8
    if not buffer.is_cuda:
        if required_size is not None:
            assert buffer.numel() >= required_size, (buffer.numel(), required_size)
            return buffer.narrow(0, 0, required_size)
        return buffer
    start_pad = (-buffer.data_ptr()) % MEMORY_ALIGNMENT
    if required_size is not None:
        assert required_size % MEMORY_ALIGNMENT == 0, (
            required_size,
            MEMORY_ALIGNMENT,
        )
        if start_pad + required_size > buffer.numel():
            raise ValueError(
                f"Cannot carve aligned activation buffer subview of size {required_size} "
                f"from raw buffer of size {buffer.numel()} with start_pad={start_pad}"
            )
        aligned_size = required_size
    else:
        aligned_size = (
            (buffer.numel() - start_pad) // MEMORY_ALIGNMENT
        ) * MEMORY_ALIGNMENT
        if aligned_size <= 0:
            raise ValueError(
                f"Aligned activation buffer size is non-positive after trimming: "
                f"raw_numel={buffer.numel()} start_pad={start_pad}"
            )
    if start_pad != 0 or aligned_size != buffer.numel():
        logger.warning(
            "Adjusting activation buffer alignment: raw_ptr=%#x raw_numel=%s start_pad=%s aligned_numel=%s",
            buffer.data_ptr(),
            buffer.numel(),
            start_pad,
            aligned_size,
        )
    buffer = buffer.narrow(0, start_pad, aligned_size)
    assert buffer.data_ptr() % MEMORY_ALIGNMENT == 0, (
        buffer.data_ptr(),
        MEMORY_ALIGNMENT,
    )
    assert buffer.numel() % MEMORY_ALIGNMENT == 0, (
        buffer.numel(),
        MEMORY_ALIGNMENT,
    )
    return buffer


def _assert_slot_boundaries_aligned(offsets: list[int]) -> None:
    """Assert that all activation-slot offsets meet kernel alignment.

    Args:
        offsets: Byte offsets to validate.
    """
    for offset in offsets:
        assert offset % MEMORY_ALIGNMENT == 0, (offset, MEMORY_ALIGNMENT)


# =============================================================================
# Public APIs
# =============================================================================


@dataclasses.dataclass
class ScratchInfo:
    """Scratch region usage snapshot from the activation buffer.

    Returned by ActivationBuffer.check_overflow(). Contains scratch region
    usage metrics for logging.

    When VMM is not enabled, the entire scratch region is device-resident:
    device_scratch_size == scratch_size and host_scratch_size is None.
    """

    # Peak minimum free bytes in the scratch region this step.
    # Negative means overflow by that many bytes.
    scratch_min_free: float
    # Total scratch region size in bytes.
    scratch_size: float
    # Total scratch bytes used this step (= scratch_size - scratch_min_free).
    scratch_max_used: float
    # Device-resident portion of scratch in bytes.
    # Equals scratch_size when VMM is not enabled.
    device_scratch_size: float
    # Device-resident scratch bytes used.
    # Equals scratch_max_used when VMM is not enabled.
    device_scratch_used: float
    # Host-resident portion of scratch in bytes, or None if VMM is not enabled.
    host_scratch_size: float | None
    # Host-resident scratch bytes used, or None if VMM is not enabled.
    host_scratch_used: float | None


@dataclasses.dataclass
class PendingOverflowCheck:
    """In-flight async D2H of peak_min_free_space.

    Returned by ActivationBuffer.begin_check_overflow(); resolved by
    finish_check_overflow(). `ready` is None when the buffer lives on CPU
    (unit tests) and the snapshot is already synchronous.
    """

    host_peaks: torch.Tensor
    ready: torch.cuda.Event | None


@dataclasses.dataclass
class ActivationBuffer:
    """Activation buffer for activation saving in distributed MoE.

    The buffer is partitioned into N activation slots plus one shared scratch
    region. In inference mode, N is canonicalized to zero
    and the single offset/peak slot tracks the shared scratch region.

    It uses N+1 pointers: one front-to-back saved-state pointer per activation
    slot followed by the shared scratch end, which grows back-to-front. Separate
    slots avoid LIFO violations when schedule lifetimes interleave.

    Memory layout:
    [Slot 0 v | Slot 1 v | ... | Slot N-1 v | Scratch ^]

    Tracks the buffer, its valid memory region, and how many tokens
    have been saved from each rank.
    """

    # The activation buffer tensor for saving activations
    buffer: torch.Tensor
    # Tensor [N+1] containing activation-slot pointers:
    # [free_start_0, ..., free_start_{N-1}, scratch_end].
    # In inference mode, N=0 and this contains only [scratch_end].
    buffer_offsets: torch.Tensor
    # Tensor [N*EP] tracking total saved activation bytes per EP rank per microbatch.
    # Used for look-ahead recompute decision: max across ranks gives worst-case usage.
    # Indexed as [microbatch_id * EP + rank]
    saved_activation_bytes_per_rank: torch.Tensor
    # Tensor [N+1] int64 tracking minimum free space per slot and scratch.
    # Indices 0..N-1 track activation slots.
    # Index N tracks the shared scratch region.
    # This captures peak memory usage, not just final state after de-allocation.
    peak_min_free_space: torch.Tensor
    # Effective bytes in each activation slot.
    # (buffer_size - scratch_size) // N, aligned to MEMORY_ALIGNMENT.
    # Zero in inference mode.
    activation_slot_bytes: int = 0
    # Number of activation slots (N).
    num_activation_slots: int = 1
    # Immutable int64 IDs for every physical activation slot. Selecting a view
    # preserves the forward's slot identity until its matching backward.
    activation_slot_ids_S: torch.Tensor = dataclasses.field(
        default_factory=lambda: torch.zeros(1, dtype=torch.int64)
    )
    # Tensor [1] int64 view selecting the current physical activation slot.
    activation_slot_id_1: torch.Tensor = dataclasses.field(
        default_factory=lambda: torch.zeros(1, dtype=torch.int64)
    )
    # Number of MoE layers in the model (for buffer capacity validation).
    num_moe_layers: int = 1
    # Static local layer depth selected for the next planner invocation.
    _num_moe_layers_in_selected_slot: int = 1
    # Terminal device-backed hot-scratch section in a VMM allocation.
    # None means non-VMM (all scratch is on device).
    device_scratch_size: int | None = None
    # Exact host-backed overflow section in a VMM allocation. A lower device
    # prefix can also become scratch in scratch-only inference.
    host_scratch_size: int | None = None
    # Capacity factor used to size the scratch region. In VMM mode this is
    # the effective imbalance (min(vmm_worst_case_imbalance, ep_size)); in
    # non-VMM mode this is chunk_capacity_factor. Used to suggest a concrete
    # knob value on scratch overflow.
    scratch_capacity_factor: float | None = None
    # In inference mode the buffer has zero-sized activation slots and the
    # full allocation is used as the shared scratch region.
    inference_mode: bool = False
    # Reference to ModelConfig for computing overflow suggestions (new scratch
    # size from a suggested factor, min_buffer_size, etc.). Set after creation.
    model_config: ModelConfig | None = None
    # Tensor [N] int64 tracking current MoE layer index per activation slot.
    # Incremented in forward, decremented in backward. Used for look-ahead
    # recompute decision: num_layers_after = (num_moe_layers - 1) - moe_layer_id.
    moe_layer_id: torch.Tensor = dataclasses.field(
        default_factory=lambda: torch.zeros(1, dtype=torch.int64)
    )
    # Device-side snapshots used to reset state without host/device copies.
    # `reset()` can run during vLLM CUDA graph capture, where Python scalar
    # writes to CUDA tensors and `.cpu()` validation are illegal.
    initial_buffer_offsets: torch.Tensor = dataclasses.field(init=False)
    initial_peak_min_free_space: torch.Tensor = dataclasses.field(init=False)
    # Pinned staging for begin_check_overflow(); allocated once on first use.
    _overflow_check_staging: torch.Tensor | None = dataclasses.field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Align the allocation and snapshot its initial planner state."""
        self.buffer = _normalize_aligned_buffer(self.buffer)
        initial_offsets = [
            i * self.activation_slot_bytes for i in range(self.num_activation_slots)
        ]
        _assert_slot_boundaries_aligned(initial_offsets + [self.buffer.numel()])
        self.initial_buffer_offsets = torch.tensor(
            initial_offsets + [self.buffer.numel()],
            dtype=self.buffer_offsets.dtype,
            device=self.buffer_offsets.device,
        )
        initial_peak_min_free_space = [self.activation_slot_bytes] * (
            self.num_activation_slots
        )
        self.initial_peak_min_free_space = torch.tensor(
            initial_peak_min_free_space + [self.scratch_region_size],
            dtype=self.peak_min_free_space.dtype,
            device=self.peak_min_free_space.device,
        )

    @property
    def ep_size(self) -> int:
        """Number of expert parallel ranks tracked for saved activations."""
        if self.num_activation_slots == 0:
            raise ValueError(
                "Inference-mode ActivationBuffer has no activation slots, so "
                "ep_size cannot be inferred from saved_activation_bytes_per_rank."
            )
        return self.saved_activation_bytes_per_rank.numel() // self.num_activation_slots

    @property
    def scratch_region_size(self) -> int:
        """Size in bytes of the shared scratch region."""
        return (
            self.buffer.numel() - self.num_activation_slots * self.activation_slot_bytes
        )

    @staticmethod
    def _canonicalize_layout(
        total_size_in_bytes: int,
        scratch_mem_size_in_bytes: int,
        num_activation_slots: int,
        inference_mode: bool,
    ) -> int:
        """Validate layout inputs and return the effective activation-slot count.

        Args:
            total_size_in_bytes: Total activation allocation size.
            scratch_mem_size_in_bytes: Requested scratch reservation.
            num_activation_slots: Requested number of activation slots.
            inference_mode: Whether the allocation is scratch-only.

        Returns:
            Effective activation-slot count, which is zero for inference.
        """
        N = num_activation_slots
        if inference_mode:
            assert N >= 0, f"num_activation_slots must be >= 0, got {N}"
            N = 0
        else:
            assert N >= 1, f"num_activation_slots must be >= 1, got {N}"
        assert scratch_mem_size_in_bytes > 0, (
            f"scratch_mem_size_in_bytes must be > 0, got {scratch_mem_size_in_bytes}"
        )
        if inference_mode:
            assert total_size_in_bytes >= scratch_mem_size_in_bytes, (
                f"total_size_in_bytes ({total_size_in_bytes}) must be >= "
                f"scratch_mem_size_in_bytes ({scratch_mem_size_in_bytes}) in inference mode"
            )
        else:
            assert total_size_in_bytes > scratch_mem_size_in_bytes, (
                f"total_size_in_bytes ({total_size_in_bytes}) must be > scratch_mem_size_in_bytes "
                f"({scratch_mem_size_in_bytes})"
            )
        return N

    @staticmethod
    def _make_layout_tensors(
        total_size_in_bytes: int,
        ep_size: int,
        device: torch.device,
        scratch_mem_size_in_bytes: int,
        num_activation_slots: int,
        inference_mode: bool,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the device tensors that describe an activation layout.

        Args:
            total_size_in_bytes: Total activation allocation size.
            ep_size: Number of expert-parallel ranks.
            device: Device for planner state tensors.
            scratch_mem_size_in_bytes: Requested scratch reservation.
            num_activation_slots: Effective activation-slot count.
            inference_mode: Whether the allocation is scratch-only.

        Returns:
            Activation slot size, offsets, per-rank saved byte counts,
            minimum free-space counters, and per-slot MoE layer IDs.
        """
        if inference_mode:
            activation_slot_bytes = 0
        else:
            activation_slot_bytes = (
                (total_size_in_bytes - scratch_mem_size_in_bytes)
                // num_activation_slots
                // MEMORY_ALIGNMENT
                * MEMORY_ALIGNMENT
            )
            if activation_slot_bytes < MEMORY_ALIGNMENT:
                raise ValueError(
                    "training activation storage must provide at least one "
                    "aligned unit per activation slot; "
                    f"total_size_in_bytes={total_size_in_bytes}, "
                    f"scratch_mem_size_in_bytes={scratch_mem_size_in_bytes}, "
                    f"num_activation_slots={num_activation_slots}"
                )
        assert activation_slot_bytes % MEMORY_ALIGNMENT == 0, activation_slot_bytes
        offsets = [i * activation_slot_bytes for i in range(num_activation_slots)] + [
            total_size_in_bytes
        ]
        _assert_slot_boundaries_aligned(offsets)

        buffer_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        saved_activation_bytes_per_rank = torch.zeros(
            num_activation_slots * ep_size, dtype=torch.int64, device=device
        )
        scratch_region_size = (
            total_size_in_bytes - num_activation_slots * activation_slot_bytes
        )
        assert scratch_region_size % MEMORY_ALIGNMENT == 0, scratch_region_size
        peak_init = [activation_slot_bytes] * num_activation_slots + [
            scratch_region_size
        ]
        peak_min_free_space = torch.tensor(peak_init, dtype=torch.int64, device=device)
        moe_layer_id = torch.zeros(
            num_activation_slots, dtype=torch.int64, device=device
        )

        return (
            activation_slot_bytes,
            buffer_offsets,
            saved_activation_bytes_per_rank,
            peak_min_free_space,
            moe_layer_id,
        )

    @classmethod
    def create(
        cls,
        total_size_in_bytes: int,
        ep_size: int,
        device: torch.device,
        scratch_mem_size_in_bytes: int,
        num_activation_slots: int,
        num_moe_layers: int,
        inference_mode: bool = False,
    ) -> ActivationBuffer:
        """Create an ActivationBuffer with fields initialized.

        The buffer is partitioned into N activation slots plus one shared
        scratch region. In inference mode, N is canonicalized to zero
        and the full buffer is used as scratch.

        Args:
            total_size_in_bytes: Total size of the buffer in bytes.
            ep_size: Number of expert parallel ranks.
            device: Device to allocate tensors on.
            scratch_mem_size_in_bytes: Size of pre-reserved scratch region in bytes.
                Must be > 0.
            num_activation_slots: Number of activation slots. Ignored in
                inference mode, where no activation slots are allocated.
            num_moe_layers: Number of MoE layers in the model. Used for
                look-ahead recompute decision and buffer capacity validation.
            inference_mode: If True, reserve the full buffer as scratch and
                create zero-sized activation slots.

        Returns:
            ActivationBuffer with uninitialized buffer and initialized offsets.
        """
        N = cls._canonicalize_layout(
            total_size_in_bytes=total_size_in_bytes,
            scratch_mem_size_in_bytes=scratch_mem_size_in_bytes,
            num_activation_slots=num_activation_slots,
            inference_mode=inference_mode,
        )

        total_size_in_bytes = _align_bytes(total_size_in_bytes)
        raw_buffer = torch.empty(
            total_size_in_bytes + MEMORY_ALIGNMENT,
            dtype=torch.uint8,
            device=device,
        )
        buffer = _normalize_aligned_buffer(
            raw_buffer,
            required_size=total_size_in_bytes,
        )

        (
            activation_slot_bytes,
            buffer_offsets,
            saved_activation_bytes_per_rank,
            peak_min_free_space,
            moe_layer_id,
        ) = cls._make_layout_tensors(
            total_size_in_bytes=total_size_in_bytes,
            ep_size=ep_size,
            device=device,
            scratch_mem_size_in_bytes=scratch_mem_size_in_bytes,
            num_activation_slots=N,
            inference_mode=inference_mode,
        )

        activation_slot_ids_S = torch.arange(
            max(N, 1), dtype=torch.int64, device=device
        )
        activation_slot_id_1 = activation_slot_ids_S.narrow(0, 0, 1)

        return cls(
            buffer=buffer,
            buffer_offsets=buffer_offsets,
            saved_activation_bytes_per_rank=saved_activation_bytes_per_rank,
            peak_min_free_space=peak_min_free_space,
            num_activation_slots=N,
            activation_slot_bytes=activation_slot_bytes,
            activation_slot_ids_S=activation_slot_ids_S,
            activation_slot_id_1=activation_slot_id_1,
            num_moe_layers=num_moe_layers,
            _num_moe_layers_in_selected_slot=num_moe_layers,
            moe_layer_id=moe_layer_id,
            inference_mode=inference_mode,
        )

    @classmethod
    def create_from_buffer(
        cls,
        buffer: torch.Tensor,
        ep_size: int,
        scratch_mem_size_in_bytes: int,
        num_activation_slots: int,
        num_moe_layers: int,
        required_size_in_bytes: int | None = None,
        device_scratch_size: int | None = None,
        host_scratch_size: int | None = None,
        inference_mode: bool = False,
    ) -> ActivationBuffer:
        """Create an ActivationBuffer from a pre-allocated buffer tensor.

        Like create(), but accepts an already-allocated buffer tensor instead
        of allocating one internally. This is used when the buffer is allocated
        via a custom memory pool (e.g., VMM) and helper tensors must be
        allocated separately on the default allocator.

        Args:
            buffer: Pre-allocated buffer tensor (uint8, on CUDA device).
            ep_size: Number of expert parallel ranks.
            scratch_mem_size_in_bytes: Size of pre-reserved scratch region in bytes.
            num_activation_slots: Number of activation slots. Ignored in
                inference mode, where no activation slots are allocated.
            num_moe_layers: Number of MoE layers in the model.
            required_size_in_bytes: Optional exact aligned view size. Extra
                bytes in the owning allocation are excluded from the buffer.
            device_scratch_size: When VMM is used, the device-resident portion
                at the high end of the scratch region. None means all scratch
                is on device.
            host_scratch_size: Exact VMM host overflow-section size. None means
                all scratch is on device.
            inference_mode: If True, reserve the full buffer as scratch and
                create zero-sized activation slots.

        Returns:
            ActivationBuffer wrapping the provided buffer.
        """
        buffer = _normalize_aligned_buffer(
            buffer,
            required_size=required_size_in_bytes,
        )
        total_size_in_bytes = buffer.numel()
        device = buffer.device
        N = cls._canonicalize_layout(
            total_size_in_bytes=total_size_in_bytes,
            scratch_mem_size_in_bytes=scratch_mem_size_in_bytes,
            num_activation_slots=num_activation_slots,
            inference_mode=inference_mode,
        )
        (
            activation_slot_bytes,
            buffer_offsets,
            saved_activation_bytes_per_rank,
            peak_min_free_space,
            moe_layer_id,
        ) = cls._make_layout_tensors(
            total_size_in_bytes=total_size_in_bytes,
            ep_size=ep_size,
            device=device,
            scratch_mem_size_in_bytes=scratch_mem_size_in_bytes,
            num_activation_slots=N,
            inference_mode=inference_mode,
        )

        activation_slot_ids_S = torch.arange(
            max(N, 1), dtype=torch.int64, device=device
        )
        activation_slot_id_1 = activation_slot_ids_S.narrow(0, 0, 1)

        return cls(
            buffer=buffer,
            buffer_offsets=buffer_offsets,
            saved_activation_bytes_per_rank=saved_activation_bytes_per_rank,
            peak_min_free_space=peak_min_free_space,
            num_activation_slots=N,
            activation_slot_bytes=activation_slot_bytes,
            activation_slot_ids_S=activation_slot_ids_S,
            activation_slot_id_1=activation_slot_id_1,
            num_moe_layers=num_moe_layers,
            _num_moe_layers_in_selected_slot=num_moe_layers,
            moe_layer_id=moe_layer_id,
            device_scratch_size=device_scratch_size,
            host_scratch_size=host_scratch_size,
            inference_mode=inference_mode,
        )

    def reset(self) -> None:
        """Reset the buffer state to initial values.

        Reset each slot pointer to its start and scratch to its upper bound.

        Zeros saved_activation_bytes_per_rank and resets peak_min_free_space.
        The buffer contents are not cleared (they will be overwritten on next use).
        Slot alignment is validated at construction; reset does not read
        device offsets back to the host.
        """
        is_capturing = self.buffer.is_cuda and torch.cuda.is_current_stream_capturing()
        if not is_capturing:
            self.buffer = _normalize_aligned_buffer(self.buffer)
        self.buffer_offsets.copy_(self.initial_buffer_offsets)
        self.saved_activation_bytes_per_rank.zero_()
        self.peak_min_free_space.copy_(self.initial_peak_min_free_space)
        self.moe_layer_id.zero_()
        self.activation_slot_id_1 = self.activation_slot_ids_S.narrow(0, 0, 1)
        self._num_moe_layers_in_selected_slot = self.num_moe_layers

    def select_activation_slot(
        self,
        activation_slot: int,
        num_moe_layers_in_slot: int,
    ) -> None:
        """Select one activation slot and its actual layer depth.

        Selection chooses an immutable device-scalar view for the physical slot
        and records the static slot depth passed to the planner kernel. A full
        PP-step CUDA graph captures the fixed view used by each scheduled call.
        Scratch-only inference has no activation slots, so selection is ignored.

        Args:
            activation_slot: Physical activation slot selected for the next
                execution interval.
            num_moe_layers_in_slot: Maximum MoE-layer depth assigned to the
                selected activation slot.

        Raises:
            ValueError: If the slot or layer count is outside the configured
                activation-buffer capacity.
        """
        if self.num_activation_slots == 0:
            return
        if not 0 <= activation_slot < self.num_activation_slots:
            raise ValueError(
                f"activation slot must be in [0, {self.num_activation_slots}), "
                f"got {activation_slot}"
            )
        if not 1 <= num_moe_layers_in_slot <= self.num_moe_layers:
            raise ValueError(
                "num_moe_layers_in_slot must be in "
                f"[1, {self.num_moe_layers}], got {num_moe_layers_in_slot}"
            )
        self.activation_slot_id_1 = self.activation_slot_ids_S.narrow(
            0, activation_slot, 1
        )
        self._num_moe_layers_in_selected_slot = num_moe_layers_in_slot

    def check_overflow(self) -> ScratchInfo:
        """Check for buffer overflow and return scratch usage info.

        Report memory statistics for every activation slot and shared scratch,
        then check for overflow.
        Returns ScratchInfo for metrics logging.

        Returns:
            ScratchInfo with device/host scratch usage information.

        Note: This is a blocking host-side check that requires a device-to-host transfer.
        """
        # Single bulk D2H transfer for all slot and scratch counters.
        peak_values = self.peak_min_free_space.tolist()
        return self._scratch_info_from_peaks(peak_values)

    def begin_check_overflow(self) -> PendingOverflowCheck:
        """Launch the overflow check's D2H without blocking the host.

        The copy is stream-ordered after all enqueued fwd/bwd work — the same
        values the synchronous check would read — but the host does not drain
        the GPU queue here. The caller must resolve the returned handle with
        finish_check_overflow() before the optimizer commits, so an overflow
        still aborts the step with full diagnostics.

        Returns:
            Pending host-side peak counters and their completion event.
        """
        src = self.peak_min_free_space
        if not src.is_cuda:
            return PendingOverflowCheck(host_peaks=src.clone(), ready=None)
        staging = self._overflow_check_staging
        if staging is None:
            staging = torch.empty_like(src, device="cpu", pin_memory=True)
            self._overflow_check_staging = staging
        staging.copy_(src, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record()
        return PendingOverflowCheck(host_peaks=staging, ready=ready)

    def finish_check_overflow(self, pending: PendingOverflowCheck) -> ScratchInfo:
        """Resolve an asynchronous overflow check.

        Args:
            pending: Handle returned by :meth:`begin_check_overflow`.

        Returns:
            Peak activation and scratch usage.

        Raises:
            RuntimeError: If the planner recorded an overflow.
        """
        if pending.ready is not None:
            pending.ready.synchronize()
        return self._scratch_info_from_peaks(pending.host_peaks.tolist())

    def _scratch_info_from_peaks(self, peak_values: list[int]) -> ScratchInfo:
        """Evaluate overflow from an already transferred peak snapshot.

        Args:
            peak_values: Minimum-free-byte counters for slots and scratch.

        Returns:
            Peak activation and scratch usage.

        Raises:
            RuntimeError: If any region overflowed.
        """
        BYTES_PER_GB = 1024 * 1024 * 1024
        N = self.num_activation_slots
        reserved_memory = self.buffer.nbytes / BYTES_PER_GB

        has_overflow = False
        activation_used_total = 0.0

        # Collect per-slot stats.
        slot_parts = []
        for i in range(N):
            slot_bytes = self.activation_slot_bytes
            min_free = peak_values[i]
            max_used = slot_bytes - min_free
            activation_used_total += max_used
            slot_parts.append(
                f"slot{i}({max_used / BYTES_PER_GB:.4f}/{slot_bytes / BYTES_PER_GB:.4f}GB)"
            )
            if min_free < 0:
                has_overflow = True

        # Compute scratch info from the already-transferred peak_values[N]
        scratch_size = float(self.scratch_region_size)
        scratch_min_free = float(peak_values[N])
        scratch_max_used = scratch_size - scratch_min_free
        if self.device_scratch_size is not None:
            hot_device_size = float(self.device_scratch_size)
            host_scratch_size = float(self.host_scratch_size or 0)
            lower_device_size = max(
                0.0, scratch_size - hot_device_size - host_scratch_size
            )
            hot_device_used = min(scratch_max_used, hot_device_size)
            host_scratch_used = min(
                max(0.0, scratch_max_used - hot_device_size),
                host_scratch_size,
            )
            lower_device_used = max(
                0.0,
                scratch_max_used - hot_device_size - host_scratch_size,
            )
            device_scratch_size = hot_device_size + lower_device_size
            device_scratch_used = hot_device_used + lower_device_used
        else:
            device_scratch_size = scratch_size
            host_scratch_size = None
            device_scratch_used = scratch_max_used
            host_scratch_used = None

        scratch_info = ScratchInfo(
            scratch_min_free=scratch_min_free,
            scratch_size=scratch_size,
            scratch_max_used=scratch_max_used,
            device_scratch_size=device_scratch_size,
            device_scratch_used=device_scratch_used,
            host_scratch_size=host_scratch_size,
            host_scratch_used=host_scratch_used,
        )

        if self.device_scratch_size is not None:
            assert host_scratch_size is not None
            slot_parts.append(
                f"scratch({scratch_max_used / BYTES_PER_GB:.4f}/"
                f"{scratch_size / BYTES_PER_GB:.4f}GB "
                f"[device:{device_scratch_size / BYTES_PER_GB:.4f}GB, "
                f"host:{host_scratch_size / BYTES_PER_GB:.4f}GB])"
            )
        else:
            slot_parts.append(
                f"scratch({scratch_max_used / BYTES_PER_GB:.4f}/{scratch_size / BYTES_PER_GB:.4f}GB)"
            )
        if scratch_min_free < 0:
            has_overflow = True

        total_max_used = activation_used_total + scratch_max_used
        if self.device_scratch_size is not None:
            assert host_scratch_size is not None and host_scratch_used is not None
            device_reserved = N * self.activation_slot_bytes + device_scratch_size
            host_reserved = host_scratch_size
            device_used = activation_used_total + device_scratch_used
            host_used = host_scratch_used
            logger.debug(
                f"ActivationBuffer: "
                f"reserved={reserved_memory:.4f}GB "
                f"[device:{device_reserved / BYTES_PER_GB:.4f}GB, "
                f"host:{host_reserved / BYTES_PER_GB:.4f}GB], "
                f"used={total_max_used / BYTES_PER_GB:.4f}GB "
                f"[device:{device_used / BYTES_PER_GB:.4f}GB, "
                f"host:{host_used / BYTES_PER_GB:.4f}GB], "
                f"slots(used/total): {', '.join(slot_parts)}"
            )
        else:
            logger.debug(
                f"ActivationBuffer: reserved={reserved_memory:.4f}GB, "
                f"used={total_max_used / BYTES_PER_GB:.4f}GB, "
                f"slots(used/total): {', '.join(slot_parts)}"
            )
        if has_overflow:
            raise ValueError(
                self._build_overflow_message(
                    peak_values, scratch_min_free, scratch_size, reserved_memory
                )
            )

        return scratch_info

    def _build_overflow_message(
        self,
        peak_values: list[int],
        scratch_min_free: float,
        scratch_size: float,
        reserved_memory: float,
    ) -> str:
        """Build a detailed overflow error message with actionable suggestions.

        Handles activation-only, scratch-only, and combined overflows. Saved
        state and scratch use independent public controls, so the message never
        conflates the two byte budgets.

        Args:
            peak_values: Peak free-space counters from one bulk D2H transfer.
            scratch_min_free: Peak min free bytes in scratch region (negative = overflow).
            scratch_size: Total scratch region size in bytes.
            reserved_memory: Total buffer size in GB.

        Returns:
            Formatted error message string.
        """
        BYTES_PER_GB = 1024 * 1024 * 1024
        N = self.num_activation_slots

        # Identify which slots overflowed.
        overflow_details = []
        max_activation_overflow = 0.0
        for i in range(N):
            if peak_values[i] < 0:
                overflow_bytes = -peak_values[i]
                overflow_details.append(
                    f"Activation slot {i}: oversubscribed by "
                    f"{overflow_bytes / BYTES_PER_GB:.4f} GBytes"
                )
                max_activation_overflow = max(max_activation_overflow, overflow_bytes)
        scratch_overflow = 0.0
        if scratch_min_free < 0:
            scratch_overflow = abs(scratch_min_free)
            overflow_details.append(
                f"Scratch region: oversubscribed by "
                f"{scratch_overflow / BYTES_PER_GB:.4f} GBytes"
            )

        # Build actionable suggestions.
        is_vmm = self.device_scratch_size is not None
        adjustments = []

        # 1. Scratch overflow -> suggest a new capacity factor
        if (
            scratch_overflow > 0
            and self.scratch_capacity_factor is not None
            and scratch_size > 0
        ):
            ratio = (scratch_size + scratch_overflow) / scratch_size
            suggested_factor = self.scratch_capacity_factor * ratio * 1.25
            if is_vmm:
                knob = "config.vmm.total_scratch_capacity_factor"
            else:
                knob = "config.device_scratch_capacity_factor"
            adjustments.append(
                f"{knob}: {self.scratch_capacity_factor:.1f} -> {suggested_factor:.1f}"
            )

        # 2. Activation overflow -> increase each slot's capacity.
        suggested_slot_bytes = self.activation_slot_bytes
        if max_activation_overflow > 0:
            suggested_slot_bytes = _align_bytes(
                int(self.activation_slot_bytes + max_activation_overflow * 1.25)
            )

        # 3. Suggest the public activation budget if it needs to increase.
        if suggested_slot_bytes > self.activation_slot_bytes:
            adjustments.append(
                f"config.activation_slot_bytes: {self.activation_slot_bytes} "
                f"({self.activation_slot_bytes / BYTES_PER_GB:.2f} GB) -> "
                f"{suggested_slot_bytes} "
                f"({suggested_slot_bytes / BYTES_PER_GB:.2f} GB)"
            )

        suggestion_block = ""
        if adjustments:
            numbered = "\n".join(
                f"  {i + 1}. {adj}" for i, adj in enumerate(adjustments)
            )
            suggestion_block = f"\nMake the following adjustments:\n{numbered}"

        return (
            "Activation buffer overflowed, numerics are corrupted. "
            f"Reserved memory: {reserved_memory:.4f} GBytes. "
            f"Overflow details: {'; '.join(overflow_details)}."
            f"{suggestion_block}"
        )


@dataclasses.dataclass(frozen=True)
class BlockscaledStorageConfig:
    """Storage geometry for block-scaled activation operands.

    A block-scaled operand is stored as a *bundle*: one contiguous allocation
    holding quantized data (qdata) immediately followed by its scale factors.
    Each scale factor covers a contiguous vector of ``sf_vec_size`` elements
    along the quantized axis.

    A block-scaled matmul quantizes along its contraction (K) axis, so the same
    activation needs two independently laid-out bundles depending on how a later
    GEMM consumes it:

    - Row-quantized: scales blocked along the feature dim (K = columns). Used
      when the tensor is the moving operand of a forward or DGRAD matmul, e.g.
      x_gathered into FC13 or h2 into FC2.
    - Column-quantized: scales blocked along the row/token dim (K = rows). Used
      when the tensor is reused *transposed* as a WGRAD operand ``grad^T @ act``,
      e.g. x_gathered and h2 into the FC13/FC2 weight gradients.

    The two layouts are not interchangeable, so a tensor consumed both ways is
    materialized as both a row bundle and a column bundle.

    Fields:
        operand_element_size: bytes per quantized storage element.
        scale_element_size: bytes per scale factor (e.g. 1 for E8M0 / E4M3).
        sf_vec_size: elements per scale block along the quantized axis
            (e.g. 32 for MXFP8, 16 for NVFP4).
        operand_values_per_storage_element: logical operand values packed into
            each quantized storage element (e.g. 2 for FP4).
        row_global_scale_element_size: bytes of per-row payload appended after
            row scale factors (e.g. 4 for NVFP4 token inverse scales).
    """

    operand_element_size: int
    scale_element_size: int
    sf_vec_size: int
    operand_values_per_storage_element: int = 1
    row_global_scale_element_size: int = 0
    row_scale_storage_multiple: int = ACTIVATION_ROW_SCALE_STORAGE_MULTIPLE

    def __post_init__(self) -> None:
        """Validate storage element and scale-vector sizes."""
        if (
            min(
                self.operand_element_size,
                self.scale_element_size,
                self.sf_vec_size,
                self.operand_values_per_storage_element,
                self.row_scale_storage_multiple,
            )
            <= 0
        ):
            raise ValueError("block-scaled storage sizes must be positive")
        if self.row_global_scale_element_size < 0:
            raise ValueError("row global scale element size must be nonnegative")

    def row_quant_sizes(self, rows: int, dim: int) -> tuple[int, int]:
        """(qdata, scales) byte sizes of a row-quantized [rows, dim] bundle.

        Scales are blocked along ``dim``: one scale per sf_vec_size columns,
        i.e. ``rows * ceil(dim / sf_vec_size)`` scale factors.

        Args:
            rows: Number of matrix rows.
            dim: Logical row width.

        Returns:
            Aligned qdata and scale byte counts.
        """
        qdata = _align_bytes(
            rows
            * (
                (dim + self.operand_values_per_storage_element - 1)
                // self.operand_values_per_storage_element
            )
            * self.operand_element_size
        )
        scales = _align_bytes(
            self._row_block_scale_size(rows, dim)
            + rows * self.row_global_scale_element_size
        )
        return qdata, scales

    def _row_block_scale_size(self, rows: int, dim: int) -> int:
        """Return unaligned bytes occupied by row-oriented block scales.

        Args:
            rows: Number of matrix rows.
            dim: Logical row width.

        Returns:
            Unaligned scale byte count.
        """
        return (
            rows
            * self.row_scale_storage_multiple
            * ((dim + self.sf_vec_size - 1) // self.sf_vec_size)
            * self.scale_element_size
        )

    def row_global_scale_offset(self, rows: int, dim: int) -> int | None:
        """Return the bundle-relative per-row global-scale offset.

        Args:
            rows: Number of matrix rows.
            dim: Logical row width.

        Returns:
            Byte offset, or ``None`` when global scales are absent.
        """
        if self.row_global_scale_element_size == 0:
            return None
        qdata_size, _ = self.row_quant_sizes(rows, dim)
        return qdata_size + self._row_block_scale_size(rows, dim)

    def col_quant_sizes(self, rows: int, dim: int) -> tuple[int, int]:
        """(qdata, scales) byte sizes of a column-quantized [rows, dim] bundle.

        Scales are blocked along ``rows``: one scale per sf_vec_size rows,
        i.e. ``dim * ceil(rows / sf_vec_size)`` scale factors.

        Args:
            rows: Number of matrix rows.
            dim: Logical row width.

        Returns:
            Aligned qdata and scale byte counts.
        """
        qdata = _align_bytes(
            dim
            * (
                (rows + self.operand_values_per_storage_element - 1)
                // self.operand_values_per_storage_element
            )
            * self.operand_element_size
        )
        scales = _align_bytes(
            dim
            * ((rows + self.sf_vec_size - 1) // self.sf_vec_size)
            * self.scale_element_size
        )
        return qdata, scales

    def row_quant_size(self, rows: int, dim: int) -> int:
        """Return total bytes for a row-quantized operand.

        Args:
            rows: Number of matrix rows.
            dim: Logical row width.

        Returns:
            Aligned qdata plus scale bytes.
        """
        return sum(self.row_quant_sizes(rows, dim))

    def packed_dispatch_size(self, rows: int, dim: int) -> int:
        """Return bytes for inference-style row-interleaved dispatch storage.

        Args:
            rows: Number of routed rows.
            dim: Logical row width.

        Returns:
            Aligned packed-dispatch byte count.
        """
        q_cols = (
            dim + self.operand_values_per_storage_element - 1
        ) // self.operand_values_per_storage_element
        scale_cols = (dim + self.sf_vec_size - 1) // self.sf_vec_size
        row_bytes = (
            q_cols * self.operand_element_size
            + scale_cols * self.scale_element_size
            + self.row_global_scale_element_size
        )
        row_stride = (
            (row_bytes + BLOCKSCALED_DISPATCH_ROW_ALIGNMENT_BYTES - 1)
            // (BLOCKSCALED_DISPATCH_ROW_ALIGNMENT_BYTES)
            * BLOCKSCALED_DISPATCH_ROW_ALIGNMENT_BYTES
        )
        return _align_bytes(rows * row_stride)

    def col_quant_size(self, rows: int, dim: int) -> int:
        """Return total bytes for a column-quantized operand.

        Args:
            rows: Number of matrix rows.
            dim: Logical row width.

        Returns:
            Aligned qdata plus scale bytes.
        """
        return sum(self.col_quant_sizes(rows, dim))


def _plan_max_recv_tokens(
    *,
    num_tokens: int,
    topk: int,
    imbalance_factor: float,
    num_local_experts: int,
    routing_world_size: int | None,
    routing_m_multiple_of: int | None,
    max_num_recv_tokens: int | None,
) -> int:
    """Return a topology- and padding-aware received-row capacity.

    Args:
        num_tokens: Input tokens local to each routing rank.
        topk: Experts selected by each input token.
        imbalance_factor: Maximum receive imbalance relative to balanced routing.
        num_local_experts: Experts owned by the receiving rank.
        routing_world_size: Number of ranks participating in routing.
        routing_m_multiple_of: Independent row-padding multiple for each expert.
        max_num_recv_tokens: Optional caller-provided capacity ceiling.

    Returns:
        Maximum padded rows that the receiving rank may consume.
    """
    capacity = int(num_tokens * topk * imbalance_factor + 0.5)
    if routing_world_size is not None:
        topology_capacity = (
            num_tokens * routing_world_size * min(topk, num_local_experts)
        )
        capacity = min(capacity, topology_capacity)
    if routing_m_multiple_of is not None:
        # Each local expert is padded independently. The sum of padded groups
        # is itself aligned, so flooring this bound remains always sufficient.
        padded_upper_bound = capacity + num_local_experts * (routing_m_multiple_of - 1)
        capacity = padded_upper_bound // routing_m_multiple_of * routing_m_multiple_of
    if max_num_recv_tokens is not None:
        capacity = min(capacity, max_num_recv_tokens)
    return capacity


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Configuration for MoE model memory calculations.

    Encapsulates all parameters needed to calculate memory sizes
    for forward pass, backward pass, and saved activations.
    """

    # Dtype of the activation tensors
    dtype: torch.dtype
    # Hidden dimension for the MoE layer
    hidden_dim: int
    # Intermediate dimension for the MoE layer
    intermediate_dim: int
    # Number of input tokens per rank to the MoE layer
    num_tokens: int
    # Number of experts selected per token
    topk: int
    # Maximum imbalance factor for token distribution
    max_imbalance_factor: float
    # Number of MoE layers in the model (for minimum buffer size estimation)
    num_moe_layers: int = 1
    # None for native BF16/FP16 activations. Block-scaled async kernels use
    # separate row-quantized compute operands and column-quantized WGRAD operands.
    blockscaled_storage: BlockscaledStorageConfig | None = None
    # Block-scaled routing pads every local expert independently.
    num_local_experts: int = 0
    routing_m_multiple_of: int | None = None
    # Optional static upper bound for the padded received-row capacity.
    max_num_recv_tokens: int | None = None
    # Mega kernels consume x_gathered and h2 row quants in one fused launch, so
    # their planner slots must be disjoint. Staged kernels serialize them. This
    # flag affects only block-scaled planning.
    mega: bool = False
    # Expert-parallel group size used to cap received rows by routing topology.
    routing_world_size: int | None = None
    # ``;bwd:<dtype>``: the backward runs the native GEMMs on operands
    # dequantized from the forward's *row* quants. That inverts which bundles are
    # persistent -- row instead of column -- and makes the backward's working set
    # dense, so both halves of the plan change shape. Block-scaled planning only.
    native_backward: bool = False
    # Interleaved FC13 consumes its epilogue output directly and therefore does
    # not materialize dense h1. This inference-only layout requires Mega.
    interleaved_fc13: bool = False

    def __post_init__(self) -> None:
        """Validate topology-aware block-scaled planning controls."""
        if self.max_num_recv_tokens is not None and self.max_num_recv_tokens <= 0:
            raise ValueError("max_num_recv_tokens must be positive")
        if self.routing_world_size is not None:
            if self.routing_world_size <= 0:
                raise ValueError("routing_world_size must be positive")
            if self.num_local_experts <= 0:
                raise ValueError(
                    "routing_world_size requires a positive num_local_experts"
                )
        storage = self.blockscaled_storage
        if self.interleaved_fc13 and (storage is None or not self.mega):
            raise ValueError(
                "interleaved_fc13 requires block-scaled storage and mega=True"
            )
        if storage is None:
            return
        self._validate_blockscaled_storage(storage)

    def _validate_blockscaled_storage(
        self,
        storage: BlockscaledStorageConfig,
    ) -> None:
        """Validate dimensions used by block-scaled planner arithmetic.

        Args:
            storage: Static low-precision storage geometry.

        Raises:
            ValueError: If routing or feature dimensions violate scale-vector
                alignment requirements.
        """
        if self.num_local_experts <= 0:
            raise ValueError(
                "block-scaled storage requires a positive num_local_experts"
            )
        if self.routing_m_multiple_of is None or self.routing_m_multiple_of <= 0:
            raise ValueError(
                "block-scaled storage requires a positive routing_m_multiple_of"
            )
        if self.routing_m_multiple_of % storage.sf_vec_size != 0:
            raise ValueError(
                "routing_m_multiple_of must be divisible by the block-scaled "
                f"sf_vec_size: {self.routing_m_multiple_of} vs {storage.sf_vec_size}"
            )
        if (
            self.max_num_recv_tokens is not None
            and self.max_num_recv_tokens % self.routing_m_multiple_of != 0
        ):
            raise ValueError(
                "max_num_recv_tokens must be divisible by routing_m_multiple_of: "
                f"{self.max_num_recv_tokens} vs {self.routing_m_multiple_of}"
            )
        for name, dim in (
            ("hidden_dim", self.hidden_dim),
            ("intermediate_dim", self.intermediate_dim),
        ):
            if dim % storage.sf_vec_size != 0:
                raise ValueError(
                    f"{name} must be divisible by the block-scaled sf_vec_size: "
                    f"{dim} vs {storage.sf_vec_size}"
                )

    @property
    def element_size(self) -> int:
        """Size of each element in bytes (e.g., 2 for bfloat16)."""
        return self.dtype.itemsize

    def fwd_act_mem_size(self, num_recv_tokens: int, recompute: bool) -> int:
        """Calculate memory size required during forward pass.

        With blockscaled_storage, row- and column-quantized bundles replace the
        dense gathered temporaries described below.

        With recompute=True:
        - Saved (front to back): x [num_tokens, hidden_dim]
        - Temporary (back to front): x_gathered, h1, h2
        - Peak = x + x_gathered + h1 + h2

        With recompute=False:
        - Saved (front to back): x_gathered, h1, h3 [num_tokens * topk, hidden_dim]
        - Temporary (back to front): h2
        - Peak = x_gathered + h1 + h3 + h2

        Args:
            num_recv_tokens: Number of tokens received.
            recompute: Whether recompute is enabled.

        Returns:
            Memory size in bytes with 128-byte alignment.
        """
        if self.blockscaled_storage is not None:
            return self._blockscaled_fwd_act_mem_size(num_recv_tokens, recompute)

        x_gathered_size = _align_bytes(
            num_recv_tokens * self.hidden_dim * self.element_size
        )
        h1_size = _align_bytes(
            num_recv_tokens * 2 * self.intermediate_dim * self.element_size
        )
        h2_size = _align_bytes(
            num_recv_tokens * self.intermediate_dim * self.element_size
        )

        if recompute:
            # With recompute: save x, temps are x_gathered + h1 + h2
            x_size = _align_bytes(self.num_tokens * self.hidden_dim * self.element_size)
            return x_size + x_gathered_size + h1_size + h2_size
        else:
            # Without recompute: save x_gathered + h1 + h3, temp is h2
            h3_size = _align_bytes(
                self.num_tokens * self.topk * self.hidden_dim * self.element_size
            )
            return x_gathered_size + h1_size + h3_size + h2_size

    def bwd_act_mem_size(self, num_recv_tokens: int, recompute: bool) -> int:
        """Calculate memory size required during backward pass.

        With blockscaled_storage, this uses the bundled row- and
        column-quantized layout from _blockscaled_bwd_act_mem_size.

        Liveness-based memory reuse (both recompute and no-recompute):
        - h2 overlaps with grad_h1 (h2 alive 1-6, grad_h1 alive 7-9)
        - grad_h3_gathered placed between grad_h2 and h2 when hidden <= intermediate
          (no overlap with either), below grad_h2 when hidden > intermediate.

        Scratch layout (top to bottom):
            grad_h1 (top, overlaps h2) | grad_h2 | [grad_h3_gathered] | ...
        With recompute, h1 and x_gathered are also in scratch below the gradients.

        Args:
            num_recv_tokens: Number of tokens received.
            recompute: Whether recompute is enabled.

        Returns:
            Memory size in bytes with 128-byte alignment.
        """
        if self.blockscaled_storage is not None:
            return self._blockscaled_bwd_act_mem_size(num_recv_tokens, recompute)

        grad_h1_size = _align_bytes(
            num_recv_tokens * 2 * self.intermediate_dim * self.element_size
        )
        grad_h2_size = _align_bytes(
            num_recv_tokens * self.intermediate_dim * self.element_size
        )
        grad_h3_gathered_size = _align_bytes(
            num_recv_tokens * self.hidden_dim * self.element_size
        )

        # h2 overlaps with grad_h1 (h2_size <= grad_h1_size always).
        # grad_h3_gathered fits between grad_h2 and h2 when hidden <= intermediate
        # (zero extra cost), otherwise goes below grad_h2.
        grad_scratch = grad_h1_size + grad_h2_size
        if self.hidden_dim > self.intermediate_dim:
            grad_scratch += grad_h3_gathered_size

        if recompute:
            x_gathered_size = _align_bytes(
                num_recv_tokens * self.hidden_dim * self.element_size
            )
            h1_size = _align_bytes(
                num_recv_tokens * 2 * self.intermediate_dim * self.element_size
            )
            return x_gathered_size + h1_size + grad_scratch
        else:
            return grad_scratch

    def saved_act_mem_size(self, num_recv_tokens: int, recompute: bool) -> int:
        """Calculate memory size for saved activations.

        With blockscaled_storage and no recompute, x and h2 are saved as
        column-quantized bundles.

        With recompute=True, saved activations:
        - x: [num_tokens, hidden_dim]

        With recompute=False, saved activations:
        - x_gathered: [num_recv_tokens, hidden_dim]
        - h1: [num_recv_tokens, 2 * intermediate_dim]
        - h3_saved: [num_tokens * topk, hidden_dim]

        Args:
            num_recv_tokens: Number of tokens received.
            recompute: Whether recompute is enabled.

        Returns:
            Memory size in bytes with 128-byte alignment.
        """
        if recompute:
            if self.blockscaled_storage is not None and self.native_backward:
                x_size = self.blockscaled_storage.packed_dispatch_size(
                    self.num_tokens, self.hidden_dim
                )
            else:
                x_size = _align_bytes(
                    self.num_tokens * self.hidden_dim * self.element_size
                )
            return x_size
        elif self.blockscaled_storage is not None:
            return self._blockscaled_no_recompute_saved_size(num_recv_tokens)
        else:
            # x_gathered + h1 + h3_saved
            x_gathered_size = _align_bytes(
                num_recv_tokens * self.hidden_dim * self.element_size
            )
            h1_size = _align_bytes(
                num_recv_tokens * 2 * self.intermediate_dim * self.element_size
            )
            h3_size = _align_bytes(
                self.num_tokens * self.topk * self.hidden_dim * self.element_size
            )
            return x_gathered_size + h1_size + h3_size

    def _blockscaled_row_scratch_size(
        self, num_recv_tokens: int, recompute: bool
    ) -> int:
        """Bytes of row-quant scratch at the backward peak.

        Without recompute, only the serialized grad_h3 and dxy producers need
        this slot. With recompute, staged kernels share the slot across every
        row operand, while Mega co-feeds x_gathered and h2 and reserves adjacent
        sub-slots for that pair.

        Args:
            num_recv_tokens: Capacity-padded received rows.
            recompute: Whether forward operands are rebuilt in backward.

        Returns:
            Required scratch bytes.
        """
        storage = self.blockscaled_storage
        assert storage is not None
        x_row = storage.row_quant_size(num_recv_tokens, self.hidden_dim)
        dxy_row = storage.row_quant_size(num_recv_tokens, 2 * self.intermediate_dim)
        if not recompute:
            return max(x_row, dxy_row)
        h2_row = storage.row_quant_size(num_recv_tokens, self.intermediate_dim)
        if self.mega:
            return max(x_row + h2_row, dxy_row)
        return max(x_row, h2_row, dxy_row)

    def _blockscaled_no_recompute_saved_size(self, num_recv_tokens: int) -> int:
        """Bytes persisted in the activation slot when a layer skips recompute.

        These live from forward until the matching backward reads them, summed
        below in order: x_gathered column bundle (FC13 WGRAD), h1 dense (SwiGLU
        backward), h2 column bundle (FC2 WGRAD), and h3 dense (combine).

        Args:
            num_recv_tokens: Capacity-padded received rows.

        Returns:
            Required saved-activation bytes.
        """
        storage = self.blockscaled_storage
        assert storage is not None
        # With native_backward the persisted bundles are the *row* quants the
        # forward GEMMs consumed rather than the column quants a block-scaled
        # WGRAD needs. Both layouts hold the same element count, so the total is
        # nearly identical; only alignment padding differs.
        bundle = (
            storage.row_quant_size if self.native_backward else storage.col_quant_size
        )
        return (
            bundle(num_recv_tokens, self.hidden_dim)
            + _align_bytes(
                num_recv_tokens * 2 * self.intermediate_dim * self.element_size
            )
            + bundle(num_recv_tokens, self.intermediate_dim)
            + _align_bytes(
                self.num_tokens * self.topk * self.hidden_dim * self.element_size
            )
        )

    def _blockscaled_fwd_row_scratch_size(self, num_recv_tokens: int) -> int:
        """Return row-quantized scratch bytes at the forward peak.

        Args:
            num_recv_tokens: Capacity-padded received rows.

        Returns:
            Required scratch bytes.
        """
        storage = self.blockscaled_storage
        assert storage is not None
        x_row = storage.row_quant_size(num_recv_tokens, self.hidden_dim)
        h2_row = storage.row_quant_size(num_recv_tokens, self.intermediate_dim)
        return x_row + h2_row if self.mega else max(x_row, h2_row)

    def _interleaved_h2_staging_size(self, num_recv_tokens: int) -> int:
        """Return dense H2 staging bytes required by interleaved NVFP4.

        Args:
            num_recv_tokens: Capacity-padded received rows.

        Returns:
            Required staging bytes.
        """
        storage = self.blockscaled_storage
        assert storage is not None
        if storage.row_global_scale_element_size == 0:
            return 0
        return _align_bytes(num_recv_tokens * self.intermediate_dim * self.element_size)

    def _blockscaled_inference_scratch_size(self, num_recv_tokens: int) -> int:
        """Return the exact scratch carve used by block-scaled inference.

        Args:
            num_recv_tokens: Capacity-padded received rows.

        Returns:
            Required scratch bytes.
        """
        row_scratch = self._blockscaled_fwd_row_scratch_size(num_recv_tokens)
        if self.interleaved_fc13:
            return row_scratch + self._interleaved_h2_staging_size(num_recv_tokens)
        return row_scratch + _align_bytes(
            num_recv_tokens * 2 * self.intermediate_dim * self.element_size
        )

    def _blockscaled_fwd_act_mem_size(
        self, num_recv_tokens: int, recompute: bool
    ) -> int:
        """Peak forward bytes = persistent tensors + transient row scratch.

        Staged kernels serialize x_gathered and h2 and share a max-sized row
        slot. A Mega launch consumes both operands together, so it reserves two
        adjacent sub-slots. With recompute, native backward saves the packed
        inference dispatch input; other modes save dense x.

        Args:
            num_recv_tokens: Capacity-padded received rows.
            recompute: Whether backward will recompute forward operands.

        Returns:
            Persistent activation plus transient scratch bytes.
        """
        storage = self.blockscaled_storage
        assert storage is not None
        row_scratch = self._blockscaled_fwd_row_scratch_size(num_recv_tokens)
        x_size = _align_bytes(self.num_tokens * self.hidden_dim * self.element_size)
        h1_size = _align_bytes(
            num_recv_tokens * 2 * self.intermediate_dim * self.element_size
        )
        if self.native_backward:
            if recompute:
                packed_x = storage.packed_dispatch_size(
                    self.num_tokens, self.hidden_dim
                )
                return packed_x + h1_size + row_scratch
            # The row bundles are persisted directly in the activation slot.
            return self.saved_act_mem_size(num_recv_tokens, False)
        if recompute:
            x_col_size = storage.col_quant_size(num_recv_tokens, self.hidden_dim)
            h2_col_size = storage.col_quant_size(num_recv_tokens, self.intermediate_dim)
            return x_size + h1_size + x_col_size + h2_col_size + row_scratch
        # No recompute: the x_gathered column bundle, dense h1, h2 column bundle,
        # and dense h3 persist; the row slot is the only extra scratch cost.
        return self.saved_act_mem_size(num_recv_tokens, False) + row_scratch

    def _blockscaled_bwd_act_mem_size(
        self, num_recv_tokens: int, recompute: bool
    ) -> int:
        """Peak backward bytes.

        Always live: the row scratch plus the gradient bundles the backward
        produces -- grad_h3 (col, for FC2 WGRAD), grad_h2 (dense), and dxy=grad_h1
        (col, for FC13 WGRAD). With recompute the forward operands must also be
        rebuilt into scratch: x_gathered (col), h1 (dense), and h2 (col). Without
        recompute those three were saved in the region during forward instead.

        Args:
            num_recv_tokens: Capacity-padded received rows.
            recompute: Whether forward operands are rebuilt in backward.

        Returns:
            Peak backward bytes.
        """
        storage = self.blockscaled_storage
        assert storage is not None
        if self.native_backward:
            return self._native_backward_bwd_act_mem_size(num_recv_tokens, recompute)
        row_scratch = self._blockscaled_row_scratch_size(num_recv_tokens, recompute)
        grad_h2_size = _align_bytes(
            num_recv_tokens * self.intermediate_dim * self.element_size
        )
        grad_h3_col_size = storage.col_quant_size(num_recv_tokens, self.hidden_dim)
        dxy_col_size = storage.col_quant_size(
            num_recv_tokens, 2 * self.intermediate_dim
        )
        if not recompute:
            return row_scratch + grad_h3_col_size + grad_h2_size + dxy_col_size

        # Recompute rebuilds the forward WGRAD operands into scratch.
        x_col_size = storage.col_quant_size(num_recv_tokens, self.hidden_dim)
        h1_size = _align_bytes(
            num_recv_tokens * 2 * self.intermediate_dim * self.element_size
        )
        h2_col_size = storage.col_quant_size(num_recv_tokens, self.intermediate_dim)
        return (
            row_scratch
            + x_col_size
            + h1_size
            + h2_col_size
            + grad_h3_col_size
            + grad_h2_size
            + dxy_col_size
        )

    def _native_backward_bwd_act_mem_size(
        self, num_recv_tokens: int, recompute: bool
    ) -> int:
        """Peak backward bytes when the backward runs the native GEMMs.

        The working set is dense, because the row bundles are dequantized before
        anything else runs: x_gathered and h2 are the dequantization
        destinations, and grad_h3_gathered / grad_h2 / grad_h1 are the native
        gradients. With recompute the forward is rerun first, so the row bundles
        and h1 have to be rebuilt in scratch too.

        Args:
            num_recv_tokens: Capacity-padded received rows.
            recompute: Whether forward operands are rebuilt in backward.

        Returns:
            Peak backward bytes.
        """
        storage = self.blockscaled_storage
        assert storage is not None
        x_gathered_size = _align_bytes(
            num_recv_tokens * self.hidden_dim * self.element_size
        )
        h2_size = _align_bytes(
            num_recv_tokens * self.intermediate_dim * self.element_size
        )
        h1_size = _align_bytes(
            num_recv_tokens * 2 * self.intermediate_dim * self.element_size
        )
        # grad_h3_gathered has x_gathered's shape, grad_h2 has h2's, grad_h1 h1's.
        dense = 2 * x_gathered_size + 2 * h2_size + h1_size
        if not recompute:
            return dense
        x_row = storage.row_quant_size(num_recv_tokens, self.hidden_dim)
        h2_row = storage.row_quant_size(num_recv_tokens, self.intermediate_dim)
        dxy_row = storage.row_quant_size(num_recv_tokens, 2 * self.intermediate_dim)
        row_scratch = max(x_row + h2_row, dxy_row)
        return dense + h1_size + row_scratch

    @functools.cached_property
    def scratch_mem_size(self) -> int:
        """Calculate maximum scratch memory size for MoE backward with recompute.

        This computes the memory needed for backward pass with recompute,
        using the maximum possible num_recv_tokens based on imbalance factor.

        Returns:
            Maximum scratch memory size in bytes required for dynamic activations.
        """
        max_recv_tokens = self.max_recv_tokens
        return self.bwd_act_mem_size(max_recv_tokens, recompute=True)

    @functools.cached_property
    def inference_scratch_mem_size(self) -> int:
        """Return forward-only scratch bytes at the configured imbalance."""
        if self.blockscaled_storage is not None:
            return self._blockscaled_inference_scratch_size(self.max_recv_tokens)
        saved_x = self.saved_act_mem_size(0, recompute=True)
        return self.fwd_act_mem_size(self.max_recv_tokens, recompute=True) - saved_x

    @functools.cached_property
    def scratch_only_mem_size(self) -> int:
        """Return scratch bytes required by a grad-free BF16 forward.

        Returns:
            Required shared-scratch capacity in bytes.
        """
        return self.min_buffer_size(num_activation_slots=0)

    @functools.cached_property
    def max_recv_tokens(self) -> int:
        """Row capacity the planner reserves for received tokens (per layer)."""
        return self._max_recv_tokens(self.max_imbalance_factor)

    def _max_recv_tokens(self, imbalance_factor: float) -> int:
        """Return the padded receive-row capacity for an imbalance factor.

        Args:
            imbalance_factor: Capacity relative to balanced routed rows.

        Returns:
            Topology-clamped and kernel-padded row capacity.
        """
        return _plan_max_recv_tokens(
            num_tokens=self.num_tokens,
            topk=self.topk,
            imbalance_factor=imbalance_factor,
            num_local_experts=self.num_local_experts,
            routing_world_size=self.routing_world_size,
            routing_m_multiple_of=(
                self.routing_m_multiple_of
                if self.blockscaled_storage is not None
                else None
            ),
            max_num_recv_tokens=self.max_num_recv_tokens,
        )

    def host_scratch_mem_size(self, effective_imbalance: float) -> int:
        """Calculate host scratch memory size for VMM layout.

        When using a VMM [device, host, device] buffer layout, the host section
        provides overflow scratch capacity for worst-case token imbalance beyond
        what the device scratch (scratch_mem_size) covers.

        Args:
            effective_imbalance: Effective imbalance factor, typically
                min(vmm_worst_case_imbalance, ep_size).

        Returns:
            Host scratch size in bytes (>= 0). Returns 0 if the device scratch
            already covers the worst-case imbalance.
        """
        worst_recv_tokens = self._max_recv_tokens(effective_imbalance)
        worst_scratch = self.bwd_act_mem_size(worst_recv_tokens, recompute=True)
        return max(0, worst_scratch - self.scratch_mem_size)

    def host_inference_scratch_mem_size(self, effective_imbalance: float) -> int:
        """Return forward-only overflow scratch beyond the device reservation.

        Args:
            effective_imbalance: Imbalance factor covered by device plus host.

        Returns:
            Additional host-mapped bytes required for inference.
        """
        worst_recv_tokens = self._max_recv_tokens(effective_imbalance)
        saved_x = self.saved_act_mem_size(0, recompute=True)
        worst_scratch = (
            self.fwd_act_mem_size(worst_recv_tokens, recompute=True) - saved_x
        )
        return max(0, worst_scratch - self.inference_scratch_mem_size)

    def min_buffer_size(self, num_activation_slots: int = 1) -> int:
        """Minimum activation buffer size assuming all layers recompute.

        This is the lower bound: every MoE layer saves only its input x
        (recompute mode), plus scratch memory for one backward pass at
        worst-case token imbalance.

        For N activation slots (pipeline parallelism):
            N * (num_moe_layers * saved_x_per_layer) + scratch_mem_size

        A buffer at this size means every layer will recompute. Larger buffers
        allow some layers to skip recompute (saving full activations instead),
        trading memory for speed.

        Args:
            num_activation_slots: Number of concurrent activation slots.
                Use N=0 for scratch-only inference buffers.

        Returns:
            Minimum buffer size in bytes.
        """
        # In recompute mode, each layer saves only x: [num_tokens, hidden_dim]
        saved_x_per_layer = self.saved_act_mem_size(0, recompute=True)
        if num_activation_slots == 0:
            return self.inference_scratch_mem_size
        return (
            num_activation_slots * self.num_moe_layers * saved_x_per_layer
            + self.scratch_mem_size
        )

    def max_buffer_size(self, num_activation_slots: int = 1) -> int:
        """Maximum useful activation buffer size assuming no layers recompute.

        This is the upper bound: every MoE layer saves full activations
        (x_gathered, h1, h3) at max_imbalance_factor (chunk_capacity_factor)
        token load, plus scratch memory for one backward pass.

        Above this size, additional buffer memory provides no benefit.

        Uses max_imbalance_factor (= chunk_capacity_factor) to estimate the
        number of received tokens per layer.

        Args:
            num_activation_slots: Number of concurrent activation slots.

        Returns:
            Maximum useful buffer size in bytes.
        """
        if num_activation_slots == 0:
            return self.inference_scratch_mem_size
        recv_tokens = self.max_recv_tokens
        saved_full_per_layer = self.saved_act_mem_size(recv_tokens, recompute=False)
        return (
            num_activation_slots * self.num_moe_layers * saved_full_per_layer
            + self.scratch_mem_size
        )


def validate_buffer_capacity(
    buffer_size: int,
    model_config: ModelConfig,
    num_activation_slots: int = 1,
) -> None:
    """Validate that buffer can hold all MoE layers in recompute mode.

    This is a host-side check that raises an exception early if the buffer is
    too small. The minimum required size is:
        N * (num_moe_layers * saved_x_per_layer) + scratch_mem_size
    where N is num_activation_slots. At this size every layer recomputes;
    larger buffers allow some layers to skip recompute.

    Args:
        buffer_size: Size of the activation buffer in bytes.
        model_config: ModelConfig with model dimensions, max_imbalance_factor,
            and num_moe_layers.
        num_activation_slots: Number of activation slots. Use zero to
            validate only scratch capacity for inference.

    Raises:
        ValueError: If buffer_size < min_buffer_size.
    """
    min_size = model_config.min_buffer_size(num_activation_slots=num_activation_slots)
    if buffer_size < min_size:
        if num_activation_slots == 0:
            raise ValueError(
                f"Activation buffer too small for inference: buffer_size={buffer_size} bytes, "
                f"but minimum required is {min_size} bytes scratch. "
                f"max_imbalance_factor={model_config.max_imbalance_factor}, "
                f"num_tokens={model_config.num_tokens}, topk={model_config.topk}."
            )
        raise ValueError(
            f"Activation buffer too small: buffer_size={buffer_size} bytes, "
            f"but minimum required is {min_size} bytes "
            f"({num_activation_slots} slots * {model_config.num_moe_layers} MoE layers "
            f"* {model_config.saved_act_mem_size(0, recompute=True)} bytes/layer "
            f"+ {model_config.scratch_mem_size} bytes scratch). "
            f"max_imbalance_factor={model_config.max_imbalance_factor}, "
            f"num_tokens={model_config.num_tokens}, topk={model_config.topk}."
        )
    if num_activation_slots == 0:
        return
    scratch_mem_size = model_config.scratch_mem_size
    activation_slot_bytes = (
        (buffer_size - scratch_mem_size)
        // num_activation_slots
        // MEMORY_ALIGNMENT
        * MEMORY_ALIGNMENT
    )
    if activation_slot_bytes <= 0:
        raise ValueError(
            f"Activation buffer too small for {num_activation_slots} activation slots: "
            f"buffer_size={buffer_size} bytes, scratch_mem_size={scratch_mem_size} bytes, "
            f"leaving {buffer_size - scratch_mem_size} bytes for "
            f"{num_activation_slots} slots (0 bytes per slot after alignment)."
        )


def validate_scratch_only_capacity(
    scratch_region_size: int,
    model_config: ModelConfig,
) -> None:
    """Validate that shared scratch can host a grad-free BF16 forward.

    Args:
        scratch_region_size: Bytes in the buffer's shared scratch region.
        model_config: Configuration used to size and plan the buffer.

    Raises:
        ValueError: If scratch cannot hold all transient forward activations.
    """
    required = model_config.scratch_only_mem_size
    if scratch_region_size < required:
        raise ValueError(
            "Activation buffer scratch region is too small for a grad-free "
            f"forward: {scratch_region_size=} bytes, required={required} bytes."
        )


@dataclasses.dataclass(frozen=True)
class BlockscaledForwardPlan:
    """Byte offsets of the block-scaled bundles used by a forward pass.

    Each field is a [1] int64 device tensor holding a byte offset into the
    activation buffer. "row"/"col" denote row- vs column-quantized bundles
    (see BlockscaledStorageConfig). The x/h2 row bundles share storage for
    staged kernels and occupy disjoint sub-slots for a Mega kernel.
    """

    # x_gathered, row-quantized: FC13 input.
    x_row_offset: torch.Tensor
    # x_gathered, column-quantized: FC13 weight-gradient operand.
    x_col_offset: torch.Tensor
    # h2, row-quantized: FC2 / combine input.
    h2_row_offset: torch.Tensor
    # h2, column-quantized: FC2 weight-gradient operand.
    h2_col_offset: torch.Tensor
    activation_offsets: torch.Tensor
    recompute_condition: torch.Tensor

    @property
    def dispatch_offsets(self) -> torch.Tensor:
        """Return the staged dispatch activation-offset slice."""
        return self.activation_offsets.narrow(
            0, FORWARD_DISPATCH_OFFSET_BASE, ACTIVATION_OFFSET_COUNT
        )

    @property
    def combine_offsets(self) -> torch.Tensor:
        """Return the staged combine activation-offset slice."""
        return self.activation_offsets.narrow(
            0, FORWARD_COMBINE_OFFSET_BASE, ACTIVATION_OFFSET_COUNT
        )

    @property
    def mega_offsets(self) -> torch.Tensor:
        """Return the Mega forward activation-offset slice."""
        return self.activation_offsets.narrow(
            0, FORWARD_DISPATCH_OFFSET_BASE, MEGA_ACTIVATION_OFFSET_COUNT
        )


@dataclasses.dataclass(frozen=True)
class ForwardPlan:
    """Plan for MoE forward pass execution.

    Contains the decision of whether to use recompute based on available
    activation buffer memory and the memory offsets for activation tensors.
    """

    # Whether to use recompute (True) or save activations (False), as a tensor
    need_recompute: torch.Tensor
    # Memory offset for x tensor (saved when recompute=True)
    x_offset: torch.Tensor
    # Memory offset for x_gathered tensor
    x_gathered_offset: torch.Tensor
    # Memory offset for h1 tensor
    h1_offset: torch.Tensor
    # Memory offset for h2 tensor
    h2_offset: torch.Tensor
    # Memory offset for h3_saved tensor (saved when recompute=False)
    h3_offset: torch.Tensor
    # Activation slot selected by forward and consumed by the matching backward.
    activation_slot_id_1: torch.Tensor
    # Owning allocation for the dense offset views above. Compiler boundaries
    # carry this tensor directly because functionalization need not preserve
    # the views' private ``_base`` references.
    packed_offsets: torch.Tensor | None = None
    # Present only for block-scaled activation-buffer plans.
    blockscaled: BlockscaledForwardPlan | None = None


@dataclasses.dataclass(frozen=True)
class BlockscaledBackwardPlan:
    """Byte offsets of the block-scaled bundles used by a backward pass.

    Each field is a [1] int64 device tensor holding a byte offset into the
    activation buffer. "row"/"col" denote row- vs column-quantized bundles
    (see BlockscaledStorageConfig). x/h2 offsets point at the operands saved in
    forward (no-recompute) or rebuilt in scratch (recompute). Recomputed x/h2
    row bundles share storage for staged kernels and are disjoint for Mega.
    """

    # x_gathered, row-quantized: FC13 DGRAD input (recompute only).
    x_row_offset: torch.Tensor
    # x_gathered, column-quantized: FC13 weight-gradient operand.
    x_col_offset: torch.Tensor
    # h2, row-quantized: FC2 DGRAD input (recompute only).
    h2_row_offset: torch.Tensor
    # h2, column-quantized: FC2 weight-gradient operand.
    h2_col_offset: torch.Tensor
    # Packed offsets consumed directly by the staged and Mega kernels.
    activation_offsets: torch.Tensor

    @property
    def recompute_dispatch_offsets(self) -> torch.Tensor:
        """Return offsets for staged W13 forward recomputation."""
        return self.activation_offsets.narrow(
            0,
            BACKWARD_RECOMPUTE_DISPATCH_OFFSET_BASE,
            ACTIVATION_OFFSET_COUNT,
        )

    @property
    def recompute_combine_offsets(self) -> torch.Tensor:
        """Return offsets for staged W2 forward recomputation."""
        return self.activation_offsets.narrow(
            0,
            BACKWARD_RECOMPUTE_COMBINE_OFFSET_BASE,
            ACTIVATION_OFFSET_COUNT,
        )

    @property
    def recompute_mega_offsets(self) -> torch.Tensor:
        """Return offsets for Mega forward recomputation."""
        return self.activation_offsets.narrow(
            0,
            BACKWARD_RECOMPUTE_DISPATCH_OFFSET_BASE,
            MEGA_ACTIVATION_OFFSET_COUNT,
        )

    @property
    def fc2_dgrad_offsets(self) -> torch.Tensor:
        """Return offsets for staged W2 DGRAD operands."""
        return self.activation_offsets.narrow(
            0, BACKWARD_FC2_DGRAD_OFFSET_BASE, ACTIVATION_OFFSET_COUNT
        )

    @property
    def fc2_wgrad_offsets(self) -> torch.Tensor:
        """Return offsets for staged W2 WGRAD operands."""
        return self.activation_offsets.narrow(
            0, BACKWARD_FC2_WGRAD_OFFSET_BASE, GEMM_OPERAND_OFFSET_COUNT
        )

    @property
    def fc2_mega_offsets(self) -> torch.Tensor:
        """Return offsets for fused W2 DGRAD and WGRAD operands."""
        return self.activation_offsets.narrow(
            0, BACKWARD_FC2_DGRAD_OFFSET_BASE, MEGA_ACTIVATION_OFFSET_COUNT
        )

    @property
    def fc13_dgrad_offsets(self) -> torch.Tensor:
        """Return offsets for staged W13 DGRAD operands."""
        return self.activation_offsets.narrow(
            0, BACKWARD_FC13_DGRAD_OFFSET_BASE, ACTIVATION_OFFSET_COUNT
        )

    @property
    def fc13_wgrad_offsets(self) -> torch.Tensor:
        """Return offsets for staged W13 WGRAD operands."""
        return self.activation_offsets.narrow(
            0, BACKWARD_FC13_WGRAD_OFFSET_BASE, GEMM_OPERAND_OFFSET_COUNT
        )

    @property
    def fc13_mega_offsets(self) -> torch.Tensor:
        """Return offsets for fused W13 DGRAD and WGRAD operands."""
        return self.activation_offsets.narrow(
            0, BACKWARD_FC13_DGRAD_OFFSET_BASE, MEGA_ACTIVATION_OFFSET_COUNT
        )


@dataclasses.dataclass(frozen=True)
class BackwardPlan:
    """Plan for MoE backward pass execution.

    Contains memory offsets for activation tensors during backward pass.
    If recompute was used in forward, x_gathered and h1 are allocated fresh.
    Otherwise, they reference the saved activations from forward pass.
    """

    # Memory offset for x_gathered tensor (saved or recomputed)
    x_gathered_offset: torch.Tensor
    # Memory offset for h1 tensor (saved or recomputed)
    h1_offset: torch.Tensor
    # Memory offset for h2 tensor
    h2_offset: torch.Tensor
    # Memory offset for grad_h2 tensor
    grad_h2_offset: torch.Tensor
    # Memory offset for grad_h3_gathered tensor
    grad_h3_gathered_offset: torch.Tensor
    # Memory offset for grad_h1 tensor
    grad_h1_offset: torch.Tensor
    # Present only for block-scaled activation-buffer plans.
    blockscaled: BlockscaledBackwardPlan | None = None


def _get_blockscaled_forward_plan(
    num_recv_tokens: torch.Tensor,
    num_recv_tokens_per_rank: torch.Tensor,
    num_recv_tokens_per_rank_snapshot: torch.Tensor | None,
    buffer_status: ActivationBuffer,
    model_config: ModelConfig,
    activation_slot_id_1: torch.Tensor,
    num_moe_layers_in_slot: int,
    inference_mode: bool,
    scratch_only: bool,
) -> ForwardPlan:
    """Build the block-scaled forward activation plan.

    Launches the block-scaled planner kernel and splits its packed int64 output
    into the dense ForwardPlan offsets plus the row/column bundle offsets in
    BlockscaledForwardPlan.

    Args:
        num_recv_tokens: Device scalar containing capacity-padded received rows.
        num_recv_tokens_per_rank: Received rows from each expert-parallel rank.
        num_recv_tokens_per_rank_snapshot: Optional independent saved row counts.
        buffer_status: Mutable activation-buffer state.
        model_config: Static planner model geometry.
        activation_slot_id_1: Device scalar selecting the physical slot.
        num_moe_layers_in_slot: Static MoE-layer depth of the selected slot.
        inference_mode: Whether to allocate scratch-only inference state.
        scratch_only: Whether a training-context forward has no backward
            consumer and therefore must not update activation-slot state.

    Returns:
        Forward offsets and recompute decision.
    """
    storage = model_config.blockscaled_storage
    assert storage is not None
    device = num_recv_tokens_per_rank.device
    need_recompute = torch.empty(1, dtype=torch.bool, device=device)
    recompute_condition = torch.empty(1, dtype=torch.int32, device=device)
    offsets = torch.empty(
        _BLOCKSCALED_FORWARD_PLAN_OFFSET_COUNT,
        dtype=torch.int64,
        device=device,
    )
    activation_offsets = torch.empty(
        FORWARD_ACTIVATION_OFFSET_COUNT,
        dtype=torch.int64,
        device=device,
    )
    N = buffer_status.num_activation_slots
    _triton_get_blockscaled_forward_plan[(1,)](
        num_recv_tokens_ptr=num_recv_tokens,
        num_recv_tokens_per_rank_ptr=num_recv_tokens_per_rank,
        num_recv_tokens_per_rank_snapshot_ptr=num_recv_tokens_per_rank_snapshot,
        saved_activation_bytes_per_rank_ptr=buffer_status.saved_activation_bytes_per_rank,
        buffer_offsets_ptr=buffer_status.buffer_offsets,
        peak_min_free_space_ptr=buffer_status.peak_min_free_space,
        need_recompute_ptr=need_recompute,
        recompute_condition_ptr=recompute_condition,
        fwd_offsets_ptr=offsets,
        fwd_activation_offsets_ptr=activation_offsets,
        EP=num_recv_tokens_per_rank.shape[0],
        num_tokens=model_config.num_tokens,
        topk=model_config.topk,
        element_size=model_config.element_size,
        operand_element_size=storage.operand_element_size,
        operand_values_per_storage_element=(storage.operand_values_per_storage_element),
        scale_element_size=storage.scale_element_size,
        row_scale_storage_multiple=storage.row_scale_storage_multiple,
        row_global_scale_element_size=storage.row_global_scale_element_size,
        sf_vec_size=storage.sf_vec_size,
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        capacity_rows=model_config.max_recv_tokens,
        microbatch_id_ptr=activation_slot_id_1,
        activation_slot_bytes=buffer_status.activation_slot_bytes,
        num_activation_slots=N,
        moe_layer_id_ptr=buffer_status.moe_layer_id,
        num_moe_layers=num_moe_layers_in_slot,
        mega=model_config.mega,
        inference_mode=inference_mode,
        scratch_only=scratch_only,
        native_backward=model_config.native_backward,
        interleaved_fc13=model_config.interleaved_fc13,
    )
    # Unpack in the same order the kernel stores into fwd_offsets_ptr.
    (
        x_offset,
        x_gathered_offset,
        h1_offset,
        h2_offset,
        h3_offset,
        x_row_offset,
        x_col_offset,
        h2_row_offset,
        h2_col_offset,
    ) = offsets.chunk(_BLOCKSCALED_FORWARD_PLAN_OFFSET_COUNT)
    return ForwardPlan(
        need_recompute=need_recompute,
        x_offset=x_offset,
        x_gathered_offset=x_gathered_offset,
        h1_offset=h1_offset,
        h2_offset=h2_offset,
        h3_offset=h3_offset,
        activation_slot_id_1=activation_slot_id_1,
        packed_offsets=offsets,
        blockscaled=BlockscaledForwardPlan(
            x_row_offset=x_row_offset,
            x_col_offset=x_col_offset,
            h2_row_offset=h2_row_offset,
            h2_col_offset=h2_col_offset,
            activation_offsets=activation_offsets,
            recompute_condition=recompute_condition,
        ),
    )


def _get_blockscaled_backward_plan(
    num_recv_tokens: torch.Tensor,
    num_recv_tokens_per_rank: torch.Tensor,
    forward_plan: ForwardPlan,
    buffer_status: ActivationBuffer,
    model_config: ModelConfig,
) -> BackwardPlan:
    """Build the block-scaled backward activation plan.

    Reuses the forward plan's saved x/h2 column bundles (no-recompute) or rebuilds
    them (recompute), then splits the kernel's packed int64 output into the dense
    BackwardPlan offsets plus the row/column bundle offsets in
    BlockscaledBackwardPlan.

    Args:
        num_recv_tokens: Device scalar containing capacity-padded received rows.
        num_recv_tokens_per_rank: Received rows from each expert-parallel rank.
        forward_plan: Matching forward plan and saved offsets.
        buffer_status: Mutable activation-buffer state.
        model_config: Static planner model geometry.

    Returns:
        Backward offsets for saved or recomputed operands and gradients.
    """
    storage = model_config.blockscaled_storage
    fwd_bs = forward_plan.blockscaled
    assert storage is not None and fwd_bs is not None
    offsets = torch.empty(
        _BLOCKSCALED_BACKWARD_PLAN_OFFSET_COUNT,
        dtype=torch.int64,
        device=buffer_status.buffer_offsets.device,
    )
    activation_offsets = torch.empty(
        BACKWARD_ACTIVATION_OFFSET_COUNT,
        dtype=torch.int64,
        device=buffer_status.buffer_offsets.device,
    )
    N = buffer_status.num_activation_slots
    _triton_get_blockscaled_backward_plan[(1,)](
        num_recv_tokens_ptr=num_recv_tokens,
        num_recv_tokens_per_rank_ptr=num_recv_tokens_per_rank,
        buffer_offsets_ptr=buffer_status.buffer_offsets,
        peak_min_free_space_ptr=buffer_status.peak_min_free_space,
        need_recompute_ptr=forward_plan.need_recompute,
        fwd_x_offset_ptr=forward_plan.x_offset,
        fwd_h1_offset_ptr=forward_plan.h1_offset,
        fwd_x_col_offset_ptr=fwd_bs.x_col_offset,
        fwd_h2_col_offset_ptr=fwd_bs.h2_col_offset,
        fwd_x_row_offset_ptr=fwd_bs.x_row_offset,
        fwd_h2_row_offset_ptr=fwd_bs.h2_row_offset,
        saved_activation_bytes_per_rank_ptr=buffer_status.saved_activation_bytes_per_rank,
        bwd_offsets_ptr=offsets,
        bwd_activation_offsets_ptr=activation_offsets,
        EP=num_recv_tokens_per_rank.shape[0],
        num_tokens=model_config.num_tokens,
        topk=model_config.topk,
        element_size=model_config.element_size,
        operand_element_size=storage.operand_element_size,
        operand_values_per_storage_element=(storage.operand_values_per_storage_element),
        scale_element_size=storage.scale_element_size,
        row_scale_storage_multiple=storage.row_scale_storage_multiple,
        row_global_scale_element_size=storage.row_global_scale_element_size,
        sf_vec_size=storage.sf_vec_size,
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        capacity_rows=model_config.max_recv_tokens,
        microbatch_id_ptr=forward_plan.activation_slot_id_1,
        activation_slot_bytes=buffer_status.activation_slot_bytes,
        num_activation_slots=N,
        moe_layer_id_ptr=buffer_status.moe_layer_id,
        mega=model_config.mega,
        native_backward=model_config.native_backward,
    )
    # chunks are ordered as the kernel stores them into bwd_offsets_ptr. Of the
    # dense BackwardPlan slots (0..5) only h1 (1) and grad_h2 (3) carry real
    # offsets; x_gathered (0), h2 (2), grad_h3_gathered (4) and grad_h1 (5) are
    # zero because block-scaling replaces them with the bundles in slots 6..13.
    chunks = offsets.chunk(_BLOCKSCALED_BACKWARD_PLAN_OFFSET_COUNT)
    return BackwardPlan(
        x_gathered_offset=chunks[0],
        h1_offset=chunks[1],
        h2_offset=chunks[2],
        grad_h2_offset=chunks[3],
        grad_h3_gathered_offset=chunks[4],
        grad_h1_offset=chunks[5],
        blockscaled=BlockscaledBackwardPlan(
            x_row_offset=chunks[6],
            x_col_offset=chunks[7],
            h2_row_offset=chunks[8],
            h2_col_offset=chunks[9],
            activation_offsets=activation_offsets,
        ),
    )


def get_forward_plan(
    num_recv_tokens: torch.Tensor,  # [1] int32 tensor
    num_recv_tokens_per_rank: torch.Tensor,  # [EP]
    buffer_status: ActivationBuffer,
    model_config: ModelConfig,
    activation_slot_id_1: torch.Tensor | None = None,
    num_moe_layers_in_slot: int | None = None,
    inference_mode: bool = False,
    num_recv_tokens_per_rank_snapshot: torch.Tensor | None = None,
    scratch_only: bool = False,
) -> ForwardPlan:
    """Create a forward execution plan for distributed MoE layer.

    This function uses a Triton kernel to calculate memory requirements and
    determine whether recompute is needed, avoiding device-to-host transfers.
    It also computes memory offsets for the activation buffer:
    - Saved activations allocated front to back:
      - Recompute mode: x
      - No-recompute mode: x_gathered, h1, h3_saved
    - Temporary activations (h2, or all if recomputing) allocated back to front

    Args:
        num_recv_tokens: Tensor [1] containing rows received by the current rank.
            With block-scaled routing this includes per-expert padding.
        num_recv_tokens_per_rank: Tensor [EP] containing the number of tokens
            routed to each rank in this forward pass.
        buffer_status: ActivationBuffer containing the activation buffer, offsets,
            and number of tokens already saved from each rank. Modified in place.
        model_config: ModelConfig containing element_size, hidden_dim,
            intermediate_dim, and other parameters needed for memory calculations.
        activation_slot_id_1: Device tensor ``[1]`` containing the
            physical activation slot. If ``None``, use the buffer's selected
            slot tensor.
        num_moe_layers_in_slot: Static number of MoE layers sharing the
            selected activation slot. If ``None``, use the buffer's current
            selection.
        inference_mode: If True, allocate only scratch temporaries and do not
            save activations or update layer/accounting state.
        num_recv_tokens_per_rank_snapshot: Optional independent output for the
            per-rank counts consumed by the corresponding backward pass.
        scratch_only: If True, use shared scratch without changing saved-state
            pointers because no backward can consume this forward.

    Returns:
        ForwardPlan with the recompute decision and memory offsets. BF16 packs
        its five offsets and selected-slot snapshot into one forward-produced
        state tensor for registered autograd.
        buffer_status is updated in place with new buffer_offsets and
        saved_activation_bytes_per_rank.
    """
    if buffer_status.inference_mode != inference_mode:
        raise ValueError(
            "ActivationBuffer.inference_mode must match get_forward_plan(inference_mode): "
            f"got activation_buffer.inference_mode={buffer_status.inference_mode}, "
            f"inference_mode={inference_mode}."
        )
    scratch_only = scratch_only or inference_mode
    if scratch_only and not inference_mode:
        validate_scratch_only_capacity(
            buffer_status.scratch_region_size,
            model_config,
        )

    EP = num_recv_tokens_per_rank.shape[0]
    device = num_recv_tokens_per_rank.device

    selection = (
        activation_slot_id_1
        if activation_slot_id_1 is not None
        else buffer_status.activation_slot_id_1
    )
    _num_moe_layers_in_selected_slot = (
        buffer_status._num_moe_layers_in_selected_slot
        if num_moe_layers_in_slot is None
        else num_moe_layers_in_slot
    )
    if not 1 <= _num_moe_layers_in_selected_slot <= buffer_status.num_moe_layers:
        raise ValueError(
            "num_moe_layers_in_slot must be in "
            f"[1, {buffer_status.num_moe_layers}], got {_num_moe_layers_in_selected_slot}"
        )

    if model_config.blockscaled_storage is not None:
        return _get_blockscaled_forward_plan(
            num_recv_tokens,
            num_recv_tokens_per_rank,
            num_recv_tokens_per_rank_snapshot,
            buffer_status,
            model_config,
            selection,
            _num_moe_layers_in_selected_slot,
            inference_mode,
            scratch_only,
        )

    # Allocate output tensors for plan offsets
    need_recompute = torch.empty(1, dtype=torch.bool, device=device)
    # Pack offsets and the selected slot into one forward-produced tensor so
    # cached SAC output owns every value needed by backward.
    fwd_offsets = torch.empty(
        _BF16_FORWARD_STATE_COUNT,
        dtype=torch.int64,
        device=device,
    )

    N = buffer_status.num_activation_slots

    # Launch kernel with single thread block
    # NOTE: buffer_offsets and saved_activation_bytes_per_rank are updated IN-PLACE
    # to ensure the shared activation buffer state is correctly maintained across layers
    _triton_get_forward_plan[(1,)](
        num_recv_tokens_ptr=num_recv_tokens,
        num_recv_tokens_per_rank_ptr=num_recv_tokens_per_rank,
        num_recv_tokens_per_rank_snapshot_ptr=num_recv_tokens_per_rank_snapshot,
        saved_activation_bytes_per_rank_ptr=buffer_status.saved_activation_bytes_per_rank,
        buffer_offsets_ptr=buffer_status.buffer_offsets,
        peak_min_free_space_ptr=buffer_status.peak_min_free_space,
        need_recompute_ptr=need_recompute,
        fwd_offsets_ptr=fwd_offsets,
        EP=EP,
        num_tokens=model_config.num_tokens,
        topk=model_config.topk,
        element_size=model_config.element_size,
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        microbatch_id_ptr=selection,
        activation_slot_bytes=buffer_status.activation_slot_bytes,
        buffer_size=buffer_status.buffer.numel(),
        num_activation_slots=N,
        moe_layer_id_ptr=buffer_status.moe_layer_id,
        num_moe_layers=_num_moe_layers_in_selected_slot,
        scratch_only=scratch_only,
        **pdl_launch_kwargs(),
    )

    # Parse individual offsets from the tensor:
    (
        x_offset,
        x_gathered_offset,
        h1_offset,
        h2_offset,
        h3_offset,
    ) = fwd_offsets[:_BF16_FORWARD_PLAN_OFFSET_COUNT].chunk(
        _BF16_FORWARD_PLAN_OFFSET_COUNT
    )

    forward_plan = ForwardPlan(
        need_recompute=need_recompute,
        x_offset=x_offset,
        x_gathered_offset=x_gathered_offset,
        h1_offset=h1_offset,
        h2_offset=h2_offset,
        h3_offset=h3_offset,
        activation_slot_id_1=selection,
        packed_offsets=fwd_offsets,
    )

    # buffer_status.buffer_offsets and buffer_status.saved_activation_bytes_per_rank
    # are already updated in-place by the kernel

    return forward_plan


def get_backward_plan(
    num_recv_tokens: torch.Tensor,
    num_recv_tokens_per_rank: torch.Tensor,  # [EP]
    forward_plan: ForwardPlan,
    buffer_status: ActivationBuffer,
    model_config: ModelConfig,
) -> BackwardPlan:
    """Create a backward execution plan for distributed MoE layer.

    This function uses a Triton kernel to calculate memory offsets for backward
    pass, avoiding device-to-host transfers. The offsets depend on whether
    recompute was used in the forward pass:
    - With recompute: x_gathered and h1 are allocated fresh from scratch region
    - Without recompute: x_gathered and h1 use saved offsets from forward pass

    All temporary tensors (h2, grad_h2, grad_h3_gathered, grad_h1) are allocated
    from the scratch region (back to front).

    After backward pass, saved activations are no longer needed, so buffer_offsets
    is updated to free the saved memory.

    saved_activation_bytes_per_rank update:
    - If recompute: decrease by x_mem_size (scalar, same for all ranks)
    - If no recompute: decrease by per-rank allocation (x_gathered + h1 + h3)

    Args:
        num_recv_tokens: Tensor [1] containing rows received in this backward pass.
            With block-scaled routing this includes per-expert padding.
        num_recv_tokens_per_rank: Tensor [EP] containing the number of tokens
            routed to each rank in this backward pass.
        forward_plan: ForwardPlan from the corresponding forward pass,
            containing need_recompute decision and saved offsets.
        buffer_status: ActivationBuffer containing the activation buffer, offsets,
            and number of tokens saved from each rank. Modified in place.
        model_config: ModelConfig containing element_size, hidden_dim,
            intermediate_dim, and other parameters needed for memory calculations.
    Returns:
        BackwardPlan with memory offsets for all backward pass tensors.
        buffer_status is updated in place with new buffer_offsets and
        saved_activation_bytes_per_rank.
    """
    device = buffer_status.buffer_offsets.device
    EP = num_recv_tokens_per_rank.shape[0]

    if model_config.blockscaled_storage is not None:
        return _get_blockscaled_backward_plan(
            num_recv_tokens,
            num_recv_tokens_per_rank,
            forward_plan,
            buffer_status,
            model_config,
        )

    # Allocate output tensors
    # bwd_offsets: [x_gathered, h1, h2, grad_h2, grad_h3_gathered, grad_h1]
    # In recompute mode, x_gathered and h1 are fresh scratch allocations
    # In no-recompute mode, they match the saved forward plan offsets
    bwd_offsets = torch.empty(6, dtype=torch.int64, device=device)

    N = buffer_status.num_activation_slots

    # Launch kernel with single thread block
    # NOTE: buffer_offsets and saved_activation_bytes_per_rank are updated IN-PLACE
    # to ensure the shared activation buffer state is correctly maintained across layers
    _triton_get_backward_plan[(1,)](
        num_recv_tokens_ptr=num_recv_tokens,
        buffer_offsets_ptr=buffer_status.buffer_offsets,
        peak_min_free_space_ptr=buffer_status.peak_min_free_space,
        need_recompute_ptr=forward_plan.need_recompute,
        fwd_x_offset_ptr=forward_plan.x_offset,
        fwd_x_gathered_offset_ptr=forward_plan.x_gathered_offset,
        fwd_h1_offset_ptr=forward_plan.h1_offset,
        saved_activation_bytes_per_rank_ptr=buffer_status.saved_activation_bytes_per_rank,
        num_recv_tokens_per_rank_ptr=num_recv_tokens_per_rank,
        bwd_offsets_ptr=bwd_offsets,
        EP=EP,
        num_tokens=model_config.num_tokens,
        topk=model_config.topk,
        element_size=model_config.element_size,
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        microbatch_id_ptr=forward_plan.activation_slot_id_1,
        activation_slot_bytes=buffer_status.activation_slot_bytes,
        buffer_size=buffer_status.buffer.numel(),
        num_activation_slots=N,
        moe_layer_id_ptr=buffer_status.moe_layer_id,
    )

    # Parse offsets from the tensor
    # [x_gathered, h1, h2, grad_h2, grad_h3_gathered, grad_h1]
    (
        x_gathered_offset,
        h1_offset,
        h2_offset,
        grad_h2_offset,
        grad_h3_gathered_offset,
        grad_h1_offset,
    ) = bwd_offsets.chunk(6)

    backward_plan = BackwardPlan(
        x_gathered_offset=x_gathered_offset,
        h1_offset=h1_offset,
        h2_offset=h2_offset,
        grad_h2_offset=grad_h2_offset,
        grad_h3_gathered_offset=grad_h3_gathered_offset,
        grad_h1_offset=grad_h1_offset,
    )

    # buffer_status.buffer_offsets and buffer_status.saved_activation_bytes_per_rank
    # are already updated in-place by the kernel

    return backward_plan
