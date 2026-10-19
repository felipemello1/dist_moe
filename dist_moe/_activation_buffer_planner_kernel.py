# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton plan-computation kernels for the DistMoE activation buffer planner.

Device side of ``activation_buffer_planner.py``: the ``@triton.jit`` kernels
that compute forward/backward activation plans on-device (so planning composes
with CUDA graphs), plus the byte-alignment constants they fold in as
``tl.constexpr``. The host-side ``ActivationBuffer`` / ``ModelConfig`` / plan
dataclasses and the launch wrappers live in ``activation_buffer_planner.py``.
"""

import triton
import triton.language as tl

from .kernels.activation_buffer import (
    ACTIVATION_A_Q_OFFSET,
    ACTIVATION_A_SCALE_OFFSET,
    ACTIVATION_B_Q_OFFSET,
    ACTIVATION_B_SCALE_OFFSET,
    ACTIVATION_C_OFFSET,
    ACTIVATION_COL_Q_OFFSET,
    ACTIVATION_COL_SCALE_OFFSET,
    ACTIVATION_SOURCE_X_OFFSET,
    ACTIVATION_SOURCE_Y_OFFSET,
    BACKWARD_FC2_DGRAD_OFFSET_BASE,
    BACKWARD_FC2_WGRAD_OFFSET_BASE,
    BACKWARD_FC13_DGRAD_OFFSET_BASE,
    BACKWARD_FC13_WGRAD_OFFSET_BASE,
    BACKWARD_RECOMPUTE_COMBINE_OFFSET_BASE,
    BACKWARD_RECOMPUTE_DISPATCH_OFFSET_BASE,
    FORWARD_COMBINE_OFFSET_BASE,
    FORWARD_DISPATCH_OFFSET_BASE,
    MISSING_ACTIVATION_OFFSET,
)
from .kernels.pdl import pdl_wait_then_release

# Memory alignment requirement in bytes for buffer sub-allocations
MEMORY_ALIGNMENT = 128
BLOCKSCALED_DISPATCH_ROW_ALIGNMENT_BYTES = 16

# Triton JIT functions cannot access module-level globals; create constexpr copy
MEMORY_ALIGNMENT_TL_CONSTEXPR = tl.constexpr(MEMORY_ALIGNMENT)
BLOCKSCALED_DISPATCH_ROW_ALIGNMENT_BYTES_TL_CONSTEXPR = tl.constexpr(
    BLOCKSCALED_DISPATCH_ROW_ALIGNMENT_BYTES
)
MISSING_ACTIVATION_OFFSET_TL_CONSTEXPR = tl.constexpr(MISSING_ACTIVATION_OFFSET)
ACTIVATION_A_Q_OFFSET_TL_CONSTEXPR = tl.constexpr(ACTIVATION_A_Q_OFFSET)
ACTIVATION_B_Q_OFFSET_TL_CONSTEXPR = tl.constexpr(ACTIVATION_B_Q_OFFSET)
ACTIVATION_C_OFFSET_TL_CONSTEXPR = tl.constexpr(ACTIVATION_C_OFFSET)
ACTIVATION_A_SCALE_OFFSET_TL_CONSTEXPR = tl.constexpr(ACTIVATION_A_SCALE_OFFSET)
ACTIVATION_B_SCALE_OFFSET_TL_CONSTEXPR = tl.constexpr(ACTIVATION_B_SCALE_OFFSET)
ACTIVATION_SOURCE_X_OFFSET_TL_CONSTEXPR = tl.constexpr(ACTIVATION_SOURCE_X_OFFSET)
ACTIVATION_SOURCE_Y_OFFSET_TL_CONSTEXPR = tl.constexpr(ACTIVATION_SOURCE_Y_OFFSET)
ACTIVATION_COL_Q_OFFSET_TL_CONSTEXPR = tl.constexpr(ACTIVATION_COL_Q_OFFSET)
ACTIVATION_COL_SCALE_OFFSET_TL_CONSTEXPR = tl.constexpr(ACTIVATION_COL_SCALE_OFFSET)
FORWARD_DISPATCH_OFFSET_BASE_TL_CONSTEXPR = tl.constexpr(FORWARD_DISPATCH_OFFSET_BASE)
FORWARD_COMBINE_OFFSET_BASE_TL_CONSTEXPR = tl.constexpr(FORWARD_COMBINE_OFFSET_BASE)
BACKWARD_RECOMPUTE_DISPATCH_OFFSET_BASE_TL_CONSTEXPR = tl.constexpr(
    BACKWARD_RECOMPUTE_DISPATCH_OFFSET_BASE
)
BACKWARD_RECOMPUTE_COMBINE_OFFSET_BASE_TL_CONSTEXPR = tl.constexpr(
    BACKWARD_RECOMPUTE_COMBINE_OFFSET_BASE
)
BACKWARD_FC2_DGRAD_OFFSET_BASE_TL_CONSTEXPR = tl.constexpr(
    BACKWARD_FC2_DGRAD_OFFSET_BASE
)
BACKWARD_FC2_WGRAD_OFFSET_BASE_TL_CONSTEXPR = tl.constexpr(
    BACKWARD_FC2_WGRAD_OFFSET_BASE
)
BACKWARD_FC13_DGRAD_OFFSET_BASE_TL_CONSTEXPR = tl.constexpr(
    BACKWARD_FC13_DGRAD_OFFSET_BASE
)
BACKWARD_FC13_WGRAD_OFFSET_BASE_TL_CONSTEXPR = tl.constexpr(
    BACKWARD_FC13_WGRAD_OFFSET_BASE
)


@triton.jit
def _tl_align(size):
    """Triton helper to align size to MEMORY_ALIGNMENT byte boundary.

    Ensures all calculations use int64 to prevent overflow with large sizes
    (e.g., num_recv_tokens=768k can produce sizes exceeding int32 max).
    """
    # Cast size to int64 to handle large values (>2GB)
    size_i64 = tl.cast(size, tl.int64)
    # Cast alignment constant to int64 to ensure all arithmetic stays in int64
    alignment = tl.cast(MEMORY_ALIGNMENT_TL_CONSTEXPR, tl.int64)
    return (size_i64 + alignment - 1) // alignment * alignment


@triton.jit
def _tl_row_quant_sizes(
    rows,
    dim: tl.constexpr,
    operand_element_size: tl.constexpr,
    operand_values_per_storage_element: tl.constexpr,
    scale_element_size: tl.constexpr,
    row_scale_storage_multiple: tl.constexpr,
    row_global_scale_element_size: tl.constexpr,
    sf_vec_size: tl.constexpr,
):
    rows = rows.to(tl.int64)
    packed_dim: tl.constexpr = (
        dim + operand_values_per_storage_element - 1
    ) // operand_values_per_storage_element
    q_size = _tl_align(rows * packed_dim * operand_element_size)
    scale_cols: tl.constexpr = (dim + sf_vec_size - 1) // sf_vec_size
    scale_size = _tl_align(
        rows * row_scale_storage_multiple * scale_cols * scale_element_size
        + rows * row_global_scale_element_size
    )
    return q_size, scale_size


@triton.jit
def _tl_packed_dispatch_size(
    rows: tl.constexpr,
    dim: tl.constexpr,
    operand_element_size: tl.constexpr,
    operand_values_per_storage_element: tl.constexpr,
    scale_element_size: tl.constexpr,
    row_global_scale_element_size: tl.constexpr,
    sf_vec_size: tl.constexpr,
):
    packed_dim: tl.constexpr = (
        dim + operand_values_per_storage_element - 1
    ) // operand_values_per_storage_element
    scale_cols: tl.constexpr = (dim + sf_vec_size - 1) // sf_vec_size
    row_bytes: tl.constexpr = (
        packed_dim * operand_element_size
        + scale_cols * scale_element_size
        + row_global_scale_element_size
    )
    row_alignment = tl.cast(
        BLOCKSCALED_DISPATCH_ROW_ALIGNMENT_BYTES_TL_CONSTEXPR, tl.int64
    )
    row_stride = (row_bytes + row_alignment - 1) // row_alignment * row_alignment
    return _tl_align(rows * row_stride)


@triton.jit
def _tl_col_quant_sizes(
    rows,
    dim: tl.constexpr,
    operand_element_size: tl.constexpr,
    operand_values_per_storage_element: tl.constexpr,
    scale_element_size: tl.constexpr,
    sf_vec_size: tl.constexpr,
):
    rows = rows.to(tl.int64)
    packed_rows = (
        rows + operand_values_per_storage_element - 1
    ) // operand_values_per_storage_element
    q_size = _tl_align(dim * packed_rows * operand_element_size)
    scale_rows = (rows + sf_vec_size - 1) // sf_vec_size
    scale_size = _tl_align(dim * scale_rows * scale_element_size)
    return q_size, scale_size


@triton.jit
def _tl_blockscaled_no_recompute_alloc_size(
    x_col_q_size,
    x_col_scale_size,
    h1_size,
    h2_col_q_size,
    h2_col_scale_size,
    h3_size,
):
    return (
        x_col_q_size
        + x_col_scale_size
        + h1_size
        + h2_col_q_size
        + h2_col_scale_size
        + h3_size
    )


@triton.jit
def _tl_blockscaled_no_recompute_alloc_per_rank(
    rank_counts,
    h3_mem_size,
    hidden_dim,
    intermediate_dim,
    element_size,
    operand_element_size,
    operand_values_per_storage_element,
    scale_element_size,
    row_scale_storage_multiple,
    row_global_scale_element_size,
    sf_vec_size,
    native_backward: tl.constexpr,
):
    """Per-rank bytes to persist this layer's bundles without recompute.

    Shared by the forward and backward planners so the two allocation models
    cannot drift.
    """
    if native_backward:
        (
            x_per_rank_q_size,
            x_per_rank_scale_size,
        ) = _tl_row_quant_sizes(
            rank_counts,
            hidden_dim,
            operand_element_size,
            operand_values_per_storage_element,
            scale_element_size,
            row_scale_storage_multiple,
            row_global_scale_element_size,
            sf_vec_size,
        )
        (
            h2_per_rank_q_size,
            h2_per_rank_scale_size,
        ) = _tl_row_quant_sizes(
            rank_counts,
            intermediate_dim,
            operand_element_size,
            operand_values_per_storage_element,
            scale_element_size,
            row_scale_storage_multiple,
            row_global_scale_element_size,
            sf_vec_size,
        )
    else:
        (
            x_per_rank_q_size,
            x_per_rank_scale_size,
        ) = _tl_col_quant_sizes(
            rank_counts,
            hidden_dim,
            operand_element_size,
            operand_values_per_storage_element,
            scale_element_size,
            sf_vec_size,
        )
        (
            h2_per_rank_q_size,
            h2_per_rank_scale_size,
        ) = _tl_col_quant_sizes(
            rank_counts,
            intermediate_dim,
            operand_element_size,
            operand_values_per_storage_element,
            scale_element_size,
            sf_vec_size,
        )
    h1_per_rank_size = _tl_align(rank_counts * 2 * intermediate_dim * element_size)
    return _tl_blockscaled_no_recompute_alloc_size(
        x_per_rank_q_size,
        x_per_rank_scale_size,
        h1_per_rank_size,
        h2_per_rank_q_size,
        h2_per_rank_scale_size,
        h3_mem_size,
    )


@triton.jit
def _tl_store_activation_offsets(
    offsets_ptr,
    base: tl.constexpr,
    a_q,
    b_q,
    c,
    a_scale,
    b_scale,
    source_x,
    source_y,
    col_q,
    col_scale,
):
    tl.store(offsets_ptr + base + ACTIVATION_A_Q_OFFSET_TL_CONSTEXPR, a_q)
    tl.store(offsets_ptr + base + ACTIVATION_B_Q_OFFSET_TL_CONSTEXPR, b_q)
    tl.store(offsets_ptr + base + ACTIVATION_C_OFFSET_TL_CONSTEXPR, c)
    tl.store(offsets_ptr + base + ACTIVATION_A_SCALE_OFFSET_TL_CONSTEXPR, a_scale)
    tl.store(offsets_ptr + base + ACTIVATION_B_SCALE_OFFSET_TL_CONSTEXPR, b_scale)
    tl.store(offsets_ptr + base + ACTIVATION_SOURCE_X_OFFSET_TL_CONSTEXPR, source_x)
    tl.store(offsets_ptr + base + ACTIVATION_SOURCE_Y_OFFSET_TL_CONSTEXPR, source_y)
    tl.store(offsets_ptr + base + ACTIVATION_COL_Q_OFFSET_TL_CONSTEXPR, col_q)
    tl.store(
        offsets_ptr + base + ACTIVATION_COL_SCALE_OFFSET_TL_CONSTEXPR,
        col_scale,
    )


@triton.jit
def _triton_get_blockscaled_forward_plan(
    num_recv_tokens_ptr,
    num_recv_tokens_per_rank_ptr,
    num_recv_tokens_per_rank_snapshot_ptr,
    saved_activation_bytes_per_rank_ptr,
    buffer_offsets_ptr,
    peak_min_free_space_ptr,
    need_recompute_ptr,
    recompute_condition_ptr,
    fwd_offsets_ptr,
    fwd_activation_offsets_ptr,
    EP: tl.constexpr,
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    element_size: tl.constexpr,
    operand_element_size: tl.constexpr,
    operand_values_per_storage_element: tl.constexpr,
    scale_element_size: tl.constexpr,
    row_scale_storage_multiple: tl.constexpr,
    row_global_scale_element_size: tl.constexpr,
    sf_vec_size: tl.constexpr,
    hidden_dim: tl.constexpr,
    intermediate_dim: tl.constexpr,
    capacity_rows: tl.constexpr,
    microbatch_id_ptr,
    activation_slot_bytes: tl.constexpr,
    num_activation_slots: tl.constexpr,
    moe_layer_id_ptr,
    num_moe_layers: tl.constexpr,
    mega: tl.constexpr,
    inference_mode: tl.constexpr,
    scratch_only: tl.constexpr,
    native_backward: tl.constexpr,
    interleaved_fc13: tl.constexpr,
):
    """Compute the block-scaled forward plan: offsets + recompute decision.

    Bundles are sized from routing's actual row counts after per-expert padding.
    ``capacity_rows`` remains the static upper bound used to size the buffer, and
    a device assertion guards the local padded count against that capacity.

    Memory grows from two ends: the selected activation slot fills
    front-to-back with persistent saved tensors, while the shared scratch region
    fills back-to-front from scratch_end with transient tensors. Staged kernels
    consume x_row (hidden) and h2_row (intermediate) in serialized launches and
    alias one max-sized row slot. A Mega launch co-feeds them, so they occupy
    adjacent sub-slots. "row scratch" below means either layout:

      no-recompute  slot [x_col | h1 | h2_col | h3];  scratch [row scratch]
      recompute     slot [x dense];  scratch [row scratch | x_col | h1 | h2_col]
      forward-only slot none;       scratch [row scratch | h1]
      inference, interleaved_fc13
                    slot none;       scratch [row scratch | nvfp4 h2 staging]

    x_col / h2_col are the column-quantized FC13 / FC2 weight-gradient bundles:
    saved in the slot for backward when not recomputing, transient otherwise.

    The h1 slot is the dense FC13 direct output: the non-interleaved fused
    kernels store it through their C tensormap and the SwiGLU quant producer
    reads it back. The interleaved (swizzled) FC13 epilogue consumes its own
    output in place, so its plan drops h1 entirely: FC13's C stays a
    host-side dummy descriptor (the dispatch group's C offset is missing),
    the MX formats quantize h2 in-epilogue into the h2 row slot, and NVFP4
    stages dense h2 in a dedicated slot below the row slot that the combine
    group's source offsets both point at. Interleaved is inference-only and
    requires the fused Mega launch (co-live x/h2 row sub-slots).

    With native_backward the backward consumes the *row* bundles instead, so the
    roles invert: no column bundle is produced at all and the row bundles become
    the persistent tensors. That removes the scratch row slot entirely when not
    recomputing, because the kernels write their A operands straight into the
    slot:

      no-recompute  slot [x_row | h1 | h2_row | h3];  scratch []
      recompute     slot [x packed]; scratch [row scratch | h1]
    """
    rank_offsets = tl.arange(0, EP)
    rank_counts = tl.load(num_recv_tokens_per_rank_ptr + rank_offsets).to(tl.int64)
    if num_recv_tokens_per_rank_snapshot_ptr is not None:
        tl.store(num_recv_tokens_per_rank_snapshot_ptr + rank_offsets, rank_counts)

    padded_rows = tl.load(num_recv_tokens_ptr).to(tl.int64)
    tl.device_assert(padded_rows <= capacity_rows, "padded receive capacity exceeded")
    rows = padded_rows
    h1_size = _tl_align(rows * 2 * intermediate_dim * element_size)  # h1 dense
    x_row_q_size, x_row_scale_size = _tl_row_quant_sizes(
        rows,
        hidden_dim,
        operand_element_size,
        operand_values_per_storage_element,
        scale_element_size,
        row_scale_storage_multiple,
        row_global_scale_element_size,
        sf_vec_size,
    )
    h2_row_q_size, h2_row_scale_size = _tl_row_quant_sizes(
        rows,
        intermediate_dim,
        operand_element_size,
        operand_values_per_storage_element,
        scale_element_size,
        row_scale_storage_multiple,
        row_global_scale_element_size,
        sf_vec_size,
    )
    x_col_q_size, x_col_scale_size = _tl_col_quant_sizes(
        rows,
        hidden_dim,
        operand_element_size,
        operand_values_per_storage_element,
        scale_element_size,
        sf_vec_size,
    )
    h2_col_q_size, h2_col_scale_size = _tl_col_quant_sizes(
        rows,
        intermediate_dim,
        operand_element_size,
        operand_values_per_storage_element,
        scale_element_size,
        sf_vec_size,
    )
    x_row_size = x_row_q_size + x_row_scale_size
    h2_row_size = h2_row_q_size + h2_row_scale_size
    scratch_end = tl.load(buffer_offsets_ptr + num_activation_slots)
    x_row_offset = scratch_end - x_row_size  # x_gathered row-quant (FC13 input)
    if mega:
        # The fused launch consumes both operands, so they must not alias.
        h2_row_offset = x_row_offset - h2_row_size
        row_slot_size = x_row_size + h2_row_size
    else:
        # Serialized staged launches share one max-sized slot.
        h2_row_offset = scratch_end - h2_row_size
        row_slot_size = tl.maximum(x_row_size, h2_row_size)
    row_slot_offset = scratch_end - row_slot_size

    zero = tl.zeros([], dtype=tl.int64)
    x_gathered_offset = zero
    h2_offset = zero
    x_col_offset = zero
    h2_col_offset = zero
    h2_staging_offset = zero
    scratch_base = (
        zero
        if inference_mode
        else tl.cast(num_activation_slots * activation_slot_bytes, tl.int64)
    )
    if inference_mode or scratch_only:
        # A forward-only call consumes row bundles and dense h1 from scratch;
        # dense x_gathered, h2, and h3 have no backward consumer.
        need_recompute = tl.full((), True, tl.int1)
        x_offset = zero
        h3_offset = zero
        if interleaved_fc13:
            # The interleaved FC13 epilogue consumes its own output in place,
            # so dense h1 is never materialized. NVFP4 stages dense h2 below
            # the row slot for its full-row global-scale pass; the MX formats
            # quantize in-epilogue and stage nothing.
            h1_offset = zero
            h2_staging_offset = row_slot_offset
            if row_global_scale_element_size > 0:
                h2_staging_offset = row_slot_offset - _tl_align(
                    rows * intermediate_dim * element_size
                )
            scratch_bottom = h2_staging_offset
        else:
            h1_offset = row_slot_offset - h1_size
            scratch_bottom = h1_offset
    else:
        microbatch_id = (tl.load(microbatch_id_ptr) % num_activation_slots).to(tl.int64)
        free_start = tl.load(buffer_offsets_ptr + microbatch_id)
        slot_end = tl.cast((microbatch_id + 1) * activation_slot_bytes, tl.int64)
        saved_offset = microbatch_id * EP
        saved_bytes = tl.load(
            saved_activation_bytes_per_rank_ptr + saved_offset + rank_offsets
        )
        tl.debug_barrier()

        if native_backward:
            x_mem_size = _tl_packed_dispatch_size(
                num_tokens,
                hidden_dim,
                operand_element_size,
                operand_values_per_storage_element,
                scale_element_size,
                row_global_scale_element_size,
                sf_vec_size,
            )
        else:
            x_mem_size = _tl_align(num_tokens * hidden_dim * element_size)
        h3_mem_size = _tl_align(num_tokens * topk * hidden_dim * element_size)
        no_recompute_alloc_per_rank = _tl_blockscaled_no_recompute_alloc_per_rank(
            rank_counts,
            h3_mem_size,
            hidden_dim,
            intermediate_dim,
            element_size,
            operand_element_size,
            operand_values_per_storage_element,
            scale_element_size,
            row_scale_storage_multiple,
            row_global_scale_element_size,
            sf_vec_size,
            native_backward,
        )
        # Look-ahead recompute decision: recompute if saving full activations
        # this layer (no_recompute_alloc_per_rank, worst rank) plus the saved input for
        # every remaining layer would overflow the selected slot.
        moe_layer_id = tl.load(moe_layer_id_ptr + microbatch_id)
        num_layers_after = (num_moe_layers - 1) - moe_layer_id
        worst_case_usage = tl.max(saved_bytes + no_recompute_alloc_per_rank, axis=0)
        need_recompute = (
            worst_case_usage + num_layers_after * x_mem_size > activation_slot_bytes
        )

        if need_recompute and native_backward:
            # Persist the inference-style packed dispatch input. The initial
            # forward only needs the serialized row slot and dense h1.
            x_offset = free_start
            new_free_start = x_offset + x_mem_size
            h3_offset = zero
            h1_offset = row_slot_offset - h1_size
            scratch_bottom = h1_offset
        elif need_recompute:
            # Save only dense x in the slot. The col bundles + h1 are transient
            # in scratch, descending below the row slot in the order:
            #   x_col (x_gathered, FC13 WGRAD), h1 (dense), h2_col (h2, FC2 WGRAD).
            x_offset = free_start
            new_free_start = x_offset + x_mem_size
            x_col_offset = row_slot_offset - x_col_q_size - x_col_scale_size
            h1_offset = x_col_offset - h1_size
            h2_col_offset = h1_offset - h2_col_q_size - h2_col_scale_size
            h3_offset = zero
            scratch_bottom = h2_col_offset
        elif native_backward:
            # The row bundles the forward GEMMs consume are exactly what the
            # backward needs, so allocate them in the slot. No extra forward
            # scratch remains live after the serialized GEMMs finish.
            x_row_offset = free_start
            h1_offset = x_row_offset + x_row_size
            h2_row_offset = h1_offset + h1_size
            h3_offset = h2_row_offset + h2_row_size
            new_free_start = h3_offset + h3_mem_size
            x_offset = zero
            scratch_bottom = scratch_end
        else:
            # Save the operands backward needs, ascending from free_start in the
            # order: x_col (x_gathered, FC13 WGRAD), h1 (dense), h2_col (h2, FC2
            # WGRAD), h3 (dense, combine). Scratch holds only the row slot.
            x_col_offset = free_start
            h1_offset = x_col_offset + x_col_q_size + x_col_scale_size
            h2_col_offset = h1_offset + h1_size
            h3_offset = h2_col_offset + h2_col_q_size + h2_col_scale_size
            new_free_start = h3_offset + h3_mem_size
            x_offset = zero
            scratch_bottom = row_slot_offset

        tl.device_assert(new_free_start <= slot_end, "OOM: activation slot overflow")
        slot_free = slot_end - new_free_start
        current_slot_peak = tl.load(peak_min_free_space_ptr + microbatch_id)
        tl.store(
            peak_min_free_space_ptr + microbatch_id,
            tl.minimum(current_slot_peak, slot_free),
        )
        tl.store(buffer_offsets_ptr + microbatch_id, new_free_start)
        tl.store(moe_layer_id_ptr + microbatch_id, moe_layer_id + 1)
        output_bytes = tl.where(
            need_recompute,
            saved_bytes + x_mem_size,
            saved_bytes + no_recompute_alloc_per_rank,
        )
        tl.store(
            saved_activation_bytes_per_rank_ptr + saved_offset + rank_offsets,
            output_bytes,
        )

    tl.device_assert(
        scratch_bottom >= scratch_base,
        "OOM: scratch overflow into activation slots",
    )
    scratch_free = scratch_bottom - scratch_base
    scratch_peak = tl.load(peak_min_free_space_ptr + num_activation_slots)
    tl.store(
        peak_min_free_space_ptr + num_activation_slots,
        tl.minimum(scratch_peak, scratch_free),
    )

    if native_backward:
        x_col_offset = x_row_offset
        h2_col_offset = h2_row_offset
    tl.store(need_recompute_ptr, need_recompute)
    tl.store(fwd_offsets_ptr, x_offset)
    tl.store(fwd_offsets_ptr + 1, x_gathered_offset)
    tl.store(fwd_offsets_ptr + 2, h1_offset)
    tl.store(fwd_offsets_ptr + 3, h2_offset)
    tl.store(fwd_offsets_ptr + 4, h3_offset)
    tl.store(fwd_offsets_ptr + 5, x_row_offset)
    tl.store(fwd_offsets_ptr + 6, x_col_offset)
    tl.store(fwd_offsets_ptr + 7, h2_row_offset)
    tl.store(fwd_offsets_ptr + 8, h2_col_offset)
    tl.store(recompute_condition_ptr, need_recompute.to(tl.int32))

    missing = tl.full((), MISSING_ACTIVATION_OFFSET_TL_CONSTEXPR, tl.int64)
    if native_backward:
        # The inference-like forward does not emit WGRAD column bundles. Keep
        # these descriptor-only slots addressable by aliasing the live row
        # bundle; return_wgrad_quant=False guarantees they are never accessed.
        dispatch_col_q = x_row_offset
        dispatch_col_scale = x_row_offset + x_row_q_size
        combine_col_q = h2_row_offset
        combine_col_scale = h2_row_offset + h2_row_q_size
    else:
        dispatch_col_q = x_col_offset
        dispatch_col_scale = x_col_offset + x_col_q_size
        combine_col_q = h2_col_offset
        combine_col_scale = h2_col_offset + h2_col_q_size
    dispatch_c = h1_offset
    combine_source_x = h1_offset
    combine_source_y = h1_offset + intermediate_dim * element_size
    if interleaved_fc13:
        # Dense h1 does not exist: FC13's C stays a host-side dummy
        # descriptor, and the SwiGLU sources are the NVFP4 dense-h2 staging
        # slot (absent on MX, which quantizes in-epilogue).
        dispatch_c = missing
        if row_global_scale_element_size > 0:
            combine_source_x = h2_staging_offset
            combine_source_y = h2_staging_offset
        else:
            combine_source_x = missing
            combine_source_y = missing
    _tl_store_activation_offsets(
        offsets_ptr=fwd_activation_offsets_ptr,
        base=FORWARD_DISPATCH_OFFSET_BASE_TL_CONSTEXPR,
        a_q=x_row_offset,
        b_q=missing,
        c=dispatch_c,
        a_scale=x_row_offset + x_row_q_size,
        b_scale=missing,
        source_x=missing,
        source_y=missing,
        col_q=dispatch_col_q,
        col_scale=dispatch_col_scale,
    )
    _tl_store_activation_offsets(
        offsets_ptr=fwd_activation_offsets_ptr,
        base=FORWARD_COMBINE_OFFSET_BASE_TL_CONSTEXPR,
        a_q=h2_row_offset,
        b_q=missing,
        c=missing,
        a_scale=h2_row_offset + h2_row_q_size,
        b_scale=missing,
        source_x=combine_source_x,
        source_y=combine_source_y,
        col_q=combine_col_q,
        col_scale=combine_col_scale,
    )


@triton.jit
def _triton_get_blockscaled_backward_plan(
    num_recv_tokens_ptr,
    num_recv_tokens_per_rank_ptr,
    buffer_offsets_ptr,
    peak_min_free_space_ptr,
    need_recompute_ptr,
    fwd_x_offset_ptr,
    fwd_h1_offset_ptr,
    fwd_x_col_offset_ptr,
    fwd_h2_col_offset_ptr,
    fwd_x_row_offset_ptr,
    fwd_h2_row_offset_ptr,
    saved_activation_bytes_per_rank_ptr,
    bwd_offsets_ptr,
    bwd_activation_offsets_ptr,
    EP: tl.constexpr,
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    element_size: tl.constexpr,
    operand_element_size: tl.constexpr,
    operand_values_per_storage_element: tl.constexpr,
    scale_element_size: tl.constexpr,
    row_scale_storage_multiple: tl.constexpr,
    row_global_scale_element_size: tl.constexpr,
    sf_vec_size: tl.constexpr,
    hidden_dim: tl.constexpr,
    intermediate_dim: tl.constexpr,
    capacity_rows: tl.constexpr,
    microbatch_id_ptr,
    activation_slot_bytes: tl.constexpr,
    num_activation_slots: tl.constexpr,
    moe_layer_id_ptr,
    mega: tl.constexpr,
    native_backward: tl.constexpr,
):
    """Compute the block-scaled backward plan (offsets only).

    Mirrors the forward planner's two-ended layout. Staged block-scaled
    recompute serializes x_row and h2_row and aliases them in one max-sized row
    slot. Mega recompute co-feeds that pair, while native backward dequantizes
    both before either dense GEMM consumes them, so those modes give the pair
    adjacent sub-slots. Without recompute only the later serialized grad_h3 and
    dxy DGRAD launches need row scratch; grad_h3_row reuses x_row's offset.

    The gradient bundles grad_h3_col, grad_h2, and dxy_col are always allocated
    in scratch (descending below the row slot). The forward operands x_col, h1,
    and h2_col are loaded from their forward-saved slot offsets (no-recompute)
    or rebuilt in scratch above the gradients (recompute):

      no-recompute  scratch [row slot | grad_h3_col | grad_h2 | dxy_col]
      recompute     scratch [row slot | x_col | h1 | h2_col | grad_h3_col | grad_h2 | dxy_col]

    With native_backward the backward is dense: the row bundles are dequantized
    up front into x_gathered and h2 and every GEMM after that is native. Scratch
    is therefore the dense working set, and the row bundles are read from the
    forward's slot offsets (no-recompute) or rebuilt in scratch (recompute):

      no-recompute  scratch [x_gathered | h2 | grad_h3_gathered | grad_h2 | grad_h1]
      recompute     scratch [row slot | h1 | x_gathered | h2 | grad_h3_gathered | grad_h2 | grad_h1]

    Finally buffer_offsets, saved bytes, and moe_layer_id are rewound to undo the
    matching forward layer's allocation. num_recv_tokens_ptr carries routing's
    per-rank row count after per-expert padding, matching the forward planner.
    """
    padded_rows = tl.load(num_recv_tokens_ptr).to(tl.int64)
    tl.device_assert(padded_rows <= capacity_rows, "padded receive capacity exceeded")
    rows = padded_rows
    need_recompute = tl.load(need_recompute_ptr)
    microbatch_id = (tl.load(microbatch_id_ptr) % num_activation_slots).to(tl.int64)
    scratch_end = tl.load(buffer_offsets_ptr + num_activation_slots)
    free_start = tl.load(buffer_offsets_ptr + microbatch_id)
    slot_end = tl.cast((microbatch_id + 1) * activation_slot_bytes, tl.int64)

    x_row_q_size, x_row_scale_size = _tl_row_quant_sizes(
        rows,
        hidden_dim,
        operand_element_size,
        operand_values_per_storage_element,
        scale_element_size,
        row_scale_storage_multiple,
        row_global_scale_element_size,
        sf_vec_size,
    )
    h2_row_q_size, h2_row_scale_size = _tl_row_quant_sizes(
        rows,
        intermediate_dim,
        operand_element_size,
        operand_values_per_storage_element,
        scale_element_size,
        row_scale_storage_multiple,
        row_global_scale_element_size,
        sf_vec_size,
    )
    dxy_row_q_size, dxy_row_scale_size = _tl_row_quant_sizes(
        rows,
        2 * intermediate_dim,
        operand_element_size,
        operand_values_per_storage_element,
        scale_element_size,
        row_scale_storage_multiple,
        row_global_scale_element_size,
        sf_vec_size,
    )
    x_row_size = x_row_q_size + x_row_scale_size
    h2_row_size = h2_row_q_size + h2_row_scale_size
    dxy_row_size = dxy_row_q_size + dxy_row_scale_size
    x_row_offset = scratch_end - x_row_size  # x_gathered row (FC13 DGRAD, recompute)
    if mega or native_backward:
        # Mega co-feeds the row bundles; native backward dequantizes both first.
        recompute_h2_row_offset = x_row_offset - h2_row_size
        recompute_row_slot_size = tl.maximum(x_row_size + h2_row_size, dxy_row_size)
    else:
        recompute_h2_row_offset = scratch_end - h2_row_size
        recompute_row_slot_size = tl.maximum(
            tl.maximum(x_row_size, h2_row_size), dxy_row_size
        )
    dxy_row_offset = scratch_end - dxy_row_size
    # Tensormap preparation still materializes h2 pointers for a skipped recompute.
    h2_row_offset = tl.where(need_recompute, recompute_h2_row_offset, dxy_row_offset)
    no_recompute_row_slot_size = tl.maximum(x_row_size, dxy_row_size)
    row_slot_size = tl.where(
        need_recompute, recompute_row_slot_size, no_recompute_row_slot_size
    )
    row_slot_offset = scratch_end - row_slot_size
    # Later DGRAD launches are serialized and reuse the row scratch.
    grad_h3_row_offset = x_row_offset  # grad_h3 row (FC2 DGRAD); reuses dead x_row slot

    x_col_q_size, x_col_scale_size = _tl_col_quant_sizes(
        rows,
        hidden_dim,
        operand_element_size,
        operand_values_per_storage_element,
        scale_element_size,
        sf_vec_size,
    )
    h2_col_q_size, h2_col_scale_size = _tl_col_quant_sizes(
        rows,
        intermediate_dim,
        operand_element_size,
        operand_values_per_storage_element,
        scale_element_size,
        sf_vec_size,
    )
    dxy_col_q_size, dxy_col_scale_size = _tl_col_quant_sizes(
        rows,
        2 * intermediate_dim,
        operand_element_size,
        operand_values_per_storage_element,
        scale_element_size,
        sf_vec_size,
    )
    h1_size = _tl_align(rows * 2 * intermediate_dim * element_size)
    grad_h2_size = _tl_align(rows * intermediate_dim * element_size)

    # Column-quantized WGRAD operands stack in scratch below the row slot, with
    # `cursor` as the descending frontier. In allocation order the bundles are:
    #   x_col      x_gathered, FC13 WGRAD
    #   h1         dense, SwiGLU backward input
    #   h2_col     h2, FC2 WGRAD
    #   grad_h3_col  grad_h3, FC2 WGRAD (x's shape, so reuses x_col_*_size)
    #   grad_h2      dense
    #   dxy_col      dxy=grad_h1, FC13 WGRAD
    # Recompute rebuilds x_col/h1/h2_col here; no-recompute instead reuses the
    # offsets the forward pass saved in the activation slot.
    x_gathered_size = _tl_align(rows * hidden_dim * element_size)
    zero = tl.zeros([], dtype=tl.int64)
    x_gathered_offset = zero
    h2_offset = zero
    grad_h3_gathered_offset = zero
    grad_h1_offset = zero

    cursor = row_slot_offset
    if native_backward:
        # Dense working set. The row bundles come from the forward's slot
        # offsets unless the forward is being rerun, in which case they and dense
        # h1 are rebuilt in the scratch row slot above this state.
        if need_recompute:
            h1_offset = cursor - h1_size
            cursor = h1_offset
            # x_row_offset already points at the scratch row slot.
            h2_row_offset = recompute_h2_row_offset
        else:
            h1_offset = tl.load(fwd_h1_offset_ptr)
            x_row_offset = tl.load(fwd_x_row_offset_ptr)
            h2_row_offset = tl.load(fwd_h2_row_offset_ptr)
            cursor = scratch_end
        x_gathered_offset = cursor - x_gathered_size
        h2_offset = x_gathered_offset - grad_h2_size
        grad_h3_gathered_offset = h2_offset - x_gathered_size
        grad_h2_offset = grad_h3_gathered_offset - grad_h2_size
        grad_h1_offset = grad_h2_offset - h1_size
        # No block-scaled backward kernel runs in this mode. Alias the unused
        # column slots to their row bundles so descriptor construction still
        # receives legal addresses without reserving discarded storage.
        x_col_offset = x_row_offset
        h2_col_offset = h2_row_offset
        grad_h3_col_offset = grad_h3_row_offset
        dxy_col_offset = dxy_row_offset
        scratch_bottom = grad_h1_offset
    else:
        if need_recompute:
            x_col_offset = cursor - x_col_q_size - x_col_scale_size
            cursor = x_col_offset
            h1_offset = cursor - h1_size
            cursor = h1_offset
            h2_col_offset = cursor - h2_col_q_size - h2_col_scale_size
            cursor = h2_col_offset
        else:
            x_col_offset = tl.load(fwd_x_col_offset_ptr)
            h1_offset = tl.load(fwd_h1_offset_ptr)
            h2_col_offset = tl.load(fwd_h2_col_offset_ptr)

        grad_h3_col_offset = cursor - x_col_q_size - x_col_scale_size
        cursor = grad_h3_col_offset
        grad_h2_offset = cursor - grad_h2_size
        cursor = grad_h2_offset
        dxy_col_offset = cursor - dxy_col_q_size - dxy_col_scale_size
        scratch_bottom = dxy_col_offset

    scratch_base = tl.cast(num_activation_slots * activation_slot_bytes, tl.int64)
    tl.device_assert(
        scratch_bottom >= scratch_base,
        "OOM: scratch overflow into activation slots",
    )
    scratch_free = scratch_bottom - scratch_base
    scratch_peak = tl.load(peak_min_free_space_ptr + num_activation_slots)
    tl.store(
        peak_min_free_space_ptr + num_activation_slots,
        tl.minimum(scratch_peak, scratch_free),
    )
    slot_peak = tl.load(peak_min_free_space_ptr + microbatch_id)
    tl.store(
        peak_min_free_space_ptr + microbatch_id,
        tl.minimum(slot_peak, slot_end - free_start),
    )

    tl.store(bwd_offsets_ptr, x_gathered_offset)
    tl.store(bwd_offsets_ptr + 1, h1_offset)
    tl.store(bwd_offsets_ptr + 2, h2_offset)
    tl.store(bwd_offsets_ptr + 3, grad_h2_offset)
    tl.store(bwd_offsets_ptr + 4, grad_h3_gathered_offset)
    tl.store(bwd_offsets_ptr + 5, grad_h1_offset)
    tl.store(bwd_offsets_ptr + 6, x_row_offset)
    tl.store(bwd_offsets_ptr + 7, x_col_offset)
    tl.store(bwd_offsets_ptr + 8, h2_row_offset)
    tl.store(bwd_offsets_ptr + 9, h2_col_offset)
    tl.store(bwd_offsets_ptr + 10, grad_h3_row_offset)
    tl.store(bwd_offsets_ptr + 11, grad_h3_col_offset)
    tl.store(bwd_offsets_ptr + 12, dxy_row_offset)
    tl.store(bwd_offsets_ptr + 13, dxy_col_offset)
    x_scale_cols: tl.constexpr = (hidden_dim + sf_vec_size - 1) // sf_vec_size
    h2_scale_cols: tl.constexpr = (intermediate_dim + sf_vec_size - 1) // sf_vec_size
    x_row_scale_storage_size = (
        rows * row_scale_storage_multiple * x_scale_cols * scale_element_size
    )
    h2_row_scale_storage_size = (
        rows * row_scale_storage_multiple * h2_scale_cols * scale_element_size
    )
    tl.store(
        bwd_offsets_ptr + 14,
        x_row_offset + x_row_q_size + x_row_scale_storage_size,
    )
    tl.store(
        bwd_offsets_ptr + 15,
        h2_row_offset + h2_row_q_size + h2_row_scale_storage_size,
    )

    missing = tl.full((), MISSING_ACTIVATION_OFFSET_TL_CONSTEXPR, tl.int64)
    if native_backward:
        x_col_scale_offset = x_row_offset + x_row_q_size
        h2_col_scale_offset = h2_row_offset + h2_row_q_size
        grad_h3_col_scale_offset = grad_h3_row_offset + x_row_q_size
        dxy_col_scale_offset = dxy_row_offset + dxy_row_q_size
    else:
        x_col_scale_offset = x_col_offset + x_col_q_size
        h2_col_scale_offset = h2_col_offset + h2_col_q_size
        grad_h3_col_scale_offset = grad_h3_col_offset + x_col_q_size
        dxy_col_scale_offset = dxy_col_offset + dxy_col_q_size
    _tl_store_activation_offsets(
        offsets_ptr=bwd_activation_offsets_ptr,
        base=BACKWARD_RECOMPUTE_DISPATCH_OFFSET_BASE_TL_CONSTEXPR,
        a_q=x_row_offset,
        b_q=missing,
        c=h1_offset,
        a_scale=x_row_offset + x_row_q_size,
        b_scale=missing,
        source_x=missing,
        source_y=missing,
        col_q=x_col_offset,
        col_scale=x_col_scale_offset,
    )
    _tl_store_activation_offsets(
        offsets_ptr=bwd_activation_offsets_ptr,
        base=BACKWARD_RECOMPUTE_COMBINE_OFFSET_BASE_TL_CONSTEXPR,
        a_q=h2_row_offset,
        b_q=missing,
        c=missing,
        a_scale=h2_row_offset + h2_row_q_size,
        b_scale=missing,
        source_x=h1_offset,
        source_y=h1_offset + intermediate_dim * element_size,
        col_q=h2_col_offset,
        col_scale=h2_col_scale_offset,
    )
    _tl_store_activation_offsets(
        offsets_ptr=bwd_activation_offsets_ptr,
        base=BACKWARD_FC2_DGRAD_OFFSET_BASE_TL_CONSTEXPR,
        a_q=grad_h3_row_offset,
        b_q=missing,
        c=grad_h2_offset,
        a_scale=grad_h3_row_offset + x_row_q_size,
        b_scale=missing,
        source_x=missing,
        source_y=missing,
        col_q=grad_h3_col_offset,
        col_scale=grad_h3_col_scale_offset,
    )
    _tl_store_activation_offsets(
        offsets_ptr=bwd_activation_offsets_ptr,
        base=BACKWARD_FC2_WGRAD_OFFSET_BASE_TL_CONSTEXPR,
        a_q=grad_h3_col_offset,
        b_q=h2_col_offset,
        c=missing,
        a_scale=grad_h3_col_scale_offset,
        b_scale=h2_col_scale_offset,
        source_x=missing,
        source_y=missing,
        col_q=missing,
        col_scale=missing,
    )
    _tl_store_activation_offsets(
        offsets_ptr=bwd_activation_offsets_ptr,
        base=BACKWARD_FC13_DGRAD_OFFSET_BASE_TL_CONSTEXPR,
        a_q=dxy_row_offset,
        b_q=missing,
        c=missing,
        a_scale=dxy_row_offset + dxy_row_q_size,
        b_scale=missing,
        source_x=grad_h2_offset,
        source_y=h1_offset,
        col_q=dxy_col_offset,
        col_scale=dxy_col_scale_offset,
    )
    _tl_store_activation_offsets(
        offsets_ptr=bwd_activation_offsets_ptr,
        base=BACKWARD_FC13_WGRAD_OFFSET_BASE_TL_CONSTEXPR,
        a_q=dxy_col_offset,
        b_q=x_col_offset,
        c=missing,
        a_scale=dxy_col_scale_offset,
        b_scale=x_col_scale_offset,
        source_x=missing,
        source_y=missing,
        col_q=missing,
        col_scale=missing,
    )

    # Rewind this activation slot's high-water mark to where the matching forward
    # layer began: recompute saved dense x at fwd_x_offset; no-recompute saved
    # the saved bundle starting at x_col_offset.
    fwd_x_offset = tl.load(fwd_x_offset_ptr)
    if native_backward:
        saved_base = tl.load(fwd_x_row_offset_ptr)
    else:
        saved_base = x_col_offset
    new_free_start = tl.where(need_recompute, fwd_x_offset, saved_base)
    tl.store(buffer_offsets_ptr + microbatch_id, new_free_start)

    rank_offsets = tl.arange(0, EP)
    rank_counts = tl.load(num_recv_tokens_per_rank_ptr + rank_offsets).to(tl.int64)
    saved_offset = microbatch_id * EP
    saved_bytes = tl.load(
        saved_activation_bytes_per_rank_ptr + saved_offset + rank_offsets
    )
    tl.debug_barrier()
    if native_backward:
        x_mem_size = _tl_packed_dispatch_size(
            num_tokens,
            hidden_dim,
            operand_element_size,
            operand_values_per_storage_element,
            scale_element_size,
            row_global_scale_element_size,
            sf_vec_size,
        )
    else:
        x_mem_size = _tl_align(num_tokens * hidden_dim * element_size)
    h3_mem_size = _tl_align(num_tokens * topk * hidden_dim * element_size)
    no_recompute_alloc_per_rank = _tl_blockscaled_no_recompute_alloc_per_rank(
        rank_counts,
        h3_mem_size,
        hidden_dim,
        intermediate_dim,
        element_size,
        operand_element_size,
        operand_values_per_storage_element,
        scale_element_size,
        row_scale_storage_multiple,
        row_global_scale_element_size,
        sf_vec_size,
        native_backward,
    )
    # Reverse the forward saved-bytes accounting for this layer (forward added
    # the same per-mode delta).
    output_bytes = tl.where(
        need_recompute,
        saved_bytes - x_mem_size,
        saved_bytes - no_recompute_alloc_per_rank,
    )
    tl.store(
        saved_activation_bytes_per_rank_ptr + saved_offset + rank_offsets,
        output_bytes,
    )
    layer_id = tl.load(moe_layer_id_ptr + microbatch_id)
    tl.store(moe_layer_id_ptr + microbatch_id, layer_id - 1)


@triton.jit
def _triton_get_forward_plan(
    # Inputs
    num_recv_tokens_ptr,  # [1] int32
    num_recv_tokens_per_rank_ptr,  # [EP]
    num_recv_tokens_per_rank_snapshot_ptr,  # optional [EP] output
    saved_activation_bytes_per_rank_ptr,  # [N*EP] - IN-PLACE updated
    buffer_offsets_ptr,  # [N+1]: slot/scratch pointers - IN-PLACE updated
    peak_min_free_space_ptr,  # [N+1] int64 - IN-PLACE updated
    # Outputs
    need_recompute_ptr,  # [1]
    # [5]: [x_offset, x_gathered_offset, h1_offset, h2_offset, h3_offset]
    fwd_offsets_ptr,
    # Constexpr parameters
    EP: tl.constexpr,
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    element_size: tl.constexpr,
    hidden_dim: tl.constexpr,
    intermediate_dim: tl.constexpr,
    microbatch_id_ptr,
    activation_slot_bytes: tl.constexpr,
    buffer_size: tl.constexpr,
    num_activation_slots: tl.constexpr,
    moe_layer_id_ptr,  # [N] int64 - IN-PLACE updated
    num_moe_layers: tl.constexpr,
    scratch_only: tl.constexpr,
    PDL: tl.constexpr,
):
    """Triton kernel to compute forward plan without D2H transfers.

    The buffer is partitioned into N activation slots + scratch.
    - free_start is loaded from buffer_offsets[microbatch_id]
    - slot_end is (microbatch_id + 1) * activation_slot_bytes
    - scratch allocations come from buffer_offsets[N] backward
    - saved_activation_bytes indexed at microbatch_id * EP

    `scratch_only` plans a forward that will never be backpropagated: every
    tensor is carved from the shared scratch region and none of the persistent
    slot state is touched, so the call is idempotent and leaves no allocation
    behind. It is set for an inference-layout buffer (where N is 0 and
    buffer_offsets[0] is the scratch end) and, on a training-layout buffer, for
    a forward autograd built no node for. The two agree because the slot state
    is unwound only by the matching backward, which neither case runs.

    NOTE: buffer_offsets_ptr, saved_activation_bytes_per_rank_ptr, and
    peak_min_free_space_ptr are updated IN-PLACE, except under `scratch_only`,
    where only the scratch entry of peak_min_free_space_ptr is updated.
    """
    pdl_wait_then_release(PDL)
    if num_recv_tokens_per_rank_snapshot_ptr is not None:
        rank_offsets = tl.arange(0, EP)
        rank_counts = tl.load(num_recv_tokens_per_rank_ptr + rank_offsets)
        tl.store(num_recv_tokens_per_rank_snapshot_ptr + rank_offsets, rank_counts)

    # Load num_recv_tokens from pointer
    num_recv_tokens = tl.load(num_recv_tokens_ptr).to(tl.int64)

    # Calculate sizes with memory alignment
    # Cast constexpr products to int64 to prevent overflow with large num_recv_tokens
    hidden_dim_bytes = tl.cast(hidden_dim * element_size, tl.int64)
    intermediate_dim_bytes = tl.cast(intermediate_dim * element_size, tl.int64)
    double_intermediate_dim_bytes = tl.cast(
        2 * intermediate_dim * element_size, tl.int64
    )

    x_gathered_size = _tl_align(num_recv_tokens * hidden_dim_bytes)
    h1_size = _tl_align(num_recv_tokens * double_intermediate_dim_bytes)
    h2_size = _tl_align(num_recv_tokens * intermediate_dim_bytes)

    # Inference buffers may have N=0; buffer_offsets[N] is still the scratch end.
    scratch_end = tl.load(buffer_offsets_ptr + num_activation_slots)
    scratch_h2_offset = scratch_end - h2_size
    scratch_h1_offset = scratch_h2_offset - h1_size
    scratch_x_gathered_offset = scratch_h1_offset - x_gathered_size

    activation_slot_id = tl.load(microbatch_id_ptr).to(tl.int64)
    if scratch_only:
        # Nothing is saved, so the recompute flag is moot; report True so any
        # consumer that still reads it takes the no-saved-activations branch.
        need_recompute = tl.full((), True, tl.int1)
        x_offset = tl.zeros([], dtype=tl.int64)
        x_gathered_offset = scratch_x_gathered_offset
        h1_offset = scratch_h1_offset
        h2_offset = scratch_h2_offset
        h3_offset = tl.zeros([], dtype=tl.int64)
    else:
        # Load the device selector and map it to an activation slot.
        microbatch_id = activation_slot_id % num_activation_slots

        # Load the selected activation slot's current offset.
        free_start = tl.load(buffer_offsets_ptr + microbatch_id)
        # Slot end: boundary for the selected activation slot.
        slot_end = tl.cast((microbatch_id + 1) * activation_slot_bytes, tl.int64)

        # Load all ranks as vectors, offset by microbatch_id * EP.
        offsets = tl.arange(0, EP)
        saved_bytes_offset = microbatch_id * EP
        recv_tokens = tl.load(num_recv_tokens_per_rank_ptr + offsets)
        saved_bytes = tl.load(
            saved_activation_bytes_per_rank_ptr + saved_bytes_offset + offsets
        )

        # Barrier: ensure all reads complete before any writes (in-place update safety).
        tl.debug_barrier()

        # x: [num_tokens, hidden_dim], h3_saved: [num_tokens * topk, hidden_dim]
        x_mem_size = _tl_align(
            tl.cast(num_tokens * hidden_dim * element_size, tl.int64)
        )
        h3_mem_size = _tl_align(
            tl.cast(num_tokens * topk * hidden_dim * element_size, tl.int64)
        )

        # Per-rank allocation sizes for no-recompute mode (vector of EP elements).
        # Each rank saves x_gathered + h1 (proportional to recv_tokens) + h3 (fixed).
        recv_tokens_i64 = recv_tokens.to(tl.int64)
        x_gathered_per_rank = _tl_align(recv_tokens_i64 * hidden_dim_bytes)
        h1_per_rank = _tl_align(recv_tokens_i64 * double_intermediate_dim_bytes)
        no_recompute_alloc_per_rank = x_gathered_per_rank + h1_per_rank + h3_mem_size

        # Load current layer index and compute remaining layers after this one.
        moe_layer_id = tl.load(moe_layer_id_ptr + microbatch_id)
        num_layers_after = (num_moe_layers - 1) - moe_layer_id

        # Look-ahead recompute decision using per-rank saved activation bytes.
        # Check if the worst-case rank (max cumulative saved bytes after this layer)
        # plus minimum space needed for remaining layers exceeds slot capacity.
        worst_case_usage = tl.max(saved_bytes + no_recompute_alloc_per_rank, axis=0)
        need_recompute = (
            worst_case_usage + num_layers_after * x_mem_size > activation_slot_bytes
        )

        if need_recompute:
            # Recompute mode: save x (front to back), all others temporary (back to front)
            x_offset = free_start
            new_free_start = x_offset + x_mem_size

            # Temporary activations allocated back to front from scratch_end
            x_gathered_offset = scratch_x_gathered_offset
            h1_offset = scratch_h1_offset
            h2_offset = scratch_h2_offset

            # h3 not saved in recompute mode, set to 0 (unused)
            h3_offset = tl.zeros([], dtype=tl.int64)
        else:
            # No-recompute mode: save x_gathered, h1, h3_saved (front to back)
            # h2 is temporary (back to front)
            x_gathered_offset = free_start
            h1_offset = x_gathered_offset + x_gathered_size
            h3_offset = h1_offset + h1_size
            # Advance free_start by local allocation (no bubbles)
            new_free_start = h3_offset + h3_mem_size

            h2_offset = scratch_h2_offset

            # x not saved in no-recompute mode, set to 0 (unused)
            x_offset = tl.zeros([], dtype=tl.int64)

    tl.store(need_recompute_ptr, need_recompute)
    tl.store(fwd_offsets_ptr, x_offset)
    tl.store(fwd_offsets_ptr + 1, x_gathered_offset)
    tl.store(fwd_offsets_ptr + 2, h1_offset)
    tl.store(fwd_offsets_ptr + 3, h2_offset)
    tl.store(fwd_offsets_ptr + 4, h3_offset)
    tl.store(fwd_offsets_ptr + 5, activation_slot_id)

    scratch_bottom = tl.where(need_recompute, x_gathered_offset, h2_offset)
    # Also 0 for an inference-layout buffer, where activation_slot_bytes is 0.
    scratch_base = tl.cast(num_activation_slots * activation_slot_bytes, tl.int64)
    tl.device_assert(
        scratch_bottom >= scratch_base,
        "OOM: scratch overflow into activation slots",
    )
    scratch_free = scratch_bottom - scratch_base
    current_scratch_peak = tl.load(peak_min_free_space_ptr + num_activation_slots)
    new_scratch_peak = tl.minimum(current_scratch_peak, scratch_free)
    tl.store(peak_min_free_space_ptr + num_activation_slots, new_scratch_peak)

    if not scratch_only:
        # Assert saved activations do not overflow into the next slot.
        tl.device_assert(new_free_start <= slot_end, "OOM: activation slot overflow")

        # Per-slot peak free-space tracking.
        # Track activation slot and scratch region separately in peak_min_free_space[N+1].
        slot_free = slot_end - new_free_start

        # Update peak for this microbatch's activation slot
        current_slot_peak = tl.load(peak_min_free_space_ptr + microbatch_id)
        new_slot_peak = tl.minimum(current_slot_peak, slot_free)
        tl.store(peak_min_free_space_ptr + microbatch_id, new_slot_peak)

        # Update the selected slot pointer in place.
        tl.store(buffer_offsets_ptr + microbatch_id, new_free_start)

        # Update moe_layer_id IN-PLACE (increment for this forward pass)
        tl.store(moe_layer_id_ptr + microbatch_id, moe_layer_id + 1)

        # Update saved_activation_bytes_per_rank IN-PLACE based on recompute decision.
        # Recompute: only x is saved (same size for all ranks, scalar add).
        # No-recompute: per-rank allocation (vector add).
        output_bytes = tl.where(
            need_recompute,
            saved_bytes + x_mem_size,
            saved_bytes + no_recompute_alloc_per_rank,
        )
        tl.store(
            saved_activation_bytes_per_rank_ptr + saved_bytes_offset + offsets,
            output_bytes,
        )


@triton.jit
def _triton_get_backward_plan(
    # Inputs
    num_recv_tokens_ptr,  # [1]
    buffer_offsets_ptr,  # [N+1]: slot/scratch pointers - IN-PLACE updated
    peak_min_free_space_ptr,  # [N+1] int64 - IN-PLACE updated
    need_recompute_ptr,  # [1]
    fwd_x_offset_ptr,  # [1]
    fwd_x_gathered_offset_ptr,  # [1]
    fwd_h1_offset_ptr,  # [1]
    saved_activation_bytes_per_rank_ptr,  # [N*EP] - IN-PLACE updated
    num_recv_tokens_per_rank_ptr,  # [EP]
    # Outputs
    bwd_offsets_ptr,  # [6]: [x_gathered, h1, h2, grad_h2, grad_h3_gathered, grad_h1]
    # Constexpr parameters
    EP: tl.constexpr,
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    element_size: tl.constexpr,
    hidden_dim: tl.constexpr,
    intermediate_dim: tl.constexpr,
    microbatch_id_ptr,
    activation_slot_bytes: tl.constexpr,
    buffer_size: tl.constexpr,
    num_activation_slots: tl.constexpr,
    moe_layer_id_ptr,  # [N] int64 - IN-PLACE updated
):
    """Triton kernel to compute backward plan without D2H transfers.

    The buffer is always partitioned into N activation slots + scratch.
    - free_start is loaded from buffer_offsets[microbatch_id]
    - scratch_end is loaded from buffer_offsets[N]
    - saved_activation_bytes indexed at microbatch_id * EP

    NOTE: buffer_offsets_ptr, saved_activation_bytes_per_rank_ptr, and
    peak_min_free_space_ptr are updated IN-PLACE.
    """
    # Load num_recv_tokens from tensor
    num_recv_tokens = tl.load(num_recv_tokens_ptr).to(tl.int64)

    # Load the device selector and map it to an activation slot.
    microbatch_id = (tl.load(microbatch_id_ptr) % num_activation_slots).to(tl.int64)

    # Load the selected activation slot's current offset.
    free_start = tl.load(buffer_offsets_ptr + microbatch_id)
    # Slot end: boundary for the selected activation slot.
    slot_end = tl.cast((microbatch_id + 1) * activation_slot_bytes, tl.int64)
    # Scratch end is the last element of buffer_offsets
    scratch_end = tl.load(buffer_offsets_ptr + num_activation_slots)

    # Load need_recompute decision from forward plan
    need_recompute = tl.load(need_recompute_ptr)

    # Load forward plan offsets
    fwd_x_offset = tl.load(fwd_x_offset_ptr)
    fwd_x_gathered_offset = tl.load(fwd_x_gathered_offset_ptr)
    fwd_h1_offset = tl.load(fwd_h1_offset_ptr)

    # Load saved_activation_bytes_per_rank and num_recv_tokens_per_rank
    offsets = tl.arange(0, EP)
    saved_bytes_offset = microbatch_id * EP
    saved_bytes = tl.load(
        saved_activation_bytes_per_rank_ptr + saved_bytes_offset + offsets
    )
    recv_tokens = tl.load(num_recv_tokens_per_rank_ptr + offsets)

    # Barrier: ensure all reads complete before any writes (in-place update safety)
    tl.debug_barrier()

    # Calculate sizes for current backward pass with memory alignment
    hidden_dim_bytes = tl.cast(hidden_dim * element_size, tl.int64)
    intermediate_dim_bytes = tl.cast(intermediate_dim * element_size, tl.int64)
    double_intermediate_dim_bytes = tl.cast(
        2 * intermediate_dim * element_size, tl.int64
    )

    x_gathered_size = _tl_align(num_recv_tokens * hidden_dim_bytes)
    h1_size = _tl_align(num_recv_tokens * double_intermediate_dim_bytes)
    h2_size = _tl_align(num_recv_tokens * intermediate_dim_bytes)
    grad_h2_size = _tl_align(num_recv_tokens * intermediate_dim_bytes)
    grad_h3_gathered_size = _tl_align(num_recv_tokens * hidden_dim_bytes)
    grad_h1_size = _tl_align(num_recv_tokens * double_intermediate_dim_bytes)

    # LIVENESS-BASED MEMORY REUSE (applies to both recompute and no-recompute):
    #
    # Backward execution order and tensor liveness:
    #   Step 1-4: forward recompute → h1, h2, h3
    #   Step 5: dgrad_dispatch(grad_h3, w2) → grad_h2, grad_h3_gathered
    #   Step 6: wgrad(grad_h3_gathered, h2) → grad_w2
    #   Step 7: swiglu_bwd(grad_h2, h1) → grad_h1
    #   Step 8: dgrad_combine(grad_h1, w13) → grad_x_gathered
    #   Step 9: wgrad(grad_h1, x_gathered) → grad_w13
    #
    # Tensor lifetimes:
    #   x_gathered: 1-9, h1: 1-7, h2: 1-6, grad_h2: 5-7,
    #   grad_h3_gathered: 5-6, grad_h1: 7-9
    #
    # Safe overlaps (non-overlapping lifetimes):
    #   h2 ↔ grad_h1 (1-6 vs 7-9)
    #   grad_h3_gathered ↔ grad_h1 (5-6 vs 7-9)
    # Forbidden overlaps:
    #   grad_h2 ↔ grad_h3_gathered (both outputs of step 5)
    #   h2 ↔ grad_h3_gathered (both read at step 6)
    #
    # Layout (top of scratch, allocated back-to-front):
    #   grad_h1 at top, h2 overlaps grad_h1, grad_h2 below grad_h1.
    #   grad_h3_gathered: below h2 when hidden <= intermediate (fits between
    #   grad_h2 and h2), below grad_h2 otherwise.
    grad_h1_offset = scratch_end - grad_h1_size
    grad_h2_offset = grad_h1_offset - grad_h2_size

    # h2 overlaps with top of grad_h1's region (h2 dies before grad_h1 born)
    h2_offset = scratch_end - h2_size

    # grad_h3_gathered must NOT overlap with grad_h2 or h2.
    # When hidden <= intermediate, it fits between grad_h2 and h2.
    # When hidden > intermediate, it must go below grad_h2.
    if hidden_dim_bytes <= intermediate_dim_bytes:
        grad_h3_gathered_offset = h2_offset - grad_h3_gathered_size
    else:
        grad_h3_gathered_offset = grad_h2_offset - grad_h3_gathered_size

    if need_recompute:
        # Recompute mode: x_gathered and h1 also in scratch, below the
        # gradient tensors.
        scratch_grad_bottom = tl.minimum(grad_h2_offset, grad_h3_gathered_offset)
        h1_offset = scratch_grad_bottom - h1_size
        x_gathered_offset = h1_offset - x_gathered_size
        scratch_bottom = x_gathered_offset
    else:
        # No-recompute mode: x_gathered and h1 saved from forward
        x_gathered_offset = fwd_x_gathered_offset
        h1_offset = fwd_h1_offset
        scratch_bottom = tl.minimum(grad_h2_offset, grad_h3_gathered_offset)

    # Assert scratch allocations don't overflow into activation slots
    tl.device_assert(
        scratch_bottom
        >= tl.cast(num_activation_slots * activation_slot_bytes, tl.int64),
        "OOM: scratch overflow into activation slots",
    )

    # Per-slot peak free-space tracking: same logic as the forward kernel.
    slot_free = slot_end - free_start
    scratch_free = scratch_bottom - tl.cast(
        num_activation_slots * activation_slot_bytes, tl.int64
    )

    # Update peak for this microbatch's activation slot
    current_slot_peak = tl.load(peak_min_free_space_ptr + microbatch_id)
    new_slot_peak = tl.minimum(current_slot_peak, slot_free)
    tl.store(peak_min_free_space_ptr + microbatch_id, new_slot_peak)

    # Update peak for scratch region
    current_scratch_peak = tl.load(peak_min_free_space_ptr + num_activation_slots)
    new_scratch_peak = tl.minimum(current_scratch_peak, scratch_free)
    tl.store(peak_min_free_space_ptr + num_activation_slots, new_scratch_peak)

    # Store offsets to bwd_offsets tensor
    # [x_gathered, h1, h2, grad_h2, grad_h3_gathered, grad_h1]
    tl.store(bwd_offsets_ptr, x_gathered_offset)
    tl.store(bwd_offsets_ptr + 1, h1_offset)
    tl.store(bwd_offsets_ptr + 2, h2_offset)
    tl.store(bwd_offsets_ptr + 3, grad_h2_offset)
    tl.store(bwd_offsets_ptr + 4, grad_h3_gathered_offset)
    tl.store(bwd_offsets_ptr + 5, grad_h1_offset)

    # Update buffer_offsets IN-PLACE to free saved activations
    if need_recompute:
        new_free_start = fwd_x_offset
    else:
        new_free_start = fwd_x_gathered_offset

    tl.store(buffer_offsets_ptr + microbatch_id, new_free_start)

    # Compute per-rank allocation sizes (same formula as forward kernel)
    # to reverse the saved_activation_bytes_per_rank update.
    recv_tokens_i64 = recv_tokens.to(tl.int64)
    x_gathered_per_rank = _tl_align(recv_tokens_i64 * hidden_dim_bytes)
    h1_per_rank = _tl_align(recv_tokens_i64 * double_intermediate_dim_bytes)
    x_mem_size = _tl_align(tl.cast(num_tokens * hidden_dim * element_size, tl.int64))
    h3_mem_size = _tl_align(
        tl.cast(num_tokens * topk * hidden_dim * element_size, tl.int64)
    )
    no_recompute_alloc_per_rank = x_gathered_per_rank + h1_per_rank + h3_mem_size

    # Update saved_activation_bytes_per_rank IN-PLACE (reverse of forward).
    # Recompute: subtract x_mem_size (scalar). No-recompute: subtract per-rank alloc.
    output_bytes = tl.where(
        need_recompute,
        saved_bytes - x_mem_size,
        saved_bytes - no_recompute_alloc_per_rank,
    )
    tl.store(
        saved_activation_bytes_per_rank_ptr + saved_bytes_offset + offsets, output_bytes
    )

    # Update moe_layer_id IN-PLACE (decrement for this backward pass)
    moe_layer_id = tl.load(moe_layer_id_ptr + microbatch_id)
    tl.store(moe_layer_id_ptr + microbatch_id, moe_layer_id - 1)
