# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl

from ..._activation_buffer_planner_kernel import (
    MEMORY_ALIGNMENT_TL_CONSTEXPR,
)
from .._environment import num_sms_per_device
from ..activation_buffer import (
    validate_conditional_execution,
)
from .repr_utils import make_dtype_repr

SCALE_AND_SUM_TILE_D_MAX_TOKENS = 128


def _to_tl_dtype(dtype: torch.dtype) -> int:
    """Convert PyTorch dtype to Triton dtype."""
    match dtype:
        case torch.float32:
            return tl.float32
        case torch.bfloat16:
            return tl.bfloat16
        case torch.float16:
            return tl.float16
        case _:
            raise NotImplementedError(f"No support for {dtype=}!")


def _validate_activation_buffer_args(
    *,
    activation_buffer: torch.Tensor | None,
    activation_offset: torch.Tensor | None,
    condition: torch.Tensor | None,
    device: torch.device,
) -> bool:
    provided = (
        activation_buffer is not None,
        activation_offset is not None,
        condition is not None,
    )
    if any(provided) and not all(provided):
        raise ValueError(
            "activation_buffer, activation_offset, and condition must be provided together"
        )
    if not all(provided):
        return False

    assert activation_buffer is not None
    assert activation_offset is not None
    assert condition is not None
    if activation_buffer.dtype is not torch.uint8:
        raise TypeError("activation_buffer must have dtype uint8")
    if activation_buffer.device != device:
        raise ValueError("activation_buffer must be on the input device")
    if activation_offset.dtype is not torch.int64 or activation_offset.numel() != 1:
        raise TypeError("activation_offset must contain one int64 element")
    if activation_offset.device != device:
        raise ValueError("activation_offset must be on the input device")
    validate_conditional_execution(condition, device=device)
    return True


def scale_and_sum(
    x: torch.Tensor,  # [T, K, D]
    scale: torch.Tensor,  # [T, K]
    expert_ids: torch.Tensor | None = None,  # optional [T, K]
    return_copy: bool = True,
    output_dtype: torch.dtype | None = None,
    x_copy_buffer: torch.Tensor | None = None,
    x_copy_offset: torch.Tensor | None = None,
    x_copy_condition: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scale post combine tokens and sum them.

    Args:
        x: Input tensor [T, K, D] - can be a buffer tensor
        scale: Scaling factors [T, K]
        expert_ids: Optional expert IDs. Negative routes are ignored without
            reading their corresponding ``x`` storage.
        return_copy: If True, return a copy of x for backward. If False, return None.
        output_dtype: dtype for the output tensor y. If None, uses x.dtype.
        x_copy_buffer: Optional uint8 activation buffer receiving x when the
            device-side condition is zero.
        x_copy_offset: Byte offset of x in x_copy_buffer.
        x_copy_condition: Device-side bool or int32 recompute condition.

    Returns:
        Tuple of (y, x_copy) where:
        - y: Scaled and summed output [T, D]
        - x_copy: Copy of input x (safe to save for backward), or None if return_copy=False
    """
    # Check dtypes
    x_dtype = x.dtype
    scale_dtype = scale.dtype
    y_dtype = output_dtype if output_dtype is not None else x_dtype

    assert x_dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"{x_dtype=} not supported"
    )
    assert scale_dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert y_dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"{y_dtype=} not supported"
    )

    # Check shape
    T, K, D = x.shape
    T_, K_ = scale.shape
    assert T == T_
    assert K == K_

    # Check strides
    assert x.is_contiguous()
    assert scale.is_contiguous()
    if expert_ids is not None:
        if expert_ids.shape != (T, K) or expert_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("expert_ids must have shape [T, K] and int32/int64 dtype")
        if not expert_ids.is_contiguous() or expert_ids.device != x.device:
            raise ValueError("expert_ids must be contiguous on the input device")
    copy_x_to_buffer = _validate_activation_buffer_args(
        activation_buffer=x_copy_buffer,
        activation_offset=x_copy_offset,
        condition=x_copy_condition,
        device=x.device,
    )

    # Allocate output and optionally copy
    # Copy is created inside the Triton kernel for efficiency
    y = torch.empty((T, D), device=x.device, dtype=y_dtype)
    x_copy = (
        torch.empty((T, K, D), device=x.device, dtype=x_dtype) if return_copy else None
    )

    # Check if we can use triton
    NUM_ELEMS = 8192
    if K <= 8:
        BLOCK_SIZE_D = 1024
        BLOCK_SIZE_T = NUM_ELEMS // (BLOCK_SIZE_D * K)
        # Round down to nearest power of 2
        BLOCK_SIZE_T = 1 << (BLOCK_SIZE_T.bit_length() - 1) if BLOCK_SIZE_T > 0 else 1
    else:
        BLOCK_SIZE_T = 1
        BLOCK_SIZE_D = NUM_ELEMS // (BLOCK_SIZE_T * K)
        # Round down to nearest power of 2
        BLOCK_SIZE_D = 1 << (BLOCK_SIZE_D.bit_length() - 1) if BLOCK_SIZE_D > 0 else 1

    # Dynamic number of CTAs based on SM count for better parallelism
    num_sms = num_sms_per_device()
    # Inference omits the copy and benefits from more reduction CTAs.
    num_ctas = num_sms * (32 if return_copy else 64)
    # Mirrors the kernel's tile count: each CTA owns cdiv(num_tiles, NUM_CTA) tiles,
    # so CTAs past num_tiles have none.
    num_tiles = triton.cdiv(T, BLOCK_SIZE_T)
    if T <= SCALE_AND_SUM_TILE_D_MAX_TOKENS:
        num_tiles *= triton.cdiv(D, BLOCK_SIZE_D)
    grid = (max(1, min(num_ctas, num_tiles)), 1, 1)
    _triton_scale_and_sum[grid](
        x,
        scale,
        expert_ids,
        y,
        x_copy,
        x_copy_buffer,
        x_copy_offset,
        x_copy_condition,
        T,
        K,
        D,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        NUM_CTA=num_ctas,
        COPY_X=return_copy,
        COPY_X_TO_BUFFER=copy_x_to_buffer,
        HAS_EXPERT_IDS=expert_ids is not None,
        X_DTYPE=_to_tl_dtype(x_dtype),
        BLOCK_K=triton.next_power_of_2(K),
        TILE_D=T <= SCALE_AND_SUM_TILE_D_MAX_TOKENS,
    )
    return y, x_copy


@triton.jit(
    do_not_specialize=["T"],
    repr=make_dtype_repr("_triton_scale_and_sum", ["x_ptr", "y_ptr"]),
)
def _triton_scale_and_sum(  # noqa: C901
    x_ptr: torch.Tensor,  # [T, K, D]
    scale_ptr: torch.Tensor,  # [T, K] or None (for unscaled sum)
    expert_ids_ptr,
    y_ptr: torch.Tensor,  # [T, D]
    x_copy_ptr: torch.Tensor,  # [T, K, D] - copy of x for backward
    x_copy_buffer_ptr,
    x_copy_offset_ptr,
    x_copy_condition_ptr,
    T,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    NUM_CTA: tl.constexpr,
    COPY_X: tl.constexpr,
    COPY_X_TO_BUFFER: tl.constexpr,
    HAS_EXPERT_IDS: tl.constexpr,
    X_DTYPE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE_D: tl.constexpr,
) -> None:
    pid = tl.program_id(axis=0).to(tl.int64)

    if COPY_X_TO_BUFFER:
        copy_x_to_buffer = (
            tl.load(x_copy_condition_ptr, eviction_policy="evict_last") == 0
        )
        x_copy_byte_offset = tl.load(
            x_copy_offset_ptr, eviction_policy="evict_last"
        ).to(tl.int64)
        x_copy_buffer_raw = (x_copy_buffer_ptr + x_copy_byte_offset).to(
            tl.pointer_type(X_DTYPE)
        )
        x_copy_align: tl.constexpr = MEMORY_ALIGNMENT_TL_CONSTEXPR // (
            X_DTYPE.primitive_bitwidth // 8
        )
        x_copy_buffer_base = tl.multiple_of(x_copy_buffer_raw, x_copy_align)

    # Compute offsets for the current program.
    block_t = tl.arange(0, BLOCK_SIZE_T)
    # BLOCK_K is host-computed (= next_power_of_2(K)); newer Triton's jit rejects
    # calling triton.next_power_of_2 in-kernel.
    block_k = tl.arange(0, BLOCK_K)
    mask_k = block_k < K
    block_d = tl.arange(0, BLOCK_SIZE_D)

    if TILE_D:
        # Flattening the T and D tile grid keeps every CTA busy for decode.
        num_d_tiles: tl.constexpr = tl.cdiv(D, BLOCK_SIZE_D)
        num_t_tiles = tl.cdiv(T, BLOCK_SIZE_T)
        num_tiles = num_t_tiles * num_d_tiles
        num_tiles_per_cta = tl.cdiv(num_tiles, NUM_CTA)
        tile_start = pid * num_tiles_per_cta
        tile_end = min(tile_start + num_tiles_per_cta, num_tiles)

        for tile_id in tl.range(tile_start, tile_end):
            start_t = (tile_id // num_d_tiles) * BLOCK_SIZE_T
            start_d = (tile_id % num_d_tiles) * BLOCK_SIZE_D

            input_offset_t = (start_t + block_t)[:, None, None]
            input_offset_k = block_k[None, :, None]
            input_mask_t = input_offset_t < T
            input_mask_tk = input_mask_t & mask_k[None, :, None]

            if HAS_EXPERT_IDS:
                expert_ids = tl.load(
                    expert_ids_ptr + input_offset_t * K + input_offset_k,
                    mask=input_mask_tk,
                    other=-1,
                )
                valid_routes = expert_ids >= 0
                input_mask_tk &= valid_routes

            if scale_ptr is not None:
                scale = tl.load(
                    scale_ptr + input_offset_t * K + input_offset_k,
                    mask=input_mask_tk,
                    other=0.0,
                ).to(tl.float32)
            else:
                scale = tl.full((BLOCK_SIZE_T, BLOCK_K, 1), 1.0, dtype=tl.float32)

            input_offset_d = (start_d + block_d)[None, None, :]
            output_mask = (input_offset_t < T) & (input_offset_d < D)
            input_mask = output_mask & input_mask_tk
            x_value = tl.load(
                x_ptr + input_offset_t * K * D + input_offset_k * D + input_offset_d,
                mask=input_mask,
                other=0.0,
            )
            x = x_value.to(tl.float32)

            if COPY_X_TO_BUFFER:
                if copy_x_to_buffer:
                    tl.store(
                        x_copy_buffer_base
                        + input_offset_t * K * D
                        + input_offset_k * D
                        + input_offset_d,
                        x_value,
                        mask=output_mask & mask_k[None, :, None],
                    )

            if COPY_X:
                tl.store(
                    x_copy_ptr
                    + input_offset_t * K * D
                    + input_offset_k * D
                    + input_offset_d,
                    x,
                    mask=output_mask & mask_k[None, :, None],
                )

            y = tl.sum(x * scale, axis=1, keep_dims=True)
            tl.store(
                y_ptr + input_offset_t * D + input_offset_d,
                y,
                mask=output_mask,
            )
    else:
        # Preserve the original T-only persistent schedule for prefill/training.
        num_tiles = tl.cdiv(T, BLOCK_SIZE_T)
        num_tiles_per_cta = tl.cdiv(num_tiles, NUM_CTA)
        tile_start = pid * num_tiles_per_cta
        tile_end = min(tile_start + num_tiles_per_cta, num_tiles)

        for tile_id in tl.range(tile_start, tile_end):
            start_t = tile_id * BLOCK_SIZE_T

            input_offset_t = (start_t + block_t)[:, None, None]
            input_offset_k = block_k[None, :, None]
            input_mask_t = input_offset_t < T
            input_mask_tk = input_mask_t & mask_k[None, :, None]

            if HAS_EXPERT_IDS:
                expert_ids = tl.load(
                    expert_ids_ptr + input_offset_t * K + input_offset_k,
                    mask=input_mask_tk,
                    other=-1,
                )
                valid_routes = expert_ids >= 0
                input_mask_tk &= valid_routes

            if scale_ptr is not None:
                scale = tl.load(
                    scale_ptr + input_offset_t * K + input_offset_k,
                    mask=input_mask_tk,
                    other=0.0,
                ).to(tl.float32)
            else:
                scale = tl.full((BLOCK_SIZE_T, BLOCK_K, 1), 1.0, dtype=tl.float32)

            for start_d in tl.range(0, D, BLOCK_SIZE_D):
                input_offset_d = (start_d + block_d)[None, None, :]
                output_mask = (input_offset_t < T) & (input_offset_d < D)
                input_mask = output_mask & input_mask_tk
                x_value = tl.load(
                    x_ptr
                    + input_offset_t * K * D
                    + input_offset_k * D
                    + input_offset_d,
                    mask=input_mask,
                    other=0.0,
                )
                x = x_value.to(tl.float32)

                if COPY_X_TO_BUFFER:
                    if copy_x_to_buffer:
                        tl.store(
                            x_copy_buffer_base
                            + input_offset_t * K * D
                            + input_offset_k * D
                            + input_offset_d,
                            x_value,
                            mask=output_mask & mask_k[None, :, None],
                        )

                if COPY_X:
                    tl.store(
                        x_copy_ptr
                        + input_offset_t * K * D
                        + input_offset_k * D
                        + input_offset_d,
                        x,
                        mask=output_mask & mask_k[None, :, None],
                    )

                y = tl.sum(x * scale, axis=1, keep_dims=True)
                tl.store(
                    y_ptr + input_offset_t * D + input_offset_d,
                    y,
                    mask=output_mask,
                )


def broadcast_and_scale(
    dy: torch.Tensor,  # [T, D]
    scale: torch.Tensor,  # [T, K]
    x: torch.Tensor,  # [T, K, D]
    expert_ids: torch.Tensor | None = None,  # optional [T, K]
    output_dtype: torch.dtype | None = None,
    dx: torch.Tensor | None = None,  # optional preallocated [T, K, D]
    x_buffer: torch.Tensor | None = None,
    x_buffer_offset: torch.Tensor | None = None,
    x_buffer_condition: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast pre dispatch gradients, scale them, and calculate scale gradients.

    If `dx` is supplied, it must be disjoint from `x` or exactly alias `x`.
    Partial/offset overlap is unsupported.

    The ``dscale`` pass reads ``x`` before the ``dx`` pass writes, so an
    exactly-aliased ``dx`` stays safe via stream ordering.
    """
    # Check dtypes
    dy_dtype = dy.dtype
    scale_dtype = scale.dtype
    x_dtype = x.dtype
    dx_dtype = output_dtype if output_dtype is not None else dy_dtype
    assert dy_dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"{dy_dtype=} not supported"
    )
    assert scale_dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert x_dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"{x_dtype=} not supported"
    )
    assert dx_dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"{dx_dtype=} not supported"
    )

    # Check shape
    T, D = dy.shape
    T_, K = scale.shape
    T__, K_, D_ = x.shape
    assert T == T_
    assert T == T__
    assert D == D_
    assert K == K_

    # Check strides
    assert dy.is_contiguous()
    assert scale.is_contiguous()
    assert x.is_contiguous()
    if expert_ids is not None:
        if expert_ids.shape != (T, K) or expert_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("expert_ids must have shape [T, K] and int32/int64 dtype")
        if not expert_ids.is_contiguous() or expert_ids.device != x.device:
            raise ValueError("expert_ids must be contiguous on the input device")
    read_x_from_buffer = _validate_activation_buffer_args(
        activation_buffer=x_buffer,
        activation_offset=x_buffer_offset,
        condition=x_buffer_condition,
        device=x.device,
    )

    dx_shape = (T, K, D)
    x_dx_exact_alias = False
    if dx is None:
        dx = torch.empty(dx_shape, device=x.device, dtype=dx_dtype)
    else:
        assert dx.shape == dx_shape
        assert dx.dtype == dx_dtype
        assert dx.device == x.device
        assert dx.is_contiguous()
        x_start = x.data_ptr()
        x_end = x_start + x.numel() * x.element_size()
        dx_start = dx.data_ptr()
        dx_end = dx_start + dx.numel() * dx.element_size()
        x_dx_exact_alias = x_start == dx_start and x_end == dx_end
        x_dx_overlap = x_start < dx_end and dx_start < x_end
        assert x_dx_exact_alias or not x_dx_overlap, (
            "`dx` must be disjoint from `x` or exactly alias `x`; "
            "partial/offset overlap is unsupported"
        )
    dscale = torch.empty((T, K), device=x.device, dtype=scale_dtype)

    # Check if we can use triton
    NUM_ELEMS = 8192
    if K <= 8:
        BLOCK_SIZE_D = 1024
        BLOCK_SIZE_T = NUM_ELEMS // (BLOCK_SIZE_D * K)
        # Round down to nearest power of 2
        BLOCK_SIZE_T = 1 << (BLOCK_SIZE_T.bit_length() - 1) if BLOCK_SIZE_T > 0 else 1
    else:
        BLOCK_SIZE_T = 1
        BLOCK_SIZE_D = NUM_ELEMS // (BLOCK_SIZE_T * K)
        # Round down to nearest power of 2
        BLOCK_SIZE_D = 1 << (BLOCK_SIZE_D.bit_length() - 1) if BLOCK_SIZE_D > 0 else 1
    # Dynamic number of CTAs based on SM count for better parallelism
    num_sms = num_sms_per_device()
    num_ctas = num_sms * 32
    grid = (num_ctas, 1, 1)
    _triton_broadcast_and_scale[grid](
        dy,
        scale,
        expert_ids,
        x,
        dx,
        dscale,
        None,  # activation_buffer_ptr
        None,  # conditional_execution_ptr
        x_buffer,
        x_buffer_offset,
        x_buffer_condition,
        T,
        K,
        D,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        DTYPE=_to_tl_dtype(dy_dtype),
        X_DTYPE=_to_tl_dtype(x_dtype),
        NUM_CTA=num_ctas,
        X_DX_MAY_ALIAS=x_dx_exact_alias,
        READ_X_FROM_BUFFER=read_x_from_buffer,
        HAS_EXPERT_IDS=expert_ids is not None,
        BLOCK_K=triton.next_power_of_2(K),
    )
    return dx, dscale


@triton.jit
def _broadcast_and_scale_d_block(
    dy_ptr,
    x_ptr,
    dx_ptr,  # None in dscale-only mode.
    dscale_ptr,
    scale,
    dscale_acc,
    input_offset_t,
    input_offset_k,
    start_d,
    block_d,
    route_mask_tk,
    mask_k,
    T,
    K: tl.constexpr,
    D: tl.constexpr,
    X_DX_MAY_ALIAS: tl.constexpr,
):
    """One D-tile of broadcast_and_scale: dx store + dscale accumulation."""
    input_offset_d = (start_d + block_d)[None, None, :]
    output_mask = (input_offset_t < T) & (input_offset_d < D)
    input_mask = output_mask & route_mask_tk
    output_route_mask = output_mask & mask_k[None, :, None]

    dy = tl.load(
        dy_ptr + input_offset_t * D + input_offset_d,
        mask=output_mask,
        other=0.0,
    ).to(tl.float32)
    dx = dy * scale

    # Compute dscale only if needed
    if dscale_ptr is not None:
        # In DistMoE recompute, x can be the local combine-buffer view
        # and dx can be that same view reused for the backward publish.
        x = tl.load(
            x_ptr + input_offset_t * K * D + input_offset_k * D + input_offset_d,
            mask=input_mask,
            other=0.0,
        ).to(tl.float32)
        # Accumulate dscale across D dimension
        dscale_acc += tl.sum(x * dy, axis=2, keep_dims=True)
        if X_DX_MAY_ALIAS:
            tl.debug_barrier()

    if dx_ptr is not None:
        tl.store(
            dx_ptr + input_offset_t * K * D + input_offset_k * D + input_offset_d,
            dx,
            mask=output_route_mask,
        )
    return dscale_acc


@triton.jit(
    do_not_specialize=["T"],
    repr=make_dtype_repr(
        "_triton_broadcast_and_scale", ["dy_ptr", "dx_ptr"], dtype_constexpr="DTYPE"
    ),
)
def _triton_broadcast_and_scale(
    # Regular mode: tensor pointers. Activation buffer mode: offset tensors
    dy_ptr,  # [T, D] tensor or [1] int64 byte offset
    scale_ptr,  # Can be None for unscaled broadcast
    expert_ids_ptr,
    x_ptr,  # [T, K, D] tensor for dscale computation
    dx_ptr,  # [T, K, D] tensor or [1] int64 byte offset
    dscale_ptr,  # Can be None when scale_ptr is None
    # Activation buffer mode
    activation_buffer_ptr,  # shared buffer (None for regular mode)
    # Conditional execution
    conditional_execution_ptr,
    x_buffer_ptr,
    x_buffer_offset_ptr,
    x_buffer_condition_ptr,
    # Common parameters
    T,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    DTYPE: tl.constexpr,
    X_DTYPE: tl.constexpr,
    NUM_CTA: tl.constexpr,
    X_DX_MAY_ALIAS: tl.constexpr,
    READ_X_FROM_BUFFER: tl.constexpr,
    HAS_EXPERT_IDS: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Unified kernel for broadcast/scale.

    Supports both regular mode and activation buffer mode.
    When activation_buffer_ptr is not None, dy_ptr and dx_ptr are byte offset tensors.
    """
    # Check conditional execution first
    if conditional_execution_ptr is not None:
        cond_val = tl.load(conditional_execution_ptr, eviction_policy="evict_last")
        if cond_val == 0:
            return

    # Setup pointers based on mode
    if activation_buffer_ptr is not None:
        # Activation buffer mode: dy_ptr and dx_ptr are offset tensors
        x_byte_offset = tl.load(dy_ptr, eviction_policy="evict_last").to(tl.int64)
        output_byte_offset = tl.load(dx_ptr, eviction_policy="evict_last").to(tl.int64)

        dy_ptr_raw = (activation_buffer_ptr + x_byte_offset).to(tl.pointer_type(DTYPE))
        dx_ptr_raw = (activation_buffer_ptr + output_byte_offset).to(
            tl.pointer_type(DTYPE)
        )
        # Apply tl.multiple_of to attach MEMORY_ALIGNMENT-byte alignment metadata to pointers
        # This enables vectorized memory operations
        align: tl.constexpr = MEMORY_ALIGNMENT_TL_CONSTEXPR // (
            DTYPE.primitive_bitwidth // 8
        )
        dy_ptr = tl.multiple_of(dy_ptr_raw, align)
        dx_ptr = tl.multiple_of(dx_ptr_raw, align)

    if READ_X_FROM_BUFFER:
        use_tensor_x = tl.load(x_buffer_condition_ptr, eviction_policy="evict_last").to(
            tl.int1
        )
        x_buffer_byte_offset = tl.load(
            x_buffer_offset_ptr, eviction_policy="evict_last"
        ).to(tl.int64)
        x_buffer_ptr_raw = (x_buffer_ptr + x_buffer_byte_offset).to(
            tl.pointer_type(X_DTYPE)
        )
        x_align: tl.constexpr = MEMORY_ALIGNMENT_TL_CONSTEXPR // (
            X_DTYPE.primitive_bitwidth // 8
        )
        x_buffer_base = tl.multiple_of(x_buffer_ptr_raw, x_align)
        x_ptr = tl.where(use_tensor_x, x_ptr, x_buffer_base)

    pid = tl.program_id(axis=0).to(tl.int64)

    # Compute offsets for the current program.
    block_t = tl.arange(0, BLOCK_SIZE_T)
    # BLOCK_K is host-computed (= next_power_of_2(K)); newer Triton's jit rejects
    # calling triton.next_power_of_2 in-kernel.
    block_k = tl.arange(0, BLOCK_K)
    mask_k = block_k < K
    block_d = tl.arange(0, BLOCK_SIZE_D)

    # Persistent kernel: each CTA processes multiple tiles in contiguous manner
    num_tiles = tl.cdiv(T, BLOCK_SIZE_T)
    num_tiles_per_cta = tl.cdiv(num_tiles, NUM_CTA)
    tile_start = pid * num_tiles_per_cta
    tile_end = min(tile_start + num_tiles_per_cta, num_tiles)

    for tile_id in tl.range(tile_start, tile_end):
        start_t = tile_id * BLOCK_SIZE_T

        input_offset_t = (start_t + block_t)[:, None, None]
        input_offset_k = block_k[None, :, None]
        input_mask_t = input_offset_t < T
        input_mask_tk = input_mask_t & mask_k[None, :, None]

        if HAS_EXPERT_IDS:
            expert_ids = tl.load(
                expert_ids_ptr + input_offset_t * K + input_offset_k,
                mask=input_mask_tk,
                other=-1,
            )
            input_mask_tk &= expert_ids >= 0

        # Load scale if provided, otherwise use 1.0
        if scale_ptr is not None:
            scale = tl.load(
                scale_ptr + input_offset_t * K + input_offset_k,
                mask=input_mask_tk,
                other=0.0,
            ).to(tl.float32)
        else:
            scale = tl.full((BLOCK_SIZE_T, BLOCK_K, 1), 1.0, dtype=tl.float32)

        # Unused (and dead-code-eliminated) when dscale_ptr is None.
        dscale_acc = tl.zeros((BLOCK_SIZE_T, BLOCK_K, 1), dtype=tl.float32)

        for start_d in tl.range(0, D, BLOCK_SIZE_D):
            dscale_acc = _broadcast_and_scale_d_block(
                dy_ptr,
                x_ptr,
                dx_ptr,
                dscale_ptr,
                scale,
                dscale_acc,
                input_offset_t,
                input_offset_k,
                start_d,
                block_d,
                input_mask_tk,
                mask_k,
                T,
                K,
                D,
                X_DX_MAY_ALIAS,
            )

        # Store final accumulated dscale only if needed
        if dscale_ptr is not None:
            tl.store(
                dscale_ptr + input_offset_t * K + input_offset_k,
                dscale_acc,
                mask=input_mask_t & mask_k[None, :, None],
            )


def reduce_from_topk(
    x: torch.Tensor,  # [T, K, D]
    expert_ids: torch.Tensor | None = None,  # optional [T, K]
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Reduce from topk dimension by summing without scaling.

    This is a special case of scale_and_sum with scale=None (no scaling).
    Reduces [T, K, D] -> [T, D] by summing across K dimension.

    Args:
        x: Input tensor [T, K, D]
        expert_ids: Optional expert IDs. Negative routes are excluded.
        out_dtype: Output dtype; defaults to ``x``'s. The kernel sums in fp32,
            so ``torch.float32`` keeps that sum instead of narrowing it.

    Returns:
        Reduced output [T, D]
    """
    # Check dtypes
    x_dtype = x.dtype
    assert x_dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"{x_dtype=} not supported"
    )

    # Check shape
    T, K, D = x.shape

    # Check strides
    assert x.is_contiguous()
    if expert_ids is not None:
        if expert_ids.shape != (T, K) or expert_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("expert_ids must have shape [T, K] and int32/int64 dtype")
        if not expert_ids.is_contiguous() or expert_ids.device != x.device:
            raise ValueError("expert_ids must be contiguous on the input device")

    # Allocate output
    y = torch.empty((T, D), device=x.device, dtype=out_dtype or x_dtype)

    # Use scale_and_sum kernel with scale=None for summing only
    # Pass None for x_copy to skip the copy operation
    NUM_ELEMS = 8192
    if K <= 8:
        BLOCK_SIZE_D = 1024
        BLOCK_SIZE_T = NUM_ELEMS // (BLOCK_SIZE_D * K)
        # Round down to nearest power of 2
        BLOCK_SIZE_T = 1 << (BLOCK_SIZE_T.bit_length() - 1) if BLOCK_SIZE_T > 0 else 1
    else:
        BLOCK_SIZE_T = 1
        BLOCK_SIZE_D = NUM_ELEMS // (BLOCK_SIZE_T * K)
        # Round down to nearest power of 2
        BLOCK_SIZE_D = 1 << (BLOCK_SIZE_D.bit_length() - 1) if BLOCK_SIZE_D > 0 else 1

    # Dynamic number of CTAs based on SM count for better parallelism
    num_sms = num_sms_per_device()
    num_ctas = num_sms * 32
    grid = (num_ctas, 1, 1)
    _triton_scale_and_sum[grid](
        x,
        None,  # scale_ptr = None (no scaling, just sum)
        expert_ids,
        y,
        None,  # x_copy_ptr = None (skip copy for forward-only reduction)
        None,  # x_copy_buffer_ptr
        None,  # x_copy_offset_ptr
        None,  # x_copy_condition_ptr
        T,
        K,
        D,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        NUM_CTA=num_ctas,
        COPY_X=False,
        COPY_X_TO_BUFFER=False,
        HAS_EXPERT_IDS=expert_ids is not None,
        X_DTYPE=_to_tl_dtype(x_dtype),
        BLOCK_K=triton.next_power_of_2(K),
        TILE_D=T <= SCALE_AND_SUM_TILE_D_MAX_TOKENS,
    )
    return y
