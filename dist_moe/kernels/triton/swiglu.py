# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SwiGLU Activation Function - Triton Implementation.

This module provides high-performance Triton GPU kernels for the SwiGLU (Swish-Gated Linear Unit)
activation function, which is commonly used in transformer models and LLMs.

SwiGLU combines the Swish (SiLU) activation with a gating mechanism:
    SwiGLU(x, y) = x * σ(x) * y
    where σ(x) = sigmoid(x)

The implementation includes:
- Forward pass: Fused computation of x * sigmoid(x) * y
- Backward pass: Efficient gradient computation for both inputs x and y

Performance Optimizations:
- Uses Triton JIT compilation for GPU acceleration
- 2D tiling strategy to maximize memory throughput and register usage
- Contiguous memory access patterns for efficient HBM/SRAM transfers
- Single sigmoid computation in backward pass (reused for both gradients)
- Float32 computation for numerical stability with mixed precision support

Key Functions:
- swiglu_fwd: Forward pass computing z = x * sigmoid(x) * y
- swiglu_bwd: Backward pass computing gradients dx and dy

Example Usage:
    >>> x = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
    >>> y = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
    >>> z = swiglu_fwd(x, y)
    >>> dz = torch.randn_like(z)
    >>> dx, dy = swiglu_bwd(dz, x, y)
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

from ..._activation_buffer_planner_kernel import (
    MEMORY_ALIGNMENT_TL_CONSTEXPR,
)
from ...formats import (
    SWIGLU_CLAMP_ALPHA_DEFAULT,
    SWIGLU_CLAMP_LIMIT_DEFAULT,
)
from .._environment import is_blackwell_gpu, num_sms_per_device
from ..pdl import pdl_launch_kwargs, pdl_wait_then_release
from .fp_math import sigmoid
from .repr_utils import make_dtype_repr


@triton.jit(
    do_not_specialize=["T"],
    repr=make_dtype_repr(
        "_triton_swiglu_fwd", ["ptr_x", "ptr_z"], dtype_constexpr="DTYPE"
    ),
)
def _triton_swiglu_fwd(  # noqa: C901
    # Regular mode pointers
    ptr_x,
    stride_x_t,
    ptr_y,
    stride_y_t,
    ptr_z,
    stride_z_t,
    # Activation buffer mode pointers (pass 0/null for regular mode)
    activation_buffer_ptr,
    xy_offset_ptr,
    z_offset_ptr,
    num_recv_tokens_ptr,
    # Conditional execution support
    conditional_execution_ptr,
    # Opt-in clip-count accumulator: float32 [3] = [gate_over, up_over, total]
    clip_stats_ptr,
    # Common parameters
    T,
    CTA_CAP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_TILES: tl.constexpr,
    DTYPE: tl.constexpr,
    FAST_MATH: tl.constexpr,
    CLIP_LIMIT: tl.constexpr,
    DO_CLIP: tl.constexpr,
    CLAMPED: tl.constexpr,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    PDL: tl.constexpr,
):
    """Unified Triton kernel for SwiGLU forward pass with persistent kernel design.

    Computes z = x * sigmoid(x) * y element-wise, or the GPT-OSS-style clamped
    variant z = g * sigmoid(ALPHA * g) * (u + 1) when CLAMPED, where
    g = min(x, LIMIT) and u = clamp(y, -LIMIT, LIMIT).
    Uses persistent kernel design where each CTA iterates over tiles of
    BLOCK_T rows by all D columns (D_TILES == 1) or by one BLOCK_D column
    block (D_TILES > 1).

    Supports both regular mode (ptr_x, ptr_y, ptr_z) and activation_buffer mode.
    When activation_buffer_ptr is not None, uses activation_buffer mode with pointer chasing.
    In activation_buffer mode, xy_offset_ptr and z_offset_ptr contain explicit byte offsets.
    """
    pdl_wait_then_release(PDL)
    # Load conditional execution value (check happens after pointer setup)
    cond_val = 1  # Default: execute
    if conditional_execution_ptr is not None:
        cond_val = tl.load(conditional_execution_ptr, eviction_policy="evict_last")

    # Check conditional execution - skip kernel if value is 0
    if cond_val == 0:
        return

    if num_recv_tokens_ptr is not None:
        active_T = tl.load(num_recv_tokens_ptr, eviction_policy="evict_last").to(
            tl.int64
        )
        if activation_buffer_ptr is None:
            T = tl.minimum(T, active_T)
        else:
            T = active_T

    if activation_buffer_ptr is not None:
        # Load byte offsets from offset tensors
        xy_data_offset = tl.load(xy_offset_ptr, eviction_policy="evict_last").to(
            tl.int64
        )
        z_data_offset = tl.load(z_offset_ptr, eviction_policy="evict_last").to(tl.int64)
        # Compute typed pointers from byte offsets
        ptr_xy_raw = (activation_buffer_ptr + xy_data_offset).to(tl.pointer_type(DTYPE))
        ptr_z_raw = (activation_buffer_ptr + z_data_offset).to(tl.pointer_type(DTYPE))

        # Apply tl.multiple_of to attach MEMORY_ALIGNMENT-byte alignment metadata to pointers
        # This enables vectorized memory operations
        align: tl.constexpr = MEMORY_ALIGNMENT_TL_CONSTEXPR // (
            DTYPE.primitive_bitwidth // 8
        )
        ptr_xy = tl.multiple_of(ptr_xy_raw, align)
        ptr_z_ab = tl.multiple_of(ptr_z_raw, align)

        # For activation_buffer mode, xy is (T, 2*D) and we split into x and y
        ptr_x = ptr_xy
        ptr_y = ptr_xy + D  # Offset by D elements for y
        ptr_z = ptr_z_ab
        # stride_x_t, stride_y_t, stride_z_t are passed from host

    # Get the program ID
    pid = tl.program_id(axis=0)

    # Per-CTA clip-count accumulators kept in registers across the tile loop,
    # flushed once per CTA via a single atomic_add per slot (minimizes atomics).
    # Dead when DO_CLIP is False (constexpr-eliminated -> zero perf impact).
    if DO_CLIP:
        gate_over_acc = tl.zeros((), tl.float32)
        up_over_acc = tl.zeros((), tl.float32)
        total_acc = tl.zeros((), tl.float32)

    # Below CTA_CAP, the launch grid still has at least one CTA per active tile.
    num_tiles = tl.cdiv(T, BLOCK_T) * D_TILES
    num_tiles_per_cta = tl.cdiv(num_tiles, CTA_CAP)
    tile_start = pid * num_tiles_per_cta
    tile_end = min(tile_start + num_tiles_per_cta, num_tiles)
    for tile_id in tl.range(tile_start, tile_end):
        if D_TILES == 1:
            row_tile = tile_id
            d_start = 0
            d_end = D
        else:
            row_tile = tile_id // D_TILES
            d_start = tile_id % D_TILES * BLOCK_D
            d_end = d_start + BLOCK_D
        # Compute row indices for this tile - use int64 to avoid overflow
        t = (row_tile * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
        # Create mask to handle boundary conditions
        mask_t = t < T

        # Loop over this tile's columns in blocks of BLOCK_D
        for offset_d in tl.range(d_start, d_end, BLOCK_D):
            # Compute column indices for this tile
            d = offset_d + tl.arange(0, BLOCK_D)
            # Create mask to handle boundary conditions in D dimension
            mask_d = d < D

            # Create 2D mask by broadcasting 1D masks
            mask = mask_t[:, None] & mask_d[None, :]

            # Load input blocks from HBM to SRAM
            # Cast to float32 for computation precision
            x = tl.load(
                ptr_x + t[:, None] * stride_x_t + d[None, :], mask=mask, other=0.0
            ).to(tl.float32)
            y = tl.load(
                ptr_y + t[:, None] * stride_y_t + d[None, :], mask=mask, other=0.0
            ).to(tl.float32)

            if CLAMPED:
                g = tl.minimum(x, LIMIT)
                u = tl.minimum(tl.maximum(y, -LIMIT), LIMIT)
                z = g * sigmoid(ALPHA * g, FAST_MATH=FAST_MATH) * (u + 1.0)
            else:
                # Compute SwiGLU: x * sigmoid(x) * y
                z = x * sigmoid(x, FAST_MATH=FAST_MATH) * y

            # Store result back to HBM
            tl.store(ptr_z + t[:, None] * stride_z_t + d[None, :], z, mask=mask)

            # Opt-in clip counting on the pre-activation gate (x) and up (y)
            # tensors already loaded above. Strict `>` matches the torch /
            # triton-path clip-logging semantics. `mask` excludes padding.
            if DO_CLIP:
                gate_over_acc += tl.sum((mask & (x > CLIP_LIMIT)).to(tl.float32))
                up_over_acc += tl.sum((mask & (tl.abs(y) > CLIP_LIMIT)).to(tl.float32))
                total_acc += tl.sum(mask.to(tl.float32))

    # Flush this CTA's accumulated counts into the shared [3] output.
    if DO_CLIP:
        tl.atomic_add(clip_stats_ptr + 0, gate_over_acc)
        tl.atomic_add(clip_stats_ptr + 1, up_over_acc)
        tl.atomic_add(clip_stats_ptr + 2, total_acc)


def _swiglu_fwd_tile(d: int) -> tuple[int, int, int, int]:
    """``(BLOCK_T, BLOCK_D, num_warps, D_TILES)`` for the forward kernel:
    narrow tiles, each its own work item, when they cut the padded lanes of
    2048-wide tiles by at least 15%."""
    wide_lanes = -(-d // 2048) * 2048
    narrow_lanes = -(-d // 512) * 512
    if narrow_lanes <= 0.85 * wide_lanes:
        return 4, 512, 4, narrow_lanes // 512
    return 2, 2048, 8, 1


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


def _validate_num_recv_tokens(
    num_recv_tokens: torch.Tensor | None,
    *,
    device: torch.device,
) -> None:
    """Validate optional device int32[1] row-count metadata without reading it."""
    if num_recv_tokens is None:
        return
    if num_recv_tokens.shape != (1,) or num_recv_tokens.dtype != torch.int32:
        raise ValueError(
            "num_recv_tokens must have shape [1] and dtype int32; got "
            f"shape={tuple(num_recv_tokens.shape)}, dtype={num_recv_tokens.dtype}"
        )
    if num_recv_tokens.device != device:
        raise ValueError(
            "num_recv_tokens must be on the input device; got "
            f"{num_recv_tokens.device} and {device}"
        )
    if not num_recv_tokens.is_contiguous():
        raise ValueError("num_recv_tokens must be contiguous")


def swiglu_fwd(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    *,
    activation_buffer: torch.Tensor | None = None,
    num_recv_tokens: torch.Tensor | None = None,
    feature_dim: int | None = None,
    dtype: torch.dtype | None = None,
    conditional_execution: torch.Tensor | None = None,
    fast_math: bool = False,
    clip_stats_out: torch.Tensor | None = None,
    clip_limit: float = 7.0,
    clamped: bool = False,
    alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    max_rows: int | None = None,
) -> torch.Tensor:
    """
    Fused silu and mul operations.

    z = x * sigmoid(x) * y

    Args:
        x: input tensor of shape (T, D) OR concatenated tensor of shape (T, 2*D) if y is None.
           When activation_buffer is provided, this is a byte offset tensor of shape [1], dtype int64,
           pointing to the input location in activation_buffer (xy_offset).
        y: optional input tensor of shape (T, D). If None, x is chunked internally.
           Not used when activation_buffer is provided.
        z: optional output tensor of shape (T, D). If None in default mode, allocated with torch.empty.
           When activation_buffer is provided, z is REQUIRED and must be a byte offset tensor of
           shape [1], dtype int64, pointing to the output location in activation_buffer (z_offset).
        activation_buffer: Optional persistent buffer for reading input and writing output.
            When provided, input is read from x (xy_offset) and output is written to z (z_offset).
        num_recv_tokens: Optional CUDA tensor containing the active row count.
            Shape: [1], dtype: torch.int32. The output keeps its capacity shape;
            rows at and beyond the bound retain caller-provided contents or are
            uninitialized when output storage is allocated internally.
        feature_dim: Optional hidden dimension (D). Required when activation_buffer is provided since
           dimension cannot be inferred from the offset tensor.
        dtype: Optional data type. Required when activation_buffer is provided since the dtype
           cannot be inferred from the offset tensor.
        conditional_execution: Optional device tensor with single integer element.
            If None, ignored. If value is 0, skip kernel execution. If non-zero, execute as usual.
        clip_stats_out: Optional float32 device tensor of shape [3] =
            [gate_over_count, up_over_count, total_count]. When provided, the
            kernel fuses a clip-count pass: it is zeroed inside this call and
            then accumulated into (gate elements > clip_limit, up elements with
            abs() > clip_limit, and the number of valid elements). When None
            (default) the counting code is constexpr-eliminated from the kernel,
            leaving behavior and performance unchanged.
        clip_limit: Threshold used by the clip-count pass (strict `>`). Only
            meaningful when clip_stats_out is provided.
        max_rows: Optional host upper bound on the device row count in
            activation_buffer mode. It sizes the launch grid, and rows past it
            are never computed, so it must not be smaller than the row count.

    Returns:
        z: output tensor of shape (T, D), or z (z_offset) when activation_buffer is used.
    """
    # Handle chunked mode (y=None) - recursive call for regular mode only
    if y is None and activation_buffer is None:
        D_doubled = x.shape[-1]
        assert D_doubled % 2 == 0, f"Last dim must be even for chunking: {D_doubled}"
        D = D_doubled // 2
        x_left = x[..., :D]
        x_right = x[..., D:]
        return swiglu_fwd(
            x_left,
            x_right,
            z=z,
            num_recv_tokens=num_recv_tokens,
            conditional_execution=conditional_execution,
            fast_math=fast_math,
            clip_stats_out=clip_stats_out,
            clip_limit=clip_limit,
            clamped=clamped,
            alpha=alpha,
            limit=limit,
        )

    # Common setup
    cta_cap = num_sms_per_device() * 32
    num_stages = 1
    _validate_num_recv_tokens(num_recv_tokens, device=x.device)

    # Opt-in fused clip counting. Validate + zero here so counts are fresh each
    # call; zeroing stays inside any captured region.
    do_clip = clip_stats_out is not None
    if do_clip:
        assert clip_stats_out.shape == (3,), (
            f"clip_stats_out must have shape [3]; got {tuple(clip_stats_out.shape)}"
        )
        assert clip_stats_out.dtype == torch.float32, (
            f"clip_stats_out must be float32; got {clip_stats_out.dtype}"
        )
        assert clip_stats_out.device == x.device, (
            f"clip_stats_out must be on the input device; got "
            f"{clip_stats_out.device} and {x.device}"
        )
        clip_stats_out.zero_()

    if activation_buffer is not None:
        # Activation buffer mode setup
        # In this mode, x is a byte offset tensor pointing to the input in activation_buffer
        # and z is a byte offset tensor pointing to the output in activation_buffer
        # Check activation_buffer
        assert activation_buffer.is_contiguous(), "activation_buffer must be contiguous"
        # Check x is an offset tensor (xy_offset)
        assert x.shape == (1,) and x.dtype == torch.int64, (
            f"x must be shape [1] and dtype int64 when activation_buffer is provided, got shape={x.shape}, dtype={x.dtype}"
        )
        # Check z is an offset tensor (z_offset)
        assert z is not None, "z (z_offset) required when activation_buffer is provided"
        assert z.shape == (1,) and z.dtype == torch.int64, (
            f"z must be shape [1] and dtype int64 when activation_buffer is provided, got shape={z.shape}, dtype={z.dtype}"
        )
        assert num_recv_tokens is not None, "num_recv_tokens required"
        # Check feature_dim
        assert feature_dim is not None, (
            "feature_dim is required when activation_buffer is provided"
        )
        # Check dtype
        assert dtype is not None, "dtype is required when activation_buffer is provided"

        # x is the input offset tensor (xy_offset)
        xy_offset = x
        # z is the output offset tensor (z_offset)
        z_offset = z

        # Calculate strides for activation_buffer mode: xy is (T, 2*D), z is (T, D)
        stride_xy_t = feature_dim * 2
        stride_z_t = feature_dim
        D = feature_dim
        T = 0  # Not used in activation_buffer mode, T comes from device tensor

        # Single kernel launch for activation_buffer mode
        BLOCK_T, BLOCK_D, num_warps, d_tiles = _swiglu_fwd_tile(D)
        # CTAs past the active tiles have none, so a host row bound caps the grid.
        grid_ctas = cta_cap
        if max_rows is not None:
            grid_ctas = max(1, min(cta_cap, triton.cdiv(max_rows, BLOCK_T) * d_tiles))
        grid = (grid_ctas, 1, 1)
        _triton_swiglu_fwd[grid](
            ptr_x=xy_offset,
            stride_x_t=stride_xy_t,
            ptr_y=xy_offset,
            stride_y_t=stride_xy_t,
            ptr_z=z_offset,
            stride_z_t=stride_z_t,
            activation_buffer_ptr=activation_buffer,
            xy_offset_ptr=xy_offset,
            z_offset_ptr=z_offset,
            num_recv_tokens_ptr=num_recv_tokens,
            conditional_execution_ptr=conditional_execution,
            clip_stats_ptr=clip_stats_out if do_clip else None,
            T=T,
            CTA_CAP=cta_cap,
            D=D,
            BLOCK_T=BLOCK_T,
            BLOCK_D=BLOCK_D,
            D_TILES=d_tiles,  # ty: ignore[invalid-argument-type]
            num_warps=num_warps,
            DTYPE=_to_tl_dtype(dtype),
            FAST_MATH=fast_math,
            CLIP_LIMIT=clip_limit,
            DO_CLIP=do_clip,
            CLAMPED=clamped,
            ALPHA=alpha,
            LIMIT=limit,
            **pdl_launch_kwargs(),
        )

        return z_offset
    else:
        # Regular mode with separate x and y tensors
        dtype_x = x.dtype
        dtype_y = y.dtype
        assert dtype_x == dtype_y, f"{dtype_x} != {dtype_y}"

        shape_x = x.shape
        shape_y = y.shape
        assert shape_x == shape_y, f"{shape_x} != {shape_y}"

        D = shape_x[-1]
        if len(shape_x) != 2:
            x = x.reshape(-1, D)
            y = y.reshape(-1, D)
            T = x.shape[0]
        else:
            T = shape_x[0]

        stride_x = x.stride()
        stride_y = y.stride()
        assert stride_x[-1] == 1, f"{stride_x=}"
        assert stride_y[-1] == 1, f"{stride_y=}"

        # Handle z output tensor: allocate if not provided
        if z is None:
            z = x.new_empty(shape_x)
        else:
            assert z.shape == shape_x, f"z shape {z.shape} != x shape {shape_x}"
            assert z.dtype == dtype_x, f"z dtype {z.dtype} != x dtype {dtype_x}"
            assert z.is_contiguous(), f"z must be contiguous: {z.stride()=}"

        # Size the CUDA-graph-stable launch from capacity. The kernel loads the
        # active row count from device when num_recv_tokens is provided.
        BLOCK_T, BLOCK_D, num_warps, d_tiles = _swiglu_fwd_tile(D)
        num_t_tiles = (T + BLOCK_T - 1) // BLOCK_T
        num_ctas = min(cta_cap, num_t_tiles * d_tiles)
        grid = (num_ctas, 1, 1)
        _triton_swiglu_fwd[grid](
            ptr_x=x,
            stride_x_t=stride_x[0],
            ptr_y=y,
            stride_y_t=stride_y[0],
            ptr_z=z,
            stride_z_t=D,
            activation_buffer_ptr=None,
            xy_offset_ptr=None,
            z_offset_ptr=None,
            num_recv_tokens_ptr=num_recv_tokens,
            conditional_execution_ptr=conditional_execution,
            clip_stats_ptr=clip_stats_out if do_clip else None,
            T=T,
            CTA_CAP=cta_cap,
            D=D,
            BLOCK_T=BLOCK_T,
            BLOCK_D=BLOCK_D,
            D_TILES=d_tiles,  # ty: ignore[invalid-argument-type]
            num_warps=num_warps,
            num_stages=num_stages,
            DTYPE=_to_tl_dtype(dtype_x),
            FAST_MATH=fast_math,
            CLIP_LIMIT=clip_limit,
            DO_CLIP=do_clip,
            CLAMPED=clamped,
            ALPHA=alpha,
            LIMIT=limit,
            **pdl_launch_kwargs(),
        )

        return z


@triton.jit(
    do_not_specialize=["T"],
    repr=make_dtype_repr(
        "_triton_swiglu_bwd", ["ptr_dz", "ptr_dx"], dtype_constexpr="DTYPE"
    ),
)
def _triton_swiglu_bwd(
    # dz input (regular tensor pointer or offset tensor)
    ptr_dz,
    stride_dz_t,
    # Regular mode pointers
    ptr_x,
    stride_x_t,
    ptr_y,
    stride_y_t,
    ptr_dx,
    stride_dx_t,
    ptr_dy,
    stride_dy_t,
    # Activation buffer mode pointers (pass 0/null for regular mode)
    activation_buffer_ptr,
    xy_offset_ptr,
    dxy_offset_ptr,
    dz_offset_ptr,  # Added: dz offset when using activation_buffer mode
    num_recv_tokens_ptr,
    # Common parameters
    T,
    CTA_CAP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DTYPE: tl.constexpr,
    FAST_MATH: tl.constexpr,
    CLAMPED: tl.constexpr,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
):
    """Unified Triton kernel for SwiGLU backward pass with persistent kernel design.

    Computes gradients for SwiGLU activation:
    dx = dz * y * sigmoid(x) * (1 + x - x * sigmoid(x))
    dy = dz * x * sigmoid(x)

    Supports both regular mode and activation_buffer mode.
    When activation_buffer_ptr is not None, uses activation_buffer mode with pointer chasing.
    In activation_buffer mode, xy_offset_ptr and dxy_offset_ptr contain explicit byte offsets.
    dz_offset_ptr is optional - if provided, dz is also loaded from activation_buffer.
    """
    if num_recv_tokens_ptr is not None:
        active_T = tl.load(num_recv_tokens_ptr, eviction_policy="evict_last").to(
            tl.int64
        )
        if activation_buffer_ptr is None:
            T = tl.minimum(T, active_T)
        else:
            T = active_T

    if activation_buffer_ptr is not None:
        # Load byte offsets from offset tensors
        xy_data_offset = tl.load(xy_offset_ptr, eviction_policy="evict_last").to(
            tl.int64
        )
        dxy_data_offset = tl.load(dxy_offset_ptr, eviction_policy="evict_last").to(
            tl.int64
        )
        # Compute typed pointers from byte offsets
        ptr_xy_raw = (activation_buffer_ptr + xy_data_offset).to(tl.pointer_type(DTYPE))
        ptr_dxy_raw = (activation_buffer_ptr + dxy_data_offset).to(
            tl.pointer_type(DTYPE)
        )

        # Apply tl.multiple_of to attach MEMORY_ALIGNMENT-byte alignment metadata to pointers
        # This enables vectorized memory operations
        align: tl.constexpr = MEMORY_ALIGNMENT_TL_CONSTEXPR // (
            DTYPE.primitive_bitwidth // 8
        )
        ptr_xy = tl.multiple_of(ptr_xy_raw, align)
        ptr_dxy = tl.multiple_of(ptr_dxy_raw, align)

        # Handle dz offset if provided (full activation_buffer mode)
        if dz_offset_ptr is not None:
            dz_data_offset = tl.load(dz_offset_ptr, eviction_policy="evict_last").to(
                tl.int64
            )
            ptr_dz_raw = (activation_buffer_ptr + dz_data_offset).to(
                tl.pointer_type(DTYPE)
            )
            ptr_dz = tl.multiple_of(ptr_dz_raw, align)

        # For activation_buffer mode, xy and dxy are (T, 2*D)
        ptr_x = ptr_xy
        ptr_y = ptr_xy + D  # Offset by D elements for y
        ptr_dx = ptr_dxy
        ptr_dy = ptr_dxy + D  # Offset by D elements for dy
        # stride_x_t, stride_y_t, stride_dx_t, stride_dy_t are passed from host

    # Get the program ID
    pid = tl.program_id(axis=0)

    # Below CTA_CAP, the launch grid still has at least one CTA per active row tile.
    num_tiles = tl.cdiv(T, BLOCK_T)
    num_tiles_per_cta = tl.cdiv(num_tiles, CTA_CAP)
    tile_start = pid * num_tiles_per_cta
    tile_end = min(tile_start + num_tiles_per_cta, num_tiles)
    for tile_id in tl.range(tile_start, tile_end):
        # Compute row indices for this tile - use int64 to avoid overflow
        t = (tile_id * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
        # Create mask to handle boundary conditions
        mask_t = t < T

        # Loop over the D dimension in blocks of BLOCK_D.
        for offset_d in tl.range(0, D, BLOCK_D):
            # Compute column indices for this tile
            d = offset_d + tl.arange(0, BLOCK_D)
            # Create mask to handle boundary conditions in D dimension
            mask_d = d < D

            # Create 2D mask by broadcasting 1D masks
            mask = mask_t[:, None] & mask_d[None, :]

            # Load input blocks from HBM to SRAM
            # Cast to float32 for computation precision
            x = tl.load(
                ptr_x + t[:, None] * stride_x_t + d[None, :], mask=mask, other=0.0
            ).to(tl.float32)
            y = tl.load(
                ptr_y + t[:, None] * stride_y_t + d[None, :], mask=mask, other=0.0
            ).to(tl.float32)
            dz = tl.load(
                ptr_dz + t[:, None] * stride_dz_t + d[None, :], mask=mask, other=0.0
            ).to(tl.float32)

            if CLAMPED:
                # Grad is zeroed where the PRE-clamp value is on the saturating
                # side of the bound. The explicit backward contract keeps the
                # grad at exactly +/-LIMIT and saturates only outside the bounds.
                g = tl.minimum(x, LIMIT)
                u = tl.minimum(tl.maximum(y, -LIMIT), LIMIT)
                s = sigmoid(ALPHA * g, FAST_MATH=FAST_MATH)
                silu = g * s
                d_silu = s + ALPHA * g * s * (1.0 - s)

                dx = dz * (u + 1.0) * d_silu
                dx = tl.where(x > LIMIT, 0.0, dx)
                tl.store(ptr_dx + t[:, None] * stride_dx_t + d[None, :], dx, mask=mask)

                dy = dz * silu
                dy = tl.where((y > LIMIT) | (y < -LIMIT), 0.0, dy)
                tl.store(ptr_dy + t[:, None] * stride_dy_t + d[None, :], dy, mask=mask)
            else:
                # Compute sigmoid(x) once and reuse
                sigmoid_x = sigmoid(x, FAST_MATH=FAST_MATH)

                # Compute gradient with respect to x using chain rule
                # d/dx[x * sigmoid(x) * y] = y * sigmoid(x) * (1 + x - x * sigmoid(x))
                dx = dz * y * sigmoid_x * (1 + x - x * sigmoid_x)
                tl.store(ptr_dx + t[:, None] * stride_dx_t + d[None, :], dx, mask=mask)

                # Compute gradient with respect to y
                # d/dy[x * sigmoid(x) * y] = x * sigmoid(x)
                dy = dz * x * sigmoid_x
                tl.store(ptr_dy + t[:, None] * stride_dy_t + d[None, :], dy, mask=mask)


def swiglu_bwd(
    dz: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    dxy: torch.Tensor | None = None,
    *,
    activation_buffer: torch.Tensor | None = None,
    num_recv_tokens: torch.Tensor | None = None,
    feature_dim: int | None = None,
    dtype: torch.dtype | None = None,
    fast_math: bool = False,
    clamped: bool = False,
    alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """
    Fused silu and mul backward operations.

    dx = dz * y * sigmoid(x) * (1 + x - x * sigmoid(x))
    dy = dz * x * sigmoid(x)

    Args:
        dz: output gradient of shape (T, D)
            When activation_buffer is provided, dz is a byte offset tensor of shape [1], dtype int64,
            pointing to the upstream gradient location in activation_buffer (dz_offset).
        x: input tensor of shape (T, D) OR concatenated tensor of shape (T, 2*D) if y is None.
           When activation_buffer is provided, x is a byte offset tensor of shape [1], dtype int64,
           pointing to the saved activations location in activation_buffer (xy_offset).
        y: optional input tensor of shape (T, D). If None, x is chunked internally and
           returns concatenated gradient of shape (T, 2*D).
           Not used when activation_buffer is provided.
        dxy: optional output gradient tensor of shape (T, 2*D). If None in default mode,
           allocated with torch.empty. When activation_buffer is provided, dxy is REQUIRED
           and must be a byte offset tensor of shape [1], dtype int64, pointing to the
           gradient output location in activation_buffer (dxy_offset).
        activation_buffer: Optional persistent buffer for reading input and writing output.
            When provided, saved activations are read from x (xy_offset) and gradients are
            written to dxy (dxy_offset). Upstream gradients are read from dz (dz_offset).
        num_recv_tokens: Optional CUDA tensor containing the active row count.
            Shape: [1], dtype: torch.int32. The output keeps its capacity shape;
            rows at and beyond the bound retain caller-provided contents or are
            uninitialized when output storage is allocated internally.
        feature_dim: Optional hidden dimension (D). Required when activation_buffer is provided since
           dimension cannot be inferred from the offset tensor.
        dtype: Data type (torch.float16 or torch.bfloat16). Required when activation_buffer
           is provided since dtype cannot be inferred from offset tensors.

    Returns:
        If y is provided: tuple of (dx, dy)
        If y is None: single tensor dxy of shape (T, 2*D) containing [dx, dy]
        If activation_buffer is used: dxy (dxy_offset)
    """
    # Common setup
    BLOCK_D = 2048
    cta_cap = num_sms_per_device() * 32
    num_warps = 8
    num_stages = 1
    _validate_num_recv_tokens(num_recv_tokens, device=x.device)

    store_dtype = dtype if activation_buffer is not None else x.dtype
    BLOCK_T = 1 if store_dtype == torch.bfloat16 and is_blackwell_gpu(x.device) else 2

    # Track mode for return logic
    chunked = False
    original_shape = None

    if activation_buffer is not None:
        # Activation buffer mode setup
        # In this mode:
        # - dz is a byte offset tensor pointing to upstream gradients (dz_offset)
        # - x is a byte offset tensor pointing to saved activations (xy_offset)
        # - dxy is a byte offset tensor pointing to gradient output (dxy_offset)
        # Check activation_buffer
        assert activation_buffer.is_contiguous(), "activation_buffer must be contiguous"
        # Check dz is an offset tensor (dz_offset)
        assert dz.shape == (1,) and dz.dtype == torch.int64, (
            f"dz must be shape [1] and dtype int64 when activation_buffer is provided, got shape={dz.shape}, dtype={dz.dtype}"
        )
        # Check x is an offset tensor (xy_offset)
        assert x.shape == (1,) and x.dtype == torch.int64, (
            f"x must be shape [1] and dtype int64 when activation_buffer is provided, got shape={x.shape}, dtype={x.dtype}"
        )
        # Check dxy is an offset tensor (dxy_offset)
        assert dxy is not None, (
            "dxy (dxy_offset) required when activation_buffer is provided"
        )
        assert dxy.shape == (1,) and dxy.dtype == torch.int64, (
            f"dxy must be shape [1] and dtype int64 when activation_buffer is provided, got shape={dxy.shape}, dtype={dxy.dtype}"
        )
        assert num_recv_tokens is not None, "num_recv_tokens required"
        # Check feature_dim
        assert feature_dim is not None, (
            "feature_dim is required when activation_buffer is provided"
        )
        # Check dtype
        assert dtype is not None, "dtype is required when activation_buffer is provided"

        # Map tensor inputs to offset tensors
        dz_offset = dz
        xy_offset = x
        dxy_offset = dxy

        # Calculate strides for activation_buffer mode: xy is (T, 2*D), dxy is (T, 2*D)
        stride_xy_t = feature_dim * 2
        stride_dz = feature_dim
        D = feature_dim
        T = 0  # Not used in activation_buffer mode, T comes from device tensor

        # Single kernel launch for activation_buffer mode
        grid = (cta_cap, 1, 1)
        _triton_swiglu_bwd[grid](
            ptr_dz=dz_offset,
            stride_dz_t=stride_dz,
            ptr_x=xy_offset,
            stride_x_t=stride_xy_t,
            ptr_y=xy_offset,
            stride_y_t=stride_xy_t,
            ptr_dx=dxy_offset,
            stride_dx_t=stride_xy_t,
            ptr_dy=dxy_offset,
            stride_dy_t=stride_xy_t,
            activation_buffer_ptr=activation_buffer,
            xy_offset_ptr=xy_offset,
            dxy_offset_ptr=dxy_offset,
            dz_offset_ptr=dz_offset,
            num_recv_tokens_ptr=num_recv_tokens,
            T=T,
            CTA_CAP=cta_cap,
            D=D,
            BLOCK_T=BLOCK_T,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
            num_stages=num_stages,
            DTYPE=_to_tl_dtype(dtype),
            FAST_MATH=fast_math,
            CLAMPED=clamped,
            ALPHA=alpha,
            LIMIT=limit,
        )

        return dxy_offset
    else:
        # Regular/chunked mode setup
        chunked = y is None
        if chunked:
            D_doubled = x.shape[-1]
            assert D_doubled % 2 == 0, (
                f"Last dim must be even for chunking: {D_doubled}"
            )
            D = D_doubled // 2
            original_shape = x.shape
            # Handle dxy output tensor: allocate if not provided
            if dxy is None:
                dxy = torch.empty_like(x, memory_format=torch.contiguous_format)
            else:
                assert dxy.shape == original_shape, (
                    f"dxy shape {dxy.shape} != x shape {original_shape}"
                )
                assert dxy.dtype == x.dtype, (
                    f"dxy dtype {dxy.dtype} != x dtype {x.dtype}"
                )
                assert dxy.is_contiguous(), f"dxy must be contiguous: {dxy.stride()=}"
            x_left = x[..., :D]
            x_right = x[..., D:]
            x = x_left
            y = x_right
            shape_x = x.shape
            shape_y = y.shape
        else:
            shape_x = x.shape
            shape_y = y.shape
            assert shape_x == shape_y, f"{shape_x} != {shape_y}"
            D = shape_x[-1]

        # Validate dtypes
        dtype_x = x.dtype
        dtype_y = y.dtype
        dtype_dz = dz.dtype
        assert dtype_x == dtype_y, f"{dtype_x} != {dtype_y}"
        assert dtype_x == dtype_dz, f"{dtype_x} != {dtype_dz}"

        # Validate output gradient shape
        shape_dz = dz.shape
        assert shape_x == shape_dz, f"{shape_x} != {shape_dz}"

        # Reshape to 2D if needed
        if len(shape_x) != 2:
            x = x.reshape(-1, D)
            y = y.reshape(-1, D)
            dz = dz.reshape(-1, D)
            T = x.shape[0]
        else:
            T = shape_x[0]

        # Validate contiguity
        stride_x = x.stride()
        stride_y = y.stride()
        stride_dz = dz.stride()
        assert stride_x[-1] == 1, f"{stride_x=}"
        assert stride_y[-1] == 1, f"{stride_y=}"
        assert stride_dz[-1] == 1, f"{stride_dz=}"

        # Allocate output tensors
        if not chunked:
            dx = x.new_empty((T, D))
            dy = y.new_empty((T, D))
        else:
            dxy_2d = dxy.view(T, 2 * D)
            dx = dxy_2d[:, :D]
            dy = dxy_2d[:, D:]

        # Size the CUDA-graph-stable launch from capacity. The kernel loads the
        # active row count from device when num_recv_tokens is provided.
        num_d_tiles = (D + BLOCK_D - 1) // BLOCK_D
        num_t_tiles = (T + BLOCK_T - 1) // BLOCK_T
        num_ctas = min(cta_cap, num_t_tiles * num_d_tiles)
        grid = (num_ctas, 1, 1)
        _triton_swiglu_bwd[grid](
            ptr_dz=dz,
            stride_dz_t=stride_dz[0],
            ptr_x=x,
            stride_x_t=stride_x[0],
            ptr_y=y,
            stride_y_t=stride_y[0],
            ptr_dx=dx,
            stride_dx_t=dx.stride()[0],
            ptr_dy=dy,
            stride_dy_t=dy.stride()[0],
            activation_buffer_ptr=None,
            xy_offset_ptr=None,
            dxy_offset_ptr=None,
            dz_offset_ptr=None,
            num_recv_tokens_ptr=num_recv_tokens,
            T=T,
            CTA_CAP=cta_cap,
            D=D,
            BLOCK_T=BLOCK_T,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
            num_stages=num_stages,
            DTYPE=_to_tl_dtype(dtype_x),
            FAST_MATH=fast_math,
            CLAMPED=clamped,
            ALPHA=alpha,
            LIMIT=limit,
        )

        # Return based on mode
        if chunked:
            return dxy
        else:
            if len(shape_x) != 2:
                dx = dx.reshape(shape_x)
                dy = dy.reshape(shape_y)
            return dx, dy
