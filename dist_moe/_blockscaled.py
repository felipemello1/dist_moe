# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Asynchronous block-scaled CuTe distributed MoE orchestration."""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.profiler import record_function
from torch.utils._python_dispatch import _get_current_dispatch_mode

from . import _blockscaled_ops as dist_bs_gemm
from ._activation_buffer import (
    ActivationBuffer,
    BLOCKSCALED_DISPATCH_ROW_ALIGNMENT_BYTES,
    BlockscaledForwardPlan,
    BlockscaledStorageConfig,
    FORWARD_ACTIVATION_OFFSET_COUNT,
    ForwardPlan,
    get_backward_plan,
    get_forward_plan,
    ModelConfig,
    validate_buffer_capacity,
)
from ._activation_buffer_offsets import (
    activation_buffer_placeholder as _activation_buffer_placeholder,
    validate_conditional_execution,
)
from ._blockscaled_ops import (
    dist_blockscaled_grouped_gemm_dgrad_dispatch,
    dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine,
    dist_blockscaled_grouped_gemm_dgrad_wgrad_dispatch,
    dist_blockscaled_grouped_gemm_dgrad_wgrad_swiglu_bwd_combine,
    dist_blockscaled_grouped_gemm_fprop_dispatch,
    dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine,
    dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_dispatch_combine,
)
from ._blockscaled_weight import _interleave_w13_rows, BlockscaledWeightSpec
from ._buffers import (
    _CommunicationBuffers,
    _initialize_fake_peer_scatter_output,
    _routing_ids_view,
    _routing_token_count_view,
    SymmetricMemoryBuffer,
)
from ._context import _get_context, _reshape_weights
from ._execution import (
    _KernelAutogradContext,
    _postprocess_wgrad,
    _resolve_parameter_grad_destinations,
    _resolve_wgrad_destinations,
    _resolve_wgrad_output_dtype,
    _weak_parameter_ref,
    _wgrad_accumulation_destinations,
    _WgradDestination,
    WgradPostprocessFn,
)
from ._postprocess import (
    _ExpertsOutputPostprocess,
    _registered_rmsnorm_args,
    _resolve_experts_output_postprocess_fn,
    _ResolvedExpertsPostprocess,
    _rmsnorm_from_registered_args,
)
from ._quantization import quantize_for_format
from ._routing_metadata import (
    _routing_ptrs_size,
    dist_dispatch_routing,
)
from ._triton_ops import (
    conditional_copy_activations,
    copy_activation_to_dispatch,
    copy_dispatch_to_activation,
    copy_routing,
    copy_routing_and_dispatch,
    reduce_from_topk,
    symmetric_memory_barrier,
)
from .formats import (
    _kernel_block_scaled_format,
    _MXFP8_DIM_MULTIPLE,
    _NVFP4_DIM_MULTIPLE,
    _NVFP4_HIDDEN_DIM_MAX,
    _NVFP4_INTERMEDIATE_DIM_MAX,
    block_scaled_format_constants,
    BlockScaledFormat,
    FP4_FORMATS,
    ScaleFactorLayout,
)
from .kernels.blockscaled_grouped_gemm import (
    blockscaled_grouped_gemm_wgrad,
    MXFP8_E4M3,
    NVFP4,
)
from .kernels.blockscaled_quantize import (
    _quantize_nvfp4_per_token,
    quantize_block_scaled,
)
from .kernels.config import BLOCKSCALED_DISPATCH_DIM_ALIGNMENT
from .kernels.inference_config import _inference_routing_multiple

if TYPE_CHECKING:
    from ._execution import (
        ExecutionOptions,
        PreparedWeight,
        WgradDestinationFn,
    )
    from .api import BlockScaledConfig, Context

_PREPARED_WEIGHT_OPERAND_COUNT = 6
_PREPARED_WEIGHT_PAIR_COUNT = 2 * _PREPARED_WEIGHT_OPERAND_COUNT
_SAVED_WEIGHT_STATE_COUNT = 5
_SAVED_WEIGHT_PAIR_COUNT = 2 * _SAVED_WEIGHT_STATE_COUNT
_W13_SAVED_WEIGHT_START = 2
_W2_SAVED_WEIGHT_START = _W13_SAVED_WEIGHT_START + _SAVED_WEIGHT_STATE_COUNT
_ASYNC_FORWARD_SAVED_TENSOR_COUNT = 34
_FORWARD_ROUTING_STATE = slice(14, 20)
_FORWARD_NEED_RECOMPUTE = 20
_FORWARD_OFFSET_INDICES = (*range(21, 30), 33)
_FORWARD_ACTIVATION_OFFSETS = 30
_FORWARD_RECOMPUTE_CONDITION = 31
_FORWARD_POSTPROCESS_CONTEXT = 32
_BACKWARD_STATE_TENSOR_COUNT = 21
_BACKWARD_WEIGHT_STATE = slice(0, _SAVED_WEIGHT_PAIR_COUNT)
_BACKWARD_ROUTING_STATE = slice(10, 16)
_BACKWARD_PLAN_STATE = slice(16, 21)
_BACKWARD_NON_WEIGHT_STATE = slice(10, None)


@torch.library.custom_op("dist_moe::prepare_nvfp4_weight", mutates_args=())
def _prepare_nvfp4_weight_operands(
    weight_ENK: torch.Tensor,
    fuse_swiglu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize grouped NVFP4 weights with the bundled Triton quantizer.

    Args:
        weight_ENK: Native grouped weight with expert, output, and input axes.
        fuse_swiglu: Interleave W13 qdata and natural scale rows for fused
            inference after quantization, preserving canonical scale groups.

    Returns:
        Packed qdata ``[E, N, K / 2]``, blocked scales, per-expert global
        scales ``[E]``, and their reciprocals ``[E]``.
    """
    if weight_ENK.ndim != 3 or weight_ENK.shape[-1] % 2:
        raise ValueError("NVFP4 preparation requires grouped weights with even K")
    from .kernels.triton_quantization.blockscaled_quantize import (
        quantize_block_scaled,
    )
    from .kernels.triton_quantization.formats import (
        BlockScaledFormatId,
        ScaleFactorLayoutId,
    )
    from .kernels.triton_quantization.global_scale import (
        expand_global_scale_for_rows,
        GlobalScaleGranularity,
        nvfp4_weight_global_scale,
    )

    global_scale_E = nvfp4_weight_global_scale(weight_ENK).detach()
    global_scale_EN = expand_global_scale_for_rows(
        global_scale_E,
        weight_ENK.shape,
        device=weight_ENK.device,
        granularity=GlobalScaleGranularity.PER_EXPERT,
    )
    qdata_ENQ, scale_RB = quantize_block_scaled(
        weight_ENK,
        format_id=BlockScaledFormatId.NVFP4.value,
        layout_id=(
            ScaleFactorLayoutId.NATURAL.value
            if fuse_swiglu
            else ScaleFactorLayoutId.CUBLAS_BLOCKED.value
        ),
        is_a=False,
        global_scale=global_scale_EN,
    )
    if fuse_swiglu:
        from .kernels.scale_factor_layout import natural_to_cublas_blocked

        qdata_ENQ = _interleave_w13_rows(qdata_ENQ)
        scale_ENB = scale_RB.view(*weight_ENK.shape[:2], -1)
        scale_ENB = _interleave_w13_rows(scale_ENB)
        scale_RB = natural_to_cublas_blocked(scale_ENB, is_a=False)
    return qdata_ENQ, scale_RB, global_scale_E, torch.reciprocal(global_scale_E)


@_prepare_nvfp4_weight_operands.register_fake
def _prepare_nvfp4_weight_fake(
    weight_ENK: torch.Tensor,
    fuse_swiglu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe inference operands without running quantization on fake data.

    Args:
        weight_ENK: Grouped BF16 weight with shape ``[E, N, K]``.
        fuse_swiglu: Whether the prepared W13 rows are interleaved.

    Returns:
        Packed qdata, padded blocked scales and two per-expert FP32 scales.
    """
    if weight_ENK.ndim != 3 or weight_ENK.shape[-1] % 2:
        raise ValueError("NVFP4 preparation requires grouped weights with even K")
    experts, rows, cols = weight_ENK.shape
    return (
        weight_ENK.new_empty((experts, rows, cols // 2), dtype=torch.float4_e2m1fn_x2),
        weight_ENK.new_empty(
            (
                experts * ((rows + 127) // 128) * 128,
                (((cols + 15) // 16 + 3) // 4) * 4,
            ),
            dtype=torch.float8_e4m3fn,
        ),
        weight_ENK.new_empty((experts,), dtype=torch.float32),
        weight_ENK.new_empty((experts,), dtype=torch.float32),
    )


def _kernel_format(format: BlockScaledFormat) -> Any:
    """Map an internal precision enum to its CuTe specialization.

    Args:
        format: Internal block-scaled format.

    Returns:
        CuTe kernel format specialization.
    """
    return {
        BlockScaledFormat.MXFP8_E4M3: MXFP8_E4M3,
        BlockScaledFormat.NVFP4: NVFP4,
    }[format]


@dataclass(frozen=True)
class _FusedBlockscaledFormatPolicy:
    """Shape, preparation, and inference constraints for one format."""

    inference_only: bool = False
    requires_prequantized_weights: bool = False
    requires_weight_global_scale: bool = False
    row_global_scale_element_size: int = 0
    input_dim_multiple: int = 128
    input_dim_max: int | None = None
    swiglu_dim_multiple: int = 128
    swiglu_dim_max: int | None = None

    def supports_swiglu_dim(self, dim: int) -> bool:
        """Return whether a dimension satisfies the SwiGLU constraints.

        Args:
            dim: Intermediate dimension.

        Returns:
            Whether the dimension is supported.
        """
        return dim % self.swiglu_dim_multiple == 0 and (
            self.swiglu_dim_max is None or dim <= self.swiglu_dim_max
        )

    def supports_input_dim(self, dim: int) -> bool:
        """Return whether a dimension satisfies the input constraints.

        Args:
            dim: Hidden dimension.

        Returns:
            Whether the dimension is supported.
        """
        return dim % self.input_dim_multiple == 0 and (
            self.input_dim_max is None or dim <= self.input_dim_max
        )

    def supports_dims(self, hidden_dim: int, intermediate_dim: int) -> bool:
        """Return whether both model dimensions satisfy the format policy.

        Args:
            hidden_dim: Model hidden dimension.
            intermediate_dim: Expert intermediate dimension.

        Returns:
            Whether both dimensions are supported.
        """
        return self.supports_input_dim(hidden_dim) and self.supports_swiglu_dim(
            intermediate_dim
        )


_DEFAULT_FUSED_BLOCKSCALED_FORMAT_POLICY = _FusedBlockscaledFormatPolicy(
    input_dim_multiple=_MXFP8_DIM_MULTIPLE,
    swiglu_dim_multiple=_MXFP8_DIM_MULTIPLE,
)


_FUSED_BLOCKSCALED_FORMAT_POLICIES: Mapping[
    BlockScaledFormat, _FusedBlockscaledFormatPolicy
] = MappingProxyType(
    {
        BlockScaledFormat.MXFP8_E4M3: _DEFAULT_FUSED_BLOCKSCALED_FORMAT_POLICY,
        BlockScaledFormat.NVFP4: _FusedBlockscaledFormatPolicy(
            inference_only=True,
            requires_prequantized_weights=True,
            requires_weight_global_scale=True,
            row_global_scale_element_size=4,
            input_dim_multiple=_NVFP4_DIM_MULTIPLE,
            input_dim_max=_NVFP4_HIDDEN_DIM_MAX,
            swiglu_dim_multiple=_NVFP4_DIM_MULTIPLE,
            swiglu_dim_max=_NVFP4_INTERMEDIATE_DIM_MAX,
        ),
    }
)

_EP_BARRIER_CHANNEL = 0


def _allocate_wgrad_output(
    *,
    weight_compute: torch.Tensor,
    output_dtype: torch.dtype | None,
) -> torch.Tensor | None:
    """Allocate an optional explicitly typed WGRAD destination.

    Args:
        weight_compute: Grouped compute view.
        output_dtype: Optional standalone gradient dtype.

    Returns:
        WGRAD destination, or ``None`` when the kernel owns allocation.
    """
    if output_dtype is not None:
        return torch.empty(
            weight_compute.shape,
            dtype=output_dtype,
            device=weight_compute.device,
        )
    return None


def _detach_blockscaled_weight_spec(
    spec: BlockscaledWeightSpec,
) -> BlockscaledWeightSpec:
    """Detach every tensor in a prepared-weight specification.

    Args:
        spec: Normalized native and quantized weight operands.

    Returns:
        A specification containing detached tensor views.
    """

    def detach(tensor: torch.Tensor | None) -> torch.Tensor | None:
        """Detach an optional tensor.

        Args:
            tensor: Optional tensor operand.

        Returns:
            Detached tensor, or ``None``.
        """
        return tensor.detach() if tensor is not None else None

    return replace(
        spec,
        w_native=spec.w_native.detach(),
        w_q_fprop=detach(spec.w_q_fprop),
        w_scale_fprop=detach(spec.w_scale_fprop),
        w_q_dgrad=detach(spec.w_q_dgrad),
        w_scale_dgrad=detach(spec.w_scale_dgrad),
        w_global_scale=detach(spec.w_global_scale),
        w_global_scale_inv=detach(spec.w_global_scale_inv),
    )


def _ep_barrier(
    symm_mem_buf: SymmetricMemoryBuffer,
    conditional_execution: torch.Tensor | None = None,
) -> None:
    """Synchronize EP ranks before consuming symmetric-memory payloads.

    Args:
        symm_mem_buf: Symmetric memory buffer handle.
        conditional_execution: Optional device tensor with one bool or int32 element.
            If provided and zero, the barrier is skipped.
            All ranks must have the same condition to avoid deadlocks.
    """
    if conditional_execution is not None:
        validate_conditional_execution(
            conditional_execution,
            device=symm_mem_buf.signal_pad_ptrs_tensor.device,
        )
    symmetric_memory_barrier(
        symm_mem_buf,
        channel=_EP_BARRIER_CHANNEL,
        conditional_execution=conditional_execution,
    )


def _blockscaled_dispatch_stride_bytes(
    x: torch.Tensor,
    cfg: _BlockscaledConfig,
) -> int:
    """Return the packed symmetric-memory row stride for inference dispatch.

    Args:
        x: Native activation rows.
        cfg: Active block-scaled format.

    Returns:
        Aligned row stride in bytes.
    """
    q_dtype, scale_dtype, sf_vec_size = block_scaled_format_constants(cfg.format)
    if q_dtype.itemsize != 1 or scale_dtype.itemsize != 1:
        raise NotImplementedError("blockscaled inference dispatch requires byte dtypes")
    cols = x.shape[1]
    dim_alignment = BLOCKSCALED_DISPATCH_DIM_ALIGNMENT
    if cols % dim_alignment != 0:
        raise ValueError(
            "blockscaled inference dispatch requires the feature dimension "
            f"to be a multiple of {dim_alignment}; got {cols}"
        )
    scale_cols = cols // sf_vec_size
    q_cols = cols // 2 if cfg.format in FP4_FORMATS else cols
    token_scale_bytes = cfg.policy.row_global_scale_element_size
    unaligned_row_bytes = q_cols + scale_cols + token_scale_bytes
    row_alignment = BLOCKSCALED_DISPATCH_ROW_ALIGNMENT_BYTES
    return (unaligned_row_bytes + row_alignment - 1) // row_alignment * row_alignment


def _local_blockscaled_dispatch_view(
    x: torch.Tensor,
    symm_mem_buf: SymmetricMemoryBuffer,
    cfg: _BlockscaledConfig,
) -> torch.Tensor:
    """Return the rank-local packed inference dispatch view.

    Args:
        x: Native activation rows defining the shape.
        symm_mem_buf: Dispatch symmetric-memory handle.
        cfg: Active block-scaled format.

    Returns:
        Rank-local byte view.
    """
    return symm_mem_buf.hdl.get_buffer(
        symm_mem_buf.hdl.rank,
        (x.shape[0], _blockscaled_dispatch_stride_bytes(x, cfg)),
        torch.uint8,
    )


def _stage_blockscaled_dispatch(
    x: torch.Tensor,
    symm_mem_buf: SymmetricMemoryBuffer,
    cfg: _BlockscaledConfig,
) -> int:
    """Quantize inference inputs directly into symmetric memory.

    Args:
        x: Native activation rows.
        symm_mem_buf: Dispatch symmetric-memory handle.
        cfg: Active block-scaled format.

    Returns:
        Packed row stride in bytes.
    """
    q_dtype, scale_dtype, sf_vec_size = block_scaled_format_constants(cfg.format)
    rows, cols = x.shape
    scale_cols = cols // sf_vec_size
    q_cols = cols // 2 if cfg.format in FP4_FORMATS else cols
    row_bytes = _blockscaled_dispatch_stride_bytes(x, cfg)
    packed = _local_blockscaled_dispatch_view(x, symm_mem_buf, cfg)
    q_out = packed[:, :q_cols].view(q_dtype)
    scale_out = packed[:, q_cols : q_cols + scale_cols].view(scale_dtype)
    if cfg.format == BlockScaledFormat.NVFP4:
        token_scale_start = q_cols + scale_cols
        token_scale_out = packed[
            :,
            token_scale_start : token_scale_start
            + cfg.policy.row_global_scale_element_size,
        ].view(torch.float32)
        _quantize_nvfp4_per_token(
            x,
            layout=ScaleFactorLayout.NATURAL,
            q_out=q_out,
            scale_out=scale_out,
            token_scale_inv_out=token_scale_out[:, 0],
        )
    else:
        quantize_block_scaled(
            x,
            format=cfg.format,
            layout=ScaleFactorLayout.NATURAL,
            q_out=q_out,
            scale_out=scale_out,
            allow_partial_rows=True,
        )
    return row_bytes


def _local_symm_mem_buffer_view(
    symm_mem_buf: SymmetricMemoryBuffer,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a typed rank-local symmetric-memory view.

    Args:
        symm_mem_buf: Symmetric-memory handle.
        shape: Requested tensor shape.
        dtype: Requested tensor dtype.

    Returns:
        Rank-local tensor view.
    """
    return symm_mem_buf.hdl.get_buffer(symm_mem_buf.hdl.rank, shape, dtype)


def _stage_async_inference_dispatch(
    x: torch.Tensor,
    symm_mem_buf: SymmetricMemoryBuffer,
    cfg: _BlockscaledConfig | None,
) -> int | None:
    """Publish native or packed inputs for async inference.

    Args:
        x: Native activation rows.
        symm_mem_buf: Dispatch symmetric-memory handle.
        cfg: Optional block-scaled format.

    Returns:
        Packed byte stride for block-scaled input, otherwise ``None``.
    """
    if cfg is not None:
        return _stage_blockscaled_dispatch(x, symm_mem_buf, cfg)
    _local_symm_mem_buffer_view(symm_mem_buf, tuple(x.shape), x.dtype).copy_(x)
    return None


_QuantPair = (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
)


@dataclass(frozen=True)
class _BlockscaledConfig:
    """Resolved block-scaled format and expert-pipeline policy."""

    format: BlockScaledFormat
    fast_math: bool = False
    mega: bool = False
    interleaved_fc13: bool = False
    routing_capacity: int | None = None
    static_inference_storage: bool = False

    @property
    def policy(self) -> _FusedBlockscaledFormatPolicy:
        """Return the static constraints for this format."""
        return _FUSED_BLOCKSCALED_FORMAT_POLICIES.get(
            self.format,
            _DEFAULT_FUSED_BLOCKSCALED_FORMAT_POLICY,
        )


@dataclass(frozen=True)
class _BlockscaledWeight:
    """Native weight plus CuTe FPROP and DGRAD representations."""

    w_native: torch.Tensor
    w_q_fprop: torch.Tensor
    w_scale_fprop: torch.Tensor
    w_q_dgrad: torch.Tensor
    w_scale_dgrad: torch.Tensor
    w_global_scale_inv: torch.Tensor | None = None


@dataclass(frozen=True)
class _ComputeDispatch:
    """Resolved block-scaled precision and kernel tuning controls."""

    config: dict | None
    blockscaled: _BlockscaledConfig

    def uses_blockscaled_dispatch(self, *, inference_mode: bool) -> bool:
        """Return whether routing publishes a packed block-scaled input.

        Args:
            inference_mode: Whether the call omits backward state.

        Returns:
            Whether dispatch uses packed row-quantized storage.
        """
        return inference_mode


def _validate_compute_config(
    config: dict | None,
) -> None:
    """Validate a block-scaled tuning override.

    Args:
        config: Optional explicit kernel configuration.

    Raises:
        ValueError: If the configuration type does not match the backend.
    """
    if config is None:
        return
    if not isinstance(config, dict):
        raise ValueError(
            "blockscaled DistMoE config must be an explicit config dict; "
            "named config strings are not supported"
        )


def _num_local_experts(
    num_experts: int,
    group: dist.ProcessGroup,
) -> int:
    """Return the rank-local expert count.

    Args:
        num_experts: Global expert count.
        group: Expert-parallel process group.

    Returns:
        Experts owned by one rank.

    Raises:
        ValueError: If experts cannot be evenly sharded.
    """
    world_size = dist.get_world_size(group)
    if num_experts < world_size or num_experts % world_size != 0:
        raise ValueError(
            "expert-parallel DistMoE requires num_experts to be divisible by "
            "world_size with at least one expert per rank; "
            f"got num_experts={num_experts}, world_size={world_size}"
        )
    return num_experts // world_size


@dataclass(frozen=True)
class _WeightHandle:
    """Native weight and its required block-scaled kernel operands."""

    w_native: torch.Tensor
    blockscaled: _BlockscaledWeight


def _blockscaled_storage_config(
    cfg: _BlockscaledConfig,
) -> BlockscaledStorageConfig:
    """Translate a format into activation-planner storage geometry.

    Args:
        cfg: Active block-scaled format.

    Returns:
        Storage sizes and packing ratios used by the planner.
    """
    operand_dtype, scale_dtype, sf_vec_size = block_scaled_format_constants(cfg.format)
    return BlockscaledStorageConfig(
        operand_element_size=operand_dtype.itemsize,
        scale_element_size=scale_dtype.itemsize,
        sf_vec_size=sf_vec_size,
        operand_values_per_storage_element=2 if cfg.format in FP4_FORMATS else 1,
        row_global_scale_element_size=cfg.policy.row_global_scale_element_size,
    )


def _async_blockscaled_col_quant_placeholder(
    activation_buffer: torch.Tensor,
    *,
    rows: int,
    dim: int,
    cfg: _BlockscaledConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create column-quantized placeholder views into the activation buffer.

    Args:
        activation_buffer: Byte-addressed activation buffer.
        rows: Logical received-token capacity.
        dim: Logical feature dimension.
        cfg: Active block-scaled format.

    Returns:
        Quantized data and scale placeholders.
    """
    operand_dtype, scale_dtype, sf_vec_size = block_scaled_format_constants(cfg.format)
    q_storage = _activation_buffer_placeholder(
        activation_buffer,
        (rows, dim),
        operand_dtype,
    )
    scale = _activation_buffer_placeholder(
        activation_buffer,
        (dim, rows // sf_vec_size),
        scale_dtype,
    )
    return q_storage.t(), scale.view(-1)


def _async_blockscaled_wgrad(
    *,
    activation_offsets: torch.Tensor,
    dy_dim: int,
    x_dim: int,
    rows: int,
    split_sizes: torch.Tensor,
    activation_buffer: torch.Tensor,
    cfg: _BlockscaledConfig,
    config: dict | None,
    dw: torch.Tensor | None,
    output_accum: bool,
    out_dtype: torch.dtype | None,
) -> torch.Tensor:
    """Run grouped WGRAD from column-quantized activation-buffer operands.

    Args:
        activation_offsets: Device-resident offsets for both operands.
        dy_dim: Gradient feature dimension.
        x_dim: Input feature dimension.
        rows: Logical received-token capacity.
        split_sizes: Rows assigned to each local expert.
        activation_buffer: Byte-addressed activation buffer.
        cfg: Active block-scaled format.
        config: Optional CuTe tuning override.
        dw: Optional direct accumulation destination.
        output_accum: Whether ``dw`` already contains data.
        out_dtype: Optional standalone WGRAD dtype.

    Returns:
        Grouped expert-weight gradient.
    """
    dy_q, dy_scale = _async_blockscaled_col_quant_placeholder(
        activation_buffer,
        rows=rows,
        dim=dy_dim,
        cfg=cfg,
    )
    x_q, x_scale = _async_blockscaled_col_quant_placeholder(
        activation_buffer,
        rows=rows,
        dim=x_dim,
        cfg=cfg,
    )
    return blockscaled_grouped_gemm_wgrad(
        dy=dy_q,
        x=x_q,
        sfa=dy_scale,
        sfb=x_scale,
        split_sizes=split_sizes,
        format=_kernel_format(cfg.format),
        out_dtype=out_dtype or (torch.float32 if dw is not None else torch.bfloat16),
        dw=dw,
        output_accum=output_accum,
        config=config,
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
    )


def _async_resolve_compute(
    *,
    config: dict | None,
    blockscaled_cfg: _BlockscaledConfig,
    inference_mode: bool,
    save_for_backward: bool,
) -> tuple[_ComputeDispatch, bool]:
    """Resolve asynchronous compute and WGRAD-save policy.

    Args:
        config: Optional kernel tuning override.
        blockscaled_cfg: Active block-scaled format.
        inference_mode: Whether inference-specialized execution is selected.
        save_for_backward: Whether this invocation has a backward consumer.
    Returns:
        Compute dispatch and whether to retain WGRAD operands.
    """
    compute = _ComputeDispatch(
        config=config,
        blockscaled=blockscaled_cfg,
    )
    if inference_mode and save_for_backward:
        raise ValueError("inference execution cannot save backward state")
    return compute, save_for_backward


def _prepare_blockscaled_weight_impl(
    w: torch.Tensor,
    cfg: _BlockscaledConfig,
    *,
    use_row_1d_quantization: bool = False,
    fuse_swiglu: bool = False,
) -> _BlockscaledWeight:
    """Quantize one native MXFP8 weight for FPROP and DGRAD.

    Args:
        w: Grouped native expert weight.
        cfg: Active block-scaled format.
        use_row_1d_quantization: Whether only the inference orientation is needed.
        fuse_swiglu: Interleave W13 quantized rows and natural scales for the
            fused inference kernel; never transpose the BF16 source.

    Returns:
        Prepared kernel operands.

    Raises:
        ValueError: If the format requires caller-owned preparation.
    """
    if cfg.policy.requires_prequantized_weights:
        raise ValueError(
            f"fused {cfg.format.value.upper()} requires pre-quantized weights "
            "with a per-expert global scale"
        )
    _, _, sf_vec_size = block_scaled_format_constants(cfg.format)
    if use_row_1d_quantization:
        if fuse_swiglu:
            from .formats import resolve_half_range_scale
            from .kernels.scale_factor_layout import natural_to_cublas_blocked

            w_q_rowmajor, w_scale_rowmajor = quantize_block_scaled(
                w,
                format=cfg.format,
                layout=ScaleFactorLayout.NATURAL,
                half_range_scale=resolve_half_range_scale(cfg.format, None),
            )
            w_q_rowmajor = _interleave_w13_rows(w_q_rowmajor)
            w_scale_ENB = w_scale_rowmajor.view(*w.shape[:2], -1)
            w_scale_ENB = _interleave_w13_rows(w_scale_ENB)
            w_scale_rowmajor = natural_to_cublas_blocked(w_scale_ENB, is_a=False)
        else:
            w_q_rowmajor, w_scale_rowmajor = quantize_for_format(
                w,
                cfg.format,
                block_size=((1, sf_vec_size),),
                layout=dist_bs_gemm.DEFAULT_LAYOUT,
            )
        assert isinstance(w_q_rowmajor, torch.Tensor)
        assert isinstance(w_scale_rowmajor, torch.Tensor)
        return _BlockscaledWeight(
            w_native=w,
            w_q_fprop=w_q_rowmajor,
            w_scale_fprop=w_scale_rowmajor,
            w_q_dgrad=w_q_rowmajor,
            w_scale_dgrad=w_scale_rowmajor,
            w_global_scale_inv=None,
        )
    (
        (w_q_colmajor, w_scale_colmajor),
        (w_q_rowmajor, w_scale_rowmajor),
    ) = quantize_for_format(
        w,
        cfg.format,
        block_size=((sf_vec_size, sf_vec_size),),
        layout=dist_bs_gemm.DEFAULT_LAYOUT,
    )
    assert isinstance(w_q_rowmajor, torch.Tensor)
    assert isinstance(w_scale_rowmajor, torch.Tensor)
    assert isinstance(w_scale_colmajor, torch.Tensor)
    if cfg.format in FP4_FORMATS:
        assert isinstance(w_q_colmajor, torch.Tensor)
        w_q_dgrad = w_q_colmajor.transpose(1, 2)
    else:
        w_q_dgrad = w_q_rowmajor
    return _BlockscaledWeight(
        w_native=w,
        w_q_fprop=w_q_rowmajor,
        w_scale_fprop=w_scale_rowmajor,
        w_q_dgrad=w_q_dgrad,
        w_scale_dgrad=w_scale_colmajor,
        w_global_scale_inv=None,
    )


def _validate_weight_global_scale(
    scale: torch.Tensor,
    name: str,
    weight: torch.Tensor,
    cfg: _BlockscaledConfig,
) -> None:
    """Validate one per-expert NVFP4 global-scale tensor.

    Args:
        scale: Scale or reciprocal scale to validate.
        name: Argument name used in the error message.
        weight: Native grouped weight that determines expert count and device.
        cfg: Active block-scaled format.

    Raises:
        ValueError: If the scale has the wrong shape, dtype, device, or layout.
    """
    if (
        scale.shape != (weight.shape[0],)
        or scale.dtype != torch.float32
        or scale.device != weight.device
        or not scale.is_contiguous()
    ):
        raise ValueError(
            f"{cfg.format.value.upper()} {name} must be contiguous float32 "
            f"with shape ({weight.shape[0]},) on {weight.device}; got "
            f"shape={tuple(scale.shape)}, dtype={scale.dtype}, "
            f"device={scale.device}"
        )


def _resolve_weight_global_scale_inv(
    w: torch.Tensor,
    prequantized: BlockscaledWeightSpec,
    cfg: _BlockscaledConfig,
) -> torch.Tensor | None:
    """Validate and resolve the per-expert NVFP4 inverse global scale.

    Args:
        w: Native grouped weight.
        prequantized: Prepared operand specification.
        cfg: Active block-scaled format.

    Returns:
        Per-expert inverse global scale, or ``None``.

    Raises:
        ValueError: If scale presence, shape, dtype, or placement is invalid.
    """
    global_scale = prequantized.w_global_scale
    global_scale_inv = prequantized.w_global_scale_inv
    if not cfg.policy.requires_weight_global_scale:
        if global_scale is not None or global_scale_inv is not None:
            raise ValueError(
                "w_global_scale is not supported for "
                f"{cfg.format.value.upper()} weights"
            )
        return None
    if global_scale is None:
        raise ValueError(
            f"pre-quantized {cfg.format.value.upper()} weights must provide "
            "w_global_scale"
        )
    _validate_weight_global_scale(global_scale, "w_global_scale", w, cfg)
    if global_scale_inv is not None:
        _validate_weight_global_scale(
            global_scale_inv,
            "w_global_scale_inv",
            w,
            cfg,
        )
        return global_scale_inv
    return torch.reciprocal(global_scale)


def _prepare_weight_handle(
    w: torch.Tensor,
    compute: _ComputeDispatch,
    *,
    inference_mode: bool,
    prequantized: BlockscaledWeightSpec | None = None,
) -> _WeightHandle:
    """Resolve native or caller-prepared weight operands.

    Args:
        w: Native grouped weight.
        compute: Selected compute path.
        inference_mode: Whether DGRAD operands may be omitted.
        prequantized: Optional caller-prepared operands.

    Returns:
        Weight handle consumed by forward and backward kernels.
    """
    has_prequantized = prequantized is not None and prequantized.is_prequantized
    if has_prequantized:
        assert prequantized is not None
        if prequantized.w_q_fprop is None or prequantized.w_scale_fprop is None:
            raise ValueError(
                "pre-quantized weight tuple must provide fprop quant slots"
            )
        w_q_dgrad = prequantized.w_q_dgrad
        w_scale_dgrad = prequantized.w_scale_dgrad
        if w_q_dgrad is None or w_scale_dgrad is None:
            if not inference_mode:
                raise ValueError(
                    "pre-quantized training weight tuple must provide dgrad quant slots"
                )
            # Inference reads only the FPROP orientation.
            w_q_dgrad = prequantized.w_q_fprop
            w_scale_dgrad = prequantized.w_scale_fprop
        w_global_scale_inv = _resolve_weight_global_scale_inv(
            w,
            prequantized,
            compute.blockscaled,
        )
        return _WeightHandle(
            w_native=w,
            blockscaled=_BlockscaledWeight(
                w_native=w,
                w_q_fprop=prequantized.w_q_fprop,
                w_scale_fprop=prequantized.w_scale_fprop,
                w_q_dgrad=w_q_dgrad,
                w_scale_dgrad=w_scale_dgrad,
                w_global_scale_inv=w_global_scale_inv,
            ),
        )
    return _WeightHandle(
        w_native=w,
        blockscaled=_prepare_blockscaled_weight_impl(
            w,
            compute.blockscaled,
            use_row_1d_quantization=inference_mode,
        ),
    )


def _weight_handle_tensors(
    handle: _WeightHandle,
    *,
    omit_dynamic_quants: bool = False,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Flatten saved operands, omitting dynamically reproducible quants.

    Args:
        handle: Prepared native and optional block-scaled weight operands.
        omit_dynamic_quants: Whether backward will regenerate block-scaled
            operands from the native weight.

    Returns:
        Five optional tensors forming the serialized block-scaled handle.
    """
    if omit_dynamic_quants:
        return (None, None, None, None, None)
    return (
        handle.blockscaled.w_q_fprop,
        handle.blockscaled.w_scale_fprop,
        handle.blockscaled.w_q_dgrad,
        handle.blockscaled.w_scale_dgrad,
        handle.blockscaled.w_global_scale_inv,
    )


def _restore_weight_handle(
    w_native: torch.Tensor,
    tensors: tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ],
    *,
    requantize_with: _ComputeDispatch | None = None,
) -> _WeightHandle:
    """Reconstruct a weight handle from saved autograd tensors.

    Args:
        w_native: Native grouped weight.
        tensors: Saved quantized operands and global scale.
        requantize_with: Optional compute policy used to regenerate omitted
            dynamic weight quants.

    Returns:
        Restored weight handle.
    """
    (
        w_q_fprop,
        w_scale_fprop,
        w_q_dgrad,
        w_scale_dgrad,
        w_global_scale_inv,
    ) = tensors
    if requantize_with is not None:
        return _prepare_weight_handle(
            w_native,
            requantize_with,
            inference_mode=False,
        )
    if w_q_fprop is None:
        assert w_scale_fprop is None
        assert w_q_dgrad is None
        assert w_scale_dgrad is None
        assert w_global_scale_inv is None
        raise RuntimeError(
            "saved block-scaled weight state is absent without a recompute policy"
        )
    assert w_scale_fprop is not None
    assert w_q_dgrad is not None
    assert w_scale_dgrad is not None
    return _WeightHandle(
        w_native=w_native,
        blockscaled=_BlockscaledWeight(
            w_native=w_native,
            w_q_fprop=w_q_fprop,
            w_scale_fprop=w_scale_fprop,
            w_q_dgrad=w_q_dgrad,
            w_scale_dgrad=w_scale_dgrad,
            w_global_scale_inv=w_global_scale_inv,
        ),
    )


def _recompute_dynamic_mega_weight_quants(
    compute: _ComputeDispatch,
    *,
    inference_mode: bool,
    weights_preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None,
) -> bool:
    """Return whether dynamic MXFP8 Mega quants should be rebuilt in backward.

    Caller-prepared weights have no preprocessing callback and remain saved.
    Only temporary quants created from callback-produced BF16 weights are
    reproducible and released after the forward.

    Args:
        compute: Resolved forward compute policy.
        inference_mode: Whether the call is forward-only inference.
        weights_preprocess_fn: Optional callback producing the compute weight.

    Returns:
        ``True`` when forward quants should be omitted from autograd state.
    """
    return (
        not inference_mode
        and weights_preprocess_fn is not None
        and compute.blockscaled.mega
        and compute.blockscaled.format is BlockScaledFormat.MXFP8_E4M3
    )


def _routing_m_multiple(
    compute: _ComputeDispatch,
    *,
    inference_mode: bool = False,
    estimated_rows: int | None = None,
    num_local_experts: int | None = None,
    fc13_output_dim: int | None = None,
    fc2_output_dim: int | None = None,
    num_sms: int | None = None,
    enable_host_tensormaps: bool = False,
) -> int | None:
    """Return the row multiple required by the selected grouped GEMM.

    Args:
        compute: Resolved kernel backend and tuning configuration.
        inference_mode: Whether the call omits backward state.
        estimated_rows: Estimated routed rows for native inference.
        num_local_experts: Experts owned by this rank.
        fc13_output_dim: Fused gate/up output width.
        fc2_output_dim: Down-projection output width.
        num_sms: Optional SM limit used for native CuTe tuning.
        enable_host_tensormaps: Whether native inference uses host tensor maps.

    Returns:
        Required routed-row multiple, or ``None`` when no padding is required.
    """
    del estimated_rows, num_local_experts, fc13_output_dim, fc2_output_dim
    del num_sms, enable_host_tensormaps
    # Staged offset-backed scratch retains the source's 128-row routing
    # contract; its paged GEMM launch multiple is resolved separately.
    if inference_mode and compute.blockscaled.mega and compute.config is not None:
        return _inference_routing_multiple(
            compute.config, mega=compute.blockscaled.mega
        )
    return dist_bs_gemm.DEFAULT_M_MULTIPLE_OF


def _async_max_num_recv_tokens(
    *,
    blockscaled_training: bool,
    world_size: int,
    num_tokens: int,
    topk: int,
    num_local_experts: int,
    routing_m_multiple: int | None,
) -> int | None:
    """Return the topology-capped training receive capacity.

    Args:
        blockscaled_training: Whether fixed activation offsets are required.
        world_size: Expert-parallel group size.
        num_tokens: Local input token count.
        topk: Experts selected per token.
        num_local_experts: Experts owned by one rank.
        routing_m_multiple: Per-expert padding multiple.

    Returns:
        Maximum padded receive rows, or ``None`` for inference.
    """
    if not blockscaled_training:
        return None
    return _routing_ptrs_size(
        world_size=world_size,
        num_tokens=num_tokens,
        topk=topk,
        num_local_experts=num_local_experts,
        m_multiple_of=0 if routing_m_multiple is None else routing_m_multiple,
    )


def _async_model_config(
    *,
    dtype: torch.dtype,
    hidden_dim: int,
    intermediate_dim: int,
    num_tokens: int,
    topk: int,
    max_imbalance_factor: float,
    num_moe_layers: int,
    num_local_experts: int,
    routing_world_size: int,
    routing_m_multiple: int | None,
    max_num_recv_tokens: int | None,
    blockscaled_cfg: _BlockscaledConfig,
) -> ModelConfig:
    """Build the planner `ModelConfig` for one async DistMoE call.

    ``routing_world_size`` must match the expert-parallel size used to allocate
    the context. Both planner paths take the tighter of the load-factor bound
    and the routing topology ceiling
    ``num_tokens * EP * min(topk, num_local_experts)``. Dropping it here makes
    execution demand more scratch than the context reserved whenever a rank
    owns fewer experts than ``topk``.

    Args:
        dtype: Dense activation dtype.
        hidden_dim: Model hidden dimension.
        intermediate_dim: Expert intermediate dimension.
        num_tokens: Local token capacity.
        topk: Routes per token.
        max_imbalance_factor: Receive capacity relative to balanced routes.
        num_moe_layers: MoE layers sharing one activation slot.
        num_local_experts: Experts owned by this rank.
        routing_world_size: Expert-parallel group size.
        routing_m_multiple: Optional per-expert row padding multiple.
        max_num_recv_tokens: Optional static receive-row ceiling.
        blockscaled_cfg: Active block-scaled policy.

    Returns:
        Activation-planner model configuration.
    """
    return ModelConfig(
        dtype=dtype,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_tokens=num_tokens,
        topk=topk,
        max_imbalance_factor=max_imbalance_factor,
        num_moe_layers=num_moe_layers,
        blockscaled_storage=_blockscaled_storage_config(blockscaled_cfg),
        num_local_experts=num_local_experts,
        routing_world_size=routing_world_size,
        routing_m_multiple_of=routing_m_multiple,
        max_num_recv_tokens=max_num_recv_tokens,
        mega=blockscaled_cfg.mega,
    )


_SWIGLU_CLAMPED = "swiglu_clamped"


def _postprocess_save_condition(
    forward_plan: ForwardPlan,
    *,
    save_for_backward: bool,
) -> torch.Tensor | None:
    """Return the planner condition for saving postprocess input.

    Args:
        forward_plan: Resolved forward allocation plan.
        save_for_backward: Whether backward may consume saved state.

    Returns:
        Device recompute condition, or ``None`` when no state is retained.
    """
    if not save_for_backward:
        return None
    assert forward_plan.blockscaled is not None
    return forward_plan.blockscaled.recompute_condition


class _BlockScaledAutograd(torch.autograd.Function):
    """Own eager autograd for activation-buffer-backed block-scaled Dist-MoE.

    Forward publishes routing and dispatch state, creates device-side buffer
    offsets, runs the staged or Mega CuTe pipeline, and retains a fixed tensor
    schema for backward. When the selected activation slot cannot retain the
    complete column-quantized WGRAD bundle, it saves only the BF16 layer input;
    backward then relaunches the same forward topology conditionally into
    scratch. Otherwise backward consumes the saved column operands directly.

    Python-only callbacks and direct WGRAD ownership use this eager boundary.
    The registered custom operations below reuse the same methods through a
    minimal context so FakeTensor, non-strict ``make_fx``, activation
    checkpointing, and CUDA graphs observe fixed schemas and stable addresses.
    """

    @staticmethod
    def forward(  # noqa: C901 -- keep the expert pipeline and its storage lifecycle together
        ctx: Any,
        x_TD: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_q_fprop: torch.Tensor | None,
        w13_scale_fprop: torch.Tensor | None,
        w13_q_dgrad: torch.Tensor | None,
        w13_scale_dgrad: torch.Tensor | None,
        w13_global_scale: torch.Tensor | None,
        w13_global_scale_inv: torch.Tensor | None,
        w2_q_fprop: torch.Tensor | None,
        w2_scale_fprop: torch.Tensor | None,
        w2_q_dgrad: torch.Tensor | None,
        w2_scale_dgrad: torch.Tensor | None,
        w2_global_scale: torch.Tensor | None,
        w2_global_scale_inv: torch.Tensor | None,
        comm_group: dist.ProcessGroup,
        comm_buffer: _CommunicationBuffers,
        activation_buffer: ActivationBuffer,
        topk: int,
        num_tokens: int,
        num_experts: int,
        max_imbalance_factor: float,
        weights_preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        experts_postprocess_fn: _ResolvedExpertsPostprocess,
        num_sms: int | None,
        blockscaled_cfg: _BlockscaledConfig,
        config: dict | None = None,
        inplace_wgrad_accum: bool = False,
        wgrad_parameter_refs: tuple[
            weakref.ReferenceType[torch.Tensor],
            weakref.ReferenceType[torch.Tensor],
        ]
        | None = None,
        wgrad_destination_fn: WgradDestinationFn | None = None,
        wgrad_output_dtype: torch.dtype | None = None,
        wgrad_postprocess_fn: WgradPostprocessFn | None = None,
        inference_mode: bool = False,
        save_for_backward: bool = True,
        activation: str = "swiglu",
        swiglu_alpha: float = 1.702,
        swiglu_limit: float = 7.0,
        activation_slot_id_1: torch.Tensor | None = None,
        num_moe_layers_in_slot: int | None = None,
        validate_expert_ids: bool = False,
    ) -> torch.Tensor:
        """Forward pass with routing.

        Args:
            ctx: Autograd context receiving execution and saved-tensor state.
            x_TD: Local input activations with shape ``[T, D]``.
            topk_expert_ids_TK: Global expert IDs with shape ``[T, K]``.
            topk_scores_TK: Router scores with shape ``[T, K]``.
            w13_weight: Native grouped weight or prepared FPROP storage.
            w2_weight: Native grouped weight or packed prepared FPROP storage;
                the context owns its logical intermediate dimension.
            w13_q_fprop: Optional prepared W13 FPROP data.
            w13_scale_fprop: Optional prepared W13 FPROP scales.
            w13_q_dgrad: Optional prepared W13 DGRAD data.
            w13_scale_dgrad: Optional prepared W13 DGRAD scales.
            w13_global_scale: Optional W13 global scale.
            w13_global_scale_inv: Optional reciprocal W13 global scale.
            w2_q_fprop: Optional prepared W2 FPROP data.
            w2_scale_fprop: Optional prepared W2 FPROP scales.
            w2_q_dgrad: Optional prepared W2 DGRAD data.
            w2_scale_dgrad: Optional prepared W2 DGRAD scales.
            w2_global_scale: Optional W2 global scale.
            w2_global_scale_inv: Optional reciprocal W2 global scale.
            comm_group: Expert-parallel process group.
            comm_buffer: Symmetric routing, dispatch, and combine buffers.
            activation_buffer: Activation and scratch buffer.
            topk: Routes per token.
            num_tokens: Local token count.
            num_experts: Global expert count.
            max_imbalance_factor: Receive capacity relative to balanced routes.
            weights_preprocess_fn: Optional native-weight preprocessing callback.
            experts_postprocess_fn: Resolved expert-output processing policy.
            num_sms: Optional SM limit.
            blockscaled_cfg: Resolved precision and pipeline policy.
            config: Optional kernel tuning override.
            inplace_wgrad_accum: Whether standard parameter gradients own the
                WGRAD destinations.
            wgrad_parameter_refs: Weak references to the logical W13 and W2
                gradient owners when parameter accumulation is enabled.
            wgrad_destination_fn: Optional integration-owned WGRAD destination
                resolver.
            wgrad_output_dtype: Optional WGRAD output dtype.
            wgrad_postprocess_fn: Optional WGRAD publication callback.
            inference_mode: Whether inference-specialized execution is selected.
            save_for_backward: Whether this invocation has a backward consumer.
            activation: SwiGLU variant.
            swiglu_alpha: Clamped-SwiGLU alpha.
            swiglu_limit: Clamped-SwiGLU input limit.
            activation_slot_id_1: Device scalar selecting the activation slot.
            num_moe_layers_in_slot: Static layer count used to bound the
                selected pipeline activation slot.
            validate_expert_ids: Whether publication validates expert-ID bounds.

        Returns:
            Combined local output activations with shape ``[T, D]``.
        """
        # 1. Resolve the static compute policy and prepare both weight layouts.
        # Prepared operands remain caller-owned; dynamic operands are retained
        # or regenerated according to the backward recomputation contract.
        input_dtype = x_TD.dtype
        auto_inference_tuning = inference_mode and config is None
        static_inference_storage = (
            inference_mode
            and blockscaled_cfg.mega
            and blockscaled_cfg.static_inference_storage
        )
        assert activation_slot_id_1 is not None
        assert num_moe_layers_in_slot is not None
        # Prepared NVFP4 W2 packs F/2 bytes; dimensions belong to the context,
        # not the physical FPROP storage passed at the registered boundary.
        assert activation_buffer.model_config is not None
        intermediate_dim = activation_buffer.model_config.intermediate_dim
        num_local_experts = _num_local_experts(num_experts, comm_group)
        _validate_compute_config(config)
        compute, save_col_quant = _async_resolve_compute(
            config=config,
            blockscaled_cfg=blockscaled_cfg,
            inference_mode=inference_mode,
            save_for_backward=save_for_backward,
        )
        # Compact decode tuning assumes the Mega inference layout. Staged
        # offset-backed execution retains the asynchronous source defaults.
        if inference_mode and blockscaled_cfg.mega:
            from .kernels._environment import num_sms_per_device
            from .kernels.inference_config import _blockscaled_inference_config

            num_sms = num_sms if num_sms is not None else num_sms_per_device()
            config = _blockscaled_inference_config(
                format=_kernel_format(blockscaled_cfg.format),
                num_tokens=num_tokens,
                topk=topk,
                num_local_experts=num_local_experts,
                hidden_dim=x_TD.shape[1],
                intermediate_dim=intermediate_dim,
                num_sms=num_sms,
                routing_capacity=blockscaled_cfg.routing_capacity,
                mega=blockscaled_cfg.mega,
                interleaved_fc13=blockscaled_cfg.interleaved_fc13,
                config=config,
                input_dtype=input_dtype,
            )
            compute = replace(compute, config=config)
        blockscaled_dispatch = compute.uses_blockscaled_dispatch(
            inference_mode=inference_mode
        )
        ctx.config = config
        ctx.blockscaled_cfg = blockscaled_cfg
        ctx.activation = activation
        ctx.swiglu_alpha = swiglu_alpha
        ctx.swiglu_limit = swiglu_limit
        ctx.wgrad_output_dtype = wgrad_output_dtype
        ctx.wgrad_destination_fn = wgrad_destination_fn
        ctx.inference_mode = inference_mode
        ctx.wgrad_postprocess_fn = wgrad_postprocess_fn
        ctx.recompute_dynamic_weight_quants = _recompute_dynamic_mega_weight_quants(
            compute,
            inference_mode=inference_mode,
            weights_preprocess_fn=weights_preprocess_fn,
        )
        # When weights are 2D, view to 3D for computation.
        # Store original shapes to reshape gradients back in backward.
        ctx.w13_orig_shape = None
        ctx.w2_orig_shape = None
        if w13_weight.ndim == 2:
            ctx.w13_orig_shape = w13_weight.shape
            ctx.w2_orig_shape = w2_weight.shape
            hidden_dim = x_TD.shape[1]
            w13_weight = w13_weight.view(num_local_experts, -1, hidden_dim)
            w2_weight = w2_weight.view(num_local_experts, hidden_dim, -1)

        ctx.weights_preprocess_fn = weights_preprocess_fn
        ctx.inplace_wgrad_accum = inplace_wgrad_accum
        if inplace_wgrad_accum:
            assert wgrad_parameter_refs is not None
            ctx.w13_param_ref, ctx.w2_param_ref = wgrad_parameter_refs
        if weights_preprocess_fn is not None:
            w13_compute_EFD = weights_preprocess_fn(w13_weight)
            w2_compute_EDF = weights_preprocess_fn(w2_weight)
        else:
            w13_compute_EFD = w13_weight
            w2_compute_EDF = w2_weight
        w13_handle = _prepare_weight_handle(
            w13_compute_EFD,
            compute,
            inference_mode=inference_mode,
            prequantized=BlockscaledWeightSpec(
                w_native=w13_compute_EFD,
                w_q_fprop=w13_q_fprop,
                w_scale_fprop=w13_scale_fprop,
                w_q_dgrad=w13_q_dgrad,
                w_scale_dgrad=w13_scale_dgrad,
                w_global_scale=w13_global_scale,
                w_global_scale_inv=w13_global_scale_inv,
            ),
        )
        w2_handle = _prepare_weight_handle(
            w2_compute_EDF,
            compute,
            inference_mode=inference_mode,
            prequantized=BlockscaledWeightSpec(
                w_native=w2_compute_EDF,
                w_q_fprop=w2_q_fprop,
                w_scale_fprop=w2_scale_fprop,
                w_q_dgrad=w2_q_dgrad,
                w_scale_dgrad=w2_scale_dgrad,
                w_global_scale=w2_global_scale,
                w_global_scale_inv=w2_global_scale_inv,
            ),
        )

        # 2. Publish local routing and dispatch payloads before constructing
        # peer pointers. The following barrier is the visibility boundary.
        routing_buffer = comm_buffer.routing
        dispatch_buffer = comm_buffer.dispatch
        combine_buffer = comm_buffer.combine

        # Copy expert_ids to routing buffer (always int16; copy_ casts).
        local_rank = dist.get_rank(comm_group)
        local_routing_buffer = _routing_ids_view(
            routing_buffer,
            local_rank,
            (topk_expert_ids_TK.shape[0], topk_expert_ids_TK.shape[1]),
        )
        routing_token_count_1 = _routing_token_count_view(
            routing_buffer,
            local_rank,
        )

        with record_function("moe_preprocess"):
            assert x_TD.ndim == 2
            dispatch_stride_bytes = None
            if inference_mode or blockscaled_dispatch:
                copy_routing(
                    expert_ids=topk_expert_ids_TK,
                    routing=local_routing_buffer,
                    routing_token_count_1=routing_token_count_1,
                    num_experts=num_experts,
                    validate_expert_ids=validate_expert_ids,
                )
                dispatch_stride_bytes = _stage_async_inference_dispatch(
                    x_TD,
                    dispatch_buffer,
                    blockscaled_cfg if blockscaled_dispatch else None,
                )
            else:
                dispatch_buffer_local = dispatch_buffer.hdl.get_buffer(
                    local_rank,
                    (num_tokens, x_TD.shape[1]),
                    x_TD.dtype,
                )
                copy_routing_and_dispatch(
                    x=x_TD,
                    dispatch=dispatch_buffer_local,
                    expert_ids=topk_expert_ids_TK,
                    routing=local_routing_buffer,
                    routing_token_count_1=routing_token_count_1,
                    num_experts=num_experts,
                    validate_expert_ids=validate_expert_ids,
                )
            # Routing metadata and dispatch rows become peer-visible together,
            # so downstream routing and GEMM need no second rendezvous.
            _ep_barrier(dispatch_buffer)

        # 3. Planning, routing, and grouped GEMM share one topology-capped
        # receive capacity; no host decision depends on the routed row count.
        hidden_dim = x_TD.shape[1]
        routing_m_multiple = _routing_m_multiple(compute, inference_mode=inference_mode)
        routing_world_size = dist.get_world_size(comm_group)
        local_expert_count = w13_compute_EFD.shape[0]
        scratch_imbalance_factor = (
            max_imbalance_factor
            if activation_buffer.scratch_capacity_factor is None
            else activation_buffer.scratch_capacity_factor
        )
        model_config = _async_model_config(
            dtype=x_TD.dtype,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_tokens=num_tokens,
            topk=topk,
            # Fixed block-scaled offsets cover the complete virtual scratch
            # range; VMM decides which suffix is backed by host memory.
            max_imbalance_factor=scratch_imbalance_factor,
            num_moe_layers=activation_buffer.num_moe_layers,
            num_local_experts=local_expert_count,
            routing_world_size=routing_world_size,
            routing_m_multiple=routing_m_multiple,
            max_num_recv_tokens=blockscaled_cfg.routing_capacity
            if inference_mode
            else _async_max_num_recv_tokens(
                blockscaled_training=not inference_mode,
                world_size=routing_world_size,
                num_tokens=num_tokens,
                topk=topk,
                num_local_experts=local_expert_count,
                routing_m_multiple=routing_m_multiple,
            ),
            blockscaled_cfg=blockscaled_cfg,
        )

        with record_function("moe_dispatch_routing"):
            routing = dist_dispatch_routing(
                tokens=x_TD,
                expert_ids=topk_expert_ids_TK,
                num_experts=num_experts,
                group=comm_group,
                comm_buffer=comm_buffer,
                expert_id_offset=None,
                m_multiple_of=routing_m_multiple,
                dispatch_stride_bytes=dispatch_stride_bytes,
                bwd_dispatch_stride_bytes=x_TD.shape[1] * x_TD.element_size(),
                generate_bwd_gather_ptrs=save_for_backward,
                use_low_latency=inference_mode,
                max_num_recv_tokens=model_config.max_recv_tokens,
                pointer_capacity_multiple_of=(
                    128 if inference_mode and blockscaled_cfg.mega else 1
                ),
            )

            num_tokens_per_local_expert_E = routing.num_tokens_per_local_experts
            dispatch_bwd_gather_ptrs = routing.bwd_gather_ptrs
            dispatch_fwd_gather_ptrs = routing.fwd_gather_ptrs
            combine_scatter_ptrs = routing.scatter_ptrs

        # 4. Static inference binds scratch directly. Training and VMM retain
        # device-side offsets and the existing recomputation/placement policy.
        forward_plan = None
        num_recv_tokens = None
        num_recv_tokens_per_rank_snapshot = None
        if not static_inference_storage:
            with record_function("moe_forward_plan"):
                # Backward must retain this call's count even when later
                # routing reuses its metadata storage.
                num_recv_tokens = routing.num_tokens_per_rank[
                    local_rank : local_rank + 1
                ].clone()
                buffer_capacity = activation_buffer.buffer.numel()
                capacity_num_activation_slots = (
                    0 if inference_mode else activation_buffer.num_activation_slots
                )
                validate_buffer_capacity(
                    buffer_size=buffer_capacity,
                    model_config=model_config,
                    num_activation_slots=capacity_num_activation_slots,
                )
                num_recv_tokens_per_rank_snapshot = (
                    torch.empty_like(routing.num_tokens_per_rank)
                    if save_for_backward
                    else None
                )
                forward_plan = get_forward_plan(
                    num_recv_tokens=num_recv_tokens,
                    num_recv_tokens_per_rank=routing.num_tokens_per_rank,
                    buffer_status=activation_buffer,
                    model_config=model_config,
                    activation_slot_id_1=activation_slot_id_1,
                    num_moe_layers_in_slot=num_moe_layers_in_slot,
                    inference_mode=inference_mode,
                    num_recv_tokens_per_rank_snapshot=num_recv_tokens_per_rank_snapshot,
                    scratch_only=not save_for_backward,
                )

        if save_for_backward:
            assert forward_plan is not None and forward_plan.blockscaled is not None
            recompute_condition = forward_plan.blockscaled.recompute_condition
            # Dispatch was already published before routing. This local-only
            # copy retains it when backward will recompute the layer.
            copy_dispatch_to_activation(
                dispatch=x_TD,
                activation_buffer=activation_buffer.buffer,
                activation_offset=forward_plan.x_offset,
                condition=recompute_condition,
            )

        # 5. Execute either the staged or Mega fused expert pipeline. Both
        # produce route-wise H3 before the common postprocessing boundary.
        _initialize_fake_peer_scatter_output(combine_buffer)
        with record_function("moe_forward_main"):
            if blockscaled_cfg is not None:
                blockscaled_forward_plan = None
                if not static_inference_storage:
                    assert (
                        forward_plan is not None
                        and forward_plan.blockscaled is not None
                    )
                    blockscaled_forward_plan = forward_plan.blockscaled
                capacity_rows = model_config.max_recv_tokens
                if inference_mode and blockscaled_cfg.mega:
                    # Descriptor/tile allocation is 128-aligned; routing's
                    # logical padded-row overflow bound remains unchanged.
                    capacity_rows = (capacity_rows + 127) // 128 * 128
                if blockscaled_cfg.mega:
                    mega_result = (
                        dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_dispatch_combine(
                            w13=w13_handle.blockscaled.w_q_fprop,
                            w13_scale=w13_handle.blockscaled.w_scale_fprop,
                            w2=w2_handle.blockscaled.w_q_fprop,
                            w2_scale=w2_handle.blockscaled.w_scale_fprop,
                            w13_global_scale_inv=(
                                w13_handle.blockscaled.w_global_scale_inv
                            ),
                            w2_global_scale_inv=(
                                w2_handle.blockscaled.w_global_scale_inv
                            ),
                            num_tokens_per_local_expert_E=num_tokens_per_local_expert_E,
                            gather_ptrs=dispatch_fwd_gather_ptrs,
                            scatter_ptrs=combine_scatter_ptrs,
                            num_out_tokens=capacity_rows,
                            symm_mem_buffer=combine_buffer,
                            num_output_tokens=num_tokens,
                            format=blockscaled_cfg.format,
                            layout=dist_bs_gemm.DEFAULT_LAYOUT,
                            out_dtype=input_dtype,
                            num_sms=num_sms,
                            sync_peers=False,
                            config=config,
                            m_multiple_of=routing_m_multiple
                            or dist_bs_gemm.DEFAULT_M_MULTIPLE_OF,
                            swiglu_fast_math=blockscaled_cfg.fast_math,
                            swiglu_clamped=activation == _SWIGLU_CLAMPED,
                            swiglu_alpha=swiglu_alpha,
                            swiglu_limit=swiglu_limit,
                            return_x_wgrad_quant=save_col_quant,
                            blockscaled_dispatch=blockscaled_dispatch,
                            return_h2_wgrad_quant=save_col_quant,
                            activation_buffer=(
                                None
                                if static_inference_storage
                                else activation_buffer.buffer
                            ),
                            operand_storage=(
                                activation_buffer.buffer
                                if static_inference_storage
                                else None
                            ),
                            activation_offsets=(
                                None
                                if blockscaled_forward_plan is None
                                else blockscaled_forward_plan.mega_offsets
                            ),
                            interleaved_fc13=blockscaled_cfg.interleaved_fc13,
                            estimate_recv_num_tokens=min(
                                num_tokens * topk, capacity_rows
                            ),
                            derive_pipeline_depth=auto_inference_tuning,
                            auto_inference_tuning=auto_inference_tuning,
                        )
                    )
                    h3_MD = mega_result[0]
                else:
                    assert (
                        forward_plan is not None
                        and forward_plan.blockscaled is not None
                    )
                    gemm_m_multiple = (
                        _inference_routing_multiple(config, mega=False)
                        if inference_mode and config is not None
                        else dist_bs_gemm.DEFAULT_M_MULTIPLE_OF
                    )
                    dist_blockscaled_grouped_gemm_fprop_dispatch(
                        x=x_TD,
                        w=w13_handle.blockscaled.w_q_fprop,
                        w_scale=w13_handle.blockscaled.w_scale_fprop,
                        num_tokens_per_local_expert_E=num_tokens_per_local_expert_E,
                        gather_ptrs=dispatch_fwd_gather_ptrs,
                        num_out_tokens=capacity_rows,
                        symm_mem_buffer=dispatch_buffer,
                        topk=topk,
                        format=blockscaled_cfg.format,
                        layout=dist_bs_gemm.DEFAULT_LAYOUT,
                        out_dtype=input_dtype,
                        num_sms=num_sms,
                        sync_peers=False,
                        copy_inputs=False,
                        config=config,
                        m_multiple_of=gemm_m_multiple,
                        return_wgrad_quant=save_col_quant,
                        blockscaled_dispatch=blockscaled_dispatch,
                        activation_buffer=activation_buffer.buffer,
                        activation_offsets=forward_plan.blockscaled.dispatch_offsets,
                        w_global_scale_inv=(w13_handle.blockscaled.w_global_scale_inv),
                        derive_pipeline_depth=auto_inference_tuning,
                    )
                    h1_placeholder = _activation_buffer_placeholder(
                        activation_buffer.buffer,
                        (capacity_rows, 2 * intermediate_dim),
                        input_dtype,
                    )
                    combine_result = (
                        dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine(
                            h1_M2F=h1_placeholder,
                            w=w2_handle.blockscaled.w_q_fprop,
                            w_scale=w2_handle.blockscaled.w_scale_fprop,
                            num_tokens_per_local_expert_E=num_tokens_per_local_expert_E,
                            scatter_ptrs=combine_scatter_ptrs,
                            symm_mem_buffer=combine_buffer,
                            format=blockscaled_cfg.format,
                            layout=dist_bs_gemm.DEFAULT_LAYOUT,
                            out_dtype=input_dtype,
                            num_sms=num_sms,
                            sync_peers=False,
                            config=config,
                            m_multiple_of=gemm_m_multiple,
                            num_output_tokens=num_tokens,
                            return_wgrad_quant=save_col_quant,
                            swiglu_fast_math=blockscaled_cfg.fast_math,
                            swiglu_clamped=activation == _SWIGLU_CLAMPED,
                            swiglu_alpha=swiglu_alpha,
                            swiglu_limit=swiglu_limit,
                            activation_buffer=activation_buffer.buffer,
                            activation_offsets=forward_plan.blockscaled.combine_offsets,
                            num_recv_tokens=capacity_rows,
                            w_global_scale_inv=(
                                w2_handle.blockscaled.w_global_scale_inv
                            ),
                            derive_pipeline_depth=auto_inference_tuning,
                        )
                    )
                    # Training also emits the column-quantized WGRAD operand;
                    # inference returns only the combined output tensor.
                    h3_MD = combine_result[0] if save_col_quant else combine_result

        w13_handle_tensors = None
        w2_handle_tensors = None
        if save_for_backward:
            w13_handle_tensors = _weight_handle_tensors(
                w13_handle,
                omit_dynamic_quants=ctx.recompute_dynamic_weight_quants,
            )
            w2_handle_tensors = _weight_handle_tensors(
                w2_handle,
                omit_dynamic_quants=ctx.recompute_dynamic_weight_quants,
            )
        del w13_compute_EFD, w2_compute_EDF, w13_handle, w2_handle

        # 6. Publish combine results, apply the resolved route-wise transform,
        # and save only the tensor state required by backward.
        with record_function("moe_postprocess"):
            _ep_barrier(combine_buffer)
            recompute_condition = (
                _postprocess_save_condition(
                    forward_plan, save_for_backward=save_for_backward
                )
                if forward_plan is not None
                else None
            )
            (
                output_TD,
                _,
                postprocess_context,
                expert_postprocess_output_dtype,
            ) = experts_postprocess_fn.forward(
                h3_MD,
                topk_scores_TK,
                output_dtype=input_dtype,
                save=save_for_backward,
                save_buffer=activation_buffer.buffer if save_for_backward else None,
                save_offset=forward_plan.h3_offset if save_for_backward else None,
                save_condition=recompute_condition,
            )

        if save_for_backward:
            ctx.input_dtype = input_dtype
            ctx.expert_postprocess_output_dtype = expert_postprocess_output_dtype
            ctx.topk = topk
            ctx.num_local_input_tokens = num_tokens
            ctx.postprocess = experts_postprocess_fn
            ctx.dispatch_buffer = dispatch_buffer
            ctx.combine_buffer = combine_buffer
            ctx.num_sms = num_sms
            # Save model_config (contains only primitives, safe to save on ctx)
            ctx.model_config = model_config
            # Save comm_group for debug printing in backward
            ctx.comm_group = comm_group
            # Save reference to shared activation_buffer for in-place updates in backward
            # This ensures backward passes update the shared buffer state, not local copies
            ctx.activation_buffer = activation_buffer

            assert num_recv_tokens_per_rank_snapshot is not None
            assert num_recv_tokens is not None
            assert w13_handle_tensors is not None and w2_handle_tensors is not None
            assert forward_plan is not None
            assert forward_plan.blockscaled is not None

            ctx.save_for_backward(
                w13_weight,
                w2_weight,
                *w13_handle_tensors,
                *w2_handle_tensors,
                topk_expert_ids_TK,
                topk_scores_TK,
                num_tokens_per_local_expert_E,
                dispatch_bwd_gather_ptrs,
                dispatch_fwd_gather_ptrs,
                combine_scatter_ptrs,
                num_recv_tokens_per_rank_snapshot,
                num_recv_tokens,
                # ForwardPlan tensors
                forward_plan.need_recompute,
                forward_plan.x_offset,
                forward_plan.x_gathered_offset,
                forward_plan.h1_offset,
                forward_plan.h2_offset,
                forward_plan.h3_offset,
                forward_plan.blockscaled.x_row_offset,
                forward_plan.blockscaled.x_col_offset,
                forward_plan.blockscaled.h2_row_offset,
                forward_plan.blockscaled.h2_col_offset,
                forward_plan.blockscaled.activation_offsets,
                forward_plan.blockscaled.recompute_condition,
                postprocess_context,
                forward_plan.activation_slot_id_1,
            )

        return output_TD

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Run the eager callback-compatible backward implementation.

        Args:
            ctx: Autograd context populated by :meth:`forward`.
            *grad_outputs: The single combined-output gradient with shape ``[T, D]``.

        Returns:
            Gradients aligned with the forward arguments.
        """
        (grad_output_TD,) = grad_outputs
        return _BlockScaledAutograd._backward_impl(ctx, grad_output_TD)

    @staticmethod
    def _backward_impl(  # noqa: C901
        ctx: Any,
        grad_output_TD: torch.Tensor,
        *,
        wgrad_destinations: tuple[_WgradDestination, _WgradDestination] | None = None,
    ) -> tuple[Any, ...]:
        """Run backward, recomputing forward intermediates when selected.

        Args:
            ctx: Autograd context populated by :meth:`forward`.
            grad_output_TD: Gradient of the combined output with shape ``[T, D]``.
            wgrad_destinations: Optional prevalidated W13/W2 destinations.

        Returns:
            Gradients aligned with the forward arguments.
        """
        config = ctx.config
        blockscaled_cfg: _BlockscaledConfig = ctx.blockscaled_cfg
        compute, _ = _async_resolve_compute(
            config=config,
            blockscaled_cfg=blockscaled_cfg,
            inference_mode=False,
            save_for_backward=True,
        )
        save_col_quant = True
        activation = ctx.activation
        swiglu_alpha = ctx.swiglu_alpha
        swiglu_limit = ctx.swiglu_limit
        expert_postprocess_output_dtype = ctx.expert_postprocess_output_dtype
        topk = ctx.topk
        num_tokens = ctx.num_local_input_tokens
        postprocess = ctx.postprocess
        dispatch_buffer = ctx.dispatch_buffer
        combine_buffer = ctx.combine_buffer
        num_sms = ctx.num_sms
        model_config = ctx.model_config
        # The context-owned planner must be mutated in place so backward frees
        # the exact activation slot reserved by its matching forward.
        activation_buffer = ctx.activation_buffer

        # 1. Restore graph-visible routing, offsets, and prepared-weight state.
        weights_preprocess_fn = ctx.weights_preprocess_fn
        (
            w13_weight,
            w2_weight,
            w13_q_fprop,
            w13_scale_fprop,
            w13_q_dgrad,
            w13_scale_dgrad,
            w13_global_scale_inv,
            w2_q_fprop,
            w2_scale_fprop,
            w2_q_dgrad,
            w2_scale_dgrad,
            w2_global_scale_inv,
            topk_expert_ids_TK,
            topk_scores_TK,
            num_tokens_per_local_expert_E,
            dispatch_bwd_gather_ptrs,
            dispatch_fwd_gather_ptrs,
            combine_scatter_ptrs,
            num_recv_tokens_per_rank,
            num_recv_tokens,
            # ForwardPlan tensors
            need_recompute,
            x_offset,
            x_gathered_offset,
            h1_offset,
            h2_offset,
            h3_offset,
            x_row_offset,
            x_col_offset,
            h2_row_offset,
            h2_col_offset,
            forward_activation_offsets,
            recompute_condition,
            postprocess_context,
            activation_slot_id_1,
        ) = ctx.saved_tensors

        # 2. Recreate callback-owned compute views before restoring or rebuilding
        # their block-scaled operands.
        if weights_preprocess_fn is not None:
            w13_compute_EFD = weights_preprocess_fn(w13_weight)
            w2_compute_EDF = weights_preprocess_fn(w2_weight)
        else:
            w13_compute_EFD = w13_weight
            w2_compute_EDF = w2_weight
        parameter_refs = (
            (ctx.w13_param_ref, ctx.w2_param_ref) if ctx.inplace_wgrad_accum else None
        )
        clear_parameter_grads_before_return = wgrad_destinations is None
        if wgrad_destinations is None:
            wgrad_destinations = _resolve_wgrad_destinations(
                inplace_wgrad_accum=ctx.inplace_wgrad_accum,
                parameter_refs=parameter_refs,
                destination_fn=ctx.wgrad_destination_fn,
                w13_EFD=w13_compute_EFD,
                w2_EDF=w2_compute_EDF,
                output_dtype=ctx.wgrad_output_dtype,
            )
        ctx.wgrad_output_dtype = _resolve_wgrad_output_dtype(
            wgrad_destinations,
            ctx.wgrad_output_dtype,
            w13_compute_EFD.dtype,
        )
        w13_handle = _restore_weight_handle(
            w13_compute_EFD,
            (
                w13_q_fprop,
                w13_scale_fprop,
                w13_q_dgrad,
                w13_scale_dgrad,
                w13_global_scale_inv,
            ),
            requantize_with=(compute if ctx.recompute_dynamic_weight_quants else None),
        )
        w2_handle = _restore_weight_handle(
            w2_compute_EDF,
            (
                w2_q_fprop,
                w2_scale_fprop,
                w2_q_dgrad,
                w2_scale_dgrad,
                w2_global_scale_inv,
            ),
            requantize_with=(compute if ctx.recompute_dynamic_weight_quants else None),
        )

        # 3. Reconstruct typed plan views from the saved scalar offsets. The
        # shared planner is mutated in place when backward releases its slot.
        forward_plan = ForwardPlan(
            need_recompute=need_recompute,
            x_offset=x_offset,
            x_gathered_offset=x_gathered_offset,
            h1_offset=h1_offset,
            h2_offset=h2_offset,
            h3_offset=h3_offset,
            activation_slot_id_1=activation_slot_id_1,
            blockscaled=BlockscaledForwardPlan(
                x_row_offset=x_row_offset,
                x_col_offset=x_col_offset,
                h2_row_offset=h2_row_offset,
                h2_col_offset=h2_col_offset,
                activation_offsets=forward_activation_offsets,
                recompute_condition=recompute_condition,
            ),
        )
        assert forward_plan.blockscaled is not None
        conditional_recompute = forward_plan.blockscaled.recompute_condition

        with record_function("moe_backward_plan"):
            backward_plan = get_backward_plan(
                num_recv_tokens=num_recv_tokens,
                num_recv_tokens_per_rank=num_recv_tokens_per_rank,
                forward_plan=forward_plan,
                buffer_status=activation_buffer,
                model_config=model_config,
            )

        # 4. Keep recomputation in the captured graph and predicate its memory
        # traffic and kernels on the device-side planner decision.
        _initialize_fake_peer_scatter_output(combine_buffer)
        with record_function("moe_forward_recompute"):
            hidden_dim = w13_compute_EFD.shape[2]
            x_placeholder = _activation_buffer_placeholder(
                activation_buffer.buffer,
                (num_tokens, hidden_dim),
                model_config.dtype,
            )
            dispatch_buffer_local = dispatch_buffer.hdl.get_buffer(
                dispatch_buffer.hdl.rank,
                (num_tokens, hidden_dim),
                model_config.dtype,
            )
            copy_activation_to_dispatch(
                dispatch=dispatch_buffer_local,
                activation_buffer=activation_buffer.buffer,
                activation_offset=forward_plan.x_offset,
                condition=conditional_recompute,
            )

            # Barrier after buffer copy — skip when not recomputing since
            # no data was written to dispatch_buffer
            _ep_barrier(dispatch_buffer, conditional_execution=conditional_recompute)

            # Recompute forward activations using activation buffer.
            intermediate_dim = w2_compute_EDF.shape[2]
            if blockscaled_cfg is not None:
                assert backward_plan.blockscaled is not None
                capacity_rows = model_config.max_recv_tokens
                if blockscaled_cfg.mega:
                    h3_saved = dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_dispatch_combine(
                        w13=w13_handle.blockscaled.w_q_fprop,
                        w13_scale=w13_handle.blockscaled.w_scale_fprop,
                        w2=w2_handle.blockscaled.w_q_fprop,
                        w2_scale=w2_handle.blockscaled.w_scale_fprop,
                        w13_global_scale_inv=(
                            w13_handle.blockscaled.w_global_scale_inv
                        ),
                        w2_global_scale_inv=(w2_handle.blockscaled.w_global_scale_inv),
                        num_tokens_per_local_expert_E=num_tokens_per_local_expert_E,
                        gather_ptrs=dispatch_fwd_gather_ptrs,
                        scatter_ptrs=combine_scatter_ptrs,
                        num_out_tokens=capacity_rows,
                        symm_mem_buffer=combine_buffer,
                        num_output_tokens=num_tokens,
                        format=blockscaled_cfg.format,
                        layout=dist_bs_gemm.DEFAULT_LAYOUT,
                        out_dtype=model_config.dtype,
                        num_sms=num_sms,
                        sync_peers=False,
                        config=config,
                        m_multiple_of=dist_bs_gemm.DEFAULT_M_MULTIPLE_OF,
                        swiglu_fast_math=blockscaled_cfg.fast_math,
                        swiglu_clamped=activation == _SWIGLU_CLAMPED,
                        swiglu_alpha=swiglu_alpha,
                        swiglu_limit=swiglu_limit,
                        blockscaled_dispatch=False,
                        return_x_wgrad_quant=save_col_quant,
                        return_h2_wgrad_quant=save_col_quant,
                        activation_buffer=activation_buffer.buffer,
                        activation_offsets=backward_plan.blockscaled.recompute_mega_offsets,
                        conditional_execution=conditional_recompute,
                    )[0]
                else:
                    dist_blockscaled_grouped_gemm_fprop_dispatch(
                        x=x_placeholder,
                        w=w13_handle.blockscaled.w_q_fprop,
                        w_scale=w13_handle.blockscaled.w_scale_fprop,
                        num_tokens_per_local_expert_E=num_tokens_per_local_expert_E,
                        gather_ptrs=dispatch_fwd_gather_ptrs,
                        num_out_tokens=capacity_rows,
                        symm_mem_buffer=dispatch_buffer,
                        topk=topk,
                        format=blockscaled_cfg.format,
                        layout=dist_bs_gemm.DEFAULT_LAYOUT,
                        out_dtype=model_config.dtype,
                        num_sms=num_sms,
                        sync_peers=False,
                        copy_inputs=False,
                        config=config,
                        m_multiple_of=dist_bs_gemm.DEFAULT_M_MULTIPLE_OF,
                        return_wgrad_quant=save_col_quant,
                        blockscaled_dispatch=False,
                        activation_buffer=activation_buffer.buffer,
                        activation_offsets=backward_plan.blockscaled.recompute_dispatch_offsets,
                        conditional_execution=conditional_recompute,
                        w_global_scale_inv=(w13_handle.blockscaled.w_global_scale_inv),
                    )
                    h1_placeholder = _activation_buffer_placeholder(
                        activation_buffer.buffer,
                        (capacity_rows, 2 * intermediate_dim),
                        model_config.dtype,
                    )
                    # Training also emits the column-quantized WGRAD operand.
                    recomputed_combine = dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine(
                        h1_M2F=h1_placeholder,
                        w=w2_handle.blockscaled.w_q_fprop,
                        w_scale=w2_handle.blockscaled.w_scale_fprop,
                        num_tokens_per_local_expert_E=num_tokens_per_local_expert_E,
                        scatter_ptrs=combine_scatter_ptrs,
                        symm_mem_buffer=combine_buffer,
                        format=blockscaled_cfg.format,
                        layout=dist_bs_gemm.DEFAULT_LAYOUT,
                        out_dtype=model_config.dtype,
                        num_sms=num_sms,
                        sync_peers=False,
                        config=config,
                        m_multiple_of=dist_bs_gemm.DEFAULT_M_MULTIPLE_OF,
                        num_output_tokens=num_tokens,
                        return_wgrad_quant=save_col_quant,
                        swiglu_fast_math=blockscaled_cfg.fast_math,
                        swiglu_clamped=activation == _SWIGLU_CLAMPED,
                        swiglu_alpha=swiglu_alpha,
                        swiglu_limit=swiglu_limit,
                        activation_buffer=activation_buffer.buffer,
                        activation_offsets=backward_plan.blockscaled.recompute_combine_offsets,
                        conditional_execution=conditional_recompute,
                        num_recv_tokens=capacity_rows,
                        w_global_scale_inv=(w2_handle.blockscaled.w_global_scale_inv),
                    )
                    h3_saved = (
                        recomputed_combine[0] if save_col_quant else recomputed_combine
                    )

            if not postprocess.fuses_saved_copy_into_reduction:
                conditional_copy_activations(
                    condition=conditional_recompute,
                    lhs=None,
                    lhs_offset=None,
                    rhs=h3_saved,
                    rhs_offset=forward_plan.h3_offset,
                    activation_buffer=activation_buffer.buffer,
                    copy_to_buffer=False,
                )

            # Barrier before consuming combine output — skip when not recomputing
            # since combine fprop was skipped and h3 was loaded from activation buffer
            _ep_barrier(combine_buffer, conditional_execution=conditional_recompute)

        # 5. Reverse postprocessing and publish route-wise gradients before
        # either staged or Mega expert backward consumes them.
        with record_function("moe_backward_postprocess"):
            grad_h3, grad_topk_scores_TK, grad_h3_is_published = postprocess.backward(
                grad_output_TD,
                h3_saved,
                topk_scores_TK,
                postprocess_output_dtype=expert_postprocess_output_dtype,
                context=postprocess_context,
                publish_view=lambda shape, dtype: _local_symm_mem_buffer_view(
                    combine_buffer,
                    shape,
                    dtype,
                ),
                x_buffer=activation_buffer.buffer,
                x_buffer_offset=forward_plan.h3_offset,
                x_buffer_condition=conditional_recompute,
            )
            del grad_output_TD, h3_saved

            if not grad_h3_is_published:
                _local_symm_mem_buffer_view(
                    combine_buffer,
                    tuple(grad_h3.shape),
                    grad_h3.dtype,
                ).copy_(grad_h3)

            # Barrier after publishing route gradients into the combine buffer.
            _ep_barrier(combine_buffer)

        # 6. Execute W2 then W13 DGRAD/WGRAD in dependency order. The Mega path
        # fuses each DGRAD/WGRAD pair; staged execution launches them separately.
        with record_function("moe_backward_main"):
            w2_wgrad_out = (
                None if wgrad_destinations is None else wgrad_destinations[1].output
            )
            w2_output_accum = (
                False
                if wgrad_destinations is None
                else wgrad_destinations[1].accumulate
            )
            fused_dgrad_wgrad = blockscaled_cfg.mega
            if fused_dgrad_wgrad and w2_wgrad_out is None:
                w2_wgrad_out = _allocate_wgrad_output(
                    weight_compute=w2_compute_EDF,
                    output_dtype=ctx.wgrad_output_dtype,
                )

            # W2 consumes the route-output gradient and the retained or
            # recomputed column-quantized H2 operand.
            grad_w2_EDF = None
            if blockscaled_cfg is not None:
                assert backward_plan.blockscaled is not None
                capacity_rows = model_config.max_recv_tokens
                if blockscaled_cfg.mega:
                    h2_col_quant = _async_blockscaled_col_quant_placeholder(
                        activation_buffer.buffer,
                        rows=capacity_rows,
                        dim=model_config.intermediate_dim,
                        cfg=blockscaled_cfg,
                    )
                    _, grad_w2_EDF, _ = (
                        dist_blockscaled_grouped_gemm_dgrad_wgrad_dispatch(
                            dy=grad_h3,
                            w=w2_handle.blockscaled.w_q_dgrad,
                            w_scale=w2_handle.blockscaled.w_scale_dgrad,
                            num_tokens_per_local_expert_E=num_tokens_per_local_expert_E,
                            gather_ptrs=combine_scatter_ptrs,
                            num_out_tokens=capacity_rows,
                            symm_mem_buffer=combine_buffer,
                            x_wgrad_quant=h2_col_quant,
                            format=blockscaled_cfg.format,
                            layout=dist_bs_gemm.DEFAULT_LAYOUT,
                            out_dtype=model_config.dtype,
                            wgrad_out_dtype=ctx.wgrad_output_dtype,
                            num_sms=num_sms,
                            sync_peers=False,
                            copy_inputs=False,
                            dw_GNK=w2_wgrad_out,
                            config=config,
                            m_multiple_of=dist_bs_gemm.DEFAULT_M_MULTIPLE_OF,
                            output_accum=w2_output_accum,
                            activation_buffer=activation_buffer.buffer,
                            activation_offsets=backward_plan.blockscaled.fc2_mega_offsets,
                        )
                    )
                else:
                    dist_blockscaled_grouped_gemm_dgrad_dispatch(
                        dy=grad_h3,
                        w=w2_handle.blockscaled.w_q_dgrad,
                        w_scale_dgrad=w2_handle.blockscaled.w_scale_dgrad,
                        num_tokens_per_local_expert_E=num_tokens_per_local_expert_E,
                        gather_ptrs=combine_scatter_ptrs,
                        num_out_tokens=capacity_rows,
                        symm_mem_buffer=combine_buffer,
                        format=blockscaled_cfg.format,
                        layout=dist_bs_gemm.DEFAULT_LAYOUT,
                        out_dtype=model_config.dtype,
                        num_sms=num_sms,
                        sync_peers=False,
                        copy_inputs=False,
                        config=config,
                        m_multiple_of=dist_bs_gemm.DEFAULT_M_MULTIPLE_OF,
                        return_wgrad_quant=save_col_quant,
                        activation_buffer=activation_buffer.buffer,
                        activation_offsets=backward_plan.blockscaled.fc2_dgrad_offsets,
                    )
            del grad_h3

            if not fused_dgrad_wgrad and w2_wgrad_out is None:
                w2_wgrad_out = _allocate_wgrad_output(
                    weight_compute=w2_compute_EDF,
                    output_dtype=ctx.wgrad_output_dtype,
                )

            if not blockscaled_cfg.mega:
                assert backward_plan.blockscaled is not None
                grad_w2_EDF = _async_blockscaled_wgrad(
                    activation_offsets=backward_plan.blockscaled.fc2_wgrad_offsets,
                    dy_dim=model_config.hidden_dim,
                    x_dim=model_config.intermediate_dim,
                    rows=model_config.max_recv_tokens,
                    split_sizes=num_tokens_per_local_expert_E,
                    activation_buffer=activation_buffer.buffer,
                    cfg=blockscaled_cfg,
                    config=config,
                    dw=w2_wgrad_out,
                    output_accum=w2_output_accum,
                    out_dtype=ctx.wgrad_output_dtype,
                )
            else:
                assert grad_w2_EDF is not None
            grad_w2_EDF = _postprocess_wgrad(
                ctx.wgrad_postprocess_fn, "w2", grad_w2_EDF
            )

            w13_wgrad_out = (
                None if wgrad_destinations is None else wgrad_destinations[0].output
            )
            w13_output_accum = (
                False
                if wgrad_destinations is None
                else wgrad_destinations[0].accumulate
            )
            if fused_dgrad_wgrad and w13_wgrad_out is None:
                w13_wgrad_out = _allocate_wgrad_output(
                    weight_compute=w13_compute_EFD,
                    output_dtype=ctx.wgrad_output_dtype,
                )

            # SwiGLU backward produces the FC13 route-gradient operand.
            intermediate_dim = w2_compute_EDF.shape[2]
            grad_w13_EFD = None
            _initialize_fake_peer_scatter_output(dispatch_buffer)
            if blockscaled_cfg is not None:
                assert backward_plan.blockscaled is not None
                capacity_rows = model_config.max_recv_tokens
                grad_h2_placeholder = _activation_buffer_placeholder(
                    activation_buffer.buffer,
                    (capacity_rows, intermediate_dim),
                    model_config.dtype,
                )
                h1_placeholder = _activation_buffer_placeholder(
                    activation_buffer.buffer,
                    (capacity_rows, 2 * intermediate_dim),
                    model_config.dtype,
                )
                if blockscaled_cfg.mega:
                    x_col_quant = _async_blockscaled_col_quant_placeholder(
                        activation_buffer.buffer,
                        rows=capacity_rows,
                        dim=model_config.hidden_dim,
                        cfg=blockscaled_cfg,
                    )
                    grad_x_gathered, grad_w13_EFD, _ = (
                        dist_blockscaled_grouped_gemm_dgrad_wgrad_swiglu_bwd_combine(
                            grad_h2_MF=grad_h2_placeholder,
                            h1_M2F=h1_placeholder,
                            w=w13_handle.blockscaled.w_q_dgrad,
                            w_scale_dgrad=w13_handle.blockscaled.w_scale_dgrad,
                            num_tokens_per_local_expert_E=num_tokens_per_local_expert_E,
                            scatter_ptrs=dispatch_bwd_gather_ptrs,
                            symm_mem_buffer=dispatch_buffer,
                            x_wgrad_quant=x_col_quant,
                            format=blockscaled_cfg.format,
                            layout=dist_bs_gemm.DEFAULT_LAYOUT,
                            out_dtype=model_config.dtype,
                            wgrad_out_dtype=ctx.wgrad_output_dtype,
                            num_sms=num_sms,
                            sync_peers=False,
                            dw_GNK=w13_wgrad_out,
                            config=config,
                            m_multiple_of=dist_bs_gemm.DEFAULT_M_MULTIPLE_OF,
                            num_output_tokens=num_tokens,
                            output_accum=w13_output_accum,
                            swiglu_fast_math=blockscaled_cfg.fast_math,
                            swiglu_clamped=activation == _SWIGLU_CLAMPED,
                            swiglu_alpha=swiglu_alpha,
                            swiglu_limit=swiglu_limit,
                            activation_buffer=activation_buffer.buffer,
                            activation_offsets=backward_plan.blockscaled.fc13_mega_offsets,
                        )
                    )
                else:
                    grad_x_gathered, _ = (
                        dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine(
                            grad_h2_MF=grad_h2_placeholder,
                            h1_M2F=h1_placeholder,
                            w=w13_handle.blockscaled.w_q_dgrad,
                            w_scale_dgrad=w13_handle.blockscaled.w_scale_dgrad,
                            num_tokens_per_local_expert_E=num_tokens_per_local_expert_E,
                            scatter_ptrs=dispatch_bwd_gather_ptrs,
                            symm_mem_buffer=dispatch_buffer,
                            format=blockscaled_cfg.format,
                            layout=dist_bs_gemm.DEFAULT_LAYOUT,
                            out_dtype=model_config.dtype,
                            num_sms=num_sms,
                            sync_peers=False,
                            config=config,
                            m_multiple_of=dist_bs_gemm.DEFAULT_M_MULTIPLE_OF,
                            num_output_tokens=num_tokens,
                            swiglu_fast_math=blockscaled_cfg.fast_math,
                            swiglu_clamped=activation == _SWIGLU_CLAMPED,
                            swiglu_alpha=swiglu_alpha,
                            swiglu_limit=swiglu_limit,
                            activation_buffer=activation_buffer.buffer,
                            activation_offsets=backward_plan.blockscaled.fc13_dgrad_offsets,
                            num_recv_tokens=capacity_rows,
                        )
                    )

            if not fused_dgrad_wgrad and w13_wgrad_out is None:
                w13_wgrad_out = _allocate_wgrad_output(
                    weight_compute=w13_compute_EFD,
                    output_dtype=ctx.wgrad_output_dtype,
                )

            if not blockscaled_cfg.mega:
                assert backward_plan.blockscaled is not None
                grad_w13_EFD = _async_blockscaled_wgrad(
                    activation_offsets=backward_plan.blockscaled.fc13_wgrad_offsets,
                    dy_dim=2 * model_config.intermediate_dim,
                    x_dim=model_config.hidden_dim,
                    rows=model_config.max_recv_tokens,
                    split_sizes=num_tokens_per_local_expert_E,
                    activation_buffer=activation_buffer.buffer,
                    cfg=blockscaled_cfg,
                    config=config,
                    dw=w13_wgrad_out,
                    output_accum=w13_output_accum,
                    out_dtype=ctx.wgrad_output_dtype,
                )
            else:
                assert grad_w13_EFD is not None
            grad_w13_EFD = _postprocess_wgrad(
                ctx.wgrad_postprocess_fn, "w13", grad_w13_EFD
            )
            if clear_parameter_grads_before_return and wgrad_destinations is not None:
                for destination in wgrad_destinations:
                    if destination.parameter is not None:
                        destination.parameter.grad = None

        with record_function("moe_backward_preprocess"):
            # Barrier before consuming dgrad_combine output
            _ep_barrier(dispatch_buffer)

            grad_x_TD = reduce_from_topk(
                grad_x_gathered,
                expert_ids=topk_expert_ids_TK,
            )
            del grad_x_gathered

        # 7. Restore logical weight shapes and suppress gradients whose storage
        # was supplied and mutated by an external owner.
        if ctx.w13_orig_shape is not None:
            if grad_w13_EFD is not None:
                grad_w13_EFD = grad_w13_EFD.view(ctx.w13_orig_shape)
            if grad_w2_EDF is not None:
                grad_w2_EDF = grad_w2_EDF.view(ctx.w2_orig_shape)

        external_wgrad_destination = ctx.wgrad_destination_fn is not None

        return (
            grad_x_TD,
            None,  # topk_expert_ids_TK
            grad_topk_scores_TK,
            None if external_wgrad_destination else grad_w13_EFD,
            None if external_wgrad_destination else grad_w2_EDF,
            *((None,) * 37),
        )


def _validate_fused_blockscaled_mode(
    cfg: _BlockscaledConfig,
    w13: BlockscaledWeightSpec,
    w2: BlockscaledWeightSpec,
    *,
    inference_mode: bool,
    hidden_dim: int,
    intermediate_dim: int,
) -> None:
    """Validate format, inference, preparation, and shape constraints.

    Args:
        cfg: Resolved block-scaled format.
        w13: First projection weight specification.
        w2: Second projection weight specification.
        inference_mode: Whether the activation buffer is scratch-only.
        hidden_dim: Model hidden width.
        intermediate_dim: Expert intermediate width.

    Raises:
        ValueError: If mode or weight preparation is invalid.
        NotImplementedError: If dimensions exceed kernel constraints.
    """
    format_name = cfg.format.value.upper()
    if cfg.policy.inference_only and not inference_mode:
        raise ValueError(
            f"async {format_name} requires an inference-mode activation buffer"
        )
    if cfg.policy.requires_prequantized_weights and (
        not w13.is_prequantized or not w2.is_prequantized
    ):
        raise ValueError(f"fused {format_name} requires pre-quantized weights")
    if not cfg.policy.supports_dims(hidden_dim, intermediate_dim):
        raise NotImplementedError(
            f"fused {format_name} requires hidden_dim to be a multiple of "
            f"{cfg.policy.input_dim_multiple}"
            + (
                ""
                if cfg.policy.input_dim_max is None
                else f" and no larger than {cfg.policy.input_dim_max}"
            )
            + ", and intermediate_dim to be a multiple of "
            f"{cfg.policy.swiglu_dim_multiple}"
            + (
                ""
                if cfg.policy.swiglu_dim_max is None
                else f" and no larger than {cfg.policy.swiglu_dim_max}"
            )
            + f"; got {hidden_dim=}, {intermediate_dim=}"
        )


def _run_blockscaled_eager(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_spec: BlockscaledWeightSpec,
    w2_spec: BlockscaledWeightSpec,
    comm_group: dist.ProcessGroup,
    comm_buffer: _CommunicationBuffers,
    activation_buffer: ActivationBuffer,
    max_imbalance_factor: float,
    *,
    save_for_backward: bool,
    weights_preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    experts_postprocess_fn: _ExpertsOutputPostprocess = None,
    num_sms: int | None = None,
    num_local_experts: int | None = None,
    inplace_wgrad_accum: bool = False,
    wgrad_parameter_owners: tuple[torch.Tensor, torch.Tensor] | None = None,
    wgrad_destination_fn: WgradDestinationFn | None = None,
    wgrad_output_dtype: torch.dtype | None = None,
    wgrad_postprocess_fn: WgradPostprocessFn | None = None,
    blockscaled_cfg: _BlockscaledConfig,
    config: dict | None = None,
    activation: str = "swiglu",
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    activation_slot_id_1: torch.Tensor | None = None,
    num_moe_layers_in_slot: int | None = None,
    validate_expert_ids: bool = False,
    zero_out_padded_callback_inputs: bool = False,
) -> torch.Tensor:
    """Run eager block-scaled distributed MoE with SwiGLU activation.

    Uses a pre-allocated shared buffer for overlapping with other operations.

    Performs distributed MoE computation with the following steps:
    1. Route tokens to experts based on top-k selection
    2. Apply first grouped GEMM (w13) with token dispatch
    3. Apply SwiGLU activation
    4. Apply second grouped GEMM (w2) with result combining
    5. Apply experts_postprocess_fn if provided (e.g., post-expert normalization)
    6. Scale and sum expert outputs

    Args:
        x_TD: Input tensor with shape ``[T, D]``.
        topk_expert_ids_TK: Expert IDs with shape ``[T, K]``.
        topk_scores_TK: Expert scores with shape ``[T, K]``.
        w13_spec: Native and optional prepared W13 operands.
        w2_spec: Native and optional prepared W2 operands.
        comm_group: Distributed process group
        comm_buffer: Pre-allocated communication buffer.
        activation_buffer: Shared buffer for activation saving.
        max_imbalance_factor: Maximum token imbalance factor for recompute decision.
            If the ratio of max tokens per rank to average exceeds this, recompute is used.
        save_for_backward: Whether this invocation has a backward consumer.
        weights_preprocess_fn: Optional function to dequantize compact weights (e.g. FP8)
            to compute dtype (e.g. BF16) before GEMM operations. If provided, the original
            compact weights are saved for backward (reducing peak memory) and re-dequantized
            in backward before gradient computation.
        experts_postprocess_fn: Optional function to apply after expert computation
            but before scale and combine. Takes and returns tensor of shape [num_tokens * topk, hidden_dim].
            Typically used for post-expert normalization.
        num_sms: Optional number of SMs to use for grouped GEMM kernels. If None, uses all
            available SMs. Setting this allows reserving SMs for DP collectives overlap.
        num_local_experts: Optional local expert count used to reshape flat weights.
        wgrad_output_dtype: Optional dtype for materialized expert weight gradients.
        wgrad_postprocess_fn: Optional callback applied when each WGRAD is ready.
        blockscaled_cfg: Resolved precision and staged/Mega pipeline policy.
        config: Optional explicit kernel configuration.
        activation: SwiGLU implementation selected by the caller.
        swiglu_alpha: Sigmoid multiplier for clamped SwiGLU.
        swiglu_limit: Symmetric preactivation limit for clamped SwiGLU.
        activation_slot_id_1: Optional graph-stable activation-stack index.
        num_moe_layers_in_slot: Static number of MoE layers sharing the
            selected activation slot.
        validate_expert_ids: Whether publication validates expert-ID bounds.
        zero_out_padded_callback_inputs: Whether invalid callback inputs are
            zeroed before invoking user code.
        inplace_wgrad_accum: Whether serialized eager backward calls may fuse
            WGRAD into the standard parameter gradient.
        wgrad_parameter_owners: Optional outer W13 and W2 parameters whose
            gradient fields own local-view WGRAD results.
        wgrad_destination_fn: Optional integration-owned WGRAD destination
            resolver. The callback owns the result and suppresses the matching
            autograd weight gradient.

    Returns:
        Output tensor of shape [num_tokens, hidden_dim]

    Raises:
        ValueError: If shapes, modes, or execution controls are incompatible.
    """
    if inplace_wgrad_accum and wgrad_postprocess_fn is not None:
        raise ValueError(
            "inplace_wgrad_accum cannot be combined with wgrad_postprocess_fn"
        )
    if wgrad_destination_fn is not None and (
        inplace_wgrad_accum or wgrad_postprocess_fn is not None
    ):
        raise ValueError(
            "wgrad_destination_fn cannot be combined with inplace_wgrad_accum "
            "or wgrad_postprocess_fn"
        )
    if wgrad_parameter_owners is not None:
        if not inplace_wgrad_accum:
            raise ValueError("wgrad_parameter_owners requires inplace_wgrad_accum=True")
        if len(wgrad_parameter_owners) != 2:
            raise ValueError("wgrad_parameter_owners must contain W13 and W2")
    if (
        w13_spec.is_prequantized or w2_spec.is_prequantized
    ) and weights_preprocess_fn is not None:
        raise ValueError(
            "pre-quantized weight tuples cannot be combined with weights_preprocess_fn"
        )
    w13_native = w13_spec.w_native
    w2_native = w2_spec.w_native

    # === Reshape if 2D weights provided ===
    hidden_dim = x_TD.shape[1]
    if num_local_experts is not None:
        w13_EFD = w13_native.view(num_local_experts, -1, hidden_dim)
        w2_EDF = w2_native.view(num_local_experts, hidden_dim, -1)
    else:
        w13_EFD = w13_native
        w2_EDF = w2_native

    wgrad_parameter_refs = None
    if inplace_wgrad_accum:
        wgrad_parameter_refs = (
            _weak_parameter_ref(
                w13_EFD,
                (None if wgrad_parameter_owners is None else wgrad_parameter_owners[0]),
            ),
            _weak_parameter_ref(
                w2_EDF,
                (None if wgrad_parameter_owners is None else wgrad_parameter_owners[1]),
            ),
        )

    # === Validation (always on 3D) ===
    assert w13_EFD.shape[0] == w2_EDF.shape[0], (
        f"w13 and w2 must have same number of local experts: "
        f"w13.shape[0]={w13_EFD.shape[0]} != w2.shape[0]={w2_EDF.shape[0]}"
    )
    assert w13_EFD.shape[2] == w2_EDF.shape[1], (
        f"Hidden dimensions must match: "
        f"w13.shape[2]={w13_EFD.shape[2]} != w2.shape[1]={w2_EDF.shape[1]}"
    )
    assert w13_EFD.shape[1] == 2 * w2_EDF.shape[2], (
        f"w13.shape[1] must be 2x w2.shape[2] for SwiGLU: "
        f"w13.shape[1]={w13_EFD.shape[1]} != 2 * w2.shape[2]={2 * w2_EDF.shape[2]}"
    )
    assert x_TD.shape[1] == w13_EFD.shape[2], (
        f"Input hidden dimension must match weight hidden dimension: "
        f"x.shape[1]={x_TD.shape[1]} != w13.shape[2]={w13_EFD.shape[2]}"
    )
    assert topk_expert_ids_TK.shape == topk_scores_TK.shape, (
        f"topk_ids and topk_scores must have same shape: "
        f"topk_ids.shape={topk_expert_ids_TK.shape} != topk_scores.shape={topk_scores_TK.shape}"
    )
    assert topk_expert_ids_TK.shape[0] == x_TD.shape[0], (
        f"Number of tokens must match: "
        f"topk_ids.shape[0]={topk_expert_ids_TK.shape[0]} != x.shape[0]={x_TD.shape[0]}"
    )

    topk = topk_expert_ids_TK.shape[1]
    num_tokens = x_TD.shape[0]
    n_local_experts = w13_EFD.shape[0]
    group_size = dist.get_world_size(comm_group)
    num_experts = n_local_experts * group_size

    _num_moe_layers_in_selected_slot = (
        activation_buffer._num_moe_layers_in_selected_slot
        if num_moe_layers_in_slot is None
        else num_moe_layers_in_slot
    )
    # Inference specialization follows the buffer; backward retention is a
    # separate per-call decision made before entering the autograd function.
    inference_mode = activation_buffer.inference_mode
    if inference_mode and torch.is_grad_enabled():
        raise ValueError(
            "An inference-mode activation buffer can only be used while grad mode is disabled."
        )
    postprocess = _resolve_experts_output_postprocess_fn(
        experts_postprocess_fn,
        x_TD=x_TD,
        topk_expert_ids_TK=topk_expert_ids_TK,
        topk_scores_TK=topk_scores_TK,
        inference_mode=inference_mode,
        zero_out_padded_callback_inputs=zero_out_padded_callback_inputs,
    )
    _validate_fused_blockscaled_mode(
        blockscaled_cfg,
        w13_spec,
        w2_spec,
        inference_mode=inference_mode,
        hidden_dim=hidden_dim,
        intermediate_dim=w2_EDF.shape[2],
    )

    if not save_for_backward:
        w13_spec = _detach_blockscaled_weight_spec(w13_spec)
        w2_spec = _detach_blockscaled_weight_spec(w2_spec)
        x_TD = x_TD.detach()
        topk_scores_TK = topk_scores_TK.detach()
        w13_native = w13_spec.w_native
        w2_native = w2_spec.w_native

    return _BlockScaledAutograd.apply(
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        w13_native,
        w2_native,
        w13_spec.w_q_fprop,
        w13_spec.w_scale_fprop,
        w13_spec.w_q_dgrad,
        w13_spec.w_scale_dgrad,
        w13_spec.w_global_scale,
        w13_spec.w_global_scale_inv,
        w2_spec.w_q_fprop,
        w2_spec.w_scale_fprop,
        w2_spec.w_q_dgrad,
        w2_spec.w_scale_dgrad,
        w2_spec.w_global_scale,
        w2_spec.w_global_scale_inv,
        comm_group,
        comm_buffer,
        activation_buffer,
        topk,
        num_tokens,
        num_experts,
        max_imbalance_factor,
        weights_preprocess_fn,
        postprocess,
        num_sms,
        blockscaled_cfg,
        config,
        inplace_wgrad_accum,
        wgrad_parameter_refs,
        wgrad_destination_fn,
        wgrad_output_dtype,
        wgrad_postprocess_fn,
        inference_mode,
        save_for_backward,
        activation,
        swiglu_alpha,
        swiglu_limit,
        (
            activation_buffer.activation_slot_id_1
            if activation_slot_id_1 is None
            else activation_slot_id_1
        ),
        _num_moe_layers_in_selected_slot,
        validate_expert_ids,
    )


def _validate_mxfp8_weight_shape(weight: torch.Tensor) -> None:
    """Validate the source quantizer's grouped weight geometry from metadata.

    Args:
        weight: Grouped weight; checking shape never reads tensor contents.

    Raises:
        ValueError: If the grouped output/input dimensions cannot form complete
            128-value blocked-scale atoms in both orientations.
    """
    if weight.ndim != 3:
        raise ValueError("MXFP8 weight preparation requires grouped 3D weights")
    if weight.shape[-2] % 128 or weight.shape[-1] % 128:
        raise ValueError("MXFP8 weight dimensions must be divisible by 128")


@torch.library.custom_op(
    "dist_moe::prepare_mxfp8_weight",
    mutates_args=(),
)
def _prepare_mxfp8_weight_op(
    weight: torch.Tensor,
    inference: bool = False,
    fuse_swiglu: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Prepare MXFP8 operands behind one eager/fake/tracing boundary.

    Args:
        weight: Grouped BF16 expert weight.
        inference: Prepare only the forward quantization orientation.
        fuse_swiglu: Interleave inference W13 rows after quantization.

    Returns:
        Qdata, FPROP scales, and DGRAD scales (empty for inference).

    Raises:
        ValueError: If ``weight`` is not a supported grouped shape.
    """
    _validate_mxfp8_weight_shape(weight)
    cfg = _BlockscaledConfig(
        format=BlockScaledFormat.MXFP8_E4M3,
        fast_math=False,
        mega=False,
    )
    prepared = _prepare_blockscaled_weight_impl(
        weight, cfg, use_row_1d_quantization=inference, fuse_swiglu=fuse_swiglu
    )
    if prepared.w_q_dgrad is not prepared.w_q_fprop:
        raise RuntimeError("MXFP8 FPROP and DGRAD must share qdata storage")
    return (
        prepared.w_q_fprop,
        prepared.w_scale_fprop,
        weight.new_empty((0,), dtype=torch.float8_e8m0fnu)
        if inference
        else prepared.w_scale_dgrad,
    )


@_prepare_mxfp8_weight_op.register_fake
def _prepare_mxfp8_weight_fake(
    weight: torch.Tensor,
    inference: bool = False,
    fuse_swiglu: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return shape-correct MXFP8 operands without reading fake data.

    Args:
        weight: Fake grouped BF16 expert weight.
        inference: Whether backward scale storage is absent.
        fuse_swiglu: Whether W13 rows have interleaved inference order.

    Returns:
        Fake shared qdata, FPROP scales, and DGRAD scales.

    Raises:
        ValueError: If ``weight`` is not a grouped three-dimensional tensor.
    """
    _validate_mxfp8_weight_shape(weight)
    groups, rows, cols = weight.shape
    qdata = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    return (
        qdata,
        torch.empty(
            (
                groups * ((rows + 127) // 128) * 128,
                ((cols // 32 + 3) // 4) * 4,
            ),
            dtype=torch.float8_e8m0fnu,
            device=weight.device,
        ),
        weight.new_empty((0,), dtype=torch.float8_e8m0fnu)
        if inference
        else torch.empty(
            (
                groups * ((cols + 127) // 128) * 128,
                ((rows // 32 + 3) // 4) * 4,
            ),
            dtype=torch.float8_e8m0fnu,
            device=weight.device,
        ),
    )


def _prepare_blockscaled_weight(  # noqa: C901 -- preparation modes share one ownership contract
    weight: torch.Tensor,
    config: BlockScaledConfig,
    *,
    inference: bool = False,
    fuse_swiglu: bool = False,
    out: PreparedWeight | None = None,
) -> PreparedWeight:
    """Prepare caller-owned FPROP and DGRAD operands for async DistMoE.

    Args:
        weight: Grouped native expert weight.
        config: Block-scaled execution policy.
        inference: Whether the prepared result may omit DGRAD operands.
        fuse_swiglu: Prepare interleaved W13 qdata/scales for Mega inference
            only. W2 and training retain their canonical row order.
        out: Optional MXFP8 training storage to refill for the same shape and
            format. Refill is unavailable for inference and NVFP4.

    Returns:
        Prepared weight accepted by :func:`_run_blockscaled`.

    Raises:
        ValueError: If weight metadata, inference layout, or ``out`` violates
            the selected format or MXFP8 training refill contract.
    """
    from ._execution import PreparedWeight

    kernel_format = _kernel_block_scaled_format(config.format)
    if weight.ndim != 3 or weight.dtype != torch.bfloat16:
        raise ValueError("weight must be grouped BF16 with shape [E, N, K]")
    if not weight.is_cuda or not weight.is_contiguous():
        raise ValueError("weight must be contiguous CUDA storage")
    if fuse_swiglu and not inference:
        raise ValueError("fuse_swiglu requires inference=True")
    if fuse_swiglu and config.pipeline != "mega":
        raise ValueError("fuse_swiglu requires the Mega inference pipeline")
    if fuse_swiglu and weight.shape[1] % 16:
        raise ValueError("fused inference W13 requires F divisible by eight")
    if out is not None:
        if inference or kernel_format != BlockScaledFormat.MXFP8_E4M3:
            raise ValueError("caller-owned refill supports MXFP8 training weights")
        if out.format != config.format:
            raise ValueError("out is not compatible MXFP8 prepared storage")
        if out.dgrad_data is None or out.dgrad_scale is None:
            raise ValueError("out must contain MXFP8 DGRAD operands")
        if out.source.shape != weight.shape or out.source.dtype != weight.dtype:
            raise ValueError("out source shape and dtype must match weight")
        with torch.no_grad():
            qdata, fprop_scale, dgrad_scale = _prepare_mxfp8_weight_op(weight)
            out.fprop_data.copy_(qdata)
            out.fprop_scale.copy_(fprop_scale)
            out.dgrad_scale.copy_(dgrad_scale)
        return out._with_source(weight)
    if kernel_format == BlockScaledFormat.NVFP4:
        if weight.ndim != 3:
            raise ValueError("NVFP4 weight preparation requires grouped 3D weights")
        with torch.no_grad():
            (
                qdata_ENQ,
                scale_RB,
                global_scale_E,
                global_scale_inv_E,
            ) = _prepare_nvfp4_weight_operands(weight, fuse_swiglu=fuse_swiglu)
        return PreparedWeight._create(
            source=weight,
            format=config.format,
            fprop_data=qdata_ENQ,
            fprop_scale=scale_RB,
            dgrad_data=None,
            dgrad_scale=None,
            global_scale=global_scale_E,
            global_scale_inv=global_scale_inv_E,
            interleaved_fc13=fuse_swiglu,
        )
    if kernel_format is BlockScaledFormat.MXFP8_E4M3:
        with torch.no_grad():
            qdata, scale_fprop, scale_dgrad = _prepare_mxfp8_weight_op(
                weight, inference=inference, fuse_swiglu=fuse_swiglu
            )
        return PreparedWeight._create(
            source=weight,
            format=config.format,
            fprop_data=qdata,
            fprop_scale=scale_fprop,
            dgrad_data=None if inference else qdata,
            dgrad_scale=None if inference else scale_dgrad,
            global_scale=None,
            interleaved_fc13=fuse_swiglu,
        )
    raise ValueError(f"unsupported prepared-weight format: {config.format}")


def _empty_private_state(reference: torch.Tensor) -> torch.Tensor:
    """Return a distinct empty tensor for an absent private state slot.

    Args:
        reference: Tensor supplying device placement.

    Returns:
        Zero-byte uint8 tensor.
    """
    return reference.new_empty((0,), dtype=torch.uint8)


def _forward_weight_state(
    saved: tuple[torch.Tensor | None, ...],
    start: int,
    supplied: bool,
    reference: torch.Tensor,
    format: BlockScaledFormat,
) -> tuple[torch.Tensor, ...]:
    """Return non-aliasing custom-op outputs for one prepared weight.

    Args:
        saved: Flattened private weight state.
        start: Starting index for this weight.
        supplied: Whether the caller supplied prepared operands.
        reference: Tensor supplying device placement for empty state.
        format: Active block-scaled format.

    Returns:
        Five graph-owned weight-state tensors.
    """
    if supplied:
        return tuple(
            _empty_private_state(reference) for _ in range(_SAVED_WEIGHT_STATE_COUNT)
        )
    q_fprop, scale_fprop, q_dgrad, scale_dgrad, global_scale_inv = saved[
        start : start + _SAVED_WEIGHT_STATE_COUNT
    ]
    assert q_fprop is not None and scale_fprop is not None and scale_dgrad is not None
    if format in FP4_FORMATS:
        assert q_dgrad is not None
    else:
        q_dgrad = _empty_private_state(reference)
    return (
        q_fprop,
        scale_fprop,
        q_dgrad,
        scale_dgrad,
        global_scale_inv
        if global_scale_inv is not None
        else _empty_private_state(reference),
    )


def _fake_weight_state(
    weight: torch.Tensor,
    supplied: bool,
    format: BlockScaledFormat,
) -> tuple[torch.Tensor, ...]:
    """Describe dynamic weight operands without reading data.

    Args:
        weight: Native grouped weight defining shape and device.
        supplied: Whether the caller supplied prepared operands.
        format: Active block-scaled format.

    Returns:
        Five FakeTensor-compatible weight-state tensors.
    """
    if supplied:
        return tuple(
            _empty_private_state(weight) for _ in range(_SAVED_WEIGHT_STATE_COUNT)
        )
    operand_dtype, scale_dtype, vector_size = block_scaled_format_constants(format)
    groups, rows, cols = weight.shape
    packed_cols = cols // 2 if format in FP4_FORMATS else cols
    q_fprop = weight.new_empty((groups, rows, packed_cols), dtype=operand_dtype)
    scale_fprop = weight.new_empty(
        (groups * rows, cols // vector_size), dtype=scale_dtype
    )
    q_dgrad = (
        weight.new_empty((groups, cols, rows // 2), dtype=operand_dtype).transpose(1, 2)
        if format in FP4_FORMATS
        else _empty_private_state(weight)
    )
    scale_dgrad = weight.new_empty(
        (groups * cols, rows // vector_size), dtype=scale_dtype
    )
    return q_fprop, scale_fprop, q_dgrad, scale_dgrad, _empty_private_state(weight)


def _blockscaled_compute_config(
    context: Context,
) -> tuple[_BlockscaledConfig, dict | None]:
    """Resolve the typed block-scaled policy and tuning configuration.

    Args:
        context: Owning DistMoE context.

    Returns:
        Private kernel policy and optional tuning dictionary.
    """
    policy = context.config.block_scaled
    assert policy is not None
    blockscaled_cfg = _BlockscaledConfig(
        format=_kernel_block_scaled_format(policy.format),
        fast_math=policy.fast_math,
        mega=policy.pipeline == "mega",
        static_inference_storage=context.config.inference
        and context.config.vmm is None,
    )
    config = (
        None
        if policy.kernel_config is None
        else policy.kernel_config._as_kernel_kwargs()
    )
    return blockscaled_cfg, config


def _pack_block_scaled_backward_state(
    saved: tuple[torch.Tensor | None, ...],
    *,
    prepared_weights: bool,
    reference: torch.Tensor,
    format: BlockScaledFormat,
) -> list[torch.Tensor]:
    """Pack eager forward state into the registered backward schema.

    Args:
        saved: Private tensors retained by the shared forward implementation.
        prepared_weights: Whether both weights remain explicit forward inputs.
        reference: Tensor supplying device placement for empty state.
        format: Active block-scaled format.

    Returns:
        Twenty-one graph-visible tensors consumed by the backward operation.
    """
    weights = (
        *_forward_weight_state(
            saved,
            _W13_SAVED_WEIGHT_START,
            prepared_weights,
            reference,
            format,
        ),
        *_forward_weight_state(
            saved,
            _W2_SAVED_WEIGHT_START,
            prepared_weights,
            reference,
            format,
        ),
    )
    offsets = []
    for index in _FORWARD_OFFSET_INDICES:
        offset = saved[index]
        assert offset is not None
        offsets.append(offset)
    packed_offsets = torch.cat(offsets)
    postprocess_context = saved[_FORWARD_POSTPROCESS_CONTEXT]
    if postprocess_context is None:
        postprocess_context = _empty_private_state(reference)
    state = list(weights)
    for tensor in (
        *saved[_FORWARD_ROUTING_STATE],
        saved[_FORWARD_NEED_RECOMPUTE],
        packed_offsets,
        saved[_FORWARD_ACTIVATION_OFFSETS],
        saved[_FORWARD_RECOMPUTE_CONDITION],
        postprocess_context,
    ):
        assert tensor is not None
        state.append(tensor)
    return state


def _block_scaled_forward_metadata(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    prepared: list[torch.Tensor | None],
    rmsnorm_enabled: bool,
    rmsnorm_recompute_rstd: bool,
    save_for_backward: bool,
    context_id: str,
) -> list[torch.Tensor]:
    """Create fixed-shape forward metadata without reading tensor data.

    Args:
        x_TD: Local activations defining output device and dtype.
        topk_expert_ids_TK: Routed IDs defining token and top-k dimensions.
        w13_EFD: Gate/up weight defining expert and operand dimensions.
        w2_EDF: Down-projection weight defining operand dimensions.
        prepared: Twelve optional prepared-weight tensors.
        rmsnorm_enabled: Whether RMSNorm requires route-wise context.
        rmsnorm_recompute_rstd: Whether backward recomputes that context.
        save_for_backward: Whether the fixed state has a backward consumer.
        context_id: Registered context selecting static topology and format.

    Returns:
        Fixed tensor-only state matching the registered forward schema.
    """
    context = _get_context(context_id)
    if not save_for_backward:
        return [_empty_private_state(x_TD) for _ in range(_BACKWARD_STATE_TENSOR_COUNT)]
    policy = context.config.block_scaled
    assert policy is not None
    kernel_format = _kernel_block_scaled_format(policy.format)
    world_size = dist.get_world_size(context.group)
    local_experts = w13_EFD.shape[0]
    pointer_count = _routing_ptrs_size(
        world_size=world_size,
        num_tokens=x_TD.shape[0],
        topk=context.top_k,
        num_local_experts=local_experts,
        m_multiple_of=dist_bs_gemm.DEFAULT_M_MULTIPLE_OF,
    )
    integer = topk_expert_ids_TK.new_empty
    w13_qdata, *_ = prepared[:_PREPARED_WEIGHT_OPERAND_COUNT]
    w2_qdata, *_ = prepared[_PREPARED_WEIGHT_OPERAND_COUNT:]
    weights = (
        *_fake_weight_state(w13_EFD, w13_qdata is not None, kernel_format),
        *_fake_weight_state(w2_EDF, w2_qdata is not None, kernel_format),
    )
    routing = (
        integer((local_experts,), dtype=torch.int32),
        integer((pointer_count,), dtype=torch.int64),
        integer((pointer_count,), dtype=torch.int64),
        integer((pointer_count,), dtype=torch.int64),
        integer((world_size,), dtype=torch.int32),
        integer((1,), dtype=torch.int32),
    )
    planner = (
        integer((1,), dtype=torch.bool),
        integer((10,), dtype=torch.int64),
        integer((FORWARD_ACTIVATION_OFFSET_COUNT,), dtype=torch.int64),
        integer((1,), dtype=torch.int32),
        (
            torch.empty(
                (x_TD.shape[0], topk_expert_ids_TK.shape[1]),
                dtype=torch.float32,
                device=x_TD.device,
            )
            if rmsnorm_enabled and not rmsnorm_recompute_rstd
            else x_TD.new_empty(0)
        ),
    )
    return [*weights, *routing, *planner]


def _run_async_forward(
    kernel_ctx: _KernelAutogradContext,
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    prepared: list[torch.Tensor | None],
    context: Context,
    postprocess: _ResolvedExpertsPostprocess,
    activation_slot_id_1: torch.Tensor,
    num_moe_layers_in_slot: int,
    save_for_backward: bool,
    validate_expert_ids: bool,
    interleaved_fc13: bool = False,
) -> torch.Tensor:
    """Invoke the asynchronous forward implementation.

    Args:
        kernel_ctx: Minimal autograd context populated by the implementation.
        x_TD: Local BF16 activations with shape ``[T, D]``.
        topk_expert_ids_TK: Routed expert IDs with shape ``[T, K]``.
        topk_scores_TK: Routing weights with shape ``[T, K]``.
        w13_EFD: Logical gate/up expert weights.
        w2_EDF: Logical down-projection expert weights.
        prepared: Twelve optional quantized weight operands.
        context: Reusable distributed execution context.
        postprocess: Resolved expert-output reduction stage.
        activation_slot_id_1: Device scalar selecting activation storage.
        num_moe_layers_in_slot: Static MoE-layer depth of the selected slot.
        save_for_backward: Whether this invocation has a backward consumer.
        validate_expert_ids: Whether publication validates expert-ID bounds.
        interleaved_fc13: Explicit inference-only prepared W13 layout.

    Returns:
        Local combined expert output with shape ``[T, D]``.
    """
    activation_buffer = context.activation_buffer
    assert activation_buffer is not None
    blockscaled_cfg, config = _blockscaled_compute_config(context)
    if interleaved_fc13 and (
        not context.config.inference or not blockscaled_cfg.mega or save_for_backward
    ):
        raise ValueError("interleaved W13 requires forward-only Mega inference")
    blockscaled_cfg = replace(
        blockscaled_cfg,
        interleaved_fc13=interleaved_fc13,
        routing_capacity=(
            context.memory_plan.total_scratch_capacity_rows
            if context.config.inference and context.memory_plan is not None
            else None
        ),
    )
    (
        w13_q_fprop,
        w13_scale_fprop,
        w13_q_dgrad,
        w13_scale_dgrad,
        w13_global_scale,
        w13_global_scale_inv,
        w2_q_fprop,
        w2_scale_fprop,
        w2_q_dgrad,
        w2_scale_dgrad,
        w2_global_scale,
        w2_global_scale_inv,
    ) = prepared
    return _BlockScaledAutograd.forward(
        kernel_ctx,
        x_TD=x_TD,
        topk_expert_ids_TK=topk_expert_ids_TK,
        topk_scores_TK=topk_scores_TK,
        w13_weight=w13_EFD,
        w2_weight=w2_EDF,
        w13_q_fprop=w13_q_fprop,
        w13_scale_fprop=w13_scale_fprop,
        w13_q_dgrad=w13_q_dgrad,
        w13_scale_dgrad=w13_scale_dgrad,
        w13_global_scale=w13_global_scale,
        w13_global_scale_inv=w13_global_scale_inv,
        w2_q_fprop=w2_q_fprop,
        w2_scale_fprop=w2_scale_fprop,
        w2_q_dgrad=w2_q_dgrad,
        w2_scale_dgrad=w2_scale_dgrad,
        w2_global_scale=w2_global_scale,
        w2_global_scale_inv=w2_global_scale_inv,
        comm_group=context.group,
        comm_buffer=context.buffers,
        activation_buffer=activation_buffer,
        topk=context.top_k,
        num_tokens=x_TD.shape[0],
        num_experts=context.config.num_experts,
        max_imbalance_factor=context.config.device_scratch_capacity_factor,
        weights_preprocess_fn=None,
        experts_postprocess_fn=postprocess,
        num_sms=context.config.num_sms,
        blockscaled_cfg=blockscaled_cfg,
        config=config,
        inplace_wgrad_accum=False,
        wgrad_parameter_refs=None,
        wgrad_output_dtype=context.config.wgrad_dtype,
        inference_mode=context.config.inference,
        save_for_backward=save_for_backward,
        activation=context.config.activation,
        swiglu_alpha=context.config.swiglu_alpha,
        swiglu_limit=context.config.swiglu_limit,
        activation_slot_id_1=activation_slot_id_1,
        num_moe_layers_in_slot=num_moe_layers_in_slot,
        validate_expert_ids=validate_expert_ids,
    )


@torch.library.custom_op(
    "dist_moe::block_scaled_forward",
    mutates_args=(),
    device_types="cuda",
)
def _block_scaled_forward_op(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    prepared: list[torch.Tensor | None],
    rmsnorm_weight_D: torch.Tensor | None,
    rmsnorm_enabled: bool,
    rmsnorm_eps: float,
    rmsnorm_norm_output_dtype: torch.dtype,
    rmsnorm_output_dtype: torch.dtype,
    rmsnorm_require_bitwise: bool,
    rmsnorm_gain_center: float,
    rmsnorm_use_kahan: bool,
    rmsnorm_recompute_rstd: bool,
    activation_slot_id_1: torch.Tensor,
    num_moe_layers_in_slot: int,
    save_for_backward: bool,
    validate_expert_ids: bool,
    context_id: str,
    interleaved_fc13: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run async MXFP8 forward and expose fixed-shape backward metadata.

    Args:
        x_TD: Local BF16 activations with shape ``[T, D]``.
        topk_expert_ids_TK: Routed expert IDs with shape ``[T, K]``.
        topk_scores_TK: Routing weights with shape ``[T, K]``.
        w13_EFD: Logical or quantized gate/up expert weight.
        w2_EDF: Logical or quantized down-projection expert weight.
        prepared: Twelve optional prepared-weight tensors.
        rmsnorm_weight_D: Optional inference-only input-scale gamma.
        rmsnorm_enabled: Whether to run fused post-expert RMSNorm.
        rmsnorm_eps: RMSNorm epsilon.
        rmsnorm_norm_output_dtype: Normalized route-output dtype.
        rmsnorm_output_dtype: Final reduction output dtype.
        rmsnorm_require_bitwise: Whether to preserve standalone reduction order.
        rmsnorm_gain_center: Constant added to ``rmsnorm_weight_D``.
        rmsnorm_use_kahan: Whether to use compensated sum-of-squares.
        rmsnorm_recompute_rstd: Whether backward recomputes reciprocal RMS values.
        activation_slot_id_1: Device scalar selecting activation storage.
        num_moe_layers_in_slot: Static MoE-layer depth of the selected slot.
        save_for_backward: Whether this invocation has a backward consumer.
        validate_expert_ids: Whether publication validates expert-ID bounds.
        context_id: Registered DistMoE context identifier.
        interleaved_fc13: Whether prepared W13 has fused inference row order.

    Returns:
        The local output and fixed-shape private backward state.

    Raises:
        RuntimeError: If prepared or saved state violates the operation schema.
    """
    if len(prepared) != _PREPARED_WEIGHT_PAIR_COUNT:
        raise RuntimeError("block-scaled forward requires twelve prepared-weight slots")
    # 1. Reconstruct process-local policy from graph-visible scalar and tensor
    # arguments, then delegate arithmetic to the shared eager implementation.
    context = _get_context(context_id)
    # 2. Rebuild the lightweight context expected by the shared eager forward;
    # it carries no independent storage or lifetime.
    kernel_ctx = _KernelAutogradContext()
    postprocess_config = _rmsnorm_from_registered_args(
        rmsnorm_weight_D,
        rmsnorm_enabled,
        rmsnorm_eps,
        rmsnorm_norm_output_dtype,
        rmsnorm_output_dtype,
        rmsnorm_require_bitwise,
        rmsnorm_gain_center,
        rmsnorm_use_kahan,
        rmsnorm_recompute_rstd,
    )
    postprocess = _resolve_experts_output_postprocess_fn(
        postprocess_config,
        x_TD=x_TD,
        topk_expert_ids_TK=topk_expert_ids_TK,
        topk_scores_TK=topk_scores_TK,
        inference_mode=context.config.inference,
    )
    output_TD = _run_async_forward(
        kernel_ctx,
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        w13_EFD,
        w2_EDF,
        prepared,
        context,
        postprocess,
        activation_slot_id_1,
        num_moe_layers_in_slot,
        save_for_backward,
        validate_expert_ids,
        interleaved_fc13,
    )
    if not save_for_backward:
        return output_TD, _block_scaled_forward_metadata(
            x_TD,
            topk_expert_ids_TK,
            w13_EFD,
            w2_EDF,
            prepared,
            rmsnorm_enabled,
            rmsnorm_recompute_rstd,
            save_for_backward,
            context_id,
        )
    # 3. Pack the eager autograd state into a fixed tensor-only schema so
    # tracing, activation checkpointing, and CUDA graphs preserve its topology.
    saved = kernel_ctx.saved_tensors
    if len(saved) != _ASYNC_FORWARD_SAVED_TENSOR_COUNT:
        raise RuntimeError(
            f"async block-scaled forward produced {len(saved)} saved tensors"
        )
    policy = context.config.block_scaled
    assert policy is not None
    kernel_format = _kernel_block_scaled_format(policy.format)
    w13_prepared = prepared[:_PREPARED_WEIGHT_OPERAND_COUNT]
    w2_prepared = prepared[_PREPARED_WEIGHT_OPERAND_COUNT:]
    w13_qdata, *_ = w13_prepared
    w2_qdata, *_ = w2_prepared
    prepared_weights = w13_qdata is not None
    if prepared_weights != (w2_qdata is not None):
        raise RuntimeError("W13 and W2 prepared state must be supplied together")
    return output_TD, _pack_block_scaled_backward_state(
        saved,
        prepared_weights=prepared_weights,
        reference=x_TD,
        format=kernel_format,
    )


@_block_scaled_forward_op.register_fake
def _block_scaled_forward_fake(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    prepared: list[torch.Tensor | None],
    rmsnorm_weight_D: torch.Tensor | None,
    rmsnorm_enabled: bool,
    rmsnorm_eps: float,
    rmsnorm_norm_output_dtype: torch.dtype,
    rmsnorm_output_dtype: torch.dtype,
    rmsnorm_require_bitwise: bool,
    rmsnorm_gain_center: float,
    rmsnorm_use_kahan: bool,
    rmsnorm_recompute_rstd: bool,
    activation_slot_id_1: torch.Tensor,
    num_moe_layers_in_slot: int,
    save_for_backward: bool,
    validate_expert_ids: bool,
    context_id: str,
    interleaved_fc13: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return fake async-forward outputs without reading tensor data.

    Args:
        x_TD: Fake local activations with shape ``[T, D]``.
        topk_expert_ids_TK: Fake routed expert IDs with shape ``[T, K]``.
        topk_scores_TK: Fake routing weights with shape ``[T, K]``.
        w13_EFD: Fake gate/up expert weight.
        w2_EDF: Fake down-projection expert weight.
        prepared: Twelve optional fake prepared-weight tensors.
        rmsnorm_weight_D: Fake optional input-scale gamma.
        rmsnorm_enabled: Whether fused post-expert RMSNorm is selected.
        rmsnorm_eps: RMSNorm epsilon.
        rmsnorm_norm_output_dtype: Normalized route-output dtype.
        rmsnorm_output_dtype: Final reduction output dtype.
        rmsnorm_require_bitwise: Whether to preserve standalone reduction order.
        rmsnorm_gain_center: Constant added to the input-scale gamma.
        rmsnorm_use_kahan: Whether to use compensated sum-of-squares.
        rmsnorm_recompute_rstd: Whether backward recomputes reciprocal RMS values.
        activation_slot_id_1: Fake device activation-slot scalar.
        num_moe_layers_in_slot: Static MoE-layer depth of the selected slot.
        save_for_backward: Whether the fixed state has a backward consumer.
        validate_expert_ids: Debug validation policy, unused by fake execution.
        context_id: Registered DistMoE context identifier.
        interleaved_fc13: Static prepared inference layout, unused by metadata.

    Returns:
        Fake output and fixed-shape private backward state.
    """
    del topk_scores_TK, rmsnorm_weight_D, rmsnorm_eps
    del rmsnorm_output_dtype, rmsnorm_require_bitwise, rmsnorm_gain_center
    del rmsnorm_use_kahan, activation_slot_id_1, num_moe_layers_in_slot
    del validate_expert_ids, interleaved_fc13
    return torch.empty_like(x_TD), _block_scaled_forward_metadata(
        x_TD,
        topk_expert_ids_TK,
        w13_EFD,
        w2_EDF,
        prepared,
        rmsnorm_enabled,
        rmsnorm_recompute_rstd,
        save_for_backward,
        context_id,
    )


_block_scaled_forward_op.register_effect(torch.library.EffectType.ORDERED)


def _optional_state(tensor: torch.Tensor) -> torch.Tensor | None:
    """Convert a zero-byte custom-op placeholder back to ``None``.

    Args:
        tensor: Graph-owned private state tensor.

    Returns:
        ``None`` for an empty placeholder, otherwise ``tensor``.
    """
    return None if tensor.numel() == 0 else tensor


def _run_block_scaled_registered_backward(
    grad_output_TD: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    state: list[torch.Tensor],
    rmsnorm_enabled: bool,
    rmsnorm_eps: float,
    rmsnorm_norm_output_dtype: torch.dtype,
    rmsnorm_output_dtype: torch.dtype,
    rmsnorm_require_bitwise: bool,
    rmsnorm_use_kahan: bool,
    rmsnorm_recompute_rstd: bool,
    wgrad_output_dtype: torch.dtype,
    context_id: str,
    *,
    wgrad_destinations: tuple[_WgradDestination, _WgradDestination] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rebuild block-scaled state and run the shared backward implementation.

    Args:
        grad_output_TD: Local BF16 output gradient with shape ``[T, D]``.
        w13_EFD: Logical or quantized gate/up expert weight.
        w2_EDF: Logical or quantized down-projection expert weight.
        topk_expert_ids_TK: Routed expert IDs with shape ``[T, K]``.
        topk_scores_TK: Routing weights with shape ``[T, K]``.
        state: Twenty-one tensors returned by the forward operation.
        rmsnorm_enabled: Whether fused post-expert RMSNorm is selected.
        rmsnorm_eps: RMSNorm epsilon.
        rmsnorm_norm_output_dtype: Normalized route-output dtype.
        rmsnorm_output_dtype: Final reduction output dtype.
        rmsnorm_require_bitwise: Whether to preserve standalone reduction order.
        rmsnorm_use_kahan: Whether to use compensated sum-of-squares.
        rmsnorm_recompute_rstd: Whether backward recomputes reciprocal RMS values.
        wgrad_output_dtype: Resolved BF16 or FP32 WGRAD output dtype.
        context_id: Registered DistMoE context identifier.
        wgrad_destinations: Validated accumulation destinations, or ``None``
            for fresh functional WGRAD outputs.

    Returns:
        Gradients for activations, routing scores, ``w13``, and ``w2``.

    Raises:
        RuntimeError: If the forward-state schema is invalid.
    """
    if len(state) != _BACKWARD_STATE_TENSOR_COUNT:
        raise RuntimeError(f"block-scaled backward received {len(state)} tensors")
    # 1. Recover process-local policy and unpack the fixed tensor schema emitted
    # by the registered forward operation.
    context = _get_context(context_id)
    policy = context.config.block_scaled
    activation_buffer = context.activation_buffer
    assert policy is not None and activation_buffer is not None
    kernel_format = _kernel_block_scaled_format(policy.format)
    blockscaled_cfg, config = _blockscaled_compute_config(context)
    postprocess_config = _rmsnorm_from_registered_args(
        None,
        rmsnorm_enabled,
        rmsnorm_eps,
        rmsnorm_norm_output_dtype,
        rmsnorm_output_dtype,
        rmsnorm_require_bitwise,
        0.0,
        rmsnorm_use_kahan,
        rmsnorm_recompute_rstd,
    )
    postprocess = _resolve_experts_output_postprocess_fn(
        postprocess_config,
        x_TD=grad_output_TD,
        topk_expert_ids_TK=topk_expert_ids_TK,
        topk_scores_TK=topk_scores_TK,
    )
    weights: list[torch.Tensor | None] = list(state[_BACKWARD_WEIGHT_STATE])
    if kernel_format not in FP4_FORMATS:
        weights[2] = weights[0]
        weights[7] = weights[5]
    assert weights[4] is not None and weights[9] is not None
    weights[4] = _optional_state(weights[4])
    weights[9] = _optional_state(weights[9])

    # 2. Rebuild the lightweight context expected by the shared eager backward;
    # it carries no independent storage or lifetime.
    kernel_ctx = _KernelAutogradContext()
    kernel_ctx.set(
        config=config,
        blockscaled_cfg=blockscaled_cfg,
        wgrad_output_dtype=wgrad_output_dtype,
        inference_mode=False,
        w13_orig_shape=None,
        w2_orig_shape=None,
        weights_preprocess_fn=None,
        recompute_dynamic_weight_quants=False,
        inplace_wgrad_accum=False,
        w13_param_ref=None,
        w2_param_ref=None,
        wgrad_destination_fn=None,
        wgrad_postprocess_fn=None,
        input_dtype=grad_output_TD.dtype,
        expert_postprocess_output_dtype=(
            rmsnorm_output_dtype if rmsnorm_enabled else grad_output_TD.dtype
        ),
        topk=context.top_k,
        num_local_input_tokens=grad_output_TD.shape[0],
        postprocess=postprocess,
        dispatch_buffer=context.buffers.dispatch,
        combine_buffer=context.buffers.combine,
        num_sms=context.config.num_sms,
        model_config=_async_model_config(
            dtype=grad_output_TD.dtype,
            hidden_dim=context.config.hidden_dim,
            intermediate_dim=context.config.intermediate_dim,
            num_tokens=grad_output_TD.shape[0],
            topk=context.top_k,
            max_imbalance_factor=(
                context.config.device_scratch_capacity_factor
                if activation_buffer.scratch_capacity_factor is None
                else activation_buffer.scratch_capacity_factor
            ),
            num_moe_layers=context.config.max_moe_layers_per_activation_slot,
            num_local_experts=w13_EFD.shape[0],
            routing_world_size=dist.get_world_size(context.group),
            routing_m_multiple=dist_bs_gemm.DEFAULT_M_MULTIPLE_OF,
            max_num_recv_tokens=None,
            blockscaled_cfg=blockscaled_cfg,
        ),
        comm_group=context.group,
        activation_buffer=activation_buffer,
        activation=context.config.activation,
        swiglu_alpha=context.config.swiglu_alpha,
        swiglu_limit=context.config.swiglu_limit,
    )
    routing = state[_BACKWARD_ROUTING_STATE]
    (
        need_recompute,
        packed_offsets,
        activation_offsets,
        recompute_condition,
        postprocess_context,
    ) = state[_BACKWARD_PLAN_STATE]
    offsets = packed_offsets.chunk(10)
    kernel_ctx.saved_tensors = (
        w13_EFD,
        w2_EDF,
        *weights,
        topk_expert_ids_TK,
        topk_scores_TK,
        *routing,
        need_recompute,
        *offsets[:9],
        activation_offsets,
        recompute_condition,
        _optional_state(postprocess_context),
        offsets[9],
    )
    # 3. Delegate to the single backward implementation and expose only the
    # four public autograd gradients from its full positional result.
    gradients = _BlockScaledAutograd._backward_impl(
        kernel_ctx,
        grad_output_TD,
        wgrad_destinations=wgrad_destinations,
    )
    return gradients[0], gradients[2], gradients[3], gradients[4]


@torch.library.custom_op(
    "dist_moe::block_scaled_backward",
    mutates_args=(),
    device_types="cuda",
)
def _block_scaled_backward_op(
    grad_output_TD: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    state: list[torch.Tensor],
    rmsnorm_enabled: bool,
    rmsnorm_eps: float,
    rmsnorm_norm_output_dtype: torch.dtype,
    rmsnorm_output_dtype: torch.dtype,
    rmsnorm_require_bitwise: bool,
    rmsnorm_use_kahan: bool,
    rmsnorm_recompute_rstd: bool,
    wgrad_output_dtype: torch.dtype,
    context_id: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run block-scaled backward and return fresh gradients."""
    return _run_block_scaled_registered_backward(
        grad_output_TD,
        w13_EFD,
        w2_EDF,
        topk_expert_ids_TK,
        topk_scores_TK,
        state,
        rmsnorm_enabled,
        rmsnorm_eps,
        rmsnorm_norm_output_dtype,
        rmsnorm_output_dtype,
        rmsnorm_require_bitwise,
        rmsnorm_use_kahan,
        rmsnorm_recompute_rstd,
        wgrad_output_dtype,
        context_id,
        wgrad_destinations=None,
    )


@torch.library.custom_op(
    "dist_moe::block_scaled_backward_accumulate_",
    mutates_args=("accumulator_grad_w13_EFD", "accumulator_grad_w2_EDF"),
    device_types="cuda",
)
def _block_scaled_backward_accumulate_op(
    grad_output_TD: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    accumulator_grad_w13_EFD: torch.Tensor,
    accumulator_grad_w2_EDF: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    state: list[torch.Tensor],
    rmsnorm_enabled: bool,
    rmsnorm_eps: float,
    rmsnorm_norm_output_dtype: torch.dtype,
    rmsnorm_output_dtype: torch.dtype,
    rmsnorm_require_bitwise: bool,
    rmsnorm_use_kahan: bool,
    rmsnorm_recompute_rstd: bool,
    wgrad_output_dtype: torch.dtype,
    context_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run block-scaled backward and add WGRAD into explicit destinations."""
    wgrad_destinations = _wgrad_accumulation_destinations(
        accumulator_grad_w13_EFD,
        accumulator_grad_w2_EDF,
        w13_EFD,
        w2_EDF,
        wgrad_output_dtype,
    )
    gradients = _run_block_scaled_registered_backward(
        grad_output_TD,
        w13_EFD,
        w2_EDF,
        topk_expert_ids_TK,
        topk_scores_TK,
        state,
        rmsnorm_enabled,
        rmsnorm_eps,
        rmsnorm_norm_output_dtype,
        rmsnorm_output_dtype,
        rmsnorm_require_bitwise,
        rmsnorm_use_kahan,
        rmsnorm_recompute_rstd,
        wgrad_output_dtype,
        context_id,
        wgrad_destinations=wgrad_destinations,
    )
    return gradients[0], gradients[1]


@_block_scaled_backward_op.register_fake
def _block_scaled_backward_fake(
    grad_output_TD: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    state: list[torch.Tensor],
    rmsnorm_enabled: bool,
    rmsnorm_eps: float,
    rmsnorm_norm_output_dtype: torch.dtype,
    rmsnorm_output_dtype: torch.dtype,
    rmsnorm_require_bitwise: bool,
    rmsnorm_use_kahan: bool,
    rmsnorm_recompute_rstd: bool,
    wgrad_output_dtype: torch.dtype,
    context_id: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fake gradients without executing asynchronous kernels.

    Args:
        grad_output_TD: Fake local output gradient with shape ``[T, D]``.
        w13_EFD: Fake gate/up expert weight.
        w2_EDF: Fake down-projection expert weight.
        topk_expert_ids_TK: Fake routed expert IDs.
        topk_scores_TK: Fake routing weights.
        state: Fake private forward state.
        rmsnorm_enabled: Whether fused post-expert RMSNorm is selected.
        rmsnorm_eps: RMSNorm epsilon.
        rmsnorm_norm_output_dtype: Normalized route-output dtype.
        rmsnorm_output_dtype: Final reduction output dtype.
        rmsnorm_require_bitwise: Whether to preserve standalone reduction order.
        rmsnorm_use_kahan: Whether to use compensated sum-of-squares.
        rmsnorm_recompute_rstd: Whether backward recomputes reciprocal RMS values.
        context_id: Registered DistMoE context identifier.

    Returns:
        Fake gradients for activations, routing scores, ``w13``, and ``w2``.
    """
    del topk_expert_ids_TK, state, rmsnorm_enabled
    del rmsnorm_eps, rmsnorm_norm_output_dtype, rmsnorm_output_dtype
    del rmsnorm_require_bitwise, rmsnorm_use_kahan, rmsnorm_recompute_rstd
    del context_id
    return (
        torch.empty_like(grad_output_TD),
        torch.empty_like(topk_scores_TK),
        w13_EFD.new_empty(w13_EFD.shape, dtype=wgrad_output_dtype),
        w2_EDF.new_empty(w2_EDF.shape, dtype=wgrad_output_dtype),
    )


@_block_scaled_backward_accumulate_op.register_fake
def _block_scaled_backward_accumulate_fake(
    grad_output_TD: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    accumulator_grad_w13_EFD: torch.Tensor,
    accumulator_grad_w2_EDF: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    state: list[torch.Tensor],
    rmsnorm_enabled: bool,
    rmsnorm_eps: float,
    rmsnorm_norm_output_dtype: torch.dtype,
    rmsnorm_output_dtype: torch.dtype,
    rmsnorm_require_bitwise: bool,
    rmsnorm_use_kahan: bool,
    rmsnorm_recompute_rstd: bool,
    wgrad_output_dtype: torch.dtype,
    context_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return non-WGRAD metadata for a fake accumulating backward."""
    del topk_expert_ids_TK, state, rmsnorm_enabled
    del rmsnorm_eps, rmsnorm_norm_output_dtype, rmsnorm_output_dtype
    del rmsnorm_require_bitwise, rmsnorm_use_kahan, rmsnorm_recompute_rstd
    del context_id
    _wgrad_accumulation_destinations(
        accumulator_grad_w13_EFD,
        accumulator_grad_w2_EDF,
        w13_EFD,
        w2_EDF,
        wgrad_output_dtype,
    )
    return torch.empty_like(grad_output_TD), torch.empty_like(topk_scores_TK)


_block_scaled_backward_op.register_effect(torch.library.EffectType.ORDERED)


def _block_scaled_setup_context(
    ctx: Any,
    inputs: tuple[object, ...],
    output: tuple[torch.Tensor, list[torch.Tensor]],
    *,
    mark_state_non_differentiable: bool = True,
) -> None:
    """Save resolved private state for the registered autograd formula.

    Args:
        ctx: Custom-op autograd context.
        inputs: Positional inputs passed to the forward operation.
        output: Forward output and its private fixed-shape state.
        mark_state_non_differentiable: Whether ``state`` is emitted by the
            current autograd boundary and should be marked.
    """
    (
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        w13_EFD,
        w2_EDF,
        prepared,
        _rmsnorm_weight_D,
        rmsnorm_enabled,
        rmsnorm_eps,
        rmsnorm_norm_output_dtype,
        rmsnorm_output_dtype,
        rmsnorm_require_bitwise,
        _rmsnorm_gain_center,
        rmsnorm_use_kahan,
        rmsnorm_recompute_rstd,
        _activation_slot_id_1,
        _num_moe_layers_in_slot,
        _save_for_backward,
        _validate_expert_ids,
        context_id,
        _interleaved_fc13,
    ) = inputs
    assert isinstance(w13_EFD, torch.Tensor) and isinstance(w2_EDF, torch.Tensor)
    assert isinstance(prepared, list)
    _result, state = output
    resolved = _resolve_block_scaled_weight_state(
        prepared,
        state[_BACKWARD_WEIGHT_STATE],
    )
    ctx.context_id = context_id
    ctx.rmsnorm_enabled = rmsnorm_enabled
    ctx.rmsnorm_eps = rmsnorm_eps
    ctx.rmsnorm_norm_output_dtype = rmsnorm_norm_output_dtype
    ctx.rmsnorm_output_dtype = rmsnorm_output_dtype
    ctx.rmsnorm_require_bitwise = rmsnorm_require_bitwise
    ctx.rmsnorm_use_kahan = rmsnorm_use_kahan
    ctx.rmsnorm_recompute_rstd = rmsnorm_recompute_rstd
    ctx.wgrad_output_dtype = w13_EFD.dtype
    ctx.inplace_wgrad_accum = False
    ctx.set_materialize_grads(False)
    if mark_state_non_differentiable:
        ctx.mark_non_differentiable(*state)
    ctx.save_for_backward(
        w13_EFD,
        w2_EDF,
        topk_expert_ids_TK,
        topk_scores_TK,
        *resolved,
        *state[_BACKWARD_NON_WEIGHT_STATE],
    )


def _resolve_block_scaled_weight_state(
    prepared: list[torch.Tensor | None],
    produced: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Resolve supplied or dynamically produced weight operands.

    Args:
        prepared: Optional caller-supplied operands for both weights.
        produced: Private forward outputs for dynamically prepared operands.

    Returns:
        Ten concrete tensors consumed by the backward operation.
    """
    prepared_weights = (
        prepared[:_PREPARED_WEIGHT_OPERAND_COUNT],
        prepared[_PREPARED_WEIGHT_OPERAND_COUNT:],
    )
    produced_weights = (
        produced[:_SAVED_WEIGHT_STATE_COUNT],
        produced[_SAVED_WEIGHT_STATE_COUNT:],
    )
    resolved: list[torch.Tensor] = []
    for supplied, generated in zip(
        prepared_weights,
        produced_weights,
        strict=True,
    ):
        (
            q_fprop,
            scale_fprop,
            q_dgrad,
            scale_dgrad,
            global_scale,
            global_scale_inv,
        ) = supplied
        if q_fprop is None:
            resolved.extend(generated)
            continue
        assert q_fprop is not None and scale_fprop is not None
        assert scale_dgrad is not None
        if global_scale_inv is None and global_scale is not None:
            global_scale_inv = torch.reciprocal(global_scale)
        resolved.extend(
            (
                q_fprop,
                scale_fprop,
                q_fprop if q_dgrad is None else q_dgrad,
                scale_dgrad,
                global_scale_inv
                if global_scale_inv is not None
                else _empty_private_state(q_fprop),
            )
        )
    return resolved


def _block_scaled_autograd_backward(
    ctx: Any,
    grad_output_TD: torch.Tensor,
    unused_state_gradients: list[torch.Tensor | None] | None,
) -> tuple[torch.Tensor | None, ...]:
    """Invoke the asynchronous backward custom operation.

    Args:
        ctx: Custom-op autograd context populated by setup.
        grad_output_TD: Gradient of the local expert output.
        unused_state_gradients: Gradients for non-differentiable private state.

    Returns:
        Gradients aligned with every registered forward input.
    """
    del unused_state_gradients
    (
        w13_EFD,
        w2_EDF,
        topk_expert_ids_TK,
        topk_scores_TK,
        *state,
    ) = ctx.saved_tensors
    context = _get_context(ctx.context_id)
    wgrad_destinations = None
    if ctx.inplace_wgrad_accum:
        wgrad_destinations = _resolve_parameter_grad_destinations(
            (ctx.w13_param_ref, ctx.w2_param_ref),
            ctx.w13_compute_shape,
            ctx.w2_compute_shape,
            context.config.wgrad_dtype,
        )
        wgrad_output_dtype = wgrad_destinations[0].dtype
    else:
        wgrad_output_dtype = context.config.wgrad_dtype or ctx.wgrad_output_dtype
    registered_args = (
        topk_expert_ids_TK,
        topk_scores_TK,
        list(state),
        ctx.rmsnorm_enabled,
        ctx.rmsnorm_eps,
        ctx.rmsnorm_norm_output_dtype,
        ctx.rmsnorm_output_dtype,
        ctx.rmsnorm_require_bitwise,
        ctx.rmsnorm_use_kahan,
        ctx.rmsnorm_recompute_rstd,
        wgrad_output_dtype,
        ctx.context_id,
    )
    if wgrad_destinations is not None and all(
        destination.accumulate for destination in wgrad_destinations
    ):
        accumulator_grad_w13_EFD = wgrad_destinations[0].output
        accumulator_grad_w2_EDF = wgrad_destinations[1].output
        assert accumulator_grad_w13_EFD is not None
        assert accumulator_grad_w2_EDF is not None
        grad_x_TD, grad_topk_scores_TK = _block_scaled_backward_accumulate_op(
            grad_output_TD,
            w13_EFD,
            w2_EDF,
            accumulator_grad_w13_EFD,
            accumulator_grad_w2_EDF,
            *registered_args,
        )
        grad_w13_EFD = grad_w2_EDF = None
    else:
        grad_x_TD, grad_topk_scores_TK, grad_w13_EFD, grad_w2_EDF = (
            _block_scaled_backward_op(
                grad_output_TD,
                w13_EFD,
                w2_EDF,
                *registered_args,
            )
        )
    return (
        grad_x_TD,
        None,  # topk_expert_ids_TK
        grad_topk_scores_TK,
        grad_w13_EFD,
        grad_w2_EDF,
        None,  # prepared
        None,  # rmsnorm_weight_D
        None,  # rmsnorm_enabled
        None,  # rmsnorm_eps
        None,  # rmsnorm_norm_output_dtype
        None,  # rmsnorm_output_dtype
        None,  # rmsnorm_require_bitwise
        None,  # rmsnorm_gain_center
        None,  # rmsnorm_use_kahan
        None,  # rmsnorm_recompute_rstd
        None,  # activation_slot_id_1
        None,  # num_moe_layers_in_slot
        None,  # save_for_backward
        None,  # validate_expert_ids
        None,  # context_id
        None,  # interleaved_fc13
    )


_block_scaled_forward_op.register_autograd(
    _block_scaled_autograd_backward,
    setup_context=_block_scaled_setup_context,
)


@torch._dynamo.allow_in_graph
class _RegisteredBlockScaledAutograd(torch.autograd.Function):
    """Attach registered execution to the logical BF16 gradient owners.

    Forward uses either caller-prepared operands or registered dynamic
    quantization while the BF16 weights remain the autograd edges. Backward
    resolves standard gradient ownership in Python, then invokes either the
    functional backward or its deterministic accumulating counterpart. Both
    kernel paths remain opaque to FakeTensor tracing.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x_TD: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        w13_EFD: torch.Tensor,
        w2_EDF: torch.Tensor,
        prepared: list[torch.Tensor | None],
        rmsnorm_weight_D: torch.Tensor | None,
        rmsnorm_enabled: bool,
        rmsnorm_eps: float,
        rmsnorm_norm_output_dtype: torch.dtype,
        rmsnorm_output_dtype: torch.dtype,
        rmsnorm_require_bitwise: bool,
        rmsnorm_gain_center: float,
        rmsnorm_use_kahan: bool,
        rmsnorm_recompute_rstd: bool,
        activation_slot_id_1: torch.Tensor,
        num_moe_layers_in_slot: int,
        context_id: str,
        options: ExecutionOptions,
    ) -> torch.Tensor:
        """Run registered forward while retaining logical gradient owners.

        Args:
            ctx: Autograd context for backward state.
            x_TD: Local input activations.
            topk_expert_ids_TK: Routed expert IDs.
            topk_scores_TK: Routed expert weights.
            w13_EFD: Logical gate/up weight used only as a gradient owner.
            w2_EDF: Logical down weight used only as a gradient owner.
            prepared: Optional FPROP and DGRAD operands for both weights.
            rmsnorm_weight_D: Optional inference-only input-scale gamma.
            rmsnorm_enabled: Whether fused post-expert RMSNorm is selected.
            rmsnorm_eps: RMSNorm epsilon.
            rmsnorm_norm_output_dtype: Normalized route-output dtype.
            rmsnorm_output_dtype: Final reduction output dtype.
            rmsnorm_require_bitwise: Whether to preserve reduction order.
            rmsnorm_gain_center: Constant added to the input-scale gamma.
            rmsnorm_use_kahan: Whether to use compensated sum-of-squares.
            rmsnorm_recompute_rstd: Whether backward recomputes reciprocal RMS values.
            activation_slot_id_1: Device scalar selecting the activation
                slot.
            num_moe_layers_in_slot: Static MoE-layer count for the selected slot.
            context_id: Registered DistMoE context identifier.
            options: Graph-compatible per-call execution controls.

        Returns:
            Local combined expert output.
        """
        w13_prepared = prepared[:_PREPARED_WEIGHT_OPERAND_COUNT]
        w2_prepared = prepared[_PREPARED_WEIGHT_OPERAND_COUNT:]
        q_w13, *_ = w13_prepared
        q_w2, *_ = w2_prepared
        compute_w13_EFD = w13_EFD if q_w13 is None else q_w13
        compute_w2_EDF = w2_EDF if q_w2 is None else q_w2
        forward_args = (
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            compute_w13_EFD,
            compute_w2_EDF,
            prepared,
            rmsnorm_weight_D,
            rmsnorm_enabled,
            rmsnorm_eps,
            rmsnorm_norm_output_dtype,
            rmsnorm_output_dtype,
            rmsnorm_require_bitwise,
            rmsnorm_gain_center,
            rmsnorm_use_kahan,
            rmsnorm_recompute_rstd,
            activation_slot_id_1,
            num_moe_layers_in_slot,
            True,
            options.validate_expert_ids,
            context_id,
            False,  # Training retains concatenated W13 and DGRAD operands.
        )
        output = _block_scaled_forward_op(*forward_args)
        _block_scaled_setup_context(
            ctx,
            forward_args,
            output,
            mark_state_non_differentiable=False,
        )
        # Prepared FP8 storage must not replace the logical BF16 WGRAD dtype.
        ctx.wgrad_output_dtype = w13_EFD.dtype
        ctx.inplace_wgrad_accum = options.inplace_wgrad_accum
        if options.inplace_wgrad_accum:
            wgrad_parameter_owners = options.wgrad_parameter_owners
            ctx.w13_param_ref = _weak_parameter_ref(
                w13_EFD,
                None if wgrad_parameter_owners is None else wgrad_parameter_owners[0],
            )
            ctx.w2_param_ref = _weak_parameter_ref(
                w2_EDF,
                None if wgrad_parameter_owners is None else wgrad_parameter_owners[1],
            )
            ctx.w13_compute_shape = w13_EFD.shape
            ctx.w2_compute_shape = w2_EDF.shape
        return output[0]

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        """Return activation, router-score, and logical-weight gradients.

        Args:
            ctx: Autograd context populated by ``forward``.
            *grad_outputs: The single combined expert-output gradient.

        Returns:
            Gradients aligned with the forward arguments.
        """
        (grad_output_TD,) = grad_outputs
        return _block_scaled_autograd_backward(ctx, grad_output_TD, None)


def _run_blockscaled(  # noqa: C901 -- validate and dispatch one public execution contract
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_weight: torch.Tensor | PreparedWeight,
    w2_weight: torch.Tensor | PreparedWeight,
    context: Context,
    options: ExecutionOptions,
    *,
    save_for_backward: bool,
) -> torch.Tensor:
    """Run the activation-buffer-backed asynchronous block-scaled backend.

    Args:
        x_TD: Local BF16 activations with shape ``[T, D]``.
        topk_expert_ids_TK: Routed expert IDs with shape ``[T, K]``.
        topk_scores_TK: Routing weights with shape ``[T, K]``.
        w13_weight: Logical or prepared gate/up expert weight.
        w2_weight: Logical or prepared down-projection expert weight.
        context: Reusable distributed execution context.
        options: Per-invocation callbacks and execution controls.
        save_for_backward: Whether this invocation has a backward consumer.

    Returns:
        Local combined expert output with shape ``[T, D]``.

    Raises:
        RuntimeError: If the context has no activation buffer.
        ValueError: If block-scaled execution is invalid.
    """
    policy = context.config.block_scaled
    if policy is None:
        raise ValueError("block-scaled execution requires a block_scaled policy")
    activation_buffer = context.activation_buffer
    if activation_buffer is None:
        raise RuntimeError("block-scaled execution requires an activation buffer")
    from ._execution import PreparedWeight

    prepared_w13 = w13_weight if isinstance(w13_weight, PreparedWeight) else None
    prepared_w2 = w2_weight if isinstance(w2_weight, PreparedWeight) else None
    interleaved_fc13 = prepared_w13 is not None and prepared_w13._interleaved_fc13
    if prepared_w2 is not None and prepared_w2._interleaved_fc13:
        raise ValueError("interleaved prepared W13 cannot be used as W2")
    if interleaved_fc13 and (save_for_backward or not context.config.inference):
        raise ValueError("interleaved W13 requires an inference-only context")
    if interleaved_fc13 and policy.pipeline != "mega":
        raise ValueError("interleaved W13 requires the Mega inference pipeline")
    source_w13_EFD = prepared_w13.source if prepared_w13 is not None else w13_weight
    source_w2_EDF = prepared_w2.source if prepared_w2 is not None else w2_weight
    assert isinstance(source_w13_EFD, torch.Tensor)
    assert isinstance(source_w2_EDF, torch.Tensor)
    source_w13_EFD, source_w2_EDF = _reshape_weights(
        source_w13_EFD,
        source_w2_EDF,
        context,
    )

    def prepared_tensors(
        weight: PreparedWeight | None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Flatten one optional public prepared-weight object.

        Args:
            weight: Optional prepared grouped weight.

        Returns:
            Six optional kernel operands: FPROP data/scales, optional DGRAD
            data/scales, and optional global scale/inverse.
        """
        if weight is None:
            return (None, None, None, None, None, None)
        return (
            weight.fprop_data,
            weight.fprop_scale,
            weight.dgrad_data,
            weight.dgrad_scale,
            weight.global_scale,
            weight.global_scale_inv,
        )

    prepared = [*prepared_tensors(prepared_w13), *prepared_tensors(prepared_w2)]
    tracing = torch.compiler.is_compiling() or _get_current_dispatch_mode() is not None
    if not options.requires_eager:
        rmsnorm_args = _registered_rmsnorm_args(options.experts_output_postprocess)
        if save_for_backward:
            return _RegisteredBlockScaledAutograd.apply(
                x_TD,
                topk_expert_ids_TK,
                topk_scores_TK,
                source_w13_EFD,
                source_w2_EDF,
                prepared,
                *rmsnorm_args,
                activation_buffer.activation_slot_id_1,
                activation_buffer._num_moe_layers_in_selected_slot,
                context.context_id,
                options,
            )
        # Like the autograd path, pass the FPROP data instead of the source:
        # without a backward the source is only an identity and may hold no
        # readable storage (e.g. an FSDP unsharded weight that keeps only the
        # quantized operands).
        output_TD, _state = _block_scaled_forward_op(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            source_w13_EFD if prepared_w13 is None else prepared_w13.fprop_data,
            source_w2_EDF if prepared_w2 is None else prepared_w2.fprop_data,
            prepared,
            *rmsnorm_args,
            activation_buffer.activation_slot_id_1,
            activation_buffer._num_moe_layers_in_selected_slot,
            save_for_backward,
            options.validate_expert_ids,
            context.context_id,
            interleaved_fc13,
        )
        return output_TD
    if tracing:
        raise RuntimeError(
            "DistMoE Python weight and WGRAD callbacks require eager execution"
        )

    def weight_spec(
        prepared_weight: PreparedWeight | None,
        source_weight: torch.Tensor,
    ) -> BlockscaledWeightSpec:
        """Return the typed eager operands for one logical weight.

        Args:
            prepared_weight: Optional caller-owned quantized representations.
            source_weight: Reshaped logical weight that owns the gradient.

        Returns:
            Native and optional quantized tensors consumed by eager execution.
        """
        if prepared_weight is None:
            return BlockscaledWeightSpec(w_native=source_weight)
        return BlockscaledWeightSpec(
            w_native=source_weight,
            w_q_fprop=prepared_weight.fprop_data,
            w_scale_fprop=prepared_weight.fprop_scale,
            w_q_dgrad=prepared_weight.dgrad_data,
            w_scale_dgrad=prepared_weight.dgrad_scale,
            w_global_scale=prepared_weight.global_scale,
            w_global_scale_inv=prepared_weight.global_scale_inv,
        )

    blockscaled_cfg, kernel_config = _blockscaled_compute_config(context)
    blockscaled_cfg = replace(
        blockscaled_cfg,
        interleaved_fc13=interleaved_fc13,
        routing_capacity=(
            context.memory_plan.total_scratch_capacity_rows
            if context.config.inference and context.memory_plan is not None
            else None
        ),
    )
    return _run_blockscaled_eager(
        x_TD=x_TD,
        topk_expert_ids_TK=topk_expert_ids_TK,
        topk_scores_TK=topk_scores_TK,
        w13_spec=weight_spec(prepared_w13, source_w13_EFD),
        w2_spec=weight_spec(prepared_w2, source_w2_EDF),
        comm_group=context.group,
        comm_buffer=context.buffers,
        activation_buffer=activation_buffer,
        max_imbalance_factor=context.config.device_scratch_capacity_factor,
        save_for_backward=save_for_backward,
        weights_preprocess_fn=options.weights_preprocess_fn,
        experts_postprocess_fn=options.experts_output_postprocess,
        num_sms=context.config.num_sms,
        inplace_wgrad_accum=options.inplace_wgrad_accum,
        wgrad_parameter_owners=options.wgrad_parameter_owners,
        wgrad_destination_fn=options.wgrad_destination_fn,
        wgrad_output_dtype=context.config.wgrad_dtype,
        wgrad_postprocess_fn=options.wgrad_postprocess_fn,
        blockscaled_cfg=blockscaled_cfg,
        config=kernel_config,
        activation=context.config.activation,
        swiglu_alpha=context.config.swiglu_alpha,
        swiglu_limit=context.config.swiglu_limit,
        activation_slot_id_1=activation_buffer.activation_slot_id_1,
        num_moe_layers_in_slot=activation_buffer._num_moe_layers_in_selected_slot,
        validate_expert_ids=options.validate_expert_ids,
        zero_out_padded_callback_inputs=(options.zero_out_padded_callback_inputs),
    )
