# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public CuTe DSL distributed MoE API.

Routing logits and top-k selection remain model responsibilities. This module
consumes precomputed expert IDs and scores, owns the communication and
activation buffers, and exposes one graph-visible operation.
"""

from __future__ import annotations

import dataclasses
import gc
import itertools
import logging
import math
import threading
from typing import Any, get_args, Literal, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.profiler import record_function
from torch.utils._python_dispatch import _get_current_dispatch_mode

from ._buffers import (
    _CommunicationBuffers,
    _initialize_fake_peer_scatter_output,
    _routing_ids_view,
    _routing_token_count_view,
    initialize_multimem_barrier_workspace,
    is_fake_process_group,
)
from ._context import (
    _get_context,
    _register_context,
    _reshape_weights,
    _unregister_context,
)
from ._execution import (
    _KernelAutogradContext,
    _postprocess_wgrad,
    _resolve_execution_options,
    _resolve_parameter_grad_destinations,
    _resolve_wgrad_destinations,
    _resolve_wgrad_output_dtype,
    _weak_parameter_ref,
    _wgrad_accumulation_destinations,
    _WgradDestination,
    BlockScaledFormat,
    ExecutionOptions,
    PreparedWeight,
)
from ._postprocess import (
    _registered_rmsnorm_args,
    _resolve_experts_output_postprocess_fn,
    _rmsnorm_from_registered_args,
    RMSNormPostprocess,
    supports_fused_post_expert_rmsnorm,
)
from .formats import (
    _kernel_block_scaled_format,
    _NVFP4_DIM_MULTIPLE,
    _NVFP4_HIDDEN_DIM_MAX,
    _NVFP4_INTERMEDIATE_DIM_MAX,
    BLOCK_MAJOR_K,
    block_scaled_format_constants,
    SWIGLU_CLAMP_ALPHA_DEFAULT,
    SWIGLU_CLAMP_LIMIT_DEFAULT,
)

if TYPE_CHECKING:
    from ._activation_buffer import (
        ActivationBuffer,
        ModelConfig,
    )


__all__ = [
    "Bf16GroupedGemmPreset",
    "BlockScaledConfig",
    "BlockScaledFormat",
    "BlockScaledKernelConfig",
    "Config",
    "Context",
    "ExecutionOptions",
    "MemoryPlan",
    "PreparedWeight",
    "VmmConfig",
    "create_context",
    "plan_memory",
    "prepare_block_scaled_weight",
    "relayout_to_block_major",
    "RMSNormPostprocess",
    "routed_experts",
    "supports_fused_post_expert_rmsnorm",
]


logger = logging.getLogger(__name__)
_MEMORY_ALIGNMENT = 128
_CUDA_LARGE_POOL_MIN_ALLOC_BYTES = 10 * 1024**2
_CUDA_LARGE_POOL_BUFFER_BYTES = 20 * 1024**2

Bf16GroupedGemmPreset = Literal[
    "1cta1mma_bm64_bn128",
    "1cta1mma_bm64_bn256",
    "1cta1mma_bm128_bn128",
    "1cta1mma_bm128_bn256",
    "1cta2mma_bm256_bn256",
    "2cta1mma_bm256_bn256",
    "2cta2mma_bm512_bn256",
]
_BF16_GROUPED_GEMM_CONFIGS = frozenset(get_args(Bf16GroupedGemmPreset))


@dataclasses.dataclass(frozen=True)
class VmmConfig:
    """Host-backed scratch policy for a distributed MoE context.

    Args:
        total_scratch_capacity_factor: Maximum routing imbalance covered by
            device and host scratch together. Runtime planning clamps this
            value to the expert-parallel group size, which is the largest
            possible imbalance.
        prefetch: Whether context construction should prepare the VMM physical
            allocation concurrently with communication-buffer initialization.
            ``False`` performs the same allocation synchronously after the
            communication buffers are ready. It changes initialization overlap,
            not the buffer layout or execution semantics.
    """

    total_scratch_capacity_factor: float = 16.0
    prefetch: bool = True

    def __post_init__(self) -> None:
        """Validate the host-scratch policy.

        Raises:
            ValueError: If the capacity factor is not positive.
            TypeError: If ``prefetch`` is not a bool.
        """
        if (
            not math.isfinite(self.total_scratch_capacity_factor)
            or self.total_scratch_capacity_factor <= 0
        ):
            raise ValueError(
                "total_scratch_capacity_factor must be finite and positive"
            )
        if not isinstance(self.prefetch, bool):
            raise TypeError("prefetch must be a bool")


@dataclasses.dataclass(frozen=True)
class BlockScaledKernelConfig:
    """Expert override for a final block-scaled CuTe pipeline.

    Omitting this object selects the original shape-aware production presets.
    The same complete override is forwarded to the selected staged or Mega
    MXFP8 or NVFP4 pipeline. All fields correspond directly to compile-time
    CuTe kernel parameters; this class validates structural invariants, not
    device resource limits or performance. Unsupported tile/resource
    combinations can still be rejected by CuTe JIT compilation, so ordinary
    users should leave ``kernel_config=None``.

    Args:
        num_ctas: Cooperating CTAs per cluster.
        block_m: Grouped M tile extent.
        block_n: Output N tile extent.
        block_k: Reduction K tile extent.
        num_smem_buffers: A/B/scale pipeline depth.
        num_c_stages: Output staging depth.
        num_tmem_buffers: TMEM accumulator slabs.
        num_tile_buffers: Scheduler ring slots.
        epilogue_subtile: Number of output N partitions.
        overlapping_accum: Whether accumulator slabs share the TMEM seam.
        swap_ab: Whether MMA operand roles are exchanged.
        kloop_unroll: Reduction-loop unroll factor.
        num_warps: Warps in the local grouped-GEMM CTA.
        static_scheduler: Whether tile assignment is static.
    """

    num_ctas: int
    block_m: int
    block_n: int
    block_k: int
    num_smem_buffers: int
    num_c_stages: int
    num_tmem_buffers: int
    num_tile_buffers: int
    epilogue_subtile: int
    overlapping_accum: bool
    swap_ab: bool
    kloop_unroll: int
    num_warps: int = 8
    static_scheduler: bool = False

    def __post_init__(self) -> None:
        """Validate structural kernel invariants.

        Raises:
            ValueError: If a pipeline or tile dimension is invalid.
        """
        positive = (
            self.num_ctas,
            self.block_m,
            self.block_n,
            self.block_k,
            self.num_smem_buffers,
            self.num_c_stages,
            self.num_tmem_buffers,
            self.num_tile_buffers,
            self.epilogue_subtile,
            self.kloop_unroll,
            self.num_warps,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("block-scaled kernel dimensions must be positive")
        if self.num_ctas not in (1, 2):
            raise ValueError("num_ctas must be 1 or 2")
        if self.block_n % self.epilogue_subtile:
            raise ValueError("epilogue_subtile must divide block_n")

    def _as_kernel_kwargs(self) -> dict[str, int | bool]:
        """Return the final dictionary consumed by the CuTe JIT.

        Returns:
            Compile-time kernel parameter dictionary.
        """
        return {
            "NUM_CTAS": self.num_ctas,
            "NUM_MMAS": 1,
            "BLOCK_SIZE_M": self.block_m,
            "BLOCK_SIZE_N": self.block_n,
            "BLOCK_SIZE_K": self.block_k,
            "NUM_SMEM_BUFFERS": self.num_smem_buffers,
            "NUM_C_STAGES": self.num_c_stages,
            "NUM_TMEM_BUFFERS": self.num_tmem_buffers,
            "NUM_TILE_BUFFERS": self.num_tile_buffers,
            "EPILOGUE_SUBTILE": self.epilogue_subtile,
            "OVERLAPPING_ACCUM": self.overlapping_accum,
            "SWAP_AB": self.swap_ab,
            "KLOOP_UNROLL": self.kloop_unroll,
            "NUM_WARPS": self.num_warps,
            "STATIC_SCHEDULER": self.static_scheduler,
        }


@dataclasses.dataclass(frozen=True)
class BlockScaledConfig:
    """CuTe block-scaled compute policy for a distributed MoE context.

    Args:
        format: MXFP8 E4M3 training/inference or NVIDIA FP4 inference operands.
        fast_math: Whether MXFP8 fused SwiGLU uses approximate sigmoid math.
            NVFP4 inference does not support this option.
        pipeline: ``"staged"`` for separate expert GEMMs or ``"mega"`` for
            the fused chunk-pipelined forward. MXFP8 Mega also fuses each
            DGRAD/WGRAD pair; NVFP4 remains forward-only.
        kernel_config: Optional expert pipeline override; ``None`` selects the
            shape-aware defaults.
    """

    format: BlockScaledFormat = BlockScaledFormat.MXFP8_E4M3
    fast_math: bool = False
    pipeline: Literal["staged", "mega"] = "staged"
    kernel_config: BlockScaledKernelConfig | None = None

    def __post_init__(self) -> None:
        """Validate the supported public precision surface.

        Raises:
            ValueError: If the format or fast-math combination is unsupported.
        """
        supported = {
            BlockScaledFormat.MXFP8_E4M3,
            BlockScaledFormat.NVFP4,
        }
        if self.format not in supported:
            raise ValueError(f"unsupported DistMoE block-scaled format {self.format}")
        if self.fast_math and self.format is BlockScaledFormat.NVFP4:
            raise ValueError("fast_math is supported only by the MXFP8 fused path")
        if self.pipeline not in ("staged", "mega"):
            raise ValueError("block-scaled pipeline must be 'staged' or 'mega'")
        if self.kernel_config is not None:
            expected_k = 128
            if self.kernel_config.block_k != expected_k:
                raise ValueError(
                    f"{self.format.value} requires block_k={expected_k}, "
                    f"got {self.kernel_config.block_k}"
                )


def _validate_block_scaled_config(config: Config) -> None:
    """Validate shape and memory controls for block-scaled execution.

    Args:
        config: Public configuration containing a block-scaled policy.

    Raises:
        ValueError: If the format or dimensions are unsupported.
    """
    policy = config.block_scaled
    assert policy is not None
    _, _, vector_size = block_scaled_format_constants(
        _kernel_block_scaled_format(policy.format)
    )
    if config.hidden_dim % vector_size or config.intermediate_dim % vector_size:
        raise ValueError(
            "block-scaled hidden_dim and intermediate_dim must be divisible "
            f"by the format vector size {vector_size}"
        )
    if policy.format is BlockScaledFormat.NVFP4:
        if not config.inference:
            raise ValueError("NVFP4 DistMoE is inference-only")
        if (
            config.hidden_dim % _NVFP4_DIM_MULTIPLE
            or config.intermediate_dim % _NVFP4_DIM_MULTIPLE
            or config.hidden_dim > _NVFP4_HIDDEN_DIM_MAX
            or config.intermediate_dim > _NVFP4_INTERMEDIATE_DIM_MAX
        ):
            raise ValueError(
                "NVFP4 requires hidden_dim to be a multiple of "
                f"{_NVFP4_DIM_MULTIPLE} and at most "
                f"{_NVFP4_HIDDEN_DIM_MAX}, and intermediate_dim to be "
                f"a multiple of {_NVFP4_DIM_MULTIPLE} and at most "
                f"{_NVFP4_INTERMEDIATE_DIM_MAX}"
            )


@dataclasses.dataclass(frozen=True)
class Config:
    """Static execution configuration for a distributed MoE context.

    Args:
        max_num_local_input_tokens: Maximum physical input-token count on every
            EP rank before top-k expansion. Each invocation may use any shared
            rank-local count ``T`` in ``[1, max_num_local_input_tokens]``.
        hidden_dim: Model hidden dimension.
        intermediate_dim: Per-expert SwiGLU intermediate dimension.
        top_k: Number of experts selected per input token.
        num_experts: Total number of experts across the expert-parallel group.
        max_moe_layers_per_activation_slot: Maximum local MoE-layer depth that
            shares one selected activation slot. Use all local MoE layers for
            microbatch slots or the largest local-stage depth for
            stage-microbatch slots.
        device_scratch_capacity_factor: Maximum routing imbalance that should fit
            entirely in device-resident scratch. A value of 1.0 represents a
            balanced ``num_tokens * top_k`` receive count. Values above the
            expert-parallel size are topology-equivalent and are capped with a
            warning during planning.
        activation_slot_bytes: Requested saved-forward-state bytes per activation
            slot, excluding shared scratch. The planner resolves its private
            alignment internally. ``None`` selects either the capacity-factor
            policy or, when that is also absent, the minimum all-recompute size.
        activation_slot_capacity_factor: Optional saved-state capacity relative
            to balanced routing. The factor scales only the optional bytes above
            mandatory layer inputs. ``1.0`` retains every eligible intermediate
            when aggregate slot usage matches balanced routing. It is mutually
            exclusive with ``activation_slot_bytes``.
        num_activation_slots: Number of concurrently live saved-activation
            slots, normally derived by coloring schedule liveness intervals.
        vmm: Optional host-backed overflow-scratch configuration.
        num_sms: Optional number of SMs assigned to each CuTe launch.
        bf16_grouped_gemm_preset: Optional explicit BF16 grouped-GEMM preset.
            During training, ``None`` selects the production FPROP/DGRAD and
            WGRAD presets. During inference it leaves kernel selection to the
            shape-aware launcher.
        block_scaled: Optional MXFP8/NVFP4 compute policy. ``None`` selects BF16.
        activation: SwiGLU variant. ``"swiglu_clamped"`` selects
            ``min(gate, limit) * sigmoid(alpha * min(gate, limit)) *
            (clamp(up, -limit, limit) + 1)``.
        swiglu_alpha: Sigmoid multiplier for clamped SwiGLU.
        swiglu_limit: Gate and up-projection bound for clamped SwiGLU.
        wgrad_dtype: Optional explicit BF16 or FP32 weight-gradient output
            dtype. ``None`` uses each parameter's non-``None`` ``grad_dtype``
            for direct in-place accumulation, then existing gradient storage or
            the parameter dtype. Ordinary functional WGRAD falls back to the
            compute-weight dtype.
        inference: Whether the context is permanently specialized for
            inference routing, layouts, formats, and scratch-only planning.
            Per-call `torch.no_grad()` evaluation on a training context is
            supported independently and does not consume activation slots.
        block_major_weights: Whether every W13 and W2 passed to this context
            was rewritten by :func:`relayout_to_block_major`, which streams
            faster. BF16 inference only (training backward reads row-major
            weights); ``intermediate_dim`` must be a multiple of 64. Weight
            callbacks receive the stored block-major tensors.
    """

    max_num_local_input_tokens: int
    hidden_dim: int
    intermediate_dim: int
    top_k: int
    num_experts: int
    max_moe_layers_per_activation_slot: int
    device_scratch_capacity_factor: float = 1.0
    activation_slot_bytes: int | None = None
    activation_slot_capacity_factor: float | None = None
    num_activation_slots: int = 1
    vmm: VmmConfig | None = None
    num_sms: int | None = None
    bf16_grouped_gemm_preset: Bf16GroupedGemmPreset | None = None
    block_scaled: BlockScaledConfig | None = None
    activation: Literal["swiglu", "swiglu_clamped"] = "swiglu"
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT
    wgrad_dtype: torch.dtype | None = None
    inference: bool = False
    block_major_weights: bool = False

    def __post_init__(self) -> None:  # noqa: C901
        """Validate static execution parameters.

        Raises:
            ValueError: If a numeric configuration value is invalid.
            TypeError: If non-``None`` ``wgrad_dtype`` is not BF16 or FP32.
        """
        if (
            min(
                self.max_num_local_input_tokens,
                self.hidden_dim,
                self.intermediate_dim,
                self.top_k,
            )
            <= 0
        ):
            raise ValueError(
                "max_num_local_input_tokens, hidden_dim, intermediate_dim, "
                "and top_k must be positive"
            )
        if self.hidden_dim % 64 != 0:
            raise ValueError("hidden_dim must be a multiple of 64 BF16 elements")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.max_moe_layers_per_activation_slot <= 0:
            raise ValueError("max_moe_layers_per_activation_slot must be positive")
        if self.activation_slot_bytes is not None:
            if isinstance(self.activation_slot_bytes, bool) or not isinstance(
                self.activation_slot_bytes, int
            ):
                raise TypeError("activation_slot_bytes must be an integer")
            if self.activation_slot_bytes < 0:
                raise ValueError("activation_slot_bytes cannot be negative")
        if self.activation_slot_capacity_factor is not None:
            if isinstance(self.activation_slot_capacity_factor, bool) or not isinstance(
                self.activation_slot_capacity_factor, (int, float)
            ):
                raise TypeError("activation_slot_capacity_factor must be a real number")
            if (
                not math.isfinite(self.activation_slot_capacity_factor)
                or self.activation_slot_capacity_factor < 0
            ):
                raise ValueError(
                    "activation_slot_capacity_factor must be finite and nonnegative"
                )
        if (
            self.activation_slot_bytes is not None
            and self.activation_slot_capacity_factor is not None
        ):
            raise ValueError(
                "activation_slot_bytes and activation_slot_capacity_factor "
                "are mutually exclusive"
            )
        if not self.inference and self.num_activation_slots <= 0:
            raise ValueError("training requires at least one activation slot")
        if self.inference and self.num_activation_slots < 0:
            raise ValueError("num_activation_slots cannot be negative")
        if self.inference and (
            self.activation_slot_bytes is not None
            or self.activation_slot_capacity_factor is not None
        ):
            raise ValueError(
                "inference does not retain saved activations; activation slot "
                "capacity controls must be None"
            )
        if (
            not math.isfinite(self.device_scratch_capacity_factor)
            or self.device_scratch_capacity_factor <= 0
        ):
            raise ValueError(
                "device_scratch_capacity_factor must be finite and positive"
            )
        if self.num_sms is not None and self.num_sms <= 0:
            raise ValueError("num_sms must be positive when specified")
        if self.block_major_weights and (
            not self.inference
            or self.block_scaled is not None
            or self.intermediate_dim % BLOCK_MAJOR_K != 0
        ):
            raise ValueError(
                "block_major_weights requires BF16 inference and an "
                "intermediate_dim that is a multiple of 64"
            )
        if (
            self.bf16_grouped_gemm_preset is not None
            and self.bf16_grouped_gemm_preset not in _BF16_GROUPED_GEMM_CONFIGS
        ):
            raise ValueError(
                "bf16_grouped_gemm_preset must name a supported BF16 preset; "
                f"got {self.bf16_grouped_gemm_preset!r}"
            )
        if self.wgrad_dtype is not None and self.wgrad_dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise TypeError(
                "wgrad_dtype must be None, torch.bfloat16, or torch.float32, got "
                f"{self.wgrad_dtype}"
            )
        if self.activation not in ("swiglu", "swiglu_clamped"):
            raise ValueError(
                "activation must be 'swiglu' or 'swiglu_clamped', got "
                f"{self.activation!r}"
            )
        if self.activation == "swiglu_clamped" and (
            not math.isfinite(self.swiglu_alpha)
            or not math.isfinite(self.swiglu_limit)
            or self.swiglu_alpha <= 0
            or self.swiglu_limit <= 0
        ):
            raise ValueError(
                "clamped SwiGLU requires finite positive swiglu_alpha and swiglu_limit"
            )
        if self.block_scaled is not None:
            _validate_block_scaled_config(self)


@dataclasses.dataclass(frozen=True)
class MemoryPlan:
    """Resolved device and host allocation sizes for distributed MoE.

    All fields are byte counts except the routing row counts and imbalance
    factors. The plan is immutable so allocation and prefetch consume exactly
    the same layout.

    Args:
        balanced_recv_rows: Balanced number of routed rows received per rank.
        device_scratch_capacity_rows: Rows covered by device-resident scratch.
        total_scratch_capacity_rows: Rows covered by device and host scratch together.
        device_scratch_capacity_factor: Configured device scratch capacity
            relative to balanced routed rows.
        total_scratch_capacity_factor: Combined device-plus-host scratch
            capacity relative to balanced routed rows.
        saved_input_bytes_per_layer: Bytes saved per recomputed layer.
        device_scratch_bytes: Device-resident hot scratch bytes.
        host_scratch_bytes: Host-resident overflow scratch bytes.
        minimum_activation_slot_bytes: Smallest correct all-recompute size of
            one activation slot.
        balanced_full_save_activation_slot_bytes: Slot bytes required to retain
            all eligible intermediates under balanced routing.
        maximum_useful_activation_slot_bytes: Slot bytes above which no
            additional executable routing state can avoid recomputation.
        activation_slot_bytes: Effective internally aligned bytes in one slot.
        activation_slot_capacity_factor: Selected balanced saved-state factor,
            or ``None`` for minimum and exact-byte policies.
        total_activation_bytes: Aggregate bytes across all activation slots.
        total_device_buffer_bytes: Saved activations plus mandatory hot scratch.
        total_virtual_bytes: Size of the contiguous virtual address range.
        num_activation_slots: Effective number of saved-activation slots.
        inference: Whether the plan uses scratch-only inference layout.
        vmm_granularity_bytes: CUDA physical-section granularity when resolved
            for a device, otherwise ``None``.
        vmm_device_prefix_bytes: Mapped device prefix containing saved
            activations and any alignment padding.
        vmm_host_section_bytes: Mapped host-overflow section size.
        vmm_device_scratch_section_bytes: Mapped hot device-scratch size.
        vmm_padding_bytes: Physical bytes added for VMM alignment and PyTorch
            caching-allocator block rounding.
        inference_operand_bytes: Device-resolved physical Mega inference
            operand requirement. ``None`` for VMM, non-Mega execution, or a
            plan without a resolved device/SM count.
    """

    balanced_recv_rows: int
    device_scratch_capacity_rows: int
    total_scratch_capacity_rows: int
    device_scratch_capacity_factor: float
    total_scratch_capacity_factor: float
    saved_input_bytes_per_layer: int
    device_scratch_bytes: int
    host_scratch_bytes: int
    minimum_activation_slot_bytes: int
    balanced_full_save_activation_slot_bytes: int
    maximum_useful_activation_slot_bytes: int
    activation_slot_bytes: int
    activation_slot_capacity_factor: float | None
    total_activation_bytes: int
    total_device_buffer_bytes: int
    total_virtual_bytes: int
    num_activation_slots: int
    inference: bool
    vmm_granularity_bytes: int | None
    vmm_device_prefix_bytes: int | None
    vmm_host_section_bytes: int | None
    vmm_device_scratch_section_bytes: int | None
    vmm_padding_bytes: int
    inference_operand_bytes: int | None = None

    @property
    def uses_host_scratch(self) -> bool:
        """Return whether the plan contains host-backed scratch.

        Returns:
            Whether the resolved host-scratch section is non-empty.
        """
        return self.host_scratch_bytes > 0

    def explain(self) -> str:
        """Return a concise, log-friendly explanation of the resolved layout.

        Returns:
            Multi-line description of routing capacity and memory placement.
        """
        gib = 1024**3

        def _size(num_bytes: int) -> str:
            """Format a byte count for memory-plan diagnostics.

            Args:
                num_bytes: Size in bytes.

            Returns:
                Human-readable GiB and exact-byte representation.
            """
            return f"{num_bytes / gib:.3f} GiB ({num_bytes:,} bytes)"

        if self.inference:
            device_prefix = "  activation slots: 0 slot(s)"
            recompute = "scratch-only inference; no saved activations"
            capacity_boundary = "  inference has no activation slots; total scratch capacity is a hard limit"
        else:
            if self.activation_slot_capacity_factor is not None:
                policy = (
                    f"balanced capacity factor {self.activation_slot_capacity_factor:g}"
                )
            elif self.activation_slot_bytes == self.minimum_activation_slot_bytes:
                policy = "minimum all-recompute"
            else:
                policy = "exact per-slot bytes"
            device_prefix = (
                f"  activation slots: {self.num_activation_slots} x "
                f"{_size(self.activation_slot_bytes)} = "
                f"{_size(self.total_activation_bytes)} ({policy})"
            )
            if self.activation_slot_bytes == self.minimum_activation_slot_bytes:
                recompute = "all layers recompute at the minimum budget"
            elif (
                self.activation_slot_bytes >= self.maximum_useful_activation_slot_bytes
            ):
                recompute = "all executable routing fits without recompute"
            elif (
                self.activation_slot_bytes
                >= self.balanced_full_save_activation_slot_bytes
            ):
                recompute = (
                    "balanced routing fits without recompute; larger saved-state "
                    "usage may recompute"
                )
            else:
                recompute = "the activation budget may avoid recompute for some layers"
            capacity_boundary = (
                "  activation slots are a soft limit (recompute on exhaustion); "
                "total scratch capacity is a hard limit"
            )
        lines = [
            "Distributed MoE memory plan:",
            f"  balanced/device/total rows: {self.balanced_recv_rows}/"
            f"{self.device_scratch_capacity_rows}/{self.total_scratch_capacity_rows}",
            f"  device/total scratch capacity factors: "
            f"{self.device_scratch_capacity_factor:g}/"
            f"{self.total_scratch_capacity_factor:g}",
            device_prefix,
            f"  device scratch: {_size(self.device_scratch_bytes)}",
            f"  host overflow scratch: {_size(self.host_scratch_bytes)}",
            f"  total device buffer: {_size(self.total_device_buffer_bytes)}",
            f"  activation slot minimum/balanced full-save/maximum useful: "
            f"{_size(self.minimum_activation_slot_bytes)} / "
            f"{_size(self.balanced_full_save_activation_slot_bytes)} / "
            f"{_size(self.maximum_useful_activation_slot_bytes)}",
            f"  virtual address range: {_size(self.total_virtual_bytes)}; {recompute}",
            capacity_boundary,
        ]
        if self.inference_operand_bytes is not None:
            lines.append(
                "  physical Mega inference operands: "
                f"{_size(self.inference_operand_bytes)} (within device scratch)"
            )
        if self.vmm_granularity_bytes is not None:
            assert self.vmm_device_prefix_bytes is not None
            assert self.vmm_host_section_bytes is not None
            assert self.vmm_device_scratch_section_bytes is not None
            lines.extend(
                (
                    f"  VMM section granularity: {_size(self.vmm_granularity_bytes)}",
                    f"  mapped device/host/device sections: "
                    f"{_size(self.vmm_device_prefix_bytes)} / "
                    f"{_size(self.vmm_host_section_bytes)} / "
                    f"{_size(self.vmm_device_scratch_section_bytes)}",
                    f"  VMM/allocator padding: {_size(self.vmm_padding_bytes)}",
                )
            )
        return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class _VmmPrefetch:
    """Own an asynchronously prefetched VMM region until it is consumed.

    Args:
        _handle: Low-level prefetch whose physical region is either consumed
            by buffer allocation or released during failure cleanup.
    """

    _handle: Any = dataclasses.field(repr=False, compare=False)

    @property
    def consumed(self) -> bool:
        """Return whether a context took ownership of the prefetched region.

        Returns:
            Whether ownership has transferred to a context.
        """
        return self._handle is not None and self._handle.consumed

    def close(self) -> None:
        """Release the prefetched region unless a context already owns it."""
        self._handle.close()

    def __del__(self) -> None:
        """Best-effort cleanup when explicit ownership was not released."""
        try:
            self.close()
        except Exception:
            pass


def _align_memory(size_bytes: int) -> int:
    """Round a byte count up to the activation planner alignment.

    Args:
        size_bytes: Unaligned byte count.

    Returns:
        Byte count aligned to the activation planner requirement.
    """
    return (size_bytes + _MEMORY_ALIGNMENT - 1) // _MEMORY_ALIGNMENT * _MEMORY_ALIGNMENT


def _align_to(size_bytes: int, alignment: int) -> int:
    """Round a positive section size to a device-specific alignment.

    Args:
        size_bytes: Requested physical section size.
        alignment: Required byte alignment.

    Returns:
        Aligned section size.
    """
    return (size_bytes + alignment - 1) // alignment * alignment


def _resolve_cuda_device(
    device: torch.device | str | int | None,
) -> torch.device:
    """Resolve a public device argument to an indexed CUDA device.

    Args:
        device: CUDA device specification, defaulting to the current device.

    Returns:
        Indexed CUDA device.

    Raises:
        ValueError: If the resolved device is not CUDA.
    """
    if device is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int):
        resolved = torch.device("cuda", device)
    else:
        resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"device must be CUDA, got {resolved}")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _model_config(
    config: Config,
    *,
    ep_size: int | None = None,
    num_local_input_tokens: int | None = None,
) -> ModelConfig:
    """Build the activation planner configuration for a public config.

    Args:
        config: Public distributed MoE configuration.
        ep_size: Expert-parallel group size. Required for topology-aware
            block-scaled capacity planning.
        num_local_input_tokens: Actual local tokens for one invocation. ``None``
            uses the configured maximum for static capacity planning.

    Returns:
        Activation planner model configuration.
    """
    from ._activation_buffer import BlockscaledStorageConfig, ModelConfig

    block_scaled_storage = None
    routing_m_multiple = None
    num_local_experts = 1 if ep_size is None else config.num_experts // ep_size
    if config.block_scaled is not None:
        if ep_size is None:
            raise ValueError("ep_size is required for block-scaled memory planning")
        operand_dtype, scale_dtype, sf_vec_size = block_scaled_format_constants(
            _kernel_block_scaled_format(config.block_scaled.format)
        )
        block_scaled_storage = BlockscaledStorageConfig(
            operand_element_size=operand_dtype.itemsize,
            scale_element_size=scale_dtype.itemsize,
            sf_vec_size=sf_vec_size,
            operand_values_per_storage_element=(
                2 if config.block_scaled.format is BlockScaledFormat.NVFP4 else 1
            ),
            row_global_scale_element_size=(
                4 if config.block_scaled.format is BlockScaledFormat.NVFP4 else 0
            ),
        )
        routing_m_multiple = 128

    return ModelConfig(
        dtype=torch.bfloat16,
        hidden_dim=config.hidden_dim,
        intermediate_dim=config.intermediate_dim,
        num_tokens=(
            config.max_num_local_input_tokens
            if num_local_input_tokens is None
            else num_local_input_tokens
        ),
        topk=config.top_k,
        max_imbalance_factor=config.device_scratch_capacity_factor,
        num_moe_layers=config.max_moe_layers_per_activation_slot,
        blockscaled_storage=block_scaled_storage,
        num_local_experts=num_local_experts,
        routing_world_size=ep_size,
        routing_m_multiple_of=routing_m_multiple,
        mega=config.block_scaled is not None and config.block_scaled.pipeline == "mega",
    )


def _bf16_compute_config(
    config: Config,
    *,
    wgrad: bool = False,
    routed_rows: int | None = None,
    num_groups: int = 1,
    combine_fprop: bool = False,
) -> str | None:
    """Resolve an explicit, inference-auto, or pinned training schedule.

    Args:
        config: Public distributed MoE configuration.
        wgrad: Whether the caller is launching WGRAD instead of FPROP/DGRAD.
        routed_rows: Host estimate ``T * top_k``; actual route counts remain
            on device and are never truncated to this estimate.
        num_groups: Local expert count used by the source's per-expert policy.
        combine_fprop: Whether this is W2 forward combine, whose source policy
            has a separate large-row tile threshold.

    Returns:
        Explicit or measured training config, or ``None`` for inference auto
        selection.
    """
    if config.bf16_grouped_gemm_preset is not None or config.inference:
        return config.bf16_grouped_gemm_preset
    from .kernels.config import grouped_gemm_training_config

    return grouped_gemm_training_config(
        wgrad=wgrad,
        routed_rows=routed_rows,
        num_groups=num_groups,
        combine_fprop=combine_fprop,
    )


def plan_memory(  # noqa: C901 -- resolve one coherent activation/scratch allocation
    config: Config,
    ep_size: int,
    *,
    device: torch.device | str | int | None = None,
) -> MemoryPlan:
    """Resolve activation, device-scratch, and host-scratch sizes.

    Args:
        config: Static distributed MoE configuration.
        ep_size: Number of expert-parallel ranks.
        device: Optional CUDA device used to resolve VMM physical-section
            alignment and the SM-dependent Mega inference operand envelope.
            An explicit ``config.num_sms`` resolves the device-only Mega
            envelope without a device. Without either, its envelope remains
            unresolved; other byte requirements remain logical.

    Returns:
        Immutable memory plan shared by prefetch and context allocation.

    Raises:
        ValueError: If the group size or requested budget is invalid.
    """
    if ep_size <= 0:
        raise ValueError("ep_size must be positive")
    effective_device_scratch_capacity_factor = min(
        config.device_scratch_capacity_factor,
        float(ep_size),
    )
    if effective_device_scratch_capacity_factor != (
        config.device_scratch_capacity_factor
    ):
        logger.warning(
            "Capping device_scratch_capacity_factor from %g to %g for "
            "ep_size=%d; routing topology cannot deliver a larger imbalance",
            config.device_scratch_capacity_factor,
            effective_device_scratch_capacity_factor,
            ep_size,
        )
    effective_config = dataclasses.replace(
        config,
        device_scratch_capacity_factor=effective_device_scratch_capacity_factor,
    )
    model_config = _model_config(effective_config, ep_size=ep_size)
    num_slots = 0 if config.inference else config.num_activation_slots
    if config.inference:
        device_scratch = _align_memory(model_config.min_buffer_size(0))
        minimum_slot_bytes = 0
        balanced_full_save_slot_bytes = 0
    else:
        device_scratch = model_config.scratch_mem_size
        saved_input_bytes_per_layer = model_config.saved_act_mem_size(0, recompute=True)
        minimum_slot_bytes = _align_memory(
            config.max_moe_layers_per_activation_slot * saved_input_bytes_per_layer
        )
        balanced_full_save_slot_bytes = _align_memory(
            config.max_moe_layers_per_activation_slot
            * model_config.saved_act_mem_size(
                model_config._max_recv_tokens(1.0), recompute=False
            )
        )
        if balanced_full_save_slot_bytes < minimum_slot_bytes:
            raise RuntimeError(
                "balanced full-save activation state is smaller than mandatory "
                "layer inputs"
            )
    total_factor = effective_device_scratch_capacity_factor
    host_scratch = 0
    if config.vmm is not None:
        total_factor = min(
            config.vmm.total_scratch_capacity_factor,
            float(ep_size),
        )
        if total_factor < effective_device_scratch_capacity_factor:
            raise ValueError(
                "total_scratch_capacity_factor must cover at least the "
                "device_scratch_capacity_factor after EP-size clamping"
            )
        if config.inference:
            total_model_config = dataclasses.replace(
                model_config,
                max_imbalance_factor=total_factor,
            )
            total_scratch = _align_memory(total_model_config.min_buffer_size(0))
            host_scratch = max(0, total_scratch - device_scratch)
        else:
            host_scratch = model_config.host_scratch_mem_size(total_factor)
    inference_operand_bytes = None
    if (
        config.inference
        and config.block_scaled is not None
        and config.block_scaled.pipeline == "mega"
        and config.vmm is None
        and (device is not None or config.num_sms is not None)
    ):
        from ._inference_memory import _inference_operand_bytes

        num_sms = config.num_sms
        if num_sms is None:
            num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        inference_operand_bytes = _inference_operand_bytes(
            config,
            ep_size=ep_size,
            capacity_rows=model_config.max_recv_tokens,
            num_sms=num_sms,
        )
        device_scratch = max(device_scratch, inference_operand_bytes)
    if config.inference:
        maximum_useful_slot_bytes = 0
        selected_slot_bytes = 0
    else:
        maximum_useful_slot_bytes = _align_memory(
            config.max_moe_layers_per_activation_slot
            * model_config.saved_act_mem_size(
                model_config._max_recv_tokens(total_factor), recompute=False
            )
        )
        if config.activation_slot_bytes is not None:
            if config.activation_slot_bytes < minimum_slot_bytes:
                raise ValueError(
                    f"activation_slot_bytes={config.activation_slot_bytes} is below "
                    f"the minimum required {minimum_slot_bytes} bytes"
                )
            selected_slot_bytes = _align_memory(config.activation_slot_bytes)
        elif config.activation_slot_capacity_factor is not None:
            optional_balanced_bytes = balanced_full_save_slot_bytes - minimum_slot_bytes
            selected_slot_bytes = _align_memory(
                minimum_slot_bytes
                + math.ceil(
                    config.activation_slot_capacity_factor * optional_balanced_bytes
                )
            )
        else:
            selected_slot_bytes = minimum_slot_bytes
        if selected_slot_bytes > maximum_useful_slot_bytes:
            logger.warning(
                "activation_slot_bytes=%d exceeds the maximum useful %d bytes "
                "for this context",
                selected_slot_bytes,
                maximum_useful_slot_bytes,
            )
    total_activation_bytes = num_slots * selected_slot_bytes
    total_device_buffer = total_activation_bytes + device_scratch
    balanced_rows = config.max_num_local_input_tokens * config.top_k
    device_rows = model_config.max_recv_tokens
    total_rows = model_config._max_recv_tokens(total_factor)
    plan = MemoryPlan(
        balanced_recv_rows=balanced_rows,
        device_scratch_capacity_rows=device_rows,
        total_scratch_capacity_rows=total_rows,
        device_scratch_capacity_factor=effective_device_scratch_capacity_factor,
        total_scratch_capacity_factor=total_factor,
        saved_input_bytes_per_layer=model_config.saved_act_mem_size(0, recompute=True),
        device_scratch_bytes=device_scratch,
        host_scratch_bytes=host_scratch,
        minimum_activation_slot_bytes=minimum_slot_bytes,
        balanced_full_save_activation_slot_bytes=(balanced_full_save_slot_bytes),
        maximum_useful_activation_slot_bytes=maximum_useful_slot_bytes,
        activation_slot_bytes=selected_slot_bytes,
        activation_slot_capacity_factor=config.activation_slot_capacity_factor,
        total_activation_bytes=total_activation_bytes,
        total_device_buffer_bytes=total_device_buffer,
        total_virtual_bytes=total_device_buffer + host_scratch,
        num_activation_slots=num_slots,
        inference=config.inference,
        vmm_granularity_bytes=None,
        vmm_device_prefix_bytes=None,
        vmm_host_section_bytes=None,
        vmm_device_scratch_section_bytes=None,
        vmm_padding_bytes=0,
        inference_operand_bytes=inference_operand_bytes,
    )
    if config.vmm is not None and host_scratch > 0 and device is not None:
        return _resolve_vmm_physical_plan(plan, _resolve_cuda_device(device))
    return plan


def _resolve_vmm_physical_plan(
    plan: MemoryPlan,
    device: torch.device,
) -> MemoryPlan:
    """Add device-specific physical VMM section sizes to a logical plan.

    CUDA maps each section at driver granularity. Aggregate activation storage
    keeps its logical size; padding at the end of the first device mapping becomes
    lower device scratch, after the host-overflow range when scratch is scanned
    from high to low addresses.

    Args:
        plan: Logical memory requirements.
        device: Indexed CUDA device that will own the mappings.

    Returns:
        Plan containing exact physical section sizes and virtual range size.
    """
    from ._vmm import (
        get_vmm_allocation_granularity,
    )

    assert device.index is not None
    granularity = get_vmm_allocation_granularity(device.index)
    device_prefix = (
        _align_to(plan.total_activation_bytes, granularity)
        if plan.total_activation_bytes
        else 0
    )
    host_section = _align_to(plan.host_scratch_bytes, granularity)
    device_scratch_section = _align_to(
        plan.device_scratch_bytes,
        granularity,
    )
    section_total = device_prefix + host_section + device_scratch_section
    # PyTorch's caching allocator reserves a 20 MiB block for large-pool
    # requests below 10 MiB. Assign those already-allocated trailing bytes to
    # hot device scratch so prefetch matches and the storage is not wasted.
    allocator_total = (
        _CUDA_LARGE_POOL_BUFFER_BYTES
        if section_total < _CUDA_LARGE_POOL_MIN_ALLOC_BYTES
        else section_total
    )
    device_scratch_section += allocator_total - section_total
    return dataclasses.replace(
        plan,
        total_virtual_bytes=allocator_total,
        vmm_granularity_bytes=granularity,
        vmm_device_prefix_bytes=device_prefix,
        vmm_host_section_bytes=host_section,
        vmm_device_scratch_section_bytes=device_scratch_section,
        vmm_padding_bytes=(
            allocator_total
            - plan.total_activation_bytes
            - plan.host_scratch_bytes
            - plan.device_scratch_bytes
        ),
    )


def _vmm_sections(
    device_prefix_bytes: int,
    host_scratch_bytes: int,
    device_scratch_bytes: int,
) -> list[Any]:
    """Build the non-empty physical sections for a DistMoE activation buffer.

    A scratch-only inference buffer can have a zero-byte activation prefix, so
    empty sections are omitted before calling the CUDA VMM APIs.

    Args:
        device_prefix_bytes: Requested leading device-section bytes.
        host_scratch_bytes: Requested host overflow-scratch bytes.
        device_scratch_bytes: Requested device hot-scratch bytes.

    Returns:
        Ordered private VMM section specifications.
    """
    from ._vmm import Location, SectionSpec

    requested = (
        (device_prefix_bytes, Location.DEVICE),
        (host_scratch_bytes, Location.HOST),
        (device_scratch_bytes, Location.DEVICE),
    )
    return [
        SectionSpec(size=size, location=location)
        for size, location in requested
        if size
    ]


def _prefetch_vmm_from_sections(
    *,
    device_prefix_bytes: int,
    host_scratch_bytes: int,
    device_scratch_bytes: int,
    device: torch.device | str | int | None = None,
) -> _VmmPrefetch:
    """Asynchronously build an exact DistMoE VMM physical layout.

    Context construction calls this helper after resolving the physical plan.

    Args:
        device_prefix_bytes: Leading device-section bytes. This physical
            section may include saved activations and alignment padding.
        host_scratch_bytes: Host overflow-scratch bytes.
        device_scratch_bytes: Device hot-scratch bytes.
        device: Target CUDA device, defaulting to the current device.

    Returns:
        Private single-use owner consumed by context construction.

    Raises:
        ValueError: If the section sizes or device are invalid.
    """
    if device_prefix_bytes < 0:
        raise ValueError("device_prefix_bytes cannot be negative")
    if host_scratch_bytes <= 0 or device_scratch_bytes <= 0:
        raise ValueError("host and device scratch sizes must be positive")
    resolved_device = _resolve_cuda_device(device)
    assert resolved_device.index is not None
    from ._vmm import (
        get_vmm_allocation_granularity,
        prefetch_vmm_region,
    )

    granularity = get_vmm_allocation_granularity(resolved_device.index)
    device_prefix_bytes = (
        _align_to(device_prefix_bytes, granularity) if device_prefix_bytes else 0
    )
    host_scratch_bytes = _align_to(host_scratch_bytes, granularity)
    device_scratch_bytes = _align_to(device_scratch_bytes, granularity)
    sections = _vmm_sections(
        device_prefix_bytes,
        host_scratch_bytes,
        device_scratch_bytes,
    )
    handle = prefetch_vmm_region(sections, device_ordinal=resolved_device.index)
    return _VmmPrefetch(handle)


@dataclasses.dataclass(init=False)
class Context:
    """Pre-allocated process-local state for one distributed MoE shape.

    Valid instances are returned by :func:`create_context`; direct
    construction is rejected because allocation, symmetric-memory rendezvous,
    process-local registration, and optional VMM ownership transfer must occur
    together.

    A context owns one mutable activation buffer, its shared scratch region, and
    one set of communication buffers. Calls using the same context must not
    overlap or execute concurrently on independent streams.

    Args:
        group: Expert-parallel process group.
        buffers: Symmetric communication buffers.
        activation_buffer: Device or VMM-backed activation buffer.
        config: Static execution configuration.
        memory_plan: Resolved activation, device-scratch, and host-scratch layout.
        _context_id: Process-local custom-operation identifier.
        _vmm_pool: Private PyTorch memory pool owning a VMM allocation.
        _closed: Whether this context has been closed.
        _vmm_teardown_device: Device whose failed VMM teardown remains retryable.
    """

    group: dist.ProcessGroup
    buffers: _CommunicationBuffers
    activation_buffer: ActivationBuffer | None
    config: Config
    memory_plan: MemoryPlan | None
    _context_id: str
    _vmm_pool: Any | None = None
    _closed: bool = False
    _vmm_teardown_device: int | None = None

    def __init__(self) -> None:
        """Reject construction outside :func:`create_context`.

        Raises:
            TypeError: Always. Context construction must establish collective
                resources and process-local registration together.
        """
        raise TypeError("use create_context()")

    @classmethod
    def _create(
        cls,
        *,
        group: dist.ProcessGroup,
        buffers: _CommunicationBuffers,
        activation_buffer: ActivationBuffer | None,
        config: Config,
        memory_plan: MemoryPlan | None,
        context_id: str,
        vmm_pool: Any | None,
    ) -> Context:
        """Construct a context after collective resources are initialized."""
        context = object.__new__(cls)
        context.group = group
        context.buffers = buffers
        context.activation_buffer = activation_buffer
        context.config = config
        context.memory_plan = memory_plan
        context._context_id = context_id
        context._vmm_pool = vmm_pool
        context._closed = False
        context._vmm_teardown_device = None
        return context

    @property
    def max_num_local_input_tokens(self) -> int:
        """Return the allocated local-token capacity on every EP rank.

        Returns:
            Maximum local input tokens accepted by one invocation.
        """
        return self.config.max_num_local_input_tokens

    @property
    def hidden_dim(self) -> int:
        """Return the configured model hidden dimension.

        Returns:
            Model hidden dimension.
        """
        return self.config.hidden_dim

    @property
    def top_k(self) -> int:
        """Return the configured experts selected per token.

        Returns:
            Number of selected experts per input token.
        """
        return self.config.top_k

    @property
    def context_id(self) -> str:
        """Return the process-local identifier used by the custom op.

        Returns:
            Opaque context identifier.

        Raises:
            RuntimeError: If the context has been closed.
        """
        if self._closed:
            raise RuntimeError("distributed MoE context is closed")
        return self._context_id

    def reset(self) -> None:
        """Reset activation-slot state for a new execution schedule.

        Raises:
            RuntimeError: If the context has already been closed.
        """
        if self._closed:
            raise RuntimeError("distributed MoE context is closed")
        if self.activation_buffer is not None:
            self.activation_buffer.reset()

    def select_activation_slot(
        self,
        activation_slot: int,
        num_moe_layers_in_slot: int,
    ) -> None:
        """Select one activation slot and its maximum local layer depth.

        Full-step CUDA graph capture records the immutable slot view selected
        for each scheduled Dist-MoE call. The fixed schedule replays the same
        per-call views, while autograd retains each forward's slot identity.

        Args:
            activation_slot: Physical activation slot selected for the next
                execution interval.
            num_moe_layers_in_slot: Maximum number of local MoE layers whose
                state may occupy the selected slot. Pipeline microbatch slots
                use all local MoE layers; stage-microbatch slots use the
                selected stage's local MoE-layer count.

        Raises:
            RuntimeError: If the context has already been closed.
            ValueError: If the slot or layer depth is outside the configured
                activation-buffer capacity.
        """
        if self._closed:
            raise RuntimeError("distributed MoE context is closed")
        if self.activation_buffer is not None:
            self.activation_buffer.select_activation_slot(
                activation_slot,
                num_moe_layers_in_slot,
            )

    def close(self) -> None:
        """Synchronize and release registry and VMM ownership.

        A repeated call retries incomplete VMM teardown. Callers must stop all
        graph replay and traced execution that references this context before
        closing it.
        """
        with _CONTEXT_LOCK:
            _unregister_context(self._context_id)
        if (
            self._closed
            and self._vmm_pool is None
            and self._vmm_teardown_device is None
        ):
            return
        self._closed = True
        if self._vmm_pool is not None:
            assert self.activation_buffer is not None
            device = self.activation_buffer.buffer.device
            assert device.index is not None
            self._vmm_teardown_device = device.index
            torch.cuda.synchronize(device)
            # Drop the tensor before the MemPool so its free callback can unmap
            # the CUDA virtual range while the allocator remains alive.
            self.activation_buffer.buffer = torch.empty(
                0, dtype=torch.uint8, device=device
            )
            self._vmm_pool = None
        if self._vmm_teardown_device is not None:
            self._finish_vmm_teardown()

    def _finish_vmm_teardown(self) -> None:
        """Drain, retry if needed, and verify one pending VMM teardown.

        Raises:
            RuntimeError: If driver cleanup fails persistently or retains a
                live virtual-memory region.
        """
        assert self._vmm_teardown_device is not None
        _finish_vmm_device_teardown(self._vmm_teardown_device)
        self._vmm_teardown_device = None


_CONTEXT_LOCK = threading.Lock()
_CONTEXT_IDS = itertools.count(1)
_VMM_ACTIVE_DEVICES: set[int | None] = set()


def _finish_vmm_device_teardown(device_index: int) -> None:
    """Release a device reservation after all VMM regions are gone.

    Args:
        device_index: CUDA device whose VMM-backed activation buffer has been released.

    Raises:
        RuntimeError: If CUDA Driver cleanup fails or retains a live region.
    """
    from ._vmm import (
        get_active_regions,
        get_last_free_error,
        retry_failed_vmm_free,
    )

    # MemPool's destructor calls releasePool. Collect it before empty_cache so
    # the private pool is freeable before cache drain.
    gc.collect()
    torch.cuda.empty_cache()
    free_error = get_last_free_error(device_index)
    if free_error is not None:
        try:
            retry_failed_vmm_free(device_index)
        except RuntimeError as retry_error:
            raise RuntimeError(
                "Failed to release the DistMoE VMM-backed activation buffer; the CUDA Driver "
                "cleanup retry also failed"
            ) from retry_error
    if get_active_regions(device_index):
        raise RuntimeError(
            "DistMoE VMM regions remain active after allocator cache teardown"
        )
    with _CONTEXT_LOCK:
        _VMM_ACTIVE_DEVICES.discard(device_index)


def _allocate_vmm_buffer(
    *,
    plan: MemoryPlan,
    device: torch.device,
    prefetch: _VmmPrefetch | None,
) -> tuple[torch.Tensor, Any]:
    """Allocate the mixed device/host byte tensor for a resolved plan.

    Args:
        plan: Resolved memory layout.
        device: CUDA device on which the virtual range is accessible.
        prefetch: Optional context-owned prefetch to consume for this allocation.

    Returns:
        Raw byte tensor and its owning PyTorch memory pool.

    Raises:
        RuntimeError: If CUDA VMM allocation fails or silently falls back.
    """
    from ._vmm import (
        configured_vmm_pool,
        get_active_regions,
        get_last_alloc_error,
    )

    assert plan.vmm_device_prefix_bytes is not None
    assert plan.vmm_host_section_bytes is not None
    assert plan.vmm_device_scratch_section_bytes is not None
    sections = _vmm_sections(
        plan.vmm_device_prefix_bytes,
        plan.vmm_host_section_bytes,
        plan.vmm_device_scratch_section_bytes,
    )
    try:
        with configured_vmm_pool(
            sections,
            prefetch=None if prefetch is None else prefetch._handle,
        ) as pool:
            with torch.cuda.use_mem_pool(pool):
                buffer = torch.empty(
                    plan.total_virtual_bytes,
                    dtype=torch.uint8,
                    device=device,
                )
    except torch.OutOfMemoryError:
        vmm_error = get_last_alloc_error(device.index)
        if vmm_error is None:
            raise
        raise RuntimeError(
            "Failed to allocate the DistMoE VMM-backed activation buffer. The CUDA OOM reported "
            "by PyTorch is a wrapper around the CUDA Driver VMM failure; "
            "inspect the chained exception for the physical allocation error."
        ) from vmm_error
    if not get_active_regions(device.index):
        raise RuntimeError(
            "The DistMoE VMM allocation unexpectedly used the default CUDA "
            "allocator, which would place host overflow scratch in device HBM."
        ) from get_last_alloc_error(device.index)
    if prefetch is not None and not prefetch.consumed:
        raise RuntimeError(
            "The DistMoE VMM prefetch was not consumed by the allocation"
        )
    return buffer, pool


def _close_vmm_prefetch(prefetch: _VmmPrefetch | None) -> None:
    """Release an optional unconsumed VMM prefetch, retrying once.

    A failed close retains the low-level region for retry. If both attempts
    fail, the caller must retain its device reservation so another context
    cannot overlap the still-owned virtual address range.

    Args:
        prefetch: Optional owner to release.

    Raises:
        RuntimeError: If both cleanup attempts fail.
    """
    if prefetch is not None:
        try:
            prefetch.close()
        except Exception:
            try:
                prefetch.close()
            except Exception as retry_error:
                raise RuntimeError(
                    "Failed to release the DistMoE VMM prefetch; cleanup retry "
                    "also failed"
                ) from retry_error


def _create_comm_buffers(
    config: Config,
    group: dist.ProcessGroup,
    device: torch.device,
    emulate_peer_buffers: bool,
) -> _CommunicationBuffers:
    """Collectively allocate the context's symmetric communication buffers.

    Args:
        config: Static execution capacity and tensor dimensions.
        group: Expert-parallel process group.
        device: CUDA device for local buffer views.
        emulate_peer_buffers: Whether virtual peers alias local storage.

    Returns:
        Communication buffers with the configured token capacity.
    """
    buffers = _CommunicationBuffers.create(
        max_num_local_input_tokens=config.max_num_local_input_tokens,
        hidden_dim=config.hidden_dim,
        top_k=config.top_k,
        group=group,
        device=device,
        emulate_peer_buffers=emulate_peer_buffers,
    )
    initialize_multimem_barrier_workspace(
        group,
        device,
        emulate_peer_buffers=emulate_peer_buffers,
    )
    return buffers


def create_context(  # noqa: C901
    *,
    group: dist.ProcessGroup,
    config: Config,
    device: torch.device | str | int | None = None,
    emulate_peer_buffers: bool | None = None,
) -> Context:
    """Collectively allocate communication and activation state.

    Every rank in ``group`` must call this function with matching shapes and in
    the same relative order because symmetric-memory rendezvous and barrier
    workspace initialization are collective. The returned context is reusable
    across sequential calls but is not reentrant. Construct it before tracing
    or CUDA-graph capture and close it only after all captured replays stop.

    Args:
        group: Expert-parallel process group.
        config: Static distributed MoE configuration.
        device: CUDA device, defaulting to the current device.
        emulate_peer_buffers: Whether every virtual peer should alias local
            storage. ``None`` enables emulation for PyTorch's built-in FakePG.
            Emulation validates shapes and memory, not distributed numerics.

    Returns:
        Registered process-local distributed MoE context.

    Raises:
        ValueError: If shapes or expert partitioning are invalid.
        RuntimeError: If CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CuTe distributed MoE requires CUDA")
    world_size = dist.get_world_size(group)
    if config.num_experts % world_size != 0:
        raise ValueError(
            f"num_experts={config.num_experts} must be divisible by "
            f"world_size={world_size}"
        )
    device = _resolve_cuda_device(device)
    plan = plan_memory(config, world_size, device=device)
    if emulate_peer_buffers is None:
        emulate_peer_buffers = is_fake_process_group(group)
    logger.info("%s", plan.explain())
    assert emulate_peer_buffers is not None
    uses_vmm = plan.uses_host_scratch

    from ._activation_buffer import ActivationBuffer

    if config.vmm is not None and not plan.uses_host_scratch:
        logger.warning(
            "VMM host scratch resolves to zero bytes; using an ordinary "
            "device allocation"
        )
    if uses_vmm:
        # Reserve before prefetch so another constructor cannot allocate a
        # competing process-global VMM-backed activation buffer while this context initializes.
        with _CONTEXT_LOCK:
            if device.index in _VMM_ACTIVE_DEVICES:
                raise RuntimeError(
                    f"CUDA device {device.index} already owns a live DistMoE VMM-backed activation buffer"
                )
            _VMM_ACTIVE_DEVICES.add(device.index)
    vmm_prefetch: _VmmPrefetch | None = None
    try:
        if uses_vmm and config.vmm is not None and config.vmm.prefetch:
            assert plan.vmm_device_prefix_bytes is not None
            assert plan.vmm_host_section_bytes is not None
            assert plan.vmm_device_scratch_section_bytes is not None
            vmm_prefetch = _prefetch_vmm_from_sections(
                device_prefix_bytes=plan.vmm_device_prefix_bytes,
                host_scratch_bytes=plan.vmm_host_section_bytes,
                device_scratch_bytes=plan.vmm_device_scratch_section_bytes,
                device=device,
            )
        buffers = _create_comm_buffers(
            config,
            group,
            device,
            emulate_peer_buffers,
        )
    except BaseException:
        _close_vmm_prefetch(vmm_prefetch)
        if uses_vmm:
            assert device.index is not None
            _finish_vmm_device_teardown(device.index)
        raise
    vmm_pool = None
    raw_buffer = None
    if uses_vmm:
        try:
            assert plan.vmm_device_prefix_bytes is not None
            assert plan.vmm_host_section_bytes is not None
            assert plan.vmm_device_scratch_section_bytes is not None
            raw_buffer, vmm_pool = _allocate_vmm_buffer(
                plan=plan,
                device=device,
                prefetch=vmm_prefetch,
            )
            activation_buffer = ActivationBuffer.create_from_buffer(
                buffer=raw_buffer,
                ep_size=world_size,
                scratch_mem_size_in_bytes=(
                    plan.total_virtual_bytes
                    if config.inference
                    else plan.total_virtual_bytes - plan.total_activation_bytes
                ),
                num_activation_slots=config.num_activation_slots,
                num_moe_layers=config.max_moe_layers_per_activation_slot,
                device_scratch_size=plan.vmm_device_scratch_section_bytes,
                host_scratch_size=plan.vmm_host_section_bytes,
                inference_mode=config.inference,
            )
        except Exception:
            _close_vmm_prefetch(vmm_prefetch)
            raw_buffer = None
            vmm_pool = None
            assert device.index is not None
            _finish_vmm_device_teardown(device.index)
            raise
    else:
        activation_buffer = ActivationBuffer.create(
            total_size_in_bytes=plan.total_device_buffer_bytes,
            ep_size=world_size,
            device=device,
            scratch_mem_size_in_bytes=plan.device_scratch_bytes,
            num_activation_slots=config.num_activation_slots,
            num_moe_layers=config.max_moe_layers_per_activation_slot,
            inference_mode=config.inference,
        )
    activation_buffer.scratch_capacity_factor = plan.total_scratch_capacity_factor
    activation_buffer.model_config = _model_config(config, ep_size=world_size)
    if plan.inference_operand_bytes is not None:
        # Static inference views reserve scratch without invoking the offset
        # planner. Reset must retain that reservation in its peak accounting.
        free_bytes = (
            activation_buffer.scratch_region_size - plan.inference_operand_bytes
        )
        activation_buffer.peak_min_free_space.fill_(free_bytes)
        activation_buffer.initial_peak_min_free_space.fill_(free_bytes)
    with _CONTEXT_LOCK:
        context_id = str(next(_CONTEXT_IDS))
        context = Context._create(
            group=group,
            buffers=buffers,
            activation_buffer=activation_buffer,
            config=config,
            memory_plan=plan,
            context_id=context_id,
            vmm_pool=vmm_pool,
        )
        _register_context(context_id, context)
    return context


def _validate_routing_shape(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    context: Context,
) -> None:
    """Validate one bounded physical token and routing shape.

    Args:
        x_TD: Local input tokens.
        topk_expert_ids_TK: Precomputed global expert IDs.
        topk_scores_TK: Precomputed expert weights.
        context: Pre-allocated execution context.

    Raises:
        ValueError: If any physical shape exceeds the context contract.
    """
    if x_TD.ndim != 2 or x_TD.shape[1] != context.hidden_dim:
        raise ValueError(
            f"x_TD must have shape [T, {context.hidden_dim}], got {tuple(x_TD.shape)}"
        )
    num_local_input_tokens = x_TD.shape[0]
    if not 1 <= num_local_input_tokens <= context.max_num_local_input_tokens:
        raise ValueError(
            "local input token count must be in "
            f"[1, context.max_num_local_input_tokens="
            f"{context.max_num_local_input_tokens}], "
            f"got {num_local_input_tokens}"
        )
    expected_routing_shape = (num_local_input_tokens, context.top_k)
    if (
        topk_expert_ids_TK.shape != expected_routing_shape
        or topk_scores_TK.shape != expected_routing_shape
    ):
        raise ValueError(
            f"topk_expert_ids_TK and topk_scores_TK must have shape {expected_routing_shape}"
        )


def _validate_inputs(  # noqa: C901
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    context: Context,
    *,
    allow_compact_weights: bool = False,
) -> None:
    """Validate the operation contract before launching kernels.

    Args:
        x_TD: Local input tokens.
        topk_expert_ids_TK: Precomputed global expert IDs.
        topk_scores_TK: Precomputed expert weights.
        w13_EFD: Local fused gate/up-projection weights.
        w2_EDF: Local down-projection weights.
        context: Pre-allocated execution context.
        allow_compact_weights: Whether prepared operands make source-weight
            dtype irrelevant to compute.

    Raises:
        ValueError: If tensor shapes, devices, or layouts are invalid.
        TypeError: If activation or weight dtypes are not BF16.
    """
    if x_TD.dtype != torch.bfloat16:
        raise TypeError("x_TD must use torch.bfloat16")
    if not allow_compact_weights and (
        w13_EFD.dtype != torch.bfloat16 or w2_EDF.dtype != torch.bfloat16
    ):
        raise TypeError("w13_EFD and w2_EDF compute weights must use torch.bfloat16")
    if topk_expert_ids_TK.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_expert_ids_TK must use torch.int32 or torch.int64")
    if topk_scores_TK.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("topk_scores_TK must use torch.bfloat16 or torch.float32")
    _validate_routing_shape(
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        context,
    )
    world_size = dist.get_world_size(context.group)
    num_local_experts = context.config.num_experts // world_size
    if w13_EFD.ndim != 3 or w2_EDF.ndim != 3:
        raise ValueError("w13_EFD and w2_EDF must be three-dimensional expert weights")
    if w13_EFD.shape[0] != num_local_experts or w2_EDF.shape[0] != num_local_experts:
        raise ValueError(f"weights must contain {num_local_experts} local experts")
    if w13_EFD.shape[1] != 2 * w2_EDF.shape[2]:
        raise ValueError(
            "w13_EFD output dimension must be twice w2_EDF's input dimension"
        )
    if w2_EDF.shape[2] != context.config.intermediate_dim:
        raise ValueError(
            "weight intermediate dimension does not match "
            f"config.intermediate_dim={context.config.intermediate_dim}"
        )
    if w13_EFD.shape[2] != context.hidden_dim or w2_EDF.shape[1] != context.hidden_dim:
        raise ValueError("weight hidden dimensions do not match the context")
    tensors = (x_TD, topk_expert_ids_TK, topk_scores_TK, w13_EFD, w2_EDF)
    if any(t.device != x_TD.device for t in tensors):
        raise ValueError("all operation tensors must reside on the same CUDA device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all operation tensors must be contiguous")


def _ep_barrier(
    buffer: Any,
    conditional_execution: torch.Tensor | None = None,
) -> None:
    """Synchronize expert-parallel ranks after symmetric-memory writes.

    Args:
        buffer: Symmetric-memory buffer whose peers should synchronize.
        conditional_execution: Optional common device-side barrier predicate.
    """
    from ._triton_ops import symmetric_memory_barrier

    symmetric_memory_barrier(
        buffer,
        channel=0,
        conditional_execution=conditional_execution,
    )


class _Bf16Autograd(torch.autograd.Function):
    """Own BF16 forward state and dispatch the fixed-topology backward.

    Forward mutates context-owned planner and communication storage and saves
    only tensors, device offsets, and the process-local context key required by
    backward. Backward reads the saved recompute predicate on device, so saved
    and recomputed layers retain one CUDA-graph launch topology. One context is
    intentionally non-reentrant because its scratch and planner state are
    shared across calls.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x_TD: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        w13_EFD: torch.Tensor,
        w2_EDF: torch.Tensor,
        activation_storage: torch.Tensor,
        activation_slot_id_1: torch.Tensor,
        num_moe_layers_in_slot: int,
        routing_storage: torch.Tensor,
        dispatch_storage: torch.Tensor,
        combine_storage: torch.Tensor,
        context_id: str,
        options: ExecutionOptions | None,
        save_for_backward: bool,
    ) -> torch.Tensor:
        """Run routing, fused dispatch, SwiGLU, fused combine, and reduction.

        Args:
            ctx: Autograd context.
            x_TD: Local BF16 input tokens.
            topk_expert_ids_TK: Precomputed global expert IDs.
            topk_scores_TK: Precomputed expert weights.
            w13_EFD: Local gate/up-projection weights.
            w2_EDF: Local down-projection weights.
            activation_storage: Graph-visible activation allocation.
            activation_slot_id_1: Selected physical activation slot.
            num_moe_layers_in_slot: Static MoE-layer count for the selected slot.
            routing_storage: Graph-visible local routing payload.
            dispatch_storage: Graph-visible local dispatch payload.
            combine_storage: Graph-visible local combine payload.
            context_id: Process-local execution context identifier.
            options: Optional eager execution controls.
            save_for_backward: Whether autograd can execute a matching backward.

        Returns:
            Local BF16 MoE output.
        """
        # 1. Recover process-local owners and reject graph-visible tensor views
        # that do not refer to this context's stable allocations.
        context = _get_context(context_id)
        options = _resolve_execution_options(options)
        ctx.w13_input_shape = w13_EFD.shape
        ctx.w2_input_shape = w2_EDF.shape
        w13_source, w2_source = w13_EFD, w2_EDF
        if options.weights_preprocess_fn is not None:
            w13_EFD = options.weights_preprocess_fn(w13_source)
            w2_EDF = options.weights_preprocess_fn(w2_source)
        w13_EFD, w2_EDF = _reshape_weights(w13_EFD, w2_EDF, context)
        _validate_inputs(
            x_TD, topk_expert_ids_TK, topk_scores_TK, w13_EFD, w2_EDF, context
        )
        activation_buffer = context.activation_buffer
        assert activation_buffer is not None
        context_storages = (
            activation_buffer.buffer,
            context.buffers.routing.local(),
            context.buffers.dispatch.local(),
            context.buffers.combine.local(),
        )
        for name, supplied, owned in zip(
            ("activation", "routing", "dispatch", "combine"),
            (
                activation_storage,
                routing_storage,
                dispatch_storage,
                combine_storage,
            ),
            context_storages,
            strict=True,
        ):
            if (
                supplied.data_ptr() != owned.data_ptr()
                or supplied.storage_offset() != owned.storage_offset()
                or supplied.shape != owned.shape
                or supplied.stride() != owned.stride()
                or supplied.dtype != owned.dtype
                or supplied.device != owned.device
                or supplied.layout != owned.layout
            ):
                raise ValueError(
                    f"{name}_storage does not match the context-owned tensor view"
                )

        from dist_moe._activation_buffer import (
            get_forward_plan,
            validate_buffer_capacity,
        )
        from dist_moe._routing_metadata import (
            dist_dispatch_routing,
        )
        from dist_moe._triton_ops import (
            copy_dispatch_to_activation,
            copy_routing_and_dispatch,
            swiglu_fwd,
        )
        from dist_moe.kernels.dist_grouped_gemm import (
            _native_swiglu_fwd_config_is_compatible,
            allocate_activation_buffer_counter_storage,
            dist_grouped_gemm_fprop_combine,
            dist_grouped_gemm_fprop_dispatch,
        )

        config = context.config
        scratch_only = config.inference or not save_for_backward
        buffers = context.buffers
        postprocess = _resolve_experts_output_postprocess_fn(
            options.experts_output_postprocess,
            x_TD=x_TD,
            topk_expert_ids_TK=topk_expert_ids_TK,
            topk_scores_TK=topk_scores_TK,
            inference_mode=config.inference,
            zero_out_padded_callback_inputs=(options.zero_out_padded_callback_inputs),
        )
        num_local_input_tokens = x_TD.shape[0]
        local_rank = dist.get_rank(context.group)
        routing_local_TK = _routing_ids_view(
            buffers.routing,
            local_rank,
            (topk_expert_ids_TK.shape[0], topk_expert_ids_TK.shape[1]),
        )
        routing_token_count_1 = _routing_token_count_view(
            buffers.routing,
            local_rank,
        )
        dispatch_local_TD = buffers.dispatch.hdl.get_buffer(
            local_rank,
            x_TD.shape,
            x_TD.dtype,
        )

        # 2. Publish routing and activation payloads before deriving peer
        # pointers. Every rank observes both payloads after the same barrier.
        with record_function("dist_moe_preprocess"):
            copy_routing_and_dispatch(
                x=x_TD,
                dispatch=dispatch_local_TD,
                expert_ids=topk_expert_ids_TK,
                routing=routing_local_TK,
                routing_token_count_1=routing_token_count_1,
                num_experts=config.num_experts,
                validate_expert_ids=options.validate_expert_ids,
            )
            _ep_barrier(buffers.dispatch)

        # Inference routing zeroes the grouped-GEMM counters, so the GEMM
        # launches skip their own memsets.
        dispatch_counter_storage = combine_counter_storage = None
        counter_workspaces = ()
        if config.inference:
            dispatch_counter_storage = allocate_activation_buffer_counter_storage(
                activation_buffer=activation_buffer.buffer, w=w13_EFD, is_dispatch=True
            )
            combine_counter_storage = allocate_activation_buffer_counter_storage(
                activation_buffer=activation_buffer.buffer, w=w2_EDF, is_dispatch=False
            )
            counter_workspaces = (dispatch_counter_storage, combine_counter_storage)

        with record_function("dist_moe_routing"):
            assert context.memory_plan is not None
            routing = dist_dispatch_routing(
                tokens=x_TD,
                expert_ids=topk_expert_ids_TK,
                num_experts=config.num_experts,
                group=context.group,
                comm_buffer=buffers,
                generate_bwd_gather_ptrs=not scratch_only,
                use_low_latency=config.inference,
                max_num_recv_tokens=context.memory_plan.total_scratch_capacity_rows,
                counter_workspaces=counter_workspaces,
            )

        num_recv_rows_1 = routing.num_tokens_per_rank[local_rank : local_rank + 1]
        # Only backward keeps the count past this call.
        if save_for_backward:
            num_recv_rows_1 = num_recv_rows_1.clone()
        model_config = _model_config(
            config,
            ep_size=dist.get_world_size(context.group),
            num_local_input_tokens=num_local_input_tokens,
        )
        validate_buffer_capacity(
            buffer_size=activation_buffer.buffer.numel(),
            model_config=model_config,
            num_activation_slots=(
                0 if config.inference else activation_buffer.num_activation_slots
            ),
        )
        num_recv_rows_per_rank_R = (
            torch.empty_like(routing.num_tokens_per_rank) if save_for_backward else None
        )
        forward_plan = get_forward_plan(
            num_recv_tokens=num_recv_rows_1,
            num_recv_tokens_per_rank=routing.num_tokens_per_rank,
            buffer_status=activation_buffer,
            model_config=model_config,
            activation_slot_id_1=activation_slot_id_1,
            num_moe_layers_in_slot=num_moe_layers_in_slot,
            inference_mode=config.inference,
            num_recv_tokens_per_rank_snapshot=num_recv_rows_per_rank_R,
            scratch_only=scratch_only,
        )
        estimated_num_recv_rows = max(1, num_local_input_tokens * context.top_k)
        fprop_config = _bf16_compute_config(
            config, routed_rows=estimated_num_recv_rows, num_groups=w13_EFD.shape[0]
        )
        combine_config = _bf16_compute_config(
            config,
            routed_rows=estimated_num_recv_rows,
            num_groups=w13_EFD.shape[0],
            combine_fprop=True,
        )
        fused_swiglu_forward = (
            options.swiglu_clip_stats_out_3 is None
            and not config.block_major_weights
            and fprop_config is not None
            and _native_swiglu_fwd_config_is_compatible(fprop_config, w2_EDF.shape[2])
        )

        # 3. Retain the local input only when backward may need device-selected
        # recomputation; all other intermediates follow the planner offsets.
        if save_for_backward:
            copy_dispatch_to_activation(
                dispatch=dispatch_local_TD,
                activation_buffer=activation_buffer.buffer,
                activation_offset=forward_plan.x_offset,
                condition=forward_plan.need_recompute,
            )

        # 4. Execute FC13 dispatch, SwiGLU, and FC2 combine against planner
        # offsets; kernels write intermediates directly into activation-buffer storage.
        _initialize_fake_peer_scatter_output(buffers.combine)
        with record_function("dist_moe_forward"):
            # The outer operation owns input publication and peer barriers.
            # Actual row counts stay on-device; this estimate only selects a
            # launch configuration and never truncates the routed rows.
            dist_grouped_gemm_fprop_dispatch(
                x=x_TD,
                w=w13_EFD,
                num_tokens_per_local_expert=routing.num_tokens_per_local_experts,
                gather_ptrs=routing.fwd_gather_ptrs,
                num_out_tokens=None,
                symm_mem_buffer=buffers.dispatch,
                topk=context.top_k,
                num_sms=config.num_sms,
                activation_buffer=activation_buffer.buffer,
                y=forward_plan.h1_offset,
                x_gathered=forward_plan.x_gathered_offset,
                config=fprop_config,
                estimate_recv_num_tokens=estimated_num_recv_rows,
                swiglu_h2=forward_plan.h2_offset if fused_swiglu_forward else None,
                swiglu_clamped=config.activation == "swiglu_clamped",
                swiglu_alpha=config.swiglu_alpha,
                swiglu_limit=config.swiglu_limit,
                precleared_counter_storage=dispatch_counter_storage,
                block_major_weights=config.block_major_weights,
            )
            # Clip observation, block-major weights and unsupported explicit
            # presets retain the standalone activation; eligible dispatch emits
            # both H1 and H2.
            if not fused_swiglu_forward:
                # A token picks each expert at most once, so every peer sends at
                # most min(top_k, local experts) routes per token.
                max_recv_rows = (
                    dist.get_world_size(context.group)
                    * num_local_input_tokens
                    * min(context.top_k, w13_EFD.shape[0])
                )
                swiglu_fwd(
                    x=forward_plan.h1_offset,
                    z=forward_plan.h2_offset,
                    clamped=config.activation == "swiglu_clamped",
                    alpha=config.swiglu_alpha,
                    limit=config.swiglu_limit,
                    activation_buffer=activation_buffer.buffer,
                    num_recv_tokens=num_recv_rows_1,
                    feature_dim=w2_EDF.shape[2],
                    dtype=w2_EDF.dtype,
                    clip_stats_out=options.swiglu_clip_stats_out_3,
                    clip_limit=options.swiglu_clip_limit,
                    max_rows=max_recv_rows,
                )
            h3_MD = dist_grouped_gemm_fprop_combine(
                x=forward_plan.h2_offset,
                w=w2_EDF,
                num_tokens_per_local_expert=routing.num_tokens_per_local_experts,
                scatter_ptrs=routing.scatter_ptrs,
                symm_mem_buffer=buffers.combine,
                num_sms=config.num_sms,
                activation_buffer=activation_buffer.buffer,
                config=combine_config,
                estimate_recv_num_tokens=estimated_num_recv_rows,
                precleared_counter_storage=combine_counter_storage,
                block_major_weights=config.block_major_weights,
            )
            h3_MD = h3_MD.view(-1, context.hidden_dim)[
                : num_local_input_tokens * context.top_k
            ]

        # 5. Make route outputs peer-visible before applying the common
        # postprocess and top-k reduction contract.
        _ep_barrier(buffers.combine)
        output_TD, _, postprocess_context, postprocess_output_dtype = (
            postprocess.forward(
                h3_MD,
                topk_scores_TK,
                output_dtype=x_TD.dtype,
                save=save_for_backward,
                save_buffer=activation_buffer.buffer if save_for_backward else None,
                save_offset=forward_plan.h3_offset if save_for_backward else None,
                save_condition=(
                    forward_plan.need_recompute if save_for_backward else None
                ),
            )
        )

        if not save_for_backward:
            ctx.mark_non_differentiable(output_TD)
            return output_TD

        # 6. Retain only graph-stable routing, planner, policy, and activation
        # state needed to reconstruct the matching backward.
        assert num_recv_rows_per_rank_R is not None
        ctx.context_id = context_id
        ctx.num_local_input_tokens = num_local_input_tokens
        ctx.model_config = model_config
        ctx.postprocess = postprocess
        ctx.postprocess_output_dtype = postprocess_output_dtype
        ctx.options = dataclasses.replace(options, wgrad_parameter_owners=None)
        assert forward_plan.packed_offsets is not None
        ctx.forward_offsets = forward_plan.packed_offsets
        if options.inplace_wgrad_accum:
            wgrad_parameter_owners = options.wgrad_parameter_owners
            ctx.w13_param_ref = _weak_parameter_ref(
                w13_source,
                None if wgrad_parameter_owners is None else wgrad_parameter_owners[0],
            )
            ctx.w2_param_ref = _weak_parameter_ref(
                w2_source,
                None if wgrad_parameter_owners is None else wgrad_parameter_owners[1],
            )
        ctx.save_for_backward(
            w13_source,
            w2_source,
            topk_expert_ids_TK,
            topk_scores_TK,
            routing.num_tokens_per_local_experts,
            routing.bwd_gather_ptrs,
            routing.fwd_gather_ptrs,
            routing.scatter_ptrs,
            num_recv_rows_per_rank_R,
            num_recv_rows_1,
            forward_plan.need_recompute,
            forward_plan.x_offset,
            forward_plan.x_gathered_offset,
            forward_plan.h1_offset,
            forward_plan.h2_offset,
            forward_plan.h3_offset,
            forward_plan.activation_slot_id_1,
            postprocess_context,
        )
        return output_TD

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[Any, ...]:
        """Run the eager callback-compatible backward implementation.

        Args:
            ctx: Autograd context populated by ``forward``.
            *grad_outputs: The single gradient of the local MoE output.

        Returns:
            Gradients matching every ``forward`` argument.
        """
        (grad_output_TD,) = grad_outputs
        return _Bf16Autograd._backward_impl(ctx, grad_output_TD)

    @staticmethod
    def _backward_impl(  # noqa: C901 -- preserve the backward scratch and gradient ordering
        ctx: Any,
        grad_output_TD: torch.Tensor,
        *,
        wgrad_destinations: tuple[_WgradDestination, _WgradDestination] | None = None,
        wgrad_output_dtype: torch.dtype | None = None,
    ) -> tuple[Any, ...]:
        """Run recompute as needed, then dgrad and two explicit wgrad GEMMs.

        Args:
            ctx: Autograd context populated by ``forward``.
            grad_output_TD: Gradient of the local MoE output.
            wgrad_destinations: Optional prevalidated W13/W2 destinations.
            wgrad_output_dtype: Optional resolved WGRAD kernel output dtype.

        Returns:
            Gradients matching every ``forward`` argument.
        """
        # 1. Recover immutable call policy plus the device-resident planner and
        # routing state saved by the matching forward invocation.
        context = _get_context(ctx.context_id)
        config = context.config
        options: ExecutionOptions = ctx.options
        buffers = context.buffers
        activation_buffer = context.activation_buffer
        assert activation_buffer is not None
        num_local_input_tokens = ctx.num_local_input_tokens
        (
            w13_source,
            w2_source,
            topk_expert_ids_TK,
            topk_scores_TK,
            num_recv_rows_per_local_expert_E,
            dispatch_bwd_gather_ptrs_M,
            dispatch_fwd_gather_ptrs_M,
            combine_scatter_ptrs_M,
            num_recv_rows_per_rank_R,
            num_recv_rows_1,
            need_recompute,
            x_offset,
            x_gathered_offset,
            h1_offset,
            h2_offset,
            h3_offset,
            activation_slot_id_1,
            postprocess_context,
        ) = ctx.saved_tensors
        if options.weights_preprocess_fn is not None:
            w13_EFD = options.weights_preprocess_fn(w13_source)
            w2_EDF = options.weights_preprocess_fn(w2_source)
        else:
            w13_EFD, w2_EDF = w13_source, w2_source
        w13_EFD, w2_EDF = _reshape_weights(w13_EFD, w2_EDF, context)
        parameter_refs = (
            (ctx.w13_param_ref, ctx.w2_param_ref)
            if options.inplace_wgrad_accum
            else None
        )
        clear_parameter_grads_before_return = wgrad_destinations is None
        if wgrad_destinations is None:
            wgrad_destinations = _resolve_wgrad_destinations(
                inplace_wgrad_accum=options.inplace_wgrad_accum,
                parameter_refs=parameter_refs,
                destination_fn=options.wgrad_destination_fn,
                w13_EFD=w13_EFD,
                w2_EDF=w2_EDF,
                output_dtype=config.wgrad_dtype,
            )
        wgrad_output_dtype = _resolve_wgrad_output_dtype(
            wgrad_destinations,
            wgrad_output_dtype
            if wgrad_output_dtype is not None
            else config.wgrad_dtype,
            w13_EFD.dtype,
        )
        from dist_moe._activation_buffer import (
            ForwardPlan,
            get_backward_plan,
        )
        from dist_moe._triton_ops import (
            conditional_copy_activations,
            copy_activation_to_dispatch,
            reduce_from_topk,
            swiglu_bwd,
            swiglu_fwd,
        )
        from dist_moe.kernels.dist_grouped_gemm import (
            dist_grouped_gemm_dgrad_combine,
            dist_grouped_gemm_dgrad_dispatch,
            dist_grouped_gemm_dgrad_swiglu_bwd_combine,
            dist_grouped_gemm_fprop_combine,
            dist_grouped_gemm_fprop_dispatch,
        )
        from dist_moe.kernels.grouped_gemm import (
            grouped_gemm_wgrad,
        )
        from dist_moe.kernels.native_combine_swiglu import (
            native_swiglu_bwd_col_multiple,
        )

        # 2. Rebuild typed plan views from saved scalar offsets, then let the
        # device planner choose saved versus recomputed intermediates without
        # changing the captured launch topology.
        forward_plan = ForwardPlan(
            need_recompute=need_recompute,
            x_offset=x_offset,
            x_gathered_offset=x_gathered_offset,
            h1_offset=h1_offset,
            h2_offset=h2_offset,
            h3_offset=h3_offset,
            activation_slot_id_1=activation_slot_id_1,
        )
        backward_plan = get_backward_plan(
            num_recv_tokens=num_recv_rows_1,
            num_recv_tokens_per_rank=num_recv_rows_per_rank_R,
            forward_plan=forward_plan,
            buffer_status=activation_buffer,
            model_config=ctx.model_config,
        )
        estimated_num_recv_rows = max(1, num_local_input_tokens * context.top_k)
        fprop_config = _bf16_compute_config(
            config, routed_rows=estimated_num_recv_rows, num_groups=w13_EFD.shape[0]
        )
        combine_config = _bf16_compute_config(
            config,
            routed_rows=estimated_num_recv_rows,
            num_groups=w13_EFD.shape[0],
            combine_fprop=True,
        )
        wgrad_config = _bf16_compute_config(
            config,
            wgrad=True,
            routed_rows=estimated_num_recv_rows,
            num_groups=w13_EFD.shape[0],
        )
        assert fprop_config is not None and wgrad_config is not None

        # 3. Recompute H1/H3 only when needed. H2 is always transient scratch,
        # so its activation kernel runs even when H1 was saved.
        _initialize_fake_peer_scatter_output(buffers.combine)
        with record_function("dist_moe_recompute"):
            dispatch_local_TD = buffers.dispatch.hdl.get_buffer(
                buffers.dispatch.hdl.rank,
                (num_local_input_tokens, context.hidden_dim),
                torch.bfloat16,
            )
            copy_activation_to_dispatch(
                dispatch=dispatch_local_TD,
                activation_buffer=activation_buffer.buffer,
                activation_offset=x_offset,
                condition=need_recompute,
            )
            _ep_barrier(buffers.dispatch, conditional_execution=need_recompute)
            dist_grouped_gemm_fprop_dispatch(
                x=x_offset,
                w=w13_EFD,
                num_tokens_per_local_expert=num_recv_rows_per_local_expert_E,
                gather_ptrs=dispatch_fwd_gather_ptrs_M,
                num_out_tokens=None,
                symm_mem_buffer=buffers.dispatch,
                topk=context.top_k,
                num_sms=config.num_sms,
                activation_buffer=activation_buffer.buffer,
                y=backward_plan.h1_offset,
                x_gathered=backward_plan.x_gathered_offset,
                conditional_execution=need_recompute,
                config=fprop_config,
                estimate_recv_num_tokens=estimated_num_recv_rows,
            )
            # H2 is always transient scratch, even when H1 was saved. Rebuild
            # it unconditionally before W2 WGRAD, matching the internal order.
            swiglu_fwd(
                x=backward_plan.h1_offset,
                z=backward_plan.h2_offset,
                clamped=config.activation == "swiglu_clamped",
                alpha=config.swiglu_alpha,
                limit=config.swiglu_limit,
                activation_buffer=activation_buffer.buffer,
                num_recv_tokens=num_recv_rows_1,
                feature_dim=w2_EDF.shape[2],
                dtype=w2_EDF.dtype,
            )
            h3_saved_MD = dist_grouped_gemm_fprop_combine(
                x=backward_plan.h2_offset,
                w=w2_EDF,
                num_tokens_per_local_expert=num_recv_rows_per_local_expert_E,
                scatter_ptrs=combine_scatter_ptrs_M,
                symm_mem_buffer=buffers.combine,
                num_sms=config.num_sms,
                activation_buffer=activation_buffer.buffer,
                conditional_execution=need_recompute,
                config=combine_config,
                estimate_recv_num_tokens=estimated_num_recv_rows,
            )
            h3_saved_MD = h3_saved_MD.view(-1, context.hidden_dim)[
                : num_local_input_tokens * context.top_k
            ]
            if not ctx.postprocess.fuses_saved_copy_into_reduction:
                conditional_copy_activations(
                    condition=need_recompute,
                    lhs=None,
                    lhs_offset=None,
                    rhs=h3_saved_MD,
                    rhs_offset=h3_offset,
                    activation_buffer=activation_buffer.buffer,
                    copy_to_buffer=False,
                )
            _ep_barrier(buffers.combine, conditional_execution=need_recompute)

        # 4. Reverse postprocessing before expert DGRAD/WGRAD so both paths
        # consume the same route-wise gradient representation.
        grad_h3_MD, grad_topk_scores_TK, grad_h3_is_published = (
            ctx.postprocess.backward(
                grad_output_TD,
                h3_saved_MD,
                topk_scores_TK,
                postprocess_output_dtype=ctx.postprocess_output_dtype,
                context=postprocess_context,
                publish_view=lambda shape, dtype: buffers.combine.hdl.get_buffer(
                    buffers.combine.hdl.rank,
                    shape,
                    dtype,
                ),
                x_buffer=activation_buffer.buffer,
                x_buffer_offset=h3_offset,
                x_buffer_condition=need_recompute,
            )
        )
        if not grad_h3_is_published:
            buffers.combine.hdl.get_buffer(
                buffers.combine.hdl.rank,
                grad_h3_MD.shape,
                grad_h3_MD.dtype,
            ).copy_(grad_h3_MD)
        _ep_barrier(buffers.combine)

        # 5. Compute W2 then W13 gradients in dependency order. Optional
        # destinations let the kernels write directly into framework storage.
        with record_function("dist_moe_backward"):
            dist_grouped_gemm_dgrad_dispatch(
                dy=grad_h3_MD,
                w=w2_EDF,
                num_tokens_per_local_expert=num_recv_rows_per_local_expert_E,
                gather_ptrs=combine_scatter_ptrs_M,
                num_out_tokens=None,
                symm_mem_buffer=buffers.combine,
                num_sms=config.num_sms,
                activation_buffer=activation_buffer.buffer,
                dx=backward_plan.grad_h2_offset,
                dy_gathered=backward_plan.grad_h3_gathered_offset,
                config=fprop_config,
            )

            if wgrad_destinations is None:
                grad_w2_output_EDF = torch.empty(
                    w2_EDF.shape,
                    dtype=wgrad_output_dtype,
                    device=w2_EDF.device,
                )
                w2_output_accum = False
            else:
                grad_w2_output_EDF = wgrad_destinations[1].output
                if grad_w2_output_EDF is None:
                    grad_w2_output_EDF = torch.empty(
                        w2_EDF.shape,
                        dtype=wgrad_output_dtype,
                        device=w2_EDF.device,
                    )
                w2_output_accum = wgrad_destinations[1].accumulate
            grad_w2_EDF = grouped_gemm_wgrad(
                dy=backward_plan.grad_h3_gathered_offset,
                x=backward_plan.h2_offset,
                split_sizes=num_recv_rows_per_local_expert_E,
                activation_buffer=activation_buffer.buffer,
                hidden_dim_dy=w2_EDF.shape[1],
                hidden_dim_x=w2_EDF.shape[2],
                dtype=w2_EDF.dtype,
                output_accum=w2_output_accum,
                wgrad=grad_w2_output_EDF,
                num_sms=config.num_sms,
                config=wgrad_config,
            )
            grad_w2_EDF = _postprocess_wgrad(
                options.wgrad_postprocess_fn,
                "w2",
                grad_w2_EDF,
            )
            _initialize_fake_peer_scatter_output(buffers.dispatch)
            # W2 WGRAD has consumed H2 before its scratch may be reused for
            # grad_H1. Fusion writes grad_H1 into that same planned destination.
            if w2_EDF.shape[2] % native_swiglu_bwd_col_multiple() == 0:
                grad_x_TKD, _ = dist_grouped_gemm_dgrad_swiglu_bwd_combine(
                    grad_h2=backward_plan.grad_h2_offset,
                    h1=backward_plan.h1_offset,
                    w=w13_EFD,
                    num_tokens_per_local_expert=num_recv_rows_per_local_expert_E,
                    scatter_ptrs=dispatch_bwd_gather_ptrs_M,
                    symm_mem_buffer=buffers.dispatch,
                    grad_h1=backward_plan.grad_h1_offset,
                    num_sms=config.num_sms,
                    activation_buffer=activation_buffer.buffer,
                    config=fprop_config,
                    estimate_recv_num_tokens=estimated_num_recv_rows,
                    swiglu_clamped=config.activation == "swiglu_clamped",
                    swiglu_alpha=config.swiglu_alpha,
                    swiglu_limit=config.swiglu_limit,
                )
            else:
                swiglu_bwd(
                    dz=backward_plan.grad_h2_offset,
                    x=backward_plan.h1_offset,
                    dxy=backward_plan.grad_h1_offset,
                    clamped=config.activation == "swiglu_clamped",
                    alpha=config.swiglu_alpha,
                    limit=config.swiglu_limit,
                    activation_buffer=activation_buffer.buffer,
                    num_recv_tokens=num_recv_rows_1,
                    feature_dim=w2_EDF.shape[2],
                    dtype=w2_EDF.dtype,
                )
                grad_x_TKD = dist_grouped_gemm_dgrad_combine(
                    dy=backward_plan.grad_h1_offset,
                    w=w13_EFD,
                    num_tokens_per_local_expert=num_recv_rows_per_local_expert_E,
                    scatter_ptrs=dispatch_bwd_gather_ptrs_M,
                    symm_mem_buffer=buffers.dispatch,
                    num_sms=config.num_sms,
                    activation_buffer=activation_buffer.buffer,
                    config=fprop_config,
                )
            grad_x_TKD = grad_x_TKD.view(-1, context.hidden_dim)[
                : num_local_input_tokens * context.top_k
            ].view(num_local_input_tokens, context.top_k, context.hidden_dim)

            if wgrad_destinations is None:
                grad_w13_output_EFD = torch.empty(
                    w13_EFD.shape,
                    dtype=wgrad_output_dtype,
                    device=w13_EFD.device,
                )
                w13_output_accum = False
            else:
                grad_w13_output_EFD = wgrad_destinations[0].output
                if grad_w13_output_EFD is None:
                    grad_w13_output_EFD = torch.empty(
                        w13_EFD.shape,
                        dtype=wgrad_output_dtype,
                        device=w13_EFD.device,
                    )
                w13_output_accum = wgrad_destinations[0].accumulate
            grad_w13_EFD = grouped_gemm_wgrad(
                dy=backward_plan.grad_h1_offset,
                x=backward_plan.x_gathered_offset,
                split_sizes=num_recv_rows_per_local_expert_E,
                activation_buffer=activation_buffer.buffer,
                hidden_dim_dy=w13_EFD.shape[1],
                hidden_dim_x=w13_EFD.shape[2],
                dtype=w13_EFD.dtype,
                output_accum=w13_output_accum,
                wgrad=grad_w13_output_EFD,
                num_sms=config.num_sms,
                config=wgrad_config,
            )
            grad_w13_EFD = _postprocess_wgrad(
                options.wgrad_postprocess_fn,
                "w13",
                grad_w13_EFD,
            )
            if clear_parameter_grads_before_return and wgrad_destinations is not None:
                for destination in wgrad_destinations:
                    if destination.parameter is not None:
                        destination.parameter.grad = None
        # 6. Publish the route-wise input gradient, reduce top-k routes back to
        # `[T, D]`, and restore logical weight shapes unless an external WGRAD
        # owner consumed those gradients directly.
        _ep_barrier(buffers.dispatch)
        grad_x_TD = reduce_from_topk(
            grad_x_TKD,
            expert_ids=topk_expert_ids_TK,
        )
        external_wgrad_destination = options.wgrad_destination_fn is not None
        return (
            grad_x_TD,
            None,  # topk_expert_ids_TK
            grad_topk_scores_TK,
            (
                None
                if external_wgrad_destination or grad_w13_EFD is None
                else grad_w13_EFD.view(ctx.w13_input_shape)
            ),
            (
                None
                if external_wgrad_destination or grad_w2_EDF is None
                else grad_w2_EDF.view(ctx.w2_input_shape)
            ),
            None,  # activation_storage
            None,  # activation_slot_id_1
            None,  # num_moe_layers_in_slot
            None,  # routing_storage
            None,  # dispatch_storage
            None,  # combine_storage
            None,  # context_id
            None,  # options
            None,  # save_for_backward
        )


# Keep context-owned mutable storage explicit so tracing and CUDA graphs retain
# the operation's ordering and lifetime dependencies.
_LIBRARY = torch.library.Library("dist_moe", "DEF")
_LIBRARY.define(
    "bf16(Tensor x_TD, Tensor topk_expert_ids_TK, Tensor topk_scores_TK, Tensor w13_EFD, Tensor w2_EDF, "
    "Tensor? rmsnorm_weight_D, bool rmsnorm_enabled, float rmsnorm_eps, "
    "ScalarType rmsnorm_norm_output_dtype, ScalarType rmsnorm_output_dtype, "
    "bool rmsnorm_require_bitwise, float rmsnorm_gain_center, bool rmsnorm_use_kahan, bool rmsnorm_recompute_rstd, "
    "Tensor(a!) activation_storage, Tensor activation_slot_id_1, int num_moe_layers_in_slot, "
    "Tensor(b!) routing_storage, "
    "Tensor(c!) dispatch_storage, Tensor(d!) combine_storage, "
    "Tensor(e!)? swiglu_clip_stats_out_3, float swiglu_clip_limit, "
    "bool validate_expert_ids, str context_id) "
    "-> Tensor"
)


@torch.library.impl(_LIBRARY, "bf16", "CUDA")
def _dist_moe_custom_op_cuda(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    rmsnorm_weight_D: torch.Tensor | None,
    rmsnorm_enabled: bool,
    rmsnorm_eps: float,
    rmsnorm_norm_output_dtype: torch.dtype,
    rmsnorm_output_dtype: torch.dtype,
    rmsnorm_require_bitwise: bool,
    rmsnorm_gain_center: float,
    rmsnorm_use_kahan: bool,
    rmsnorm_recompute_rstd: bool,
    activation_storage: torch.Tensor,
    activation_slot_id_1: torch.Tensor,
    num_moe_layers_in_slot: int,
    routing_storage: torch.Tensor,
    dispatch_storage: torch.Tensor,
    combine_storage: torch.Tensor,
    swiglu_clip_stats_out_3: torch.Tensor | None,
    swiglu_clip_limit: float,
    validate_expert_ids: bool,
    context_id: str,
) -> torch.Tensor:
    """Run the CUDA implementation through its autograd function.

    Args:
        x_TD: Local BF16 input tokens.
        topk_expert_ids_TK: Precomputed global expert IDs.
        topk_scores_TK: Precomputed expert weights.
        w13_EFD: Local gate/up-projection weights.
        w2_EDF: Local down-projection weights.
        rmsnorm_weight_D: Optional inference-only input-scale gamma.
        rmsnorm_enabled: Whether to run fused post-expert RMSNorm.
        rmsnorm_eps: RMSNorm epsilon.
        rmsnorm_norm_output_dtype: Normalized route-output dtype.
        rmsnorm_output_dtype: Final reduction output dtype.
        rmsnorm_require_bitwise: Whether to preserve standalone reduction order.
        rmsnorm_gain_center: Constant added to ``rmsnorm_weight_D``.
        rmsnorm_use_kahan: Whether to use compensated sum-of-squares.
        rmsnorm_recompute_rstd: Whether backward recomputes reciprocal RMS values.
        activation_storage: Graph-visible activation allocation.
        activation_slot_id_1: Selected physical activation slot.
        num_moe_layers_in_slot: Static MoE-layer count for the selected slot.
        routing_storage: Graph-visible routing payload.
        dispatch_storage: Graph-visible dispatch payload.
        combine_storage: Graph-visible combine payload.
        swiglu_clip_stats_out_3: Optional graph-visible FP32 SwiGLU counters.
        swiglu_clip_limit: Threshold for the optional clip counters.
        validate_expert_ids: Whether publication validates expert-ID bounds.
        context_id: Process-local execution context identifier.

    Returns:
        Local BF16 MoE output.
    """
    kernel_ctx = _KernelAutogradContext()
    postprocess = _rmsnorm_from_registered_args(
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
    # Registered-op inference launches its kernel chain with PDL (kernels/pdl.py);
    # inference with Python callbacks (requires_eager) does not. Imported here so
    # `import dist_moe` does not import Triton.
    from .kernels.pdl import pdl_launches

    with pdl_launches(_get_context(context_id).config.inference):
        return _Bf16Autograd.forward(
            kernel_ctx,
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            w13_EFD,
            w2_EDF,
            activation_storage,
            activation_slot_id_1,
            num_moe_layers_in_slot,
            routing_storage,
            dispatch_storage,
            combine_storage,
            context_id,
            ExecutionOptions(
                experts_output_postprocess=postprocess,
                swiglu_clip_stats_out_3=swiglu_clip_stats_out_3,
                swiglu_clip_limit=swiglu_clip_limit,
                validate_expert_ids=validate_expert_ids,
            ),
            False,
        )


@torch.library.impl(_LIBRARY, "bf16", "Meta")
def _dist_moe_custom_op_meta(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    rmsnorm_weight_D: torch.Tensor | None,
    rmsnorm_enabled: bool,
    rmsnorm_eps: float,
    rmsnorm_norm_output_dtype: torch.dtype,
    rmsnorm_output_dtype: torch.dtype,
    rmsnorm_require_bitwise: bool,
    rmsnorm_gain_center: float,
    rmsnorm_use_kahan: bool,
    rmsnorm_recompute_rstd: bool,
    activation_storage: torch.Tensor,
    activation_slot_id_1: torch.Tensor,
    num_moe_layers_in_slot: int,
    routing_storage: torch.Tensor,
    dispatch_storage: torch.Tensor,
    combine_storage: torch.Tensor,
    swiglu_clip_stats_out_3: torch.Tensor | None,
    swiglu_clip_limit: float,
    validate_expert_ids: bool,
    context_id: str,
) -> torch.Tensor:
    """Describe the inference custom-op output without communication.

    Args:
        x_TD: Local BF16 input tokens.
        topk_expert_ids_TK: Expert IDs, unused for metadata propagation.
        topk_scores_TK: Expert scores, unused for metadata propagation.
        w13_EFD: Gate/up weights, unused for metadata propagation.
        w2_EDF: Down weights, unused for metadata propagation.
        rmsnorm_weight_D: Optional input scale, unused for metadata propagation.
        rmsnorm_enabled: RMSNorm selection, unused for metadata propagation.
        rmsnorm_eps: RMSNorm epsilon, unused for metadata propagation.
        rmsnorm_norm_output_dtype: Norm dtype, unused for metadata propagation.
        rmsnorm_output_dtype: Output dtype, unused for metadata propagation.
        rmsnorm_require_bitwise: Numerics policy, unused for metadata propagation.
        rmsnorm_gain_center: Input-scale offset, unused for metadata propagation.
        rmsnorm_use_kahan: Reduction policy, unused for metadata propagation.
        activation_storage: Activation allocation, unused by the fake kernel.
        activation_slot_id_1: Selected activation slot, unused by the fake kernel.
        num_moe_layers_in_slot: Static slot depth, unused by the fake kernel.
        routing_storage: Routing allocation, unused by the fake kernel.
        dispatch_storage: Dispatch allocation, unused by the fake kernel.
        combine_storage: Combine allocation, unused by the fake kernel.
        swiglu_clip_stats_out_3: Clip counters, unused by the fake kernel.
        swiglu_clip_limit: Clip threshold, unused by the fake kernel.
        validate_expert_ids: Debug validation policy, unused by the fake kernel.
        context_id: Context identifier, unused by the fake kernel.

    Returns:
        Empty tensor with the output shape, dtype, and device metadata.
    """
    del topk_expert_ids_TK, topk_scores_TK, w13_EFD
    del w2_EDF, rmsnorm_weight_D, rmsnorm_enabled
    del rmsnorm_eps, rmsnorm_norm_output_dtype, rmsnorm_output_dtype
    del rmsnorm_require_bitwise, rmsnorm_gain_center, rmsnorm_use_kahan
    del rmsnorm_recompute_rstd, activation_storage, activation_slot_id_1
    del num_moe_layers_in_slot, routing_storage, dispatch_storage
    del combine_storage, swiglu_clip_stats_out_3, swiglu_clip_limit
    del validate_expert_ids, context_id
    return torch.empty_like(x_TD)


def _run_registered_bf16_forward(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
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
    validate_expert_ids: bool,
    context_id: str,
    options: ExecutionOptions | None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run BF16 forward and expose fixed tensor state for registered autograd.

    Args:
        x_TD: Local BF16 input tokens.
        topk_expert_ids_TK: Precomputed global expert IDs.
        topk_scores_TK: Precomputed expert weights.
        w13_EFD: Local gate/up-projection weights.
        w2_EDF: Local down-projection weights.
        rmsnorm_weight_D: Optional inference-only input-scale gamma.
        rmsnorm_enabled: Whether to run fused post-expert RMSNorm.
        rmsnorm_eps: RMSNorm epsilon.
        rmsnorm_norm_output_dtype: Normalized route-output dtype.
        rmsnorm_output_dtype: Final reduction output dtype.
        rmsnorm_require_bitwise: Whether to preserve standalone reduction order.
        rmsnorm_gain_center: Constant added to ``rmsnorm_weight_D``.
        rmsnorm_use_kahan: Whether to use compensated sum-of-squares.
        rmsnorm_recompute_rstd: Whether backward recomputes reciprocal RMS values.
        activation_slot_id_1: Selected physical activation slot.
        num_moe_layers_in_slot: Static MoE-layer count for the selected slot.
        validate_expert_ids: Whether publication validates expert-ID bounds.
        context_id: Process-local execution context identifier.
        options: Optional eager controls used inside the opaque operation.

    Returns:
        Output and fixed-shape routing/planner state for backward. The packed
        planner tensor contains five offsets followed by the selected-slot
        snapshot produced by this forward.
    """
    context = _get_context(context_id)
    activation_buffer = context.activation_buffer
    assert activation_buffer is not None
    kernel_ctx = _KernelAutogradContext()
    postprocess = _rmsnorm_from_registered_args(
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
    output_TD = _Bf16Autograd.forward(
        kernel_ctx,
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        w13_EFD,
        w2_EDF,
        activation_buffer.buffer,
        activation_slot_id_1,
        num_moe_layers_in_slot,
        context.buffers.routing.local(),
        context.buffers.dispatch.local(),
        context.buffers.combine.local(),
        context_id,
        dataclasses.replace(
            _resolve_execution_options(options),
            experts_output_postprocess=postprocess,
            validate_expert_ids=validate_expert_ids,
        ),
        True,
    )
    saved_tensors = kernel_ctx.saved_tensors
    if len(saved_tensors) != 18:
        raise RuntimeError(
            f"BF16 training forward produced {len(saved_tensors)} saved tensors"
        )
    forward_state_6 = kernel_ctx.forward_offsets
    if forward_state_6.numel() != 6:
        raise RuntimeError("BF16 forward planner produced invalid cached state")
    postprocess_context = saved_tensors[-1]
    if postprocess_context is None:
        postprocess_context = x_TD.new_empty(0)
    state_tensors = []
    for tensor in (
        *saved_tensors[4:11],
        forward_state_6,
        postprocess_context,
    ):
        assert tensor is not None
        state_tensors.append(tensor)
    return output_TD, state_tensors


@torch.library.custom_op(
    "dist_moe::bf16_forward",
    mutates_args=(),
    device_types="cuda",
)
def _bf16_forward_op(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
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
    validate_expert_ids: bool,
    context_id: str,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run BF16 training forward and expose fixed backward state.

    Args:
        x_TD: Local BF16 input tokens.
        topk_expert_ids_TK: Precomputed global expert IDs.
        topk_scores_TK: Precomputed expert weights.
        w13_EFD: Local gate/up-projection weights.
        w2_EDF: Local down-projection weights.
        rmsnorm_weight_D: Optional inference-only input-scale gamma.
        rmsnorm_enabled: Whether to run fused post-expert RMSNorm.
        rmsnorm_eps: RMSNorm epsilon.
        rmsnorm_norm_output_dtype: Normalized route-output dtype.
        rmsnorm_output_dtype: Final reduction output dtype.
        rmsnorm_require_bitwise: Whether to preserve standalone reduction order.
        rmsnorm_gain_center: Constant added to ``rmsnorm_weight_D``.
        rmsnorm_use_kahan: Whether to use compensated sum-of-squares.
        rmsnorm_recompute_rstd: Whether backward recomputes reciprocal RMS values.
        activation_slot_id_1: Selected physical activation slot.
        num_moe_layers_in_slot: Static MoE-layer count for the selected slot.
        validate_expert_ids: Whether publication validates expert-ID bounds.
        context_id: Process-local execution context identifier.

    Returns:
        Output followed by routing metadata and activation-plan tensors. The
        planner state contains five offsets followed by the selected-slot
        snapshot produced by this forward.
    """
    return _run_registered_bf16_forward(
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        w13_EFD,
        w2_EDF,
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
        validate_expert_ids,
        context_id,
        None,
    )


def _fake_registered_bf16_forward(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
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
    validate_expert_ids: bool,
    context_id: str,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Build BF16 output and saved-state metadata without reading data.

    Args:
        x_TD: Fake local input tokens.
        topk_expert_ids_TK: Fake expert IDs.
        topk_scores_TK: Fake expert weights.
        w13_EFD: Fake local gate/up weights.
        w2_EDF: Fake local down weights.
        rmsnorm_weight_D: Fake optional input-scale gamma.
        rmsnorm_enabled: Whether fused post-expert RMSNorm is selected.
        rmsnorm_eps: RMSNorm epsilon.
        rmsnorm_norm_output_dtype: Normalized route-output dtype.
        rmsnorm_output_dtype: Final reduction output dtype.
        rmsnorm_require_bitwise: Whether to preserve reduction order.
        rmsnorm_gain_center: Constant added to the input-scale gamma.
        rmsnorm_use_kahan: Whether to use compensated sum-of-squares.
        rmsnorm_recompute_rstd: Whether backward recomputes reciprocal RMS values.
        activation_slot_id_1: Fake selected physical activation slot.
        num_moe_layers_in_slot: Static slot depth, unused by the fake kernel.
        validate_expert_ids: Debug validation policy, unused by the fake kernel.
        context_id: Process-local context identifier supplying static config.

    Returns:
        Fake output followed by fixed-shape routing and planner tensors,
        including the six-element BF16 planner state.
    """
    del topk_scores_TK, w2_EDF, rmsnorm_weight_D
    del rmsnorm_eps, rmsnorm_output_dtype, rmsnorm_require_bitwise
    del rmsnorm_gain_center, rmsnorm_use_kahan, activation_slot_id_1
    del num_moe_layers_in_slot, validate_expert_ids
    context = _get_context(context_id)
    num_local_experts = w13_EFD.shape[0]
    world_size = context.config.num_experts // num_local_experts
    pointer_count = context.config.num_experts * x_TD.shape[0]
    new_integer_tensor = topk_expert_ids_TK.new_empty
    state_tensors = [
        new_integer_tensor((num_local_experts,), dtype=torch.int32),
        new_integer_tensor((pointer_count,), dtype=torch.int64),
        new_integer_tensor((pointer_count,), dtype=torch.int64),
        new_integer_tensor((pointer_count,), dtype=torch.int64),
        new_integer_tensor((world_size,), dtype=torch.int32),
        new_integer_tensor((1,), dtype=torch.int32),
        new_integer_tensor((1,), dtype=torch.bool),
        new_integer_tensor((6,), dtype=torch.int64),
        (
            torch.empty(
                (x_TD.shape[0], topk_expert_ids_TK.shape[1]),
                dtype=torch.float32,
                device=x_TD.device,
            )
            if rmsnorm_enabled and not rmsnorm_recompute_rstd
            else x_TD.new_empty(0)
        ),
    ]
    return torch.empty_like(x_TD), state_tensors


@_bf16_forward_op.register_fake
def _bf16_forward_fake(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
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
    validate_expert_ids: bool,
    context_id: str,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return BF16 forward metadata for FakeTensor propagation."""
    return _fake_registered_bf16_forward(
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        w13_EFD,
        w2_EDF,
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
        validate_expert_ids,
        context_id,
    )


# BF16 owns a persistent device-side activation planner. An ordered effect token
# preserves its push/pop sequence without auto-functionalization copying the
# activation buffer or invalidating symmetric-memory identities.
_bf16_forward_op.register_effect(torch.library.EffectType.ORDERED)


@torch.library.custom_op(
    "dist_moe::bf16_forward_with_clip_stats",
    mutates_args=(),
    device_types="cuda",
)
def _bf16_forward_with_clip_stats_op(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
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
    validate_expert_ids: bool,
    swiglu_clip_limit: float,
    context_id: str,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Run functional BF16 training forward and return clip counters.

    Args:
        x_TD: Local BF16 input tokens.
        topk_expert_ids_TK: Precomputed global expert IDs.
        topk_scores_TK: Precomputed expert weights.
        w13_EFD: Local gate/up-projection weights.
        w2_EDF: Local down-projection weights.
        rmsnorm_weight_D: Optional inference-only input-scale gamma.
        rmsnorm_enabled: Whether to run fused post-expert RMSNorm.
        rmsnorm_eps: RMSNorm epsilon.
        rmsnorm_norm_output_dtype: Normalized route-output dtype.
        rmsnorm_output_dtype: Final reduction output dtype.
        rmsnorm_require_bitwise: Whether to preserve standalone reduction order.
        rmsnorm_gain_center: Constant added to ``rmsnorm_weight_D``.
        rmsnorm_use_kahan: Whether to use compensated sum-of-squares.
        rmsnorm_recompute_rstd: Whether backward recomputes reciprocal RMS values.
        activation_slot_id_1: Selected physical activation slot.
        num_moe_layers_in_slot: Static MoE-layer count for the selected slot.
        validate_expert_ids: Whether publication validates expert-ID bounds.
        swiglu_clip_limit: Threshold for the returned clip counters.
        context_id: Process-local execution context identifier.

    Returns:
        Output, fixed backward state, and FP32 ``[3]`` clip counters.
    """
    clip_stats_out_3 = x_TD.new_empty((3,), dtype=torch.float32)
    output_TD, state_tensors = _run_registered_bf16_forward(
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        w13_EFD,
        w2_EDF,
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
        validate_expert_ids,
        context_id,
        ExecutionOptions(
            swiglu_clip_stats_out_3=clip_stats_out_3,
            swiglu_clip_limit=swiglu_clip_limit,
        ),
    )
    return output_TD, state_tensors, clip_stats_out_3


@_bf16_forward_with_clip_stats_op.register_fake
def _bf16_forward_with_clip_stats_fake(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
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
    validate_expert_ids: bool,
    swiglu_clip_limit: float,
    context_id: str,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Return BF16 forward and clip-stat metadata for FakeTensor propagation."""
    output_TD, state_tensors = _fake_registered_bf16_forward(
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        w13_EFD,
        w2_EDF,
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
        validate_expert_ids,
        context_id,
    )
    return output_TD, state_tensors, x_TD.new_empty((3,), dtype=torch.float32)


_bf16_forward_with_clip_stats_op.register_effect(torch.library.EffectType.ORDERED)


def _run_bf16_registered_backward(
    grad_output_TD: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    state_tensors: list[torch.Tensor],
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
    """Rebuild BF16 state and run the shared backward implementation.

    Args:
        grad_output_TD: Gradient of the local MoE output.
        w13_EFD: Local gate/up weights.
        w2_EDF: Local down weights.
        topk_expert_ids_TK: Expert IDs from forward.
        topk_scores_TK: Router scores from forward.
        state_tensors: Routing metadata followed by packed planner state whose
            final value is the original forward's selected-slot snapshot.
        rmsnorm_enabled: Whether fused post-expert RMSNorm is selected.
        rmsnorm_eps: RMSNorm epsilon.
        rmsnorm_norm_output_dtype: Normalized route-output dtype.
        rmsnorm_output_dtype: Final reduction output dtype.
        rmsnorm_require_bitwise: Whether to preserve standalone reduction order.
        rmsnorm_use_kahan: Whether to use compensated sum-of-squares.
        rmsnorm_recompute_rstd: Whether backward recomputes reciprocal RMS values.
        wgrad_output_dtype: Resolved BF16 or FP32 WGRAD output dtype.
        context_id: Process-local execution context identifier.
        wgrad_destinations: Validated accumulation destinations, or ``None``
            for fresh functional WGRAD outputs.

    Returns:
        Gradients for input, routing scores, w13, and w2.
    """
    if len(state_tensors) != 9:
        raise RuntimeError(f"BF16 backward received {len(state_tensors)} state tensors")
    (
        num_recv_rows_per_local_expert_E,
        dispatch_bwd_gather_ptrs_M,
        dispatch_fwd_gather_ptrs_M,
        combine_scatter_ptrs_M,
        num_recv_rows_per_rank_R,
        num_recv_rows_1,
        need_recompute,
        forward_state_6,
        postprocess_context,
    ) = state_tensors
    context = _get_context(context_id)
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
    (
        x_offset,
        x_gathered_offset,
        h1_offset,
        h2_offset,
        h3_offset,
        activation_slot_id_1,
    ) = forward_state_6.chunk(6)
    kernel_ctx = _KernelAutogradContext()
    kernel_ctx.w13_input_shape = w13_EFD.shape
    kernel_ctx.w2_input_shape = w2_EDF.shape
    kernel_ctx.set(
        context_id=context_id,
        num_local_input_tokens=grad_output_TD.shape[0],
        model_config=_model_config(
            context.config,
            ep_size=dist.get_world_size(context.group),
            num_local_input_tokens=grad_output_TD.shape[0],
        ),
        options=ExecutionOptions(experts_output_postprocess=postprocess_config),
        postprocess=postprocess,
        postprocess_output_dtype=(
            rmsnorm_output_dtype if rmsnorm_enabled else grad_output_TD.dtype
        ),
    )
    kernel_ctx.saved_tensors = (
        w13_EFD,
        w2_EDF,
        topk_expert_ids_TK,
        topk_scores_TK,
        num_recv_rows_per_local_expert_E,
        dispatch_bwd_gather_ptrs_M,
        dispatch_fwd_gather_ptrs_M,
        combine_scatter_ptrs_M,
        num_recv_rows_per_rank_R,
        num_recv_rows_1,
        need_recompute,
        x_offset,
        x_gathered_offset,
        h1_offset,
        h2_offset,
        h3_offset,
        activation_slot_id_1,
        None if postprocess_context.numel() == 0 else postprocess_context,
    )
    gradients = _Bf16Autograd._backward_impl(
        kernel_ctx,
        grad_output_TD,
        wgrad_destinations=wgrad_destinations,
        wgrad_output_dtype=wgrad_output_dtype,
    )
    return gradients[0], gradients[2], gradients[3], gradients[4]


@torch.library.custom_op(
    "dist_moe::bf16_backward",
    mutates_args=(),
    device_types="cuda",
)
def _bf16_backward_op(
    grad_output_TD: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    state_tensors: list[torch.Tensor],
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
    """Run BF16 backward and return fresh input, score, and weight gradients."""
    return _run_bf16_registered_backward(
        grad_output_TD,
        w13_EFD,
        w2_EDF,
        topk_expert_ids_TK,
        topk_scores_TK,
        state_tensors,
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
    "dist_moe::bf16_backward_accumulate_",
    mutates_args=("accumulator_grad_w13_EFD", "accumulator_grad_w2_EDF"),
    device_types="cuda",
)
def _bf16_backward_accumulate_op(
    grad_output_TD: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    accumulator_grad_w13_EFD: torch.Tensor,
    accumulator_grad_w2_EDF: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    state_tensors: list[torch.Tensor],
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
    """Run BF16 backward and add WGRAD into explicit destinations."""
    wgrad_destinations = _wgrad_accumulation_destinations(
        accumulator_grad_w13_EFD,
        accumulator_grad_w2_EDF,
        w13_EFD,
        w2_EDF,
        wgrad_output_dtype,
    )
    gradients = _run_bf16_registered_backward(
        grad_output_TD,
        w13_EFD,
        w2_EDF,
        topk_expert_ids_TK,
        topk_scores_TK,
        state_tensors,
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


@_bf16_backward_op.register_fake
def _bf16_backward_fake(
    grad_output_TD: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    state_tensors: list[torch.Tensor],
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
    """Return BF16 gradient metadata without executing kernels.

    Args:
        grad_output_TD: Fake output gradient.
        w13_EFD: Fake gate/up weights.
        w2_EDF: Fake down weights.
        topk_expert_ids_TK: Fake expert IDs.
        topk_scores_TK: Fake router scores.
        state_tensors: Fake routing and activation-plan state.
        rmsnorm_enabled: Whether fused post-expert RMSNorm is selected.
        rmsnorm_eps: RMSNorm epsilon.
        rmsnorm_norm_output_dtype: Normalized route-output dtype.
        rmsnorm_output_dtype: Final reduction output dtype.
        rmsnorm_require_bitwise: Whether to preserve standalone reduction order.
        rmsnorm_use_kahan: Whether to use compensated sum-of-squares.
        rmsnorm_recompute_rstd: Whether backward recomputes reciprocal RMS values.
        wgrad_output_dtype: Resolved BF16 or FP32 weight-gradient dtype.
        context_id: Process-local context identifier.

    Returns:
        Fake gradients for input, routing scores, w13, and w2.
    """
    del topk_expert_ids_TK, state_tensors, rmsnorm_enabled
    del rmsnorm_eps, rmsnorm_norm_output_dtype, rmsnorm_output_dtype
    del rmsnorm_require_bitwise, rmsnorm_use_kahan, rmsnorm_recompute_rstd
    del context_id
    return (
        torch.empty_like(grad_output_TD),
        torch.empty_like(topk_scores_TK),
        w13_EFD.new_empty(w13_EFD.shape, dtype=wgrad_output_dtype),
        w2_EDF.new_empty(w2_EDF.shape, dtype=wgrad_output_dtype),
    )


@_bf16_backward_accumulate_op.register_fake
def _bf16_backward_accumulate_fake(
    grad_output_TD: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    accumulator_grad_w13_EFD: torch.Tensor,
    accumulator_grad_w2_EDF: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    state_tensors: list[torch.Tensor],
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
    """Return non-WGRAD metadata for a fake accumulating BF16 backward."""
    del topk_expert_ids_TK, state_tensors, rmsnorm_enabled
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


_bf16_backward_op.register_effect(torch.library.EffectType.ORDERED)


def _bf16_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[torch.Tensor, list[torch.Tensor]],
    *,
    mark_state_non_differentiable: bool = True,
) -> None:
    """Save BF16 forward-produced state for the registered autograd formula.

    The selected-slot snapshot comes from ``output``. The live selector input
    may name a later slot during selective activation-checkpoint recomputation
    and is never authoritative for a cached forward result.

    Args:
        ctx: PyTorch library autograd context.
        inputs: BF16 forward operator inputs.
        output: BF16 output and private saved-state tensors.
        mark_state_non_differentiable: Whether ``state_tensors`` are outputs of
            the current autograd boundary and should be marked.
    """
    (
        _x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        w13_EFD,
        w2_EDF,
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
        _validate_expert_ids,
        context_id,
    ) = inputs
    _output_TD, state_tensors = output
    ctx.context_id = context_id
    ctx.rmsnorm_enabled = rmsnorm_enabled
    ctx.rmsnorm_eps = rmsnorm_eps
    ctx.rmsnorm_norm_output_dtype = rmsnorm_norm_output_dtype
    ctx.rmsnorm_output_dtype = rmsnorm_output_dtype
    ctx.rmsnorm_require_bitwise = rmsnorm_require_bitwise
    ctx.rmsnorm_use_kahan = rmsnorm_use_kahan
    ctx.rmsnorm_recompute_rstd = rmsnorm_recompute_rstd
    ctx.set_materialize_grads(False)
    if mark_state_non_differentiable:
        ctx.mark_non_differentiable(*state_tensors)
    ctx.save_for_backward(
        w13_EFD,
        w2_EDF,
        topk_expert_ids_TK,
        topk_scores_TK,
        *state_tensors,
    )


def _bf16_autograd_backward(
    ctx: Any,
    grad_output_TD: torch.Tensor,
    unused_state_gradients: list[torch.Tensor | None] | None,
) -> tuple[torch.Tensor | None, ...]:
    """Invoke the opaque BF16 backward operator.

    Args:
        ctx: PyTorch library autograd context.
        grad_output_TD: Gradient of the public output.
        unused_state_gradients: Gradients for private non-differentiable state.

    Returns:
        Gradients matching the BF16 forward operator inputs.
    """
    del unused_state_gradients
    w13_EFD, w2_EDF, topk_expert_ids_TK, topk_scores_TK, *state_tensors = (
        ctx.saved_tensors
    )
    inplace_wgrad_accum = getattr(ctx, "inplace_wgrad_accum", False)
    w13_compute_EFD = (
        w13_EFD.view(ctx.w13_compute_shape) if inplace_wgrad_accum else w13_EFD
    )
    w2_compute_EDF = (
        w2_EDF.view(ctx.w2_compute_shape) if inplace_wgrad_accum else w2_EDF
    )
    context = _get_context(ctx.context_id)
    wgrad_destinations = None
    if inplace_wgrad_accum:
        wgrad_destinations = _resolve_parameter_grad_destinations(
            (ctx.w13_param_ref, ctx.w2_param_ref),
            ctx.w13_compute_shape,
            ctx.w2_compute_shape,
            context.config.wgrad_dtype,
        )
        wgrad_output_dtype = wgrad_destinations[0].dtype
    else:
        wgrad_output_dtype = context.config.wgrad_dtype or w13_compute_EFD.dtype
    registered_args = (
        topk_expert_ids_TK,
        topk_scores_TK,
        state_tensors,
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
        grad_x_TD, grad_topk_scores_TK = _bf16_backward_accumulate_op(
            grad_output_TD,
            w13_compute_EFD,
            w2_compute_EDF,
            accumulator_grad_w13_EFD,
            accumulator_grad_w2_EDF,
            *registered_args,
        )
        grad_w13_EFD = grad_w2_EDF = None
    else:
        grad_x_TD, grad_topk_scores_TK, grad_w13_EFD, grad_w2_EDF = _bf16_backward_op(
            grad_output_TD,
            w13_compute_EFD,
            w2_compute_EDF,
            *registered_args,
        )
        if inplace_wgrad_accum:
            grad_w13_EFD = grad_w13_EFD.view(ctx.w13_input_shape)
            grad_w2_EDF = grad_w2_EDF.view(ctx.w2_input_shape)
    return (
        grad_x_TD,
        None,  # topk_expert_ids_TK
        grad_topk_scores_TK,
        grad_w13_EFD,
        grad_w2_EDF,
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
        None,  # validate_expert_ids
        None,  # context_id
    )


class _RegisteredBf16Autograd(torch.autograd.Function):
    """Connect the registered BF16 forward to standard gradient ownership.

    The forward remains opaque to FakeTensor tracing. Backward resolves the two
    parameter gradients in Python, then invokes either the functional backward
    or its deterministic accumulating counterpart with explicit tensor inputs.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x_TD: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        w13_EFD: torch.Tensor,
        w2_EDF: torch.Tensor,
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
        swiglu_clip_stats_out_3: torch.Tensor | None,
        swiglu_clip_limit: float,
        options: ExecutionOptions,
    ) -> torch.Tensor:
        """Run the opaque forward and retain parameter-gradient owners."""
        forward_args = (
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            w13_EFD,
            w2_EDF,
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
            options.validate_expert_ids,
            context_id,
        )
        if swiglu_clip_stats_out_3 is None:
            output = _bf16_forward_op(*forward_args)
        else:
            output_with_stats = _bf16_forward_with_clip_stats_op(
                *forward_args[:-1],
                swiglu_clip_limit,
                context_id,
            )
            swiglu_clip_stats_out_3.copy_(output_with_stats[2])
            output = output_with_stats[:2]
        _bf16_setup_context(
            ctx,
            forward_args,
            output,
            mark_state_non_differentiable=False,
        )
        ctx.inplace_wgrad_accum = options.inplace_wgrad_accum
        if options.inplace_wgrad_accum:
            context = _get_context(context_id)
            w13_compute_EFD, w2_compute_EDF = _reshape_weights(
                w13_EFD,
                w2_EDF,
                context,
            )
            owners = options.wgrad_parameter_owners
            ctx.w13_input_shape = w13_EFD.shape
            ctx.w2_input_shape = w2_EDF.shape
            ctx.w13_compute_shape = w13_compute_EFD.shape
            ctx.w2_compute_shape = w2_compute_EDF.shape
            ctx.w13_param_ref = _weak_parameter_ref(
                w13_EFD,
                None if owners is None else owners[0],
            )
            ctx.w2_param_ref = _weak_parameter_ref(
                w2_EDF,
                None if owners is None else owners[1],
            )
        return output[0]

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[Any, ...]:
        """Run the registered functional or accumulating BF16 backward."""
        (grad_output_TD,) = grad_outputs
        return (*_bf16_autograd_backward(ctx, grad_output_TD, None), None, None, None)


_bf16_forward_op.register_autograd(
    _bf16_autograd_backward,
    setup_context=_bf16_setup_context,
)


def _bf16_clip_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[torch.Tensor, list[torch.Tensor], torch.Tensor],
) -> None:
    """Save functional BF16 state and mark returned clip counters as metadata."""
    # The clip-enabled wrapper inserts swiglu_clip_limit at input index 17.
    # Remove exactly that non-differentiable scalar before sharing base setup.
    base_inputs = (*inputs[:17], *inputs[18:])
    _bf16_setup_context(ctx, base_inputs, output[:2])
    ctx.mark_non_differentiable(output[2])


def _bf16_clip_autograd_backward(
    ctx: Any,
    grad_output_TD: torch.Tensor,
    unused_state_gradients: list[torch.Tensor | None] | None,
    unused_clip_stats_gradient: torch.Tensor | None,
) -> tuple[torch.Tensor | None, ...]:
    """Run BF16 backward and omit gradients for clip configuration.

    Args:
        ctx: PyTorch library autograd context.
        grad_output_TD: Gradient of the public output.
        unused_state_gradients: Gradients for private non-differentiable state.
        unused_clip_stats_gradient: Gradient for non-differentiable counters.

    Returns:
        Gradients matching the clip-enabled BF16 forward inputs.
    """
    del unused_clip_stats_gradient
    gradients = _bf16_autograd_backward(
        ctx,
        grad_output_TD,
        unused_state_gradients,
    )
    # Restore the omitted scalar's gradient slot at the same input index.
    return (*gradients[:17], None, *gradients[17:])


_bf16_forward_with_clip_stats_op.register_autograd(
    _bf16_clip_autograd_backward,
    setup_context=_bf16_clip_setup_context,
)


def routed_experts(  # noqa: C901
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_weight: torch.Tensor | PreparedWeight,
    w2_weight: torch.Tensor | PreparedWeight,
    context: Context,
    *,
    options: ExecutionOptions | None = None,
) -> torch.Tensor:
    """Apply BF16 or block-scaled distributed MoE to routed tokens.

    Args:
        x_TD: Contiguous CUDA BF16 local tokens with shape ``[T, D]``, where
            every EP rank uses the same
            ``1 <= T <= context.max_num_local_input_tokens`` for this call.
        topk_expert_ids_TK: Contiguous CUDA int32 or int64 global expert IDs
            with shape ``[T, K]`` and values in ``[-1, num_experts)``. ``-1``
            disables that route. Nonnegative IDs in each row must be distinct;
            repeated ``-1`` padding is allowed. Callers enforce uniqueness.
        topk_scores_TK: Contiguous CUDA BF16 or FP32 expert weights with shape
            ``[T, K]``. Callers must use score zero for ``-1`` routes.
        w13_weight: Local BF16 gate/up weight with shape
            ``[E_local, 2, F, D]``, ``[E_local, 2F, D]``, or
            ``[E_local * 2F, D]``, or a prepared MXFP8/NVFP4 operand whose
            source has one of those shapes. NVFP4 requires prepared weights
            and an inference context.
        w2_weight: Local BF16 down weight with shape ``[E_local, D, F]`` or
            ``[E_local * D, F]``, or a matching prepared MXFP8/NVFP4 operand.
        context: Factory-created execution context for these static shapes.
        options: Per-call execution controls. ``None`` selects graph-friendly
            defaults with no callback or clip-statistics side effect.

    Note:
        This is collective over ``context.group``. Every group rank must call
        it in the same order and the same number of times with compatible
        context shapes and execution options. Rank-divergent invocation can
        leave peers waiting at the operation's device-side barriers.
        Execution mutates context-owned communication, planner, activation, and
        scratch storage. Optional clip counters and WGRAD destinations are also
        caller-visible mutations. Standard ``parameter.grad`` accumulation uses
        the registered path. Interleaved Python callbacks require eager
        execution. FakeTensor metadata, non-strict ``make_fx``, activation
        checkpointing, and CUDA graphs are supported; full ``torch.compile``
        and strict export are not.

        Unequal physical token counts across EP ranks trigger a device-side
        trap before routing metadata is generated. This terminates the
        distributed iteration and invalidates the CUDA execution context;
        callers cannot catch the failure and continue.

        Exceeding total scratch capacity triggers a production device trap in
        the routing prefix or direct-decode kernel before final metadata is
        published. The overflowing rank reports required receive rows and
        total capacity; peers may time out or be terminated by the job
        supervisor. The failed CUDA context cannot be reused.

        Backward retention is selected per invocation. A training context used
        under ``torch.no_grad()`` keeps training-compatible forward behavior,
        uses shared scratch only, and leaves activation-slot state unchanged;
        it does not acquire inference-only behavior from grad mode.

    Returns:
        Fresh, non-aliasing local BF16 output with shape ``[T, D]``.

    Raises:
        TypeError: If an input dtype is unsupported.
        ValueError: If an input shape, layout, device, or expert partition does
            not match ``context``.
        RuntimeError: If gradients are requested from an inference context.

    """
    options = _resolve_execution_options(options)
    clip_stats_out_3 = options.swiglu_clip_stats_out_3
    if clip_stats_out_3 is not None:
        if clip_stats_out_3.device != x_TD.device:
            raise ValueError(
                "swiglu_clip_stats_out_3 must be on the Dist-MoE input device"
            )
        if context.config.block_scaled is not None:
            raise ValueError("SwiGLU clip statistics are supported only by BF16")
    _validate_routing_shape(
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        context,
    )
    prepared = isinstance(w13_weight, PreparedWeight) or isinstance(
        w2_weight, PreparedWeight
    )
    if prepared and not (
        isinstance(w13_weight, PreparedWeight) and isinstance(w2_weight, PreparedWeight)
    ):
        raise TypeError(
            "w13_weight and w2_weight must either both be prepared or both be tensors"
        )
    if prepared:
        if context.config.block_scaled is None:
            raise ValueError("prepared weights require block-scaled execution")
        if options.weights_preprocess_fn is not None:
            raise ValueError(
                "prepared weights cannot be combined with weights_preprocess_fn"
            )
        assert isinstance(w13_weight, PreparedWeight)
        assert isinstance(w2_weight, PreparedWeight)
        w13_EFD, w2_EDF = _reshape_weights(
            w13_weight.source,
            w2_weight.source,
            context,
        )
        logical_w13_weight = logical_w2_weight = None
    else:
        assert isinstance(w13_weight, torch.Tensor)
        assert isinstance(w2_weight, torch.Tensor)
        logical_w13_weight, logical_w2_weight = w13_weight, w2_weight
        w13_EFD, w2_EDF = _reshape_weights(w13_weight, w2_weight, context)
    if options.weights_preprocess_fn is None and not prepared:
        _validate_inputs(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            w13_EFD,
            w2_EDF,
            context,
        )
    postprocess = options.experts_output_postprocess
    rmsnorm_weight_D = (
        postprocess.weight if isinstance(postprocess, RMSNormPostprocess) else None
    )
    if context.config.inference and any(
        tensor is not None and tensor.requires_grad
        for tensor in (
            x_TD,
            topk_scores_TK,
            w13_EFD,
            w2_EDF,
            rmsnorm_weight_D,
        )
    ):
        raise RuntimeError("inference contexts do not support autograd")
    save_for_backward = torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad
        for tensor in (
            x_TD,
            topk_scores_TK,
            w13_EFD,
            w2_EDF,
            rmsnorm_weight_D,
        )
    )
    if context.config.block_scaled is not None:
        # Importing registers the separate dispatcher schema without adding
        # low-precision compiler dependencies to BF16-only processes.
        from . import _blockscaled  # noqa: F401
        from ._blockscaled import _run_blockscaled

        return _run_blockscaled(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            w13_weight,
            w2_weight,
            context,
            options,
            save_for_backward=save_for_backward,
        )
    if prepared:
        raise ValueError("prepared weights require block-scaled execution")
    activation_buffer = context.activation_buffer
    assert activation_buffer is not None
    tracing = torch.compiler.is_compiling() or _get_current_dispatch_mode() is not None
    if save_for_backward and not options.requires_eager:
        assert logical_w13_weight is not None and logical_w2_weight is not None
        return _RegisteredBf16Autograd.apply(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            logical_w13_weight,
            logical_w2_weight,
            *_registered_rmsnorm_args(postprocess),
            activation_buffer.activation_slot_id_1,
            activation_buffer._num_moe_layers_in_selected_slot,
            context.context_id,
            clip_stats_out_3,
            options.swiglu_clip_limit,
            options,
        )
    if options.requires_eager:
        if tracing:
            raise RuntimeError(
                "DistMoE Python weight, expert-output, and WGRAD callbacks "
                "require eager execution"
            )
        assert logical_w13_weight is not None and logical_w2_weight is not None
        return _Bf16Autograd.apply(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            logical_w13_weight,
            logical_w2_weight,
            activation_buffer.buffer,
            activation_buffer.activation_slot_id_1,
            activation_buffer._num_moe_layers_in_selected_slot,
            context.buffers.routing.local(),
            context.buffers.dispatch.local(),
            context.buffers.combine.local(),
            context.context_id,
            options,
            save_for_backward,
        )
    rmsnorm_args = _registered_rmsnorm_args(postprocess)
    if save_for_backward:
        assert logical_w13_weight is not None and logical_w2_weight is not None
        forward_args = (
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            logical_w13_weight,
            logical_w2_weight,
            *rmsnorm_args,
            activation_buffer.activation_slot_id_1,
            activation_buffer._num_moe_layers_in_selected_slot,
            options.validate_expert_ids,
        )
        if clip_stats_out_3 is None:
            output_TD, _state_tensors = _bf16_forward_op(
                *forward_args,
                context.context_id,
            )
            return output_TD
        output_TD, _state_tensors, computed_clip_stats_3 = (
            _bf16_forward_with_clip_stats_op(
                *forward_args,
                options.swiglu_clip_limit,
                context.context_id,
            )
        )
        clip_stats_out_3.copy_(computed_clip_stats_3)
        return output_TD
    return torch.ops.dist_moe.bf16.default(
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        w13_EFD,
        w2_EDF,
        *rmsnorm_args,
        activation_buffer.buffer,
        activation_buffer.activation_slot_id_1,
        activation_buffer._num_moe_layers_in_selected_slot,
        context.buffers.routing.local(),
        context.buffers.dispatch.local(),
        context.buffers.combine.local(),
        clip_stats_out_3,
        options.swiglu_clip_limit,
        options.validate_expert_ids,
        context.context_id,
    )


def relayout_to_block_major(weight: torch.Tensor) -> None:
    """Rewrite each expert's ``[N, K]`` weight in place as ``[K / 64, N, 64]``.

    This is the layout a context with ``Config.block_major_weights`` reads.
    The shape is unchanged, so afterwards only Dist-MoE may read the weight, and
    a second call permutes it again. With ``K = 128``::

        row-major:    r0[0:64] r0[64:128] r1[0:64] r1[64:128] ...
        block-major:  r0[0:64] r1[0:64] ... r(N-1)[0:64] r0[64:128] r1[64:128] ...

    Args:
        weight: Contiguous expert weight ``[E, N, K]``: ``[E, 2F, D]`` for W13,
            ``[E, D, F]`` for W2. K must be a multiple of 64.

    Raises:
        ValueError: If ``weight`` is not a contiguous 3-D tensor with K a
            multiple of 64.
    """
    if (
        weight.ndim != 3
        or not weight.is_contiguous()
        or weight.shape[2] % BLOCK_MAJOR_K != 0
    ):
        raise ValueError(
            "relayout_to_block_major needs a contiguous [E, N, K] weight whose K "
            "is a multiple of 64"
        )
    num_rows, k = weight.shape[1:]
    # One expert at a time: the temp is one [N, K] weight, not a second copy.
    block_BN64 = weight.new_empty(k // BLOCK_MAJOR_K, num_rows, BLOCK_MAJOR_K)
    with torch.inference_mode():
        for weight_NK in weight:
            block_BN64.copy_(
                weight_NK.view(num_rows, k // BLOCK_MAJOR_K, BLOCK_MAJOR_K).transpose(
                    0, 1
                )
            )
            weight_NK.view(block_BN64.shape).copy_(block_BN64)


def prepare_block_scaled_weight(
    weight: torch.Tensor,
    config: BlockScaledConfig,
    *,
    inference: bool = False,
    fuse_swiglu: bool = False,
    out: PreparedWeight | None = None,
) -> PreparedWeight:
    """Prepare caller-owned block-scaled FPROP and DGRAD operands.

    Args:
        weight: Contiguous CUDA BF16 expert weight with shape ``[E, N, K]``.
            Use ``[E, 2F, D]`` for W13 and ``[E, D, F]`` for W2.
            MXFP8 requires N and K divisible by128 for its blocked scale
            orientations. NVFP4 preparation requires an even K.
        config: Block-scaled execution policy.
        inference: Whether to prepare FPROP-only inference operands. MXFP8
            training requires both FPROP and DGRAD scale orientations.
        fuse_swiglu: Prepare W13 for fused inference SwiGLU by interleaving
            quantized gate/up rows and their scales in groups of eight.
            Requires ``inference=True``, ``config.pipeline="mega"`` and W13
            shape ``[E, 2F, D]``. Do not enable this for W2. The logical source
            weight remains unchanged; the result cannot be used for training
            backward or the staged pipeline.
        out: Optional MXFP8 training storage to refill for the same logical
            weight, shape, format, and device. Refill preserves the identical
            storage tensors and returns a prepared wrapper associated with
            ``weight``. It is unavailable for inference and NVFP4.

    Returns:
        Prepared weight accepted by :func:`routed_experts`.

    Raises:
        ValueError: If ``weight`` or ``out`` violates the selected format,
            shape, dtype, device, contiguity, inference, or refill contract.
    """
    from ._blockscaled import _prepare_blockscaled_weight as prepare

    return prepare(
        weight, config, inference=inference, fuse_swiglu=fuse_swiglu, out=out
    )
