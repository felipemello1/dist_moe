# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared execution policies for the CuTe distributed-MoE implementations."""

from __future__ import annotations

import dataclasses
import enum
import math
import weakref
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import torch
from torch.distributed.tensor import DTensor

from ._postprocess import (
    _postprocess_requires_eager,
    RMSNormPostprocess,
)
from .formats import BlockScaledFormat as _KernelBlockScaledFormat

WgradName: TypeAlias = Literal["w13", "w2"]
WeightPreprocessFn: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
WgradPostprocessFn: TypeAlias = Callable[[WgradName, torch.Tensor], torch.Tensor | None]
WgradDestinationFn: TypeAlias = Callable[
    [WgradName, torch.Size, torch.dtype, torch.device],
    tuple[torch.Tensor, bool],
]


class BlockScaledFormat(enum.StrEnum):
    """Block-scaled operand formats supported by the public API.

    Attributes:
        MXFP8_E4M3: FP8 E4M3 operands with E8M0 scale factors for training and
            inference.
        NVFP4: FP4 E2M1 operands with E4M3 scale factors for inference only.
    """

    MXFP8_E4M3 = _KernelBlockScaledFormat.MXFP8_E4M3.value
    NVFP4 = _KernelBlockScaledFormat.NVFP4.value


class _KernelAutogradContext:
    """Collect state while reusing eager autograd kernel orchestration.

    Mutable custom operators cannot use ``torch.library.register_autograd``.
    Their CUDA implementations call the existing eager forward and backward
    methods directly through this minimal context, while the compiler-facing
    autograd function contains only opaque custom-op calls.
    """

    forward_offsets: torch.Tensor
    w13_input_shape: torch.Size
    w2_input_shape: torch.Size

    def __init__(self) -> None:
        """Initialize an empty saved-tensor collection."""
        self.saved_tensors: tuple[torch.Tensor | None, ...] = ()

    def save_for_backward(self, *tensors: torch.Tensor | None) -> None:
        """Retain tensors produced by the CUDA forward implementation.

        Args:
            *tensors: Tensors or absent optional state consumed by backward.
        """
        self.saved_tensors = tensors

    def mark_non_differentiable(self, *tensors: torch.Tensor) -> None:
        """Accept the inference-only autograd context operation.

        Args:
            *tensors: Outputs that eager autograd marks non-differentiable.
        """
        del tensors

    def set(self, **values: Any) -> None:
        """Attach non-tensor state expected by an eager kernel method.

        Args:
            **values: Attribute names and values to attach.
        """
        for name, value in values.items():
            setattr(self, name, value)


@dataclasses.dataclass(frozen=True)
class ExecutionOptions:
    """Per-invocation controls layered on a reusable DistMoE context.

    The default value preserves the registered execution path. Standard
    ``parameter.grad`` accumulation also uses that path; only callbacks whose
    semantics execute inside the kernel schedule require eager orchestration.

    Args:
        inplace_wgrad_accum: Fuse serialized WGRAD contributions into the
            standard ``parameter.grad`` buffer. The registered backward
            operation receives that local buffer explicitly.
        wgrad_parameter_owners: Optional outer parameters that own W13 and W2
            gradients when the compute tensors are local views. Execution
            retains only weak references to these parameters.
        weights_preprocess_fn: Materialize BF16 compute weights from compact
            source weights in forward and again immediately before backward.
            It must be deterministic and return a contiguous CUDA tensor on
            the source device with a shape accepted for the same projection.
        experts_output_postprocess: Optional deterministic callable applied to
            each route-wise expert output before score weighting and top-k
            reduction, or ``RMSNormPostprocess`` for fused RMSNorm plus top-k
            reduction. A callable must preserve shape,
            device, contiguity, and dtype across forward and backward replay.
            Closure parameters accumulate gradients through nested backward
            and are not explicit inputs to the outer custom autograd function.
        wgrad_postprocess_fn: Hook invoked as soon as each WGRAD is available.
            A replacement must preserve shape, device, and contiguity and use
            BF16 or FP32. Returning ``None`` declares that the callback consumed
            and now owns the gradient.
        swiglu_clip_stats_out_3: Optional contiguous CUDA FP32 tensor with
            shape ``[3]``. Plain BF16 SwiGLU rewrites it with gate-over-limit,
            absolute-up-over-limit, and valid-element counts. Training returns
            functional counter state and performs an explicit graph-visible
            copy into this tensor; inference passes it to its mutable op.
            Callers own aggregation.
        swiglu_clip_limit: Finite threshold used for strict clip comparisons.
        wgrad_destination_fn: Eager-only hook invoked before each WGRAD kernel.
            It receives the projection name, grouped output shape, dtype, and
            device. It returns a contiguous tensor with that exact shape,
            dtype, and device plus whether the kernel should overwrite or
            accumulate into its existing contents. The callback owns and
            exposes that mutated destination, so WGRAD is not returned to
            autograd.
        validate_expert_ids: Whether routing publication checks every expert ID
            is in ``[-1, num_experts)`` before routing. Training folds the check
            into fused publication; inference performs an opt-in device check
            before native publication. Violations device-trap.
        zero_out_padded_callback_inputs: Whether invalid route rows are zeroed
            before Python postprocess and observer callbacks. This is required
            when callbacks coexist with ``-1`` routes unless the caller
            otherwise guarantees callback-safe storage.
        w13_bias_EF: Optional gate/up bias ``[E_local, 2F]``; row ``e`` is
            ``[gate (F) | up (F)]``, like ``w13_EFD[e]``.
        w2_bias_ED: Optional down-projection bias ``[E_local, D]``. Both
            biases: BF16 inference only (no training, no block-scaled
            experts). Each is added in fp32 and rounded to BF16, like a BF16
            ``projection + bias``.
    """

    inplace_wgrad_accum: bool = False
    wgrad_parameter_owners: tuple[torch.Tensor, torch.Tensor] | None = (
        dataclasses.field(default=None, repr=False, compare=False)
    )
    weights_preprocess_fn: WeightPreprocessFn | None = None
    experts_output_postprocess: (
        Callable[[torch.Tensor], torch.Tensor] | RMSNormPostprocess | None
    ) = None
    wgrad_postprocess_fn: WgradPostprocessFn | None = None
    swiglu_clip_stats_out_3: torch.Tensor | None = None
    swiglu_clip_limit: float = 7.0
    wgrad_destination_fn: WgradDestinationFn | None = None
    validate_expert_ids: bool = False
    zero_out_padded_callback_inputs: bool = False
    w13_bias_EF: torch.Tensor | None = None
    w2_bias_ED: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate graph-visible observability and WGRAD controls.

        Raises:
            TypeError: If a tensor, threshold, callback, or owner has an
                unsupported type.
            ValueError: If clip state is invalid or WGRAD ownership modes are
                inconsistent.
        """
        self._validate_clip_observability()
        self._validate_wgrad_ownership()
        for name in ("validate_expert_ids", "zero_out_padded_callback_inputs"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

    def _validate_clip_observability(self) -> None:
        """Validate the optional graph-visible SwiGLU clip counters."""
        output_3 = self.swiglu_clip_stats_out_3
        if output_3 is not None:
            if not isinstance(output_3, torch.Tensor):
                raise TypeError("swiglu_clip_stats_out_3 must be a tensor")
            if output_3.dtype is not torch.float32:
                raise TypeError("swiglu_clip_stats_out_3 must have dtype float32")
            if output_3.shape != (3,) or not output_3.is_contiguous():
                raise ValueError(
                    "swiglu_clip_stats_out_3 must be contiguous with shape [3]"
                )
            if output_3.device.type != "cuda":
                raise ValueError("swiglu_clip_stats_out_3 must be a CUDA tensor")
        if not isinstance(self.swiglu_clip_limit, (int, float)) or isinstance(
            self.swiglu_clip_limit, bool
        ):
            raise TypeError("swiglu_clip_limit must be a real number")
        if not math.isfinite(self.swiglu_clip_limit):
            raise ValueError("swiglu_clip_limit must be finite")

    def _validate_wgrad_ownership(self) -> None:
        """Validate mutually exclusive WGRAD destination ownership modes."""
        if not isinstance(self.inplace_wgrad_accum, bool):
            raise TypeError("inplace_wgrad_accum must be a bool")
        if self.inplace_wgrad_accum and self.wgrad_postprocess_fn is not None:
            raise ValueError(
                "inplace_wgrad_accum cannot be combined with wgrad_postprocess_fn"
            )
        if self.wgrad_destination_fn is not None and (
            self.inplace_wgrad_accum or self.wgrad_postprocess_fn is not None
        ):
            raise ValueError(
                "wgrad_destination_fn cannot be combined with "
                "inplace_wgrad_accum or wgrad_postprocess_fn"
            )
        if self.wgrad_destination_fn is not None and not callable(
            self.wgrad_destination_fn
        ):
            raise TypeError("wgrad_destination_fn must be callable")
        if self.wgrad_parameter_owners is not None:
            if not self.inplace_wgrad_accum:
                raise ValueError(
                    "wgrad_parameter_owners requires inplace_wgrad_accum=True"
                )
            if (
                not isinstance(self.wgrad_parameter_owners, tuple)
                or len(self.wgrad_parameter_owners) != 2
                or not all(
                    isinstance(owner, torch.Tensor)
                    for owner in self.wgrad_parameter_owners
                )
            ):
                raise TypeError(
                    "wgrad_parameter_owners must be a pair of W13 and W2 tensors"
                )

    @property
    def requires_eager(self) -> bool:
        """Return whether these controls require Python execution.

        Returns:
            ``True`` when a Python callback must run inside kernel orchestration.
        """
        return (
            self.weights_preprocess_fn is not None
            or _postprocess_requires_eager(self.experts_output_postprocess)
            or self.wgrad_postprocess_fn is not None
            or self.wgrad_destination_fn is not None
        )


@dataclasses.dataclass(frozen=True, init=False)
class PreparedWeight:
    """Caller-owned block-scaled operands for one expert projection.

    Args:
        source: Logical high-precision weight identity. It receives WGRAD for
            training formats; inference-only formats retain it only for
            ownership and lifecycle coordination. Prepared execution does not
            require readable source storage; Python preprocessing callbacks
            that read the source still require it.
        format: Block-scaled format used by every quantized operand.
        fprop_data: Format-specific contiguous CUDA FPROP qdata. MXFP8 stores
            E4M3 values; NVFP4 stores two packed E2M1 values per byte.
        fprop_scale: Contiguous CUDA scale storage in the blocked layout
            consumed directly by the FPROP kernel.
        dgrad_data: Contiguous transposed MXFP8 E4M3 DGRAD qdata. It is required
            for MXFP8 training and absent for inference-only NVFP4.
        dgrad_scale: Contiguous blocked DGRAD scales with the orientation
            matching ``dgrad_data``; absent for NVFP4.
        global_scale: Optional per-expert FP32 global scale required by NVFP4.
        global_scale_inv: Optional precomputed reciprocal of ``global_scale``.
            Fused NVFP4 inference consumes this directly so CUDA graph replay
            does not launch a reciprocal on every invocation.
    """

    source: torch.Tensor
    format: BlockScaledFormat
    fprop_data: torch.Tensor
    fprop_scale: torch.Tensor
    dgrad_data: torch.Tensor | None
    dgrad_scale: torch.Tensor | None
    global_scale: torch.Tensor | None = None
    global_scale_inv: torch.Tensor | None = None
    _interleaved_fc13: bool = False

    def __init__(self) -> None:
        """Reject construction outside :func:`prepare_block_scaled_weight`.

        Raises:
            TypeError: Always. The factory validates all layouts together.
        """
        raise TypeError("use prepare_block_scaled_weight()")

    @classmethod
    def _create(
        cls,
        *,
        source: torch.Tensor,
        format: BlockScaledFormat,
        fprop_data: torch.Tensor,
        fprop_scale: torch.Tensor,
        dgrad_data: torch.Tensor | None,
        dgrad_scale: torch.Tensor | None,
        global_scale: torch.Tensor | None = None,
        global_scale_inv: torch.Tensor | None = None,
        interleaved_fc13: bool = False,
    ) -> PreparedWeight:
        """Construct one validated prepared-weight owner.

        Args:
            source: Logical high-precision weight identity. Training formats
                use it as the WGRAD owner; inference-only formats retain it for
                ownership and lifecycle coordination.
            format: Public block-scaled format shared by all operands.
            fprop_data: Quantized forward weight data.
            fprop_scale: Forward-oriented scale storage.
            dgrad_data: Optional quantized DGRAD weight data.
            dgrad_scale: Optional DGRAD-oriented scale storage.
            global_scale: Optional per-expert global scale.
            global_scale_inv: Optional reciprocal global scale.
            interleaved_fc13: Whether FPROP rows contain adjacent gate/up
                groups for the inference-only fused SwiGLU kernels.

        Returns:
            Factory-owned prepared-weight state.
        """
        prepared = object.__new__(cls)
        for name, value in (
            ("source", source),
            ("format", format),
            ("fprop_data", fprop_data),
            ("fprop_scale", fprop_scale),
            ("dgrad_data", dgrad_data),
            ("dgrad_scale", dgrad_scale),
            ("global_scale", global_scale),
            ("global_scale_inv", global_scale_inv),
            ("_interleaved_fc13", interleaved_fc13),
        ):
            object.__setattr__(prepared, name, value)
        return prepared

    def storage_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return one representative tensor per owned quantized storage.

        Storage identity, rather than its current address or size, determines
        uniqueness. The returned membership therefore remains stable after an
        external owner releases storage with ``resize_(0)``. A releasing owner
        must record each storage's required byte count before the first resize.

        Returns:
            Unique data, scale, and global-scale storage representatives.
        """
        tensors = (
            self.fprop_data,
            self.fprop_scale,
            self.dgrad_data,
            self.dgrad_scale,
            self.global_scale,
            self.global_scale_inv,
        )
        unique_tensors: list[torch.Tensor] = []
        unique_storages: list[torch.UntypedStorage] = []
        for tensor in tensors:
            if tensor is None:
                continue
            storage = tensor.untyped_storage()
            if any(storage is existing for existing in unique_storages):
                continue
            unique_storages.append(storage)
            unique_tensors.append(tensor)
        return tuple(unique_tensors)

    def _with_source(self, source: torch.Tensor) -> PreparedWeight:
        """Return the same prepared storage associated with a new source."""
        return self._create(
            source=source,
            format=self.format,
            fprop_data=self.fprop_data,
            fprop_scale=self.fprop_scale,
            dgrad_data=self.dgrad_data,
            dgrad_scale=self.dgrad_scale,
            global_scale=self.global_scale,
            global_scale_inv=self.global_scale_inv,
            interleaved_fc13=self._interleaved_fc13,
        )


_DEFAULT_EXECUTION_OPTIONS = ExecutionOptions()


def _resolve_execution_options(
    options: ExecutionOptions | None,
) -> ExecutionOptions:
    """Return caller controls or the shared immutable default.

    Args:
        options: Caller-provided controls or ``None``.

    Returns:
        Validated per-call execution controls.
    """
    if options is None:
        return _DEFAULT_EXECUTION_OPTIONS
    if not isinstance(options, ExecutionOptions):
        raise TypeError("options must be ExecutionOptions or None")
    return options


@dataclasses.dataclass(frozen=True)
class _WgradDestination:
    """One WGRAD destination resolved before kernel launches.

    Args:
        parameter: Leaf parameter whose gradient field owns the result, or
            ``None`` when an integration owns ``output`` directly.
        output: Existing gradient view, or ``None`` before the first write.
        accumulate: Whether the kernel must add to ``output``.
        dtype: Resolved kernel output dtype.
    """

    parameter: torch.Tensor | None
    output: torch.Tensor | None
    accumulate: bool
    dtype: torch.dtype


def _parameter_base(tensor: torch.Tensor) -> torch.Tensor:
    """Return the leaf owning a possibly reshaped expert-weight view.

    Args:
        tensor: Expert weight or view passed to DistMoE.

    Returns:
        Outermost tensor base that owns the autograd gradient field.
    """
    base = tensor
    while base._base is not None:
        base = base._base
    return base


def _weak_parameter_ref(
    tensor: torch.Tensor,
    owner: torch.Tensor | None = None,
) -> weakref.ReferenceType[torch.Tensor]:
    """Create a weak reference to the leaf owning an expert weight.

    Args:
        tensor: Expert weight or view passed to DistMoE.
        owner: Optional outer parameter whose ``grad`` field owns the result.

    Returns:
        Weak reference suitable for saving on an autograd context.

    Raises:
        ValueError: If the resolved owner cannot own an autograd gradient.
    """
    parameter = _parameter_base(tensor) if owner is None else owner
    if not parameter.is_leaf:
        raise ValueError(
            "in-place WGRAD accumulation requires a view of an autograd leaf"
        )
    if not parameter.requires_grad:
        raise ValueError(
            "in-place WGRAD accumulation requires a parameter that needs gradients"
        )
    return weakref.ref(parameter)


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local tensor represented by a Tensor or DTensor.

    Args:
        tensor: Tensor whose local storage is required by a CUDA kernel.

    Returns:
        Local tensor for a DTensor, otherwise ``tensor`` itself.
    """
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _resolve_wgrad_dtype(
    parameter: torch.Tensor,
    output_dtype: torch.dtype | None,
) -> torch.dtype:
    """Resolve one parameter-owned WGRAD output dtype without mutation.

    A non-``None`` ``parameter.grad_dtype`` is the parameter's declaration and
    must agree with an explicit ``output_dtype``. Otherwise an explicit dtype
    wins, followed by existing gradient storage and then ``parameter.dtype``.
    Only BF16 and FP32 WGRAD are supported, and existing storage must match the
    resolved dtype exactly.

    Args:
        parameter: Leaf tensor whose standard ``grad`` field owns WGRAD.
        output_dtype: Optional explicit dtype selected by ``Config.wgrad_dtype``.

    Returns:
        BF16 or FP32 dtype for the WGRAD kernel output.

    Raises:
        TypeError: If the resolved dtype is not BF16 or FP32.
        RuntimeError: If an explicit dtype conflicts with the parameter
            declaration or existing gradient storage.
    """
    declared_grad_dtype = parameter.grad_dtype
    grad = parameter.grad
    if output_dtype is None:
        if declared_grad_dtype is not None:
            output_dtype = declared_grad_dtype
        elif grad is not None:
            output_dtype = grad.dtype
        else:
            output_dtype = parameter.dtype
    elif declared_grad_dtype is not None and output_dtype != declared_grad_dtype:
        raise RuntimeError(
            "wgrad_dtype conflicts with the parameter's declared grad_dtype"
        )
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            "resolved WGRAD dtype must be torch.bfloat16 or torch.float32, got "
            f"{output_dtype}"
        )
    if grad is not None and grad.dtype != output_dtype:
        raise RuntimeError(
            "existing parameter gradient dtype does not match the resolved WGRAD dtype"
        )
    return output_dtype


def _resolve_wgrad_output_dtype(
    destinations: tuple[_WgradDestination, _WgradDestination] | None,
    output_dtype: torch.dtype | None,
    fallback_dtype: torch.dtype,
) -> torch.dtype:
    """Resolve one kernel dtype from optional validated destinations.

    Args:
        destinations: Optional W13 and W2 destinations.
        output_dtype: Optional explicit functional output dtype.
        fallback_dtype: Compute-weight dtype used when neither is present.

    Returns:
        Shared W13/W2 destination dtype, explicit dtype, or fallback dtype.

    Raises:
        RuntimeError: If destination dtypes differ or conflict with the
            explicit dtype.
    """
    if destinations is None:
        return fallback_dtype if output_dtype is None else output_dtype
    if destinations[0].dtype != destinations[1].dtype:
        raise RuntimeError("W13 and W2 destinations have different dtypes")
    destination_dtype = destinations[0].dtype
    if output_dtype is not None and output_dtype != destination_dtype:
        raise RuntimeError("WGRAD output dtype does not match its destinations")
    return destination_dtype


def _resolve_parameter_grad(
    reference: weakref.ReferenceType[torch.Tensor],
    compute_shape: torch.Size,
    output_dtype: torch.dtype | None,
) -> _WgradDestination:
    """Resolve a standard parameter-gradient destination for one WGRAD.

    Args:
        reference: Weak reference to the leaf parameter.
        compute_shape: Three-dimensional grouped-GEMM output shape.
        output_dtype: Optional dtype selected by ``Config.wgrad_dtype``.

    Returns:
        Parameter, grouped output view, and kernel accumulation mode.

    Raises:
        RuntimeError: If the parameter was released or an existing gradient is
            incompatible with the unsharded WGRAD contract.
    """
    parameter = reference()
    if parameter is None:
        raise RuntimeError("expert parameter was released before backward")
    dtype = _resolve_wgrad_dtype(parameter, output_dtype)
    local_parameter = _local_tensor(parameter)
    if local_parameter.numel() != math.prod(compute_shape):
        raise RuntimeError("expert parameter shape does not match its WGRAD output")
    grad = parameter.grad
    if grad is None:
        return _WgradDestination(parameter, None, False, dtype)
    local_grad = _local_tensor(grad)
    if (
        local_grad.shape != local_parameter.shape
        or local_grad.dtype != dtype
        or local_grad.device != local_parameter.device
        or not local_grad.is_contiguous()
    ):
        raise RuntimeError(
            "in-place WGRAD accumulation requires a contiguous unsharded "
            "parameter gradient with matching shape, dtype, and device"
        )
    return _WgradDestination(parameter, local_grad.view(compute_shape), True, dtype)


def _resolve_parameter_grad_destinations(
    parameter_refs: tuple[
        weakref.ReferenceType[torch.Tensor],
        weakref.ReferenceType[torch.Tensor],
    ],
    w13_shape: torch.Size,
    w2_shape: torch.Size,
    output_dtype: torch.dtype | None,
) -> tuple[_WgradDestination, _WgradDestination]:
    """Resolve matching W13 and W2 parameter-gradient destinations.

    Args:
        parameter_refs: Weak W13 and W2 parameter references.
        w13_shape: Grouped W13 kernel output shape.
        w2_shape: Grouped W2 kernel output shape.
        output_dtype: Optional dtype selected by ``Config.wgrad_dtype``.

    Returns:
        Both validated destinations with one shared WGRAD dtype.

    Raises:
        RuntimeError: If W13 and W2 resolve to different dtypes. Both
            destinations are validated before either kernel can mutate state.
    """
    destinations = (
        _resolve_parameter_grad(parameter_refs[0], w13_shape, output_dtype),
        _resolve_parameter_grad(parameter_refs[1], w2_shape, output_dtype),
    )
    if destinations[0].dtype != destinations[1].dtype:
        raise RuntimeError("W13 and W2 parameters resolve to different WGRAD dtypes")
    return destinations


def _wgrad_accumulation_destinations(
    accumulator_grad_w13_EFD: torch.Tensor,
    accumulator_grad_w2_EDF: torch.Tensor,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    output_dtype: torch.dtype | None,
) -> tuple[_WgradDestination, _WgradDestination]:
    """Validate explicit destinations for one accumulating backward.

    Args:
        accumulator_grad_w13_EFD: Existing W13 gradient to update.
        accumulator_grad_w2_EDF: Existing W2 gradient to update.
        w13_EFD: Grouped W13 compute weight defining output shape and device.
        w2_EDF: Grouped W2 compute weight defining output shape and device.
        output_dtype: Requested WGRAD dtype, or ``None`` for weight dtype.

    Returns:
        Validated W13 and W2 accumulation destinations.

    Raises:
        RuntimeError: If either accumulator is incompatible. Both are validated
            before any WGRAD kernel can mutate them.
    """

    def resolve(
        accumulator_EFD: torch.Tensor,
        weight_EFD: torch.Tensor,
    ) -> _WgradDestination:
        dtype = weight_EFD.dtype if output_dtype is None else output_dtype
        if (
            accumulator_EFD.shape != weight_EFD.shape
            or accumulator_EFD.dtype != dtype
            or accumulator_EFD.device != weight_EFD.device
            or not accumulator_EFD.is_contiguous()
        ):
            raise RuntimeError(
                "WGRAD accumulator must be a contiguous tensor with the "
                "requested shape, dtype, and device"
            )
        return _WgradDestination(None, accumulator_EFD, True, dtype)

    destinations = (
        resolve(accumulator_grad_w13_EFD, w13_EFD),
        resolve(
            accumulator_grad_w2_EDF,
            w2_EDF,
        ),
    )
    if destinations[0].dtype != destinations[1].dtype:
        raise RuntimeError("W13 and W2 accumulators have different WGRAD dtypes")
    return destinations


def _resolve_wgrad_destination(
    fn: WgradDestinationFn,
    name: WgradName,
    weight: torch.Tensor,
    output_dtype: torch.dtype | None,
) -> _WgradDestination:
    """Resolve and validate an integration-owned WGRAD destination.

    Args:
        fn: Integration callback that owns the destination.
        name: Expert projection whose WGRAD will be produced.
        weight: Grouped compute weight defining shape and device.
        output_dtype: Requested WGRAD dtype, or ``None`` for weight dtype.

    Returns:
        Validated destination and kernel accumulation mode.

    Raises:
        TypeError: If the callback does not return ``(Tensor, bool)``.
        RuntimeError: If the returned tensor cannot be used by the kernel.
    """
    dtype = weight.dtype if output_dtype is None else output_dtype
    resolved = fn(name, weight.shape, dtype, weight.device)
    if (
        not isinstance(resolved, tuple)
        or len(resolved) != 2
        or not isinstance(resolved[0], torch.Tensor)
        or not isinstance(resolved[1], bool)
    ):
        raise TypeError("wgrad_destination_fn must return (Tensor, bool)")
    output, accumulate = resolved
    if (
        output.shape != weight.shape
        or output.dtype != dtype
        or output.device != weight.device
        or not output.is_contiguous()
    ):
        raise RuntimeError(
            "wgrad_destination_fn must return a contiguous tensor with the "
            "requested shape, dtype, and device"
        )
    return _WgradDestination(None, output, accumulate, dtype)


def _resolve_wgrad_destinations(
    *,
    inplace_wgrad_accum: bool,
    parameter_refs: tuple[
        weakref.ReferenceType[torch.Tensor],
        weakref.ReferenceType[torch.Tensor],
    ]
    | None,
    destination_fn: WgradDestinationFn | None,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    output_dtype: torch.dtype | None,
) -> tuple[_WgradDestination, _WgradDestination] | None:
    """Resolve both expert WGRAD destinations before either kernel writes.

    Args:
        inplace_wgrad_accum: Whether standard parameter gradients own output.
        parameter_refs: Weak W13 and W2 parameter references for that mode.
        destination_fn: Optional integration-owned destination resolver.
        w13_EFD: Grouped W13 compute weight.
        w2_EDF: Grouped W2 compute weight.
        output_dtype: Requested WGRAD dtype, or ``None`` for weight dtype.

    Returns:
        Both validated destinations, or ``None`` for ordinary autograd output.
    """
    if inplace_wgrad_accum:
        assert parameter_refs is not None
        return _resolve_parameter_grad_destinations(
            parameter_refs,
            w13_EFD.shape,
            w2_EDF.shape,
            output_dtype,
        )
    if destination_fn is None:
        return None
    return (
        _resolve_wgrad_destination(
            destination_fn,
            "w13",
            w13_EFD,
            output_dtype,
        ),
        _resolve_wgrad_destination(
            destination_fn,
            "w2",
            w2_EDF,
            output_dtype,
        ),
    )


def _postprocess_wgrad(
    fn: WgradPostprocessFn | None,
    name: WgradName,
    gradient: torch.Tensor | None,
) -> torch.Tensor | None:
    """Apply the optional immediate WGRAD postprocessing hook.

    Args:
        fn: Optional postprocessing callable.
        name: Projection whose gradient was produced.
        gradient: Materialized WGRAD, or ``None``.

    Returns:
        Original, replaced, consumed, or absent gradient.
    """
    if fn is None or gradient is None:
        return gradient
    return fn(name, gradient)
