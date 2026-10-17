# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Post-expert RMSNorm configuration and resolution for distributed MoE."""

import math
from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch

from .kernels._tensor_views import _views_overlap

_ExpertsOutputPostprocessFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class RMSNormPostprocess:
    """Configuration for fused RMSNorm and top-k reduction.

    Scalarless by default. An input-scale gamma — the
    ``rmsnorm((weight + gain_center) * x)`` form used by weighted post-expert
    norms — is supported for INFERENCE ONLY by setting ``weight`` (and, for
    zero-centered gamma, ``gain_center=1.0``): the gamma is applied inside the
    single fused kernel, with no extra pre-scaling launch. Training with a
    weighted norm must use a callback, or fold the effective gamma (including
    any gain center) into the corresponding output rows of ``w2`` for
    inference from pre-quantized weights.

    Supported hidden dimensions are multiples of 128 from 256 through 131072.
    Bitwise backward follows the standalone projection and score-gradient
    reduction orders, using register, chunked, or shared-gradient schedules
    selected from the input geometry.

    ``require_bitwise=False`` permits DIM clustering in forward and selects the
    HBM-optimized backward at compute capability 10 or newer. Both can reduce
    latency, but their results are only numerically equivalent to the standalone operators. Bitwise
    equivalence remains the default. Non-bitwise training is supported through
    DIM 12288; wider non-bitwise dimensions are inference-only.

    ``use_kahan`` selects Kahan-compensated sum-of-squares in every launch that
    computes the normalization, forward and backward alike.
    ``recompute_rstd`` omits the reciprocal RMS values from forward state and
    recomputes them from the saved expert output in backward.

    The observers exist because the fused path never routes the expert output
    through caller code: ``observe_expert_output_fn`` receives the pre-norm
    expert output on forward passes (including the backward's grad-disabled
    recompute replay), and ``observe_expert_output_grad_fn`` receives the
    saved/recomputed expert output together with its gradient once backward
    has computed it. Both are for stats/SDC logging only and must not mutate
    their arguments.

    Args:
        eps: Positive epsilon added to the mean square.
        norm_output_dtype: Dtype used for normalized route outputs.
        output_dtype: Dtype of the score-weighted token output.
        require_bitwise: Whether to preserve standalone reduction order.
        weight: Optional contiguous CUDA input-scale gamma with shape ``[D]``
            and the DistMoE input dtype. It is inference-only and cannot be
            combined with Kahan accumulation; full validation is deferred
            until :func:`dist_moe.routed_experts` knows the selected execution mode.
        gain_center: Constant added to ``weight`` before input scaling.
        use_kahan: Whether to use compensated sum-of-squares.
        observe_expert_output_fn: Optional read-only forward observer.
        observe_expert_output_grad_fn: Optional read-only backward observer.
        recompute_rstd: Whether backward recomputes reciprocal RMS values.
    """

    eps: float
    norm_output_dtype: torch.dtype
    output_dtype: torch.dtype
    require_bitwise: bool = True
    # Optional inference-only input-scale gamma: a contiguous [DIM] tensor
    # matching the DistMoE input dtype. ``gain_center`` is added to it inside
    # the kernel.
    weight: torch.Tensor | None = None
    gain_center: float = 0.0
    use_kahan: bool = False
    observe_expert_output_fn: Callable[[torch.Tensor], None] | None = None
    observe_expert_output_grad_fn: (
        Callable[[torch.Tensor, torch.Tensor], None] | None
    ) = None
    recompute_rstd: bool = False


_ExpertsOutputPostprocess = _ExpertsOutputPostprocessFn | RMSNormPostprocess | None


def _postprocess_requires_eager(postprocess: _ExpertsOutputPostprocess) -> bool:
    """Return whether postprocess execution requires Python-owned callables.

    Args:
        postprocess: Public expert-output postprocess value.

    Returns:
        ``True`` for a callback or an RMSNorm config with observer callbacks.
    """
    _validate_experts_output_postprocess_type(postprocess)
    return callable(postprocess) or (
        isinstance(postprocess, RMSNormPostprocess)
        and (
            postprocess.observe_expert_output_fn is not None
            or postprocess.observe_expert_output_grad_fn is not None
        )
    )


def _registered_rmsnorm_args(
    postprocess: _ExpertsOutputPostprocess,
) -> tuple[
    torch.Tensor | None,
    bool,
    float,
    torch.dtype,
    torch.dtype,
    bool,
    float,
    bool,
    bool,
]:
    """Flatten an RMSNorm config into graph-visible operator arguments.

    Args:
        postprocess: ``None`` or a callback-free ``RMSNormPostprocess``.

    Returns:
        Optional weight followed by the static kernel policy.

    Raises:
        TypeError: If a Python callback or observer is present.
    """
    if postprocess is None:
        return (
            None,
            False,
            0.0,
            torch.float32,
            torch.float32,
            True,
            0.0,
            False,
            False,
        )
    if not isinstance(postprocess, RMSNormPostprocess):
        raise TypeError("callable expert postprocessing requires eager execution")
    if _postprocess_requires_eager(postprocess):
        raise TypeError("RMSNorm observer callbacks require eager execution")
    return (
        postprocess.weight,
        True,
        postprocess.eps,
        postprocess.norm_output_dtype,
        postprocess.output_dtype,
        postprocess.require_bitwise,
        postprocess.gain_center,
        postprocess.use_kahan,
        postprocess.recompute_rstd,
    )


def _rmsnorm_from_registered_args(
    weight_D: torch.Tensor | None,
    enabled: bool,
    eps: float,
    norm_output_dtype: torch.dtype,
    output_dtype: torch.dtype,
    require_bitwise: bool,
    gain_center: float,
    use_kahan: bool,
    recompute_rstd: bool,
) -> RMSNormPostprocess | None:
    """Reconstruct the exact RMSNorm policy inside an opaque operator.

    Args:
        weight_D: Optional inference-only input-scale gamma with shape ``[D]``.
        enabled: Whether fused post-expert RMSNorm is selected.
        eps: RMSNorm epsilon.
        norm_output_dtype: Dtype of normalized route outputs.
        output_dtype: Dtype of the score-weighted reduction output.
        require_bitwise: Whether to preserve standalone reduction order.
        gain_center: Constant added to ``weight_D`` before input scaling.
        use_kahan: Whether to use compensated sum-of-squares.
        recompute_rstd: Whether backward recomputes reciprocal RMS values.

    Returns:
        Reconstructed RMSNorm config, or ``None`` for plain reduction.
    """
    if not enabled:
        return None
    return RMSNormPostprocess(
        eps=eps,
        norm_output_dtype=norm_output_dtype,
        output_dtype=output_dtype,
        require_bitwise=require_bitwise,
        weight=weight_D,
        gain_center=gain_center,
        use_kahan=use_kahan,
        recompute_rstd=recompute_rstd,
    )


def supports_fused_post_expert_rmsnorm(
    *,
    input_dtype: torch.dtype,
    score_dtype: torch.dtype,
    norm_output_dtype: torch.dtype,
    output_dtype: torch.dtype,
    topk: int,
    dim: int,
) -> bool:
    """Return whether the fused forward supports the base tensor contract.

    This predicate covers forward dtypes, top-k, hidden dimension, and the
    DistMoE output-dtype requirement. It intentionally does not validate a
    particular :class:`RMSNormPostprocess`: training mode, optional weight,
    and Kahan summation are validated by :func:`dist_moe.routed_experts` when
    the complete policy is available. It returns
    ``False`` rather than raising when the CuTe DSL kernel is unavailable.

    Args:
        input_dtype: Expert-output dtype entering RMSNorm.
        score_dtype: Router-score dtype.
        norm_output_dtype: Dtype used for normalized route outputs.
        output_dtype: Dtype of the reduced token output.
        topk: Number of selected experts per token.
        dim: Expert-output hidden dimension.

    Returns:
        Whether the fused forward kernel supports the supplied base contract.
    """
    if output_dtype != input_dtype:
        # dist_moe's combine output must preserve the op input dtype
        # (_validate_rmsnorm_postprocess re-enforces this at resolve time).
        return False
    try:
        # Import lazily: Cutlass imports break CPU-only targets (no libcuda).
        from .kernels.norm.fused_rmsnorm_combine import (
            supports_accelerated_fwd_contract,
        )
    except ImportError:
        return False
    return supports_accelerated_fwd_contract(
        input_dtype=input_dtype,
        weight_dtype=score_dtype,
        norm_output_dtype=norm_output_dtype,
        output_dtype=output_dtype,
        topk=topk,
        dim=dim,
    )


@dataclass(frozen=True)
class _ResolvedExpertsPostprocess:
    """The combine-output stage: postprocess + top-k scale-and-sum, both passes.

    ``_resolve_experts_output_postprocess_fn`` maps every accepted public shape
    of ``experts_output_postprocess`` (None, plain callback,
    `RMSNormPostprocess`) onto one of three variants of this structure, each a
    straight-line implementation. Backends make exactly two calls —
    ``forward`` and ``backward`` — and never branch on whether the norm is
    fused with the reduction.

    ``forward(h3, scores, *, output_dtype, save, save_requires_copy,
    save_buffer, save_offset, save_condition)`` returns
    ``(output, h3_saved, context, postprocess_output_dtype)``. ``save`` asks
    for backward state: either materialized (``h3_saved``; a copy when
    ``save_requires_copy`` says ``h3`` aliases reused storage) or written into
    ``save_buffer`` at ``save_offset`` gated on ``save_condition``.
    ``context`` is the fused variant's rstd.

    ``backward(grad_output, h3_saved, scores, *, postprocess_output_dtype,
    context, publish_view, x_buffer,
    x_buffer_offset, x_buffer_condition)`` returns
    ``(grad_h3, grad_scores, published)``. ``publish_view`` is a
    ``(shape, dtype) -> Tensor`` factory for the buffer the caller wants
    ``grad_h3`` published into; variants that cannot write their gradient
    directly ignore it and return ``published=False``. ``x_buffer*`` let the
    postprocess-free variant read the reduction input straight from the
    activation buffer.
    """

    forward: Callable[
        ...,
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.dtype],
    ]
    backward: Callable[..., tuple[torch.Tensor, torch.Tensor, bool]]
    includes_scale_and_sum: bool
    # True when forward fuses the saved-h3 copy into the reduction launch and
    # backward reads the reduction input straight from an aliased buffer (the
    # postprocess-free variant). Callers use it to decide buffer staging.
    fuses_saved_copy_into_reduction: bool = False


def _validate_experts_output_postprocess_type(
    experts_output_postprocess: object,
) -> None:
    """Validate the accepted public post-expert processing forms.

    Args:
        experts_output_postprocess: ``None``, a direct route-wise callback,
            or an ``RMSNormPostprocess`` configuration.

    Raises:
        ValueError: If the value is not one of the supported forms.
    """
    if (
        experts_output_postprocess is None
        or callable(experts_output_postprocess)
        or isinstance(experts_output_postprocess, RMSNormPostprocess)
    ):
        return
    raise ValueError(
        "experts_output_postprocess must be callable, RMSNormPostprocess, "
        "or None; "
        f"got {experts_output_postprocess!r}"
    )


def _fused_post_expert_rmsnorm(
    h3_MD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    config: RMSNormPostprocess,
    *,
    num_tokens: int,
    topk: int,
) -> torch.Tensor:
    """Run fused RMSNorm, score weighting, and top-k reduction.

    Args:
        h3_MD: Route-wise W2 output with shape ``[T * K, D]``.
        topk_expert_ids_TK: Routed expert IDs with shape ``[T, K]``.
        topk_scores_TK: Router scores with shape ``[T, K]``.
        config: Validated fused RMSNorm policy.
        num_tokens: Local token count ``T``.
        topk: Routes per token ``K``.

    Returns:
        Reduced output with shape ``[T, D]``.
    """
    # Import lazily so CPU-only processes do not require libcuda.
    from .kernels.norm.fused_rmsnorm_combine import (
        _fused_rmsnorm_combine_fwd,
    )

    output_TD, _ = _fused_rmsnorm_combine_fwd(
        h3_MD.reshape(num_tokens, topk, -1),
        topk_scores_TK,
        expert_ids=topk_expert_ids_TK,
        eps=config.eps,
        norm_output_dtype=config.norm_output_dtype,
        output_dtype=config.output_dtype,
        require_bitwise=config.require_bitwise,
        use_kahan=config.use_kahan,
        weight=config.weight,
        gain_center=config.gain_center,
    )
    return output_TD


def _fused_post_expert_rmsnorm_with_input_copy(
    h3_MD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    config: RMSNormPostprocess,
    *,
    num_tokens: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fused RMSNorm while materializing backward state.

    Args:
        h3_MD: Route-wise W2 output with shape ``[T * K, D]``.
        topk_expert_ids_TK: Routed expert IDs with shape ``[T, K]``.
        topk_scores_TK: Router scores with shape ``[T, K]``.
        config: Validated fused RMSNorm policy.
        num_tokens: Local token count ``T``.
        topk: Routes per token ``K``.

    Returns:
        Reduced output, copied route output, and reciprocal RMS values.
    """
    from .kernels.norm.fused_rmsnorm_combine import (
        _fused_rmsnorm_combine_fwd,
    )

    h3_TKD = h3_MD.reshape(num_tokens, topk, -1)
    h3_copy_TKD = torch.empty_like(h3_TKD)
    output_TD, rstd_TK = _fused_rmsnorm_combine_fwd(
        h3_TKD,
        topk_scores_TK,
        expert_ids=topk_expert_ids_TK,
        eps=config.eps,
        use_kahan=config.use_kahan,
        norm_output_dtype=config.norm_output_dtype,
        output_dtype=config.output_dtype,
        require_bitwise=config.require_bitwise,
        x_copy=h3_copy_TKD,
    )
    return output_TD, h3_copy_TKD.reshape_as(h3_MD), rstd_TK


def _fused_post_expert_rmsnorm_with_context(
    h3_MD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    config: RMSNormPostprocess,
    *,
    num_tokens: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused RMSNorm and return its compact backward context.

    Args:
        h3_MD: Stable route-wise W2 output with shape ``[T * K, D]``.
        topk_expert_ids_TK: Routed expert IDs with shape ``[T, K]``.
        topk_scores_TK: Router scores with shape ``[T, K]``.
        config: Validated fused RMSNorm policy.
        num_tokens: Local token count ``T``.
        topk: Routes per token ``K``.

    Returns:
        Reduced output and reciprocal RMS values.
    """
    from .kernels.norm.fused_rmsnorm_combine import (
        _fused_rmsnorm_combine_fwd,
    )

    return _fused_rmsnorm_combine_fwd(
        h3_MD.reshape(num_tokens, topk, -1),
        topk_scores_TK,
        expert_ids=topk_expert_ids_TK,
        eps=config.eps,
        use_kahan=config.use_kahan,
        norm_output_dtype=config.norm_output_dtype,
        output_dtype=config.output_dtype,
        require_bitwise=config.require_bitwise,
    )


def _fused_post_expert_rmsnorm_backward_to(
    grad_output_TD: torch.Tensor,
    h3_MD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    grad_h3_output_MD: torch.Tensor,
    rstd_TK: torch.Tensor | None,
    config: RMSNormPostprocess,
    *,
    num_tokens: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused RMSNorm and score-reduction backward into a destination.

    Args:
        grad_output_TD: Gradient of the reduced output with shape ``[T, D]``.
        h3_MD: Saved or recomputed route output with shape ``[T * K, D]``.
        topk_expert_ids_TK: Routed expert IDs with shape ``[T, K]``.
        topk_scores_TK: Router scores with shape ``[T, K]``.
        grad_h3_output_MD: Destination for the route gradient.
        rstd_TK: Reciprocal RMS context from forward, or ``None`` when recomputed.
        config: Validated fused RMSNorm policy.
        num_tokens: Local token count ``T``.
        topk: Routes per token ``K``.

    Returns:
        Route-output and router-score gradients.
    """
    from .kernels.norm.fused_rmsnorm_combine import (
        _fused_rmsnorm_combine_bwd,
        _prepare_bwd_grad_y,
    )

    h3_TKD = h3_MD.reshape(num_tokens, topk, -1)
    grad_y_TD = _prepare_bwd_grad_y(
        grad_output_TD,
        config.norm_output_dtype,
        require_bitwise=config.require_bitwise,
        x=h3_TKD,
    )
    grad_h3_TKD, grad_topk_scores_TK = _fused_rmsnorm_combine_bwd(
        grad_y_TD,
        h3_TKD,
        topk_scores_TK,
        rstd_TK,
        eps=config.eps,
        require_bitwise=config.require_bitwise,
        use_kahan=config.use_kahan,
        grad_x=grad_h3_output_MD.reshape_as(h3_TKD),
        expert_ids=topk_expert_ids_TK,
    )
    if not topk_scores_TK.requires_grad:
        grad_topk_scores_TK.zero_()
    return grad_h3_TKD.reshape_as(h3_MD), grad_topk_scores_TK


def _scale_and_sum(
    x_TKD: torch.Tensor,
    scale_TK: torch.Tensor,
    expert_ids_TK: torch.Tensor,
    *,
    return_copy: bool,
    output_dtype: torch.dtype,
    x_copy_buffer: torch.Tensor | None = None,
    x_copy_offset: torch.Tensor | None = None,
    x_copy_condition: torch.Tensor | None = None,
    select_routes: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply router scores and reduce routes, optionally copying the input.

    Args:
        x_TKD: Route output with shape ``[T, K, D]``.
        scale_TK: Router scores with shape ``[T, K]``.
        expert_ids_TK: Global expert IDs with shape ``[T, K]``. Negative routes
            are inactive.
        return_copy: Whether to return a standalone copy for backward.
        output_dtype: Dtype of the reduced output.
        x_copy_buffer: Optional activation buffer containing the saved copy.
        x_copy_offset: Byte offset of the saved copy in ``x_copy_buffer``.
        x_copy_condition: Device predicate controlling the buffer copy.
        select_routes: Inference only: issue the ID, score and row loads
            together (``scale_and_sum``).

    Returns:
        Reduced output and optional copied route output.
    """
    from .kernels.triton.broadcast_n_reduction import scale_and_sum

    if x_TKD.dtype is not torch.float64:
        return scale_and_sum(
            x=x_TKD,
            scale=scale_TK,
            expert_ids=expert_ids_TK,
            return_copy=return_copy,
            output_dtype=output_dtype,
            x_copy_buffer=x_copy_buffer,
            x_copy_offset=x_copy_offset,
            x_copy_condition=x_copy_condition,
            select_routes=select_routes,
        )

    assert x_copy_buffer is None
    assert x_copy_offset is None
    assert x_copy_condition is None
    valid_TK = expert_ids_TK >= 0
    masked_x_TKD = torch.where(valid_TK.unsqueeze(-1), x_TKD, 0)
    output_TD = (masked_x_TKD * scale_TK.unsqueeze(-1)).sum(dim=1).to(output_dtype)
    return output_TD, masked_x_TKD.clone() if return_copy else None


def _reduction_backward(
    grad_output_TD: torch.Tensor,
    scores_TK: torch.Tensor,
    expert_ids_TK: torch.Tensor,
    x_TKD: torch.Tensor,
    *,
    reduction_input_dtype: torch.dtype,
    out_TKD: torch.Tensor | None = None,
    x_buffer: torch.Tensor | None = None,
    x_buffer_offset: torch.Tensor | None = None,
    x_buffer_condition: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate score weighting and top-k reduction.

    Args:
        grad_output_TD: Gradient of the reduced output with shape ``[T, D]``.
        scores_TK: Router scores with shape ``[T, K]``.
        expert_ids_TK: Global expert IDs with shape ``[T, K]``. Negative routes
            receive zero gradients.
        x_TKD: Saved route output with shape ``[T, K, D]``.
        reduction_input_dtype: Dtype used by forward score reduction.
        out_TKD: Optional destination for the route gradient.
        x_buffer: Optional activation buffer containing ``x_TKD``.
        x_buffer_offset: Byte offset of ``x_TKD`` in the activation buffer.
        x_buffer_condition: Device predicate selecting the activation-buffer value.

    Returns:
        Route-output and router-score gradients.
    """
    from .kernels.triton.broadcast_n_reduction import broadcast_and_scale

    grad_output_TD = grad_output_TD.to(reduction_input_dtype).contiguous()
    if reduction_input_dtype is torch.float64:
        assert x_buffer is None
        assert x_buffer_offset is None
        assert x_buffer_condition is None
        valid_TK = expert_ids_TK >= 0
        grad_x_TKD = torch.where(
            valid_TK.unsqueeze(-1),
            grad_output_TD.unsqueeze(1) * scores_TK.unsqueeze(-1),
            0,
        )
        grad_scores_TK = (
            torch.where(valid_TK.unsqueeze(-1), grad_output_TD.unsqueeze(1) * x_TKD, 0)
            .sum(dim=-1)
            .to(scores_TK.dtype)
        )
        if out_TKD is not None:
            out_TKD.copy_(grad_x_TKD)
            grad_x_TKD = out_TKD
        return grad_x_TKD, grad_scores_TK

    grad_x_TKD, grad_scores_TK = broadcast_and_scale(
        dy=grad_output_TD,
        scale=scores_TK,
        x=x_TKD,
        expert_ids=expert_ids_TK,
        dx=out_TKD,
        x_buffer=x_buffer,
        x_buffer_offset=x_buffer_offset,
        x_buffer_condition=x_buffer_condition,
    )
    return grad_x_TKD, grad_scores_TK


def _grad_publish_target(
    publish_view: Callable[[tuple[int, ...], torch.dtype], torch.Tensor] | None,
    h3_saved_MD: torch.Tensor,
    *,
    preserve_h3_saved: bool,
) -> tuple[torch.Tensor, bool]:
    """Return the route-gradient destination and publication state.

    Zerocopy recompute paths deliberately leave ``h3_saved`` in the very
    buffer the caller publishes grad_h3 into (async: the symm-mem combine
    buffer); the fused backward supports that in-place store. What it cannot
    support is the caller consuming ``h3_saved`` afterwards: when the grad
    observer needs it (``preserve_h3_saved``), an overlapping view is
    refused and the gradient is materialized for the caller to stage
    instead.

    Args:
        publish_view: Optional destination-view factory.
        h3_saved_MD: Saved or recomputed route output.
        preserve_h3_saved: Whether a later observer still needs ``h3_saved``.

    Returns:
        Gradient destination and whether it directly publishes the gradient.
    """
    if publish_view is not None:
        out_MD = publish_view(tuple(h3_saved_MD.shape), h3_saved_MD.dtype)
        if not (preserve_h3_saved and _views_overlap(out_MD, h3_saved_MD)):
            return out_MD, True
    return torch.empty_like(h3_saved_MD), False


def _validate_rmsnorm_postprocess(
    config: RMSNormPostprocess,
    *,
    expert_output_dtype: torch.dtype,
    expert_output_dim: int,
    topk_scores_TK: torch.Tensor,
    inference_mode: bool,
) -> None:
    """Validate the complete fused RMSNorm execution contract.

    Args:
        config: Public fused RMSNorm policy.
        expert_output_dtype: Dtype of the W2 combine output.
        expert_output_dim: Hidden dimension ``D``.
        topk_scores_TK: Router scores with shape ``[T, K]``.
        inference_mode: Whether backward is disabled.

    Raises:
        ValueError: If numerics, shape, dtype, or mode are unsupported.
    """
    if not math.isfinite(config.eps) or config.eps <= 0:
        raise ValueError(
            f"RMSNormPostprocess eps must be finite and positive; got eps={config.eps}"
        )
    if config.output_dtype != expert_output_dtype:
        raise ValueError(
            "RMSNormPostprocess output_dtype must match the DistMoE input dtype; "
            f"got output_dtype={config.output_dtype}, input_dtype={expert_output_dtype}"
        )

    # Import lazily so CPU-only processes do not require libcuda.
    from .kernels.norm._fused_rmsnorm_combine_bwd_kernel import (
        FusedRMSNormCombineBwd,
    )
    from .kernels.norm.fused_rmsnorm_combine import (
        supports_accelerated_fwd_contract,
    )

    _, topk = topk_scores_TK.shape
    if not supports_accelerated_fwd_contract(
        input_dtype=expert_output_dtype,
        weight_dtype=topk_scores_TK.dtype,
        norm_output_dtype=config.norm_output_dtype,
        output_dtype=config.output_dtype,
        topk=topk,
        dim=expert_output_dim,
    ):
        raise ValueError(
            "RMSNormPostprocess requires bf16 expert outputs, fp32 scores, "
            "topk=2/8, a supported expert dimension, fp32 norm output, and "
            "bf16/fp32 reduction output; got "
            f"input_dtype={expert_output_dtype}, score_dtype={topk_scores_TK.dtype}, "
            f"topk={topk}, dim={expert_output_dim}, "
            f"norm_output_dtype={config.norm_output_dtype}, "
            f"output_dtype={config.output_dtype}, "
            f"require_bitwise={config.require_bitwise}"
        )
    if (
        not inference_mode
        and not config.require_bitwise
        and expert_output_dim > FusedRMSNormCombineBwd._MAX_DIM
    ):
        raise ValueError(
            "non-bitwise RMSNormPostprocess dimensions above "
            f"{FusedRMSNormCombineBwd._MAX_DIM} are inference-only"
        )
    _validate_rmsnorm_weight(
        config,
        expert_output_dtype=expert_output_dtype,
        expert_output_dim=expert_output_dim,
        inference_mode=inference_mode,
    )


def _validate_rmsnorm_weight(
    config: RMSNormPostprocess,
    *,
    expert_output_dtype: torch.dtype,
    expert_output_dim: int,
    inference_mode: bool,
) -> None:
    """Validate the optional inference-only RMSNorm input scale.

    Args:
        config: Public fused RMSNorm policy.
        expert_output_dtype: Dtype of the W2 combine output.
        expert_output_dim: Hidden dimension ``D``.
        inference_mode: Whether backward is disabled.

    Raises:
        ValueError: If the input scale or its mode is unsupported.
    """
    if config.weight is None:
        if config.gain_center != 0.0:
            raise ValueError(
                "RMSNormPostprocess gain_center requires a weight; got "
                f"gain_center={config.gain_center} with weight=None"
            )
        return
    if config.use_kahan:
        # The fused kernel rejects the combination with NotImplementedError;
        # surface it here so it fails the advertised entry-time contract
        # check instead of the forward launch.
        raise ValueError(
            "RMSNormPostprocess use_kahan is not supported together with "
            "weight (input-scale gamma)"
        )
    if (
        config.weight.shape != (expert_output_dim,)
        or config.weight.dtype != expert_output_dtype
        or not config.weight.is_cuda
        or not config.weight.is_contiguous()
    ):
        raise ValueError(
            "RMSNormPostprocess weight must be a contiguous CUDA [DIM] tensor "
            "matching the DistMoE input dtype; got weight shape "
            f"{tuple(config.weight.shape)} dtype {config.weight.dtype} for "
            f"dim={expert_output_dim}, input_dtype={expert_output_dtype}"
        )
    if not inference_mode:
        raise ValueError(
            "RMSNormPostprocess weight (input-scale gamma) is "
            "inference/forward-only; training with a weighted post-expert "
            "norm must use a callback postprocess"
        )


def _resolve_experts_output_postprocess_fn(
    experts_output_postprocess: _ExpertsOutputPostprocess | _ResolvedExpertsPostprocess,
    *,
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    inference_mode: bool = False,
    zero_out_padded_callback_inputs: bool = False,
) -> _ResolvedExpertsPostprocess:
    """Resolve the public postprocess value to straight-line execution hooks.

    Args:
        experts_output_postprocess: Public callback, RMSNorm config, ``None``,
            or an already resolved private runtime value.
        x_TD: Dist-MoE input, which defines output dtype and hidden width.
        topk_expert_ids_TK: Routed expert IDs with shape ``[T, K]``.
        topk_scores_TK: Router scores with shape ``[T, K]``.
        inference_mode: Whether backward is disabled.
        zero_out_padded_callback_inputs: Whether Python callbacks receive zeroed
            invalid route rows.

    Returns:
        Forward/backward hooks for exactly one supported execution variant.
    """
    # Idempotent so callers may resolve once before entering the backend.
    if isinstance(experts_output_postprocess, _ResolvedExpertsPostprocess):
        return experts_output_postprocess
    _validate_experts_output_postprocess_type(experts_output_postprocess)
    num_tokens, topk = topk_scores_TK.shape
    if experts_output_postprocess is None:
        return _resolve_reduction_only(
            num_tokens, topk, topk_expert_ids_TK, inference_mode=inference_mode
        )
    if callable(experts_output_postprocess):
        return _resolve_callback(
            experts_output_postprocess,
            num_tokens,
            topk,
            topk_expert_ids_TK,
            zero_out_padded_inputs=zero_out_padded_callback_inputs,
        )
    return _resolve_rmsnorm_config(
        experts_output_postprocess,
        x_TD=x_TD,
        topk_expert_ids_TK=topk_expert_ids_TK,
        topk_scores_TK=topk_scores_TK,
        inference_mode=inference_mode,
        zero_out_padded_callback_inputs=zero_out_padded_callback_inputs,
    )


def _resolve_reduction_only(
    num_tokens: int,
    topk: int,
    expert_ids_TK: torch.Tensor,
    *,
    inference_mode: bool = False,
) -> _ResolvedExpertsPostprocess:
    """Resolve the plain top-k scale-and-sum stage.

    Args:
        num_tokens: Local token count ``T``.
        topk: Routes per token ``K``.
        expert_ids_TK: Routed expert IDs defining active routes.
        inference_mode: Whether backward is disabled; selects the inference
            reduction loads.

    Returns:
        Forward and backward hooks for reduction without postprocessing.
    """

    def forward(
        h3_MD: torch.Tensor,
        scores_TK: torch.Tensor,
        *,
        output_dtype: torch.dtype,
        save: bool,
        save_requires_copy: bool = True,
        save_buffer: torch.Tensor | None = None,
        save_offset: torch.Tensor | None = None,
        save_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.dtype]:
        """Reduce route outputs and retain only the state backward needs.

        Args:
            h3_MD: Route-wise W2 output with shape ``[T * K, D]``.
            scores_TK: Router scores with shape ``[T, K]``.
            output_dtype: Requested reduced-output dtype.
            save: Whether backward state is required.
            save_requires_copy: Whether ``h3_MD`` aliases reusable storage.
            save_buffer: Optional activation buffer for the saved route output.
            save_offset: Byte offset into ``save_buffer``.
            save_condition: Device predicate controlling the buffer save.

        Returns:
            Output, optional saved route output, no auxiliary context, and the
            reduction-input dtype.
        """
        h3_TKD = h3_MD.view(num_tokens, topk, -1)
        if save and save_buffer is not None:
            output_TD, _ = _scale_and_sum(
                x_TKD=h3_TKD,
                scale_TK=scores_TK,
                expert_ids_TK=expert_ids_TK,
                return_copy=False,
                output_dtype=output_dtype,
                x_copy_buffer=save_buffer,
                x_copy_offset=save_offset,
                x_copy_condition=save_condition,
            )
            return output_TD, None, None, h3_TKD.dtype
        if save and not save_requires_copy:
            # The caller vouches h3_MD is stable storage, so saving it directly
            # skips the copy fused into the reduction launch.
            output_TD, _ = _scale_and_sum(
                x_TKD=h3_TKD,
                scale_TK=scores_TK,
                expert_ids_TK=expert_ids_TK,
                return_copy=False,
                output_dtype=output_dtype,
            )
            return output_TD, h3_MD, None, h3_TKD.dtype
        output_TD, h3_saved_TKD = _scale_and_sum(
            x_TKD=h3_TKD,
            scale_TK=scores_TK,
            expert_ids_TK=expert_ids_TK,
            return_copy=save,
            output_dtype=output_dtype,
            select_routes=inference_mode,
        )
        return output_TD, h3_saved_TKD, None, h3_TKD.dtype

    def backward(
        grad_output_TD: torch.Tensor,
        h3_saved_MD: torch.Tensor,
        scores_TK: torch.Tensor,
        *,
        postprocess_output_dtype: torch.dtype,
        context: torch.Tensor | None = None,
        publish_view: Callable[[tuple[int, ...], torch.dtype], torch.Tensor]
        | None = None,
        x_buffer: torch.Tensor | None = None,
        x_buffer_offset: torch.Tensor | None = None,
        x_buffer_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Differentiate plain score weighting and top-k reduction.

        Args:
            grad_output_TD: Gradient of the reduced output with shape ``[T, D]``.
            h3_saved_MD: Saved route output or activation-buffer placeholder.
            scores_TK: Router scores with shape ``[T, K]``.
            postprocess_output_dtype: Forward reduction-input dtype.
            context: Unused auxiliary context.
            publish_view: Optional symmetric-memory gradient destination.
            x_buffer: Optional activation buffer containing the route output.
            x_buffer_offset: Byte offset into ``x_buffer``.
            x_buffer_condition: Device predicate selecting the activation-buffer value.

        Returns:
            Route gradient, router-score gradient, and whether the route
            gradient was written directly to ``publish_view``.
        """
        h3_saved_TKD = h3_saved_MD.view(num_tokens, topk, -1)
        grad_h3_output_TKD = (
            publish_view(tuple(h3_saved_TKD.shape), postprocess_output_dtype)
            if publish_view is not None
            else None
        )
        grad_h3_TKD, grad_scores_TK = _reduction_backward(
            grad_output_TD,
            scores_TK,
            expert_ids_TK,
            h3_saved_TKD,
            reduction_input_dtype=postprocess_output_dtype,
            out_TKD=grad_h3_output_TKD,
            x_buffer=x_buffer,
            x_buffer_offset=x_buffer_offset,
            x_buffer_condition=x_buffer_condition,
        )
        return (
            grad_h3_TKD.view_as(h3_saved_MD),
            grad_scores_TK,
            grad_h3_output_TKD is not None,
        )

    return _ResolvedExpertsPostprocess(
        forward=forward,
        backward=backward,
        includes_scale_and_sum=False,
        fuses_saved_copy_into_reduction=True,
    )


def _validate_callback_output(
    output_MD: torch.Tensor,
    input_MD: torch.Tensor,
    *,
    expected_dtype: torch.dtype | None = None,
) -> None:
    """Validate one eager route-wise postprocess result.

    Args:
        output_MD: Callback output to validate.
        input_MD: Route-wise callback input with shape ``[T * K, D]``.
        expected_dtype: Optional dtype recorded by the matching forward call.

    Raises:
        TypeError: If the callback did not return a supported tensor dtype.
        ValueError: If shape, device, layout, or replay dtype changed.
    """
    if not isinstance(output_MD, torch.Tensor):
        raise TypeError("expert postprocess callback must return a tensor")
    if output_MD.shape != input_MD.shape:
        raise ValueError(
            "expert postprocess callback must preserve route shape; "
            f"got {tuple(output_MD.shape)} for input {tuple(input_MD.shape)}"
        )
    if output_MD.device != input_MD.device:
        raise ValueError("expert postprocess callback must preserve the input device")
    if not output_MD.is_contiguous():
        raise ValueError("expert postprocess callback must return contiguous output")
    if output_MD.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "expert postprocess callback must return float16, bfloat16, or "
            f"float32 output; got {output_MD.dtype}"
        )
    if expected_dtype is not None and output_MD.dtype != expected_dtype:
        raise ValueError(
            "expert postprocess callback changed dtype between forward and "
            f"backward replay: {expected_dtype} -> {output_MD.dtype}"
        )


def _resolve_callback(
    fn: _ExpertsOutputPostprocessFn,
    num_tokens: int,
    topk: int,
    expert_ids_TK: torch.Tensor,
    *,
    zero_out_padded_inputs: bool,
) -> _ResolvedExpertsPostprocess:
    """Transform callback (e.g. an unfused norm), then the plain reduction.

    The backward replays the callback under ``enable_grad`` on the saved
    pre-postprocess expert output — one replay serves both the reduction input
    and the gradient chain, so no postprocessed copy is ever saved.

    The callback must be deterministic, side-effect-free except for ordinary
    parameter-gradient accumulation, and return a contiguous tensor with the
    same shape and device as its input. Closure parameters receive gradients
    through the nested backward replay; they are not explicit inputs of the
    outer DistMoE autograd function.

    Args:
        fn: Route-wise eager callback.
        num_tokens: Local token count ``T``.
        topk: Routes per token ``K``.
        expert_ids_TK: Routed expert IDs defining active routes.
        zero_out_padded_inputs: Whether invalid route rows are zeroed before
            invoking the callback.

    Returns:
        Straight-line eager forward and backward callback handlers.
    """
    from ._triton_ops import conditional_copy_activations, zero_out_padded_rows_
    from .kernels.triton.broadcast_n_reduction import scale_and_sum

    def forward(
        h3_MD: torch.Tensor,
        scores_TK: torch.Tensor,
        *,
        output_dtype: torch.dtype,
        save: bool,
        save_requires_copy: bool = True,
        save_buffer: torch.Tensor | None = None,
        save_offset: torch.Tensor | None = None,
        save_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.dtype]:
        """Apply a direct callback, reduce routes, and retain its input.

        Args:
            h3_MD: Route-wise W2 output with shape ``[T * K, D]``.
            scores_TK: Router scores with shape ``[T, K]``.
            output_dtype: Requested reduced-output dtype.
            save: Whether backward state is required.
            save_requires_copy: Whether ``h3_MD`` aliases reusable storage.
            save_buffer: Optional activation buffer for the callback input.
            save_offset: Byte offset into ``save_buffer``.
            save_condition: Device predicate controlling the buffer save.

        Returns:
            Output, optional saved callback input, no auxiliary context, and
            the callback-output dtype used by reduction.
        """
        if zero_out_padded_inputs:
            zero_out_padded_rows_(h3_MD, expert_ids_TK)
        h3_saved_MD = None
        if save:
            if save_buffer is not None:
                assert save_condition is not None
                conditional_copy_activations(
                    condition=save_condition,
                    lhs=None,
                    lhs_offset=None,
                    rhs=h3_MD,
                    rhs_offset=save_offset,
                    activation_buffer=save_buffer,
                    copy_to_buffer=True,
                )
            else:
                h3_saved_MD = h3_MD.clone() if save_requires_copy else h3_MD
        with torch.no_grad():
            h3_postprocessed_MD = fn(h3_MD)
        _validate_callback_output(h3_postprocessed_MD, h3_MD)
        output_TD, _ = scale_and_sum(
            x=h3_postprocessed_MD.view(num_tokens, topk, -1),
            scale=scores_TK,
            expert_ids=expert_ids_TK,
            return_copy=False,
            output_dtype=output_dtype,
        )
        return output_TD, h3_saved_MD, None, h3_postprocessed_MD.dtype

    def backward(
        grad_output_TD: torch.Tensor,
        h3_saved_MD: torch.Tensor,
        scores_TK: torch.Tensor,
        *,
        postprocess_output_dtype: torch.dtype,
        context: torch.Tensor | None = None,
        publish_view: Callable[[tuple[int, ...], torch.dtype], torch.Tensor]
        | None = None,
        x_buffer: torch.Tensor | None = None,
        x_buffer_offset: torch.Tensor | None = None,
        x_buffer_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Replay a direct callback and differentiate it plus reduction.

        Args:
            grad_output_TD: Gradient of the reduced output with shape ``[T, D]``.
            h3_saved_MD: Saved pre-callback route output with shape
                ``[T * K, D]``.
            scores_TK: Router scores with shape ``[T, K]``.
            postprocess_output_dtype: Forward callback-output dtype.
            context: Unused auxiliary context.
            publish_view: Unused direct-publication destination.
            x_buffer: Unused activation buffer.
            x_buffer_offset: Unused activation-buffer offset.
            x_buffer_condition: Unused activation-buffer predicate.

        Returns:
            Callback-input gradient, router-score gradient, and ``False``
            because the gradient has not been published.
        """
        if zero_out_padded_inputs:
            zero_out_padded_rows_(h3_saved_MD, expert_ids_TK)
        h3_pre_MD = h3_saved_MD.detach().requires_grad_(True)
        with torch.enable_grad():
            h3_postprocessed_MD = fn(h3_pre_MD)
        _validate_callback_output(
            h3_postprocessed_MD,
            h3_pre_MD,
            expected_dtype=postprocess_output_dtype,
        )
        h3_postprocessed_TKD = h3_postprocessed_MD.detach().view(num_tokens, topk, -1)
        grad_h3_post_TKD, grad_scores_TK = _reduction_backward(
            grad_output_TD,
            scores_TK,
            expert_ids_TK,
            h3_postprocessed_TKD,
            reduction_input_dtype=postprocess_output_dtype,
        )
        with torch.enable_grad():
            h3_postprocessed_MD.backward(grad_h3_post_TKD.view_as(h3_postprocessed_MD))
        grad_h3_MD = h3_pre_MD.grad
        assert grad_h3_MD is not None
        return grad_h3_MD, grad_scores_TK, False

    return _ResolvedExpertsPostprocess(
        forward=forward,
        backward=backward,
        includes_scale_and_sum=False,
    )


def _validate_rmsnorm_stage_inputs(
    h3_MD: torch.Tensor,
    scores_TK: torch.Tensor,
    *,
    expected_h3_shapes: tuple[tuple[int, ...], ...],
    num_tokens: int,
    topk: int,
    input_dtype: torch.dtype,
) -> None:
    """Validate tensors presented to a resolved fused RMSNorm stage.

    Args:
        h3_MD: Route-wise W2 output.
        scores_TK: Router scores.
        expected_h3_shapes: Accepted flattened and unflattened route shapes.
        num_tokens: Local token count ``T``.
        topk: Routes per token ``K``.
        input_dtype: Required expert-output dtype.

    Raises:
        ValueError: If shape, dtype, device, or layout is unsupported.
    """
    if tuple(h3_MD.shape) not in expected_h3_shapes or tuple(scores_TK.shape) != (
        num_tokens,
        topk,
    ):
        raise ValueError(
            "RMSNormPostprocess shape mismatch: expected expert outputs "
            f"with shape in {expected_h3_shapes} and scores "
            f"{(num_tokens, topk)}, got "
            f"{tuple(h3_MD.shape)} and {tuple(scores_TK.shape)}"
        )
    if h3_MD.dtype != input_dtype:
        raise ValueError(
            "RMSNormPostprocess requires expert output to preserve the "
            f"DistMoE input dtype; got {h3_MD.dtype}, expected {input_dtype}"
        )
    if not (
        h3_MD.is_cuda
        and scores_TK.is_cuda
        and h3_MD.device == scores_TK.device
        and h3_MD.is_contiguous()
        and scores_TK.is_contiguous()
    ):
        raise ValueError(
            "RMSNormPostprocess requires contiguous CUDA tensors on one device"
        )


def _resolve_rmsnorm_config(
    config: RMSNormPostprocess,
    *,
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    inference_mode: bool,
    zero_out_padded_callback_inputs: bool,
) -> _ResolvedExpertsPostprocess:
    """Resolve fused RMSNorm plus scale-and-sum execution hooks.

    Args:
        config: Public fused RMSNorm policy.
        x_TD: Dist-MoE input that defines the output dtype and hidden width.
        topk_expert_ids_TK: Routed expert IDs with shape ``[T, K]``.
        topk_scores_TK: Router scores with shape ``[T, K]``.
        inference_mode: Whether backward is disabled.
        zero_out_padded_callback_inputs: Whether observer callbacks receive
            zeroed invalid route rows.

    Returns:
        Validated forward and backward hooks for fused RMSNorm reduction.
    """
    from ._triton_ops import conditional_copy_activations, zero_out_padded_rows_

    num_tokens, topk = topk_scores_TK.shape
    _validate_rmsnorm_postprocess(
        config,
        expert_output_dtype=x_TD.dtype,
        expert_output_dim=x_TD.shape[-1],
        topk_scores_TK=topk_scores_TK,
        inference_mode=inference_mode,
    )
    expected_h3_shapes = (
        (num_tokens, topk, x_TD.shape[-1]),
        (num_tokens * topk, x_TD.shape[-1]),
    )
    validate_inputs = partial(
        _validate_rmsnorm_stage_inputs,
        expected_h3_shapes=expected_h3_shapes,
        num_tokens=num_tokens,
        topk=topk,
        input_dtype=x_TD.dtype,
    )

    def observe_forward(h3_MD: torch.Tensor) -> None:
        """Invoke the optional forward observer on the route output.

        Args:
            h3_MD: Route-wise W2 output before normalization.
        """
        if config.observe_expert_output_fn is not None:
            if zero_out_padded_callback_inputs:
                zero_out_padded_rows_(h3_MD, topk_expert_ids_TK)
            config.observe_expert_output_fn(h3_MD)

    def forward(
        h3_MD: torch.Tensor,
        scores_TK: torch.Tensor,
        *,
        output_dtype: torch.dtype,
        save: bool,
        save_requires_copy: bool = True,
        save_buffer: torch.Tensor | None = None,
        save_offset: torch.Tensor | None = None,
        save_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.dtype]:
        """Run fused RMSNorm/reduction and retain its minimal backward state.

        Args:
            h3_MD: Route-wise W2 output.
            scores_TK: Router scores.
            output_dtype: Requested reduced-output dtype.
            save: Whether backward state is required.
            save_requires_copy: Whether ``h3_MD`` aliases reusable storage.
            save_buffer: Optional activation buffer for the saved route output.
            save_offset: Byte offset into ``save_buffer``.
            save_condition: Device predicate controlling the buffer save.

        Returns:
            Reduced output, optional saved route output, reciprocal RMS values,
            and the reduction-input dtype.
        """
        assert output_dtype == config.output_dtype, (
            "RMSNormPostprocess output_dtype must match the reduction dtype; "
            f"got config {config.output_dtype} vs requested {output_dtype}"
        )
        validate_inputs(h3_MD, scores_TK)
        observe_forward(h3_MD)
        if not save:
            output_TD = _fused_post_expert_rmsnorm(
                h3_MD,
                topk_expert_ids_TK,
                scores_TK,
                config,
                num_tokens=num_tokens,
                topk=topk,
            )
            return output_TD, None, None, config.output_dtype
        if save_buffer is not None:
            assert save_condition is not None
            conditional_copy_activations(
                condition=save_condition,
                lhs=None,
                lhs_offset=None,
                rhs=h3_MD,
                rhs_offset=save_offset,
                activation_buffer=save_buffer,
                copy_to_buffer=True,
            )
            output_TD, rstd_TK = _fused_post_expert_rmsnorm_with_context(
                h3_MD,
                topk_expert_ids_TK,
                scores_TK,
                config,
                num_tokens=num_tokens,
                topk=topk,
            )
            return (
                output_TD,
                None,
                None if config.recompute_rstd else rstd_TK,
                config.output_dtype,
            )
        if not save_requires_copy:
            # The caller vouches h3_MD is stable storage (e.g. the composed
            # backend's materialized combine output), so saving it directly
            # avoids a second full-activation copy.
            output_TD, rstd_TK = _fused_post_expert_rmsnorm_with_context(
                h3_MD,
                topk_expert_ids_TK,
                scores_TK,
                config,
                num_tokens=num_tokens,
                topk=topk,
            )
            return (
                output_TD,
                h3_MD,
                None if config.recompute_rstd else rstd_TK,
                config.output_dtype,
            )
        output_TD, h3_saved_MD, rstd_TK = _fused_post_expert_rmsnorm_with_input_copy(
            h3_MD,
            topk_expert_ids_TK,
            scores_TK,
            config,
            num_tokens=num_tokens,
            topk=topk,
        )
        return (
            output_TD,
            h3_saved_MD,
            None if config.recompute_rstd else rstd_TK,
            config.output_dtype,
        )

    def backward(
        grad_output_TD: torch.Tensor,
        h3_saved_MD: torch.Tensor,
        scores_TK: torch.Tensor,
        *,
        postprocess_output_dtype: torch.dtype,
        context: torch.Tensor | None = None,
        publish_view: Callable[[tuple[int, ...], torch.dtype], torch.Tensor]
        | None = None,
        x_buffer: torch.Tensor | None = None,
        x_buffer_offset: torch.Tensor | None = None,
        x_buffer_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Run fused RMSNorm and score-reduction backward.

        Args:
            grad_output_TD: Gradient of the reduced output with shape ``[T, D]``.
            h3_saved_MD: Saved or recomputed pre-normalization route output.
            scores_TK: Router scores with shape ``[T, K]``.
            postprocess_output_dtype: Forward reduction-input dtype.
            context: Reciprocal RMS values produced by forward.
            publish_view: Optional symmetric-memory gradient destination.
            x_buffer: Unused activation buffer.
            x_buffer_offset: Unused activation-buffer offset.
            x_buffer_condition: Unused activation-buffer predicate.

        Returns:
            Route gradient, router-score gradient, and whether the route
            gradient was written directly to ``publish_view``.
        """
        validate_inputs(h3_saved_MD, scores_TK)
        if (
            zero_out_padded_callback_inputs
            and config.observe_expert_output_grad_fn is not None
        ):
            zero_out_padded_rows_(h3_saved_MD, topk_expert_ids_TK)
        grad_h3_output_MD, published = _grad_publish_target(
            publish_view,
            h3_saved_MD,
            preserve_h3_saved=config.observe_expert_output_grad_fn is not None,
        )
        grad_h3_MD, grad_scores_TK = _fused_post_expert_rmsnorm_backward_to(
            grad_output_TD,
            h3_saved_MD,
            topk_expert_ids_TK,
            scores_TK,
            grad_h3_output_MD,
            context,
            config,
            num_tokens=num_tokens,
            topk=topk,
        )
        if config.observe_expert_output_grad_fn is not None:
            config.observe_expert_output_grad_fn(h3_saved_MD, grad_h3_MD)
        return grad_h3_MD, grad_scores_TK, published

    return _ResolvedExpertsPostprocess(
        forward=forward,
        backward=backward,
        includes_scale_and_sum=True,
    )
