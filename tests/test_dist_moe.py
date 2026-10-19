# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the BF16 CuTe DSL distributed MoE implementation."""

import contextlib
import dataclasses
import datetime
import tempfile
import threading
import unittest
from functools import partial, wraps
from typing import Any, Callable, get_args, Literal
from unittest import mock

import dist_moe
import dist_moe._blockscaled as blockscaled_impl
import dist_moe.api as api_impl
import dist_moe.kernels.pdl as pdl_impl
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.testing._internal.distributed.fake_pg  # noqa: F401
from dist_moe._activation_buffer import (
    ActivationBuffer,
    BlockscaledStorageConfig,
    get_backward_plan,
    get_forward_plan,
    ModelConfig,
)
from dist_moe._buffers import (
    _CommunicationBuffers,
    _ROUTING_HEADER_SIZE_BYTES,
    _routing_ids_view,
    _routing_token_count_view,
    is_fake_symmetric_memory,
)
from dist_moe._execution import (
    _resolve_parameter_grad,
    _resolve_parameter_grad_destinations,
    _resolve_wgrad_destination,
    _resolve_wgrad_dtype,
    _weak_parameter_ref,
)
from dist_moe._postprocess import (
    _registered_rmsnorm_args,
    _rmsnorm_from_registered_args,
    _validate_callback_output,
)
from dist_moe._triton_ops import (
    copy_dispatch_to_activation,
    copy_routing_and_dispatch,
)
from dist_moe.formats import BlockScaledFormat as _KernelBlockScaledFormat
from dist_moe.kernels import (
    config as grouped_gemm_config_impl,
    dist_grouped_gemm as dist_grouped_gemm_impl,
    grouped_gemm as grouped_gemm_impl,
    grouped_gemm_kernel as grouped_gemm_kernel_impl,
    mega_blockscaled_grouped_gemm as mega_impl,
)
from torch._higher_order_ops.effects import has_effects
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)


def _is_blackwell() -> bool:
    """Return whether the current process can execute SM100 CuTe kernels.

    Returns:
        ``True`` when CUDA is available with compute capability 10.0 or newer.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Compute relative L2 error in FP32.

    Args:
        actual: Tensor produced by the implementation.
        expected: Reference tensor.

    Returns:
        Relative L2 error as a Python float.
    """
    numerator = torch.linalg.vector_norm(actual.float() - expected.float())
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return float((numerator / denominator).detach())


def _cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Compute whole-tensor cosine similarity in FP32.

    Args:
        actual: Tensor produced by the implementation.
        expected: Reference tensor.

    Returns:
        Cosine similarity as a Python float.
    """
    return float(
        F.cosine_similarity(
            actual.float().flatten(),
            expected.float().flatten(),
            dim=0,
        ).detach()
    )


def _run_two_rank_test(rank: int, test_name: str, rendezvous_file: str) -> None:
    """Run one decorated test body in a two-rank NCCL process group."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(minutes=5),
        device_id=torch.device("cuda", rank),
    )
    try:
        case = DistMoeTwoRankTest(methodName=test_name)
        test = getattr(type(case), test_name).__wrapped__
        test(case)
    finally:
        dist.destroy_process_group()


def with_comms(test: Callable[..., None]) -> Callable[..., None]:
    """Run one unittest method in two local NCCL worker processes."""

    @wraps(test)
    def wrapped(self: unittest.TestCase) -> None:
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _run_two_rank_test,
                args=(test.__name__, f"{directory}/rendezvous"),
                nprocs=2,
                join=True,
            )

    return wrapped


class _BackwardOpRecorder(TorchDispatchMode):
    """Record explicit accumulators at the registered WGRAD boundary."""

    def __init__(self) -> None:
        """Initialize an empty call record."""
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Run one operation and retain its two mutable accumulator inputs."""
        del types
        result = func(*args, **({} if kwargs is None else kwargs))
        if func in (
            torch.ops.dist_moe.bf16_backward_accumulate_.default,
            torch.ops.dist_moe.block_scaled_backward_accumulate_.default,
        ):
            self.calls.append((args[3], args[4]))
        return result


def _reference_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scores: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    experts_output_postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
    clamped: bool = False,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a small routed MoE with ordinary PyTorch operations.

    Args:
        x: Input tokens with shape ``[T, D]``.
        topk_ids: Expert IDs with shape ``[T, K]``.
        topk_scores: Expert weights with shape ``[T, K]``.
        w13: Fused local gate/up weights.
        w2: Local down-projection weights.
        experts_output_postprocess: Optional route-wise output transform.
        clamped: Whether to apply the clamped SwiGLU formula.
        swiglu_alpha: Sigmoid multiplier for clamped SwiGLU.
        swiglu_limit: Preactivation bound for clamped SwiGLU.
        w13_bias: Optional gate/up bias ``[E, 2F]`` added after W13.
        w2_bias: Optional down-projection bias ``[E, D]`` added after W2.

    Returns:
        Reference output with shape ``[T, D]``.
    """
    outputs = []
    intermediate_dim = w2.shape[2]
    for token_index in range(x.shape[0]):
        token_outputs = []
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token_index, slot])
            if expert < 0:
                token_outputs.append(torch.zeros_like(x[token_index]))
                continue
            h13 = F.linear(x[token_index], w13[expert])
            if w13_bias is not None:
                h13 = h13 + w13_bias[expert]
            gate, up = h13.split(intermediate_dim)
            gate_f32, up_f32 = gate.float(), up.float()
            if clamped:
                gate_f32 = torch.minimum(
                    gate_f32,
                    gate_f32.new_tensor(swiglu_limit),
                )
                up_f32 = torch.clamp(
                    up_f32,
                    min=-swiglu_limit,
                    max=swiglu_limit,
                )
                h2 = (
                    gate_f32 * torch.sigmoid(swiglu_alpha * gate_f32) * (up_f32 + 1.0)
                ).to(x.dtype)
            else:
                h2 = (gate_f32 * torch.sigmoid(gate_f32) * up_f32).to(x.dtype)
            route = F.linear(h2, w2[expert])
            if w2_bias is not None:
                route = route + w2_bias[expert]
            token_outputs.append(route)
        stacked = torch.stack(token_outputs)
        if experts_output_postprocess is not None:
            stacked = experts_output_postprocess(stacked)
        outputs.append(
            torch.sum(stacked.float() * topk_scores[token_index, :, None], dim=0).to(
                x.dtype
            )
        )
    return torch.stack(outputs)


def _reference_swiglu_clip_stats(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    w13_EFD: torch.Tensor,
    *,
    clip_limit: float,
) -> torch.Tensor:
    """Count pre-activation clipping events with ordinary PyTorch operations.

    Args:
        x_TD: Local BF16 input tokens.
        topk_expert_ids_TK: Global expert IDs for each route.
        w13_EFD: Local grouped gate/up weights in a single-rank test.
        clip_limit: Strict absolute threshold used by the fused kernel.

    Returns:
        FP32 gate, up-projection, and valid-element counts with shape ``[3]``.
    """
    intermediate_dim = w13_EFD.shape[1] // 2
    gate_count = 0
    up_count = 0
    valid_routes = 0
    for token_index in range(x_TD.shape[0]):
        for route_index in range(topk_expert_ids_TK.shape[1]):
            expert = int(topk_expert_ids_TK[token_index, route_index])
            if expert < 0:
                continue
            valid_routes += 1
            h13_2F = F.linear(x_TD[token_index], w13_EFD[expert])
            gate_F, up_F = h13_2F.split(intermediate_dim)
            gate_count += int((gate_F.float() > clip_limit).sum())
            up_count += int((up_F.float().abs() > clip_limit).sum())
    total = valid_routes * intermediate_dim
    return torch.tensor(
        (gate_count, up_count, total),
        dtype=torch.float32,
        device=x_TD.device,
    )


def _all_gather_cat(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather equal-shaped tensors and concatenate their leading axes.

    Args:
        tensor: Local contiguous tensor.

    Returns:
        Rank-ordered concatenation of every peer tensor.
    """
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered)


def _blockscaled_tolerance(
    format: dist_moe.BlockScaledFormat,
    direction: str,
) -> tuple[float, float, float, float]:
    """Return calibrated elementwise and aggregate numerical bounds.

    Args:
        format: Block-scaled operand format under test.
        direction: ``fwd``, ``grad``, or ``wgrad``.

    Returns:
        Relative tolerance, absolute tolerance, relative-L2 bound, and cosine
        similarity floor.

    Raises:
        ValueError: If the format or direction is unknown.
    """
    bounds = {
        dist_moe.BlockScaledFormat.MXFP8_E4M3: {
            "fwd": (0.05, 0.002, 0.075, 0.9975),
            "grad": (0.05, 0.04, 0.075, 0.9975),
            "wgrad": (0.05, 0.025, 0.075, 0.9975),
        },
        dist_moe.BlockScaledFormat.NVFP4: {
            "fwd": (0.05, 0.007, 0.29, 0.955),
        },
    }
    try:
        return bounds[format][direction]
    except KeyError as error:
        raise ValueError(
            f"unknown block-scaled tolerance {format}/{direction}"
        ) from error


class DistMoeConfigTest(unittest.TestCase):
    """CPU validation tests for the public configuration surface."""

    def test_pdl_launches_is_per_thread_and_restored(self) -> None:
        """PDL is off by default, nests, is restored after an exception, and stays per thread."""
        self.assertFalse(pdl_impl.pdl_enabled())
        with pdl_impl.pdl_launches(True):
            self.assertEqual(
                pdl_impl.pdl_launch_kwargs(), {"PDL": True, "launch_pdl": True}
            )
            with pdl_impl.pdl_launches(False):
                self.assertFalse(pdl_impl.pdl_enabled())
            self.assertTrue(pdl_impl.pdl_enabled())
            other_thread = []
            thread = threading.Thread(
                target=lambda: other_thread.append(pdl_impl.pdl_enabled())
            )
            thread.start()
            thread.join()
            self.assertEqual(other_thread, [False])
        with self.assertRaises(RuntimeError):
            with pdl_impl.pdl_launches(True):
                raise RuntimeError
        self.assertFalse(pdl_impl.pdl_enabled())

    def test_resource_owner_types_are_factory_only(self) -> None:
        """Factory-owned resource types reject incomplete construction."""
        for owner_type in (
            _CommunicationBuffers,
            dist_moe.Context,
            dist_moe.PreparedWeight,
        ):
            with self.subTest(owner_type=owner_type.__name__):
                with self.assertRaisesRegex(TypeError, "use"):
                    owner_type()

    def test_vmm_prefetch_policy_requires_a_boolean(self) -> None:
        """Reject ambiguous truthy values for the context-owned VMM policy."""
        with self.assertRaisesRegex(TypeError, "prefetch must be a bool"):
            dist_moe.VmmConfig(prefetch=1)  # type: ignore[arg-type]

    def test_activation_slot_capacity_policy_is_unambiguous(self) -> None:
        """Reject malformed, conflicting, and inference-only slot policies."""
        base = {
            "max_num_local_input_tokens": 4,
            "hidden_dim": 128,
            "intermediate_dim": 128,
            "top_k": 2,
            "num_experts": 4,
            "max_moe_layers_per_activation_slot": 1,
        }
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            dist_moe.Config(
                **base,
                activation_slot_bytes=4096,
                activation_slot_capacity_factor=1.0,
            )
        for value in (-1, True, 1.5):
            with self.subTest(activation_slot_bytes=value):
                with self.assertRaises((TypeError, ValueError)):
                    dist_moe.Config(**base, activation_slot_bytes=value)
        for value in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    dist_moe.Config(
                        **base,
                        activation_slot_capacity_factor=value,
                    )
        with self.assertRaisesRegex(ValueError, "does not retain saved activations"):
            dist_moe.Config(
                **base,
                activation_slot_capacity_factor=1.0,
                num_activation_slots=0,
                inference=True,
            )

    def test_execution_options_validate_clip_statistics(self) -> None:
        """Clip statistics require a finite limit and CUDA FP32 output."""
        with self.assertRaisesRegex(ValueError, "finite"):
            dist_moe.ExecutionOptions(swiglu_clip_limit=float("inf"))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            dist_moe.ExecutionOptions(
                swiglu_clip_stats_out_3=torch.empty(3, dtype=torch.float32)
            )
        for name in ("validate_expert_ids", "zero_out_padded_callback_inputs"):
            with self.subTest(name=name), self.assertRaisesRegex(TypeError, "bool"):
                dist_moe.ExecutionOptions(**{name: 1})

    def test_registered_bf16_schema_exposes_clip_statistics_mutation(self) -> None:
        """Inference mutates counters while training returns functional state."""
        inference_schema = str(torch.ops.dist_moe.bf16.default._schema)
        training_schema = str(torch.ops.dist_moe.bf16_forward.default._schema)
        clip_training_schema = str(
            torch.ops.dist_moe.bf16_forward_with_clip_stats.default._schema
        )
        self.assertIn("Tensor(e!)? swiglu_clip_stats_out_3", inference_schema)
        self.assertNotIn("!", training_schema)
        self.assertIn("float swiglu_clip_limit", clip_training_schema)
        self.assertNotIn("!", clip_training_schema)

    def test_rejects_nonfinite_scratch_capacity_factors(self) -> None:
        """Scratch capacity factors must define finite allocation bounds."""
        base = {
            "max_num_local_input_tokens": 4,
            "hidden_dim": 128,
            "intermediate_dim": 128,
            "top_k": 2,
            "num_experts": 4,
            "max_moe_layers_per_activation_slot": 1,
        }
        for value in (float("nan"), float("inf"), 0.0):
            with (
                self.subTest(device=value),
                self.assertRaisesRegex(
                    ValueError,
                    "device_scratch_capacity_factor must be finite and positive",
                ),
            ):
                dist_moe.Config(**base, device_scratch_capacity_factor=value)
            with (
                self.subTest(total=value),
                self.assertRaisesRegex(
                    ValueError,
                    "total_scratch_capacity_factor must be finite and positive",
                ),
            ):
                dist_moe.VmmConfig(total_scratch_capacity_factor=value)

    def test_blockscaled_routing_multiple_matches_scale_page_layout(self) -> None:
        """Keep async routing capacity separate from paged GEMM scale alignment."""
        staged = blockscaled_impl._BlockscaledConfig(
            format=_KernelBlockScaledFormat.MXFP8_E4M3,
        )
        cases = (
            (staged, {"BLOCK_SIZE_N": 32, "SWAP_AB": True}, 128, 32),
            (staged, {"BLOCK_SIZE_N": 64, "SWAP_AB": True}, 128, 64),
            (
                blockscaled_impl._BlockscaledConfig(
                    format=_KernelBlockScaledFormat.MXFP8_E4M3,
                    mega=True,
                ),
                {"BLOCK_SIZE_N": 256},
                256,
                256,
            ),
            (staged, None, 128, 128),
        )
        for blockscaled, config, expected_routing, expected_launch in cases:
            compute = blockscaled_impl._ComputeDispatch(
                config=config,
                blockscaled=blockscaled,
            )
            with self.subTest(config=config, mega=blockscaled.mega):
                self.assertEqual(
                    blockscaled_impl._routing_m_multiple(
                        compute,
                        inference_mode=True,
                    ),
                    expected_routing,
                )
                launch_multiple = (
                    blockscaled_impl._inference_routing_multiple(
                        config, mega=blockscaled.mega
                    )
                    if config is not None
                    else blockscaled_impl.dist_bs_gemm.DEFAULT_M_MULTIPLE_OF
                )
                self.assertEqual(launch_multiple, expected_launch)
                self.assertEqual(expected_routing % launch_multiple, 0)

    def test_mega_bprop_rejects_swap_ab_paged_scale_rows(self) -> None:
        """Mega DGRAD+WGRAD rejects SWAP_AB tiles with paged row scales.

        Both bprop entry points must raise before launch because their row-quant
        scale storage is sized by logical rows while the producer writes paged
        rows. Unpaged or unswapped tiles pass the same check.
        """
        rows = dim = 128
        common: dict[str, Any] = {
            "sfb": torch.empty(0),
            "split_sizes": torch.tensor([rows], dtype=torch.int32),
            "symm_mem_buffer": None,
            "x_wgrad_quant": (torch.empty(0), torch.empty(0)),
            "num_sms": 148,
            "config": {"SWAP_AB": True, "BLOCK_SIZE_N": 64},
        }
        with self.assertRaisesRegex(NotImplementedError, "SWAP_AB paged scale"):
            mega_impl.mega_blockscaled_grouped_gemm_dgrad_wgrad_dispatch(
                dy=torch.empty(rows, dim),
                w=torch.empty(1, dim, dim),
                gather_ptrs=torch.zeros(rows, dtype=torch.int64),
                num_out_tokens=rows,
                **common,
            )
        with self.assertRaisesRegex(NotImplementedError, "SWAP_AB paged scale"):
            mega_impl.mega_blockscaled_grouped_gemm_dgrad_wgrad_swiglu_bwd_combine(
                grad_h2=torch.empty(rows, dim),
                h1=torch.empty(rows, 2 * dim),
                w=torch.empty(1, 2 * dim, dim),
                scatter_ptrs=torch.zeros(rows, dtype=torch.int64),
                **common,
            )
        for swap_ab, block_n in ((True, 128), (True, 256), (False, 64)):
            with self.subTest(swap_ab=swap_ab, block_n=block_n):
                mega_impl._reject_paged_swap_ab_bprop(
                    {"SWAP_AB": swap_ab, "BLOCK_SIZE_N": block_n}
                )

    def test_dynamic_mxfp8_mega_weight_quant_recompute_policy(self) -> None:
        """Only callback-generated training quants are omitted from saved state."""
        compute = blockscaled_impl._ComputeDispatch(
            config=None,
            blockscaled=blockscaled_impl._BlockscaledConfig(
                format=_KernelBlockScaledFormat.MXFP8_E4M3,
                mega=True,
            ),
        )

        def preprocess(weight: torch.Tensor) -> torch.Tensor:
            return weight

        self.assertTrue(
            blockscaled_impl._recompute_dynamic_mega_weight_quants(
                compute,
                inference_mode=False,
                weights_preprocess_fn=preprocess,
            )
        )
        self.assertFalse(
            blockscaled_impl._recompute_dynamic_mega_weight_quants(
                compute,
                inference_mode=True,
                weights_preprocess_fn=preprocess,
            )
        )
        self.assertFalse(
            blockscaled_impl._recompute_dynamic_mega_weight_quants(
                compute,
                inference_mode=False,
                weights_preprocess_fn=None,
            )
        )

    def test_bf16_training_compute_configs_are_pinned_by_value(self) -> None:
        """Pin source presets and their row/expert selection boundaries."""
        fprop = grouped_gemm_config_impl.GROUPED_GEMM_CONFIGS[
            grouped_gemm_config_impl.grouped_gemm_training_config()
        ]
        wgrad = grouped_gemm_config_impl.GROUPED_GEMM_CONFIGS[
            grouped_gemm_config_impl.grouped_gemm_training_config(wgrad=True)
        ]
        self.assertEqual(
            fprop,
            {
                "NUM_CTAS": 2,
                "NUM_MMAS": 2,
                "BLOCK_SIZE_M": 512,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "NUM_SMEM_BUFFERS": 4,
                "NUM_TMEM_BUFFERS": 1,
                "NUM_TILE_BUFFERS": 2,
                "EPILOGUE_SUBTILE": 4,
            },
        )
        self.assertEqual(
            wgrad,
            {
                "NUM_CTAS": 2,
                "NUM_MMAS": 2,
                "BLOCK_SIZE_M": 512,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "NUM_SMEM_BUFFERS": 4,
                "NUM_C_STAGES": 2,
                "NUM_TMEM_BUFFERS": 1,
                "NUM_TILE_BUFFERS": 2,
                "EPILOGUE_SUBTILE": 4,
            },
        )
        for is_wgrad, rows, groups, combine, block_m in (
            (False, 8192, 1, False, 256),
            (False, 8193, 1, False, 512),
            (True, 8192, 1, False, 256),
            (True, 8193, 1, False, 512),
            (False, 16384, 64, False, 256),
            (False, 32768, 64, False, 512),
            (False, 16384, 1, True, 256),
        ):
            with self.subTest(
                wgrad=is_wgrad, rows=rows, groups=groups, combine=combine
            ):
                selected = grouped_gemm_config_impl.grouped_gemm_training_config(
                    wgrad=is_wgrad,
                    routed_rows=rows,
                    num_groups=groups,
                    combine_fprop=combine,
                )
                self.assertEqual(
                    grouped_gemm_config_impl.GROUPED_GEMM_CONFIGS[selected][
                        "BLOCK_SIZE_M"
                    ],
                    block_m,
                )

    def test_bf16_inference_fprop_tile_fits_the_busiest_expert(self) -> None:
        """Auto M tile: rows per expert + 2 sqrt, rounded to 64, 1-CTA up to 128."""
        for rows_per_expert, block_m in (
            (49, 64),
            (50, 128),
            (64, 128),
            (65, 128),
            (129, 256),
        ):
            with self.subTest(rows_per_expert=rows_per_expert):
                name = grouped_gemm_config_impl.auto_grouped_gemm_config(
                    GM=rows_per_expert * 64, G=64, N=1024
                )
                self.assertEqual(
                    grouped_gemm_config_impl.GROUPED_GEMM_CONFIGS[name]["BLOCK_SIZE_M"],
                    block_m,
                )

    def test_bf16_inference_fused_swiglu_dispatch_keeps_two_accumulators(
        self,
    ) -> None:
        """Above 256 rows per expert, the SwiGLU-writing dispatch keeps 2 TMEM buffers.

        Plain FPROP still resolves the 512-row tile, whose single accumulator
        would make the next tile's MMAs wait for the slow SwiGLU epilogue.
        """
        multiple = dist_grouped_gemm_impl.NATIVE_SWIGLU_EPILOGUE_TILE_N_MULTIPLE
        for rows_per_expert in (257, 512, 1024):
            for epilogue_tile_n_multiple, expected in (
                (None, {"BLOCK_SIZE_M": 512, "NUM_TMEM_BUFFERS": 1}),
                (multiple, {"NUM_TMEM_BUFFERS": 2}),
            ):
                with self.subTest(
                    rows_per_expert=rows_per_expert,
                    epilogue_tile_n_multiple=epilogue_tile_n_multiple,
                ):
                    name = dist_grouped_gemm_impl._resolve_dist_grouped_gemm_config(
                        config=None,
                        kernel_M=rows_per_expert * 64,
                        G=64,
                        problem_type=grouped_gemm_impl._FPROP,
                        N=8192,
                        num_sms=148,
                        epilogue_tile_n_multiple=epilogue_tile_n_multiple,
                    )
                    config = grouped_gemm_config_impl.GROUPED_GEMM_CONFIGS[name]
                    self.assertEqual({key: config[key] for key in expected}, expected)

    def test_wide_tmem_load_workaround_targets_blackwell(self) -> None:
        """Apply the TMEM-load cap to both Blackwell targets and nothing else."""
        check = grouped_gemm_kernel_impl._tmem_ld_wide_fragment_nvvm_broken
        check.cache_clear()
        self.addCleanup(check.cache_clear)
        for version, capability, expected in (
            ("4.6.1", (9, 0), False),
            ("4.6.1", (10, 0), True),
            ("4.6.1", (10, 3), True),
            ("4.7.1", (10, 3), False),
            ("4.8.0", (10, 3), False),
        ):
            check.cache_clear()
            with (
                mock.patch.object(
                    grouped_gemm_kernel_impl.cutlass, "__version__", version
                ),
                mock.patch.object(torch.cuda, "is_available", return_value=True),
                mock.patch.object(
                    torch.cuda, "get_device_capability", return_value=capability
                ),
            ):
                self.assertEqual(check(), expected)

        self.assertEqual(
            check.cache_info().currsize,
            1,
        )

    def test_wide_tmem_load_workaround_caps_registers_per_thread(self) -> None:
        """Cap each recognized load shape without widening narrow atoms."""
        self.assertIsNone(
            grouped_gemm_kernel_impl.GroupedGemmKernel.TMEM_LOAD_MAX_REGS_PER_THREAD
        )
        self.assertEqual(
            dist_grouped_gemm_impl.DistGroupedGemmKernel.TMEM_LOAD_MAX_REGS_PER_THREAD,
            32,
        )
        tcgen05 = grouped_gemm_kernel_impl.tcgen05
        cases = (
            (tcgen05.Ld32x32bOp, 1, tcgen05.Repetition.x128),
            (tcgen05.Ld16x64bOp, 1, tcgen05.Repetition.x128),
            (tcgen05.Ld16x32bx2Op, 1, tcgen05.Repetition.x128),
            (tcgen05.Ld16x128bOp, 2, tcgen05.Repetition.x64),
            (tcgen05.Ld16x256bOp, 4, tcgen05.Repetition.x32),
        )
        for op_type, regs_per_repetition, widest in cases:
            cap = tcgen05.Repetition(32 // regs_per_repetition)
            for repeat in tcgen05.Repetition:
                if repeat.value > widest.value:
                    continue
                capped = grouped_gemm_kernel_impl._capped_tmem_ld_repetition(
                    op_type(repeat)
                )
                if repeat.value * regs_per_repetition > 32:
                    self.assertIs(capped, cap, (op_type, repeat))
                else:
                    self.assertIsNone(capped, (op_type, repeat))
        self.assertIsNone(grouped_gemm_kernel_impl._capped_tmem_ld_repetition(object()))

    def test_deep_prefill_uses_measured_training_pipeline(self) -> None:
        """Large inference dispatch reuses the measured deep pipeline."""
        config = dist_grouped_gemm_impl._resolve_dist_grouped_gemm_config(
            config=None,
            kernel_M=4 * 8192,
            G=4,
            problem_type=grouped_gemm_impl._FPROP,
            N=8192,
            num_sms=148,
        )
        self.assertEqual(
            config, grouped_gemm_config_impl.grouped_gemm_training_config()
        )

    def test_rejects_invalid_wgrad_dtype(self) -> None:
        """Only BF16 and FP32 weight-gradient destinations are accepted."""
        config_args = dict(
            max_num_local_input_tokens=4,
            hidden_dim=128,
            intermediate_dim=128,
            top_k=2,
            num_experts=4,
            max_moe_layers_per_activation_slot=1,
            activation_slot_bytes=4096,
        )
        self.assertIsNone(dist_moe.Config(**config_args).wgrad_dtype)
        with self.assertRaisesRegex(TypeError, "wgrad_dtype"):
            dist_moe.Config(
                **config_args,
                wgrad_dtype=torch.float16,
            )

    def test_resolves_parameter_wgrad_dtype_contract(self) -> None:
        """Cover public WGRAD dtype precedence and side-effect-free failures."""
        cases = (
            ("parameter fallback", None, None, None, torch.bfloat16, None),
            ("declared BF16", torch.bfloat16, None, None, torch.bfloat16, None),
            ("declared FP32", torch.float32, None, None, torch.float32, None),
            ("configured FP32", None, torch.float32, None, torch.float32, None),
            ("existing FP32", None, None, torch.float32, torch.float32, None),
            (
                "declaration conflict",
                torch.bfloat16,
                torch.float32,
                None,
                None,
                RuntimeError,
            ),
            (
                "destination conflict",
                None,
                torch.float32,
                torch.bfloat16,
                None,
                RuntimeError,
            ),
            ("unsupported dtype", None, torch.float16, None, None, TypeError),
        )
        for name, declared, configured, existing, expected, error_type in cases:
            with self.subTest(name=name):
                parameter = torch.nn.Parameter(torch.empty(6, dtype=torch.bfloat16))
                parameter.grad_dtype = declared
                if existing is not None:
                    parameter.grad = torch.empty_like(parameter, dtype=existing)
                if error_type is not None:
                    with self.assertRaises(error_type):
                        _resolve_wgrad_dtype(parameter, configured)
                else:
                    self.assertIs(
                        _resolve_wgrad_dtype(parameter, configured),
                        expected,
                    )

        w13 = torch.nn.Parameter(torch.empty(6, dtype=torch.bfloat16))
        w2 = torch.nn.Parameter(torch.empty(6, dtype=torch.bfloat16))
        w13.grad_dtype = torch.bfloat16
        w2.grad_dtype = torch.float32
        with self.assertRaisesRegex(RuntimeError, "different WGRAD dtypes"):
            _resolve_parameter_grad_destinations(
                (_weak_parameter_ref(w13), _weak_parameter_ref(w2)),
                torch.Size((1, 2, 3)),
                torch.Size((1, 2, 3)),
                None,
            )
        self.assertIsNone(w13.grad)
        self.assertIsNone(w2.grad)

    def test_rejects_invalid_clamped_swiglu_parameters(self) -> None:
        """Clamped SwiGLU requires finite positive static parameters."""
        for field in ("swiglu_alpha", "swiglu_limit"):
            for value in (0.0, float("nan"), float("inf")):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(
                        ValueError,
                        "finite positive",
                    ),
                ):
                    dist_moe.Config(
                        max_num_local_input_tokens=4,
                        hidden_dim=128,
                        intermediate_dim=128,
                        top_k=2,
                        num_experts=4,
                        max_moe_layers_per_activation_slot=1,
                        activation="swiglu_clamped",
                        **{field: value},
                    )

    def test_block_major_weights_require_bf16_inference(self) -> None:
        """Block-major weights are an opt-in for BF16 inference with 64-aligned F.

        Config-only; the GPU test below covers execution.
        """
        base = dict(
            max_num_local_input_tokens=4,
            hidden_dim=128,
            intermediate_dim=128,
            top_k=2,
            num_experts=4,
            max_moe_layers_per_activation_slot=1,
            block_major_weights=True,
        )
        dist_moe.Config(**base, inference=True, num_activation_slots=0)
        inference = {"inference": True, "num_activation_slots": 0}
        for overrides in (
            {},
            {**inference, "intermediate_dim": 96},
            {**inference, "block_scaled": dist_moe.BlockScaledConfig()},
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, "block_major_weights"),
            ):
                dist_moe.Config(**{**base, **overrides})
        with self.assertRaisesRegex(ValueError, "interleaved_gate_up"):
            dist_moe.Config(
                **{**base, **inference, "block_major_weights": False},
                interleaved_gate_up=True,
            )

    def test_mxfp8_weight_preparation_supports_fake_tensor_tracing(self) -> None:
        """Fake preparation preserves exact packed shapes and backward ownership.

        MXFP8 training shares qdata; inference omits DGRAD. Interleaved Mega
        W13 retains its layout identity without changing the operand shapes.
        """
        cases = (
            (dist_moe.BlockScaledFormat.MXFP8_E4M3, False, False),
            (dist_moe.BlockScaledFormat.MXFP8_E4M3, True, False),
            (dist_moe.BlockScaledFormat.MXFP8_E4M3, True, True),
            (dist_moe.BlockScaledFormat.NVFP4, True, False),
            (dist_moe.BlockScaledFormat.NVFP4, True, True),
        )
        for block_format, inference, fuse_swiglu in cases:
            with (
                self.subTest(
                    format=block_format,
                    inference=inference,
                    fuse_swiglu=fuse_swiglu,
                ),
                FakeTensorMode(),
            ):
                weight = torch.empty(2, 128, 256, dtype=torch.bfloat16, device="cuda")
                prepared = dist_moe.prepare_block_scaled_weight(
                    weight,
                    dist_moe.BlockScaledConfig(
                        format=block_format,
                        pipeline="mega" if fuse_swiglu else "staged",
                    ),
                    inference=inference,
                    fuse_swiglu=fuse_swiglu,
                )

                self.assertIs(prepared.source, weight)
                self.assertIs(prepared.format, block_format)
                self.assertIs(prepared._interleaved_fc13, fuse_swiglu)
                if block_format is dist_moe.BlockScaledFormat.NVFP4:
                    self.assertEqual(prepared.fprop_data.shape, (2, 128, 128))
                    self.assertIs(prepared.fprop_data.dtype, torch.float4_e2m1fn_x2)
                    self.assertEqual(prepared.fprop_scale.shape, (256, 16))
                    self.assertIs(prepared.fprop_scale.dtype, torch.float8_e4m3fn)
                    for scale in (prepared.global_scale, prepared.global_scale_inv):
                        assert scale is not None
                        self.assertEqual(scale.shape, (2,))
                        self.assertIs(scale.dtype, torch.float32)
                else:
                    self.assertEqual(prepared.fprop_data.shape, weight.shape)
                    self.assertIs(prepared.fprop_data.dtype, torch.float8_e4m3fn)
                    self.assertEqual(prepared.fprop_scale.shape, (256, 8))
                    self.assertIs(prepared.fprop_scale.dtype, torch.float8_e8m0fnu)
                    self.assertIsNone(prepared.global_scale)
                    self.assertIsNone(prepared.global_scale_inv)
                if inference:
                    self.assertIsNone(prepared.dgrad_data)
                    self.assertIsNone(prepared.dgrad_scale)
                else:
                    self.assertIs(prepared.fprop_data, prepared.dgrad_data)
                    assert prepared.dgrad_scale is not None
                    self.assertEqual(prepared.dgrad_scale.shape, (512, 4))
                    self.assertIs(prepared.dgrad_scale.dtype, torch.float8_e8m0fnu)

        for inference in (False, True):
            for shape in ((2, 64, 128), (2, 128, 64)):
                with (
                    self.subTest(inference=inference, unsupported_shape=shape),
                    FakeTensorMode(),
                    self.assertRaisesRegex(ValueError, "divisible by 128"),
                ):
                    dist_moe.prepare_block_scaled_weight(
                        torch.empty(shape, dtype=torch.bfloat16, device="cuda"),
                        dist_moe.BlockScaledConfig(),
                        inference=inference,
                    )

    def test_nvfp4_weight_preparation_rejects_odd_columns(self) -> None:
        """Reject an unpaired NVFP4 column before packed preparation loses it."""
        with FakeTensorMode():
            weight = torch.empty(2, 128, 255, dtype=torch.bfloat16, device="cuda")
            with self.assertRaisesRegex(ValueError, "even K"):
                dist_moe.prepare_block_scaled_weight(
                    weight,
                    dist_moe.BlockScaledConfig(format=dist_moe.BlockScaledFormat.NVFP4),
                    inference=True,
                )

    def test_fused_weight_preparation_requires_mega_inference(self) -> None:
        """Interleaved W13 must not enter training or a staged consumer."""
        with FakeTensorMode():
            weight = torch.empty(2, 128, 256, dtype=torch.bfloat16, device="cuda")
            for block_format in (
                dist_moe.BlockScaledFormat.MXFP8_E4M3,
                dist_moe.BlockScaledFormat.NVFP4,
            ):
                for pipeline, inference, message in (
                    ("mega", False, "inference=True"),
                    ("staged", True, "Mega inference pipeline"),
                ):
                    with (
                        self.subTest(format=block_format, pipeline=pipeline),
                        self.assertRaisesRegex(ValueError, message),
                    ):
                        dist_moe.prepare_block_scaled_weight(
                            weight,
                            dist_moe.BlockScaledConfig(
                                format=block_format, pipeline=pipeline
                            ),
                            inference=inference,
                            fuse_swiglu=True,
                        )

    def test_inference_allows_zero_activation_stacks(self) -> None:
        """Inference configuration permits a scratch-only zero-stack layout."""
        config = dist_moe.Config(
            max_num_local_input_tokens=4,
            hidden_dim=128,
            intermediate_dim=128,
            top_k=2,
            num_experts=4,
            max_moe_layers_per_activation_slot=1,
            activation_slot_bytes=None,
            num_activation_slots=0,
            inference=True,
        )
        self.assertTrue(config.inference)
        self.assertEqual(config.num_activation_slots, 0)

    def test_buffer_view_uses_exact_required_size(self) -> None:
        """Exclude surplus backing storage from a VMM-backed activation buffer tensor view."""
        activation_buffer = ActivationBuffer.create_from_buffer(
            buffer=torch.empty(4096, dtype=torch.uint8),
            ep_size=1,
            scratch_mem_size_in_bytes=1024,
            num_activation_slots=1,
            num_moe_layers=1,
            required_size_in_bytes=2048,
        )

        self.assertEqual(activation_buffer.buffer.numel(), 2048)
        self.assertEqual(activation_buffer.activation_slot_bytes, 1024)

    def test_inference_rejects_saved_activation_storage(self) -> None:
        """Inference cannot reserve unused saved-activation storage."""
        with self.assertRaisesRegex(ValueError, "does not retain saved activations"):
            dist_moe.Config(
                max_num_local_input_tokens=4,
                hidden_dim=128,
                intermediate_dim=128,
                top_k=2,
                num_experts=4,
                max_moe_layers_per_activation_slot=1,
                activation_slot_bytes=4096,
                num_activation_slots=0,
                inference=True,
            )

    def test_rejects_unknown_bf16_grouped_gemm_preset(self) -> None:
        """BF16 expert overrides must name an exported preset."""
        with self.assertRaisesRegex(ValueError, "supported BF16 preset"):
            dist_moe.Config(
                max_num_local_input_tokens=4,
                hidden_dim=128,
                intermediate_dim=128,
                top_k=2,
                num_experts=4,
                max_moe_layers_per_activation_slot=1,
                bf16_grouped_gemm_preset="unknown",
            )

    def test_blockscaled_public_formats_are_explicit(self) -> None:
        """The public enum omits unsupported MXFP8 E5M2 and MXFP4 modes."""
        self.assertEqual(dist_moe.BlockScaledFormat.__name__, "BlockScaledFormat")
        self.assertEqual(
            dist_moe.BlockScaledConfig().format,
            dist_moe.BlockScaledFormat.MXFP8_E4M3,
        )
        self.assertEqual(
            set(dist_moe.BlockScaledFormat),
            {
                dist_moe.BlockScaledFormat.MXFP8_E4M3,
                dist_moe.BlockScaledFormat.NVFP4,
            },
        )
        for public_format, kernel_format in (
            (
                dist_moe.BlockScaledFormat.MXFP8_E4M3,
                _KernelBlockScaledFormat.MXFP8_E4M3,
            ),
            (
                dist_moe.BlockScaledFormat.NVFP4,
                _KernelBlockScaledFormat.NVFP4,
            ),
        ):
            with self.subTest(public_format=public_format):
                self.assertIs(
                    blockscaled_impl._kernel_block_scaled_format(public_format),
                    kernel_format,
                )
        self.assertEqual(
            dist_moe.BlockScaledConfig(format=dist_moe.BlockScaledFormat.NVFP4).format,
            dist_moe.BlockScaledFormat.NVFP4,
        )
        with self.assertRaisesRegex(ValueError, "fast_math"):
            dist_moe.BlockScaledConfig(
                format=dist_moe.BlockScaledFormat.NVFP4,
                fast_math=True,
            )

    def test_nvfp4_requires_inference_and_aligned_dimensions(self) -> None:
        """Reject NVFP4 training and shapes outside the kernel contract."""
        nvfp4 = dist_moe.BlockScaledConfig(format=dist_moe.BlockScaledFormat.NVFP4)
        common = {
            "max_num_local_input_tokens": 4,
            "intermediate_dim": 256,
            "top_k": 2,
            "num_experts": 4,
            "max_moe_layers_per_activation_slot": 1,
            "block_scaled": nvfp4,
        }
        with self.assertRaisesRegex(ValueError, "inference-only"):
            dist_moe.Config(hidden_dim=256, **common)
        with self.assertRaisesRegex(ValueError, "multiple of 256"):
            dist_moe.Config(hidden_dim=128, inference=True, **common)

    def test_prepared_weight_storage_membership_survives_release(self) -> None:
        """Storage discovery remains stable after external zero-size release."""
        qdata = torch.empty(8)
        fprop_scale = torch.empty(4)
        dgrad_scale = torch.empty(4)
        prepared = dist_moe.PreparedWeight._create(
            source=torch.empty(8),
            format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
            fprop_data=qdata,
            fprop_scale=fprop_scale,
            dgrad_data=qdata.view(2, 4),
            dgrad_scale=dgrad_scale,
        )

        before = prepared.storage_tensors()
        self.assertEqual(before, (qdata, fprop_scale, dgrad_scale))
        for tensor in before:
            tensor.untyped_storage().resize_(0)
        after = prepared.storage_tensors()

        self.assertEqual(len(after), len(before))
        for actual, expected in zip(after, before, strict=True):
            self.assertIs(actual.untyped_storage(), expected.untyped_storage())

    def test_blockscaled_memory_plan_reports_padded_receive_capacity(self) -> None:
        """The public memory plan exposes topology-aware padded row capacity."""
        config = dist_moe.Config(
            max_num_local_input_tokens=100,
            hidden_dim=128,
            intermediate_dim=128,
            top_k=2,
            num_experts=8,
            max_moe_layers_per_activation_slot=1,
            device_scratch_capacity_factor=1.5,
            block_scaled=dist_moe.BlockScaledConfig(),
        )
        plan = dist_moe.plan_memory(config, ep_size=2)

        self.assertEqual(plan.balanced_recv_rows, 200)
        self.assertEqual(plan.device_scratch_capacity_rows, 768)

    def test_blockscaled_uses_activation_buffer_and_vmm_controls(self) -> None:
        """VMM preserves offset scratch rather than static inference layouts."""
        config = dist_moe.Config(
            max_num_local_input_tokens=128,
            hidden_dim=128,
            intermediate_dim=128,
            top_k=2,
            num_experts=4,
            max_moe_layers_per_activation_slot=1,
            block_scaled=dist_moe.BlockScaledConfig(),
            vmm=dist_moe.VmmConfig(total_scratch_capacity_factor=2.0),
        )
        plan = dist_moe.plan_memory(config, ep_size=2)
        self.assertGreater(plan.total_activation_bytes, 0)
        self.assertGreater(plan.host_scratch_bytes, 0)
        for block_format in (
            dist_moe.BlockScaledFormat.MXFP8_E4M3,
            dist_moe.BlockScaledFormat.NVFP4,
        ):
            with self.subTest(format=block_format):
                inference_config = dataclasses.replace(
                    config,
                    hidden_dim=256,
                    intermediate_dim=256,
                    inference=True,
                    num_sms=8,
                    block_scaled=dist_moe.BlockScaledConfig(
                        format=block_format, pipeline="mega"
                    ),
                )
                vmm_plan = dist_moe.plan_memory(inference_config, ep_size=2)
                model_config = api_impl._model_config(inference_config, ep_size=2)
                self.assertIsNone(vmm_plan.inference_operand_bytes)
                self.assertEqual(
                    vmm_plan.device_scratch_bytes,
                    api_impl._align_memory(model_config.min_buffer_size(0)),
                )
                self.assertGreater(vmm_plan.host_scratch_bytes, 0)
                device_plan = dist_moe.plan_memory(
                    dataclasses.replace(inference_config, vmm=None), ep_size=2
                )
                self.assertIsNotNone(device_plan.inference_operand_bytes)
                self.assertGreater(device_plan.inference_operand_bytes, 0)

    def test_callback_options_require_eager_execution(self) -> None:
        """Only Python-owned execution controls leave the custom-op path."""
        registered_options = dist_moe.ExecutionOptions()
        accumulation_options = dist_moe.ExecutionOptions(inplace_wgrad_accum=True)
        callback_options = dist_moe.ExecutionOptions(
            experts_output_postprocess=lambda value: value
        )
        rmsnorm_options = dist_moe.ExecutionOptions(
            experts_output_postprocess=dist_moe.RMSNormPostprocess(
                eps=1e-6,
                norm_output_dtype=torch.float32,
                output_dtype=torch.bfloat16,
            )
        )
        observed_rmsnorm_options = dist_moe.ExecutionOptions(
            experts_output_postprocess=dist_moe.RMSNormPostprocess(
                eps=1e-6,
                norm_output_dtype=torch.float32,
                output_dtype=torch.bfloat16,
                observe_expert_output_fn=lambda _value: None,
            )
        )
        self.assertFalse(registered_options.requires_eager)
        self.assertFalse(rmsnorm_options.requires_eager)
        self.assertFalse(accumulation_options.requires_eager)
        self.assertTrue(callback_options.requires_eager)
        self.assertTrue(observed_rmsnorm_options.requires_eager)
        self.assertTrue(
            dist_moe.ExecutionOptions(
                weights_preprocess_fn=lambda weight: weight
            ).requires_eager
        )
        self.assertTrue(
            dist_moe.ExecutionOptions(
                wgrad_postprocess_fn=lambda _name, gradient: gradient
            ).requires_eager
        )

    def test_registered_rmsnorm_policy_round_trips_exactly(self) -> None:
        """Preserve every fused RMSNorm field across the custom-op boundary."""
        weight_D = torch.randn(128)
        expected = dist_moe.RMSNormPostprocess(
            eps=1e-6,
            norm_output_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            require_bitwise=False,
            weight=weight_D,
            gain_center=1.0,
            use_kahan=True,
            recompute_rstd=True,
        )

        actual = _rmsnorm_from_registered_args(*_registered_rmsnorm_args(expected))

        assert actual is not None
        self.assertIs(actual.weight, weight_D)
        self.assertEqual(
            dataclasses.replace(actual, weight=None),
            dataclasses.replace(expected, weight=None),
        )
        self.assertIsNone(
            _rmsnorm_from_registered_args(*_registered_rmsnorm_args(None))
        )

    def test_wgrad_destination_contract(self) -> None:
        """Validate an integration-owned destination before kernel mutation."""
        weight_EFD = torch.empty(2, 6, 4, dtype=torch.bfloat16)
        output_EFD = torch.empty_like(weight_EFD, dtype=torch.float32)

        destination = _resolve_wgrad_destination(
            lambda name, shape, dtype, device: (
                output_EFD,
                name == "w13" and shape == weight_EFD.shape,
            ),
            "w13",
            weight_EFD,
            torch.float32,
        )

        self.assertIs(destination.output, output_EFD)
        self.assertTrue(destination.accumulate)
        self.assertIsNone(destination.parameter)
        with self.assertRaisesRegex(RuntimeError, "requested shape"):
            _resolve_wgrad_destination(
                lambda *_args: (torch.empty(1), False),
                "w13",
                weight_EFD,
                torch.float32,
            )

    def test_wgrad_destination_is_an_exclusive_eager_policy(self) -> None:
        """Reject ambiguous ownership and keep destination callbacks eager."""

        def destination_fn(*_args: object) -> tuple[torch.Tensor, bool]:
            """Return a placeholder destination for option validation."""
            return torch.empty(1), False

        self.assertTrue(
            dist_moe.ExecutionOptions(
                wgrad_destination_fn=destination_fn
            ).requires_eager
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            dist_moe.ExecutionOptions(
                inplace_wgrad_accum=True,
                wgrad_destination_fn=destination_fn,
            )

    def test_callback_output_contract(self) -> None:
        """Reject callback results that cannot be replayed safely."""
        input_RD = torch.randn(4, 8)
        _validate_callback_output(input_RD.clone(), input_RD)

        with self.assertRaisesRegex(TypeError, "must return a tensor"):
            _validate_callback_output(None, input_RD)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "preserve route shape"):
            _validate_callback_output(torch.randn(2, 8), input_RD)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            _validate_callback_output(torch.randn(8, 4).T, input_RD)
        with self.assertRaisesRegex(TypeError, "float16, bfloat16, or float32"):
            _validate_callback_output(input_RD.double(), input_RD)
        with self.assertRaisesRegex(ValueError, "changed dtype"):
            _validate_callback_output(
                input_RD.bfloat16(), input_RD, expected_dtype=torch.float32
            )

    def test_wgrad_accumulation_rejects_wgrad_postprocess(self) -> None:
        """A direct destination cannot also use the WGRAD callback."""
        with self.assertRaisesRegex(ValueError, "wgrad_postprocess_fn"):
            dist_moe.ExecutionOptions(
                inplace_wgrad_accum=True,
                wgrad_postprocess_fn=lambda _name, gradient: gradient,
            )

    def test_wgrad_accumulation_resolves_explicit_parameter_owner(self) -> None:
        """A local compute tensor may accumulate into an outer parameter."""
        compute = torch.empty(2, 3, requires_grad=True)
        owner = torch.nn.Parameter(torch.empty(6, dtype=torch.bfloat16))
        owner.grad_dtype = torch.float32
        owner.grad = torch.arange(6, dtype=torch.float32)

        destination = _resolve_parameter_grad(
            _weak_parameter_ref(compute, owner),
            torch.Size((1, 2, 3)),
            None,
        )

        self.assertIs(destination.parameter, owner)
        self.assertTrue(destination.accumulate)
        self.assertIs(destination.dtype, torch.float32)
        assert destination.output is not None
        self.assertEqual(destination.output.data_ptr(), owner.grad.data_ptr())
        with self.assertRaisesRegex(ValueError, "needs gradients"):
            _weak_parameter_ref(torch.empty(2, 3))

    def test_wgrad_parameter_owners_require_accumulation(self) -> None:
        """Explicit owners are meaningful only for direct accumulation."""
        owner = torch.nn.Parameter(torch.empty(1))
        with self.assertRaisesRegex(ValueError, "inplace_wgrad_accum"):
            dist_moe.ExecutionOptions(wgrad_parameter_owners=(owner, owner))

    def test_registered_backward_schemas_separate_wgrad_ownership(self) -> None:
        """Functional and accumulating backwards have fixed ownership modes."""
        cases = (
            (
                torch.ops.dist_moe.bf16_backward.default,
                torch.ops.dist_moe.bf16_backward_accumulate_.default,
            ),
            (
                torch.ops.dist_moe.block_scaled_backward.default,
                torch.ops.dist_moe.block_scaled_backward_accumulate_.default,
            ),
        )
        for functional_op, accumulate_op in cases:
            with self.subTest(op=functional_op):
                self.assertTrue(
                    all(
                        arg.alias_info is None
                        for arg in functional_op._schema.arguments
                    )
                )
            with self.subTest(op=accumulate_op):
                arguments = accumulate_op._schema.arguments
                self.assertEqual(
                    [argument.name for argument in arguments[3:5]],
                    ["accumulator_grad_w13_EFD", "accumulator_grad_w2_EDF"],
                )
                self.assertTrue(
                    all(
                        argument.alias_info is not None and argument.alias_info.is_write
                        for argument in arguments[3:5]
                    )
                )
                self.assertTrue(
                    all(ret.alias_info is None for ret in accumulate_op._schema.returns)
                )


@pytest.mark.gpus_needed_1
@pytest.mark.gb10x
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DistMoeBufferTest(unittest.TestCase):
    """Buffer lifecycle tests that do not compile CuTe kernels."""

    def tearDown(self) -> None:
        """Destroy a process group created by a buffer test."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_training_requires_one_aligned_unit_per_activation_slot(self) -> None:
        """Reject a training buffer whose aligned activation slots are empty."""
        with self.assertRaisesRegex(
            ValueError,
            "at least one aligned unit per activation slot",
        ):
            ActivationBuffer.create_from_buffer(
                buffer=torch.empty(2176, dtype=torch.uint8, device="cuda"),
                ep_size=1,
                scratch_mem_size_in_bytes=2048,
                num_activation_slots=2,
                num_moe_layers=1,
            )

    def test_multi_rank_fake_process_group_models_virtual_peers(self) -> None:
        """Fake symmetric memory exposes one alias for every virtual peer."""
        dist.init_process_group(
            backend="fake",
            store=dist.HashStore(),
            rank=0,
            world_size=2,
        )
        buffers = _CommunicationBuffers.create(
            max_num_local_input_tokens=4,
            hidden_dim=128,
            top_k=2,
            group=dist.group.WORLD,
            device=torch.device("cuda"),
        )

        for buffer in (buffers.routing, buffers.dispatch, buffers.combine):
            self.assertEqual(buffer.hdl.world_size, 2)
            self.assertEqual(buffer.buffer_ptrs_tensor.numel(), 2)
            self.assertEqual(buffer.signal_pad_ptrs_tensor.numel(), 2)
            self.assertEqual(
                buffer.hdl.get_buffer(0, buffer.shape, buffer.dtype).data_ptr(),
                buffer.hdl.get_buffer(1, buffer.shape, buffer.dtype).data_ptr(),
            )
        routing_ids_TK = _routing_ids_view(buffers.routing, 0, (4, 2))
        self.assertEqual(
            routing_ids_TK.data_ptr() - buffers.routing.local().data_ptr(),
            _ROUTING_HEADER_SIZE_BYTES,
        )
        self.assertEqual(
            _routing_token_count_view(buffers.routing, 0).item(),
            0,
        )

    def test_explicit_peer_emulation_does_not_depend_on_backend_name(self) -> None:
        """Custom fake groups can request local peer emulation explicitly."""
        dist.init_process_group(
            backend="fake",
            store=dist.HashStore(),
            rank=0,
            world_size=2,
        )
        with mock.patch.object(dist, "get_backend", return_value="custom_fake"):
            context = dist_moe.create_context(
                group=dist.group.WORLD,
                config=dist_moe.Config(
                    max_num_local_input_tokens=4,
                    hidden_dim=128,
                    intermediate_dim=128,
                    top_k=2,
                    num_experts=4,
                    max_moe_layers_per_activation_slot=1,
                ),
                emulate_peer_buffers=True,
            )
        self.addCleanup(context.close)

        for buffer in (
            context.buffers.routing,
            context.buffers.dispatch,
            context.buffers.combine,
        ):
            self.assertTrue(is_fake_symmetric_memory(buffer))
            self.assertEqual(
                buffer.hdl.get_buffer(0, buffer.shape, buffer.dtype).data_ptr(),
                buffer.hdl.get_buffer(1, buffer.shape, buffer.dtype).data_ptr(),
            )

    def test_activation_slot_selection_is_context_local(self) -> None:
        """Selecting one activation buffer must not mutate another buffer."""
        buffers = [
            ActivationBuffer.create(
                total_size_in_bytes=4096,
                ep_size=1,
                device=torch.device("cuda"),
                scratch_mem_size_in_bytes=2048,
                num_activation_slots=2,
                num_moe_layers=1,
            )
            for _ in range(2)
        ]

        buffers[0].select_activation_slot(1, 1)

        self.assertEqual(buffers[0].activation_slot_id_1.item(), 1)
        self.assertEqual(buffers[1].activation_slot_id_1.item(), 0)
        self.assertEqual(buffers[0].moe_layer_id[1].item(), 0)
        self.assertEqual(buffers[1].moe_layer_id[0].item(), 0)
        with self.assertRaisesRegex(ValueError, "activation slot"):
            buffers[0].select_activation_slot(-1, 1)

    def test_multi_rank_fake_context_supports_bf16_and_mxfp8(self) -> None:
        """Both public training formats build their real two-rank memory plan."""
        dist.init_process_group(
            backend="fake",
            store=dist.HashStore(),
            rank=0,
            world_size=2,
        )
        for blockscaled in (None, dist_moe.BlockScaledConfig()):
            with self.subTest(block_scaled=blockscaled):
                config = dist_moe.Config(
                    max_num_local_input_tokens=128,
                    hidden_dim=128,
                    intermediate_dim=128,
                    top_k=2,
                    num_experts=4,
                    max_moe_layers_per_activation_slot=2,
                    block_scaled=blockscaled,
                )
                context = dist_moe.create_context(
                    group=dist.group.WORLD,
                    config=config,
                )
                try:
                    self.assertEqual(
                        context.memory_plan,
                        dist_moe.plan_memory(config, ep_size=2),
                    )
                    for buffer in (
                        context.buffers.routing,
                        context.buffers.dispatch,
                        context.buffers.combine,
                    ):
                        self.assertEqual(buffer.hdl.world_size, 2)
                        self.assertEqual(buffer.buffer_ptrs_tensor.numel(), 2)
                finally:
                    context.close()

    def test_inference_buffer_uses_full_allocation_as_scratch(self) -> None:
        """Zero-stack inference exposes the complete allocation as scratch."""
        size = 4 * 1024 * 1024
        buffer = ActivationBuffer.create(
            total_size_in_bytes=size,
            ep_size=1,
            device=torch.device("cuda"),
            scratch_mem_size_in_bytes=size,
            num_activation_slots=0,
            num_moe_layers=1,
            inference_mode=True,
        )
        self.assertEqual(buffer.num_activation_slots, 0)
        self.assertEqual(buffer.activation_slot_bytes, 0)
        self.assertEqual(buffer.scratch_region_size, size)
        self.assertEqual(buffer.buffer_offsets.tolist(), [size])
        buffer.select_activation_slot(1, 1)

    def test_select_activation_slot_tracks_stage_depth(self) -> None:
        """Select a physical slot with the stage's actual MoE depth."""
        buffer = ActivationBuffer.create(
            total_size_in_bytes=4096,
            ep_size=1,
            device=torch.device("cuda"),
            scratch_mem_size_in_bytes=2048,
            num_activation_slots=2,
            num_moe_layers=3,
        )

        buffer.select_activation_slot(0, 2)

        self.assertEqual(buffer.activation_slot_id_1.item(), 0)
        self.assertEqual(buffer._num_moe_layers_in_selected_slot, 2)
        self.assertEqual(buffer.moe_layer_id[0].item(), 0)
        with self.assertRaisesRegex(ValueError, "activation slot"):
            buffer.select_activation_slot(2, 1)
        with self.assertRaisesRegex(ValueError, "num_moe_layers"):
            buffer.select_activation_slot(0, 4)


@pytest.mark.gpus_needed_1
@pytest.mark.gb10x
@unittest.skipUnless(_is_blackwell(), "Blackwell CUDA device required")
class DistMoeKernelTest(unittest.TestCase):
    """Numerical tests for CuTe grouped GEMM and distributed MoE."""

    def test_scale_and_sum_selected_routes_match_masked_loads(self) -> None:
        """The inference route select gives the masked path's exact sum.

        Inactive routes hold NaN rows and NaN scores, as unwritten combine
        storage and dropped-route scores can; both paths must ignore them and
        agree bitwise at decode token counts.
        """
        from dist_moe.kernels.triton.broadcast_n_reduction import scale_and_sum

        generator = torch.Generator(device="cuda").manual_seed(5)
        for num_tokens in (1, 64, 128):
            x = torch.randn(
                num_tokens, 8, 2048, device="cuda", generator=generator
            ).bfloat16()
            scale = torch.rand(num_tokens, 8, device="cuda", generator=generator)
            expert_ids = torch.randint(
                0, 256, (num_tokens, 8), device="cuda", generator=generator
            )
            inactive = (
                torch.rand(num_tokens, 8, device="cuda", generator=generator) < 0.4
            )
            expert_ids[inactive] = -1
            x[inactive] = float("nan")
            scale[inactive] = float("nan")
            outputs = [
                scale_and_sum(
                    x,
                    scale,
                    expert_ids=expert_ids,
                    return_copy=False,
                    select_routes=select_routes,
                )[0]
                for select_routes in (False, True)
            ]
            self.assertTrue(torch.isfinite(outputs[0]).all())
            torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)

    def _run_fake_residual_chain(
        self,
        config: dist_moe.Config,
        base_x_TD: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        base_topk_scores_TK: torch.Tensor,
        base_weights: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[torch.Tensor, ...]:
        """Run a multi-rank FakePG residual chain and return owned results.

        Args:
            config: Dist-MoE memory and precision policy.
            base_x_TD: Shared input values for policy comparisons.
            topk_expert_ids_TK: Deterministic routed expert IDs.
            base_topk_scores_TK: Shared router-score values.
            base_weights: Shared high-precision expert-weight values.

        Returns:
            Output, input gradient, and one router-score gradient per layer.
        """
        context = dist_moe.create_context(group=dist.group.WORLD, config=config)
        try:
            x_TD = base_x_TD.detach().clone().requires_grad_()
            score_tensors = []
            output_TD = x_TD
            context.select_activation_slot(
                0,
                config.max_moe_layers_per_activation_slot,
            )
            for w13_E2FD, w2_EDF in base_weights:
                topk_scores_TK = base_topk_scores_TK.detach().clone().requires_grad_()
                score_tensors.append(topk_scores_TK)
                if config.block_scaled is None:
                    w13_operand, w2_operand = w13_E2FD, w2_EDF
                else:
                    w13_operand = dist_moe.prepare_block_scaled_weight(
                        w13_E2FD.flatten(1, 2),
                        config.block_scaled,
                    )
                    w2_operand = dist_moe.prepare_block_scaled_weight(
                        w2_EDF,
                        config.block_scaled,
                    )
                output_TD = output_TD + dist_moe.routed_experts(
                    output_TD,
                    topk_expert_ids_TK,
                    topk_scores_TK,
                    w13_operand,
                    w2_operand,
                    context,
                )
            output_TD.float().square().mean().backward()
            gradient_sources = (x_TD, *score_tensors)
            gradients = []
            for tensor in gradient_sources:
                gradient = tensor.grad
                self.assertIsNotNone(gradient)
                assert gradient is not None
                gradients.append(gradient.detach().clone())
            return output_TD.detach().clone(), *gradients
        finally:
            context.close()

    def test_multi_rank_fake_recompute_matches_saved_activations(self) -> None:
        """Fake peer scatters make saved and recomputed training bitwise equal."""
        dist.init_process_group(
            backend="fake",
            store=dist.HashStore(),
            rank=0,
            world_size=2,
        )
        torch.manual_seed(314159)
        num_tokens, hidden_dim, intermediate_dim = 128, 256, 256
        num_experts, top_k, num_layers = 4, 2, 2
        token_T = torch.arange(num_tokens, device="cuda")[:, None]
        slot_K = torch.arange(top_k, device="cuda")[None, :]
        topk_expert_ids_TK = (token_T * top_k + slot_K) % num_experts
        topk_scores_TK = torch.rand(num_tokens, top_k, device="cuda")
        topk_scores_TK /= topk_scores_TK.sum(dim=1, keepdim=True)

        def randn(*shape: int) -> torch.Tensor:
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16)

        x_TD = randn(num_tokens, hidden_dim)
        weights = tuple(
            (
                randn(
                    num_experts // 2,
                    2,
                    intermediate_dim,
                    hidden_dim,
                )
                * hidden_dim**-0.5,
                randn(
                    num_experts // 2,
                    hidden_dim,
                    intermediate_dim,
                )
                * intermediate_dim**-0.5,
            )
            for _ in range(num_layers)
        )

        for block_scaled in (None, dist_moe.BlockScaledConfig()):
            with self.subTest(block_scaled=block_scaled):
                minimum = dist_moe.Config(
                    max_num_local_input_tokens=num_tokens,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    top_k=top_k,
                    num_experts=num_experts,
                    max_moe_layers_per_activation_slot=num_layers,
                    device_scratch_capacity_factor=1.0,
                    num_activation_slots=1,
                    block_scaled=block_scaled,
                    wgrad_dtype=torch.float32,
                )
                maximum = dataclasses.replace(
                    minimum,
                    activation_slot_bytes=dist_moe.plan_memory(
                        minimum,
                        ep_size=2,
                    ).maximum_useful_activation_slot_bytes,
                )
                expected = self._run_fake_residual_chain(
                    maximum,
                    x_TD,
                    topk_expert_ids_TK,
                    topk_scores_TK,
                    weights,
                )
                actual = self._run_fake_residual_chain(
                    minimum,
                    x_TD,
                    topk_expert_ids_TK,
                    topk_scores_TK,
                    weights,
                )
                self.assertEqual(len(actual), len(expected))
                for actual_tensor, expected_tensor in zip(
                    actual,
                    expected,
                    strict=True,
                ):
                    torch.testing.assert_close(
                        actual_tensor,
                        expected_tensor,
                        rtol=0,
                        atol=0,
                    )

    def test_nvfp4_weight_preparation_owns_inverse_global_scale(self) -> None:
        """Prepared NVFP4 weights retain both global-scale orientations."""
        weight = torch.randn(
            2,
            128,
            256,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        prepared = dist_moe.prepare_block_scaled_weight(
            weight,
            dist_moe.BlockScaledConfig(format=dist_moe.BlockScaledFormat.NVFP4),
        )

        assert prepared.global_scale is not None
        assert prepared.global_scale_inv is not None
        torch.testing.assert_close(
            prepared.global_scale_inv,
            torch.reciprocal(prepared.global_scale),
            rtol=0,
            atol=0,
        )
        storage = prepared.storage_tensors()
        self.assertTrue(all(not tensor.requires_grad for tensor in storage))
        self.assertTrue(all(tensor.grad_fn is None for tensor in storage))
        self.assertTrue(any(tensor is prepared.global_scale for tensor in storage))
        self.assertTrue(any(tensor is prepared.global_scale_inv for tensor in storage))

    def test_nvfp4_token_scale_writes_directly_to_strided_output(self) -> None:
        """Dynamic NVFP4 quantization writes its inverse scale in place."""
        rows, cols = 32, 256
        x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
        expected_q, expected_scale, expected_scale_inv = (
            blockscaled_impl._quantize_nvfp4_per_token(
                x,
                layout=blockscaled_impl.ScaleFactorLayout.NATURAL,
            )
        )
        q_cols, scale_cols = cols // 2, cols // 16
        row_bytes = (q_cols + scale_cols + 4 + 15) // 16 * 16
        packed = torch.empty((rows, row_bytes), dtype=torch.uint8, device="cuda")
        q_out = packed[:, :q_cols].view(torch.float4_e2m1fn_x2)
        scale_out = packed[:, q_cols : q_cols + scale_cols].view(torch.float8_e4m3fn)
        scale_inv_out = packed[:, q_cols + scale_cols : q_cols + scale_cols + 4].view(
            torch.float32
        )[:, 0]

        actual_q, actual_scale, actual_scale_inv = (
            blockscaled_impl._quantize_nvfp4_per_token(
                x,
                layout=blockscaled_impl.ScaleFactorLayout.NATURAL,
                q_out=q_out,
                scale_out=scale_out,
                token_scale_inv_out=scale_inv_out,
            )
        )
        self.assertEqual(actual_q.data_ptr(), q_out.data_ptr())
        self.assertEqual(actual_scale.data_ptr(), scale_out.data_ptr())
        self.assertEqual(actual_scale_inv.data_ptr(), scale_inv_out.data_ptr())
        self.assertTrue(
            torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(
                actual_scale.view(torch.uint8), expected_scale.view(torch.uint8)
            )
        )
        torch.testing.assert_close(actual_scale_inv, expected_scale_inv, rtol=0, atol=0)

    def test_scratch_only_plan_preserves_activation_stack(self) -> None:
        """A grad-free BF16 plan must not leave state for a nonexistent backward."""
        config = ModelConfig(
            dtype=torch.bfloat16,
            hidden_dim=64,
            intermediate_dim=64,
            num_tokens=8,
            topk=1,
            max_imbalance_factor=1.0,
            num_moe_layers=2,
        )
        buffer = ActivationBuffer.create(
            total_size_in_bytes=config.min_buffer_size(),
            ep_size=1,
            device=torch.device("cuda"),
            scratch_mem_size_in_bytes=config.scratch_mem_size,
            num_activation_slots=1,
            num_moe_layers=2,
        )
        recv = torch.tensor([8], dtype=torch.int32, device="cuda")
        offsets_before = buffer.buffer_offsets.clone()
        saved_before = buffer.saved_activation_bytes_per_rank.clone()
        layer_ids_before = buffer.moe_layer_id.clone()

        for _ in range(4):
            plan = get_forward_plan(
                recv,
                recv,
                buffer,
                config,
                scratch_only=True,
            )
            self.assertTrue(plan.need_recompute.item())

        self.assertTrue(torch.equal(buffer.buffer_offsets, offsets_before))
        self.assertTrue(
            torch.equal(buffer.saved_activation_bytes_per_rank, saved_before)
        )
        self.assertTrue(torch.equal(buffer.moe_layer_id, layer_ids_before))

    def test_blockscaled_plan_snapshots_and_rewinds_actual_rows(self) -> None:
        """Save routed-row ownership in the planner launch and rewind its bytes."""
        for mega in (False, True):
            with self.subTest(mega=mega):
                config = ModelConfig(
                    dtype=torch.bfloat16,
                    hidden_dim=256,
                    intermediate_dim=512,
                    num_tokens=128,
                    topk=2,
                    max_imbalance_factor=1.0,
                    num_moe_layers=1,
                    blockscaled_storage=BlockscaledStorageConfig(1, 1, 128),
                    num_local_experts=2,
                    routing_m_multiple_of=128,
                    mega=mega,
                )
                capacity_bytes = config.saved_act_mem_size(
                    config.max_recv_tokens,
                    recompute=False,
                )
                buffer = ActivationBuffer.create(
                    total_size_in_bytes=capacity_bytes + config.scratch_mem_size,
                    ep_size=2,
                    device=torch.device("cuda"),
                    scratch_mem_size_in_bytes=config.scratch_mem_size,
                    num_activation_slots=1,
                    num_moe_layers=1,
                )
                recv = torch.tensor([128], dtype=torch.int32, device="cuda")
                recv_per_rank = torch.tensor(
                    [128, 256], dtype=torch.int64, device="cuda"
                )
                snapshot = torch.empty_like(recv_per_rank)

                forward = get_forward_plan(
                    recv,
                    recv_per_rank,
                    buffer,
                    config,
                    num_recv_tokens_per_rank_snapshot=snapshot,
                )
                actual_bytes = config.saved_act_mem_size(128, recompute=False)
                self.assertTrue(torch.equal(snapshot, recv_per_rank))
                self.assertEqual(buffer.buffer_offsets[0].item(), actual_bytes)
                self.assertLess(actual_bytes, capacity_bytes)

                get_backward_plan(recv, snapshot, forward, buffer, config)
                self.assertEqual(buffer.buffer_offsets[0].item(), 0)
                self.assertEqual(
                    torch.count_nonzero(buffer.saved_activation_bytes_per_rank).item(),
                    0,
                )

    def test_backward_uses_forward_activation_slot(self) -> None:
        """A later stage selection cannot redirect an earlier forward's pop."""
        for storage in (None, BlockscaledStorageConfig(1, 1, 32)):
            with self.subTest(blockscaled=storage is not None):
                num_tokens = 128 if storage is not None else 8
                config = ModelConfig(
                    dtype=torch.bfloat16,
                    hidden_dim=64,
                    intermediate_dim=64,
                    num_tokens=num_tokens,
                    topk=1,
                    max_imbalance_factor=1.0,
                    num_moe_layers=2,
                    blockscaled_storage=storage,
                    num_local_experts=1 if storage is not None else 0,
                    routing_m_multiple_of=128 if storage is not None else None,
                )
                buffer = ActivationBuffer.create(
                    total_size_in_bytes=config.max_buffer_size(2),
                    ep_size=1,
                    device=torch.device("cuda"),
                    scratch_mem_size_in_bytes=config.scratch_mem_size,
                    num_activation_slots=2,
                    num_moe_layers=2,
                )
                recv = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
                buffer.select_activation_slot(0, 1)
                forward = get_forward_plan(recv, recv, buffer, config)
                self.assertEqual(forward.activation_slot_id_1.item(), 0)
                self.assertGreater(buffer.buffer_offsets[0].item(), 0)

                buffer.select_activation_slot(1, 2)
                get_backward_plan(recv, recv, forward, buffer, config)

                self.assertEqual(buffer.buffer_offsets[0].item(), 0)
                self.assertEqual(buffer.moe_layer_id.tolist(), [0, 0])

    def test_balanced_slot_factor_recomputes_only_above_its_soft_budget(self) -> None:
        """Balanced state fits while larger routed state falls back to recompute."""
        for block_scaled in (None, dist_moe.BlockScaledConfig()):
            with self.subTest(block_scaled=block_scaled is not None):
                num_tokens = 128 if block_scaled is not None else 8
                config = dist_moe.Config(
                    max_num_local_input_tokens=num_tokens,
                    hidden_dim=128,
                    intermediate_dim=128,
                    top_k=1,
                    num_experts=4,
                    max_moe_layers_per_activation_slot=1,
                    device_scratch_capacity_factor=2.0,
                    activation_slot_capacity_factor=1.0,
                    block_scaled=block_scaled,
                )
                memory_plan = dist_moe.plan_memory(config, ep_size=2)
                model_config = api_impl._model_config(config, ep_size=2)
                buffer = ActivationBuffer.create(
                    total_size_in_bytes=memory_plan.total_device_buffer_bytes,
                    ep_size=2,
                    device=torch.device("cuda"),
                    scratch_mem_size_in_bytes=memory_plan.device_scratch_bytes,
                    num_activation_slots=1,
                    num_moe_layers=1,
                )
                balanced_rows = model_config._max_recv_tokens(1.0)
                larger_rows = model_config._max_recv_tokens(2.0)

                balanced = torch.tensor(
                    [balanced_rows], dtype=torch.int32, device="cuda"
                )
                balanced_per_rank = torch.full(
                    (2,), balanced_rows, dtype=torch.int64, device="cuda"
                )
                balanced_plan = get_forward_plan(
                    balanced,
                    balanced_per_rank,
                    buffer,
                    model_config,
                )
                self.assertFalse(balanced_plan.need_recompute.item())

                buffer.reset()
                larger = torch.tensor([larger_rows], dtype=torch.int32, device="cuda")
                larger_per_rank = torch.tensor(
                    [larger_rows, 0], dtype=torch.int64, device="cuda"
                )
                larger_plan = get_forward_plan(
                    larger,
                    larger_per_rank,
                    buffer,
                    model_config,
                )
                self.assertTrue(larger_plan.need_recompute.item())

    def test_prepared_mxfp8_weight_refill_is_allocation_stable(self) -> None:
        """Refill preserves storage addresses and exact quantized operands."""
        torch.manual_seed(0)
        weight = torch.randn(
            2,
            128,
            128,
            dtype=torch.bfloat16,
            device="cuda",
        )
        config = dist_moe.BlockScaledConfig()
        prepared = dist_moe.prepare_block_scaled_weight(weight, config)
        storage_tensors = (
            prepared.fprop_data,
            prepared.fprop_scale,
            prepared.dgrad_scale,
        )
        addresses = tuple(tensor.data_ptr() for tensor in storage_tensors)

        weight.add_(0.25)
        refilled = dist_moe.prepare_block_scaled_weight(weight, config, out=prepared)
        reference = dist_moe.prepare_block_scaled_weight(weight, config)

        self.assertEqual(
            addresses,
            tuple(
                tensor.data_ptr()
                for tensor in (
                    refilled.fprop_data,
                    refilled.fprop_scale,
                    refilled.dgrad_scale,
                )
            ),
        )
        for actual, expected in (
            (refilled.fprop_data, reference.fprop_data),
            (refilled.fprop_scale, reference.fprop_scale),
            (refilled.dgrad_data, reference.dgrad_data),
            (refilled.dgrad_scale, reference.dgrad_scale),
        ):
            assert actual is not None and expected is not None
            self.assertTrue(
                torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            )

    def test_publication_helpers_preserve_payloads_and_saved_dispatch(self) -> None:
        """Publish both peer payloads and save the published dispatch conditionally."""
        x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
        expert_ids = torch.arange(16, dtype=torch.int64, device="cuda").view(8, 2)
        dispatch = torch.empty_like(x)
        routing = torch.empty_like(expert_ids, dtype=torch.int16)
        routing_token_count_1 = torch.zeros(1, dtype=torch.int32, device="cuda")
        copy_routing_and_dispatch(
            x=x,
            dispatch=dispatch,
            expert_ids=expert_ids,
            routing=routing,
            routing_token_count_1=routing_token_count_1,
            num_experts=16,
            validate_expert_ids=True,
        )

        offset_bytes = 256
        activation = torch.full(
            (offset_bytes + dispatch.numel() * dispatch.element_size(),),
            0xA5,
            dtype=torch.uint8,
            device="cuda",
        )
        offset = torch.tensor([offset_bytes], dtype=torch.int64, device="cuda")
        copy_dispatch_to_activation(
            dispatch=dispatch,
            activation_buffer=activation,
            activation_offset=offset,
            condition=torch.ones(1, dtype=torch.bool, device="cuda"),
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(dispatch, x, rtol=0, atol=0)
        torch.testing.assert_close(routing, expert_ids.to(torch.int16), rtol=0, atol=0)
        self.assertEqual(routing_token_count_1.item(), x.shape[0])
        saved = activation[offset_bytes:].view(torch.bfloat16).view_as(dispatch)
        torch.testing.assert_close(saved, dispatch, rtol=0, atol=0)

        activation.fill_(0xA5)
        copy_dispatch_to_activation(
            dispatch=dispatch,
            activation_buffer=activation,
            activation_offset=offset,
            condition=torch.zeros(1, dtype=torch.bool, device="cuda"),
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.all(activation == 0xA5))

    def test_fused_publication_preserves_cow_inputs_in_cuda_graph(self) -> None:
        """Read-only publication inputs retain COW storage in eager and replay."""
        x_base = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
        expert_ids_base = torch.arange(16, dtype=torch.int64, device="cuda").view(8, 2)
        x = x_base._lazy_clone()
        expert_ids = expert_ids_base._lazy_clone()
        dispatch = torch.empty_like(x)
        routing = torch.empty_like(expert_ids, dtype=torch.int16)
        routing_token_count_1 = torch.zeros(1, dtype=torch.int32, device="cuda")

        def read_only_data_ptr(tensor: torch.Tensor) -> int:
            return torch._C._data_address(tensor) + (
                tensor.storage_offset() * tensor.element_size()
            )

        x_ptr = read_only_data_ptr(x)
        expert_ids_ptr = read_only_data_ptr(expert_ids)

        def publish() -> None:
            copy_routing_and_dispatch(
                x=x,
                dispatch=dispatch,
                expert_ids=expert_ids,
                routing=routing,
                routing_token_count_1=routing_token_count_1,
                num_experts=16,
                validate_expert_ids=True,
            )

        publish()  # Compile before capture and exercise the eager boundary.
        torch.cuda.synchronize()

        def assert_publication() -> None:
            self.assertTrue(torch._C._is_cow_tensor(x))
            self.assertTrue(torch._C._is_cow_tensor(expert_ids))
            self.assertEqual(read_only_data_ptr(x), x_ptr)
            self.assertEqual(read_only_data_ptr(expert_ids), expert_ids_ptr)
            torch.testing.assert_close(dispatch, x, rtol=0, atol=0)
            torch.testing.assert_close(
                routing, expert_ids.to(torch.int16), rtol=0, atol=0
            )
            self.assertEqual(routing_token_count_1.item(), x.shape[0])

        assert_publication()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            publish()

        dispatch.zero_()
        routing.zero_()
        routing_token_count_1.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert_publication()

    def _assert_planner_reset(self, activation_buffer: ActivationBuffer) -> None:
        """Assert that backward restored the BF16 activation planner.

        Args:
            activation_buffer: Planner state to validate.
        """
        actual = (
            activation_buffer.buffer_offsets,
            activation_buffer.saved_activation_bytes_per_rank,
            activation_buffer.moe_layer_id,
        )
        expected = (
            activation_buffer.initial_buffer_offsets,
            *tuple(torch.zeros_like(tensor) for tensor in actual[1:]),
        )
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)

    def tearDown(self) -> None:
        """Destroy the single-rank fake group used by a kernel test."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def _create_case(
        self,
        *,
        num_experts: int = 4,
        top_k: int = 2,
        max_moe_layers_per_activation_slot: int = 1,
        num_activation_slots: int = 1,
        activation: Literal["swiglu", "swiglu_clamped"] = "swiglu",
        swiglu_alpha: float = 1.702,
        swiglu_limit: float = 7.0,
        inference: bool = True,
        block_scaled: dist_moe.BlockScaledConfig | None = None,
        num_tokens: int = 8,
    ) -> tuple:
        """Create two input sets and one standalone DistMoE context.

        Args:
            num_experts: Number of global experts.
            top_k: Number of unique experts selected per token.
            max_moe_layers_per_activation_slot: Sequential MoE layers sharing one slot.
            num_activation_slots: Number of independent activation slots.
            activation: SwiGLU variant used by the expert MLP.
            swiglu_alpha: Sigmoid multiplier for clamped SwiGLU.
            swiglu_limit: Preactivation bound for clamped SwiGLU.
            inference: Whether to create a scratch-only inference context.
            block_scaled: Optional block-scaled execution policy.
            num_tokens: Tokens per input set (and the context's capacity).

        Returns:
            dist_moe.Context, two ``(x, topk_ids, topk_scores)`` tuples, and the two
            expert weight tensors.
        """
        if not dist.is_initialized():
            dist.init_process_group(
                backend="fake",
                store=dist.HashStore(),
                rank=0,
                world_size=1,
            )
        torch.manual_seed(43)
        hidden_dim, intermediate_dim = 256, 256
        ids_a = (
            torch.arange(num_tokens, device="cuda")[:, None] * top_k
            + torch.arange(top_k, device="cuda")[None, :]
        ) % num_experts
        ids_b = (ids_a + 1) % num_experts
        scores_a = torch.rand(num_tokens, top_k, device="cuda")
        scores_a /= scores_a.sum(dim=1, keepdim=True)
        scores_b = torch.rand(num_tokens, top_k, device="cuda")
        scores_b /= scores_b.sum(dim=1, keepdim=True)
        x_a = torch.randn(
            num_tokens,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        x_b = torch.randn_like(x_a)
        w13 = torch.randn(
            num_experts,
            2 * intermediate_dim,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        w13.mul_(hidden_dim**-0.5)
        w2 = torch.randn(
            num_experts,
            hidden_dim,
            intermediate_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        w2.mul_(intermediate_dim**-0.5)
        context = dist_moe.create_context(
            group=dist.group.WORLD,
            config=dist_moe.Config(
                max_num_local_input_tokens=num_tokens,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                top_k=top_k,
                num_experts=num_experts,
                max_moe_layers_per_activation_slot=max_moe_layers_per_activation_slot,
                device_scratch_capacity_factor=1.0,
                activation_slot_capacity_factor=(None if inference else 1.0),
                num_activation_slots=0 if inference else num_activation_slots,
                block_scaled=block_scaled,
                activation=activation,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
                inference=inference,
            ),
        )
        self.addCleanup(context.close)
        return context, (x_a, ids_a, scores_a), (x_b, ids_b, scores_b), w13, w2

    def test_context_accepts_bounded_token_shapes(self) -> None:
        """Reuse one training context from one row through its capacity."""
        context, case_a, _, w13, w2 = self._create_case(inference=False)
        for num_tokens in (1, 4, 8):
            with self.subTest(num_tokens=num_tokens):
                x_TD, expert_ids_TK, scores_TK = (
                    tensor[:num_tokens].contiguous() for tensor in case_a
                )
                actual_TD = dist_moe.routed_experts(
                    x_TD,
                    expert_ids_TK,
                    scores_TK,
                    w13,
                    w2,
                    context,
                )
                expected_TD = _reference_moe(
                    x_TD,
                    expert_ids_TK,
                    scores_TK,
                    w13,
                    w2,
                )
                torch.testing.assert_close(
                    actual_TD,
                    expected_TD,
                    rtol=2e-2,
                    atol=2e-2,
                )

        for invalid_tokens in (0, 9):
            with self.subTest(invalid_tokens=invalid_tokens):
                tensors = tuple(
                    (
                        tensor[:0]
                        if invalid_tokens == 0
                        else torch.cat((tensor, tensor[:1]), dim=0)
                    ).contiguous()
                    for tensor in case_a
                )
                with self.assertRaisesRegex(ValueError, "must be in"):
                    dist_moe.routed_experts(*tensors, w13, w2, context)

    def test_prepared_weights_accept_bounded_token_shape(self) -> None:
        """Prepared MXFP8 inference accepts a call below context capacity."""
        policy = dist_moe.BlockScaledConfig()
        context, case_a, _, w13_EFD, w2_EDF = self._create_case(
            inference=True,
            block_scaled=policy,
        )
        short_x_TD, short_expert_ids_TK, short_scores_TK = (
            tensor[:4].contiguous() for tensor in case_a
        )
        prepared_w13 = dist_moe.prepare_block_scaled_weight(
            w13_EFD,
            policy,
            inference=True,
        )
        prepared_w2 = dist_moe.prepare_block_scaled_weight(
            w2_EDF,
            policy,
            inference=True,
        )

        with torch.no_grad():
            actual_TD = dist_moe.routed_experts(
                short_x_TD,
                short_expert_ids_TK,
                short_scores_TK,
                prepared_w13,
                prepared_w2,
                context,
                options=dist_moe.ExecutionOptions(validate_expert_ids=True),
            )
        self.assertEqual(actual_TD.shape, short_x_TD.shape)

    def test_padding_routes_match_reference_forward_and_backward(self) -> None:
        """BF16/MXFP8 ignore mixed and fully padded routes in both passes."""
        for block_scaled in (None, dist_moe.BlockScaledConfig()):
            with self.subTest(block_scaled=block_scaled is not None):
                context, case_a, _, base_w13, base_w2 = self._create_case(
                    inference=False,
                    block_scaled=block_scaled,
                )
                base_x, base_ids, base_scores = case_a
                ids_TK = base_ids.clone()
                ids_TK[0] = -1
                ids_TK[1, 1] = -1
                scores_TK = base_scores.clone()
                scores_TK[ids_TK < 0] = 0
                actual_inputs = tuple(
                    tensor.detach().clone().requires_grad_()
                    for tensor in (base_x, scores_TK, base_w13, base_w2)
                )
                reference_inputs = tuple(
                    tensor.detach().clone().requires_grad_()
                    for tensor in (base_x, scores_TK, base_w13, base_w2)
                )

                actual_TD = dist_moe.routed_experts(
                    actual_inputs[0],
                    ids_TK,
                    actual_inputs[1],
                    actual_inputs[2],
                    actual_inputs[3],
                    context,
                    options=dist_moe.ExecutionOptions(validate_expert_ids=True),
                )
                reference_TD = _reference_moe(
                    reference_inputs[0],
                    ids_TK,
                    reference_inputs[1],
                    reference_inputs[2],
                    reference_inputs[3],
                )
                self.assertEqual(torch.count_nonzero(actual_TD[0]).item(), 0)

                grad_TD = torch.randn_like(actual_TD)
                actual_TD.backward(grad_TD)
                reference_TD.backward(grad_TD)
                if block_scaled is None:
                    torch.testing.assert_close(
                        actual_TD,
                        reference_TD,
                        rtol=3e-2,
                        atol=3e-2,
                    )
                else:
                    output_tolerance = _blockscaled_tolerance(
                        block_scaled.format,
                        "fwd",
                    )
                    self.assertLess(
                        _relative_l2(actual_TD, reference_TD),
                        output_tolerance[2],
                    )
                    self.assertGreater(
                        _cosine_similarity(actual_TD, reference_TD),
                        output_tolerance[3],
                    )
                for name, actual, expected in zip(
                    ("x", "scores", "w13", "w2"),
                    actual_inputs,
                    reference_inputs,
                    strict=True,
                ):
                    if block_scaled is None:
                        torch.testing.assert_close(
                            actual.grad,
                            expected.grad,
                            rtol=3e-2,
                            atol=1.0 if name in ("w13", "w2") else 3e-2,
                        )
                    else:
                        grad_tolerance = _blockscaled_tolerance(
                            block_scaled.format,
                            "wgrad" if name in ("w13", "w2") else "grad",
                        )
                        self.assertLess(
                            _relative_l2(actual.grad, expected.grad),
                            grad_tolerance[2],
                        )
                        self.assertGreater(
                            _cosine_similarity(actual.grad, expected.grad),
                            grad_tolerance[3],
                        )
                self.assertEqual(
                    torch.count_nonzero(actual_inputs[0].grad[0]).item(),
                    0,
                )
                self.assertEqual(
                    torch.count_nonzero(actual_inputs[1].grad[ids_TK < 0]).item(),
                    0,
                )
                self._assert_planner_reset(context.activation_buffer)

    def test_mxfp8_staged_and_mega_inference_match_reference(self) -> None:
        """Match staged/Mega MXFP8 inference to the dense reference.

        Dynamic and prepared weights must also produce bitwise-identical
        outputs, proving that preparation changes ownership rather than math.
        """
        for pipeline in ("staged", "mega"):
            with self.subTest(pipeline=pipeline):
                policy = dist_moe.BlockScaledConfig(
                    format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
                    pipeline=pipeline,
                )
                context, case_a, _, w13_EFD, w2_EDF = self._create_case(
                    inference=True,
                    block_scaled=policy,
                )
                x_TD, topk_expert_ids_TK, topk_scores_TK = case_a
                expected_TD = _reference_moe(
                    x_TD,
                    topk_expert_ids_TK,
                    topk_scores_TK,
                    w13_EFD,
                    w2_EDF,
                )
                outputs = []
                for prepared in (False, True):
                    with self.subTest(pipeline=pipeline, prepared=prepared):
                        w13_operand = (
                            dist_moe.prepare_block_scaled_weight(
                                w13_EFD,
                                policy,
                                inference=True,
                            )
                            if prepared
                            else w13_EFD
                        )
                        w2_operand = (
                            dist_moe.prepare_block_scaled_weight(
                                w2_EDF,
                                policy,
                                inference=True,
                            )
                            if prepared
                            else w2_EDF
                        )
                        with torch.no_grad():
                            actual_TD = dist_moe.routed_experts(
                                x_TD,
                                topk_expert_ids_TK,
                                topk_scores_TK,
                                w13_operand,
                                w2_operand,
                                context,
                            )
                        _rtol, _atol, relative_l2, cosine = _blockscaled_tolerance(
                            policy.format,
                            "fwd",
                        )
                        self.assertLess(
                            _relative_l2(actual_TD, expected_TD),
                            relative_l2,
                        )
                        self.assertGreater(
                            _cosine_similarity(actual_TD, expected_TD),
                            cosine,
                        )
                        outputs.append(actual_TD)
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

    def test_blockscaled_modes_are_visible_to_non_strict_tracing(self) -> None:
        """Trace every supported block-scaled training and inference mode."""

        def bind_run(
            w13_EFD: torch.Tensor,
            w2_EDF: torch.Tensor,
            context: dist_moe.Context,
        ):
            """Bind non-tensor tracing state outside the pipeline loop."""

            def run(
                x_TD: torch.Tensor,
                topk_expert_ids_TK: torch.Tensor,
                topk_scores_TK: torch.Tensor,
            ) -> torch.Tensor:
                """Run block-scaled forward through the registered operation."""
                return dist_moe.routed_experts(
                    x_TD,
                    topk_expert_ids_TK,
                    topk_scores_TK,
                    w13_EFD,
                    w2_EDF,
                    context,
                )

            return run

        cases = (
            (dist_moe.BlockScaledFormat.MXFP8_E4M3, False),
            (dist_moe.BlockScaledFormat.MXFP8_E4M3, True),
            (dist_moe.BlockScaledFormat.NVFP4, True),
        )
        for block_format, inference in cases:
            for pipeline in ("staged", "mega"):
                with self.subTest(
                    format=block_format,
                    inference=inference,
                    pipeline=pipeline,
                ):
                    policy = dist_moe.BlockScaledConfig(
                        format=block_format,
                        pipeline=pipeline,
                    )
                    context, case_a, case_b, w13_EFD, w2_EDF = self._create_case(
                        inference=inference,
                        block_scaled=policy,
                    )
                    if inference:
                        w13_operand = dist_moe.prepare_block_scaled_weight(
                            w13_EFD,
                            policy,
                            inference=True,
                        )
                        w2_operand = dist_moe.prepare_block_scaled_weight(
                            w2_EDF,
                            policy,
                            inference=True,
                        )
                    else:
                        w13_operand, w2_operand = w13_EFD, w2_EDF
                        for tensor in (
                            case_a[0],
                            case_a[2],
                            case_b[0],
                            case_b[2],
                            w13_EFD,
                            w2_EDF,
                        ):
                            tensor.requires_grad_()

                    run = bind_run(w13_operand, w2_operand, context)
                    with torch.set_grad_enabled(not inference):
                        traced = make_fx(run)(*case_a)
                    targets = {
                        node.target
                        for node in traced.graph.nodes
                        if node.op == "call_function"
                    }
                    self.assertIn(
                        torch.ops.dist_moe.block_scaled_forward.default,
                        targets,
                    )

                    for x_TD, topk_expert_ids_TK, topk_scores_TK in (case_a, case_b):
                        context.reset()
                        with torch.set_grad_enabled(not inference):
                            expected_TD = run(
                                x_TD,
                                topk_expert_ids_TK,
                                topk_scores_TK,
                            ).detach()
                        context.reset()
                        with torch.set_grad_enabled(not inference):
                            actual_TD = traced(
                                x_TD,
                                topk_expert_ids_TK,
                                topk_scores_TK,
                            )
                        torch.testing.assert_close(
                            actual_TD,
                            expected_TD,
                            rtol=0,
                            atol=0,
                        )
                        context.reset()

    def test_training_cuda_graph_matches_eager_forward_and_backward(self) -> None:
        """Match BF16 and MXFP8 graph outputs and gradients to eager execution."""

        def bind_run(
            context: dist_moe.Context,
            options: dist_moe.ExecutionOptions | None,
        ):
            """Bind context and execution policy outside the pipeline loop."""

            def run(
                x_TD: torch.Tensor,
                topk_expert_ids_TK: torch.Tensor,
                topk_scores_TK: torch.Tensor,
                w13_EFD: torch.Tensor,
                w2_EDF: torch.Tensor,
            ) -> torch.Tensor:
                """Run one graph-captured training invocation."""
                return dist_moe.routed_experts(
                    x_TD,
                    topk_expert_ids_TK,
                    topk_scores_TK,
                    w13_EFD,
                    w2_EDF,
                    context,
                    options=options,
                )

            return run

        policies = (
            ("bf16", None, False, None),
            (
                "mxfp8_staged",
                dist_moe.BlockScaledConfig(
                    format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
                    pipeline="staged",
                ),
                False,
                None,
            ),
            (
                "mxfp8_mega",
                dist_moe.BlockScaledConfig(
                    format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
                    pipeline="mega",
                ),
                False,
                None,
            ),
            ("bf16_accumulate_bf16", None, True, torch.bfloat16),
            ("bf16_accumulate_fp32", None, True, torch.float32),
            (
                "mxfp8_staged_accumulate_bf16",
                dist_moe.BlockScaledConfig(
                    format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
                    pipeline="staged",
                ),
                True,
                torch.bfloat16,
            ),
            (
                "mxfp8_staged_accumulate_fp32",
                dist_moe.BlockScaledConfig(
                    format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
                    pipeline="staged",
                ),
                True,
                torch.float32,
            ),
        )
        for name, policy, inplace_wgrad_accum, grad_dtype in policies:
            with self.subTest(mode=name):
                context, case_a, case_b, w13_EFD, w2_EDF = self._create_case(
                    inference=False,
                    block_scaled=policy,
                )
                w13_EFD.requires_grad_()
                w2_EDF.requires_grad_()
                if grad_dtype is not None:
                    w13_EFD.grad_dtype = grad_dtype
                    w2_EDF.grad_dtype = grad_dtype

                eager_results = []
                for x_TD, ids_TK, scores_TK in (case_a, case_b):
                    eager_args = (
                        x_TD.clone().requires_grad_(),
                        ids_TK.clone(),
                        scores_TK.clone().requires_grad_(),
                        w13_EFD.detach().clone().requires_grad_(),
                        w2_EDF.detach().clone().requires_grad_(),
                    )
                    if inplace_wgrad_accum:
                        assert grad_dtype is not None
                        eager_args[3].grad_dtype = grad_dtype
                        eager_args[4].grad_dtype = grad_dtype
                        eager_args[3].grad = torch.zeros_like(
                            eager_args[3], dtype=grad_dtype
                        )
                        eager_args[4].grad = torch.zeros_like(
                            eager_args[4], dtype=grad_dtype
                        )
                    context.reset()
                    eager_output_TD = bind_run(
                        context,
                        dist_moe.ExecutionOptions(inplace_wgrad_accum=True)
                        if inplace_wgrad_accum
                        else None,
                    )(*eager_args)
                    eager_output_TD.sum().backward()
                    eager_results.append(
                        (
                            eager_output_TD.detach().clone(),
                            tuple(
                                tensor.grad.detach().clone()
                                for tensor in (
                                    eager_args[0],
                                    eager_args[2],
                                    eager_args[3],
                                    eager_args[4],
                                )
                            ),
                        )
                    )
                context.reset()

                sample_args = (
                    case_a[0].clone().requires_grad_(),
                    case_a[1].clone(),
                    case_a[2].clone().requires_grad_(),
                    w13_EFD,
                    w2_EDF,
                )
                if inplace_wgrad_accum:
                    assert grad_dtype is not None
                    w13_EFD.grad = torch.zeros_like(w13_EFD, dtype=grad_dtype)
                    w2_EDF.grad = torch.zeros_like(w2_EDF, dtype=grad_dtype)
                graphed = torch.cuda.make_graphed_callables(
                    bind_run(
                        context,
                        dist_moe.ExecutionOptions(inplace_wgrad_accum=True)
                        if inplace_wgrad_accum
                        else None,
                    ),
                    sample_args,
                    num_warmup_iters=1,
                    allow_unused_input=inplace_wgrad_accum,
                )
                for (x_TD, ids_TK, scores_TK), (eager_output_TD, eager_grads) in zip(
                    (case_a, case_b), eager_results, strict=True
                ):
                    if inplace_wgrad_accum:
                        assert w13_EFD.grad is not None and w2_EDF.grad is not None
                        w13_EFD.grad.zero_()
                        w2_EDF.grad.zero_()
                    else:
                        w13_EFD.grad = None
                        w2_EDF.grad = None
                    graph_args = (
                        x_TD.clone().requires_grad_(),
                        ids_TK.clone(),
                        scores_TK.clone().requires_grad_(),
                        w13_EFD,
                        w2_EDF,
                    )
                    output_TD = graphed(*graph_args)
                    output_TD.sum().backward()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        output_TD,
                        eager_output_TD,
                        rtol=0,
                        atol=0,
                    )
                    for tensor, eager_grad in zip(
                        (graph_args[0], graph_args[2], w13_EFD, w2_EDF),
                        eager_grads,
                        strict=True,
                    ):
                        self.assertIsNotNone(tensor.grad)
                        torch.testing.assert_close(
                            tensor.grad,
                            eager_grad,
                            rtol=0,
                            atol=0,
                        )

    def test_mxfp8_inference_cuda_graph_has_no_autograd_state(self) -> None:
        """Prepared no-grad calls never read storage-free logical weights.

        Compare ordinary prepared execution with metadata-only sources in
        eager, make_fx and CUDA graphs. Both context modes preserve slot state;
        packed NVFP4 must retain logical dimensions for inference planning.
        """

        class StorageFreeWeight(torch.Tensor):
            """Logical weight metadata that rejects every dispatched data op."""

            @staticmethod
            def __new__(cls, source: torch.Tensor) -> torch.Tensor:
                """Keep grouped shape/dtype/device metadata without storage."""
                return torch.Tensor._make_wrapper_subclass(
                    cls,
                    source.shape,
                    strides=source.stride(),
                    dtype=source.dtype,
                    layout=source.layout,
                    device=source.device,
                    requires_grad=source.requires_grad,
                )

            @classmethod
            def __torch_dispatch__(
                cls,
                func: Any,
                types: tuple[type, ...],
                args: tuple[Any, ...] = (),
                kwargs: dict[str, Any] | None = None,
            ) -> Any:
                """Fail before a storage-free source can enter a data operation."""
                raise RuntimeError(f"storage-free logical weight entered {func}")

        for block_format, inference in (
            (dist_moe.BlockScaledFormat.MXFP8_E4M3, False),
            (dist_moe.BlockScaledFormat.MXFP8_E4M3, True),
            (dist_moe.BlockScaledFormat.NVFP4, True),
        ):
            for pipeline in ("staged", "mega"):
                policy = dist_moe.BlockScaledConfig(
                    format=block_format, pipeline=pipeline
                )
                context, case_a, case_b, w13_EFD, w2_EDF = self._create_case(
                    inference=inference,
                    block_scaled=policy,
                    num_experts=64 if pipeline == "staged" else 4,
                )
                prepared_w2 = dist_moe.prepare_block_scaled_weight(
                    w2_EDF, policy, inference=inference
                )
                activation_buffer = context.activation_buffer
                assert activation_buffer is not None
                slot_tensors = (
                    activation_buffer.buffer_offsets,
                    activation_buffer.saved_activation_bytes_per_rank,
                    activation_buffer.moe_layer_id,
                )
                initial_slot_state = tuple(tensor.clone() for tensor in slot_tensors)
                for fuse_swiglu in (
                    (False, True) if pipeline == "mega" and inference else (False,)
                ):
                    with self.subTest(
                        format=block_format,
                        pipeline=pipeline,
                        inference=inference,
                        fuse_swiglu=fuse_swiglu,
                    ):
                        prepared_w13 = dist_moe.prepare_block_scaled_weight(
                            w13_EFD,
                            policy,
                            inference=inference,
                            fuse_swiglu=fuse_swiglu,
                        )
                        reference_run = partial(
                            dist_moe.routed_experts,
                            w13_weight=prepared_w13,
                            w2_weight=prepared_w2,
                            context=context,
                        )
                        run = partial(
                            dist_moe.routed_experts,
                            w13_weight=prepared_w13._with_source(
                                StorageFreeWeight(w13_EFD)
                            ),
                            w2_weight=prepared_w2._with_source(
                                StorageFreeWeight(w2_EDF)
                            ),
                            context=context,
                        )
                        static_inputs = tuple(tensor.clone() for tensor in case_a)
                        with torch.no_grad():
                            eager_TD = run(*static_inputs)
                            torch.testing.assert_close(
                                eager_TD,
                                reference_run(*static_inputs),
                                rtol=0,
                                atol=0,
                            )
                            traced = make_fx(run)(*static_inputs)
                            self.assertIn(
                                torch.ops.dist_moe.block_scaled_forward.default,
                                {
                                    node.target
                                    for node in traced.graph.nodes
                                    if node.op == "call_function"
                                },
                            )
                            torch.testing.assert_close(
                                traced(*static_inputs), eager_TD, rtol=0, atol=0
                            )
                            torch.cuda.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                output_TD = run(*static_inputs)

                            self.assertIsNone(output_TD.grad_fn)
                            for inputs in (case_a, case_b):
                                for static, value in zip(
                                    static_inputs, inputs, strict=True
                                ):
                                    static.copy_(value)
                                eager_TD = reference_run(*inputs)
                                torch.testing.assert_close(
                                    run(*inputs), eager_TD, rtol=0, atol=0
                                )
                                torch.testing.assert_close(
                                    traced(*inputs), eager_TD, rtol=0, atol=0
                                )
                                graph.replay()
                                torch.cuda.synchronize()
                                torch.testing.assert_close(
                                    output_TD, eager_TD, rtol=0, atol=0
                                )
                                expected_TD = _reference_moe(*inputs, w13_EFD, w2_EDF)
                                _rtol, _atol, relative_l2, cosine = (
                                    _blockscaled_tolerance(block_format, "fwd")
                                )
                                self.assertLess(
                                    _relative_l2(output_TD, expected_TD), relative_l2
                                )
                                self.assertGreater(
                                    _cosine_similarity(output_TD, expected_TD), cosine
                                )
                            for actual, expected in zip(
                                slot_tensors, initial_slot_state, strict=True
                            ):
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )

    def test_non_power_of_two_topk_matches_reference(self) -> None:
        """DSV3-style non-power-of-two top-k supports forward and backward."""
        context, case_a, _, w13, w2 = self._create_case(
            num_experts=8,
            top_k=6,
            inference=False,
        )
        x, ids, scores = case_a
        x.requires_grad_()
        scores.requires_grad_()
        w13.requires_grad_()
        w2.requires_grad_()
        reference_inputs = [
            tensor.detach().clone().requires_grad_() for tensor in (x, scores, w13, w2)
        ]

        actual = dist_moe.routed_experts(x, ids, scores, w13, w2, context)
        expected = _reference_moe(
            reference_inputs[0],
            ids,
            reference_inputs[1],
            reference_inputs[2],
            reference_inputs[3],
        )

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        self.assertLess(_relative_l2(actual, expected), 5e-3)
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)
        for name, tensor, reference_tensor in zip(
            ("x", "topk_scores", "w13", "w2"),
            (x, scores, w13, w2),
            reference_inputs,
        ):
            self.assertIsNotNone(tensor.grad, f"missing {name} gradient")
            torch.testing.assert_close(
                tensor.grad,
                reference_tensor.grad,
                rtol=2e-2,
                atol=2e-2,
            )

    def test_bf16_registered_rmsnorm_backward_matches_reference(self) -> None:
        """Match registered BF16 RMSNorm output and gradients to reference.

        Planner reset is also required so the fused callback cannot leak buffer
        state into the next layer invocation.
        """
        context, case_a, _, w13_EFD, w2_EDF = self._create_case(inference=False)
        x_TD, topk_expert_ids_TK, topk_scores_TK = case_a
        topk_expert_ids_TK = topk_expert_ids_TK.clone()
        topk_expert_ids_TK[0] = -1
        topk_scores_TK = topk_scores_TK.clone()
        topk_scores_TK[0] = 0
        actual_inputs = tuple(
            tensor.detach().clone().requires_grad_()
            for tensor in (x_TD, topk_scores_TK, w13_EFD, w2_EDF)
        )
        reference_inputs = tuple(
            tensor.detach().clone().requires_grad_() for tensor in actual_inputs
        )
        eps = 1e-6
        options = dist_moe.ExecutionOptions(
            experts_output_postprocess=dist_moe.RMSNormPostprocess(
                eps=eps,
                norm_output_dtype=torch.float32,
                output_dtype=torch.bfloat16,
                require_bitwise=False,
                recompute_rstd=True,
            )
        )

        actual_TD = dist_moe.routed_experts(
            actual_inputs[0],
            topk_expert_ids_TK,
            actual_inputs[1],
            actual_inputs[2],
            actual_inputs[3],
            context,
            options=options,
        )
        expected_TD = _reference_moe(
            reference_inputs[0],
            topk_expert_ids_TK,
            reference_inputs[1],
            reference_inputs[2],
            reference_inputs[3],
            experts_output_postprocess=lambda h3_KD: F.rms_norm(
                h3_KD.float(),
                (h3_KD.shape[-1],),
                eps=eps,
            ),
        )
        grad_output_TD = torch.randn_like(actual_TD)
        actual_TD.backward(grad_output_TD)
        expected_TD.backward(grad_output_TD)

        self.assertEqual(torch.count_nonzero(actual_TD[0]).item(), 0)
        torch.testing.assert_close(actual_TD, expected_TD, rtol=2e-2, atol=2e-2)
        for name, actual, expected in zip(
            ("x", "topk_scores", "w13", "w2"),
            actual_inputs,
            reference_inputs,
            strict=True,
        ):
            torch.testing.assert_close(
                actual.grad,
                expected.grad,
                rtol=3e-2,
                atol=3e-2,
                msg=lambda message, name=name: f"{name}: {message}",
            )
        self._assert_planner_reset(context.activation_buffer)

    def test_bf16_inference_presets_match_default(self) -> None:
        """Every BF16 grouped-GEMM preset compiles and matches the default bitwise."""
        context, case_a, _, w13_EFD, w2_EDF = self._create_case(inference=True)
        with torch.no_grad():
            expected_TD = dist_moe.routed_experts(*case_a, w13_EFD, w2_EDF, context)
            for preset in get_args(dist_moe.Bf16GroupedGemmPreset):
                with self.subTest(preset=preset):
                    config = dataclasses.replace(
                        context.config, bf16_grouped_gemm_preset=preset
                    )
                    preset_context = dist_moe.create_context(
                        group=dist.group.WORLD, config=config
                    )
                    self.addCleanup(preset_context.close)
                    actual_TD = dist_moe.routed_experts(
                        *case_a, w13_EFD, w2_EDF, preset_context
                    )
                    torch.testing.assert_close(actual_TD, expected_TD, rtol=0, atol=0)

    def test_bf16_callback_accumulates_closed_parameter_grad(self) -> None:
        """Match callback padding sanitation and its closed parameter gradient."""
        context, case_a, _, w13_EFD, w2_EDF = self._create_case(inference=False)
        x_TD, topk_expert_ids_TK, topk_scores_TK = case_a
        topk_expert_ids_TK = topk_expert_ids_TK.clone()
        topk_expert_ids_TK[0] = -1
        topk_scores_TK = topk_scores_TK.clone()
        topk_scores_TK[0] = 0
        actual_inputs = tuple(
            tensor.detach().clone().requires_grad_()
            for tensor in (x_TD, topk_scores_TK, w13_EFD, w2_EDF)
        )
        reference_inputs = tuple(
            tensor.detach().clone().requires_grad_() for tensor in actual_inputs
        )
        scale_D = torch.randn(
            x_TD.shape[-1],
            dtype=x_TD.dtype,
            device=x_TD.device,
            requires_grad=True,
        )
        reference_scale_D = scale_D.detach().clone().requires_grad_()

        def callback(h3_MD: torch.Tensor) -> torch.Tensor:
            """Require Dist-MoE to zero invalid callback-input rows."""
            h3_TKD = h3_MD.view(*topk_expert_ids_TK.shape, -1)
            self.assertEqual(
                torch.count_nonzero(h3_TKD[topk_expert_ids_TK < 0]).item(),
                0,
            )
            return h3_MD * scale_D

        actual_TD = dist_moe.routed_experts(
            actual_inputs[0],
            topk_expert_ids_TK,
            actual_inputs[1],
            actual_inputs[2],
            actual_inputs[3],
            context,
            options=dist_moe.ExecutionOptions(
                experts_output_postprocess=callback,
                zero_out_padded_callback_inputs=True,
            ),
        )
        expected_TD = _reference_moe(
            reference_inputs[0],
            topk_expert_ids_TK,
            reference_inputs[1],
            reference_inputs[2],
            reference_inputs[3],
            experts_output_postprocess=lambda h3_RD: h3_RD * reference_scale_D,
        )
        grad_output_TD = torch.randn_like(actual_TD)
        actual_TD.backward(grad_output_TD)
        expected_TD.backward(grad_output_TD)

        torch.testing.assert_close(actual_TD, expected_TD, rtol=2e-2, atol=2e-2)
        for name, actual, expected in zip(
            ("x", "topk_scores", "w13", "w2", "callback_scale"),
            (*actual_inputs, scale_D),
            (*reference_inputs, reference_scale_D),
            strict=True,
        ):
            torch.testing.assert_close(
                actual.grad,
                expected.grad,
                rtol=3e-2,
                atol=3e-2,
                msg=lambda message, name=name: f"{name}: {message}",
            )
        self._assert_planner_reset(context.activation_buffer)

    def test_bf16_modes_are_visible_to_non_strict_tracing(self) -> None:
        """Trace BF16 training and inference through their registered ops."""

        def bind_run(
            w13_EFD: torch.Tensor,
            w2_EDF: torch.Tensor,
            context: dist_moe.Context,
        ):
            """Bind context and weights outside the training/inference loop."""

            def run(
                x_TD: torch.Tensor,
                topk_expert_ids_TK: torch.Tensor,
                topk_scores_TK: torch.Tensor,
            ) -> torch.Tensor:
                """Run BF16 forward through the registered operation."""
                return dist_moe.routed_experts(
                    x_TD,
                    topk_expert_ids_TK,
                    topk_scores_TK,
                    w13_EFD,
                    w2_EDF,
                    context,
                )

            return run

        for inference in (False, True):
            with self.subTest(inference=inference):
                context, case_a, case_b, w13, w2 = self._create_case(
                    inference=inference
                )
                if not inference:
                    for tensor in (case_a[0], case_a[2], w13, w2):
                        tensor.requires_grad_()
                run = bind_run(w13, w2, context)
                traced = make_fx(run)(*case_a)
                targets = {
                    node.target
                    for node in traced.graph.nodes
                    if node.op == "call_function"
                }
                expected_target = (
                    torch.ops.dist_moe.bf16.default
                    if inference
                    else torch.ops.dist_moe.bf16_forward.default
                )
                self.assertIn(expected_target, targets)

                for x_TD, ids_TK, scores_TK in (case_a, case_b):
                    context.reset()
                    expected_TD = run(
                        x_TD,
                        ids_TK,
                        scores_TK,
                    ).detach()
                    context.reset()
                    actual_TD = traced(x_TD, ids_TK, scores_TK)
                    torch.testing.assert_close(
                        actual_TD,
                        expected_TD,
                        rtol=0,
                        atol=0,
                    )
                    context.reset()

    def test_clip_statistics_are_graph_visible(self) -> None:
        """Clip counters are exact, optional, and visible to tracing and AC."""
        context, case_a, _, w13, w2 = self._create_case(inference=False)
        x_TD, topk_expert_ids_TK, topk_scores_TK = case_a
        x_TD.copy_(
            torch.where(
                torch.arange(x_TD.shape[0], device=x_TD.device)[:, None] % 2 == 0,
                torch.ones_like(x_TD),
                -torch.ones_like(x_TD),
            )
        )
        w13.zero_()
        w13[:, : w2.shape[2]].fill_(1)
        w13[:, w2.shape[2] :].fill_(-1)
        case_a[0].requires_grad_()
        case_a[2].requires_grad_()
        w13.requires_grad_()
        w2.requires_grad_()
        x_TD, topk_expert_ids_TK, topk_scores_TK = case_a
        stats_out_3 = torch.full((3,), -1, dtype=torch.float32, device="cuda")
        options = dist_moe.ExecutionOptions(
            swiglu_clip_stats_out_3=stats_out_3,
            swiglu_clip_limit=0.0,
        )
        baseline_TD = dist_moe.routed_experts(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            w13,
            w2,
            context,
        ).detach()
        context.reset()
        observed_TD = dist_moe.routed_experts(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            w13,
            w2,
            context,
            options=options,
        )
        torch.testing.assert_close(observed_TD, baseline_TD, rtol=0, atol=0)
        torch.testing.assert_close(
            stats_out_3,
            _reference_swiglu_clip_stats(
                x_TD,
                topk_expert_ids_TK,
                w13,
                clip_limit=0.0,
            ),
            rtol=0,
            atol=0,
        )
        context.reset()
        traced = make_fx(
            lambda x_TD, topk_expert_ids_TK, topk_scores_TK: dist_moe.routed_experts(
                x_TD,
                topk_expert_ids_TK,
                topk_scores_TK,
                w13,
                w2,
                context,
                options=options,
            )
        )(*case_a)

        context.reset()
        traced(*case_a)
        expected_total = case_a[0].shape[0] * case_a[1].shape[1] * w2.shape[2]
        self.assertEqual(stats_out_3[2].item(), expected_total)
        self.assertGreaterEqual(stats_out_3[0].item(), 0)
        self.assertGreaterEqual(stats_out_3[1].item(), 0)
        self.assertLessEqual(stats_out_3[0].item(), expected_total)
        self.assertLessEqual(stats_out_3[1].item(), expected_total)
        custom_ops = [
            node
            for node in traced.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.dist_moe.bf16_forward_with_clip_stats.default
        ]
        self.assertEqual(len(custom_ops), 1)
        self.assertEqual(len(custom_ops[0].args), 19)
        self.assertTrue(
            any(
                node.op == "call_function"
                and node.target is torch.ops.aten.copy_.default
                for node in traced.graph.nodes
            )
        )

    def test_rmsnorm_inference_is_visible_to_non_strict_tracing(self) -> None:
        """Non-strict tracing preserves fused RMSNorm inside the opaque op."""
        context, case_a, case_b, w13_EFD, w2_EDF = self._create_case()
        eps = 1e-6
        options = dist_moe.ExecutionOptions(
            experts_output_postprocess=dist_moe.RMSNormPostprocess(
                eps=eps,
                norm_output_dtype=torch.float32,
                output_dtype=torch.bfloat16,
                require_bitwise=False,
            )
        )
        traced = make_fx(
            lambda x_TD, topk_expert_ids_TK, topk_scores_TK: dist_moe.routed_experts(
                x_TD,
                topk_expert_ids_TK,
                topk_scores_TK,
                w13_EFD,
                w2_EDF,
                context,
                options=options,
            )
        )(*case_a)
        targets = {
            node.target for node in traced.graph.nodes if node.op == "call_function"
        }
        self.assertIn(torch.ops.dist_moe.bf16.default, targets)

        for x_TD, topk_expert_ids_TK, topk_scores_TK in (case_a, case_b):
            expected_TD = _reference_moe(
                x_TD,
                topk_expert_ids_TK,
                topk_scores_TK,
                w13_EFD,
                w2_EDF,
                experts_output_postprocess=lambda h3_KD: F.rms_norm(
                    h3_KD.float(),
                    (h3_KD.shape[-1],),
                    eps=eps,
                ),
            )
            actual_TD = traced(x_TD, topk_expert_ids_TK, topk_scores_TK)
            torch.testing.assert_close(actual_TD, expected_TD, rtol=2e-2, atol=2e-2)

    def test_custom_op_rejects_storage_from_another_owner(self) -> None:
        """Reject every graph-visible storage that is not context-owned."""
        context, case_a, _, w13_EFD, w2_EDF = self._create_case()
        x_TD, topk_expert_ids_TK, topk_scores_TK = case_a
        storages = [
            context.activation_buffer.buffer,
            context.buffers.routing.local(),
            context.buffers.dispatch.local(),
            context.buffers.combine.local(),
        ]
        for index, name in enumerate(("activation", "routing", "dispatch", "combine")):
            supplied = list(storages)
            supplied[index] = torch.empty(1, dtype=torch.uint8, device=x_TD.device)
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    ValueError,
                    f"{name}_storage does not match the context-owned tensor view",
                ),
            ):
                torch.ops.dist_moe.bf16.default(
                    x_TD,
                    topk_expert_ids_TK,
                    topk_scores_TK,
                    w13_EFD,
                    w2_EDF,
                    None,
                    None,
                    *_registered_rmsnorm_args(None),
                    supplied[0],
                    context.activation_buffer.activation_slot_id_1,
                    context.activation_buffer._num_moe_layers_in_selected_slot,
                    *supplied[1:],
                    None,
                    7.0,
                    False,
                    context.context_id,
                )

        activation_alias = storages[0][:-1]
        self.assertEqual(activation_alias.data_ptr(), storages[0].data_ptr())
        with self.assertRaisesRegex(
            ValueError,
            "activation_storage does not match the context-owned tensor view",
        ):
            torch.ops.dist_moe.bf16.default(
                x_TD,
                topk_expert_ids_TK,
                topk_scores_TK,
                w13_EFD,
                w2_EDF,
                None,
                None,
                *_registered_rmsnorm_args(None),
                activation_alias,
                context.activation_buffer.activation_slot_id_1,
                context.activation_buffer._num_moe_layers_in_selected_slot,
                *storages[1:],
                None,
                7.0,
                False,
                context.context_id,
            )

    def test_training_call_uses_selected_microbatch_stack(self) -> None:
        """Backward pops the slot selected by its matching forward."""
        context, case_a, _, w13, w2 = self._create_case(
            max_moe_layers_per_activation_slot=1,
            num_activation_slots=2,
            inference=False,
        )
        context.select_activation_slot(1, 1)
        x_TD, topk_expert_ids_TK, topk_scores_TK = case_a
        inputs = (x_TD, topk_scores_TK, w13, w2)
        for tensor in inputs:
            tensor.requires_grad_()
        reference_inputs = tuple(
            tensor.detach().clone().requires_grad_() for tensor in inputs
        )

        actual_TD = dist_moe.routed_experts(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            w13,
            w2,
            context,
        )
        expected_TD = _reference_moe(
            reference_inputs[0],
            topk_expert_ids_TK,
            reference_inputs[1],
            reference_inputs[2],
            reference_inputs[3],
        )
        grad_TD = torch.randn_like(actual_TD)
        actual_TD.backward(grad_TD)
        expected_TD.backward(grad_TD)

        torch.testing.assert_close(actual_TD, expected_TD, rtol=2e-2, atol=2e-2)
        for actual, expected in zip(inputs, reference_inputs, strict=True):
            torch.testing.assert_close(
                actual.grad,
                expected.grad,
                rtol=3e-2,
                atol=3e-2,
            )
        self.assertEqual(
            context.activation_buffer.buffer_offsets[1],
            context.activation_buffer.initial_buffer_offsets[1],
        )
        self.assertEqual(
            context.activation_buffer.saved_activation_bytes_per_rank[1].item(),
            0,
        )

    def test_sac_preserves_forward_activation_slot(self) -> None:
        """SAC backward consumes the slot selected by the original forward."""
        cases = (
            ("bf16", None, torch.ops.dist_moe.bf16_forward.default),
            (
                "mxfp8",
                dist_moe.BlockScaledConfig(
                    format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
                    pipeline="staged",
                ),
                torch.ops.dist_moe.block_scaled_forward.default,
            ),
        )
        for name, block_scaled, forward_op in cases:
            with self.subTest(name=name):
                context, case_a, _, w13, w2 = self._create_case(
                    num_activation_slots=2,
                    inference=False,
                    block_scaled=block_scaled,
                )
                x, ids, scores = case_a
                base_inputs = (x, scores, w13, w2)
                grad_output = torch.randn_like(x)

                def inputs(
                    values: tuple[torch.Tensor, ...] = base_inputs,
                ) -> tuple[torch.Tensor, ...]:
                    """Clone one independent differentiable input set."""
                    return tuple(
                        tensor.detach().clone().requires_grad_() for tensor in values
                    )

                def layer(
                    input_x: torch.Tensor,
                    input_scores: torch.Tensor,
                    input_w13: torch.Tensor,
                    input_w2: torch.Tensor,
                    expert_ids: torch.Tensor = ids,
                    owner: dist_moe.Context = context,
                ) -> torch.Tensor:
                    """Run one registered DistMoE layer."""
                    return dist_moe.routed_experts(
                        input_x,
                        expert_ids,
                        input_scores,
                        input_w13,
                        input_w2,
                        owner,
                    )

                context.select_activation_slot(0, 1)
                reference_inputs = inputs()
                reference = layer(*reference_inputs)
                reference.backward(grad_output)
                reference_gradients = tuple(
                    tensor.grad.clone() for tensor in reference_inputs
                )
                self._assert_planner_reset(context.activation_buffer)
                context.reset()

                calls = 0
                effectful_ops = []

                def checkpointed_layer(
                    *values: torch.Tensor,
                    owner: dist_moe.Context = context,
                    run=layer,
                ) -> torch.Tensor:
                    """Select a different live slot during SAC recomputation."""
                    nonlocal calls
                    owner.select_activation_slot(min(calls, 1), 1)
                    calls += 1
                    return run(*values)

                def policy(
                    _ctx,
                    op,
                    *_args,
                    saved_ops=effectful_ops,
                    **_kwargs,
                ) -> CheckpointPolicy:
                    """Cache ordered operations while recomputing pure work."""
                    if has_effects(op):
                        saved_ops.append(op)
                        return CheckpointPolicy.MUST_SAVE
                    return CheckpointPolicy.PREFER_RECOMPUTE

                checkpoint_inputs = inputs()
                actual = checkpoint(
                    checkpointed_layer,
                    *checkpoint_inputs,
                    use_reentrant=False,
                    context_fn=lambda: create_selective_checkpoint_contexts(policy),
                    early_stop=False,
                )
                self.assertEqual(calls, 1)
                actual.backward(grad_output)
                self.assertEqual(calls, 2)

                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                for actual_input, expected_gradient in zip(
                    checkpoint_inputs,
                    reference_gradients,
                    strict=True,
                ):
                    torch.testing.assert_close(
                        actual_input.grad,
                        expected_gradient,
                        rtol=0,
                        atol=0,
                    )
                self.assertIn(forward_op, effectful_ops)
                self._assert_planner_reset(context.activation_buffer)

    def test_clip_statistics_survive_activation_checkpointing(self) -> None:
        """Checkpoint recomputation preserves output, gradients, and counters."""
        context, case_a, _, w13_EFD, w2_EDF = self._create_case(inference=False)
        x_TD, topk_expert_ids_TK, topk_scores_TK = case_a
        inputs = (x_TD, topk_scores_TK, w13_EFD, w2_EDF)
        for tensor in inputs:
            tensor.requires_grad_()
        stats_out_3 = torch.zeros(3, dtype=torch.float32, device="cuda")
        options = dist_moe.ExecutionOptions(
            swiglu_clip_stats_out_3=stats_out_3,
            swiglu_clip_limit=0.0,
        )

        def run(
            input_TD: torch.Tensor,
            scores_TK: torch.Tensor,
            input_w13_EFD: torch.Tensor,
            input_w2_EDF: torch.Tensor,
        ) -> torch.Tensor:
            """Run the clip-enabled registered BF16 boundary."""
            return dist_moe.routed_experts(
                input_TD,
                topk_expert_ids_TK,
                scores_TK,
                input_w13_EFD,
                input_w2_EDF,
                context,
                options=options,
            )

        eager_inputs = tuple(
            tensor.detach().clone().requires_grad_() for tensor in inputs
        )
        eager_TD = run(*eager_inputs)
        grad_output_TD = torch.randn_like(eager_TD)
        eager_TD.backward(grad_output_TD)
        eager_grads = tuple(tensor.grad.clone() for tensor in eager_inputs)
        expected_stats_3 = stats_out_3.clone()
        context.reset()
        stats_out_3.zero_()

        checkpoint_inputs = tuple(
            tensor.detach().clone().requires_grad_() for tensor in inputs
        )
        checkpointed_TD = checkpoint(
            run,
            *checkpoint_inputs,
            use_reentrant=False,
        )
        checkpointed_TD.backward(grad_output_TD)

        torch.testing.assert_close(checkpointed_TD, eager_TD, rtol=0, atol=0)
        torch.testing.assert_close(stats_out_3, expected_stats_3, rtol=0, atol=0)
        for actual, expected in zip(checkpoint_inputs, eager_grads, strict=True):
            torch.testing.assert_close(actual.grad, expected, rtol=0, atol=0)

    def test_clip_statistics_replay_in_inference_cuda_graph(self) -> None:
        """Inference graph replay rewrites clip counters at stable addresses."""
        context, case_a, case_b, w13_EFD, w2_EDF = self._create_case()
        static_x_TD, static_ids_TK, static_scores_TK = (
            tensor.clone() for tensor in case_a
        )
        padded_case = tuple(tensor.clone() for tensor in case_b)
        padded_case[1][-1] = -1
        padded_case[2][-1] = 0
        stats_out_3 = torch.full((3,), -1, dtype=torch.float32, device="cuda")
        options = dist_moe.ExecutionOptions(
            swiglu_clip_stats_out_3=stats_out_3,
            swiglu_clip_limit=0.0,
        )
        with torch.no_grad():
            dist_moe.routed_experts(
                static_x_TD,
                static_ids_TK,
                static_scores_TK,
                w13_EFD,
                w2_EDF,
                context,
                options=options,
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output_TD = dist_moe.routed_experts(
                    static_x_TD,
                    static_ids_TK,
                    static_scores_TK,
                    w13_EFD,
                    w2_EDF,
                    context,
                    options=options,
                )

        for x_TD, ids_TK, scores_TK in (case_a, case_b, padded_case):
            static_x_TD.copy_(x_TD)
            static_ids_TK.copy_(ids_TK)
            static_scores_TK.copy_(scores_TK)
            stats_out_3.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output_TD,
                _reference_moe(x_TD, ids_TK, scores_TK, w13_EFD, w2_EDF),
                rtol=2e-2,
                atol=2e-2,
            )
            torch.testing.assert_close(
                stats_out_3,
                _reference_swiglu_clip_stats(
                    x_TD,
                    ids_TK,
                    w13_EFD,
                    clip_limit=0.0,
                ),
                rtol=0,
                atol=0,
            )

    def _assert_mxfp8_training_effects_preserve_eager_bitwise(
        self,
        *,
        activation: Literal["swiglu", "swiglu_clamped"],
        pipeline: Literal["staged", "mega"],
        options: dist_moe.ExecutionOptions | None = None,
    ) -> None:
        """Compare eager, registered, and checkpointed MXFP8 execution.

        Args:
            activation: SwiGLU variant under test.
            pipeline: Block-scaled pipeline under test.
            options: Optional invocation controls under test.
        """
        dist.init_process_group(
            backend="fake",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
        )
        torch.manual_seed(71)
        num_tokens, hidden_dim, intermediate_dim = 128, 256, 256
        num_experts, top_k = 4, 2
        policy = dist_moe.BlockScaledConfig(
            format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
            pipeline=pipeline,
        )
        context = dist_moe.create_context(
            group=dist.group.WORLD,
            config=dist_moe.Config(
                max_num_local_input_tokens=num_tokens,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                top_k=top_k,
                num_experts=num_experts,
                max_moe_layers_per_activation_slot=1,
                num_activation_slots=2,
                block_scaled=policy,
                activation=activation,
            ),
        )
        self.addCleanup(context.close)
        ids = (
            torch.arange(num_tokens, device="cuda")[:, None] * top_k
            + torch.arange(top_k, device="cuda")[None, :]
        ) % num_experts
        base_inputs = (
            torch.randn(
                num_tokens,
                hidden_dim,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            torch.rand(num_tokens, top_k, device="cuda"),
            torch.randn(
                num_experts,
                2 * intermediate_dim,
                hidden_dim,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            torch.randn(
                num_experts,
                hidden_dim,
                intermediate_dim,
                device="cuda",
                dtype=torch.bfloat16,
            ),
        )
        grad_output = torch.randn_like(base_inputs[0])

        def inputs() -> tuple[torch.Tensor, ...]:
            """Clone one independent differentiable input set."""
            return tuple(
                tensor.detach().clone().requires_grad_() for tensor in base_inputs
            )

        context.select_activation_slot(0, 1)
        eager_inputs = inputs()
        calls = mock.Mock()
        with (
            mock.patch.object(
                blockscaled_impl,
                "_ep_barrier",
                wraps=blockscaled_impl._ep_barrier,
            ) as barrier,
            mock.patch.object(
                blockscaled_impl,
                "dist_dispatch_routing",
                wraps=blockscaled_impl.dist_dispatch_routing,
            ) as routing,
            mock.patch.object(
                blockscaled_impl,
                "copy_dispatch_to_activation",
                wraps=blockscaled_impl.copy_dispatch_to_activation,
            ) as save_dispatch,
        ):
            calls.attach_mock(barrier, "barrier")
            calls.attach_mock(routing, "routing")
            calls.attach_mock(save_dispatch, "save_dispatch")
            eager = dist_moe.routed_experts(
                eager_inputs[0],
                ids,
                eager_inputs[1],
                eager_inputs[2],
                eager_inputs[3],
                context,
                options=options,
            )
        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            ["barrier", "routing", "save_dispatch", "barrier"],
        )
        # Spy cycles must not retain CUDA arguments for GC during later captures.
        calls.reset_mock()
        eager.backward(grad_output)
        eager_gradients = tuple(tensor.grad.clone() for tensor in eager_inputs)

        def run(
            input_x: torch.Tensor,
            input_scores: torch.Tensor,
            input_w13: torch.Tensor,
            input_w2: torch.Tensor,
        ) -> torch.Tensor:
            """Run block-scaled DistMoE through the async memory planner."""
            return dist_moe.routed_experts(
                input_x,
                ids,
                input_scores,
                input_w13,
                input_w2,
                context,
                options=options,
            )

        self.assertTrue(has_effects(torch.ops.dist_moe.block_scaled_forward.default))
        self.assertTrue(has_effects(torch.ops.dist_moe.block_scaled_backward.default))
        effectful_ops = []

        def effect_policy(_ctx, op, *_args, **_kwargs) -> CheckpointPolicy:
            """Save hidden state transitions and recompute pure operations."""
            if has_effects(op):
                effectful_ops.append(op)
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        def checkpointed(*values: torch.Tensor) -> torch.Tensor:
            """Apply the same effect-aware policy used by full checkpointing."""
            return checkpoint(
                run,
                *values,
                use_reentrant=False,
                context_fn=lambda: create_selective_checkpoint_contexts(effect_policy),
                early_stop=False,
            )

        context.select_activation_slot(1, 1)
        actual_inputs = inputs()
        actual = checkpointed(*actual_inputs)
        actual.backward(grad_output)

        torch.testing.assert_close(actual, eager, rtol=0, atol=0)
        for actual_input, expected_gradient in zip(
            actual_inputs,
            eager_gradients,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_input.grad,
                expected_gradient,
                rtol=0,
                atol=0,
            )
        if options is None or not options.requires_eager:
            self.assertIn(
                torch.ops.dist_moe.block_scaled_forward.default,
                effectful_ops,
            )
        torch.cuda.synchronize()
        context.close()

    def test_mxfp8_training_effects_preserve_eager_bitwise(self) -> None:
        """Deterministic MXFP8 preserves effectful execution bitwise."""
        self._assert_mxfp8_training_effects_preserve_eager_bitwise(
            activation="swiglu",
            pipeline="staged",
        )

    def test_wgrad_accumulators_are_visible_to_joint_make_fx(self) -> None:
        """BF16 and MXFP8 joint graphs expose both WGRAD destinations."""
        dist.init_process_group(
            backend="fake",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
        )
        options = dist_moe.ExecutionOptions(inplace_wgrad_accum=True)
        cases = (
            (None, torch.ops.dist_moe.bf16_backward_accumulate_.default),
            (
                dist_moe.BlockScaledConfig(
                    format=dist_moe.BlockScaledFormat.MXFP8_E4M3
                ),
                torch.ops.dist_moe.block_scaled_backward_accumulate_.default,
            ),
        )

        def bind_joint_forward_backward(context):
            """Bind the process-local context around one joint trace."""

            def joint_forward_backward(
                input_x_TD: torch.Tensor,
                input_expert_ids_TK: torch.Tensor,
                input_scores_TK: torch.Tensor,
                input_w13_EFD: torch.Tensor,
                input_w2_EDF: torch.Tensor,
                output_grad_TD: torch.Tensor,
                accumulator_w13_EFD: torch.Tensor,
                accumulator_w2_EDF: torch.Tensor,
            ) -> torch.Tensor:
                """Run one GraphTrainer-shaped forward and backward."""
                input_w13_EFD.grad_dtype = accumulator_w13_EFD.dtype
                input_w2_EDF.grad_dtype = accumulator_w2_EDF.dtype
                input_w13_EFD.grad = accumulator_w13_EFD
                input_w2_EDF.grad = accumulator_w2_EDF
                output_TD = dist_moe.routed_experts(
                    input_x_TD,
                    input_expert_ids_TK,
                    input_scores_TK,
                    input_w13_EFD,
                    input_w2_EDF,
                    context,
                    options=options,
                )
                output_TD.backward(output_grad_TD)
                return output_TD

            return joint_forward_backward

        cases_with_dtype = (
            (policy, expected_backward, grad_dtype)
            for policy, expected_backward in cases
            for grad_dtype in (torch.bfloat16, torch.float32)
        )
        for policy, expected_backward, grad_dtype in cases_with_dtype:
            with self.subTest(policy=policy, grad_dtype=grad_dtype):
                context, case, _, w13_EFD, w2_EDF = self._create_case(
                    inference=False,
                    block_scaled=policy,
                )
                x_TD, topk_expert_ids_TK, topk_scores_TK = case
                inputs = (x_TD, topk_expert_ids_TK, topk_scores_TK, w13_EFD, w2_EDF)
                for tensor in (x_TD, topk_scores_TK, w13_EFD, w2_EDF):
                    tensor.requires_grad_()
                w13_EFD.grad_dtype = grad_dtype
                w2_EDF.grad_dtype = grad_dtype
                grad_output_TD = torch.randn_like(x_TD)
                accumulator_grad_w13_EFD = torch.zeros_like(w13_EFD, dtype=grad_dtype)
                accumulator_grad_w2_EDF = torch.zeros_like(w2_EDF, dtype=grad_dtype)

                traced = make_fx(
                    bind_joint_forward_backward(context),
                    tracing_mode="fake",
                    _allow_non_fake_inputs=True,
                )(
                    *inputs,
                    grad_output_TD,
                    accumulator_grad_w13_EFD,
                    accumulator_grad_w2_EDF,
                )
                placeholders = [
                    node for node in traced.graph.nodes if node.op == "placeholder"
                ]
                backward_nodes = [
                    node
                    for node in traced.graph.nodes
                    if node.op == "call_function" and node.target is expected_backward
                ]
                self.assertEqual(len(backward_nodes), 1)
                self.assertTrue(backward_nodes[0].is_impure())
                self.assertIs(backward_nodes[0].args[-2], grad_dtype)
                for argument, placeholder in zip(
                    backward_nodes[0].args[3:5],
                    placeholders[-2:],
                    strict=True,
                ):
                    self.assertIs(argument.target, torch.ops.aten.view.default)
                    self.assertIs(argument.args[0], placeholder)
                context.close()

    def test_mxfp8_clamped_staged_effects_preserve_eager_bitwise(self) -> None:
        """Clamped staged MXFP8 preserves effectful execution bitwise."""
        self._assert_mxfp8_training_effects_preserve_eager_bitwise(
            activation="swiglu_clamped",
            pipeline="staged",
        )

    def test_mxfp8_clamped_mega_effects_preserve_eager_bitwise(self) -> None:
        """Clamped mega MXFP8 preserves effectful execution bitwise."""
        self._assert_mxfp8_training_effects_preserve_eager_bitwise(
            activation="swiglu_clamped",
            pipeline="mega",
        )

    def test_mxfp8_training_context_no_grad_is_scratch_only(self) -> None:
        """No-grad staged/Mega evaluation leaves training slot state untouched."""
        for pipeline in ("staged", "mega"):
            with self.subTest(pipeline=pipeline):
                context, case_a, case_b, w13_EFD, w2_EDF = self._create_case(
                    inference=False,
                    block_scaled=dist_moe.BlockScaledConfig(pipeline=pipeline),
                )
                x_TD, topk_expert_ids_TK, topk_scores_TK = case_a
                for tensor in (x_TD, topk_scores_TK, w13_EFD, w2_EDF):
                    tensor.requires_grad_()
                activation_buffer = context.activation_buffer
                assert activation_buffer is not None
                slot_state = tuple(
                    tensor.clone()
                    for tensor in (
                        activation_buffer.buffer_offsets,
                        activation_buffer.saved_activation_bytes_per_rank,
                        activation_buffer.moe_layer_id,
                    )
                )

                def run(
                    context: dist_moe.Context,
                    input_TD: torch.Tensor,
                    ids_TK: torch.Tensor,
                    scores_TK: torch.Tensor,
                    input_w13_EFD: torch.Tensor,
                    input_w2_EDF: torch.Tensor,
                ) -> torch.Tensor:
                    """Run one MXFP8 call through the public training context."""
                    return dist_moe.routed_experts(
                        input_TD,
                        ids_TK,
                        scores_TK,
                        input_w13_EFD,
                        input_w2_EDF,
                        context,
                    )

                def assert_slot_state(
                    activation_buffer: ActivationBuffer,
                    slot_state: tuple[torch.Tensor, ...],
                ) -> None:
                    """Require every activation-slot counter to remain unchanged."""
                    for actual, expected in zip(
                        (
                            activation_buffer.buffer_offsets,
                            activation_buffer.saved_activation_bytes_per_rank,
                            activation_buffer.moe_layer_id,
                        ),
                        slot_state,
                        strict=True,
                    ):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

                bound_run = partial(run, context)
                bound_assert_slot_state = partial(
                    assert_slot_state,
                    activation_buffer,
                    slot_state,
                )

                with torch.no_grad():
                    eager_TD = bound_run(
                        x_TD,
                        topk_expert_ids_TK,
                        topk_scores_TK,
                        w13_EFD,
                        w2_EDF,
                    )
                    self.assertFalse(eager_TD.requires_grad)
                    bound_assert_slot_state()

                    traced = make_fx(bound_run)(
                        x_TD,
                        topk_expert_ids_TK,
                        topk_scores_TK,
                        w13_EFD,
                        w2_EDF,
                    )
                    custom_ops = [
                        node
                        for node in traced.graph.nodes
                        if node.op == "call_function"
                        and node.target
                        is torch.ops.dist_moe.block_scaled_forward.default
                    ]
                    self.assertEqual(len(custom_ops), 1)
                    self.assertIs(custom_ops[0].args[-2], False)
                    traced_TD = traced(
                        x_TD,
                        topk_expert_ids_TK,
                        topk_scores_TK,
                        w13_EFD,
                        w2_EDF,
                    )
                    torch.testing.assert_close(traced_TD, eager_TD, rtol=0, atol=0)
                    bound_assert_slot_state()

                    static_inputs = tuple(tensor.detach().clone() for tensor in case_a)
                    static_w13_EFD = w13_EFD.detach().clone()
                    static_w2_EDF = w2_EDF.detach().clone()
                    bound_run(*static_inputs, static_w13_EFD, static_w2_EDF)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_output_TD = bound_run(
                            *static_inputs,
                            static_w13_EFD,
                            static_w2_EDF,
                        )
                    for inputs in (case_a, case_b):
                        for static, value in zip(static_inputs, inputs, strict=True):
                            static.copy_(value)
                        expected_TD = bound_run(
                            *inputs,
                            static_w13_EFD,
                            static_w2_EDF,
                        )
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            graph_output_TD,
                            expected_TD,
                            rtol=0,
                            atol=0,
                        )
                        bound_assert_slot_state()

                training_TD = bound_run(
                    x_TD,
                    topk_expert_ids_TK,
                    topk_scores_TK,
                    w13_EFD,
                    w2_EDF,
                )
                torch.testing.assert_close(training_TD, eager_TD, rtol=0, atol=0)
                training_TD.backward(torch.randn_like(training_TD))
                self._assert_planner_reset(activation_buffer)

    def test_inference_cuda_graphs_replay_bounded_shapes(self) -> None:
        """One context backs fixed CUDA graphs at multiple bounded T values."""
        context, case_a, case_b, w13, w2 = self._create_case()
        padded_case = tuple(tensor.clone() for tensor in case_b)
        padded_case[1][-1] = -1
        padded_case[2][-1] = 0
        captures = []
        for num_tokens in (4, 8):
            static_inputs = tuple(
                tensor[:num_tokens].contiguous().clone() for tensor in case_a
            )
            dist_moe.routed_experts(*static_inputs, w13, w2, context)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_output = dist_moe.routed_experts(
                    *static_inputs,
                    w13,
                    w2,
                    context,
                )
            captures.append((num_tokens, static_inputs, graph, captured_output))

        for num_tokens, static_inputs, graph, captured_output in captures:
            for values in (case_a, padded_case):
                inputs = tuple(tensor[:num_tokens].contiguous() for tensor in values)
                for static, value in zip(static_inputs, inputs, strict=True):
                    static.copy_(value)
                graph.replay()
                torch.cuda.synchronize()
                actual = captured_output.clone()
                expected = _reference_moe(*inputs, w13, w2)
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
                self.assertLess(_relative_l2(actual, expected), 5e-3)

    def _run_nvfp4_inference_cuda_graph_replay(
        self,
        pipeline: Literal["staged", "mega"],
        *,
        activation: Literal["swiglu", "swiglu_clamped"] = "swiglu",
    ) -> None:
        """Replay prepared NVFP4 inference through one public pipeline.

        Args:
            pipeline: Block-scaled pipeline name.
            activation: SwiGLU variant under test.
        """
        policy = dist_moe.BlockScaledConfig(
            format=dist_moe.BlockScaledFormat.NVFP4,
            pipeline=pipeline,
        )
        context, case_a, case_b, w13_E2FD, w2_EDF = self._create_case(
            activation=activation,
            block_scaled=policy,
        )
        static_x_TD, static_ids_TK, static_scores_TK = (
            tensor.clone() for tensor in case_a
        )
        padded_case = tuple(tensor.clone() for tensor in case_b)
        padded_case[1][-1] = -1
        padded_case[2][-1] = 0
        prepared_w13 = dist_moe.prepare_block_scaled_weight(w13_E2FD, policy)
        prepared_w2 = dist_moe.prepare_block_scaled_weight(w2_EDF, policy)

        with torch.no_grad():
            dist_moe.routed_experts(
                static_x_TD,
                static_ids_TK,
                static_scores_TK,
                prepared_w13,
                prepared_w2,
                context,
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_output_TD = dist_moe.routed_experts(
                    static_x_TD,
                    static_ids_TK,
                    static_scores_TK,
                    prepared_w13,
                    prepared_w2,
                    context,
                )

            replayed_outputs = []
            for x_TD, ids_TK, scores_TK in (case_a, case_b, padded_case):
                static_x_TD.copy_(x_TD)
                static_ids_TK.copy_(ids_TK)
                static_scores_TK.copy_(scores_TK)
                graph.replay()
                torch.cuda.synchronize()
                actual_TD = captured_output_TD.clone()
                expected_TD = dist_moe.routed_experts(
                    x_TD,
                    ids_TK,
                    scores_TK,
                    prepared_w13,
                    prepared_w2,
                    context,
                )
                torch.testing.assert_close(actual_TD, expected_TD, rtol=0, atol=0)
                if torch.all(ids_TK[-1] < 0):
                    self.assertEqual(torch.count_nonzero(actual_TD[-1]).item(), 0)
                replayed_outputs.append(actual_TD)
            self.assertFalse(torch.equal(replayed_outputs[0], replayed_outputs[1]))

    def test_nvfp4_inference_cuda_graph_matrix_matches_eager(self) -> None:
        """Match staged/Mega and plain/clamped NVFP4 replay to eager bitwise."""
        for pipeline in ("staged", "mega"):
            for activation in ("swiglu", "swiglu_clamped"):
                with self.subTest(pipeline=pipeline, activation=activation):
                    self._run_nvfp4_inference_cuda_graph_replay(
                        pipeline,
                        activation=activation,
                    )

    def test_single_rank_forward_and_backward_match_reference(self) -> None:
        """Fused operation matches a routed PyTorch reference and all gradients."""
        dist.init_process_group(
            backend="fake",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
        )
        torch.manual_seed(29)
        num_tokens, hidden_dim, intermediate_dim = 8, 256, 256
        num_experts, top_k = 4, 2
        ids = (
            torch.arange(num_tokens, device="cuda")[:, None] * top_k
            + torch.arange(top_k, device="cuda")[None, :]
        ) % num_experts
        scores = torch.rand(
            num_tokens,
            top_k,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        scores = (scores / scores.sum(dim=1, keepdim=True)).detach().requires_grad_()
        x = torch.randn(
            num_tokens,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        w13 = (
            torch.randn(
                num_experts,
                2,
                intermediate_dim,
                hidden_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * hidden_dim**-0.5
        ).requires_grad_()
        w2 = (
            torch.randn(
                num_experts,
                hidden_dim,
                intermediate_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * intermediate_dim**-0.5
        ).requires_grad_()
        config = dist_moe.Config(
            max_num_local_input_tokens=num_tokens,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            top_k=top_k,
            num_experts=num_experts,
            max_moe_layers_per_activation_slot=1,
            activation_slot_bytes=64 * 1024 * 1024,
            wgrad_dtype=torch.float32,
        )
        context = dist_moe.create_context(
            group=dist.group.WORLD,
            config=config,
        )

        reference_inputs = [
            tensor.detach().clone().requires_grad_(tensor.requires_grad)
            for tensor in (x, scores, w13, w2)
        ]
        reference = _reference_moe(
            reference_inputs[0],
            ids,
            reference_inputs[1],
            reference_inputs[2].flatten(1, 2),
            reference_inputs[3],
        )
        actual = dist_moe.routed_experts(x, ids, scores, w13, w2, context)
        torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
        self.assertLess(_relative_l2(actual, reference), 5e-3)
        self.assertGreater(_cosine_similarity(actual, reference), 0.999)

        grad = torch.randn_like(actual)
        actual.backward(grad)
        reference.backward(grad)
        for name, actual_grad, expected_tensor in zip(
            ("x", "topk_scores", "w13", "w2"),
            (x.grad, scores.grad, w13.grad, w2.grad),
            reference_inputs,
        ):
            expected_grad = expected_tensor.grad
            self.assertIsNotNone(actual_grad, f"missing {name} gradient")
            self.assertIsNotNone(expected_grad, f"missing reference {name} gradient")
            torch.testing.assert_close(
                actual_grad.float(),
                expected_grad.float(),
                rtol=3e-2,
                atol=3e-2,
                msg=lambda message, name=name: f"{name}: {message}",
            )
            self.assertLess(
                _relative_l2(actual_grad, expected_grad),
                1e-2,
                f"{name} relative L2 error",
            )
            self.assertGreater(
                _cosine_similarity(actual_grad, expected_grad),
                0.995,
                f"{name} cosine similarity",
            )
        context.close()

    def test_bf16_clamped_swiglu_matches_reference(self) -> None:
        """Clamped BF16 forward and backward match the PyTorch formula."""
        swiglu_alpha = 1.5
        swiglu_limit = 5.0
        context, case_a, _, w13_EFD, w2_EDF = self._create_case(
            activation="swiglu_clamped",
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            inference=False,
        )
        x_TD, topk_expert_ids_TK, topk_scores_TK = case_a
        inputs = (x_TD, topk_scores_TK, w13_EFD, w2_EDF)
        for tensor in inputs:
            tensor.requires_grad_()
        reference_inputs = tuple(
            tensor.detach().clone().requires_grad_() for tensor in inputs
        )

        actual_TD = dist_moe.routed_experts(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            w13_EFD,
            w2_EDF,
            context,
        )
        expected_TD = _reference_moe(
            reference_inputs[0],
            topk_expert_ids_TK,
            reference_inputs[1],
            reference_inputs[2],
            reference_inputs[3],
            clamped=True,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
        )
        grad_TD = torch.randn_like(actual_TD)
        actual_TD.backward(grad_TD)
        expected_TD.backward(grad_TD)

        torch.testing.assert_close(actual_TD, expected_TD, rtol=2e-2, atol=2e-2)
        for actual, expected in zip(inputs, reference_inputs, strict=True):
            torch.testing.assert_close(
                actual.grad,
                expected.grad,
                rtol=3e-2,
                atol=3e-2,
            )

    def test_bf16_inference_block_major_weights_match_row_major(self) -> None:
        """Block-major weights give the row-major inference output exactly.

        The same routed tokens run once with row-major W13/W2 and once with the
        weights rewritten by ``relayout_to_block_major`` on a context with
        ``block_major_weights``; outputs must be bitwise equal. Cases: the auto
        config at D = F = 256 and at D = 320, F = 192 (a W2 N tail and three K
        blocks), and every BF16 grouped-GEMM preset.
        """
        context, case_a, _, w13_EFD, w2_EDF = self._create_case()
        x_TD, ids_TK, _ = case_a
        torch.manual_seed(7)
        cases = [
            (case_a, w13_EFD, w2_EDF, context.config),
            *(
                (
                    case_a,
                    w13_EFD,
                    w2_EDF,
                    dataclasses.replace(context.config, bf16_grouped_gemm_preset=p),
                )
                for p in get_args(dist_moe.Bf16GroupedGemmPreset)
            ),
            (
                (
                    torch.randn(len(x_TD), 320, device="cuda").bfloat16(),
                    ids_TK,
                    case_a[2],
                ),
                torch.randn(4, 384, 320, device="cuda", dtype=torch.bfloat16) * 0.05,
                torch.randn(4, 320, 192, device="cuda", dtype=torch.bfloat16) * 0.07,
                dataclasses.replace(
                    context.config, hidden_dim=320, intermediate_dim=192
                ),
            ),
        ]
        for (x, ids, scores), w13, w2, config in cases:
            with self.subTest(
                preset=config.bf16_grouped_gemm_preset, hidden_dim=config.hidden_dim
            ):
                row_context = dist_moe.create_context(
                    group=dist.group.WORLD, config=config
                )
                block_context = dist_moe.create_context(
                    group=dist.group.WORLD,
                    config=dataclasses.replace(config, block_major_weights=True),
                )
                block_w13, block_w2 = w13.clone(), w2.clone()
                dist_moe.relayout_to_block_major(block_w13)
                dist_moe.relayout_to_block_major(block_w2)
                with torch.inference_mode():
                    row_major_TD = dist_moe.routed_experts(
                        x, ids, scores, w13, w2, row_context
                    )
                    block_major_TD = dist_moe.routed_experts(
                        x, ids, scores, block_w13, block_w2, block_context
                    )
                row_context.close()
                block_context.close()
                torch.testing.assert_close(block_major_TD, row_major_TD, rtol=0, atol=0)

    def test_bf16_inference_swap_ab_matches_default(self) -> None:
        """Decode-sized swapped GEMMs give the default tiles' exact output.

        The shared case has 4 rows per expert, so dispatch and combine build
        swapped kernels on the single activation descriptor, with row-major and
        block-major weights; both outputs must equal the call with the gate off.
        """
        import dist_moe.kernels.dist_grouped_gemm as gemm_impl

        context, case_a, _, w13_EFD, w2_EDF = self._create_case()
        block_context = dist_moe.create_context(
            group=dist.group.WORLD,
            config=dataclasses.replace(context.config, block_major_weights=True),
        )
        block_w13, block_w2 = w13_EFD.clone(), w2_EDF.clone()
        dist_moe.relayout_to_block_major(block_w13)
        dist_moe.relayout_to_block_major(block_w2)
        get_kernel = gemm_impl._get_dist_kernel
        kernels = []

        def record_kernel(*args, **kwargs):
            kernels.append(get_kernel(*args, **kwargs))
            return kernels[-1]

        with torch.inference_mode():
            with mock.patch.object(
                api_impl, "_decode_swap_ab_config", lambda *args: None
            ):
                default_TD = dist_moe.routed_experts(*case_a, w13_EFD, w2_EDF, context)
            with mock.patch.object(gemm_impl, "_get_dist_kernel", record_kernel):
                swapped_TD = dist_moe.routed_experts(*case_a, w13_EFD, w2_EDF, context)
                block_TD = dist_moe.routed_experts(
                    *case_a, block_w13, block_w2, block_context
                )
        context.close()
        block_context.close()
        torch.testing.assert_close(swapped_TD, default_TD, rtol=0, atol=0)
        torch.testing.assert_close(block_TD, default_TD, rtol=0, atol=0)
        self.assertEqual(
            {
                (kernel.mode, kernel.block_major_weights)
                for kernel in kernels
                if kernel.SWAP_AB and kernel.SINGLE_A_TENSORMAP
            },
            {
                (mode, block_major)
                for mode in (
                    gemm_impl.DistGroupedGemmKernel.DISPATCH,
                    gemm_impl.DistGroupedGemmKernel.COMBINE,
                )
                for block_major in (False, True)
            },
        )

    def test_bf16_inference_projection_biases_match_reference(self) -> None:
        """BF16 inference adds W13/W2 biases like a BF16 ``projection + bias``.

        Tolerance-based against the PyTorch formula with clamped SwiGLU on 24
        experts, 8 of which get no rows, and one token whose routes are all
        ``-1``. Bias std 0.5 moves the output far beyond the tolerance, so a
        dropped bias fails.
        """
        context, case_a, _, w13_EFD, w2_EDF = self._create_case(
            num_experts=24, activation="swiglu_clamped"
        )
        x_TD, topk_expert_ids_TK, topk_scores_TK = case_a
        topk_expert_ids_TK = topk_expert_ids_TK.clone()
        topk_scores_TK = topk_scores_TK.clone()
        topk_expert_ids_TK[0] = -1
        topk_scores_TK[0] = 0.0
        w13_bias_EF = 0.5 * torch.randn(
            w13_EFD.shape[:2], device="cuda", dtype=torch.bfloat16
        )
        w2_bias_ED = 0.5 * torch.randn(
            w2_EDF.shape[:2], device="cuda", dtype=torch.bfloat16
        )
        options = dist_moe.ExecutionOptions(
            w13_bias_EF=w13_bias_EF, w2_bias_ED=w2_bias_ED
        )

        with torch.inference_mode():
            actual_TD = dist_moe.routed_experts(
                x_TD,
                topk_expert_ids_TK,
                topk_scores_TK,
                w13_EFD,
                w2_EDF,
                context,
                options=options,
            )
        expected_TD = _reference_moe(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            w13_EFD,
            w2_EDF,
            clamped=True,
            w13_bias=w13_bias_EF,
            w2_bias=w2_bias_ED,
        )
        context.close()

        torch.testing.assert_close(actual_TD, expected_TD, rtol=2e-2, atol=2e-2)

    def test_projection_biases_require_bf16_inference_and_weight_shapes(self) -> None:
        """Training contexts, interleaved W13 and wrong-shape biases raise first."""
        interleaved = {"block_major_weights": True, "interleaved_gate_up": True}
        for inference, w13_width_delta, config_fields, message in (
            (False, 0, {}, "only by BF16 inference"),
            (True, 2, {}, "16-byte-aligned"),
            (True, 0, interleaved, "not supported with interleaved_gate_up"),
        ):
            with self.subTest(message=message):
                context, case_a, _, w13_EFD, w2_EDF = self._create_case(
                    inference=inference
                )
                if config_fields:
                    context.close()
                    context = dist_moe.create_context(
                        group=dist.group.WORLD,
                        config=dataclasses.replace(context.config, **config_fields),
                    )
                num_experts, w13_width, _ = w13_EFD.shape
                options = dist_moe.ExecutionOptions(
                    w13_bias_EF=torch.zeros(
                        num_experts,
                        w13_width + w13_width_delta,
                        device="cuda",
                        dtype=torch.bfloat16,
                    ),
                    w2_bias_ED=torch.zeros(
                        w2_EDF.shape[:2], device="cuda", dtype=torch.bfloat16
                    ),
                )
                with (
                    torch.inference_mode(),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    dist_moe.routed_experts(
                        *case_a, w13_EFD, w2_EDF, context, options=options
                    )
                context.close()

    def test_bf16_inference_interleaved_gate_up_matches_row_major(self) -> None:
        """The dispatch-epilogue SwiGLU of interleaved W13 is bitwise exact.

        Row-major weights with the standalone SwiGLU vs block-major weights with
        W13 gate/up rows interleaved (``Config.interleaved_gate_up``). Cases:
        swapped decode tiles (auto config), clamped SwiGLU, a 128-row preset and
        a 2-CTA preset (2-CTA tiles are auto at 2048+ tokens per GPU) with 300
        rows per expert, so CTA rank 1 and MMA atom 1 store rows too.
        """
        for activation, preset, num_tokens in (
            ("swiglu", None, 8),
            ("swiglu_clamped", None, 8),
            ("swiglu", "1cta1mma_bm128_bn128", 8),
            ("swiglu", "2cta2mma_bm512_bn256", 600),
        ):
            with self.subTest(activation=activation, preset=preset):
                context, (x, ids, scores), _, w13, w2 = self._create_case(
                    activation=activation, num_tokens=num_tokens
                )
                config = dataclasses.replace(
                    context.config, bf16_grouped_gemm_preset=preset
                )
                row_context = dist_moe.create_context(
                    group=dist.group.WORLD, config=config
                )
                fused_context = dist_moe.create_context(
                    group=dist.group.WORLD,
                    config=dataclasses.replace(
                        config, block_major_weights=True, interleaved_gate_up=True
                    ),
                )
                fused_w13, fused_w2 = w13.clone(), w2.clone()
                dist_moe.relayout_to_block_major(fused_w13, interleave_gate_up=True)
                dist_moe.relayout_to_block_major(fused_w2)
                with torch.inference_mode():
                    row_major_TD = dist_moe.routed_experts(
                        x, ids, scores, w13, w2, row_context
                    )
                    fused_TD = dist_moe.routed_experts(
                        x, ids, scores, fused_w13, fused_w2, fused_context
                    )
                context.close()
                row_context.close()
                fused_context.close()
                torch.testing.assert_close(fused_TD, row_major_TD, rtol=0, atol=0)

    def test_two_saved_layers_match_reference_without_recompute(self) -> None:
        """Each layer retains its own post-combine state for score gradients."""
        context, case_a, case_b, w13_first, w2_first = self._create_case(
            max_moe_layers_per_activation_slot=2, inference=False
        )
        self.assertGreaterEqual(
            context.memory_plan.activation_slot_bytes,
            context.memory_plan.maximum_useful_activation_slot_bytes,
        )
        x_TD, topk_expert_ids_TK, topk_scores_first_TK = case_a
        _, _, topk_scores_second_TK = case_b
        w13_second_EFD = torch.randn_like(w13_first).mul_(context.hidden_dim**-0.5)
        w2_second_EDF = torch.randn_like(w2_first).mul_(
            context.config.intermediate_dim**-0.5
        )
        tensors = (
            x_TD,
            topk_scores_first_TK,
            topk_scores_second_TK,
            w13_first,
            w2_first,
            w13_second_EFD,
            w2_second_EDF,
        )
        for tensor in tensors:
            tensor.requires_grad_()
        reference_tensors = tuple(
            tensor.detach().clone().requires_grad_() for tensor in tensors
        )

        h_TD = dist_moe.routed_experts(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_first_TK,
            w13_first,
            w2_first,
            context,
        )
        actual_TD = dist_moe.routed_experts(
            h_TD,
            topk_expert_ids_TK,
            topk_scores_second_TK,
            w13_second_EFD,
            w2_second_EDF,
            context,
        )
        expected_h_TD = _reference_moe(
            reference_tensors[0],
            topk_expert_ids_TK,
            reference_tensors[1],
            reference_tensors[3],
            reference_tensors[4],
        )
        expected_TD = _reference_moe(
            expected_h_TD,
            topk_expert_ids_TK,
            reference_tensors[2],
            reference_tensors[5],
            reference_tensors[6],
        )

        grad_TD = torch.randn_like(actual_TD)
        actual_TD.backward(grad_TD)
        expected_TD.backward(grad_TD)
        torch.testing.assert_close(actual_TD, expected_TD, rtol=2e-2, atol=2e-2)
        for actual, expected in zip(tensors, reference_tensors):
            torch.testing.assert_close(
                actual.grad,
                expected.grad,
                rtol=2e-2,
                atol=2e-2,
            )

    def test_eager_flat_weights_return_logical_gradient_shapes(self) -> None:
        """The eager autograd boundary restores flattened parameter shapes."""
        context, case_a, _, w13, w2 = self._create_case(inference=False)
        x, ids, scores = case_a
        flat_w13 = w13.detach().flatten(0, 1).requires_grad_()
        flat_w2 = w2.detach().flatten(0, 1).requires_grad_()

        output = dist_moe.routed_experts(
            x.detach().requires_grad_(),
            ids,
            scores.detach().requires_grad_(),
            flat_w13,
            flat_w2,
            context,
            options=dist_moe.ExecutionOptions(weights_preprocess_fn=torch.clone),
        )
        output.backward(torch.randn_like(output))

        self.assertEqual(flat_w13.grad.shape, flat_w13.shape)
        self.assertEqual(flat_w2.grad.shape, flat_w2.shape)
        context.close()


@unittest.skipUnless(
    torch.cuda.device_count() >= 2 and _is_blackwell(),
    "two Blackwell CUDA devices required",
)
@pytest.mark.gpus_needed_2
@pytest.mark.gb10x
class DistMoeTwoRankTest(unittest.TestCase):
    """Two-rank NCCL numerics for real symmetric-memory peer paths."""

    @property
    def world_size(self) -> int:
        """Return the expert-parallel world size used by this test.

        Returns:
            Two ranks.
        """
        return 2

    @property
    def device_type(self) -> str:
        """Return the distributed test device type.

        Returns:
            CUDA device type.
        """
        return "cuda"

    @with_comms
    def test_dtensor_gradient_owner_resolves_local_storage(self) -> None:
        """Use FP32 DTensor-local storage for a BF16 parameter WGRAD."""
        device = torch.device("cuda", dist.get_rank())
        mesh = init_device_mesh("cuda", (self.world_size,))
        local_parameter_EFD = torch.ones(
            1,
            2,
            3,
            dtype=torch.bfloat16,
            device=device,
        )
        owner = (
            DTensor.from_local(
                local_parameter_EFD,
                device_mesh=mesh,
                placements=(Shard(0),),
                run_check=False,
            )
            .detach()
            .requires_grad_()
        )
        owner.grad_dtype = torch.float32
        local_grad_EFD = torch.zeros_like(
            local_parameter_EFD,
            dtype=torch.float32,
        )
        owner.grad = DTensor.from_local(
            local_grad_EFD,
            device_mesh=mesh,
            placements=(Shard(0),),
            run_check=False,
        )

        destination = _resolve_parameter_grad(
            _weak_parameter_ref(local_parameter_EFD, owner),
            local_parameter_EFD.shape,
            None,
        )

        self.assertIs(destination.parameter, owner)
        self.assertTrue(destination.accumulate)
        self.assertIs(destination.dtype, torch.float32)
        assert destination.output is not None
        self.assertEqual(destination.output.data_ptr(), local_grad_EFD.data_ptr())

    def _run_two_rank_numerics(  # noqa: C901
        self,
        block_scaled: dist_moe.BlockScaledConfig | None,
        *,
        options: dist_moe.ExecutionOptions | None = None,
        prepared_weights: bool = False,
        flattened_weights: bool = False,
        activation: Literal["swiglu", "swiglu_clamped"] = "swiglu",
        swiglu_alpha: float = 1.702,
        swiglu_limit: float = 7.0,
        reference_output_scale: float = 1.0,
        reference_postprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        repeat_execution: bool = False,
        accumulate_execution: bool = False,
        parameter_grad_dtype: torch.dtype = torch.bfloat16,
        max_num_tokens: int | None = None,
        padding_rank: int | None = None,
        wgrad_dtype: torch.dtype | None = torch.float32,
    ) -> None:
        """Run one precision through real peer dispatch and combine paths.

        Args:
            block_scaled: Low-precision policy, or ``None`` for BF16 with VMM.
            options: Optional per-invocation execution controls.
            prepared_weights: Whether to provide caller-owned quantized weights.
            flattened_weights: Whether to exercise two-dimensional parameters.
            activation: SwiGLU variant used by the expert MLP.
            swiglu_alpha: Sigmoid multiplier for clamped SwiGLU.
            swiglu_limit: Preactivation bound for clamped SwiGLU.
            reference_output_scale: Factor applied by an expert-output callback.
            reference_postprocess_fn: Optional reference route-wise transform.
            repeat_execution: Whether to repeat training on the same context.
            accumulate_execution: Whether to accumulate a second identical
                backward into the existing expert gradients.
            parameter_grad_dtype: Dtype declared by both expert parameters.
            max_num_tokens: Optional context capacity above the actual input T.
            padding_rank: Optional rank whose every route is inactive.
            wgrad_dtype: Explicit WGRAD output dtype, or the logical-weight
                dtype when omitted by the public configuration.
        """
        rank = dist.get_rank()
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        num_tokens, hidden_dim, intermediate_dim = 128, 1024, 1024
        num_experts, top_k = 4, 2
        num_local_experts = num_experts // self.world_size
        inference = (
            block_scaled is not None
            and block_scaled.format is dist_moe.BlockScaledFormat.NVFP4
        )
        prepared_weights = prepared_weights or inference

        # Every token selects one expert on each rank, forcing dispatch gather
        # and combine scatter to cross the NVLink peer mapping in both directions.
        global_token = rank * num_tokens + torch.arange(num_tokens, device=device)
        first_expert = global_token % num_experts
        ids = torch.stack(
            (first_expert, (first_expert + num_local_experts) % num_experts),
            dim=1,
        ).to(torch.int64)
        if rank == padding_rank:
            ids.fill_(-1)

        torch.manual_seed(1000 + rank)
        x = torch.randn(
            num_tokens,
            hidden_dim,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        if block_scaled is not None:
            x = (x * 0.1).detach().requires_grad_()
        scores = torch.rand(
            num_tokens,
            top_k,
            device=device,
            dtype=torch.float32,
        )
        scores = (scores / scores.sum(dim=1, keepdim=True)).requires_grad_()
        if rank == padding_rank:
            with torch.no_grad():
                scores.zero_()
        grad_output = torch.randn_like(x)

        # Generate identical global weights on both ranks, then give the fused
        # operation only its rank-local expert shard.
        torch.manual_seed(2027)
        full_w13 = torch.randn(
            num_experts,
            2 * intermediate_dim,
            hidden_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        full_w2 = torch.randn(
            num_experts,
            hidden_dim,
            intermediate_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        full_w13.mul_(hidden_dim**-0.5)
        full_w2.mul_(intermediate_dim**-0.5)
        expert_start = rank * num_local_experts
        w13 = (
            full_w13[expert_start : expert_start + num_local_experts]
            .clone()
            .requires_grad_()
        )
        w2 = (
            full_w2[expert_start : expert_start + num_local_experts]
            .clone()
            .requires_grad_()
        )
        w13.grad_dtype = parameter_grad_dtype
        w2.grad_dtype = parameter_grad_dtype
        if inference:
            x = x.detach()
            scores = scores.detach()
            w13 = w13.detach()
            w2 = w2.detach()

        common_config = {
            "max_num_local_input_tokens": (
                num_tokens if max_num_tokens is None else max_num_tokens
            ),
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "top_k": top_k,
            "num_experts": num_experts,
            "max_moe_layers_per_activation_slot": 1,
            "activation": activation,
            "swiglu_alpha": swiglu_alpha,
            "swiglu_limit": swiglu_limit,
            "wgrad_dtype": None if accumulate_execution else wgrad_dtype,
        }
        if block_scaled is None:
            config = dist_moe.Config(
                **common_config,
                device_scratch_capacity_factor=1.0,
                activation_slot_bytes=None,
            )
        else:
            config = dist_moe.Config(
                **common_config,
                block_scaled=block_scaled,
                inference=inference,
            )
        context = dist_moe.create_context(
            group=dist.group.WORLD,
            config=config,
            device=device,
        )
        try:
            call_w13 = w13.view(-1, hidden_dim) if flattened_weights else w13
            call_w2 = w2.view(-1, intermediate_dim) if flattened_weights else w2
            if prepared_weights:
                assert block_scaled is not None
                call_w13 = dist_moe.prepare_block_scaled_weight(
                    w13, block_scaled, inference=inference
                )
                call_w2 = dist_moe.prepare_block_scaled_weight(
                    w2, block_scaled, inference=inference
                )
            with torch.no_grad() if inference else torch.enable_grad():
                actual = dist_moe.routed_experts(
                    x,
                    ids,
                    scores,
                    call_w13,
                    call_w2,
                    context,
                    options=options,
                )
            if rank == padding_rank:
                self.assertEqual(torch.count_nonzero(actual).item(), 0)

            global_actual = _all_gather_cat(actual.detach())
            global_x = _all_gather_cat(x.detach()).requires_grad_()
            global_ids = _all_gather_cat(ids)
            global_scores = _all_gather_cat(scores.detach()).requires_grad_()
            global_grad_output = _all_gather_cat(grad_output)
            reference_w13 = full_w13.detach().clone().requires_grad_()
            reference_w2 = full_w2.detach().clone().requires_grad_()
            reference = _reference_moe(
                global_x,
                global_ids,
                global_scores,
                reference_w13,
                reference_w2,
                experts_output_postprocess=reference_postprocess_fn,
                clamped=activation == "swiglu_clamped",
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
            )
            reference = reference * reference_output_scale
            if block_scaled is None:
                fwd_tolerance = (2e-2, 2e-2, 5e-3, 0.999)
            else:
                fwd_tolerance = _blockscaled_tolerance(block_scaled.format, "fwd")
            rtol, atol, rel_l2, cosine = fwd_tolerance
            check_elementwise = block_scaled is None or (
                reference_postprocess_fn is None and activation == "swiglu"
            )
            # The dense reference does not fake-quantize MXFP8 intermediates;
            # RMSNorm and clamped SwiGLU redistribute that error, so compare
            # their aggregate quality instead.
            if check_elementwise:
                torch.testing.assert_close(
                    global_actual.float(),
                    reference.float(),
                    rtol=rtol,
                    atol=atol,
                )
            self.assertLess(
                _relative_l2(global_actual, reference),
                rel_l2,
                "output relative L2 error",
            )
            self.assertGreater(
                _cosine_similarity(global_actual, reference),
                cosine,
                "output cosine similarity",
            )

            if inference:
                return

            hook_counts = {"w13": 0, "w2": 0}
            if accumulate_execution:

                def record_accumulation(
                    name: str,
                    parameter: torch.Tensor,
                ) -> None:
                    """Record that autograd observed the final owned gradient."""
                    self.assertIsNotNone(parameter.grad)
                    hook_counts[name] += 1

                w13.register_post_accumulate_grad_hook(
                    partial(record_accumulation, "w13")
                )
                w2.register_post_accumulate_grad_hook(
                    partial(record_accumulation, "w2")
                )

            actual.backward(grad_output)
            reference.backward(global_grad_output)
            reference_grads = (
                global_x.grad,
                global_scores.grad,
                reference_w13.grad,
                reference_w2.grad,
            )
            comparisons = zip(
                ("x", "topk_scores", "w13", "w2"),
                (
                    _all_gather_cat(x.grad),
                    _all_gather_cat(scores.grad),
                    _all_gather_cat(w13.grad),
                    _all_gather_cat(w2.grad),
                ),
                reference_grads,
            )
            for name, actual_grad, expected_grad in comparisons:
                self.assertIsNotNone(actual_grad, f"missing {name} gradient")
                self.assertIsNotNone(
                    expected_grad, f"missing reference {name} gradient"
                )
                if block_scaled is None:
                    grad_tolerance = (
                        3e-2,
                        1.0 if name in ("w13", "w2") else 3e-2,
                        1.5e-2 if name in ("w13", "w2") else 1e-2,
                        0.995,
                    )
                else:
                    direction = "wgrad" if name in ("w13", "w2") else "grad"
                    grad_tolerance = _blockscaled_tolerance(
                        block_scaled.format, direction
                    )
                rtol, atol, rel_l2, cosine = grad_tolerance
                if check_elementwise:
                    torch.testing.assert_close(
                        actual_grad.float(),
                        expected_grad.float(),
                        rtol=rtol,
                        atol=atol,
                        msg=lambda message, name=name: f"{name}: {message}",
                    )
                self.assertLess(
                    _relative_l2(actual_grad, expected_grad),
                    rel_l2,
                    f"{name} relative L2 error",
                )
                self.assertGreater(
                    _cosine_similarity(actual_grad, expected_grad),
                    cosine,
                    f"{name} cosine similarity",
                )
            if accumulate_execution:
                self.assertEqual(hook_counts, {"w13": 1, "w2": 1})
                self.assertIsNotNone(w13.grad)
                self.assertIsNotNone(w2.grad)
                self.assertIs(w13.grad.dtype, parameter_grad_dtype)
                self.assertIs(w2.grad.dtype, parameter_grad_dtype)
                self.assertFalse(hasattr(w13, "main_grad"))
                self.assertFalse(hasattr(w2, "main_grad"))
                first_w13 = w13.grad.detach().clone()
                first_w2 = w2.grad.detach().clone()
                w13_pointer = w13.grad.data_ptr()
                w2_pointer = w2.grad.data_ptr()
                repeated = dist_moe.routed_experts(
                    x,
                    ids,
                    scores,
                    call_w13,
                    call_w2,
                    context,
                    options=options,
                )
                recorder = _BackwardOpRecorder()
                with recorder:
                    repeated.backward(grad_output)
                self.assertEqual(hook_counts, {"w13": 2, "w2": 2})
                self.assertEqual(len(recorder.calls), 1)
                for expected_pointer, accumulator in zip(
                    (w13_pointer, w2_pointer),
                    recorder.calls[0],
                    strict=True,
                ):
                    self.assertEqual(accumulator.data_ptr(), expected_pointer)
                self.assertEqual(w13.grad.data_ptr(), w13_pointer)
                self.assertEqual(w2.grad.data_ptr(), w2_pointer)
                torch.testing.assert_close(
                    w13.grad,
                    first_w13 * 2,
                    rtol=5e-2,
                    atol=2e-2,
                )
                torch.testing.assert_close(
                    w2.grad,
                    first_w2 * 2,
                    rtol=5e-2,
                    atol=2e-2,
                )
            if repeat_execution:
                first_gradients = tuple(
                    tensor.grad.detach().clone() for tensor in (x, scores, w13, w2)
                )
                for tensor in (x, scores, w13, w2):
                    tensor.grad = None
                repeated = dist_moe.routed_experts(
                    x,
                    ids,
                    scores,
                    w13,
                    w2,
                    context,
                    options=options,
                )
                repeated.backward(grad_output)
                torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
                for tensor, expected_gradient in zip(
                    (x, scores, w13, w2),
                    first_gradients,
                    strict=True,
                ):
                    torch.testing.assert_close(
                        tensor.grad,
                        expected_gradient,
                        rtol=0,
                        atol=0,
                    )
        finally:
            context.close()

    @with_comms
    def test_forward_and_backward_exercise_peer_paths(self) -> None:
        """Match two-rank BF16 output and gradients to the routed reference.

        A bitwise repeat proves context reuse across real peer dispatch and
        combine without involving VMM.
        """
        self._run_two_rank_numerics(None, repeat_execution=True)

    @with_comms
    def test_bounded_token_shape_exercises_peer_paths(self) -> None:
        """Run actual T below one BF16/MXFP8 context capacity over NCCL."""
        for block_scaled in (None, dist_moe.BlockScaledConfig()):
            with self.subTest(block_scaled=block_scaled is not None):
                self._run_two_rank_numerics(
                    block_scaled,
                    max_num_tokens=256,
                )

    @with_comms
    def test_bf16_inference_single_a_tensormap_matches_per_group(self) -> None:
        """One A descriptor per CTA gives the per-group descriptors' exact output.

        Two ranks, BF16 inference, the auto (one-CTA) config. Routing is uneven:
        expert 0 gets more than one 64-row tile, most experts get no rows, rank 1
        sends nothing in one case, and NaN tokens sit next to valid ones, so tile
        tails load other experts' rows, including NaN rows. Valid outputs must be
        bitwise equal to the same call with per-group descriptors, and both GEMMs
        must have used the single descriptor. Forced swapped tiles (2 to 180 rows
        per expert) must match too: the only swapped combine over real peers. No
        other test runs BF16 inference over real peers.
        """
        import dist_moe.kernels.dist_grouped_gemm as gemm_impl

        rank = dist.get_rank()
        device = torch.device("cuda", rank)
        num_experts, top_k, hidden_dim, intermediate_dim = 16, 4, 256, 256
        config = dist_moe.Config(
            max_num_local_input_tokens=128,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            top_k=top_k,
            max_moe_layers_per_activation_slot=1,
            device_scratch_capacity_factor=float(self.world_size),
            inference=True,
        )
        context = dist_moe.create_context(group=dist.group.WORLD, config=config)
        generator = torch.Generator(device=device).manual_seed(11)
        num_local_experts = num_experts // self.world_size
        w13 = (
            torch.randn(
                num_local_experts,
                2 * intermediate_dim,
                hidden_dim,
                device=device,
                generator=generator,
            )
            .mul_(hidden_dim**-0.5)
            .bfloat16()
        )
        w2 = (
            torch.randn(
                num_local_experts,
                hidden_dim,
                intermediate_dim,
                device=device,
                generator=generator,
            )
            .mul_(intermediate_dim**-0.5)
            .bfloat16()
        )
        get_kernel = gemm_impl._get_dist_kernel
        single_flags = []

        def record_kernel(*args, **kwargs):
            single_flags.append((kwargs["mode"], kwargs["single_a_tensormap"]))
            return get_kernel(*args, **kwargs)

        launch = gemm_impl._fused_dist_grouped_gemm_impl
        swap_config = "1cta1mma_bm128_bn32_bk128_swap_ab"

        def per_group_launch(*args, **kwargs):
            return launch(*args, **{**kwargs, "single_a_tensormap": False})

        # (tokens per rank, experts every token picks, rank-1 sends nothing)
        for num_tokens, picks, rank1_empty in (
            (90, (0, 3, 9, 14), False),
            (1, (0, 7, 8, 15), False),
            (33, (0, 1, 2, 3), True),
        ):
            x = torch.randn(
                num_tokens, hidden_dim, device=device, generator=generator
            ).bfloat16()
            x[1::5] = float("nan")
            ids = torch.tensor(picks, device=device).expand(num_tokens, -1).clone()
            scores = torch.rand(
                num_tokens, top_k, device=device, generator=generator
            ).softmax(dim=-1)
            if rank1_empty and rank == 1:
                ids.fill_(-1)
                scores.zero_()
            single_flags.clear()
            # Per-group descriptors exist for non-swapped tiles only.
            with (
                torch.inference_mode(),
                mock.patch.object(
                    api_impl, "_decode_swap_ab_config", lambda *args: None
                ),
            ):
                with mock.patch.object(gemm_impl, "_get_dist_kernel", record_kernel):
                    single = dist_moe.routed_experts(x, ids, scores, w13, w2, context)
                with mock.patch.object(
                    gemm_impl, "_fused_dist_grouped_gemm_impl", per_group_launch
                ):
                    per_group = dist_moe.routed_experts(
                        x, ids, scores, w13, w2, context
                    )
            # Forced swapped tiles (2 to 180 rows per expert, up to six 32-token
            # tiles) must give the single descriptor's output too.
            with (
                torch.inference_mode(),
                mock.patch.object(
                    api_impl, "_decode_swap_ab_config", lambda *args: swap_config
                ),
            ):
                swapped = dist_moe.routed_experts(x, ids, scores, w13, w2, context)
            self.assertTrue(single_flags and all(flag for _, flag in single_flags))
            valid = torch.isfinite(x).all(dim=1)
            self.assertTrue(torch.isfinite(single[valid]).all())
            torch.testing.assert_close(single[valid], per_group[valid], rtol=0, atol=0)
            torch.testing.assert_close(swapped[valid], single[valid], rtol=0, atol=0)
        context.close()

    @with_comms
    def test_fully_padded_rank_exercises_peer_paths(self) -> None:
        """A rank with only `-1` routes contributes no BF16/MXFP8 work."""
        for block_scaled in (None, dist_moe.BlockScaledConfig()):
            with self.subTest(block_scaled=block_scaled is not None):
                self._run_two_rank_numerics(
                    block_scaled,
                    padding_rank=1,
                )

    @with_comms
    def test_bf16_clip_statistics_exercise_peer_paths(self) -> None:
        """Clip counters include route rows read from both symmetric peers."""
        stats_out_3 = torch.zeros(
            3,
            dtype=torch.float32,
            device=torch.device("cuda", dist.get_rank()),
        )
        self._run_two_rank_numerics(
            None,
            options=dist_moe.ExecutionOptions(
                swiglu_clip_stats_out_3=stats_out_3,
                swiglu_clip_limit=0.0,
            ),
        )
        expected_valid_elements = 128 * 2 * 1024
        self.assertEqual(stats_out_3[2].item(), expected_valid_elements)

    @with_comms
    def test_flat_weights_exercise_peer_paths(self) -> None:
        """Match flattened-weight output and gradients over real peer paths.

        This protects logical gradient-shape restoration for the accepted 2D
        expert-weight representation.
        """
        self._run_two_rank_numerics(
            None,
            flattened_weights=True,
        )

    @with_comms
    def test_bf16_accumulates_parameter_grad_over_peer_paths(self) -> None:
        """Accumulate BF16 WGRAD into stable standard gradient storage."""
        for grad_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(grad_dtype=grad_dtype):
                self._run_two_rank_numerics(
                    None,
                    options=dist_moe.ExecutionOptions(inplace_wgrad_accum=True),
                    flattened_weights=True,
                    accumulate_execution=True,
                    parameter_grad_dtype=grad_dtype,
                )

    @with_comms
    def test_mxfp8_e4m3_forward_and_backward_exercise_peer_paths(self) -> None:
        """Bound staged MXFP8 output and gradient error over two real ranks.

        This covers fused peer dispatch and combine under quantized execution.
        """
        self._run_two_rank_numerics(
            dist_moe.BlockScaledConfig(format=dist_moe.BlockScaledFormat.MXFP8_E4M3)
        )

    @with_comms
    def test_post_expert_rmsnorm_exercises_peer_paths(self) -> None:
        """Bound fused-RMSNorm output and gradients across training backends.

        BF16, staged/Mega MXFP8, and direct WGRAD destinations must preserve the
        same two-rank post-expert semantics.
        """
        eps = 1e-6

        def reference_rmsnorm(h3_KD: torch.Tensor) -> torch.Tensor:
            """Apply the unfused FP32 RMSNorm reference to each route."""
            return F.rms_norm(h3_KD.float(), (h3_KD.shape[-1],), eps=eps)

        postprocess = dist_moe.RMSNormPostprocess(
            eps=eps,
            norm_output_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            require_bitwise=False,
            recompute_rstd=True,
        )
        cases = (
            (None, False),
            (None, True),
            (
                dist_moe.BlockScaledConfig(
                    format=dist_moe.BlockScaledFormat.MXFP8_E4M3
                ),
                False,
            ),
            (
                dist_moe.BlockScaledConfig(
                    format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
                    pipeline="mega",
                ),
                False,
            ),
        )
        for block_scaled, inplace_wgrad_accum in cases:
            with self.subTest(
                block_scaled=block_scaled,
                inplace_wgrad_accum=inplace_wgrad_accum,
            ):
                self._run_two_rank_numerics(
                    block_scaled,
                    options=dist_moe.ExecutionOptions(
                        experts_output_postprocess=postprocess,
                        inplace_wgrad_accum=inplace_wgrad_accum,
                    ),
                    reference_postprocess_fn=reference_rmsnorm,
                    accumulate_execution=inplace_wgrad_accum,
                )

    @with_comms
    def test_mxfp8_mega_forward_and_backward_exercise_peer_paths(self) -> None:
        """Bound two-rank Mega MXFP8 output and gradients against reference.

        This independently covers the fused Mega path rather than relying on
        staged-pipeline evidence.
        """
        self._run_two_rank_numerics(
            dist_moe.BlockScaledConfig(
                format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
                pipeline="mega",
            )
        )

    @with_comms
    def test_mxfp8_clamped_swiglu_matches_reference(self) -> None:
        """Match clamped staged and mega MXFP8 against PyTorch math."""
        for pipeline in ("staged", "mega"):
            with self.subTest(pipeline=pipeline):
                self._run_two_rank_numerics(
                    dist_moe.BlockScaledConfig(
                        format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
                        pipeline=pipeline,
                    ),
                    activation="swiglu_clamped",
                    swiglu_alpha=1.5,
                    swiglu_limit=5.0,
                )

    @with_comms
    def test_mxfp8_mega_rebuilds_callback_weight_quants_in_backward(self) -> None:
        """Regenerate callback-owned MXFP8 operands instead of saving them."""

        def preprocess_weight(weight: torch.Tensor) -> torch.Tensor:
            """Return the callback-owned native compute weight unchanged."""
            return weight

        with mock.patch.object(
            blockscaled_impl,
            "_prepare_blockscaled_weight_impl",
            wraps=blockscaled_impl._prepare_blockscaled_weight_impl,
        ) as prepare_weight:
            self._run_two_rank_numerics(
                dist_moe.BlockScaledConfig(
                    format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
                    pipeline="mega",
                ),
                options=dist_moe.ExecutionOptions(
                    weights_preprocess_fn=preprocess_weight
                ),
            )

        self.assertEqual(
            prepare_weight.call_count,
            4,
            "W13 and W2 must each quantize once in forward and once in backward",
        )

    def _run_prepared_mxfp8_parameter_grad_accumulation(
        self,
        pipeline: str,
        parameter_grad_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Accumulate prepared MXFP8 WGRAD through one async pipeline."""
        self._run_two_rank_numerics(
            dist_moe.BlockScaledConfig(
                format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
                pipeline=pipeline,
            ),
            options=dist_moe.ExecutionOptions(inplace_wgrad_accum=True),
            prepared_weights=True,
            accumulate_execution=True,
            parameter_grad_dtype=parameter_grad_dtype,
        )

    @with_comms
    def test_prepared_mxfp8_staged_accumulates_parameter_grad(self) -> None:
        """Accumulate prepared MXFP8 WGRAD through the staged pipeline."""
        for grad_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(grad_dtype=grad_dtype):
                self._run_prepared_mxfp8_parameter_grad_accumulation(
                    "staged",
                    grad_dtype,
                )

    @with_comms
    def test_prepared_mxfp8_mega_accumulates_parameter_grad(self) -> None:
        """Accumulate prepared MXFP8 WGRAD through the mega pipeline."""
        self._run_prepared_mxfp8_parameter_grad_accumulation("mega")

    @with_comms
    def test_prepared_mxfp8_forward_and_backward_exercise_peer_paths(self) -> None:
        """Bound prepared MXFP8 output and gradients against reference.

        This proves the storage-free logical-weight autograd bridge preserves
        the same two-rank result as dynamically prepared execution.
        """
        for pipeline in ("staged", "mega"):
            with self.subTest(pipeline=pipeline):
                self._run_two_rank_numerics(
                    dist_moe.BlockScaledConfig(
                        format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
                        pipeline=pipeline,
                    ),
                    prepared_weights=True,
                    wgrad_dtype=None,
                )

    @with_comms
    def test_expert_and_wgrad_postprocess_exercise_peer_paths(self) -> None:
        """Differentiate an expert hook and invoke each WGRAD hook promptly."""
        seen_wgrads: list[str] = []
        preprocess_calls: list[torch.Size] = []

        def preprocess_weight(weight: torch.Tensor) -> torch.Tensor:
            """Record rematerialization and preserve the compute weight."""
            preprocess_calls.append(weight.shape)
            return weight

        def scale_expert_output(output: torch.Tensor) -> torch.Tensor:
            """Scale each expert-route result before score reduction."""
            return output * 0.5

        def record_wgrad(name: str, gradient: torch.Tensor) -> torch.Tensor:
            """Record and preserve each generated expert-weight gradient."""
            seen_wgrads.append(name)
            return gradient

        self._run_two_rank_numerics(
            None,
            options=dist_moe.ExecutionOptions(
                weights_preprocess_fn=preprocess_weight,
                experts_output_postprocess=scale_expert_output,
                wgrad_postprocess_fn=record_wgrad,
            ),
            reference_output_scale=0.5,
        )
        self.assertCountEqual(seen_wgrads, ["w2", "w13"])
        self.assertEqual(len(preprocess_calls), 4)

    @with_comms
    def test_bf16_inference_pdl_graph_matches_non_pdl(self) -> None:
        """PDL-launched BF16 inference replays bitwise equal to launches without PDL.

        Two ranks capture the same four-layer CUDA graph with and without PDL,
        with non-PDL fill, copy and elementwise nodes between layers, then replay
        both on new routing (inactive routes, one fully padded rank). Every
        replay must match exactly. A kernel launched with PDL but without its
        wait fails here; eager calls usually do not expose it.
        """
        rank = dist.get_rank()
        device = torch.device("cuda", rank)
        num_experts, top_k, hidden_dim, intermediate_dim, num_tokens = (
            8,
            4,
            1024,
            512,
            64,
        )
        num_local_experts = num_experts // self.world_size
        config = dist_moe.Config(
            max_num_local_input_tokens=num_tokens,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            top_k=top_k,
            max_moe_layers_per_activation_slot=1,
            device_scratch_capacity_factor=float(self.world_size),
            inference=True,
        )
        context = dist_moe.create_context(group=dist.group.WORLD, config=config)
        generator = torch.Generator(device=device).manual_seed(11)
        w13 = (
            torch.randn(
                num_local_experts,
                2 * intermediate_dim,
                hidden_dim,
                device=device,
                generator=generator,
            )
            .mul_(hidden_dim**-0.5)
            .bfloat16()
        )
        w2 = (
            torch.randn(
                num_local_experts,
                hidden_dim,
                intermediate_dim,
                device=device,
                generator=generator,
            )
            .mul_(intermediate_dim**-0.5)
            .bfloat16()
        )

        def capture(pdl: bool):
            x = torch.zeros(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
            ids = torch.full((num_tokens, top_k), -1, device=device, dtype=torch.int64)
            scores = torch.zeros(num_tokens, top_k, device=device)
            launches = (
                pdl_impl.pdl_launches
                if pdl
                else lambda enabled: contextlib.nullcontext()
            )
            with mock.patch.object(pdl_impl, "pdl_launches", launches), torch.no_grad():
                dist_moe.routed_experts(x, ids, scores, w13, w2, context)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    h = x
                    for _ in range(4):
                        scratch = torch.empty_like(h)
                        scratch.fill_(1.0)
                        h = dist_moe.routed_experts(h, ids, scores, w13, w2, context)
                        h = (h.clone() + x) * 0.5
                        scratch.zero_()
            return graph, (x, ids, scores), h

        try:
            pdl_graph, pdl_inputs, pdl_out = capture(pdl=True)
            plain_graph, plain_inputs, plain_out = capture(pdl=False)
            mismatched_steps = 0
            for step in range(6):
                generator.manual_seed(100 * step + rank)
                x = torch.randn(
                    num_tokens, hidden_dim, device=device, generator=generator
                ).bfloat16()
                ids = torch.rand(
                    num_tokens, num_experts, device=device, generator=generator
                ).argsort(dim=1)[:, :top_k]
                scores = torch.rand(
                    num_tokens, top_k, device=device, generator=generator
                )
                inactive = (
                    torch.rand(num_tokens, top_k, device=device, generator=generator)
                    < 0.3
                )
                if rank == 1 and step % 3 == 0:
                    inactive.fill_(True)
                ids = ids.masked_fill(inactive, -1)
                scores = scores.masked_fill(inactive, 0.0)
                for static in (pdl_inputs, plain_inputs):
                    for target, value in zip(static, (x, ids, scores)):
                        target.copy_(value)
                plain_graph.replay()
                pdl_graph.replay()
                torch.cuda.synchronize()
                dist.barrier()
                # Both ranks replay every step before asserting, so a mismatch
                # fails instead of leaving the peer waiting in its next replay.
                mismatched_steps += not torch.equal(pdl_out, plain_out)
            self.assertEqual(mismatched_steps, 0)
        finally:
            context.close()

    @with_comms
    def test_bf16_decode_routing_matches_general_routing(self) -> None:
        """Chunked decode routing gives the general routing path's exact output.

        BF16 inference on 384 experts (192 per rank, not a power of two) at 8,
        2048 and 8192 route slots per rank, i.e. one, two and eight decode
        routing CTAs per source rank. Each output must be bitwise equal to the
        same call with decode routing disabled, and the decode path must have
        run; no other test reaches these expert counts or more than 1024 slots.
        """
        import dist_moe._routing_metadata as routing_impl

        rank = dist.get_rank()
        device = torch.device("cuda", rank)
        num_experts, top_k, hidden_dim, intermediate_dim = 384, 8, 256, 256
        num_local_experts = num_experts // self.world_size
        config = dist_moe.Config(
            max_num_local_input_tokens=1024,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            top_k=top_k,
            max_moe_layers_per_activation_slot=1,
            device_scratch_capacity_factor=float(self.world_size),
            inference=True,
        )
        context = dist_moe.create_context(group=dist.group.WORLD, config=config)
        generator = torch.Generator(device=device).manual_seed(7)
        w13 = (
            torch.randn(
                num_local_experts,
                2 * intermediate_dim,
                hidden_dim,
                device=device,
                generator=generator,
            )
            .mul_(hidden_dim**-0.5)
            .bfloat16()
        )
        w2 = (
            torch.randn(
                num_local_experts,
                hidden_dim,
                intermediate_dim,
                device=device,
                generator=generator,
            )
            .mul_(intermediate_dim**-0.5)
            .bfloat16()
        )
        decode_routing = mock.Mock(wraps=routing_impl._decode_dispatch_routing)
        try:
            for num_tokens in (1, 256, 1024):
                generator.manual_seed(100 * num_tokens + rank)
                x = torch.randn(
                    num_tokens, hidden_dim, device=device, generator=generator
                ).bfloat16()
                ids = (
                    torch.rand(
                        num_tokens, num_experts, device=device, generator=generator
                    )
                    .topk(top_k, dim=1)
                    .indices
                )
                scores = torch.rand(
                    num_tokens, top_k, device=device, generator=generator
                ).softmax(dim=-1)
                with torch.inference_mode():
                    decode_routing.reset_mock()
                    with mock.patch.object(
                        routing_impl, "_decode_dispatch_routing", decode_routing
                    ):
                        decoded = dist_moe.routed_experts(
                            x, ids, scores, w13, w2, context
                        )
                    self.assertEqual(decode_routing.call_count, 1)
                    with mock.patch.object(
                        routing_impl, "_can_use_decode_routing", return_value=False
                    ):
                        general = dist_moe.routed_experts(
                            x, ids, scores, w13, w2, context
                        )
                torch.testing.assert_close(decoded, general, rtol=0, atol=0)
        finally:
            context.close()

    @with_comms
    def test_nvfp4_inference_exercises_peer_paths(self) -> None:
        """Bound prepared NVFP4 inference against the routed reference.

        The test supplies real two-rank dispatch and combine evidence for the
        inference-only format.
        """
        self._run_two_rank_numerics(
            dist_moe.BlockScaledConfig(format=dist_moe.BlockScaledFormat.NVFP4)
        )
