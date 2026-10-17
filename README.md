# Dist-MoE

Dist-MoE provides speed-of-light distributed Mixture-of-Experts kernels for
NVIDIA SM100 and newer GPUs. Its fused communication-compute pipelines combine
token dispatch, grouped expert GEMMs, and token combine for high-throughput
training and inference.

The `dist_moe` package consumes caller-provided top-k expert IDs and scores,
and owns the communication buffers, activation planning, scratch memory, and
graph-stable PyTorch operations needed to execute routed experts.

## Contents

- [Supported modes](#supported-modes)
- [Install](#install)
- [Run the BF16 quickstart](#run-the-bf16-quickstart)
- [Understand the execution model](#understand-the-execution-model)
- [Choose a precision and pipeline](#choose-a-precision-and-pipeline)
- [Plan memory](#plan-memory)
- [Manage context lifetime](#manage-context-lifetime)
- [Use graph and checkpoint features](#use-graph-and-checkpoint-features)
- [Use advanced features](#use-advanced-features)
- [Public API](#public-api)
- [Read the detailed guides](#read-the-detailed-guides)
- [Acknowledgements](#acknowledgements)

<a id="supported-modes"></a>
## Supported modes

| Precision | Pipeline | Training | Inference |
| --- | --- | :---: | :---: |
| BF16 | Standard | Yes | Yes |
| MXFP8 E4M3 | Staged or Mega | Yes | Yes |
| NVFP4 | Staged or Mega | No | Yes |

All supported modes work with non-strict `make_fx`, CUDA graphs, and optional
VMM host-backed scratch. Training modes include forward and backward.

The package also supports prepared MXFP8 and NVFP4 weights, device-selected
activation slots, plain and clamped SwiGLU, fused post-expert RMSNorm, clip
statistics, and optional in-place accumulation into standard
`parameter.grad` buffers.

<details>
<summary>Not supported in this release</summary>

Full `torch.compile`, strict export, TLX or hierarchical execution,
synchronous block-scaled backends, MXFP8 E5M2, MXFP4, A8W4, NVFP4 training,
expert borrowing, activation rings, and communication quantization.

</details>

<a id="install"></a>
## Install

Install from a source checkout into an environment that already provides a
supported NVIDIA driver and CUDA toolkit. With pip:

```bash
python -m pip install .
```

Or with uv:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install .
```

The initial release is validated with this runtime:

| Layer | Validated release stack |
| --- | --- |
| Platform | Linux ARM64; NVIDIA Blackwell (SM100 class) |
| Framework | Python 3.12; PyTorch 2.14.0+cu130 |
| GPU runtime | CUDA toolkit 13.0.3.0; NVRTC 13.0.88; CUDA Python 13.4.1; NCCL 2.30.7 |
| Kernel toolchain | Triton 3.8 or 3.9; NVIDIA CUTLASS DSL 4.6.1 |

The PyTorch wheel runtime and the toolkit used for JIT compilation are separate
compatibility requirements. Importing `dist_moe` does not initialize CUDA or a
process group, but execution has no CPU fallback.

### Verify the installation

The complete runtime test file requires at least two supported Blackwell GPUs.
For an editable pip installation:

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest -q tests/test_dist_moe.py
```

The equivalent uv workflow is:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
uv pip install pytest
python -m pytest -q tests/test_dist_moe.py
```

This test file covers configuration, kernels, autograd, non-strict `make_fx`,
CUDA graphs, and real two-rank execution. Real VMM tests use a separate isolated
test target.

This project leverages third-party code made available under its own licenses.
See [NOTICE](NOTICE) and [LICENSES](LICENSES).

<a id="run-the-bf16-quickstart"></a>
## BF16 Quickstart

This quickstart shows the complete process lifecycle. The checked-in
[BF16 training example](examples/bf16_training.py) extends the same flow with
advanced BF16 options. Run one process per expert-parallel GPU:

These examples run on one host and use the loopback address so rendezvous does
not depend on whether the host name is reachable from child processes.

```bash
torchrun --standalone --local-addr=127.0.0.1 --nproc-per-node=2 examples/bf16_training.py
```

```python
import os

import torch
import torch.distributed as dist

import dist_moe


device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group("nccl")
ep_group = dist.group.WORLD

T, D, F, K, E_local = 16, 256, 256, 2, 4
rank = dist.get_rank(ep_group)
ep_size = dist.get_world_size(ep_group)
E = E_local * ep_size

config = dist_moe.Config(
    max_num_local_input_tokens=T,
    hidden_dim=D,
    intermediate_dim=F,
    top_k=K,
    num_experts=E,
    max_moe_layers_per_activation_slot=1,
    device_scratch_capacity_factor=1.0,
    num_activation_slots=1,
)
print(dist_moe.plan_memory(config, ep_size=ep_size, device=device).explain())

context = dist_moe.create_context(
    group=ep_group,
    config=config,
    device=device,
)

try:
    x_TD = torch.randn(
        T,
        D,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    global_token_ids_T = rank * T + torch.arange(T, device=device)
    topk_expert_ids_TK = (
        global_token_ids_T[:, None] * K
        + torch.arange(K, device=device)[None, :]
    ) % E
    topk_scores_TK = torch.full(
        (T, K),
        1.0 / K,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    w13_E2FD = torch.randn(
        E_local,
        2,
        F,
        D,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w2_EDF = torch.randn(
        E_local,
        D,
        F,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )

    output_TD = dist_moe.routed_experts(
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        w13_E2FD,
        w2_EDF,
        context,
    )
    output_TD.float().sum().backward()
    assert output_TD.shape == x_TD.shape
    assert x_TD.grad is not None
finally:
    context.close()
    dist.destroy_process_group()
```

Here `T` is the number of local input tokens, `D` is the model hidden width,
`F` is the per-expert intermediate width, `K` is top-k, and `E_local` is the
number of experts owned by this rank. The global expert count is
`E_local * ep_size`.

The W13 weight may have shape `[E_local, 2, F, D]` or `[E_local, 2F, D]`.
The flattened form `[E_local * 2F, D]` is also accepted
when W2 is `[E_local * D, F]`. Inputs and weights must be contiguous CUDA BF16
tensors. Expert IDs must be contiguous integers in `[-1, num_experts)`, where
`-1` disables a route and requires score zero. Scores must be contiguous BF16
or FP32 tensors. Nonnegative expert IDs in each token's top-k row must be
distinct; repeated `-1` padding is allowed. Uniqueness is a caller requirement,
not part of the optional range-only debug check.

<a id="understand-the-execution-model"></a>
## Understand the execution model

One call has four inputs owned by the caller and one reusable context owned by
Dist-MoE:

```text
caller routing
  x_TD + expert IDs + scores
        |
        v
publish rows -> route on GPU -> W13 -> SwiGLU -> W2 -> combine
        |                                           |
        +------ context-owned communication --------+
                    and temporary memory
```

The context owns:

- **communication buffers**: symmetric device memory used to exchange routed
  rows and gradients with expert-parallel peers;
- **activation buffer**: one context-owned virtual address range containing
  device-backed activation slots and scratch storage;
- **activation slots**: logical regions that retain forward state until its
  matching backward;
- **device scratch**: temporary memory reused by one active Dist-MoE call; and
- optional **host-backed scratch**: VMM overflow for routing beyond the device
  scratch capacity.

Forward expert matrix multiplication is called **FPROP**. Backward computes
input gradients with **DGRAD** and weight gradients with **WGRAD**. The
detailed BF16 and MXFP8 guides show every kernel and saved value.

The context is collective to create, sequentially reusable, and not reentrant.
All expert-parallel ranks must create matching contexts in the same order.

`Config.max_num_local_input_tokens` is the context's allocation capacity. Each
invocation may pass any physical row count `T` in
`[1, max_num_local_input_tokens]`, and different invocations may use different
counts. Every expert-parallel rank must use the same `T` for one invocation;
unequal counts trigger a device-side trap before routing metadata is generated.
That failure terminates the distributed iteration and invalidates the CUDA
execution context; it is not a recoverable Python input error.

Padding remains part of the physical `T` rows. Set an inactive route's expert
ID to `-1` and its score to zero. Dist-MoE excludes that route from routing,
communication, scratch demand, expert computation, output, and gradients. A
row whose `K` IDs are all `-1` therefore produces an exact zero row while the
returned tensor remains `[T, D]`; mixed valid and inactive routes are allowed.
Values outside `[-1, num_experts)` violate the caller contract. For integration
debugging, `ExecutionOptions(validate_expert_ids=True)` checks that range in
fused training publication or before native inference publication, and
device-traps on failure.

<a id="choose-a-precision-and-pipeline"></a>
## Choose a precision and pipeline

Start with BF16 unless the model and hardware require block-scaled compute.

### BF16

Leave `block_scaled=None`, which is the default. Activations and weights enter
the expert GEMMs in BF16; tensor-core products accumulate in FP32. See
[BF16 execution](docs/bf16_execution.md).

### MXFP8

MXFP8 E4M3 supports training and inference. Activations are quantized after
routing, and weights may be prepared once and reused. Run the complete
[training](examples/mxfp8_training.py) or
[inference](examples/block_scaled_inference.py) program with:

The block-scaled paths use the concatenated W13 shape `[E_local, 2F, D]`, named
`w13_EFD`. The BF16 quickstart keeps the gate/up axis explicit as
`w13_E2FD`, with shape `[E_local, 2, F, D]`.

```bash
torchrun --standalone --local-addr=127.0.0.1 --nproc-per-node=2 examples/mxfp8_training.py
torchrun --standalone --local-addr=127.0.0.1 --nproc-per-node=2 examples/block_scaled_inference.py
```

```python
policy = dist_moe.BlockScaledConfig(
    format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
    pipeline="staged",
)
config = dist_moe.Config(
    max_num_local_input_tokens=T,
    hidden_dim=D,
    intermediate_dim=F,
    top_k=K,
    num_experts=E,
    max_moe_layers_per_activation_slot=1,
    block_scaled=policy,
)
```

The **staged** pipeline uses separate distributed W13 and SwiGLU/W2 kernels.
The **Mega** pipeline fuses both projections into a chunked pipeline. They use
the same public tensor and memory contracts. See
[MXFP8 execution](docs/mxfp8_execution.md).

### NVFP4

NVFP4 is inference-only. It requires `Config(inference=True)`, zero activation
slots, and prepared weights. The complete
[NVFP4 inference example](examples/block_scaled_inference.py) runs with:

```bash
torchrun --standalone --local-addr=127.0.0.1 --nproc-per-node=2 examples/block_scaled_inference.py
```

```python
policy = dist_moe.BlockScaledConfig(
    format=dist_moe.BlockScaledFormat.NVFP4,
    pipeline="staged",
)
config = dist_moe.Config(
    max_num_local_input_tokens=T,
    hidden_dim=D,
    intermediate_dim=F,
    top_k=K,
    num_experts=E,
    max_moe_layers_per_activation_slot=1,
    num_activation_slots=0,
    block_scaled=policy,
    inference=True,
)
```

See [NVFP4 inference](docs/nvfp4_execution.md).

<a id="plan-memory"></a>
## Plan memory

Plan before creating the context. The complete
[memory-planning example](examples/memory_planning.py) runs without a process
group:

```bash
python examples/memory_planning.py
```

```python
config = dist_moe.Config(
    max_num_local_input_tokens=4,
    hidden_dim=128,
    intermediate_dim=128,
    top_k=2,
    num_experts=4,
    max_moe_layers_per_activation_slot=2,
    activation_slot_capacity_factor=1.0,
    device_scratch_capacity_factor=1.0,
    num_activation_slots=1,
    vmm=dist_moe.VmmConfig(total_scratch_capacity_factor=2.0),
)
print(dist_moe.plan_memory(config, ep_size=2).explain())
```

The three capacity controls have different meanings:

| Setting | Controls | Limit | If exceeded |
| --- | --- | --- | --- |
| `activation_slot_capacity_factor` | Optional saved state across all layers sharing one activation slot | Soft | The device planner recomputes eligible intermediates in backward |
| `device_scratch_capacity_factor` | Temporary scratch for one active layer in device memory | Hard without VMM | Execution reports overflow, or uses host-backed scratch when configured |
| `VmmConfig.total_scratch_capacity_factor` | Device plus host-backed scratch for one active layer | Hard | Execution reports overflow |

Leave both activation-slot controls unset to retain only mandatory layer inputs
and recompute all eligible intermediates. Set
`activation_slot_capacity_factor=1.0` to retain all eligible state when the
slot's aggregate routing matches balanced routing. Expert users may instead
set an exact `activation_slot_bytes`; the two controls are mutually exclusive.

An activation-slot overflow is safe and selects dynamic recomputation. A total
scratch overflow is fatal: the first routing kernel that knows the final local
receive-row count prints the required and available rows, then traps before it
publishes unsafe routing metadata. Only the overflowing rank is guaranteed to
print that diagnostic; peers may time out or be terminated by the job
supervisor. The failed CUDA context cannot be reused. See the
[memory planner guide](docs/memory_planner.md) for formulas and a complete
`plan.explain()` example.

### Reading `plan.explain()`

Memory planning is pure: it does not allocate a context or initialize
communication. The example above makes every part of the plan visible.

Read the result from routing capacity to physical storage:

- **Receive capacity:** `8 balanced -> 8 device -> 16 total`. Balanced routing
  stays in HBM; VMM covers up to twice that routing load.
- **Activation slot:** `1 x 16 KiB`. Capacity factor 1 retains every eligible
  intermediate at balanced routing.
- **Scratch:** `12 KiB device + 12 KiB host`. Scratch grows into host backing
  only after the device budget is exhausted.
- **Allocation:** `28 KiB device / 40 KiB virtual`. Device bytes contain the
  slot and hot scratch; the virtual range also includes host overflow.

The useful range for one slot is 2 KiB for mandatory inputs, 16 KiB for a
balanced full save, and 28 KiB for the maximum executable routing load.

<details>
<summary>Exact <code>plan.explain()</code> output</summary>

```text
Distributed MoE memory plan:
  balanced/device/total rows: 8/8/16
  device/total scratch capacity factors: 1/2
  activation slots: 1 x 0.000 GiB (16,384 bytes) = 0.000 GiB (16,384 bytes) (balanced capacity factor 1)
  device scratch: 0.000 GiB (12,288 bytes)
  host overflow scratch: 0.000 GiB (12,288 bytes)
  total device buffer: 0.000 GiB (28,672 bytes)
  activation slot minimum/balanced full-save/maximum useful: 0.000 GiB (2,048 bytes) / 0.000 GiB (16,384 bytes) / 0.000 GiB (28,672 bytes)
  virtual address range: 0.000 GiB (40,960 bytes); balanced routing fits without recompute; larger saved-state usage may recompute
  activation slots are a soft limit (recompute on exhaustion); total scratch capacity is a hard limit
```

</details>

The activation slot is a soft budget: exhaustion selects recomputation. Total
scratch is a hard bound. Exact byte alignment remains the planner's
responsibility.

<a id="manage-context-lifetime"></a>
## Manage context lifetime

A context's static mode and an individual call's autograd behavior are
different decisions:

| Context | Invocation | Behavior |
| --- | --- | --- |
| Training | Gradients required | Saves eligible state or records dynamic recomputation in the selected activation slot |
| Training | `torch.no_grad()` or no differentiable input | Uses training-compatible kernels, puts intermediates in scratch, and leaves activation-slot state unchanged |
| Inference | Forward only | Uses inference-specialized routing, formats, and scratch-only planning |

The [MXFP8 training example](examples/mxfp8_training.py) demonstrates that
evaluation can reuse a training context:

```python
with torch.no_grad():
    dist_moe.routed_experts(
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        w13_EFD,
        w2_EDF,
        context,
    )
```

`torch.no_grad()` does not turn a training context into an inference context.
Use `Config(inference=True)` for an inference-only context. Training,
training-context evaluation, and specialized inference have different CUDA
graph topologies and must be captured separately.

Call `context.reset()` only after an aborted schedule. Call `context.close()`
after every eager call, trace, and CUDA-graph replay using the context has
finished.

<a id="use-graph-and-checkpoint-features"></a>
## Use graph and checkpoint features

The supported graph boundary includes:

- FakeTensor metadata propagation;
- non-strict `make_fx` with opaque registered Dist-MoE operations;
- activation checkpointing with effectful planner transitions; and
- CUDA graph capture and replay over stable context-owned addresses.

Full `torch.compile`, strict export, and general functionalization of mutable
activation and communication buffers are not supported. A registered operation
does not imply those broader compiler guarantees.

Dynamic activation recomputation remains CUDA-graph compatible because the
host launches one fixed sequence. Device predicates decide whether the
recompute kernels perform work; the host does not branch on the planner result.
Each training forward also returns an immutable snapshot of its selected
activation slot with its planner offsets. Autograd saves that produced state,
so selective activation-checkpoint recomputation cannot substitute a newer
live slot selection. See the
[BF16 execution guide](docs/bf16_execution.md#execution-and-graph-boundary) for
the exact saved-state contract.

<a id="use-advanced-features"></a>
## Use advanced features

### Prepared block-scaled weights

`dist_moe.prepare_block_scaled_weight()` creates caller-owned quantized data
and scales. Prepared execution can use logical sources without readable
storage; see [weight ownership and reuse](docs/mxfp8_execution.md#prepare-and-reuse-weights).
The complete
[block-scaled inference example](examples/block_scaled_inference.py)
prepares both expert weights before execution:

```python
policy = dist_moe.BlockScaledConfig(pipeline="mega")
prepared_w13 = dist_moe.prepare_block_scaled_weight(
    w13_EFD,
    policy,
    inference=True,
    fuse_swiglu=True,
)
prepared_w2 = dist_moe.prepare_block_scaled_weight(
    w2_EDF,
    policy,
    inference=True,
)
```

`fuse_swiglu=True` is a Mega-inference W13 option. Preparation quantizes the
canonical gate/up weights first, then places quantized rows and their scales
in adjacent groups of eight. This lets the inference kernel fuse SwiGLU
without changing the logical BF16 source. Do not enable it for W2 or training.
The default remains concatenated weight preparation, including the staged
pipeline. Device-only Mega inference uses aligned views of context-owned
scratch for either layout, with the automatic tensor-map policy. VMM inference
keeps GPU-planned offsets and device-prepared tensor maps; it does not use the
static scratch layout. Both preparation modes are registered operations with FakeTensor
metadata; actual quantization occurs only when the operation executes.

Training integrations may refill compatible prepared MXFP8 storage with the
`out=` argument after a new weight unshard. Refill preserves the quantized
storage tensors and returns a wrapper associated with the current source weight.

See [MXFP8 execution](docs/mxfp8_execution.md) and
[NVFP4 inference](docs/nvfp4_execution.md) for format-specific fields.

### In-place WGRAD accumulation

`ExecutionOptions(inplace_wgrad_accum=True)` lets serialized backward calls
accumulate W13 and W2 gradients directly into existing standard
`parameter.grad` storage. The first contribution is returned normally; later
contributions use graph-visible mutable destinations and return no duplicate
weight gradient to autograd. See the complete
[BF16 training example](examples/bf16_training.py).

For direct accumulation, `Config.wgrad_dtype` and the parameter declaration
resolve as follows. The examples assume a BF16 parameter.

| `parameter.grad_dtype` | `Config.wgrad_dtype` | Existing `parameter.grad` | Result |
| --- | --- | --- | --- |
| `None` | `None` | `None` | BF16 from `parameter.dtype` |
| FP32 | `None` | `None` | FP32 from the parameter declaration |
| `None` | FP32 | `None` | Explicit FP32 |
| `None` | `None` | FP32 | FP32 from existing storage |
| BF16 | FP32 | any | Error: conflicting declaration |
| `None` | FP32 | BF16 | Error: incompatible storage |

Only BF16 and FP32 are supported, and W13 and W2 must resolve to the same
dtype before either kernel writes. Outside direct accumulation, `None` uses the
compute-weight dtype. A parameter's default `grad_dtype` is its own dtype; set
it to `None` only when an explicitly selected or pre-existing gradient dtype
should take precedence.

```python
options = dist_moe.ExecutionOptions(inplace_wgrad_accum=True)
for _ in range(2):
    output_TD = dist_moe.routed_experts(
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        w13_E2FD,
        w2_EDF,
        context,
        options=options,
    )
    output_TD.float().sum().backward()
```

This is an opt-in serialized-writer contract. Read
[WGRAD accumulation](docs/wgrad_accumulation.md) before integrating it with a
distributed optimizer or pipeline schedule.

### Post-expert processing

`ExecutionOptions.experts_output_postprocess` accepts either a direct eager
callback or a typed `RMSNormPostprocess`. The typed policy fuses route-wise
RMSNorm, router-score weighting, and top-k reduction. See the complete
[BF16 training example](examples/bf16_training.py).

```python
options = dist_moe.ExecutionOptions(
    experts_output_postprocess=dist_moe.RMSNormPostprocess(
        eps=1e-6,
        norm_output_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        require_bitwise=False,
        recompute_rstd=True,
    )
)
```

Callback-free typed policies use the registered graph path. Arbitrary Python
callbacks and observers use the eager extension path. See
[post-expert processing](docs/postprocess.md).

Callback-free reduction and typed RMSNorm mask `-1` routes inside kernels they
already launch. A Python callback normally receives valid expert rows plus
unspecified storage for inactive rows. Set
`zero_out_padded_callback_inputs=True` to zero inactive rows before callback or
observer code; this opt-in policy may add one eager-only bandwidth launch when
the zeroing cannot be fused with an existing save/copy.

### Clamped SwiGLU

Set `activation="swiglu_clamped"` with positive `swiglu_alpha` and
`swiglu_limit`. The fused kernels use the same parameters in forward,
backward, and dynamic recomputation. The complete
[BF16 training example](examples/bf16_training.py) configures the public
policy:

```python
config = dataclasses.replace(
    config,
    activation="swiglu_clamped",
    swiglu_alpha=1.702,
    swiglu_limit=7.0,
)
```

See the precision-specific execution guides for the exact formula.

### Block-major BF16 inference weights

BF16 inference streams W13/W2 faster when each expert's `[N, K]` weight is
stored as `[K / 64, N, 64]`. Rewrite caller-owned weights once per load, then
use a context with `block_major_weights=True`:

```python
dist_moe.relayout_to_block_major(w13_EFD)
dist_moe.relayout_to_block_major(w2_EDF)
config = dataclasses.replace(config, block_major_weights=True)
```

See [the BF16 kernel design](docs/bf16_kernel_design.md#block-major-weights).

### Clip statistics

BF16 execution can rewrite caller-owned SwiGLU counters on every forward. The
complete [BF16 training example](examples/bf16_training.py) supplies the
graph-visible destination:

```python
clip_stats_out_3 = torch.empty(3, dtype=torch.float32, device=device)
options = dist_moe.ExecutionOptions(
    swiglu_clip_stats_out_3=clip_stats_out_3,
    swiglu_clip_limit=7.0,
)
```

### Pipeline activation slots

A pipeline framework derives static activation-slot assignments from schedule
liveness. The complete
[BF16 training example](examples/bf16_training.py) selects one
resolved physical slot:

```python
context.select_activation_slot(
    activation_slot=1,
    num_moe_layers_in_slot=1,
)
```

The context saves the selected device scalar for backward. See
[pipeline activation slots](docs/pipeline_activation_slots.md) for microbatch
and stage-microbatch coloring.

### VMM host-backed scratch

VMM can extend device scratch with pinned host memory while preserving one
stable virtual address range. The complete
[memory-planning example](examples/memory_planning.py) resolves both device and
host-backed scratch before context creation:

```python
config = dataclasses.replace(
    config,
    device_scratch_capacity_factor=1.0,
    vmm=dist_moe.VmmConfig(
        total_scratch_capacity_factor=2.0,
        prefetch=False,
    ),
)
```

Saved activations always remain in device memory. Only scratch beyond the
device capacity may use host backing. See [VMM host-backed scratch](docs/vmm.md).

<a id="public-api"></a>
## Public API

| API | Purpose |
| --- | --- |
| [`dist_moe.Config`](docs/memory_planner.md) | Static shape, precision, capacity, activation-slot, and execution policy |
| [`dist_moe.VmmConfig`](docs/vmm.md) | Optional device-plus-host scratch policy |
| [`dist_moe.Bf16GroupedGemmPreset`](docs/bf16_execution.md) | Named BF16 FPROP/DGRAD launch preset |
| [`dist_moe.BlockScaledFormat`](docs/mxfp8_execution.md) | Supported MXFP8 and NVFP4 formats |
| [`dist_moe.BlockScaledConfig`](docs/mxfp8_execution.md) | Block-scaled precision and staged/Mega pipeline policy |
| [`dist_moe.BlockScaledKernelConfig`](docs/mxfp8_execution.md) | Expert-only complete CuTe launch override |
| [`dist_moe.ExecutionOptions`](docs/wgrad_accumulation.md) | Per-call accumulation, postprocess, observer, and callback controls |
| [`dist_moe.RMSNormPostprocess`](docs/postprocess.md) | Typed fused post-expert RMSNorm policy |
| [`dist_moe.MemoryPlan`](docs/memory_planner.md) / [`dist_moe.plan_memory`](docs/memory_planner.md) | Immutable memory plan and pure planning function |
| [`dist_moe.Context`](docs/bf16_execution.md) / [`dist_moe.create_context`](docs/bf16_execution.md) | Reusable distributed runtime state and collective factory |
| [`dist_moe.PreparedWeight`](docs/mxfp8_execution.md) / [`dist_moe.prepare_block_scaled_weight`](docs/mxfp8_execution.md) | Caller-owned block-scaled weight storage and preparation |
| [`dist_moe.relayout_to_block_major`](docs/bf16_kernel_design.md#block-major-weights) | In-place block-major rewrite of BF16 inference weights |
| [`dist_moe.routed_experts`](docs/bf16_execution.md) | BF16/MXFP8 training or BF16/MXFP8/NVFP4 inference |
| [`dist_moe.supports_fused_post_expert_rmsnorm`](docs/postprocess.md) | Query fused RMSNorm shape/dtype support |

The API docstrings are authoritative for individual arguments and validation
errors. The guides below explain how those APIs compose.

<a id="read-the-detailed-guides"></a>
## Read the detailed guides

Start with the execution guide for the selected precision:

- [BF16 end-to-end execution](docs/bf16_execution.md)
- [Asynchronous MXFP8 end-to-end execution](docs/mxfp8_execution.md)
- [NVFP4 inference execution](docs/nvfp4_execution.md)

Then use the focused guides as needed:

- [Activation and scratch memory planner](docs/memory_planner.md)
- [VMM host-backed scratch](docs/vmm.md)
- [Pipeline activation slots](docs/pipeline_activation_slots.md)
- [Post-expert processing](docs/postprocess.md)
- [WGRAD accumulation](docs/wgrad_accumulation.md)
- [BF16 kernel design](docs/bf16_kernel_design.md)
- [MXFP8 kernel design](docs/mxfp8_kernel_design.md)
- [NVFP4 kernel design](docs/nvfp4_kernel_design.md)

<a id="acknowledgements"></a>
## Acknowledgements

Shikai Li is the primary author of this work, for designing and building its
core: the distributed MoE implementation, the SyncFree execution model, the
activation planning and memory management layers, and the grouped-GEMM and
low-precision kernels. Additional contributions came from Jongsoo Park, Summer
Deng, Jianyu Huang, Jie Wang, Xiaozhu Meng, Vijay Thakkar, Jiecao Yu, Edward
Yang, and others. We also acknowledge the earlier work on mxfp8 dropless
CUDA-Graphable MoE by Daniel Haziza, Luca Wehrstedt, Simon Layton and Driss
Guessous. This work is open-sourced by Sanket Purandare, advised by Tianyu Liu,
Natalia Gimelshein, and Edward Yang.
