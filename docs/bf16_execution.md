# BF16 End-To-End Execution

This guide follows one BF16 training call from context construction through
backward. For kernel internals, see [BF16 kernel design](bf16_kernel_design.md).
For allocation formulas, see [the memory planner](memory_planner.md).

Forward expert matrix multiplication is abbreviated **FPROP**. Backward uses
**DGRAD** for the input gradient and **WGRAD** for the weight gradient. This
guide defines where each value lives, how long it remains live, and which
synchronization makes peer data safe to consume.

## Outline

- [Inputs and ownership](#inputs-and-ownership)
- [Running example](#running-example)
- [Execution and graph boundary](#execution-and-graph-boundary)
- [Forward](#forward)
- [State retained for backward](#state-retained-for-backward)
- [Backward](#backward)
- [Framework integration](#framework-integration)
- [Grouped-GEMM presets](#grouped-gemm-presets)

<a id="inputs-and-ownership"></a>
## Inputs and ownership

For `T` local tokens, hidden width `D`, intermediate width `F`, top-k `K`, and
`E_local` experts, the public call is:

```python
output_TD = dist_moe.routed_experts(
    x_TD,
    topk_expert_ids_TK,
    topk_scores_TK,
    w13_E2FD,
    w2_EDF,
    context,
)
```

| Value | Shape | Dtype | Initial location |
| --- | --- | --- | --- |
| `x_TD` | `[T, D]` | BF16 | ordinary local HBM |
| `topk_expert_ids_TK` | `[T, K]` | integer | ordinary local HBM |
| `topk_scores_TK` | `[T, K]` | floating | ordinary local HBM |
| `w13_E2FD` | `[E_local, 2, F, D]` | BF16 | caller/FSDP-owned local HBM |
| `w2_EDF` | `[E_local, D, F]` | BF16 | caller/FSDP-owned local HBM |

Routing logits and top-k selection happen before this call. The operation does
not contain a router.

`dist_moe.create_context()` allocates capacity-sized symmetric routing,
dispatch, and combine payloads, rendezvouses peer pointers, allocates the
rank-local activation buffer, and registers the context used by graph-visible
operators. It is the only Dist-MoE-owned persistent allocation phase required
by the steady-state BF16 call; caller inputs/outputs and one-time compiler
artifacts remain outside this ownership contract.

<a id="running-example"></a>
## Running example

The rest of this guide follows one small two-rank call. Each rank starts with
two tokens and owns two of four global experts:

| Source rank | Token | Selected experts |
| --- | --- | --- |
| 0 | `A` | `E0`, `E2` |
| 0 | `B` | `E3`, `E1` |
| 1 | `C` | `E2`, `E0` |
| 1 | `D` | `E1`, `E3` |

Rank 0 owns `E0` and `E1`, so its grouped GEMMs process `A, C` for `E0` and
`B, D` for `E1`. Rank 1 similarly processes `A, C` for `E2` and `B, D` for
`E3`. A token is published once on its source rank; the routing metadata may
point several expert assignments at that source row.

From rank 0's perspective, the forward flow is:

```text
ordinary local HBM: A, B
  -> symmetric dispatch: publish A, B
  -> peer gather: read A and B locally, C and D from rank 1
  -> expert-major activation rows: E0[A,C], E1[B,D]
  -> local W13, SwiGLU, and W2
  -> symmetric combine: write each route result to A/B/C/D's owner
  -> score and sum A/B's two local route results
```

Backward reverses that ownership. Each token owner publishes route-output
gradients, each expert rank gathers the rows needed for DGRAD and WGRAD, and
the input-gradient routes are scattered back to their token owners. This
example is deliberately shape-independent: the same movement applies whether
the planner saves intermediates or recreates them in scratch.

<a id="execution-and-graph-boundary"></a>
## Execution and graph boundary

Inference uses one mutable registered operation. Training uses functional,
ordered forward/backward operations so registered autograd and activation
checkpointing preserve the planner transition:

```text
inference: dist_moe.routed_experts -> dist_moe::bf16
training:  dist_moe.routed_experts -> dist_moe::bf16_forward
                                   -> registered autograd
                                   -> dist_moe::bf16_backward
                                      or dist_moe::bf16_backward_accumulate_
```

The inference schema names activation, routing, dispatch, combine, and optional
clip-stat storage as mutated inputs. Training state is returned explicitly from
the functional forward. The planner-produced BF16 state contains six device
`int64` values: five activation-buffer offsets followed by the physical
activation-slot ID selected by that forward. Registered autograd saves this
produced state rather than rereading the live context selector. Selective
activation checkpointing can therefore cache a forward from slot A while a
recomputation observes slot B without redirecting A's backward state.
FakeTensor propagation uses only metadata and never inspects routing or
activation contents. Non-strict `make_fx` retains the opaque operations, and
CUDA graphs replay the same preallocated addresses. Full `torch.compile` and
strict export are not part of this release contract.

### Forward-only evaluation with a training context

The public boundary derives backward retention for each call. Under
`torch.no_grad()`, or when no differentiable operand requires gradients, BF16
uses the same training-compatible forward kernels with `scratch_only=True`.
It omits backward routing metadata and saved activations and leaves every
activation-slot pointer and layer counter unchanged. Training may resume on
the same context without a reset. This is distinct from
`Config.inference=True`, which selects inference-specialized context behavior.

<a id="forward"></a>
## Forward

| Order | API or kernel | Reads | Writes | Residency and lifetime | Synchronization |
| ---: | --- | --- | --- | --- | --- |
| 1 | `copy_routing_and_dispatch()` | local `x_TD` and `topk_expert_ids_TK` | local symmetric routing/dispatch payloads | producer HBM until peers finish the layer | launch-ordered local writes |
| 2 | symmetric-memory barrier | published payloads and signal workspace | peer-visible ordering state | device signal storage | EP device barrier; no CPU synchronization |
| 3 | `dist_dispatch_routing()` | route IDs and peer base pointers | expert row counts plus gather/scatter pointer tables | ordinary local HBM, saved through backward | follows the publication barrier |
| 4 | `get_forward_plan()` | row counts and selected activation slot | `need_recompute` plus five byte offsets and a slot-ID snapshot | device planner state, saved through backward | launch ordered; no host read |
| 5 | `dist_grouped_gemm_fprop_dispatch()` | peer/local BF16 dispatch rows and W13 | gathered `x`, BF16 `h1`, and eligible fused `h2` | `x`/`h1` in the slot or scratch; `h2` always scratch | pointer-driven peer reads after barrier |
| 6 | standalone `swiglu_fwd()`, when required | BF16 `h1` and optional W13 bias | BF16 `h2` and optional clip counters | the same planned `h2` destination | same-stream dependency |
| 7 | `dist_grouped_gemm_fprop_combine()` | BF16 `h2`, W2 and optional W2 bias | BF16 route outputs in peer combine buffers | symmetric peer HBM, optionally copied to the slot | remote stores complete before the following barrier |
| 8 | postprocess plus `scale_and_sum()` | local route outputs and router scores | BF16 `output_TD` | ordinary local HBM returned to the caller | consumes combine rows after the EP barrier |

In inference the registered op launches this chain with programmatic dependent
launch (`kernels/pdl.py`): each kernel waits for its predecessor before its first
global read or write, then lets its successor be scheduled. Launches overlap the
predecessor's tail; data order is unchanged.

### 1. Publish route IDs and source activations

`copy_routing_and_dispatch()` places local route IDs and source activations in
their symmetric HBM payloads. After planning, `copy_dispatch_to_activation()`
conditionally saves `x` at `forward_plan.x_offset` when recomputation is
selected. A GPU symmetric-memory barrier makes the symmetric writes visible to
peers.

The source rank still owns the physical dispatch allocation. A peer expert
rank reads it through a rendezvoused pointer; the row is never copied to CPU.
In the example, rank 0's dispatch buffer contains `A` and `B`, while rank 1's
contains `C` and `D`.

### 2. Build routing metadata

`dist_dispatch_routing()` runs GPU routing metadata kernels that count and pack
rows per local expert and produce:

- `num_tokens_per_local_experts`;
- `fwd_gather_ptrs`, addresses of source BF16 route rows;
- `bwd_gather_ptrs`, addresses used to return input gradients;
- `scatter_ptrs`, addresses of route outputs in peer combine buffers;
- `num_tokens_per_rank` and the local received-row count.

These tensors live in ordinary local HBM or context-owned routing workspace.
Inference routing also zeroes the two grouped-GEMM counter tensors, so the
dispatch and combine launches skip their own memsets.
The BF16 path does not add per-expert row padding. The operation never
transfers route counts to the CPU.

Expert ID `-1` marks an inactive route. Routing does not produce a pointer or
expert row for that slot, and the later score/reduction kernels mask its
unwritten combine storage at load time. An all-`-1` token therefore remains in
the physical `[T, D]` output as an exact zero row.

For rank 0, the resulting expert-major pointer order is `E0[A,C], E1[B,D]`.
Pointers for `A` and `B` address rank 0's dispatch buffer; pointers for `C` and
`D` address rank 1's buffer.

### 3. Plan activation offsets

`get_forward_plan()` updates the selected activation slot on device. It
returns `need_recompute` and a packed six-element state containing offsets for
`x`, `x_gathered`, `h1`, `h2`, and `h3`, followed by the selected physical
slot ID. The offsets are device `int64` byte positions within
`ActivationBuffer.buffer`; the final scalar is the immutable slot snapshot
for the matching backward.

If the slot has space, it retains `x_gathered`, `h1`, and `h3`. Otherwise it
retains only `x`, and backward reconstructs those intermediates in scratch.
`h2` is always temporary scratch, including the full-save case: it can be
recreated from `h1` without retaining another activation. Its recorded offset
does not preserve its contents across later forwards.

### 4. Dispatch, W13, and native SwiGLU

`dist_grouped_gemm_fprop_dispatch()` follows each `fwd_gather_ptr`, reads a BF16
row from local or peer symmetric dispatch HBM, and stores the gathered row at
`x_gathered_offset` in the local activation buffer. It then runs the grouped
W13 GEMM and stores BF16 `h1` at `h1_offset`. Eligible launches also write `h2_offset`
from the same native epilogue:

```text
peer/local dispatch BF16 -> local x_gathered BF16
  -> W13 BF16 x BF16, FP32 accumulation
  -> round-to-nearest h1 BF16 [received_rows, 2F]
  -> SwiGLU on those BF16 values, in FP32 registers
  -> h2 BF16 [received_rows, F]
```

Rank 0 now owns expert-local `h1` rows for `E0[A,C]` and `E1[B,D]`. The rows
are local even when their source token was remote.

W13 and `h1` remain in canonical `[gate; up]` order. The kernel uses an
interleaved TMA view to bring each gate/up pair into one epilogue subtile;
it does not allocate or retain a second weight or activation layout. Both
`h1` and `h2` are written to the existing planner destinations. Their
`int64[1]` byte offsets are read on device, not copied to the host.

Forward fusion is selected only when clip counters and `w13_bias_EF` are
absent and the resolved preset passes the native geometry check: `F` is
divisible by 32,
the preset is non-swapped, its epilogue subtile is 64 columns, and its M tile
contains 128 rows per CTA MMA atom. Both default training presets meet this
contract. An incompatible explicit preset or unresolved inference-auto preset
retains the standalone activation. Existing weight, expert-output, and WGRAD
callbacks keep their original boundaries and do not themselves disable this
fusion. See [the kernel flow](bf16_kernel_design.md#swiglu) for the inline and
deferred epilogues.

### 5. Activation variants and clip observation

The standalone `swiglu_fwd()` reads `h1_offset`, evaluates the nonlinear
expression in FP32 registers, and writes the same BF16 `h2_offset`. When a
caller-owned FP32 `[3]` clip-stat tensor is supplied through
`dist_moe.ExecutionOptions`,
this variant resets and writes gate, up-projection, and valid-element
counters. Training returns those counters as functional private state and the
public wrapper records an explicit `aten.copy_` into the caller tensor;
inference exposes the tensor as a mutated input of `dist_moe::bf16`.
`swiglu_clip_limit` sets the counting threshold; it does not clamp the
activation. The native fused epilogue does not collect these counters.

With `dist_moe.Config(activation="swiglu_clamped")`, either execution variant
evaluates `min(g, limit) * sigmoid(alpha * min(g, limit)) *
(clamp(u, -limit, limit) + 1)`. `swiglu_alpha` and `swiglu_limit` are static
context values used identically in forward, backward, and conditional forward
recomputation.

### 6. W2 plus combine

`dist_grouped_gemm_fprop_combine()` reads `h2`, runs W2 with FP32
accumulation, and stores each BF16 route output through its `scatter_ptr`
directly into the route owner's symmetric combine HBM. A barrier establishes
completion before each owner reads its local combine view.

Without postprocessing, `scale_and_sum()` reshapes route outputs to
`[T, K, D]`, multiplies by `topk_scores`, reduces K in FP32, and returns BF16
`[T, D]`. When backward may need expert outputs, it conditionally saves the
pre-reduction `h3` at `h3_offset`.

In the example, rank 0 writes `E0(C)` and `E1(D)` results to rank 1's combine
buffer and receives its `E2(A)` and `E3(B)` results from rank 1. It then reduces
the two route results for local tokens `A` and `B`.

`dist_moe.ExecutionOptions.experts_output_postprocess` may instead provide a
direct callable or `dist_moe.RMSNormPostprocess`. A callable transforms route-wise `h3`
before the ordinary reduction and selects eager execution. The RMSNorm policy
uses the fused kernel after W2 combine to normalize each route in FP32,
apply router scores, and reduce K. Scalarless RMSNorm supports training; the
optional input-scale weight is inference-only. The fused path saves its rstd
context and conditionally saves the pre-normalization `h3` in the same activation buffer.

<a id="state-retained-for-backward"></a>
## State retained for backward

Autograd saves references to the logical weights, top-k scores, expert row
counts, gather/scatter pointer tensors, per-rank counts, received-row count,
`need_recompute`, and the packed six-element planner state. That state owns all
five forward offsets and the original forward's resolved activation slot.
Backward never derives the slot from the context's current selector.

The offset tensor is meaningful only with the context's activation buffer. It
does not describe a standalone PyTorch allocation. The activation buffer owns the bytes;
the offset identifies the typed view that a Dist-MoE kernel reconstructs.

<a id="backward"></a>
## Backward

For the running example, rank 0 begins with gradients for output tokens `A`
and `B`. Score broadcasting produces four route gradients. W2 DGRAD gathers
the two remote-owned routes needed by `E0` and `E1`; W13 DGRAD later scatters
the corresponding input-gradient routes back to the owners of `A`, `B`, `C`,
and `D`. W13 and W2 WGRAD remain local because rank 0 owns those expert
weights.

| Order | API or kernel | Reads | Writes | Residency and lifetime | Synchronization |
| ---: | --- | --- | --- | --- | --- |
| 1 | `get_backward_plan()` | saved forward offsets and planner state | saved-or-scratch offsets for this layer | device-only planner decision | launch ordered; no host read |
| 2 | conditional FC13/W2 recompute plus standalone SwiGLU | saved `x` or `h1` and fixed scratch offsets | reconstructed intermediates and fresh scratch `h2` | fixed launches; GEMMs are device-predicated, SwiGLU always runs | conditional peer publication and barrier |
| 3 | postprocess backward / `broadcast_and_scale()` | `grad_output_TD`, router scores, and `h3` | BF16 `grad_h3` in symmetric combine HBM plus `grad_topk_scores_TK` | peer-visible until W2 DGRAD completes | combine barrier precedes peer reads |
| 4 | `dist_grouped_gemm_dgrad_dispatch()` | peer `grad_h3` rows and W2 | BF16 `grad_h2` and gathered `grad_h3` | shared scratch | pointer-driven peer reads |
| 5 | `grouped_gemm_wgrad()` | `grad_h3` and `h2` offsets | W2 gradient | caller-selected BF16/FP32 output or owned destination | same-stream dependency |
| 6 | `dist_grouped_gemm_dgrad_swiglu_bwd_combine()` when `F % 256 == 0` | `grad_h2`, `h1`, and W13 | planned BF16 `grad_h1` plus peer route input gradients | existing scratch and symmetric dispatch HBM | producer counters order SwiGLU stores before W13 TMA reads |
| 7 | otherwise `swiglu_bwd()` then `dist_grouped_gemm_dgrad_combine()` | the same inputs | the same `grad_h1` and peer destinations | the same planned storage | same-stream dependency; scatter precedes the dispatch barrier |
| 8 | `grouped_gemm_wgrad()` + `reduce_from_topk()` | `grad_h1`, `x_gathered`, and peer route gradients | W13 gradient and BF16 `grad_x_TD` | gradient outputs returned to autograd | reduction follows the dispatch barrier |

### 1. Resolve the backward plan and optional recompute

`get_backward_plan()` allocates scratch offsets for activation gradients and
selects either the saved forward offsets or recompute scratch offsets. The host
launches the fixed recompute sequence:

1. `copy_activation_to_dispatch()` republishes saved `x` when required.
2. Conditional `dist_grouped_gemm_fprop_dispatch()` recreates
   `x_gathered/h1` without forward activation fusion.
3. Standalone `swiglu_fwd()` always recreates scratch `h2` from the selected
   saved or recomputed `h1`.
4. Conditional `dist_grouped_gemm_fprop_combine()` recreates `h3`.

The host never reads `need_recompute`. FC13 and W2 recompute launches are
device-predicated no-ops when their state was saved, but SwiGLU must still
rebuild `h2`: another forward may have overwritten that shared scratch. This
is why recompute keeps the standalone activation even when the original
forward was fused. It does not recollect clip counters. No host branch on the
saved-state decision or new activation allocation occurs during graph replay.

### 2. Router-score gradient and route output gradient

Without postprocessing, `broadcast_and_scale()` combines `grad_output`,
`topk_scores`, and saved or recomputed `h3`. It publishes BF16 `grad_h3` in
symmetric combine HBM and returns `grad_topk_scores`. A direct callback is
replayed under autograd to recover its input gradient. `dist_moe.RMSNormPostprocess`
uses the matching fused backward to produce both the normalization gradient and
router-score gradient, then publishes the same `grad_h3` contract.

### 3. W2 DGRAD and WGRAD

`dist_grouped_gemm_dgrad_dispatch()` gathers peer `grad_h3` rows, runs W2
DGRAD, and writes BF16 `grad_h2` plus gathered `grad_h3` to scratch.

`grouped_gemm_wgrad()` consumes the device offsets for gathered `grad_h3` and
`h2` and computes W2 WGRAD with FP32 MMA accumulation. The output dtype is
the explicit `config.wgrad_dtype` or, when it is `None`, the compute-weight
dtype. Direct parameter-gradient accumulation additionally respects the
parameter's declared `grad_dtype` and existing gradient storage. The
underlying grouped-GEMM kernel also accepts an explicit destination and
overwrite or reduce-add semantics.

W2 WGRAD must consume `h2` before the next step writes `grad_h1`. The scratch
plan reuses the former `h2` range for `grad_h1`; forward fusion and backward
fusion do not change that lifetime or add another gradient allocation.

### 4. SwiGLU backward, W13 DGRAD, and W13 WGRAD

For `F` divisible by 256, `dist_grouped_gemm_dgrad_swiglu_bwd_combine()` reads
the planned `grad_h2` and saved or recomputed `h1` offsets. Its native producer
computes the derivative in FP32, stores BF16 `grad_h1` in canonical
`[gate; up]` order, and signals each completed activation M tile. The same
launch runs W13 DGRAD after the corresponding signal and scatters route input
gradients to peer symmetric dispatch HBM through `bwd_gather_ptrs`.

Other intermediate widths use `swiglu_bwd()` followed by
`dist_grouped_gemm_dgrad_combine()`. Forward and backward eligibility are
independent: forward needs a 32-column half-subtile, while the native backward
producer needs complete 256-column passes. Both backward variants write the
same `grad_h1_offset`; neither allocates a second `grad_h1`.
`grouped_gemm_wgrad()` then consumes that canonical `grad_h1` and
`x_gathered` to produce W13 WGRAD.

After the dispatch barrier, `reduce_from_topk()` sums the K route gradients for
each local token and returns BF16 `grad_x`.

By default, the functional backward returns fresh W13 and W2 gradients to
ordinary autograd. With `ExecutionOptions(inplace_wgrad_accum=True)`, later
serialized contributions use `dist_moe::bf16_backward_accumulate_` and mutate
both existing `parameter.grad` tensors explicitly. See
[WGRAD accumulation](wgrad_accumulation.md) for ownership and graph-capture
requirements.

<a id="framework-integration"></a>
## Framework integration

The BF16 package accepts ordinary logical weight tensors and does not own FSDP.
A framework may all-gather BF16 weights and pass their logical tensor into this
call. A pipeline integration sizes the configured activation-slot count from
schedule liveness and calls `context.select_activation_slot(slot, slot_depth)`
before each forward. Here `slot_depth` is the selected stage's MoE depth for
stage-microbatch coloring; microbatch coloring instead passes the total number
of local MoE layers sharing that slot. Backward uses the slot and layer-depth
bound saved by forward.
See [activation-slot scheduling](pipeline_activation_slots.md) for sizing and
static assignment.

Call `context.close()` only after all eager work, traced execution, and CUDA
graph replay that reference the context have stopped.

<a id="grouped-gemm-presets"></a>
## Grouped-GEMM presets

`dist_moe.Config.bf16_grouped_gemm_preset` accepts a named compile-time launch
preset for BF16 grouped GEMMs. Names use
`<cluster CTAs>cta<MMA atoms>mma_bm<M>_bn<N>`; for example,
`2cta1mma_bm256_bn256` means a two-CTA cluster, one MMA atom per CTA, and a
256-by-256 output tile. An explicit preset is passed to FPROP, DGRAD, and
WGRAD; it also overrides the separate default W2-combine choice.

With no explicit preset, training uses the retained native policy with
`R = max(1, T * top_k)` and `E_local` local experts:

| Launch | BM256 selection | Otherwise |
| --- | --- | --- |
| W13 FPROP and both DGRADs | `R <= 8192` or `R < 512 * E_local` | BM512, one TMEM buffer |
| W2 FPROP combine | the same small-row condition, or `R >= 16384 * E_local` | BM512, one TMEM buffer |
| WGRAD | `R <= 8192` | its BM512 preset, with two C stages |

The BM256 training preset has two TMEM buffers. These are source launch
policies, not a speedup guarantee. `R` uses the current invocation's local
`T`, not the context maximum or a host copy of received counts. Actual routed
rows remain device-selected and may differ from the estimate; the hard
capacity bound comes from [the memory plan](memory_planner.md), not `R`.
Inference with no explicit preset leaves selection to the shape-aware launcher.

An explicit preset is appropriate only after benchmarking the exact received-row
distribution, hidden and intermediate dimensions, local expert count, and GPU
architecture. A preset changes launch geometry, not arithmetic or the memory
planner contract.
