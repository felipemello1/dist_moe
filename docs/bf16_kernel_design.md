# BF16 Kernel Design

This guide explains the GPU kernels behind the BF16 execution flow. Read
[BF16 end-to-end execution](bf16_execution.md) first if you need the public API,
saved-state, or backward-lifetime model.

**CTA** means cooperative thread array, the unit of thread-block scheduling.
**TMA** moves tensor tiles between global memory (GMEM) and shared memory
(SMEM). **TMEM** is tensor memory used by Blackwell tensor cores.

## Outline

- [Scope](#scope)
- [Persistent grouped GEMM](#persistent-grouped-gemm)
- [Block-major weights](#block-major-weights)
- [CTA roles](#cta-roles)
- [SwiGLU](#swiglu)
- [Launch policy](#launch-policy)
- [WGRAD](#wgrad)
- [Synchronization](#synchronization)
- [Numerical invariants](#numerical-invariants)

<a id="scope"></a>
## Scope

The BF16 path fuses expert-parallel data movement with grouped expert GEMMs.
Routing scores and top-k IDs are inputs; routing itself is not part of the
expert kernels. Matrix products use BF16 operands and FP32 tensor-core
accumulation. Outputs are converted to BF16 at the epilogue unless WGRAD is
configured for FP32 output.

The kernel families are:

| File | Responsibility |
| --- | --- |
| `kernels/dist_grouped_gemm.py`, `kernels/dist_grouped_gemm_kernel.py` | Distributed launchers and warp-specialized dispatch/combine kernels |
| `kernels/grouped_gemm.py`, `kernels/grouped_gemm_kernel.py` | Local grouped FPROP/DGRAD/WGRAD launchers and shared device machinery |
| `kernels/swiglu_fragment.py` | Natural-h1 forward epilogue, using an interleaved TMA view |
| `kernels/native_combine_swiglu.py` | Native SwiGLU-backward producer inside W13 DGRAD combine |
| `kernels/tile_scheduler.py` | Persistent grouped tile assignment and producer/consumer handoff |

<a id="persistent-grouped-gemm"></a>
## Persistent grouped GEMM

One launch covers all local experts. `split_sizes[e]` gives the number of rows
owned by expert `e`; a persistent scheduler maps each work ID to:

```text
(expert, expert-local M tile, output N tile)
```

Rows never cross expert boundaries. Tail rows and columns are predicated.
Host-side configuration chooses one- or two-CTA clusters, tile M/N/K, pipeline
depth, epilogue subdivision, and static or dynamic scheduling. These choices
are compile-time CuTe parameters and form a performance-sensitive
specialization contract.

Training and backward patch the activation (A) TMA descriptor once per expert,
after the previous expert's last load lands. BF16 inference FPROP (one-CTA,
unpadded; A, or B when swapped) instead patches one descriptor per CTA over every
expert's rows and addresses expert `e` at row `sum(split_sizes[:e])`, so decode
tiles, whose consecutive tiles belong to different experts, skip the patch, the
fence and the wait. A tile's last rows may then load the next expert's rows:
each output row depends only on its own A row, and every epilogue stores only
rows below that expert's `split_sizes[e]`. The descriptor extent is the exact
row total, so TMA zero-fills past the last expert.

<a id="block-major-weights"></a>
## Block-major weights

A B tile of a row-major `[N, K]` weight is `BLOCK_SIZE_N` strided 128-byte rows.
With `Config.block_major_weights`, every W13 and W2 was rewritten by
`relayout_to_block_major`, so each expert stores `[K / 64, N, 64]` and a 64-wide
K tile of all its rows is contiguous:

```text
row-major:    r0[0:64] r0[64:128] r1[0:64] r1[64:128] ...
block-major:  r0[0:64] r1[0:64] ... r(N-1)[0:64] r0[64:128] r1[64:128] ...
```

The FPROP dispatch and combine kernels view the weight operand (B, or A when
swapped) as `(N, (64, K/64), G) : (64, (1, 64N), NK)`; the TMA atom and
partitions are unchanged. Only BF16 inference supports it: DGRAD and WGRAD
read the weights row-major, and the fused SwiGLU dispatch epilogue runs with it
only for [interleaved W13](#interleaved-w13).

<a id="cta-roles"></a>
## CTA roles

The ordinary grouped GEMM divides work into pipelined roles:

```text
TMA producer     GMEM -> staged A/B SMEM, descriptor updates, full barriers
MMA consumer     SMEM -> tcgen05 MMA -> FP32 TMEM accumulators
epilogue group   TMEM -> registers -> C SMEM -> 16-byte/TMA GMEM stores
idle warps       participate only in required cluster/warpgroup ordering
```

The distributed dispatch kernel adds gather warps. They load source rows using
`gather_ptrs`; a pointer may address the local symmetric dispatch buffer or a
peer rank's dispatch buffer. Gathered BF16 rows are written to the local
activation buffer and signal a per-M-tile counter. The TMA producer waits for
that counter before staging the row tile.

```text
local/peer symmetric dispatch HBM
  -> gather warps
  -> x_gathered BF16 in local activation buffer
  -> TMA to SMEM
  -> tcgen05 BF16 x BF16, FP32 TMEM accumulation
  -> BF16 epilogue in SMEM
  -> local activation buffer (dispatch result)
```

The plain combine kernel reverses the communication direction. It reads
expert-local rows, runs the grouped GEMM, and resolves each route's
`scatter_ptr` in its epilogue. Sixteen-byte stores write the BF16 result directly to the owning
rank's symmetric combine HBM.

```text
h2 BF16 in local activation buffer
  -> TMA to SMEM
  -> tcgen05 BF16 x BF16, FP32 TMEM accumulation
  -> BF16 in SMEM
  -> 16-byte stores to local or peer symmetric combine HBM
```

Native distributed epilogues bound each TMEM load to 32 FP32 destination
registers per thread. This avoids a wider instruction's allocation constraint
within the fixed 168-register epilogue budget. The logical fragment and its
lifetime through H1 stores and fused SwiGLU stay unchanged. A narrower copy atom
covers the same tile without adding a kernel, allocation, epilogue pass, or
different rounding point. Plain and block-scaled GEMMs retain their separate
compiler-specific load selection.

<a id="swiglu"></a>
## SwiGLU

### Forward state flow

The native forward mode combines dispatch, W13, and SwiGLU in one launch.
W13 remains in canonical `[gate; up]` order. `interleaved_fc13_view` changes
the TMA view of the weight and output axes so one 64-column epilogue subtile
contains 32 gate columns and their 32 up columns. C stores map the result back
to natural `h1`; this view does not create another tensor allocation.

The BF16 rounding point remains before the nonlinear operation:

```text
peer/local BF16 -> gathered x -> W13 -> FP32 TMEM accumulator
  -> round-to-nearest BF16 h1 fragment
       -> C TMA store: natural h1 at its planned byte offset
       -> inline fragment or deferred h1 load
            -> FP32 SwiGLU -> BF16 h2 at its planned byte offset
```

The selected source pipeline determines where that activation work happens:

- With two TMEM buffers, the epilogue computes `h2` from its rounded BF16
  register fragment while the MMA can work on the other accumulator buffer.
- With one TMEM buffer, it releases the accumulator first, waits for the h1
  TMA stores, fences the async proxy, and computes `h2` from stored BF16 h1.

No new barriers are substituted for this source ordering. The forward kernel
loads H1/H2 offsets on device and checks their packed-row byte extents against
the existing activation buffer. Gathered `x`, natural `h1`, and compact `h2`
occupy the planner's existing disjoint live ranges.

Fusion requires `F % 32 == 0`, no A/B swap, a 64-column epilogue subtile, and
`BLOCK_SIZE_M == 128 * NUM_CTAS * NUM_MMAS`. The public orchestration selects
it only for a resolved compatible preset and without clip counters. Otherwise
the Triton `swiglu_fwd` operates on the same H1/H2 offsets. Clip observation
resets and accumulates its FP32 `[3]` counters in that standalone variant;
it is not a capability of the native epilogue.

<a id="interleaved-w13"></a>
### Interleaved W13 (inference)

With `Config.interleaved_gate_up`, W13 itself stores gate and up rows in groups
of 16 (`relayout_to_block_major(w13, interleave_gate_up=True)`), so a C staging
tile already holds each gate column next to its up column. The dispatch
epilogue then writes only `h2`, swapped or not: no `h1` store and no standalone
SwiGLU. Each group of 16 gate/up values is split over two threads, one batched
sigmoid of 8 values each. It adds no clip counters; see
`Config.interleaved_gate_up` for where it loses.

Forward fusion does not extend H2's lifetime. The planner always places it in
shared scratch, while H1 may be saved in an activation slot. Backward therefore
conditionally reruns FC13 and always runs standalone SwiGLU to reconstruct H2
from the selected saved or recomputed H1. This restores the W2 WGRAD input
without retaining another activation or changing the scratch layout.

### Backward state flow

When `F % 256 == 0`, the W13 DGRAD combine kernel uses its extra producer
warps for native SwiGLU backward:

```text
planned grad_h2 BF16 + saved/recomputed natural h1 BF16
  -> FP32 derivative
  -> canonical [gate; up] grad_h1 BF16 at the planned scratch offset
  -> per-M-tile completion signal
  -> TMA -> W13 DGRAD -> peer input-gradient scatter
```

The producer works in complete 256-column passes and signals each activation
M tile only after its row items are stored. The GEMM TMA producer waits on
that tile's counter, not on a host read or an additional collective. Tail rows
are masked while the tile-completion protocol still covers the partial tile.
For other widths, Triton `swiglu_bwd` and ordinary W13 DGRAD combine run as
two launches over the same planned storage.

W2 WGRAD runs before either backward variant. It still needs forward `h2`,
whose scratch range is reused for `grad_h1`. Fusion does not move that overwrite
earlier or allocate a second gradient tensor. W13 WGRAD subsequently reads
canonical `grad_h1` and gathered `x`. See
[the backward lifetime](bf16_execution.md#backward) for the complete order.

Both native and standalone variants implement clamped SwiGLU when requested.
They clamp the gate above by `limit`, clamp the up branch to `[-limit, limit]`,
add one to the up branch, and evaluate the sigmoid with multiplier `alpha`.
The arithmetic and derivative remain in FP32 registers before BF16 stores.

<a id="launch-policy"></a>
## Launch policy

The retained configuration chooses tile geometry and pipeline depth from the
host estimate `max(1, T * top_k)` and local expert count, without inspecting
actual received counts on the host. W2 FPROP combine has a distinct large-row
selection rule; WGRAD also has its own default pipeline. The exact policy and
explicit-preset behavior are documented in
[grouped-GEMM presets](bf16_execution.md#grouped-gemm-presets).

Distributed scheduling preserves the source's rank rotation, gather-K segment
signals, and combine N-band choices. These controls are part of the compiled
kernel/cache specialization. They do not change routing capacity or authorize
a different physical scratch layout. A source-selected pipeline or possible
overlap is not evidence of a measured speedup.

<a id="wgrad"></a>
## WGRAD

`grouped_gemm_wgrad` computes, per expert:

```text
dW[e] = dY[e].T @ X[e]
```

Both operands may be ordinary tensors or device byte offsets into the shared
activation buffer. The kernel reads those offsets on device, so dynamic save or
recompute policy causes no CPU synchronization. MMA accumulates in FP32 TMEM.
The epilogue either overwrites a BF16/FP32 destination or uses reduce-add stores
for configured in-place accumulation.

<a id="synchronization"></a>
## Synchronization

- Symmetric-memory barriers establish visibility after source publication and
  before consuming peer-written combine output.
- Gather-completion counters order peer loads before TMA consumes a local tile.
- Fused backward producer counters order `grad_h1` stores before W13 TMA reads.
- SMEM full/empty mbarriers order TMA and MMA stages.
- TMEM full/empty mbarriers order MMA and epilogue stages.
- Two-CTA configurations use cluster barriers and multicast masks for their
  cooperative tile.

These are GPU ordering operations. They are not CPU/GPU synchronizations.

<a id="numerical-invariants"></a>
## Numerical invariants

Byte parity depends on preserving the selected tile configuration, row order,
expert grouping, FP32 accumulation, BF16 conversion point, SwiGLU expression,
barrier order, and peer-store order. Refactoring host orchestration must not
change any of those properties.

The kernels use deterministic round-to-nearest conversions; stochastic
rounding and RNG state are not part of this execution path. Native forward
keeps the BF16 h1 rounding boundary, and native backward keeps the BF16
grad_h1 boundary before W13 DGRAD/WGRAD. Numerical and performance acceptance
must compare the same preset, activation flags, routed population, and saved
or recomputed state rather than inferring equivalence from fusion alone.
