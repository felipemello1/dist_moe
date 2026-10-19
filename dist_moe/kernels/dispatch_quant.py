# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-side dispatch-quant producer helpers shared by the DistMoE kernels.

Free ``@cute.jit`` functions hoisted out of ``DistBlockScaledGroupedGemmKernel``.
The wide quantize/store pair reads its compile-time attributes through a
``DispatchQuantParams`` record (the ``arguments.py`` ``CuteParamsBase``
convention); the narrower helpers take explicit ``cutlass.Constexpr``
parameters (the ``swiglu_epilogue.py`` convention).
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op

from ..formats import (
    CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE,
    CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM,
)
from ._quant_conversion import (
    _unpack_b16x2,
)
from .blockscaled_grouped_gemm import (
    NVFP4,
)
from .blockscaled_quantization_common import (
    _compute_amax_nonfinite_b16x2,
    _compute_scale_and_recip_from_amax,
    _cublas_blockscaled_qscale_offset,
    _max_b16x2,
    _reduce_amax_b16x2,
    _store_fp4_tile_qdata,
    _store_fp8_qwords,
)
from .grouped_gemm import (
    _FPROP,
)
from .params import (
    ceil_div,
    DispatchQuantParams,
)
from .tile_scheduler import (
    GroupedProblemVisitor,
)


@dsl_user_op
def _dispatch_quant_nanosleep(
    sleep_ns,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [cutlass.Uint32(sleep_ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _wait_counter_at_least(ptr: cute.Pointer, expected: cutlass.Uint32) -> None:
    """Warp-uniform acquire-spin until the counter behind ``ptr`` reaches ``expected``."""
    # Fast path: every lane performs the acquire probe, then the ballot makes
    # the slow-path branch warp-uniform. If any lane observes a stale value,
    # the whole warp waits; otherwise every lane has observed the release.
    done = cute.arch.load(ptr, cutlass.Uint32, sem="acquire", scope="gpu")
    needs_wait = cute.arch.vote_ballot_sync(done < expected) != 0
    if needs_wait:
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            # Check-first spin at fixed 50 ns granularity. Exponential
            # backoff here is a measured regression: sleep-before-check with
            # 64->2048 ns growth cost +3-8% on Mega blockscaled small-token
            # decode across EP16-64, and even check-first growth after a
            # 1.6 us grace window still measured +2.7% (square NVFP4 decode 89.7
            # and 86.4 us respectively vs 84.1 us for this form, which is
            # trunk parity) -- decode's per-tile handoff waits are long
            # enough to engage any backoff but too latency-critical to
            # tolerate coarse discovery.
            ready = False
            while not ready:
                spin_done = cute.arch.load(
                    ptr, cutlass.Uint32, sem="acquire", scope="gpu"
                )
                ready = spin_done >= expected
                if not ready:
                    _dispatch_quant_nanosleep(50)
        cute.arch.sync_warp()


@cute.jit
def _wait_lane_counters_at_least(
    ptr: cute.Pointer, active: cutlass.Boolean, expected: cutlass.Uint32
) -> None:
    """Warp-uniform acquire-spin until every active lane's counter behind
    ``ptr`` reaches ``expected``.

    Each lane probes its own counter, so a wait on up to 32 counters costs one
    round trip instead of one serial acquire per counter. ``ptr`` must be
    readable on inactive lanes too.
    """
    done = cute.arch.load(ptr, cutlass.Uint32, sem="acquire", scope="gpu")
    pending = cute.arch.vote_ballot_sync(active & (done < expected)) != 0
    while pending:
        _dispatch_quant_nanosleep(50)
        done = cute.arch.load(ptr, cutlass.Uint32, sem="acquire", scope="gpu")
        pending = cute.arch.vote_ballot_sync(active & (done < expected)) != 0
    # Other lanes performed some of the acquires; this orders them before the
    # dependent loads issued by any single lane.
    cute.arch.sync_warp()


def _make_u32x4_vector_type():
    """MLIR ir.VectorType<4xi32> for 16-byte vectorized load/store."""
    return ir.VectorType.get([4], cutlass.Uint32.mlir_type)


@dsl_user_op
def _load_u32x4_or_zero(
    addr: cutlass.Int64, pred: cutlass.Boolean, *, loc=None, ip=None
) -> ir.Value:
    """16-byte ``ld.global.cg`` when ``pred``, else zeros and no load.

    ``cute.arch.load`` takes no predicate; this is CUTLASS ``arch::global_load``.
    Issue a batch of these before any store so their round trips overlap.
    """
    words = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [
            cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p;\n"
        "setp.ne.b32 p, $5, 0;\n"
        "mov.b32 $0, 0;\n"
        "mov.b32 $1, 0;\n"
        "mov.b32 $2, 0;\n"
        "mov.b32 $3, 0;\n"
        "@p ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4]; }",
        "=r,=r,=r,=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
    )
    return vector.from_elements(
        _make_u32x4_vector_type(),
        [llvm.extractvalue(T.i32(), words, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )


@cute.jit
def _load_group_global_scale_inv(
    b_global_scale_inv_ptr: cutlass.Int64,
    g: cutlass.Int32,
    use_global_scale_inv: cutlass.Constexpr[bool],
) -> cutlass.Float32:
    if cutlass.const_expr(not use_global_scale_inv):
        return cutlass.Float32(1.0)
    return cute.arch.load(
        cute.make_ptr(
            cutlass.Float32,
            b_global_scale_inv_ptr,
            cute.AddressSpace.gmem,
            assumed_align=4,
        )
        + g,
        cutlass.Float32,
        cop="cg",
    )


@cute.jit
def _load_gather_row_as_b16x2(
    row_addr: cutlass.Int64,
    k_base: cutlass.Int32,
    K: cutlass.Int32,
    values: cute.Tensor,
    lane_elems: cutlass.Constexpr[int],
    source_dtype: type[cutlass.Numeric],
) -> None:
    if cutlass.const_expr(
        source_dtype != cutlass.BFloat16 and source_dtype != cutlass.Float16
    ):
        raise TypeError("packed gather-row source loads require BF16 or FP16")
    if cutlass.const_expr(values.element_type != cutlass.Int32):
        raise TypeError("packed gather-row source load destination mismatch")
    if cutlass.const_expr(lane_elems % 2 != 0):
        raise ValueError("packed b16x2 source loads require even lane_elems")
    pair_elems: cutlass.Constexpr[int] = lane_elems // 2
    chunk_pairs: cutlass.Constexpr[int] = 128 // cutlass.Int32.width
    if cutlass.const_expr(pair_elems % chunk_pairs != 0):
        raise ValueError("packed lane pairs must be divisible by 128-bit load width")

    values.fill(cutlass.Int32(0))

    if row_addr != cutlass.Int64(0):
        row_ptr = cute.make_ptr(
            cutlass.Int32,
            row_addr,
            cute.AddressSpace.gmem,
            assumed_align=128,
        )
        row_tensor = cute.make_tensor(
            row_ptr,
            cute.make_ordered_layout((K // cutlass.Int32(2),), order=(0,)),
        )
        row_chunks = cute.tiled_divide(row_tensor, (chunk_pairs,))
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Int32,
            num_bits_per_copy=128,
        )
        for chunk in cutlass.range_constexpr(pair_elems // chunk_pairs):
            loaded = cute.make_rmem_tensor(chunk_pairs, cutlass.Int32)
            cute.copy(
                copy_atom,
                row_chunks[None, k_base // cutlass.Int32(2 * chunk_pairs) + chunk],
                loaded,
            )
            for i in cutlass.range_constexpr(chunk_pairs):
                values[chunk * chunk_pairs + i] = loaded[i]


@cute.jit
def _dispatch_quant_fetch_group_tile_id(
    mWorkCounter: cute.Tensor,
    tile_smem_ptr: cute.Pointer,
    quant_group: cutlass.Int32,
    quant_group_lane: cutlass.Int32,
    DISPATCH_QUANT_SYNC_BAR: cutlass.Constexpr[int],
    DISPATCH_QUANT_GROUP_THREADS: cutlass.Constexpr[int],
    barrier_id_base: cutlass.Constexpr[int] | None = None,
    num_threads: cutlass.Constexpr[int] | None = None,
    smem_stride: cutlass.Constexpr[int] = 1,
    relaxed_work_counter: cutlass.Constexpr[bool] = False,
) -> cutlass.Int32:
    if cutlass.const_expr(barrier_id_base is None):
        barrier_id_base = DISPATCH_QUANT_SYNC_BAR
    if cutlass.const_expr(num_threads is None):
        num_threads = DISPATCH_QUANT_GROUP_THREADS
    barrier_id = barrier_id_base + quant_group
    tile_idx_raw = cutlass.Uint32(0)
    group_tile_smem_ptr = tile_smem_ptr + quant_group * cutlass.Int32(smem_stride)
    if quant_group_lane == cutlass.Int32(0):
        work_counter_ptr = cute.recast_ptr(
            mWorkCounter.iterator,
            dtype=cutlass.Uint32,
        )
        # Relaxed callers only allocate unique IDs; readiness is published
        # separately through the release-ordered done counter.
        tile_idx_raw = cute.arch.atomic_add(
            work_counter_ptr,
            cutlass.Uint32(1),
            sem="relaxed" if relaxed_work_counter else "release",
            scope="gpu",
        )
        cute.arch.store(
            group_tile_smem_ptr,
            cutlass.Int32(tile_idx_raw),
            ss="cta",
        )
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)
    tile_idx = cute.arch.load(group_tile_smem_ptr, cutlass.Int32, ss="cta")
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)
    return tile_idx


@cute.jit
def _store_dispatch_qwords(
    params: DispatchQuantParams,
    mQWords: cute.Tensor,
    q_vals: cute.Tensor,
    row: cutlass.Int32,
    q_word_col: cutlass.Int32,
    row_lane_in_tile: cutlass.Int32,
    col_quant: cutlass.Constexpr[bool] = False,
) -> None:
    if cutlass.const_expr(params.is_fp4):
        _store_fp4_tile_qdata(
            mQWords,
            q_vals,
            row,
            row_lane_in_tile,
            q_word_col,
            params.DISPATCH_QUANT_ELEMS_PER_LANE,
            params.DISPATCH_QUANT_COL_LANES,
            params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD,
            params.format_id,
            col_quant,
        )
    else:
        _store_fp8_qwords(
            mQWords,
            q_vals,
            row,
            q_word_col,
            params.DISPATCH_QUANT_ELEMS_PER_LANE,
            params.format_id,
            params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD,
            cache_modifier="cs",
        )


@cute.jit
def _dispatch_copy_blockscaled_work_tile(  # noqa: C901
    gather_ptrs: cute.Pointer,
    mRowQWords: cute.Tensor,
    mRowScale: cute.Tensor,
    mRowGlobalScaleInv: cute.Tensor,
    row_start: cutlass.Int32,
    row_scale_start: cutlass.Int32,
    scale_col_start: cutlass.Int32,
    K: cutlass.Int32,
    group_lane: cutlass.Int32,
    group_threads: cutlass.Constexpr[int],
    scale_cols_per_work_tile: cutlass.Constexpr[int],
    qdata_prefetch: cutlass.Constexpr[int],
    use_global_scale_inv: cutlass.Constexpr[bool],
    DISPATCH_QUANT_QDATA_ELEMS_PER_WORD: cutlass.Constexpr[int],
    DISPATCH_QUANT_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    VECTOR_DISPATCH_SCALE_COPY: cutlass.Constexpr[bool],
    sf_vec_size: cutlass.Constexpr[int],
    gsi_row_start: cutlass.Int32 | None = None,
) -> None:
    scale_cols = K // cutlass.Int32(sf_vec_size)
    scale_col_blocks = ceil_div(
        scale_cols,
        cutlass.Int32(CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM),
    )
    q_words_per_scale: cutlass.Constexpr[int] = (
        sf_vec_size // DISPATCH_QUANT_QDATA_ELEMS_PER_WORD
    )
    q_words_per_work_tile: cutlass.Constexpr[int] = (
        scale_cols_per_work_tile * q_words_per_scale
    )
    q_word_cols = K // cutlass.Int32(DISPATCH_QUANT_QDATA_ELEMS_PER_WORD)
    qdata_row_bytes = q_word_cols * cutlass.Int32(4)
    qword_vecs_per_work_tile: cutlass.Constexpr[int] = q_words_per_work_tile // 4
    qword_vec_count: cutlass.Constexpr[int] = sf_vec_size * qword_vecs_per_work_tile
    # MXFP8 rows: padded rows (null gather pointer) load zeros inside the batch,
    # so the qdata loads stay in flight, and each row's scales come in 16-byte
    # loads issued before the qdata. NVFP4 and MXFP4 keep the old copy (not timed).
    is_mxfp8: cutlass.Constexpr[bool] = (
        sf_vec_size == 32 and DISPATCH_QUANT_QDATA_ELEMS_PER_WORD == 4
    )
    scale_cols_per_load: cutlass.Constexpr[int] = 16  # 1-byte scales, 16-byte load
    scale_loads_per_row: cutlass.Constexpr[int] = (
        scale_cols_per_work_tile // scale_cols_per_load
    )
    load_row_scales: cutlass.Constexpr[bool] = (
        is_mxfp8
        and scale_cols_per_work_tile % scale_cols_per_load == 0
        and sf_vec_size * scale_loads_per_row <= group_threads
    )
    if cutlass.const_expr(load_row_scales):
        is_scale_lane = group_lane < cutlass.Int32(sf_vec_size * scale_loads_per_row)
        scale_row = group_lane // cutlass.Int32(scale_loads_per_row)
        scale_load_col = scale_col_start + (
            group_lane - scale_row * cutlass.Int32(scale_loads_per_row)
        ) * cutlass.Int32(scale_cols_per_load)
        scale_row_addr = cutlass.Int64(0)
        if is_scale_lane:
            scale_row_addr = cute.arch.load(
                gather_ptrs + scale_row,
                cutlass.Int64,
                ss="cta",
            )
        # Rows, K and `scale_col_start` are multiples of 16 bytes, so a load that
        # starts in the scales stays in the row. K % 128 (checked for every inference
        # dispatch in `_blockscaled_dispatch_stride_bytes`) makes the 4-byte stores
        # whole 4-column atoms.
        scale_words = _load_u32x4_or_zero(
            scale_row_addr
            + cutlass.Int64(qdata_row_bytes)
            + cutlass.Int64(scale_load_col),
            scale_row_addr != cutlass.Int64(0) and scale_load_col < scale_cols,
        )
    # Static q-word row stride: an Int32 row would wrap row * q_word_cols past 2**31.
    mRowQWords = cute.domain_offset((cutlass.Int64(row_start), 0), mRowQWords)
    for linear_qword_vec_base in cutlass.range(
        group_lane,
        cutlass.Int32(qword_vec_count),
        cutlass.Int32(group_threads * qdata_prefetch),
        unroll=1,
    ):
        local_rows: list = []
        qword_cols: list = []
        source_ptrs: list = []
        output_ptrs: list = []
        input_valid: list = []
        linear_valids: list = []
        bundle_valid = cutlass.Boolean(True)
        for prefetch_idx in cutlass.range_constexpr(qdata_prefetch):
            linear_qword_vec = linear_qword_vec_base + cutlass.Int32(
                prefetch_idx * group_threads
            )
            linear_valid = linear_qword_vec < cutlass.Int32(qword_vec_count)
            local_row = linear_qword_vec // cutlass.Int32(qword_vecs_per_work_tile)
            local_qword_vec = linear_qword_vec - local_row * cutlass.Int32(
                qword_vecs_per_work_tile
            )
            qword_col = scale_col_start * cutlass.Int32(
                q_words_per_scale
            ) + local_qword_vec * cutlass.Int32(4)
            row_addr = cutlass.Int64(0)
            if linear_valid:
                row_addr = cute.arch.load(
                    gather_ptrs + local_row,
                    cutlass.Int64,
                    ss="cta",
                )
            valid = (
                linear_valid
                and row_addr != cutlass.Int64(0)
                and qword_col + 4 <= q_word_cols
            )
            source_ptrs.append(
                cute.make_ptr(
                    cutlass.Uint32,
                    row_addr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                + qword_col
            )
            output_idx = cute.crd2idx((local_row, qword_col), mRowQWords.layout)
            output_ptrs.append(mRowQWords.iterator + output_idx)
            local_rows.append(local_row)
            qword_cols.append(qword_col)
            input_valid.append(valid)
            linear_valids.append(linear_valid)
            bundle_valid &= valid

        if bundle_valid:
            values: list = []
            for prefetch_idx in cutlass.range_constexpr(qdata_prefetch):
                values.append(
                    cute.arch.load(
                        source_ptrs[prefetch_idx],
                        _make_u32x4_vector_type(),
                        cop="cg",
                    )
                )
            for prefetch_idx in cutlass.range_constexpr(qdata_prefetch):
                cute.arch.store(
                    output_ptrs[prefetch_idx],
                    values[prefetch_idx],
                )
        elif cutlass.const_expr(is_mxfp8):
            # Same bytes as below: data where valid, zeros on padded rows.
            values: list = []
            for prefetch_idx in cutlass.range_constexpr(qdata_prefetch):
                values.append(
                    _load_u32x4_or_zero(
                        source_ptrs[prefetch_idx].toint(), input_valid[prefetch_idx]
                    )
                )
            for prefetch_idx in cutlass.range_constexpr(qdata_prefetch):
                if (
                    linear_valids[prefetch_idx]
                    and qword_cols[prefetch_idx] < q_word_cols
                ):
                    cute.arch.store(output_ptrs[prefetch_idx], values[prefetch_idx])
        else:
            for prefetch_idx in cutlass.range_constexpr(qdata_prefetch):
                qword_col = qword_cols[prefetch_idx]
                if input_valid[prefetch_idx]:
                    value = cute.arch.load(
                        source_ptrs[prefetch_idx],
                        _make_u32x4_vector_type(),
                        cop="cg",
                    )
                    cute.arch.store(output_ptrs[prefetch_idx], value)
                elif linear_valids[prefetch_idx] and qword_col < q_word_cols:
                    local_row = local_rows[prefetch_idx]
                    for value_idx in cutlass.range_constexpr(4):
                        mRowQWords[
                            local_row,
                            qword_col + cutlass.Int32(value_idx),
                        ] = cutlass.Uint32(0)

    if cutlass.const_expr(load_row_scales):
        if is_scale_lane:
            for word in cutlass.range_constexpr(
                scale_cols_per_load // CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
            ):
                scale_col = scale_load_col + cutlass.Int32(
                    word * CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
                )
                if scale_col < scale_cols:
                    scale_offset = _cublas_blockscaled_qscale_offset(
                        row_scale_start + scale_row,
                        scale_col,
                        scale_col_blocks,
                    )
                    scale_ptr = cute.recast_ptr(
                        mRowScale.iterator + scale_offset,
                        dtype=cutlass.Uint32,
                    )
                    scale_word = vector.extract(
                        scale_words, dynamic_position=[], static_position=[word]
                    )
                    cute.arch.store(scale_ptr, cutlass.Uint32(scale_word))
    else:
        vector_scale_copy: cutlass.Constexpr[bool] = VECTOR_DISPATCH_SCALE_COPY
        scale_cols_per_copy: cutlass.Constexpr[int] = (
            CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM if vector_scale_copy else 1
        )
        scale_value_dtype = cutlass.Uint32 if vector_scale_copy else cutlass.Uint8
        assert DISPATCH_QUANT_SCALE_COLS_PER_TILE % scale_cols_per_copy == 0
        assert scale_cols_per_work_tile % scale_cols_per_copy == 0
        scale_copies_per_work_tile: cutlass.Constexpr[int] = (
            scale_cols_per_work_tile // scale_cols_per_copy
        )
        scale_copies: cutlass.Constexpr[int] = sf_vec_size * scale_copies_per_work_tile
        for linear_scale_copy in cutlass.range(
            group_lane,
            cutlass.Int32(scale_copies),
            cutlass.Int32(group_threads),
            unroll=1,
        ):
            local_row = linear_scale_copy // cutlass.Int32(scale_copies_per_work_tile)
            local_scale_copy = linear_scale_copy - local_row * cutlass.Int32(
                scale_copies_per_work_tile
            )
            scale_col = scale_col_start + local_scale_copy * cutlass.Int32(
                scale_cols_per_copy
            )
            value = scale_value_dtype(0)
            row_addr = cute.arch.load(
                gather_ptrs + local_row,
                cutlass.Int64,
                ss="cta",
            )
            if row_addr != cutlass.Int64(0) and scale_col < scale_cols:
                source = cute.make_ptr(
                    scale_value_dtype,
                    row_addr
                    + cutlass.Int64(qdata_row_bytes)
                    + cutlass.Int64(scale_col),
                    cute.AddressSpace.gmem,
                    assumed_align=scale_cols_per_copy,
                )
                value = cute.arch.load(source, scale_value_dtype)
            if scale_col < scale_cols:
                scale_offset = _cublas_blockscaled_qscale_offset(
                    row_scale_start + local_row,
                    scale_col,
                    scale_col_blocks,
                )
                scale_ptr = cute.recast_ptr(
                    mRowScale.iterator + scale_offset,
                    dtype=scale_value_dtype,
                )
                cute.arch.store(scale_ptr, value)
    if cutlass.const_expr(use_global_scale_inv):
        if scale_col_start == 0 and group_lane < cutlass.Int32(sf_vec_size):
            value = cutlass.Float32(1.0)
            row_addr = cute.arch.load(
                gather_ptrs + group_lane,
                cutlass.Int64,
                ss="cta",
            )
            if row_addr != cutlass.Int64(0):
                source = cute.make_ptr(
                    cutlass.Float32,
                    row_addr
                    + cutlass.Int64(qdata_row_bytes)
                    + cutlass.Int64(scale_cols),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                value = cute.arch.load(source, cutlass.Float32, cop="cg")
            gsi_row = row_start
            if cutlass.const_expr(gsi_row_start is not None):
                # Ring mode relocates the quantized rows, but the per-row
                # global scales are consumed at logical rows.
                gsi_row = gsi_row_start
            mRowGlobalScaleInv[gsi_row + group_lane] = value


@cute.jit
def _dispatch_quantize_tile(  # noqa: C901
    params: DispatchQuantParams,
    warp_lane: cutlass.Int32,
    mGatherPtrs: cute.Tensor,
    gather_ptr_smem_ptr,
    mRowQWords: cute.Tensor,
    mRowScale: cute.Tensor,
    mColQWords: cute.Tensor,
    mColScale: cute.Tensor,
    row_start: cutlass.Int32,
    row_scale_start: cutlass.Int32,
    row_block: cutlass.Int32,
    col_block: cutlass.Int32,
    row_blocks: cutlass.Int32,
    K: cutlass.Int32,
    row_quant: cutlass.Constexpr[bool] = True,
    col_quant: cutlass.Constexpr[bool] = True,
) -> None:
    source_dtype = params.dispatch_quant_source_dtype
    if cutlass.const_expr(
        params.sf_vec_size != params.DISPATCH_QUANT_ELEMS_PER_SCALE_COL
    ):
        raise ValueError("dispatch quant lane layout must cover one scale column")
    if cutlass.const_expr(params.DISPATCH_QUANT_ELEMS_PER_LANE % 2 != 0):
        raise ValueError("dispatch quant packed shuffle requires even lane elems")
    if cutlass.const_expr(
        params.DISPATCH_QUANT_ELEMS_PER_LANE
        % params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD
        != 0
    ):
        raise ValueError("dispatch quant lane elems must divide qdata words")
    if cutlass.const_expr(
        params.sf_vec_size % params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD != 0
    ):
        raise ValueError("dispatch quant sf_vec_size must divide qdata words")
    if cutlass.const_expr(
        mRowScale.element_type not in (cutlass.Uint8, cutlass.Int8)
        or mColScale.element_type not in (cutlass.Uint8, cutlass.Int8)
    ):
        raise TypeError(
            "dispatch quant qscale stores require byte tensors; "
            f"got row={mRowScale.element_type}, col={mColScale.element_type}"
        )
    col_lanes = cutlass.Int32(params.DISPATCH_QUANT_COL_LANES)
    row_lane = warp_lane // col_lanes
    col_lane_block = warp_lane - row_lane * col_lanes
    col_lane_pair = col_lane_block // cutlass.Int32(
        params.DISPATCH_QUANT_COL_BLOCKS_PER_SCALE
    )
    col_scale_cols = K // cutlass.Int32(params.sf_vec_size)
    row_scale_n_col_blocks = ceil_div(
        col_scale_cols,
        cutlass.Int32(CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM),
    )
    col_tile_elem_offset = col_block * cutlass.Int32(params.sf_vec_size)
    col_lane_elem_offset = col_lane_block * cutlass.Int32(
        params.DISPATCH_QUANT_ELEMS_PER_LANE
    )
    col_start = col_tile_elem_offset + col_lane_elem_offset
    col_q_word = col_start // cutlass.Int32(params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD)

    row_vals_packed = cute.make_rmem_tensor(
        (
            params.DISPATCH_QUANT_ROW_REPS,
            params.DISPATCH_QUANT_ELEMS_PER_LANE // 2,
        ),
        cutlass.Int32,
    )

    # Gather source rows for this warp's 32x64 tile.
    for rep in cutlass.range_constexpr(params.DISPATCH_QUANT_ROW_REPS):
        row_rep_offset = cutlass.Int32(rep * params.DISPATCH_QUANT_ROW_LANES)
        row = row_start + row_lane + row_rep_offset
        if cutlass.const_expr(gather_ptr_smem_ptr is not None):
            row_addr = cute.arch.load(
                gather_ptr_smem_ptr + row_lane + row_rep_offset,
                cutlass.Int64,
                ss="cta",
            )
        else:
            row_addr = mGatherPtrs[row]
        row_loaded_packed = cute.make_rmem_tensor(
            params.DISPATCH_QUANT_ELEMS_PER_LANE // 2,
            cutlass.Int32,
        )
        _load_gather_row_as_b16x2(
            row_addr,
            col_start,
            K,
            row_loaded_packed,
            params.DISPATCH_QUANT_ELEMS_PER_LANE,
            source_dtype,
        )
        for pair in cutlass.range_constexpr(params.DISPATCH_QUANT_ELEMS_PER_LANE // 2):
            row_vals_packed[rep, pair] = row_loaded_packed[pair]

    row_scale_group_first_col = col_lane_pair * cutlass.Int32(
        params.DISPATCH_QUANT_COL_BLOCKS_PER_SCALE
    )
    row_scale_lane = (col_lane_block - row_scale_group_first_col) == cutlass.Int32(0)
    row_scale_offset_base = cutlass.Int32(0)
    if row_scale_lane:
        row_scale_col = col_block + col_lane_pair
        row_scale_offset_base = _cublas_blockscaled_qscale_offset(
            row_scale_start + row_lane,
            row_scale_col,
            row_scale_n_col_blocks,
        )

    row_recips = cute.make_rmem_tensor(
        params.DISPATCH_QUANT_ROW_REPS,
        cutlass.Float32,
    )
    for rep in cutlass.range_constexpr(params.DISPATCH_QUANT_ROW_REPS):
        if cutlass.const_expr(row_quant):
            row_amax_packed = cutlass.Int32(0)
            for pair in cutlass.range_constexpr(
                params.DISPATCH_QUANT_ELEMS_PER_LANE // 2
            ):
                row_amax_packed = _max_b16x2(
                    row_amax_packed,
                    _compute_amax_nonfinite_b16x2(
                        row_vals_packed[rep, pair],
                        source_dtype,
                    ),
                    source_dtype,
                )
            row_amax0, row_amax1 = _reduce_amax_b16x2(
                row_amax_packed,
                1,
                params.DISPATCH_QUANT_ROW_REDUCE_STAGES,
                source_dtype,
            )
            row_amax = max(row_amax0, row_amax1)
            row_scale_u8, row_recip = _compute_scale_and_recip_from_amax(
                cutlass.Float32(row_amax),
                params.format_id,
                params.half_range_scale,
            )
            row_recips[rep] = row_recip
            if row_scale_lane:
                row_scale_offset = row_scale_offset_base + cutlass.Int32(
                    rep
                    * params.DISPATCH_QUANT_ROW_LANES
                    * CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE
                )
                row_scale_ptr = cute.recast_ptr(
                    mRowScale.iterator + row_scale_offset,
                    dtype=cutlass.Uint8,
                )
                cute.arch.store(row_scale_ptr, row_scale_u8, cop="cs")

    if cutlass.const_expr(col_quant):
        col_recips = cute.make_rmem_tensor(
            params.DISPATCH_QUANT_ELEMS_PER_LANE, cutlass.Float32
        )
        col_scale_n_col_blocks = ceil_div(
            row_blocks,
            cutlass.Int32(CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM),
        )
        col_scale_offset_base = cutlass.Int32(0)
        if row_lane == cutlass.Int32(0):
            col_scale_offset_base = _cublas_blockscaled_qscale_offset(
                col_start,
                row_block,
                col_scale_n_col_blocks,
            )
        for pair in cutlass.range_constexpr(params.DISPATCH_QUANT_ELEMS_PER_LANE // 2):
            col_amax_packed = cutlass.Int32(0)
            for rep in cutlass.range_constexpr(params.DISPATCH_QUANT_ROW_REPS):
                col_amax_packed = _max_b16x2(
                    col_amax_packed,
                    _compute_amax_nonfinite_b16x2(
                        row_vals_packed[rep, pair],
                        source_dtype,
                    ),
                    source_dtype,
                )
            col_amax0, col_amax1 = _reduce_amax_b16x2(
                col_amax_packed,
                params.DISPATCH_QUANT_COL_LANES,
                params.DISPATCH_QUANT_COL_REDUCE_STAGES,
                source_dtype,
            )
            for pair_idx in cutlass.range_constexpr(2):
                col_amax = col_amax0
                if cutlass.const_expr(pair_idx == 1):
                    col_amax = col_amax1
                j: cutlass.Constexpr[int] = 2 * pair + pair_idx
                col_scale_u8, col_recip = _compute_scale_and_recip_from_amax(
                    cutlass.Float32(col_amax),
                    params.format_id,
                    params.half_range_scale,
                )
                col_recips[j] = col_recip
                if row_lane == cutlass.Int32(0):
                    col_scale_offset = col_scale_offset_base + cutlass.Int32(
                        j * CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE
                    )
                    col_scale_ptr = cute.recast_ptr(
                        mColScale.iterator + col_scale_offset,
                        dtype=cutlass.Uint8,
                    )
                    cute.arch.store(col_scale_ptr, col_scale_u8, cop="cs")

    if cutlass.const_expr(not params.is_fp4):
        # Static q-word row stride: an Int32 row would wrap row * K / 4 past 2**31.
        mRowQWords = cute.domain_offset((cutlass.Int64(row_start), 0), mRowQWords)
        mColQWords = cute.domain_offset((cutlass.Int64(row_start), 0), mColQWords)
        row_start = cutlass.Int32(0)
    for rep in cutlass.range_constexpr(params.DISPATCH_QUANT_ROW_REPS):
        row_rep_offset = cutlass.Int32(rep * params.DISPATCH_QUANT_ROW_LANES)
        row = row_start + row_lane + row_rep_offset
        row_lane_in_tile = row_lane + row_rep_offset
        if cutlass.const_expr(row_quant):
            row_q_vals = cute.make_rmem_tensor(
                params.DISPATCH_QUANT_ELEMS_PER_LANE, cutlass.Float32
            )
            for pair in cutlass.range_constexpr(
                params.DISPATCH_QUANT_ELEMS_PER_LANE // 2
            ):
                row_val0, row_val1 = _unpack_b16x2(
                    row_vals_packed[rep, pair],
                    source_dtype,
                )
                row_q_vals[2 * pair] = cutlass.Float32(row_val0) * row_recips[rep]
                row_q_vals[2 * pair + 1] = cutlass.Float32(row_val1) * row_recips[rep]
            _store_dispatch_qwords(
                params,
                mRowQWords,
                row_q_vals,
                row,
                col_q_word,
                row_lane_in_tile,
                False,
            )
        if cutlass.const_expr(col_quant):
            col_q_vals = cute.make_rmem_tensor(
                params.DISPATCH_QUANT_ELEMS_PER_LANE, cutlass.Float32
            )
            for pair in cutlass.range_constexpr(
                params.DISPATCH_QUANT_ELEMS_PER_LANE // 2
            ):
                col_val0, col_val1 = _unpack_b16x2(
                    row_vals_packed[rep, pair],
                    source_dtype,
                )
                col_q_vals[2 * pair] = cutlass.Float32(col_val0) * col_recips[2 * pair]
                col_q_vals[2 * pair + 1] = (
                    cutlass.Float32(col_val1) * col_recips[2 * pair + 1]
                )
            _store_dispatch_qwords(
                params,
                mColQWords,
                col_q_vals,
                row,
                col_q_word,
                row_lane_in_tile,
                True,
            )


@cute.jit
def _dispatch_quant_tma_per_tile_wait(
    act_tile_idx: cutlass.Int32,
    act_tile_start_per_group: cutlass.Int32,
    act_rows: cutlass.Int32,
    mDoneCounter: cute.Tensor,
    K: cutlass.Int32,
    ACT_BLOCK_SIZE: cutlass.Constexpr[int],
    COMBINE_SWIGLU_BWD_MODE: cutlass.Constexpr[int],
    COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    COMBINE_SWIGLU_FWD_MODE: cutlass.Constexpr[int],
    COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    DISPATCH_QUANT_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    format: cutlass.Constexpr,
    mode: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
) -> None:
    tile_row_start = act_tile_idx * cutlass.Int32(ACT_BLOCK_SIZE)
    valid_rows = act_rows - tile_row_start
    if valid_rows > cutlass.Int32(ACT_BLOCK_SIZE):
        valid_rows = cutlass.Int32(ACT_BLOCK_SIZE)
    valid_row_blocks = valid_rows // cutlass.Int32(sf_vec_size)
    col_scale_cols = (
        (K // cutlass.Int32(2)) // cutlass.Int32(sf_vec_size)
        if cutlass.const_expr(mode == COMBINE_SWIGLU_BWD_MODE)
        else K // cutlass.Int32(sf_vec_size)
    )
    scale_cols_per_tile: cutlass.Constexpr[int] = (
        (
            COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE
            if cutlass.const_expr(mode == COMBINE_SWIGLU_BWD_MODE)
            else COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE
        )
        if cutlass.const_expr(
            mode == COMBINE_SWIGLU_FWD_MODE or mode == COMBINE_SWIGLU_BWD_MODE
        )
        else DISPATCH_QUANT_SCALE_COLS_PER_TILE
    )
    col_work_tiles = ceil_div(
        col_scale_cols,
        cutlass.Int32(scale_cols_per_tile),
    )
    valid_quant_tiles = valid_row_blocks * col_work_tiles
    if cutlass.const_expr(format is NVFP4 and mode == COMBINE_SWIGLU_FWD_MODE):
        valid_quant_tiles = valid_rows
    ptr = cute.recast_ptr(
        mDoneCounter.iterator + act_tile_start_per_group + act_tile_idx,
        dtype=cutlass.Uint32,
    )
    _wait_counter_at_least(ptr, cutlass.Uint32(valid_quant_tiles))
    cute.arch.fence_proxy("async.global")


@cute.jit
def _mega_dispatch_quant_signal_group_tile_done(
    mDoneCounter: cute.Tensor,
    row_done_counter_offset: cutlass.Int32,
    col_done_counter_offset: cutlass.Int32,
    act_tile_slot: cutlass.Int32,
    col_done_slot: cutlass.Int32,
    quant_group: cutlass.Int32,
    quant_group_lane: cutlass.Int32,
    DISPATCH_QUANT_GROUP_THREADS: cutlass.Constexpr[int],
    DISPATCH_QUANT_SYNC_BAR: cutlass.Constexpr[int],
    MEGA_FORWARD_MODE: cutlass.Constexpr[int],
    mode: cutlass.Constexpr[int],
    col_tiles_per_work: cutlass.Constexpr[int] = 1,
    col_done_limit=None,
) -> None:
    # Quant stores use the generic proxy while the GEMM consumes them via
    # TMA, so publish to the async proxy before releasing the counters.
    cute.arch.fence_proxy("async.global")
    cute.arch.fence_acq_rel_gpu()
    cute.arch.barrier(
        barrier_id=DISPATCH_QUANT_SYNC_BAR + quant_group,
        number_of_threads=DISPATCH_QUANT_GROUP_THREADS,
    )
    if quant_group_lane == cutlass.Int32(0):
        valid_tiles = cutlass.Int32(col_tiles_per_work)
        if cutlass.const_expr(col_done_limit is not None):
            valid_tiles = min(valid_tiles, col_done_limit - col_done_slot)
        row_counter_ptr = cute.recast_ptr(
            mDoneCounter.iterator + row_done_counter_offset + act_tile_slot,
            dtype=cutlass.Uint32,
        )
        cute.arch.atomic_add(
            row_counter_ptr,
            cutlass.Uint32(valid_tiles),
            sem="release",
            scope="gpu",
        )
        # Forward FC13 consumes only row-major quantization. Column
        # readiness is needed by the wgrad path's feature-tile waits.
        if cutlass.const_expr(mode != MEGA_FORWARD_MODE):
            for tile_delta in cutlass.range_constexpr(col_tiles_per_work):
                if cutlass.Int32(tile_delta) < valid_tiles:
                    col_counter_ptr = cute.recast_ptr(
                        mDoneCounter.iterator
                        + col_done_counter_offset
                        + col_done_slot
                        + cutlass.Int32(tile_delta),
                        dtype=cutlass.Uint32,
                    )
                    cute.arch.atomic_add(
                        col_counter_ptr,
                        cutlass.Uint32(1),
                        sem="release",
                        scope="gpu",
                    )


@cute.jit
def _mega_dispatch_quant_wgrad_col_tile_wait(
    mDoneCounter: cute.Tensor,
    col_done_counter_offset: cutlass.Int32,
    g: cutlass.Int32,
    col_work_tile: cutlass.Int32,
    group_row_blocks: cutlass.Int32,
    col_work_tiles: cutlass.Int32,
) -> None:
    col_done_slot = g * col_work_tiles + col_work_tile
    ptr = cute.recast_ptr(
        mDoneCounter.iterator + col_done_counter_offset + col_done_slot,
        dtype=cutlass.Uint32,
    )
    _wait_counter_at_least(ptr, cutlass.Uint32(group_row_blocks))


@cute.jit
def _mega_dispatch_quant_dgrad_tile_wait(
    mDoneCounter: cute.Tensor,
    row_done_counter_offset: cutlass.Int32,
    act_tile_idx: cutlass.Int32,
    act_tile_start_per_group: cutlass.Int32,
    act_rows: cutlass.Int32,
    K: cutlass.Int32,
    ACT_BLOCK_SIZE: cutlass.Constexpr[int],
    DISPATCH_QUANT_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
) -> None:
    tile_row_start = act_tile_idx * cutlass.Int32(ACT_BLOCK_SIZE)
    valid_rows = act_rows - tile_row_start
    if valid_rows > cutlass.Int32(ACT_BLOCK_SIZE):
        valid_rows = cutlass.Int32(ACT_BLOCK_SIZE)
    valid_row_blocks = valid_rows // cutlass.Int32(sf_vec_size)
    col_scale_cols = K // cutlass.Int32(sf_vec_size)
    col_work_tiles = ceil_div(
        col_scale_cols,
        cutlass.Int32(DISPATCH_QUANT_SCALE_COLS_PER_TILE),
    )
    valid_quant_tiles = valid_row_blocks * col_work_tiles
    ptr = cute.recast_ptr(
        mDoneCounter.iterator
        + row_done_counter_offset
        + act_tile_start_per_group
        + act_tile_idx,
        dtype=cutlass.Uint32,
    )
    _wait_counter_at_least(ptr, cutlass.Uint32(valid_quant_tiles))


@cute.jit
def _mega_dispatch_quant_wgrad_feature_tile_wait(
    mDoneCounter: cute.Tensor,
    col_done_counter_offset: cutlass.Int32,
    g: cutlass.Int32,
    feature_tile_idx: cutlass.Int32,
    feature_tile_elems: cutlass.Constexpr[int],
    feature_size: cutlass.Int32,
    group_row_blocks: cutlass.Int32,
    col_work_tiles: cutlass.Int32,
    DISPATCH_QUANT_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
) -> None:
    first_feature_elem = feature_tile_idx * cutlass.Int32(feature_tile_elems)
    last_feature_elem = first_feature_elem + cutlass.Int32(feature_tile_elems)
    if last_feature_elem > feature_size:
        last_feature_elem = feature_size
    first_scale_col = first_feature_elem // cutlass.Int32(sf_vec_size)
    last_scale_col_excl = ceil_div(
        last_feature_elem,
        cutlass.Int32(sf_vec_size),
    )
    first_col_work_tile = first_scale_col // cutlass.Int32(
        DISPATCH_QUANT_SCALE_COLS_PER_TILE
    )
    last_col_work_tile_excl = ceil_div(
        last_scale_col_excl,
        cutlass.Int32(DISPATCH_QUANT_SCALE_COLS_PER_TILE),
    )
    for col_work_tile in cutlass.range(
        first_col_work_tile,
        last_col_work_tile_excl,
        unroll=1,
    ):
        _mega_dispatch_quant_wgrad_col_tile_wait(
            mDoneCounter=mDoneCounter,
            col_done_counter_offset=col_done_counter_offset,
            g=g,
            col_work_tile=col_work_tile,
            group_row_blocks=group_row_blocks,
            col_work_tiles=col_work_tiles,
        )


@cute.jit
def _mega_wait_quant_ready(
    wait_axis: cutlass.Constexpr[int],
    tile_m_idx: cutlass.Int32,
    tile_n_idx: cutlass.Int32,
    dgrad_act_off_tiles: cutlass.Int32,
    g: cutlass.Int32,
    split_sizes: cute.Tensor,
    m_size: cutlass.Int32,
    n_size: cutlass.Int32,
    dispatch_dim: cutlass.Int32,
    mDoneCounter: cute.Tensor,
    row_done_counter_offset: cutlass.Int32,
    col_done_counter_offset: cutlass.Int32,
    BLOCK_SIZE_M: cutlass.Constexpr[int],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
    DISPATCH_QUANT_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    SWAP_AB: cutlass.Constexpr[bool],
    sf_vec_size: cutlass.Constexpr[int],
) -> None:
    # The dispatched dy is the A operand of both GEMMs, so a tile's quant
    # dependency is its output-M tile (output-N under SWAP_AB). Each GEMM
    # waits on the completion counter for the axis it reads: wait_axis 0
    # consumes the row-major quant -> row counter; wait_axis 1 consumes the
    # col-major quant -> col counter. Both key on the same out-M tile.
    if cutlass.const_expr(SWAP_AB):
        out_m_tile = tile_n_idx
        OUT_M_BLOCK: cutlass.Constexpr[int] = BLOCK_SIZE_N
        out_m_size = n_size
    else:
        out_m_tile = tile_m_idx
        OUT_M_BLOCK: cutlass.Constexpr[int] = BLOCK_SIZE_M
        out_m_size = m_size
    if cutlass.const_expr(wait_axis == 0):
        _mega_dispatch_quant_dgrad_tile_wait(
            mDoneCounter=mDoneCounter,
            row_done_counter_offset=row_done_counter_offset,
            act_tile_idx=out_m_tile,
            act_tile_start_per_group=dgrad_act_off_tiles,
            act_rows=out_m_size,
            K=dispatch_dim,
            ACT_BLOCK_SIZE=OUT_M_BLOCK,
            DISPATCH_QUANT_SCALE_COLS_PER_TILE=DISPATCH_QUANT_SCALE_COLS_PER_TILE,
            sf_vec_size=sf_vec_size,
        )
    else:
        group_row_blocks = cutlass.Int32(split_sizes[g]) // cutlass.Int32(sf_vec_size)
        col_scale_cols = dispatch_dim // cutlass.Int32(sf_vec_size)
        col_work_tiles = ceil_div(
            col_scale_cols,
            cutlass.Int32(DISPATCH_QUANT_SCALE_COLS_PER_TILE),
        )
        _mega_dispatch_quant_wgrad_feature_tile_wait(
            mDoneCounter=mDoneCounter,
            col_done_counter_offset=col_done_counter_offset,
            g=g,
            feature_tile_idx=out_m_tile,
            feature_tile_elems=OUT_M_BLOCK,
            feature_size=dispatch_dim,
            group_row_blocks=group_row_blocks,
            col_work_tiles=col_work_tiles,
            DISPATCH_QUANT_SCALE_COLS_PER_TILE=DISPATCH_QUANT_SCALE_COLS_PER_TILE,
            sf_vec_size=sf_vec_size,
        )


@cute.jit
def _mega_activation_wgrad_tile_wait(
    mDoneCounter: cute.Tensor,
    col_done_counter_offset: cutlass.Int32,
    g: cutlass.Int32,
    feature_tile_idx: cutlass.Int32,
    feature_tile_elems: cutlass.Constexpr[int],
    activation_dim: cutlass.Int32,
    group_row_blocks: cutlass.Int32,
    DISPATCH_QUANT_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
) -> None:
    first_scale_col = (
        feature_tile_idx * cutlass.Int32(feature_tile_elems)
    ) // cutlass.Int32(sf_vec_size)
    last_scale_col_excl = ceil_div(
        (feature_tile_idx + cutlass.Int32(1)) * cutlass.Int32(feature_tile_elems),
        cutlass.Int32(sf_vec_size),
    )
    col_work_tiles = ceil_div(
        activation_dim // cutlass.Int32(sf_vec_size),
        cutlass.Int32(DISPATCH_QUANT_SCALE_COLS_PER_TILE),
    )
    first_work_tile = first_scale_col // cutlass.Int32(
        DISPATCH_QUANT_SCALE_COLS_PER_TILE
    )
    last_work_tile_excl = ceil_div(
        last_scale_col_excl,
        cutlass.Int32(DISPATCH_QUANT_SCALE_COLS_PER_TILE),
    )
    for col_work_tile in cutlass.range(
        first_work_tile,
        last_work_tile_excl,
        unroll=1,
    ):
        _mega_dispatch_quant_wgrad_col_tile_wait(
            mDoneCounter=mDoneCounter,
            col_done_counter_offset=col_done_counter_offset,
            g=g,
            col_work_tile=col_work_tile,
            group_row_blocks=group_row_blocks,
            col_work_tiles=col_work_tiles,
        )


# Each gather row sub-tile splits its K range into this many work items so
# peer-bound gathers spread across CTAs. Measured at 16- and 32-row
# sub-tiles: 2 items leave the 16-row gather under-spread, and 8 lose to the
# per-item claim/fence/signal overhead.
_GATHER_K_SEGMENTS: int = 4
# One item claims every K segment of its row sub-tile once the sub-tiles
# fill this many waves of gather WGs. At one wave, T1024 lost 2-7% on
# EP16-EP64 to the coarser spread.
_GATHER_CLAIM_ALL_MIN_WAVES: int = 8
# The split only pays while the GEMM out-computes its gather, i.e. for wide
# enough outputs per gathered row. Measured on GB300 bf16: the split cost
# 3-7% at N=1792 (16-19% on a skew-hot rank) and 4% at N=2048, and saved
# 2% at N=2560 and 0.4-1.2% at N=3072-4096.
_GATHER_K_SPLIT_MIN_OUT_FEATURES: int = 2560


def _gather_k_segments(out_features: int) -> int:
    """K segments per gather row sub-tile of a GEMM that produces
    `out_features` outputs per gathered row."""
    if out_features < _GATHER_K_SPLIT_MIN_OUT_FEATURES:
        return 1
    return _GATHER_K_SEGMENTS


def _gather_chunk_prefetch(
    rows_per_warp: int, k_segments: int, loads_per_lane: int = 32
) -> int:
    """K chunks a row gather loads per round with `rows_per_warp` rows per
    warp and `k_segments` K segments per row sub-tile; narrow sub-tiles
    batch chunks up to `loads_per_lane` loads in flight per lane."""
    row_prefetch = min(rows_per_warp, 8)
    if row_prefetch < 8:
        return loads_per_lane // row_prefetch
    # A second in-flight K-chunk only pays while the gather bounds the GEMM.
    # K-split (wide-output) GEMMs out-compute their gather, and the deeper
    # batch steals from the overlapped MMA instead: measured on GB300 bf16
    # N=3584 (skew-hot rank, 28672 rows per expert), tensor-active fell from
    # 95% to 88% during the gather and the GEMM ran 4.5% longer.
    return 1 if k_segments > 1 else 2


def _gather_k_segment_columns(
    K,
    AG_BLOCK_SIZE_M: int,
    AG_BLOCK_SIZE_K: int,
    K_SEGMENTS: int,
    NUM_GATHER_WARPS: int,
):
    """Columns per K segment of a row gather. Segments are whole load
    rounds, so the last one may be short or empty."""
    round_cols = (
        _gather_chunk_prefetch(AG_BLOCK_SIZE_M // NUM_GATHER_WARPS, K_SEGMENTS)
        * AG_BLOCK_SIZE_K
    )
    num_rounds = (K + round_cols - 1) // round_cols
    return (num_rounds + K_SEGMENTS - 1) // K_SEGMENTS * round_cols


@cute.jit
def _gather_k_segment_tma_wait(
    kernel,
    tile_m_idx: cutlass.Int32,
    tile_m_start_per_group: cutlass.Int32,
    a_buff_counter_addr_i64: cutlass.Int64,
    kk: cutlass.Int32,
    num_k_tiles: cutlass.Int32,
    K: cutlass.Int32,
    NUM_SUB_PER_GEMM_M: cutlass.Constexpr[int],
) -> cutlass.Int32:
    """`_tma_k_segment_wait` of a kernel whose gather WG runs
    `_dispatch_gather_producer_body`: spins (warp-wide) until the K segments
    the M-tile's gather has signalled cover k-tile `kk`, and returns how many
    leading k-tiles they cover."""
    K_SEGMENTS = kernel.GATHER_K_SEGMENTS
    NUM_GATHER_WARPS = kernel.GATHER_WG_THREADS // 32
    seg_cols = _gather_k_segment_columns(
        K, kernel.AG_BLOCK_SIZE_M, kernel.AG_BLOCK_SIZE_K, K_SEGMENTS, NUM_GATHER_WARPS
    )
    ptr = cute.make_ptr(
        cutlass.Uint32,
        a_buff_counter_addr_i64
        + cutlass.Int64(tile_m_start_per_group + tile_m_idx) * cutlass.Int64(4),
        cute.AddressSpace.gmem,
    )
    cute.arch.sync_warp()
    ready_k = cutlass.Int32(0)
    while ready_k <= kk:
        v = cute.arch.load(ptr, cutlass.Uint32, sem="acquire", scope="gpu")
        # Segment items may finish out of order; only a ready prefix counts.
        ready_segs = cutlass.Int32(0)
        for s in cutlass.range_constexpr(K_SEGMENTS):
            seg_count = (v >> cutlass.Uint32(8 * s)) & cutlass.Uint32(0xFF)
            if ready_segs == cutlass.Int32(s) and seg_count >= cutlass.Uint32(
                NUM_SUB_PER_GEMM_M * NUM_GATHER_WARPS
            ):
                ready_segs = cutlass.Int32(s + 1)
        ready_k = ready_segs * seg_cols // cutlass.Int32(kernel.BLOCK_SIZE_K)
        if ready_segs == cutlass.Int32(K_SEGMENTS):
            ready_k = num_k_tiles
    return cute.arch.shuffle_sync(ready_k, cutlass.Int32(0))


@cute.jit
def _dispatch_gather_producer_body(  # noqa: C901
    gather_counter_addr_i64: cutlass.Int64,
    a_buff_counter_addr_i64: cutlass.Int64,  # global "this CTA's gather done" counter
    gather_a_ptrs_addr_i64: cutlass.Int64,
    padding_row_addr_i64: cutlass.Int64,
    a_local_addr_i64: cutlass.Int64,
    stride_am_elems: cutlass.Int32,
    gather_tile_idx_smem_ptr: cute.Pointer,
    split_sizes: cute.Tensor,
    elem_size_bytes_a: cutlass.Constexpr[int],
    G: cutlass.Constexpr[int],
    K: cutlass.Int32,
    local_rank: cutlass.Int32,
    WORLD_SIZE: cutlass.Constexpr[int],
    AG_BLOCK_SIZE_M: cutlass.Constexpr[int],
    AG_BLOCK_SIZE_K: cutlass.Constexpr[int],
    BLOCK_SIZE_M: cutlass.Constexpr[int],
    has_padding_sentinels: cutlass.Constexpr[bool],
    _BAR_GATHER_WG_INTERNAL: cutlass.Constexpr[int],
    _DIST_GATHER_WG_THREADS: cutlass.Constexpr[int],
    SEGMENT_SIGNALS: cutlass.Constexpr[bool] = True,
    LOADS_PER_LANE: cutlass.Constexpr[int] = 32,
    gather_columns: cutlass.Constexpr[bool] = False,
    K_SEGMENTS: cutlass.Constexpr[int] = 1,
    M_INTERLEAVE: cutlass.Constexpr[int] = 0,
) -> None:
    """Gather producer body (GATHER WG, 128 threads): pulls A from
    peer GPUs via per-row pointer indirection, signals per-M-tile
    completion to the TMA producer.

    `K_SEGMENTS > 1` splits each row sub-tile's K range into that many
    segments and counts them per segment: byte `s` of the M-tile counter
    collects one increment per gather warp and row sub-tile once segment
    `s` has landed, so the consumer can start on a segment as soon as its
    byte reaches `BLOCK_SIZE_M // AG_BLOCK_SIZE_M * NUM_GATHER_WARPS`. A
    work item covers every segment of its sub-tile, signalling each as it
    finishes, once the sub-tiles holding rows fill
    `_GATHER_CLAIM_ALL_MIN_WAVES` waves of gather WGs; smaller gathers
    claim one segment per item to spread across more WGs.
    Without `SEGMENT_SIGNALS` (consumers that wait for whole tiles) every
    item covers one segment and signals once, at its end."""
    assert not (gather_columns and K_SEGMENTS > 1), "column gather has no K split"
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % cutlass.Int32(128)
    gemm_m_idx = cutlass.Int32(0)
    sub_offset = cutlass.Int32(0)

    NUM_SUB_PER_GEMM_M: cutlass.Constexpr[int] = BLOCK_SIZE_M // AG_BLOCK_SIZE_M
    NUM_GATHER_WARPS: cutlass.Constexpr[int] = _DIST_GATHER_WG_THREADS // 32
    claim_segments = cutlass.Int32(1)
    claim_step = cutlass.Uint32(1)
    if cutlass.const_expr(K_SEGMENTS > 1):
        assert K_SEGMENTS <= 4 and NUM_SUB_PER_GEMM_M * NUM_GATHER_WARPS < 256, (
            "segment counts must fit one byte of the M-tile counter each"
        )
    if cutlass.const_expr(K_SEGMENTS > 1 and SEGMENT_SIGNALS):
        num_row_subtiles = cutlass.Int32(0)
        for g in cutlass.range(G, unroll=1):
            num_row_subtiles += (
                cutlass.Int32(split_sizes[g]) + cutlass.Int32(AG_BLOCK_SIZE_M - 1)
            ) // cutlass.Int32(AG_BLOCK_SIZE_M)
        grid_dim = cute.arch.grid_dim()
        if num_row_subtiles >= cutlass.Int32(
            _GATHER_CLAIM_ALL_MIN_WAVES * grid_dim[0] * grid_dim[1] * grid_dim[2]
        ):
            claim_segments = cutlass.Int32(K_SEGMENTS)
        claim_step = cutlass.Uint32(claim_segments)

    # Initial fetch: lane 0 atomic-fetch-add on the global counter,
    # broadcast via SMEM scratch + WG barrier.
    if lane == cutlass.Int32(0):
        tile_idx_raw = cute.arch.atomic_add(
            cute.make_ptr(
                cutlass.Uint32, gather_counter_addr_i64, cute.AddressSpace.gmem
            ),
            claim_step,
            sem="release",
            scope="gpu",
        )
        cute.arch.store(
            gather_tile_idx_smem_ptr,
            cutlass.Int32(tile_idx_raw),
            ss="cta",
        )
    cute.arch.barrier(
        barrier_id=_BAR_GATHER_WG_INTERNAL,
        number_of_threads=_DIST_GATHER_WG_THREADS,
    )
    tile_idx = cute.arch.load(
        gather_tile_idx_smem_ptr + cutlass.Int32(0), cutlass.Int32, ss="cta"
    )
    visitor = GroupedProblemVisitor.create(
        split_sizes,
        cutlass.Int32(0),
        K if gather_columns else cutlass.Int32(1),
        K,
        G,
        _FPROP,
        BLOCK_SIZE_M,
        AG_BLOCK_SIZE_K if gather_columns else 1,
        AG_BLOCK_SIZE_K,
        gather_columns,
        1,
        local_rank,
        WORLD_SIZE,
        M_SUBTILES=NUM_SUB_PER_GEMM_M * K_SEGMENTS,
        M_INTERLEAVE=M_INTERLEAVE,
    )
    work = visitor.get_work(tile_idx)

    while work.is_valid_tile:
        m_size = work.m_size
        start_am = work.split_prefix
        tile_m_start_per_group = work.m_tile_prefix

        if work.is_valid_tile:
            gemm_m_idx = work.tile_m_idx
            sub_offset = work.subtile_idx // cutlass.Int32(K_SEGMENTS)
            k_seg = work.subtile_idx % cutlass.Int32(K_SEGMENTS)
            feature_tile_idx = work.tile_n_idx
            if cutlass.const_expr(gather_columns):
                row_tiles = (m_size + AG_BLOCK_SIZE_M - 1) // AG_BLOCK_SIZE_M
                group_tile_idx = tile_idx - visitor.tile_start
                feature_tile_idx = group_tile_idx // row_tiles
                gemm_m_idx = group_tile_idx % row_tiles

            base_m = gemm_m_idx * cutlass.Int32(
                BLOCK_SIZE_M
            ) + sub_offset * cutlass.Int32(AG_BLOCK_SIZE_M)

            num_k_chunks = (K + AG_BLOCK_SIZE_K - 1) // AG_BLOCK_SIZE_K
            seg_signals = cutlass.Uint32(0)
            seg_counter_addr = a_buff_counter_addr_i64 + cutlass.Int64(
                tile_m_start_per_group + gemm_m_idx
            ) * cutlass.Int64(4)

            # Rows beyond m_size are masked off; we still signal at
            # the end so the consumer observes NUM_SUB_PER_GEMM_M
            # increments per parent GEMM M-tile.
            if base_m < m_size:
                # Row-parallel decomposition: 4 warps × 32 lanes = 128
                # threads. Warp w handles ROWS_PER_WARP = AG_BLOCK_M/4
                # rows; within a warp each lane owns one 16-byte
                # col-vec across all rows (one peer ptr per row, reused
                # across k_chunks).
                ELEMS_PER_VEC: cutlass.Constexpr[int] = 4
                ROWS_PER_WARP: cutlass.Constexpr[int] = (
                    AG_BLOCK_SIZE_M // NUM_GATHER_WARPS
                )
                warp_id_in_wg = lane // cutlass.Int32(32)
                lane_in_warp = lane % cutlass.Int32(32)
                NUM_VECS_PER_ROW: cutlass.Constexpr[int] = (
                    AG_BLOCK_SIZE_K // 2
                ) // ELEMS_PER_VEC
                assert NUM_VECS_PER_ROW == 32, (
                    "row-parallel gather assumes 32 col-vecs/row, got "
                    f"{NUM_VECS_PER_ROW}"
                )
                col_b32 = lane_in_warp * cutlass.Int32(ELEMS_PER_VEC)
                row_bundle_start = warp_id_in_wg * cutlass.Int32(ROWS_PER_WARP)
                # Hoist per-row peer-ptr loads out of the k_chunk loop
                # (invariant). Loads are unconditional so all entries
                # of `peer_bases` share the same cute SSA type; the
                # STG is gated by `row_in_bounds[r]`.
                #
                # CORRECTNESS: clamp `safe_row` to [0, m_size-1] before
                # using it as a `gather_a_ptrs` index. Without the
                # clamp, last-group tail sub-tiles can speculatively
                # index past the table's `max_recv` size and segfault.
                # `cutlass.max(m_size, 1)` handles m_size == 0 (empty
                # group) where m_size - 1 would underflow to -1; for
                # empty groups the loaded value is never dereferenced.
                m_size_pos = cutlass.max(m_size, cutlass.Int32(1))
                peer_bases: list = []
                dst_row_bases: list = []
                row_in_bounds: list = []
                for r in cutlass.range_constexpr(ROWS_PER_WARP):
                    global_row = base_m + row_bundle_start + cutlass.Int32(r)
                    row_in_bounds.append(global_row < m_size)
                    safe_row = cutlass.min(global_row, m_size_pos - cutlass.Int32(1))
                    peer_ptr_addr = gather_a_ptrs_addr_i64 + cutlass.Int64(
                        start_am + safe_row
                    ) * cutlass.Int64(8)
                    peer_base = cute.arch.load(
                        cute.make_ptr(
                            cutlass.Int64,
                            peer_ptr_addr,
                            cute.AddressSpace.gmem,
                        ),
                        cutlass.Int64,
                    )
                    if cutlass.const_expr(has_padding_sentinels):
                        if peer_base == cutlass.Int64(0):
                            peer_base = padding_row_addr_i64
                    peer_bases.append(peer_base)
                    dst_row_bases.append(
                        a_local_addr_i64
                        + cutlass.Int64(start_am + global_row)
                        * cutlass.Int64(stride_am_elems)
                        * cutlass.Int64(elem_size_bytes_a)
                    )
                # Batch loads ahead of stores so the scheduler can
                # overlap multiple remote NVLink loads before the
                # dependent local stores. Immediate LDG->STG pairs
                # otherwise dominate long-scoreboard samples (PC
                # sampling).
                ROW_PREFETCH: cutlass.Constexpr[int] = (
                    8 if ROWS_PER_WARP >= 8 else ROWS_PER_WARP
                )
                row_bundle_in_bounds = (
                    base_m + row_bundle_start + cutlass.Int32(ROWS_PER_WARP)
                ) <= m_size
                # A warp whose rows all lie past m_size copies nothing but
                # still runs the K loop: its segment signals are counted.
                row_bundle_any = (base_m + row_bundle_start) < m_size
                # Per-lane 8-bf16-element vector covers cols
                # [col_b16, col_b16 + 8). Predicate on K so the partial
                # last k_chunk masks OOB lanes (matches TLX's `mask_ak`
                # in `_gather_tile_inner`). Upstream
                # `_validate_cute_vector_width` already guarantees
                # K % 8 == 0, so per-vector granularity is sufficient.
                ELEMS_PER_VEC_BF16: cutlass.Constexpr[int] = 8
                # With few rows per warp (small gather sub-tiles, i.e. the
                # decode regime), row batching alone leaves only
                # ROWS_PER_WARP peer-NVLink loads in flight per lane and the
                # k_chunk loop becomes a serial latency chain along K
                # (~1.9 us per 256-element chunk: fc13 gather cost measured
                # 7.5 us per 1024 rows of K). Batch loads across k_chunks as
                # well, targeting 32 loads in flight. Wide sub-tiles
                # (ROW_PREFETCH == 8, the prefill shapes) batch two
                # k_chunks (16 loads in flight per lane) unless K-split.
                CHUNK_PREFETCH: cutlass.Constexpr[int] = (
                    1
                    if gather_columns
                    else _gather_chunk_prefetch(
                        ROWS_PER_WARP, K_SEGMENTS, LOADS_PER_LANE
                    )
                )
                # K segments are whole CHUNK_PREFETCH batches, so every
                # batch but the tensor's last stays on the batched path.
                num_k_batches = (
                    num_k_chunks + cutlass.Int32(CHUNK_PREFETCH - 1)
                ) // cutlass.Int32(CHUNK_PREFETCH)
                seg_batches = (
                    num_k_batches + cutlass.Int32(K_SEGMENTS - 1)
                ) // cutlass.Int32(K_SEGMENTS)
                item_batches = seg_batches * claim_segments
                num_k_outer = cutlass.max(
                    cutlass.min(item_batches, num_k_batches - k_seg * seg_batches),
                    cutlass.Int32(0),
                )
                if cutlass.const_expr(gather_columns):
                    num_k_outer = cutlass.Int32(1)
                for k_outer in cutlass.range(num_k_outer, unroll=1):
                    k_chunk0 = (k_seg * seg_batches + k_outer) * cutlass.Int32(
                        CHUNK_PREFETCH
                    )
                    if cutlass.const_expr(gather_columns):
                        k_chunk0 = feature_tile_idx
                    batch_col_end = (
                        col_b32 * cutlass.Int32(2)
                        + (k_chunk0 + cutlass.Int32(CHUNK_PREFETCH - 1))
                        * cutlass.Int32(AG_BLOCK_SIZE_K)
                        + cutlass.Int32(ELEMS_PER_VEC_BF16)
                    )
                    # Partial bundles batch too: the per-chunk path below
                    # pays one NVLink round trip per K chunk, which bounds
                    # decode-sized groups.
                    if (
                        cutlass.const_expr(CHUNK_PREFETCH > 1)
                        and (batch_col_end <= K)
                        and row_bundle_any
                    ):
                        chunk_vecs: list = []
                        for cc in cutlass.range_constexpr(CHUNK_PREFETCH):
                            cc_offset_bytes = cutlass.Int64(
                                col_b32 * cutlass.Int32(2)
                                + (k_chunk0 + cutlass.Int32(cc))
                                * cutlass.Int32(AG_BLOCK_SIZE_K)
                            ) * cutlass.Int64(elem_size_bytes_a)
                            row_vecs: list = []
                            for r in cutlass.range_constexpr(ROWS_PER_WARP):
                                src_addr = peer_bases[r] + cc_offset_bytes
                                row_vecs.append(
                                    cute.arch.load(
                                        cute.make_ptr(
                                            cutlass.Uint32,
                                            src_addr,
                                            cute.AddressSpace.gmem,
                                        ),
                                        _make_u32x4_vector_type(),
                                        cop="cg",
                                    )
                                )
                            chunk_vecs.append(row_vecs)
                        for cc in cutlass.range_constexpr(CHUNK_PREFETCH):
                            cc_offset_bytes = cutlass.Int64(
                                col_b32 * cutlass.Int32(2)
                                + (k_chunk0 + cutlass.Int32(cc))
                                * cutlass.Int32(AG_BLOCK_SIZE_K)
                            ) * cutlass.Int64(elem_size_bytes_a)
                            for r in cutlass.range_constexpr(ROWS_PER_WARP):
                                if row_in_bounds[r]:
                                    dst_addr = dst_row_bases[r] + cc_offset_bytes
                                    cute.arch.store(
                                        cute.make_ptr(
                                            cutlass.Uint32,
                                            dst_addr,
                                            cute.AddressSpace.gmem,
                                        ),
                                        chunk_vecs[cc][r],
                                        cop="cg",
                                    )
                    else:
                        for cc in cutlass.range_constexpr(CHUNK_PREFETCH):
                            k_chunk = k_chunk0 + cutlass.Int32(cc)
                            col_b16 = col_b32 * cutlass.Int32(
                                2
                            ) + k_chunk * cutlass.Int32(AG_BLOCK_SIZE_K)
                            col_offset_bytes = cutlass.Int64(col_b16) * cutlass.Int64(
                                elem_size_bytes_a
                            )
                            col_in_bounds = (
                                col_b16 + cutlass.Int32(ELEMS_PER_VEC_BF16)
                            ) <= K
                            if col_in_bounds and row_bundle_in_bounds:
                                for r_base in cutlass.range_constexpr(
                                    0, ROWS_PER_WARP, ROW_PREFETCH
                                ):
                                    vecs: list = []
                                    for rr in cutlass.range_constexpr(ROW_PREFETCH):
                                        r: cutlass.Constexpr[int] = r_base + rr
                                        src_addr = peer_bases[r] + col_offset_bytes
                                        vecs.append(
                                            cute.arch.load(
                                                cute.make_ptr(
                                                    cutlass.Uint32,
                                                    src_addr,
                                                    cute.AddressSpace.gmem,
                                                ),
                                                _make_u32x4_vector_type(),
                                                cop="cg",
                                            )
                                        )
                                    for rr in cutlass.range_constexpr(ROW_PREFETCH):
                                        r: cutlass.Constexpr[int] = r_base + rr
                                        dst_addr = dst_row_bases[r] + col_offset_bytes
                                        cute.arch.store(
                                            cute.make_ptr(
                                                cutlass.Uint32,
                                                dst_addr,
                                                cute.AddressSpace.gmem,
                                            ),
                                            vecs[rr],
                                            cop="cg",
                                        )
                            elif col_in_bounds and row_bundle_any:
                                # Rows past m_size reload the clamped
                                # `safe_row` so the bundle's loads still
                                # issue together. A load-store pair per
                                # row costs one NVLink round trip per row
                                # and K chunk: a group's partial tail item
                                # ran ~4x a full one and, claimed last,
                                # set the gather's span.
                                for r_base in cutlass.range_constexpr(
                                    0, ROWS_PER_WARP, ROW_PREFETCH
                                ):
                                    vecs: list = []
                                    for rr in cutlass.range_constexpr(ROW_PREFETCH):
                                        r: cutlass.Constexpr[int] = r_base + rr
                                        src_addr = peer_bases[r] + col_offset_bytes
                                        vecs.append(
                                            cute.arch.load(
                                                cute.make_ptr(
                                                    cutlass.Uint32,
                                                    src_addr,
                                                    cute.AddressSpace.gmem,
                                                ),
                                                _make_u32x4_vector_type(),
                                                cop="cg",
                                            )
                                        )
                                    for rr in cutlass.range_constexpr(ROW_PREFETCH):
                                        r: cutlass.Constexpr[int] = r_base + rr
                                        if row_in_bounds[r]:
                                            dst_addr = (
                                                dst_row_bases[r] + col_offset_bytes
                                            )
                                            cute.arch.store(
                                                cute.make_ptr(
                                                    cutlass.Uint32,
                                                    dst_addr,
                                                    cute.AddressSpace.gmem,
                                                ),
                                                vecs[rr],
                                                cop="cg",
                                            )
                    if cutlass.const_expr(K_SEGMENTS > 1 and SEGMENT_SIGNALS):
                        # Per-warp segment signal; the item-end signal
                        # covers the item's last segment.
                        if (k_outer + cutlass.Int32(1)) % seg_batches == cutlass.Int32(
                            0
                        ) and k_outer + cutlass.Int32(1) < num_k_outer:
                            cute.arch.fence_proxy("async.global")
                            cute.arch.fence_acq_rel_gpu()
                            cute.arch.sync_warp()
                            seg_signal = cutlass.Uint32(1) << cutlass.Uint32(
                                (
                                    k_seg
                                    + (k_outer + cutlass.Int32(1)) // seg_batches
                                    - 1
                                )
                                * 8
                            )
                            if lane_in_warp == cutlass.Int32(0):
                                cute.arch.atomic_add(
                                    cute.make_ptr(
                                        cutlass.Uint32,
                                        seg_counter_addr,
                                        cute.AddressSpace.gmem,
                                    ),
                                    seg_signal,
                                    sem="release",
                                    scope="gpu",
                                )
                            seg_signals = seg_signals + seg_signal

            # Sub-tile work for (group g, gemm_m_idx, sub_offset) is
            # complete: atomic-add 1 to the per-M-tile counter slot.
            # The TMA producer body busy-waits on this slot per tile
            # until it reaches NUM_SUB_PER_GEMM_M, restoring the
            # gather/TMA pipeline that the old global wait collapsed.
            # The barrier below also ensures all threads' stores to
            # `a_local` are visible before the signal.
            cute.arch.fence_proxy("async.global")
            cute.arch.fence_acq_rel_gpu()
            cute.arch.barrier(
                barrier_id=_BAR_GATHER_WG_INTERNAL,
                number_of_threads=_DIST_GATHER_WG_THREADS,
            )
            if lane == cutlass.Int32(0):
                counter_slot_addr = seg_counter_addr
                if cutlass.const_expr(gather_columns):
                    counter_slot_addr = a_buff_counter_addr_i64 + cutlass.Int64(
                        work.group_idx * num_k_chunks + feature_tile_idx
                    ) * cutlass.Int64(4)
                done_signal = cutlass.Uint32(1)
                if cutlass.const_expr(K_SEGMENTS > 1):
                    # Every warp sent the same per-segment signals.
                    done_signal = cutlass.Uint32(NUM_GATHER_WARPS) << cutlass.Uint32(
                        k_seg * 8
                    )
                    if cutlass.const_expr(SEGMENT_SIGNALS):
                        if claim_segments > cutlass.Int32(1):
                            done_signal = cutlass.Uint32(
                                NUM_GATHER_WARPS
                                * (((1 << (8 * K_SEGMENTS)) - 1) // 255)
                            )
                        done_signal = (
                            done_signal - cutlass.Uint32(NUM_GATHER_WARPS) * seg_signals
                        )
                cute.arch.atomic_add(
                    cute.make_ptr(
                        cutlass.Uint32, counter_slot_addr, cute.AddressSpace.gmem
                    ),
                    done_signal,
                    sem="release",
                    scope="gpu",
                )
                next_tile_idx_raw = cute.arch.atomic_add(
                    cute.make_ptr(
                        cutlass.Uint32,
                        gather_counter_addr_i64,
                        cute.AddressSpace.gmem,
                    ),
                    claim_step,
                    sem="release",
                    scope="gpu",
                )
                cute.arch.store(
                    gather_tile_idx_smem_ptr,
                    cutlass.Int32(next_tile_idx_raw),
                    ss="cta",
                )
            cute.arch.barrier(
                barrier_id=_BAR_GATHER_WG_INTERNAL,
                number_of_threads=_DIST_GATHER_WG_THREADS,
            )
            tile_idx = cute.arch.load(
                gather_tile_idx_smem_ptr + cutlass.Int32(0),
                cutlass.Int32,
                ss="cta",
            )
            work = visitor.get_work(tile_idx)


@cute.jit
def _mega_forward_wait_h1_tile(
    mDoneCounter: cute.Tensor,
    h1_done_counter_offset: cutlass.Int32,
    act_tile_slot: cutlass.Int32,
    h2_dim: cutlass.Int32,
    BLOCK_SIZE_M: cutlass.Constexpr[int],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
    NUM_CTAS: cutlass.Constexpr[int],
    SWAP_AB: cutlass.Constexpr[bool],
) -> None:
    FEATURE_BLOCK_SIZE: cutlass.Constexpr[int] = (
        BLOCK_SIZE_M if SWAP_AB else BLOCK_SIZE_N
    )
    expected_tiles = ceil_div(
        2 * h2_dim,
        cutlass.Int32(FEATURE_BLOCK_SIZE),
    ) * cutlass.Int32(NUM_CTAS)
    ptr = cute.recast_ptr(
        mDoneCounter.iterator + h1_done_counter_offset + act_tile_slot,
        dtype=cutlass.Uint32,
    )
    _wait_counter_at_least(ptr, cutlass.Uint32(expected_tiles))
