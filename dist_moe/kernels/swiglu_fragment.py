# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SwiGLU forward on the native grouped-GEMM TMA-store epilogue fragment.

FC13 weights stay in natural `[gate; up]` order and are read through an
interleaved view with group `NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE`, half the
64-column epilogue subtile. Each subtile then holds `[32 gate | 32 up]` columns of one
natural 32-column block: the C TMA store writes both halves back to their
natural h1 columns through the same view, and each epilogue thread, which owns
one row of 64 consecutive accumulator columns, has complete gate/up pairs.

With two TMEM buffers h2 comes from the accumulator fragment in the TMEM drain
(`store_swiglu_h2_fragment`), hidden behind the other buffer's MMA. With one,
that drain would stall the MMA, so the epilogue releases TMEM first and
recomputes h2 from the h1 it just stored (`store_swiglu_h2_from_h1`); both are
bitwise identical to plain h1 + `swiglu_fwd`.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import dsl_user_op

from ._grouped_gemm_config import NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE
from ._quant_conversion import (
    _cvt_f16x2_to_f32x2,
    _cvt_f32x2_to_f16x2_rn,
)
from .blockscaled_quantization_common import (
    _copy_store_u32_words,
    _mul_f32x2,
    _rcp_rn_f32_fast_path,
    _rcp_rn_f32_ge1,
    _sigmoid_f32,
    _swiglu_clamped_fwd_f32,
)

_EPILOG_ROWS_PER_CTA_ATOM = 128
_SIGMOID_BATCH = 8


class SwigluH2Epilogue(NamedTuple):
    m_h2_words: cute.Tensor
    m_h1_words: cute.Tensor
    fast_math: bool
    clamped: bool
    alpha: float
    limit: float


def h1_swiglu_epilogue(
    kernel,
    m_h2_words: cute.Tensor,
    h1_base_ptr: cutlass.Int64,
    rows,
    h1_row_stride: cutlass.Int32,
) -> SwigluH2Epilogue | None:
    """The epilogue's `SwigluH2Epilogue` if `kernel.swiglu_emit_h1`, else None."""
    if not kernel.swiglu_emit_h1:
        return None
    words = h1_row_stride // 2
    m_h1_words = cute.make_tensor(
        cute.make_ptr(
            cutlass.Uint32, h1_base_ptr, cute.AddressSpace.gmem, assumed_align=16
        ),
        # Int64 row stride: row * words passes 2**31 at ~2**31 / I routed rows.
        cute.make_layout((rows, words), stride=(cutlass.Int64(words), 1)),
    )
    return SwigluH2Epilogue(
        m_h2_words,
        m_h1_words,
        kernel.swiglu_fast_math,
        kernel.swiglu_clamped,
        kernel.swiglu_alpha,
        kernel.swiglu_limit,
    )


def interleaved_fc13_view(
    tensor: cute.Tensor,
    mode: int,
    swizzle_group_size: int = NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE,
) -> cute.Tensor:
    """View the natural `[gate; up]` FC13 axis `mode` in the order
    `_interleave_fc13_rows` produces for `swizzle_group_size`."""
    shape = list(tensor.shape)
    stride = list(tensor.stride)
    n, s = shape[mode], stride[mode]
    shape[mode] = (swizzle_group_size, 2, n // (2 * swizzle_group_size))
    stride[mode] = (s, (n // 2) * s, swizzle_group_size * s)
    return cute.make_tensor(
        tensor.iterator, cute.make_layout(tuple(shape), stride=tuple(stride))
    )


@dsl_user_op
def _no_if_conversion(*, loc=None, ip=None) -> None:
    """Side-effecting no-op; ptxas will not predicate the block holding it."""
    llvm.inline_asm(
        None,
        [],
        "nanosleep.u32 0;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _sigmoid_rn_f32_batch(
    x_r: cute.Tensor, sig_r: cute.Tensor, N: cutlass.Constexpr[int]
) -> None:
    """Bitwise ``_sigmoid_f32(x, fast_math=False)`` over ``N`` values.

    ``rcp.rn.f32`` branches to its slow path per element, and each branch
    closes a scheduling block, so one producer warp per SMSP ran every
    element's EX2 -> RCP chain back to back. One slow-path check per batch
    lets ptxas interleave the chains.

    Unit-scale random weights put ~10% of gates below -87.3, so nearly every
    warp takes the fallback and it must stay branch-free. Left alone, ptxas
    if-converts it into the hot path (+7-13% even when never taken); the
    warp vote keeps the branch uniform and ``_no_if_conversion`` keeps it a
    branch.
    """
    den = cute.make_rmem_tensor(N, Float32)
    slow = cutlass.Uint32(0)
    for i in cutlass.range_constexpr(N):
        den[i] = Float32(1.0) + cute_math.exp(-x_r[i], fastmath=False)
        r, s = _rcp_rn_f32_fast_path(den[i])
        sig_r[i] = r
        slow = slow | s
    if cute.arch.vote_any_sync(slow != cutlass.Uint32(0)):
        _no_if_conversion()
        for i in cutlass.range_constexpr(N):
            sig_r[i] = _rcp_rn_f32_ge1(den[i])


@cute.jit
def store_swiglu_h2_fragment(
    swiglu: SwigluH2Epilogue,
    tRS_rC: cute.Tensor,
    tidx: cutlass.Int32,
    cm_start: cutlass.Int32,
    m_size: cutlass.Int32,
    n_size: cutlass.Int32,
    tile_m_idx: cutlass.Int32,
    tile_n_idx: cutlass.Int32,
    mma_m_idx: cutlass.Constexpr[int],
    mma_n_idx: cutlass.Constexpr[int],
    subtile_idx: cutlass.Int32,
    num_mma_atoms_m: cutlass.Constexpr[int],
    num_mma_atoms_n: cutlass.Constexpr[int],
    block_size_m: cutlass.Constexpr[int],
    block_size_n: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    cluster_cta_rank: cutlass.Int32,
) -> None:
    group: cutlass.Constexpr[int] = NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE
    atom_m: cutlass.Constexpr[int] = block_size_m // num_mma_atoms_m
    atom_cta_m: cutlass.Constexpr[int] = atom_m // num_ctas
    if cutlass.const_expr(
        cute.size(tRS_rC) != 2 * group or atom_cta_m != _EPILOG_ROWS_PER_CTA_ATOM
    ):
        raise ValueError(
            "fused SwiGLU h2 needs one 128-lane accumulator row of "
            f"{2 * group} columns per thread; got {cute.size(tRS_rC)} values "
            f"and {atom_cta_m} rows per CTA atom"
        )
    row_in_group = (
        tile_m_idx * cutlass.Int32(block_size_m)
        + cutlass.Int32(mma_m_idx * atom_m)
        + cluster_cta_rank * cutlass.Int32(atom_cta_m)
        + tidx
    )
    input_col = (
        tile_n_idx * cutlass.Int32(block_size_n)
        + cutlass.Int32(mma_n_idx * (block_size_n // num_mma_atoms_n))
        + subtile_idx * cutlass.Int32(2 * group)
    )
    # Columns are warp-uniform; rows are not, and the batched sigmoid votes.
    if input_col + cutlass.Int32(2 * group) <= n_size:
        # Physical register i holds accumulator column i of this subtile.
        words = cute.make_tensor(
            cute.recast_ptr(tRS_rC.iterator, dtype=cutlass.Int32),
            cute.make_layout(group),
        )
        out = cute.make_rmem_tensor(group // 2, cutlass.Uint32)
        _swiglu_words(
            words,
            out,
            swiglu.fast_math,
            swiglu.clamped,
            swiglu.alpha,
            swiglu.limit,
            tRS_rC.element_type,
        )
        if row_in_group < m_size:
            for chunk in cutlass.range_constexpr(2):
                _copy_store_u32_words(
                    swiglu.m_h2_words,
                    cm_start + row_in_group,
                    input_col // cutlass.Int32(4) + cutlass.Int32(chunk * group // 4),
                    out,
                    group // 4,
                    chunk * group // 4,
                )


@cute.jit
def _load_u32_words(
    m_words: cute.Tensor,
    row: cutlass.Int32,
    word_col: cutlass.Int32,
    dst: cute.Tensor,
    dst_offset: cutlass.Constexpr[int],
    num_words: cutlass.Constexpr[int],
) -> None:
    src_ptr = cute.make_ptr(
        dtype=cutlass.Uint32,
        value=(
            m_words.iterator + cute.crd2idx((row, word_col), m_words.layout)
        ).toint(),
        mem_space=m_words.iterator.memspace,
        assumed_align=4 * num_words,
    )
    src = cute.make_tensor(src_ptr, cute.make_layout((num_words,)))
    tmp = cute.make_rmem_tensor(num_words, cutlass.Uint32)
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Uint32, num_bits_per_copy=32 * num_words
    )
    cute.copy(copy_atom, src, tmp)
    for i in cutlass.range_constexpr(num_words):
        dst[dst_offset + i] = tmp[i]


@cute.jit
def store_swiglu_h2_from_h1(
    swiglu: SwigluH2Epilogue,
    is_tma_store_warp,
    barrier_id: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
    tidx: cutlass.Int32,
    cm_start: cutlass.Int32,
    m_size: cutlass.Int32,
    n_size: cutlass.Int32,
    tile_m_idx: cutlass.Int32,
    tile_n_idx: cutlass.Int32,
    num_mma_atoms_m: cutlass.Constexpr[int],
    num_mma_atoms_n: cutlass.Constexpr[int],
    subtile_cnt: cutlass.Constexpr[int],
    block_size_m: cutlass.Constexpr[int],
    block_size_n: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    cluster_cta_rank: cutlass.Int32,
    dtype: cutlass.Constexpr,
) -> None:
    """h2 of this CTA's rows of the tile from the h1 it just TMA-stored.

    Runs after the accumulator is released, so SwiGLU overlaps the next
    tile's MMA instead of holding TMEM.
    """
    group: cutlass.Constexpr[int] = NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE
    atom_m: cutlass.Constexpr[int] = block_size_m // num_mma_atoms_m
    atom_cta_m: cutlass.Constexpr[int] = atom_m // num_ctas
    atom_n: cutlass.Constexpr[int] = block_size_n // num_mma_atoms_n
    if cutlass.const_expr(
        atom_n != subtile_cnt * 2 * group or atom_cta_m != _EPILOG_ROWS_PER_CTA_ATOM
    ):
        raise ValueError(
            f"fused SwiGLU h2 needs {2 * group}-column subtiles over "
            f"{_EPILOG_ROWS_PER_CTA_ATOM} rows per CTA atom; got atom_n={atom_n}, "
            f"{subtile_cnt} subtiles and {atom_cta_m} rows"
        )
    # The async-proxy h1 writes must land before generic loads read them.
    if is_tma_store_warp:
        cute.arch.cp_async_bulk_wait_group(0, read=False)
        cute.arch.fence_proxy("async.global")
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)
    up_word_col = n_size // cutlass.Int32(4)
    for mma_m_idx in cutlass.range_constexpr(num_mma_atoms_m):
        row_in_group = (
            tile_m_idx * cutlass.Int32(block_size_m)
            + cutlass.Int32(mma_m_idx * atom_m)
            + cluster_cta_rank * cutlass.Int32(atom_cta_m)
            + tidx
        )
        row = cm_start + row_in_group
        for mma_n_idx in cutlass.range_constexpr(num_mma_atoms_n):
            for subtile_idx in cutlass.range(subtile_cnt, unroll=1):
                input_col = (
                    tile_n_idx * cutlass.Int32(block_size_n)
                    + cutlass.Int32(mma_n_idx * atom_n)
                    + subtile_idx * cutlass.Int32(2 * group)
                )
                # Columns are warp-uniform; rows are not, and the batched
                # sigmoid votes.
                if input_col + cutlass.Int32(2 * group) <= n_size:
                    _swiglu_h2_row_subtile(
                        swiglu,
                        row,
                        row_in_group < m_size,
                        input_col // cutlass.Int32(4),
                        up_word_col,
                        dtype,
                    )


@cute.jit
def _swiglu_h2_row_subtile(
    swiglu: SwigluH2Epilogue,
    row: cutlass.Int32,
    row_valid,
    word_col: cutlass.Int32,
    up_word_col: cutlass.Int32,
    dtype: cutlass.Constexpr,
) -> None:
    """h2 words `[word_col, word_col + group / 2)` of `row` from its h1.

    Invalid rows compute on zeros and store nothing.
    """
    group: cutlass.Constexpr[int] = NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE
    chunk: cutlass.Constexpr[int] = group // 4
    words = cute.make_rmem_tensor(group, cutlass.Uint32)
    words.fill(0)
    if row_valid:
        for c in cutlass.range_constexpr(2):
            _load_u32_words(
                swiglu.m_h1_words,
                row,
                word_col + cutlass.Int32(c * chunk),
                words,
                c * chunk,
                chunk,
            )
            _load_u32_words(
                swiglu.m_h1_words,
                row,
                up_word_col + word_col + cutlass.Int32(c * chunk),
                words,
                group // 2 + c * chunk,
                chunk,
            )
    out = cute.make_rmem_tensor(group // 2, cutlass.Uint32)
    _swiglu_words(
        words,
        out,
        swiglu.fast_math,
        swiglu.clamped,
        swiglu.alpha,
        swiglu.limit,
        dtype,
    )
    if row_valid:
        for c in cutlass.range_constexpr(2):
            _copy_store_u32_words(
                swiglu.m_h2_words,
                row,
                word_col + cutlass.Int32(c * chunk),
                out,
                chunk,
                c * chunk,
            )


@cute.jit
def _swiglu_words(
    words: cute.Tensor,
    out: cute.Tensor,
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    dtype: cutlass.Constexpr,
) -> None:
    """SwiGLU of `[gate | up]` b16x2 `words` into `out`."""
    half: cutlass.Constexpr[int] = cute.size(out)
    if cutlass.const_expr(fast_math or clamped):
        for pair in cutlass.range_constexpr(half):
            gate = _cvt_f16x2_to_f32x2(words[pair], dtype)
            up = _cvt_f16x2_to_f32x2(words[pair + half], dtype)
            if cutlass.const_expr(clamped):
                result = tuple(
                    _swiglu_clamped_fwd_f32(gate[i], up[i], alpha, limit, fast_math)
                    for i in range(2)
                )
            else:
                sigmoid = (_sigmoid_f32(gate[0], True), _sigmoid_f32(gate[1], True))
                result = _mul_f32x2(_mul_f32x2(gate, sigmoid), up)
            out[pair] = _cvt_f32x2_to_f16x2_rn(result[0], result[1], dtype)
    else:
        batches: cutlass.Constexpr[int] = 2 * half // _SIGMOID_BATCH
        gate = cute.make_rmem_tensor((_SIGMOID_BATCH, batches), Float32)
        sig = cute.make_rmem_tensor((_SIGMOID_BATCH, batches), Float32)
        for pair in cutlass.range_constexpr(half):
            g0, g1 = _cvt_f16x2_to_f32x2(words[pair], dtype)
            gate[2 * pair] = g0
            gate[2 * pair + 1] = g1
        for b in cutlass.range_constexpr(batches):
            _sigmoid_rn_f32_batch(gate[None, b], sig[None, b], _SIGMOID_BATCH)
        for pair in cutlass.range_constexpr(half):
            result = _mul_f32x2(
                _mul_f32x2(
                    (gate[2 * pair], gate[2 * pair + 1]),
                    (sig[2 * pair], sig[2 * pair + 1]),
                ),
                _cvt_f16x2_to_f32x2(words[pair + half], dtype),
            )
            out[pair] = _cvt_f32x2_to_f16x2_rn(result[0], result[1], dtype)


def h1_epi_smem_layout(
    c_dtype: type[cutlass.Numeric], epi_tile: cute.Tile, num_stages: int
) -> cute.ComposedLayout:
    """C staging layout for the TMA store through `interleaved_fc13_view`.

    Each TMA box row is one gate or up half of the subtile, and TMA pads a box
    row to the swizzle span, so the 128B swizzle `make_smem_layout_epi` picks
    for a 64-column tile would drop every up half. Tile 64B-swizzled atoms
    M-first so each half is its own dense `(rows, group)` box.
    """
    if c_dtype.width * NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE != 64 * 8:
        raise ValueError(f"h1 staging needs 64-byte halves; got {c_dtype}")
    atom = sm100_utils.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW64, c_dtype)
    shape = cute.append(cute.product_each(cute.shape(epi_tile)), num_stages)
    return cute.tile_to_shape(atom, shape, order=(0, 1, 2))
