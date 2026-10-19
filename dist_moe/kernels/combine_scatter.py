# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Row-pointer scatter of staged COMBINE epilogue tiles to peer outputs."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import vector
from cutlass.cutlass_dsl import dsl_user_op

from ._quant_conversion import _cvt_bf16x2_to_f32x2, _cvt_f32x2_to_f16x2_rn
from .dispatch_quant import _make_u32x4_vector_type


@dsl_user_op
def _add_bf16x8(vec, bias_vec, *, loc=None, ip=None):
    """Return ``bf16(f32(vec) + f32(bias_vec))`` for 8 BF16 values packed as 4 x u32."""

    def unpack(v, i):
        word = vector.extract(v, dynamic_position=[], static_position=[i])
        return _cvt_bf16x2_to_f32x2(cutlass.Uint32(word))

    words = []
    for i in range(4):
        x_lo, x_hi = unpack(vec, i)
        b_lo, b_hi = unpack(bias_vec, i)
        word = _cvt_f32x2_to_f16x2_rn(x_lo + b_lo, x_hi + b_hi, cutlass.BFloat16)
        words.append(word.ir_value(loc=loc, ip=ip))
    return vector.from_elements(_make_u32x4_vector_type(), words, loc=loc, ip=ip)


class CombineScatterMixin:
    """Requires `has_padding_sentinels` and `EPILOG_WG_THREADS` on the kernel.

    A kernel with `has_bias` adds a BF16 bias row to every scattered row.
    """

    has_padding_sentinels: bool
    EPILOG_WG_THREADS: int
    has_bias: bool = False

    @cute.jit
    def _combine_epilog_load_scatter_ptrs(
        self,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
        scatter_ptr_smem_ptr: cute.Pointer,
        tile_m_idx: cutlass.Int32,
        mma_m_idx: cutlass.Constexpr[int],
        cm_start: cutlass.Int32,
        m_size: cutlass.Int32,
        scatter_ptrs_addr_i64: cutlass.Int64,
        padding_row_addr_i64: cutlass.Int64,
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        NUM_MMA_ATOMS_M: cutlass.Constexpr[int],
        NUM_CTAS: cutlass.Constexpr[int],
        cluster_cta_rank: cutlass.Int32,
        subtile_idx: cutlass.Constexpr[int] = 0,
        subtile_axis_m: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Cache per-row scatter base pointers in CTA SMEM."""
        EPILOGUE_M: cutlass.Constexpr[int] = cute.size(sC.shape, mode=[0])
        ATOM_M: cutlass.Constexpr[int] = BLOCK_SIZE_M // NUM_MMA_ATOMS_M
        ATOM_CTA_M: cutlass.Constexpr[int] = ATOM_M // NUM_CTAS

        if tidx < cutlass.Int32(EPILOGUE_M):
            local_row = tidx
            row_in_group = (
                tile_m_idx * cutlass.Int32(BLOCK_SIZE_M)
                + cutlass.Int32(mma_m_idx * ATOM_M)
                + cluster_cta_rank * cutlass.Int32(ATOM_CTA_M)
                + local_row
            )
            if cutlass.const_expr(subtile_axis_m):
                row_in_group += cutlass.Int32(subtile_idx * EPILOGUE_M)
            if row_in_group < m_size:
                peer_addr_ptr = scatter_ptrs_addr_i64 + cutlass.Int64(
                    cm_start + row_in_group
                ) * cutlass.Int64(8)
                peer_base_i64 = cute.arch.load(
                    cute.make_ptr(cutlass.Int64, peer_addr_ptr, cute.AddressSpace.gmem),
                    cutlass.Int64,
                )
                if cutlass.const_expr(self.has_padding_sentinels):
                    if peer_base_i64 == cutlass.Int64(0):
                        peer_base_i64 = padding_row_addr_i64
                cute.arch.store(
                    scatter_ptr_smem_ptr + local_row,
                    peer_base_i64,
                    ss="cta",
                )

    @cute.jit
    def _combine_epilog_scatter_smem_tile(
        self,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
        scatter_ptr_smem_ptr: cute.Pointer,
        tile_m_idx: cutlass.Int32,
        tile_n_idx: cutlass.Int32,
        mma_m_idx: cutlass.Constexpr[int],
        mma_n_idx: cutlass.Constexpr[int],
        subtile_idx: cutlass.Constexpr[int],
        NUM_MMA_ATOMS_M: cutlass.Constexpr[int],
        NUM_MMA_ATOMS_N: cutlass.Constexpr[int],
        cm_start: cutlass.Int32,
        m_size: cutlass.Int32,
        n_size: cutlass.Int32,
        elem_size_bytes_c: cutlass.Constexpr[int],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        NUM_CTAS: cutlass.Constexpr[int],
        cluster_cta_rank: cutlass.Int32,
        ELEMS_PER_VEC: cutlass.Constexpr[int] = 8,
        subtile_axis_m: cutlass.Constexpr[bool] = False,
        bias_row_addr_i64: cutlass.Int64 | None = None,
    ) -> None:
        """Scatter from the staged epilogue tile. Each lane stores one
        16-byte contiguous-N chunk; with 64-elem N subtiles, lanes 0..7
        cover one row's chunks, 8..15 the next, etc. With `has_bias`, the
        tile's expert bias row at `bias_row_addr_i64` is added first."""
        EPILOGUE_M: cutlass.Constexpr[int] = cute.size(sC.shape, mode=[0])
        EPILOGUE_N: cutlass.Constexpr[int] = cute.size(sC.shape, mode=[1])
        CHUNKS_PER_ROW: cutlass.Constexpr[int] = EPILOGUE_N // ELEMS_PER_VEC
        ROWS_PER_PASS: cutlass.Constexpr[int] = self.EPILOG_WG_THREADS // CHUNKS_PER_ROW
        NUM_ROW_PASSES: cutlass.Constexpr[int] = (
            EPILOGUE_M + ROWS_PER_PASS - 1
        ) // ROWS_PER_PASS
        ATOM_M: cutlass.Constexpr[int] = BLOCK_SIZE_M // NUM_MMA_ATOMS_M
        ATOM_CTA_M: cutlass.Constexpr[int] = ATOM_M // NUM_CTAS
        N_ATOM_SPAN: cutlass.Constexpr[int] = BLOCK_SIZE_N // NUM_MMA_ATOMS_N

        chunk = tidx % CHUNKS_PER_ROW
        row_lane = tidx // CHUNKS_PER_ROW
        local_col = chunk * cutlass.Int32(ELEMS_PER_VEC)

        for row_pass in cutlass.range_constexpr(NUM_ROW_PASSES):
            local_row = row_lane + cutlass.Int32(row_pass * ROWS_PER_PASS)
            if local_row < cutlass.Int32(EPILOGUE_M):
                row_in_group = (
                    tile_m_idx * cutlass.Int32(BLOCK_SIZE_M)
                    + cutlass.Int32(mma_m_idx * ATOM_M)
                    + cluster_cta_rank * cutlass.Int32(ATOM_CTA_M)
                    + local_row
                )
                if cutlass.const_expr(subtile_axis_m):
                    row_in_group += cutlass.Int32(subtile_idx * EPILOGUE_M)
                n_global_first = (
                    tile_n_idx * cutlass.Int32(BLOCK_SIZE_N)
                    + cutlass.Int32(mma_n_idx * N_ATOM_SPAN)
                    + local_col
                )
                if cutlass.const_expr(not subtile_axis_m):
                    n_global_first += cutlass.Int32(subtile_idx * EPILOGUE_N)

                if (row_in_group < m_size) and (
                    n_global_first + cutlass.Int32(ELEMS_PER_VEC) <= n_size
                ):
                    peer_base_i64 = cute.arch.load(
                        scatter_ptr_smem_ptr + local_row,
                        cutlass.Int64,
                        ss="cta",
                    )

                    col0 = local_col
                    ptr_vec = cute.recast_ptr(
                        sC.iterator
                        + cute.crd2idx(
                            (local_row, col0 + cutlass.Int32(0), cutlass.Int32(0)),
                            sC.layout,
                        ),
                        dtype=cutlass.Uint32,
                    )
                    vec = cute.arch.load(
                        ptr_vec,
                        _make_u32x4_vector_type(),
                        ss="cta",
                    )
                    if cutlass.const_expr(self.has_bias):
                        bias_addr = bias_row_addr_i64 + cutlass.Int64(
                            n_global_first
                        ) * cutlass.Int64(elem_size_bytes_c)
                        bias_vec = cute.arch.load(
                            cute.make_ptr(
                                cutlass.Uint32, bias_addr, cute.AddressSpace.gmem
                            ),
                            _make_u32x4_vector_type(),
                        )
                        vec = _add_bf16x8(vec, bias_vec)
                    dst_addr = peer_base_i64 + cutlass.Int64(
                        n_global_first
                    ) * cutlass.Int64(elem_size_bytes_c)
                    cute.arch.store(
                        cute.make_ptr(cutlass.Uint32, dst_addr, cute.AddressSpace.gmem),
                        vec,
                        cop="cg",
                    )
