# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape- and format-driven schedule and topology policies of the chunked Mega MoE host."""

from .blockscaled_grouped_gemm import (
    BlockScaledFormatSpec,
    MXFP4,
    MXFP8_E4M3,
    NVFP4,
)
from .config import (
    DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
)
from .dist_blockscaled_config import (
    _dispatch_scale_copy_atom_aligned,
    _use_nvfp4_high_throughput_producers,
)

_MX_PREFILL_VECTOR_COPY_MIN_SCALE_STRIP_BYTES = 256
_A8W4_WIDE_GATHER_MAX_INTERMEDIATE_DIM = 3072
_A8W4_WIDE_GATHER_MIN_ROUTED_ROWS_PER_GROUP = 8192
_DEEP_FC2_ONE_TILE_MIN_ROUTED_ROWS = 16384
_DEEP_FC2_MULTI_TILE_MIN_ROUTED_ROWS_PER_GROUP = 16384
_A8W4_DEEP_FC2_MULTI_TILE_MIN_ROUTED_ROWS_PER_GROUP = 8192


def _use_nvfp4_high_throughput_topology(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    num_sms: int,
    hidden_dim: int,
    intermediate_dim: int,
) -> bool:
    # The square decode shape regressed with the extra producer warps; its
    # measured winner is the dedicated wide-row quant topology below.
    return _use_nvfp4_high_throughput_producers(
        format=format,
        rows=rows,
        num_sms=num_sms,
        hidden_dim=hidden_dim,
        mega=True,
    ) and (hidden_dim, intermediate_dim) != (4096, 4096)


def _use_nvfp4_wide_row_quant(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    routed_prefill: bool,
    routed_chunks_per_group: float | None,
    num_sms: int,
    hidden_dim: int,
    intermediate_dim: int,
) -> bool:
    # Routed prefill wins at any capacity; decode past the capacity floor was
    # never tuned for it. Past 16 routed chunks per group it measured
    # 3.6-10.5% slower on multi-node EP32/EP64, so launches without a routed
    # estimate (training, the async fused path) keep the capacity floor.
    return (
        format is NVFP4
        and (
            rows < num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
            or (routed_prefill and routed_chunks_per_group is not None)
        )
        and (routed_chunks_per_group is None or routed_chunks_per_group <= 16)
        and (hidden_dim, intermediate_dim) == (4096, 4096)
    )


def _use_vector_dispatch_scale_copy(
    *,
    format: BlockScaledFormatSpec,
    hidden_dim: int,
    intermediate_dim: int,
    wide_gather: bool,
    rows: int,
    num_sms: int,
) -> bool:
    # The blockscaled-dispatch copy producer moves scale factors one Uint8 at
    # a time unless this is set, and each byte is a dependent peer-NVLink
    # load+store -- at decode that latency chain dominates the whole fused
    # kernel (square EP4-proxy MXFP4 88.2 -> 65.9 us, contraction 107.0 -> 86.8 us with
    # 4-byte copies). NVFP4 keeps its measured policy (wide-gather bundles the
    # flag; the (12288, 3072) narrow shape enables it explicitly). Other
    # formats vectorize whenever a whole 4-column scale atom exists, except at
    # prefill row counts with short per-row scale strips, where the wider
    # copies serialize behind the gather producer instead: with 128-byte
    # strips (D=4096) square MXFP4 at 65,536 tokens/rank measures 13,824 -> 14,093
    # us vectorized, while 384-byte strips (D=12288) keep large vectorized
    # wins through 64K tokens (contraction A8W4 at 32,768: 28,333 -> 22,909 us).
    # MXFP8 rows ignore this flag: they always copy scales in 16-byte row loads
    # (`_dispatch_copy_blockscaled_work_tile`); for them it only picks the narrow
    # decode work item.
    if format is NVFP4:
        return (hidden_dim, intermediate_dim) == (12288, 3072) and not wide_gather
    if not _dispatch_scale_copy_atom_aligned(format=format, hidden_dim=hidden_dim):
        return False
    return (
        rows < num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
        or hidden_dim // format.sf_vec_size
        >= _MX_PREFILL_VECTOR_COPY_MIN_SCALE_STRIP_BYTES
    )


def _use_wide_gather_topology(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    num_sms: int,
    hidden_dim: int,
    intermediate_dim: int,
    interleaved_fc13: bool = False,
    mixed_operand_widths: bool = False,
    routed_rows_per_group: float | None = None,
) -> bool:
    # Wide gather is valid for every shape below. Square and contraction workloads
    # set the measured threshold; other MXFP4 contractions use it as a
    # performance heuristic rather than a capability requirement.
    if format is NVFP4:
        high_throughput = _use_nvfp4_high_throughput_topology(
            format=format,
            rows=rows,
            num_sms=num_sms,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )
        return high_throughput or (hidden_dim == 4096 and intermediate_dim == 4096)
    if rows < num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF:
        return False
    if format is MXFP8_E4M3 and hidden_dim >= 2 * intermediate_dim:
        # MXFP8 contraction-shape training forward is gather-throughput-bound:
        # the gathered operand scales with hidden_dim while FC13 work scales
        # with intermediate_dim. The (12288/3072) contraction set the measurement: 8-warp
        # gather is EP64 Mega fprop 4158->3493 us (-16.0%) and E2E -3.9% on
        # GB300 at E256/NTPE8192. Interleaved
        # inference prefill with MXFP8 weights keeps the narrow topology (wide
        # measured 0.9-7.1% slower for this contraction at EP8-EP64). With 4-bit weights FC13 is
        # cheap enough that the gather binds once each expert has 16 routed
        # 512-row chunks: contraction A8W4 at 16-256 chunks measured 7.3-22.4% faster on
        # EP8-EP64;
        # at 8 it was mixed (+1.9% to -5.6%). The threshold is in rows because
        # mixed-width launches derive chunk_rows only after this decision. FC13
        # does 2*intermediate_dim MACs per gathered byte, so the gather binds
        # less as that grows; the cap is the measured contraction shape because the larger shape
        # (14336/3584, same D/I ratio) measured -1.5% to +2.4% at 16-256 chunks
        return not interleaved_fc13 or (
            mixed_operand_widths
            and intermediate_dim <= _A8W4_WIDE_GATHER_MAX_INTERMEDIATE_DIM
            and routed_rows_per_group is not None
            and routed_rows_per_group >= _A8W4_WIDE_GATHER_MIN_ROUTED_ROWS_PER_GROUP
        )
    return format is MXFP4 and hidden_dim >= intermediate_dim


def _use_two_slot_schedule_ring(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    num_sms: int,
    hidden_dim: int,
    intermediate_dim: int,
) -> bool:
    # Two slots improved square MXFP4 prefill but reduced overlap elsewhere.
    prefill = rows >= num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
    return (
        format is MXFP4 and prefill and hidden_dim == 4096 and intermediate_dim == 4096
    )


def _use_t6_mxfp8_prefill_lead(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    num_sms: int,
    hidden_dim: int,
    intermediate_dim: int,
    interleaved_fc13: bool,
) -> bool:
    # The 12K-to-3K MXFP8 prefill shape benefits from leading FC13 publication.
    return (
        format is MXFP8_E4M3
        and interleaved_fc13
        and rows >= num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
        and hidden_dim == 12288
        and intermediate_dim == 3072
    )


def _use_smaller_swiglu_epilogue(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    num_sms: int,
    interleaved_fc13: bool,
) -> bool:
    # Measured square/contraction FP4 and MXFP8 prefill variants reduce epilogue register
    # pressure; other MXFP8 decode shapes retain the wider fragment.
    if not interleaved_fc13:
        return False
    if format is MXFP4:
        return True
    return format is MXFP8_E4M3 and (
        rows >= num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
    )


def _use_staged_swiglu_fragment(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    num_sms: int,
    hidden_dim: int,
    intermediate_dim: int,
    interleaved_fc13: bool,
) -> bool:
    # The square MXFP8 prefill shape gains from staging a wider fragment;
    # the contraction and decode rows measured slower with it.
    return (
        format is MXFP8_E4M3
        and interleaved_fc13
        and rows >= num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
        and hidden_dim == 4096
        and intermediate_dim == 4096
    )


# Measured crossover for the mixed-width (A8W4) chunked mega on the wide-
# intermediate shapes (hidden <= 2*intermediate, e.g. the square): the
# stage-all schedule loses to the wave schedule once the batch reaches a few
# row waves -- EP4 100-iter CUDA-graph means square 2,048 tok 1,017.8 -> 904.9 us
# (-11.1%) and 1,024 tok -5.5%; EP8 confirms -8.6% / -8.3% on the same square
# cells. The contraction shapes do NOT join the narrowing: larger contraction measured only
# noise-level gains at EP4 and +4.1% at EP8 T=2,048 under the wave schedule,
# and the pure formats measured +10.5% / +22.5% on their cells, so both keep
# the 128-row/SM stage-all band. True decode stays staged everywhere.
_STAGE_ALL_MIXED_WIDTH_MIN_ROWS_PER_SM: int = 32


def _stage_all_fc13(
    *,
    interleaved_fc13: bool,
    rows: int,
    num_sms: int,
    hidden_dim: int,
    intermediate_dim: int,
    mixed_weight_format: bool = False,
    format: BlockScaledFormatSpec | None = None,
    world_size: int | None = None,
    routed_rows: int | None = None,
) -> bool:
    # Avoid parking persistent clusters on FC2 dependencies when FC2 has a
    # large task frontier or FC13 publishes activations from its epilogue.
    narrow_stage_all = mixed_weight_format and hidden_dim <= 2 * intermediate_dim
    decode_rows_per_sm = (
        _STAGE_ALL_MIXED_WIDTH_MIN_ROWS_PER_SM
        if narrow_stage_all
        else DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
    )
    decode = rows < num_sms * decode_rows_per_sm
    if decode:
        return interleaved_fc13 or hidden_dim >= 2 * intermediate_dim
    # Interleaved MXFP8-activation prefill up to a few waves gains from
    # staging on local EP4 (contraction 32768 rows/rank -14.5%, A8W4 -10.0%), but the
    # same schedule measured 5-16% slower against the ambient-drift control on
    # contraction at EP8-EP32 for exactly the staged tiers. So only EP <= 4 stages
    # MXFP8 fused-SwiGLU prefill, up to 4 waves of routed rows (`rows` is the padded
    # capacity). 65,536 routed rows/rank at EP4: Qwen3.5-35B-A3B 758.8 -> 666.5 us,
    # DeepSeek-V3 3169 -> 2794 us.
    return (
        format is MXFP8_E4M3
        and interleaved_fc13
        and not mixed_weight_format
        and world_size is not None
        and world_size <= 4
        and routed_rows is not None
        and routed_rows < 4 * num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
    )


def _mega_fp4_mma_atom_n(
    *,
    config: dict,
    format: BlockScaledFormatSpec,
    hidden_dim: int,
    intermediate_dim: int,
    mixed_operand_widths: bool,
) -> int:
    block_n = int(config["BLOCK_SIZE_N"])
    if (
        mixed_operand_widths
        or format is NVFP4
        or (format is MXFP4 and (hidden_dim, intermediate_dim) == (12288, 3072))
    ):
        return block_n
    preferred_atom_n = min(128, block_n)
    return preferred_atom_n if block_n % preferred_atom_n == 0 else block_n


def _fc2_tiles_per_schedule_item(*, max_tiles: int, row_waves: int) -> int:
    candidate = min(max_tiles, max(1, row_waves))
    return 1 << (candidate.bit_length() - 1)
