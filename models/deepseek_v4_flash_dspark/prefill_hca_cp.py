# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 context-parallel prefill HCA (ratio-128): one rank's token slice, with the
window KV cache, the compressor state and the compressed KV cache all fully replicated.
--cp picks the CP world size: 2 or 4 ranks over the same token run."""


# T_LOC freezes into the kernel shapes at import time, so read --cp from argv and
# override config before importing the sub-kernels below.
import sys

import config

_CP_CHOICES = (2, 4)
_CP_DEFAULT = 2


def _parse_cp_argv():
    for i, tok in enumerate(sys.argv):
        if tok == "--cp" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith("--cp="):
            return int(tok.split("=", 1)[1])
    return _CP_DEFAULT


CP = _parse_cp_argv()
config.CP = CP
config.TP_O = CP

import functools

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    BLOCK_SIZE,
    FLASH as M,
    HCA_STATE_PHYSICAL_BLOCKS,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    PREFILL_BATCH,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_BLOCK_NUM,
    PREFILL_ORI_MAX_BLOCKS,
    PREFILL_SEQ,
)
from hc_post import golden_hc_post_prefill, hc_post_prefill
from hc_pre import golden_hc_pre, hc_pre
from prefill_compressor_ratio128 import (
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    golden_prefill_compressor_ratio128,
    prefill_compressor_ratio128,
)
from qkv_proj_rope import (
    golden_qkv_proj_rope,
    kv_proj_rope,
    materialize_kv_rope_rows,
    materialize_rope_rows,
    q_proj_rope,
    rope_prepare,
)
from rmsnorm import golden_rms_norm, rms_norm
from prefill_sparse_attn import (
    O_GROUPS_LOC,
    PREFILL_ATTN_TILE,
    SPARSE_BIAS_COLS,
    T_PAD,
    VALID_BLOCK_MASK_COLS,
    golden_prefill_sparse_attn,
    o_proj,
    sparse_attn_core,
)


# Dynamic shape variables.
ORI_BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CMP_BLOCK_NUM_DYN")
STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_STATE_BLOCK_NUM_DYN")

# model config
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
T_LOC = T // CP  # per-rank token slice; the flat token stream is split into CP equal runs
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_DIM = ROPE_HEAD_DIM
Q_LORA = M.q_lora_rank
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window
IDX_TOPK = M.index_topk
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
O_COLS_LOC = O_GROUPS_LOC * O_GROUP_IN  # == H_LOC * HEAD_DIM

COMPRESS_RATIO = 128
MAIN_OUT_DIM = HEAD_DIM
MAIN_COMPRESS_STATE_DIM = 2 * MAIN_OUT_DIM
START_POS = 0

# paged KV cache
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
SPARSE_ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
SPARSE_CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
HCA_ORI_BLOCK_NUM = PREFILL_ORI_BLOCK_NUM
HCA_CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM

# tiling
GATHER_ROW_TILE = 8   # x_normed all-gather publish / stage-out row block
A2A_ROW_TILE = 2      # attention-output all-to-all publish / stage-out row block
RS_ROW_TILE = 4       # reduce-scatter publish / accumulate row block

assert CP in _CP_CHOICES, f"--cp must be one of {_CP_CHOICES} (got {CP})"
assert S % COMPRESS_RATIO == 0, "prefill HCA expects whole ratio-128 compression chunks"
assert WIN == BLOCK_SIZE, "prefill HCA currently assumes one window page per batch"
assert T_LOC % 16 == 0, "cube M and the bias token tile both need a 16-row multiple"
# HCA has no indexer: the compressed tail is every slot the cache holds, so the
# shared prefill pruning width must cover the whole cache, not a top-k budget.
assert MAX_SEQ_LEN // COMPRESS_RATIO <= PREFILL_MAX_COMPRESSED, (
    f"prefill HCA compressed tail ({PREFILL_MAX_COMPRESSED} slots) must cover "
    f"MAX_SEQ_LEN={MAX_SEQ_LEN} ({MAX_SEQ_LEN // COMPRESS_RATIO} slots)")


@pl.jit.inline
def prefill_hca_cp(
    x_hc: pl.Tensor[[T_LOC, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    position_ids_local: pl.Tensor[[T_LOC], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS_LOC, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS_LOC * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_norm_window: pld.DistributedTensor[[T, D], pl.BF16],
    ready: pld.DistributedTensor[[CP, 1], pl.INT32],
    o_window: pld.DistributedTensor[[T, O_COLS_LOC], pl.BF16],
    a2a_ready: pld.DistributedTensor[[CP, 1], pl.INT32],
    rs_window: pld.DistributedTensor[[T, D], pl.BF16],
    rs_ready: pld.DistributedTensor[[CP, 1], pl.INT32],
    x_out: pl.Out[pl.Tensor[[T_LOC, HC_MULT, D], pl.FP32]],
    num_tokens_full: pl.Scalar[pl.INT32],
    num_tokens_local: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """One CP rank: hc_pre and rms_norm over its token slice, all-gather the normalized
    run, then q on the local rows and kv plus the compressor on the full run."""
    x_mixed = pl.create_tensor([T_LOC, D], dtype=pl.BF16)
    post = pl.create_tensor([T_LOC, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T_LOC, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post, comb)

    x_normed_local = pl.create_tensor([T_LOC, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed_local)
    # Defers kv_proj_matmul one hop behind rms_norm so qr_proj_matmul dispatches first.
    late_dep = pl.system.task_dummy(deps=[rms_tid])

    # All-gather the normalized slices. Carries BF16 [T_LOC, D] rather than the pre-hc
    # FP32 lanes, which are 32x larger.
    publish_row = my_rank * T_LOC
    x_normed_full = pl.create_tensor([T, D], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_x_normed_gather"):
        for peer in pl.range(CP):
            pld.tensor.put(
                dst=x_norm_window, peer=peer, src=x_normed_local,
                dst_offsets=[publish_row, 0], src_offsets=[0, 0], shape=[T_LOC, D],
                chunk_rows=GATHER_ROW_TILE, chunk_cols=D, pipeline=True,
            )
        for peer in pl.range(CP):
            if peer != my_rank:
                pld.system.notify(
                    target=ready, peer=peer, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(CP):
            if src != my_rank:
                pld.system.wait(signal=ready, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)

        for t0 in pl.range(0, T, GATHER_ROW_TILE):
            x_normed_full[t0 : t0 + GATHER_ROW_TILE, 0:D] = x_norm_window[t0 : t0 + GATHER_ROW_TILE, 0:D]

    # q branch: local rows, rope rows at this rank's global absolute positions.
    rope_cos_loc = pl.create_tensor([T_LOC, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_loc = pl.create_tensor([T_LOC, ROPE_HEAD_DIM], dtype=pl.BF16)
    materialize_rope_rows(freqs_cos, freqs_sin, position_ids_local, num_tokens_local, rope_cos_loc, rope_sin_loc)

    q_cos_il = pl.create_tensor([T_LOC, ROPE_DIM], dtype=pl.FP32)
    q_sin_signed = pl.create_tensor([T_LOC, ROPE_DIM], dtype=pl.FP32)
    q_swap_idx = pl.create_tensor([T_LOC, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_cos_loc, rope_sin_loc, q_cos_il, q_sin_signed, q_swap_idx)

    q = pl.create_tensor([T_LOC, H, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T_LOC, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T_LOC, 1], dtype=pl.FP32)
    q_proj_rope(
        x_normed_local, wq_a, wq_b, wq_b_scale, gamma_cq,
        q_cos_il, q_sin_signed, q_swap_idx,
        q, qr, qr_scale,
    )

    # kv branch: full token run, so every rank writes an identical complete kv_cache.
    # The kv rope rows must ride KV_T_DYN, not the q-side T_DYN: one dynamic axis driven at
    # two row counts binds to the first and silently clips the second.
    rope_cos_full = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_full = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    materialize_kv_rope_rows(freqs_cos, freqs_sin, position_ids, num_tokens_full, rope_cos_full, rope_sin_full)

    kv_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    kv_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    kv_swap_idx = pl.create_tensor([T, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_cos_full, rope_sin_full, kv_cos_il, kv_sin_signed, kv_swap_idx)

    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(x_normed_full, wkv, gamma_ckv, kv_cos_il, kv_sin_signed, kv_swap_idx, kv, late_dep)

    # Writeback stays global: the sliding window of a local query reaches tokens owned
    # by other ranks, and it reads them from this rank's own complete copy.
    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_cp_cache_write"):
        for write_t in pl.range(T):
            if write_t < num_tokens_full:
                write_row_raw = pl.read(ori_slot_mapping, [write_t])
                if write_row_raw >= 0:
                    write_row = pl.cast(write_row_raw, pl.INDEX)
                    kv_cache_flat[write_row : write_row + 1, :] = kv[write_t : write_t + 1, :]

    # Compressor stays global too, and for the same reason: its softmax pool looks back a
    # fixed 128 absolute positions, a window that crosses rank boundaries. Every rank runs
    # the full T rows against the global slot mappings and ends up with identical
    # compress_state / cmp_kv replicas.
    prefill_compressor_ratio128(
        x_normed_full, compress_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        freqs_cos, freqs_sin, cmp_kv,
        position_ids, num_tokens_full, cmp_slot_mapping, state_slot_mapping,
    )

    # Index build is the only CP-local piece: visible_cmp depends solely on the global
    # absolute position, so a local row matches the non-CP row at the same global index.
    swa_indices = pl.create_tensor([T_LOC, WIN], dtype=pl.INT32)
    cmp_indices = pl.create_tensor([T_LOC, IDX_TOPK], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([T_LOC, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_cp_sparse_indices"):
        for idx_t in pl.range(T_LOC):
            swa_row = pl.full([1, WIN], dtype=pl.INT32, value=-1)
            cmp_row = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
            mask_row = pl.full([1, VALID_BLOCK_MASK_COLS], dtype=pl.INT32, value=0)
            if idx_t < num_tokens_local:
                abs_pos = pl.read(position_ids_local, [idx_t])
                window_valid = pl.min(pl.cast(WIN, pl.INT32), abs_pos + 1)
                key_start_abs = abs_pos + 1 - window_valid
                for win_col in pl.range(WIN):
                    win_col_i32 = pl.cast(win_col, pl.INT32)
                    if win_col_i32 < window_valid:
                        key_abs = key_start_abs + win_col_i32
                        blk_slot = key_abs // BLOCK_SIZE
                        blk = pl.read(ori_block_table, [pl.cast(blk_slot, pl.INDEX)])
                        if blk >= 0:
                            row = pl.cast(blk * BLOCK_SIZE + (key_abs - blk_slot * BLOCK_SIZE), pl.INT32)
                            pl.write(swa_row, [0, win_col], row)
                            if win_col < SPARSE_BIAS_COLS:
                                pl.write(mask_row, [0, win_col // PREFILL_ATTN_TILE], pl.cast(1, pl.INT32))
                visible_cmp = (abs_pos + 1) // COMPRESS_RATIO
                for cmp_col in pl.range(IDX_TOPK):
                    cmp_col_i32 = pl.cast(cmp_col, pl.INT32)
                    if cmp_col_i32 < visible_cmp:
                        if cmp_col_i32 < pl.cast(SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE, pl.INT32):
                            pl.write(cmp_row, [0, cmp_col], cmp_col_i32)
                            sparse_col = WIN + cmp_col
                            if sparse_col < SPARSE_BIAS_COLS:
                                pl.write(mask_row, [0, sparse_col // PREFILL_ATTN_TILE], pl.cast(1, pl.INT32))
            swa_indices[idx_t : idx_t + 1, 0:WIN] = swa_row
            cmp_indices[idx_t : idx_t + 1, 0:IDX_TOPK] = cmp_row
            valid_block_mask[idx_t : idx_t + 1, 0:VALID_BLOCK_MASK_COLS] = mask_row

    # The inverse RoPE inside the core un-rotates by the LOCAL token positions, so it has
    # to run before the all-to-all redistributes the token rows.
    o_local = pl.create_tensor([T_PAD, H * HEAD_DIM], dtype=pl.BF16)
    sparse_attn_core(
        q, kv_cache, swa_indices,
        cmp_kv, cmp_block_table,
        cmp_indices,
        valid_block_mask, attn_sink,
        rope_cos_loc, rope_sin_loc,
        o_local, num_tokens_local,
    )

    # All-to-all: token-split -> head-split. o_local is token-major and head-minor, so
    # peer p's heads are the column band [p*O_COLS_LOC, (p+1)*O_COLS_LOC) and land in
    # peer p's window at row my_rank*T_LOC. The window is then [T, O_COLS_LOC] with
    # global token index s*T_LOC + t, ready for o_proj with no local reorder.
    o_full = pl.create_tensor([T_PAD, O_COLS_LOC], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_o_all_to_all") as a2a_tid:
        for peer in pl.range(CP):
            pld.tensor.put(
                dst=o_window, peer=peer, src=o_local,
                dst_offsets=[my_rank * T_LOC, 0], src_offsets=[0, peer * O_COLS_LOC],
                shape=[T_LOC, O_COLS_LOC],
                chunk_rows=A2A_ROW_TILE, chunk_cols=O_COLS_LOC, pipeline=True,
            )
        for peer in pl.range(CP):
            if peer != my_rank:
                pld.system.notify(
                    target=a2a_ready, peer=peer, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(CP):
            if src != my_rank:
                pld.system.wait(signal=a2a_ready, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)

        for t0 in pl.range(0, T, A2A_ROW_TILE):
            o_full[t0 : t0 + A2A_ROW_TILE, 0:O_COLS_LOC] = o_window[t0 : t0 + A2A_ROW_TILE, 0:O_COLS_LOC]

    # o_proj over this rank's group band only, so attn_partial is a partial sum over D.
    attn_partial = pl.create_tensor([T, D], dtype=pl.BF16)
    o_proj(o_full, wo_a, wo_b, wo_b_scale, attn_partial, a2a_tid, num_tokens_full)

    # Reduce-scatter: rank r keeps the sum over ranks of rows [r*T_LOC, (r+1)*T_LOC).
    # Each rank drops its slice for peer p into slot my_rank of p's window, so the sum is
    # a local accumulate and needs no atomics or a zeroed window.
    attn_out = pl.create_tensor([T_LOC, D], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_o_reduce_scatter"):
        for peer in pl.range(CP):
            pld.tensor.put(
                dst=rs_window, peer=peer, src=attn_partial,
                dst_offsets=[my_rank * T_LOC, 0], src_offsets=[peer * T_LOC, 0],
                shape=[T_LOC, D],
                chunk_rows=RS_ROW_TILE, chunk_cols=D, pipeline=True,
            )
        for peer in pl.range(CP):
            if peer != my_rank:
                pld.system.notify(
                    target=rs_ready, peer=peer, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(CP):
            if src != my_rank:
                pld.system.wait(signal=rs_ready, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)

        for t0 in pl.range(0, T_LOC, RS_ROW_TILE):
            rs_acc = pl.full([RS_ROW_TILE, D], dtype=pl.FP32, value=0.0)
            for src in pl.range(CP):
                rs_slab = rs_window[src * T_LOC + t0 : src * T_LOC + t0 + RS_ROW_TILE, 0:D]
                rs_acc = pl.add(rs_acc, pl.cast(rs_slab, target_type=pl.FP32))
            attn_out[t0 : t0 + RS_ROW_TILE, 0:D] = pl.cast(rs_acc, target_type=pl.BF16, mode="rint")

    hc_post_prefill(attn_out, x_hc, post, comb, x_out, num_tokens_local)


@pl.jit
def l2_prefill_hca_cp(
    x_hc: pl.Tensor[[T_LOC, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    position_ids_local: pl.Tensor[[T_LOC], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS_LOC, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS_LOC * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_norm_window: pld.DistributedTensor[[T, D], pl.BF16],
    ready: pld.DistributedTensor[[CP, 1], pl.INT32],
    o_window: pld.DistributedTensor[[T, O_COLS_LOC], pl.BF16],
    a2a_ready: pld.DistributedTensor[[CP, 1], pl.INT32],
    rs_window: pld.DistributedTensor[[T, D], pl.BF16],
    rs_ready: pld.DistributedTensor[[CP, 1], pl.INT32],
    x_out: pl.Out[pl.Tensor[[T_LOC, HC_MULT, D], pl.FP32]],
    num_tokens_full: pl.Scalar[pl.INT32],
    num_tokens_local: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    prefill_hca_cp(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, ori_slot_mapping, ori_block_table,
        cmp_kv, cmp_block_table,
        position_ids, position_ids_local, cmp_slot_mapping, state_slot_mapping,
        attn_sink, wo_a, wo_b, wo_b_scale,
        x_norm_window, ready, o_window, a2a_ready, rs_window, rs_ready,
        x_out, num_tokens_full, num_tokens_local, my_rank,
    )
    return kv_cache, compress_state, cmp_kv, x_out


@pl.jit.host
def l3_prefill_hca_cp(
    x_hc: pl.Tensor[[CP, T_LOC, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[CP, HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[CP, HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.InOut[pl.Tensor[[CP, HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    position_ids_local: pl.Tensor[[CP, T_LOC], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[CP, O_GROUPS_LOC, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[CP, D, O_GROUPS_LOC * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[CP, T_LOC, HC_MULT, D], pl.FP32]],
    num_tokens_full: pl.Scalar[pl.INT32],
    num_tokens_local: pl.Scalar[pl.INT32],
):
    """Launch one CP rank per device. Weights and the global token metadata are shared; the
    token slice, its positions, the output and the three cache copies are per rank."""
    window_buf = pld.alloc_window_buffer([T, D], dtype=pl.BF16)
    ready_buf = pld.alloc_window_buffer([CP, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([T, O_COLS_LOC], dtype=pl.BF16)
    a2a_ready_buf = pld.alloc_window_buffer([CP, 1], dtype=pl.INT32)
    rs_window_buf = pld.alloc_window_buffer([T, D], dtype=pl.BF16)
    rs_ready_buf = pld.alloc_window_buffer([CP, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        x_norm_window = pld.window(window_buf, [T, D], dtype=pl.BF16)
        ready = pld.window(ready_buf, [CP, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [T, O_COLS_LOC], dtype=pl.BF16)
        a2a_ready = pld.window(a2a_ready_buf, [CP, 1], dtype=pl.INT32)
        rs_window = pld.window(rs_window_buf, [T, D], dtype=pl.BF16)
        rs_ready = pld.window(rs_ready_buf, [CP, 1], dtype=pl.INT32)
        l2_prefill_hca_cp(
            x_hc[rank],
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
            freqs_cos, freqs_sin,
            cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
            compress_state[rank], compress_state_block_table,
            kv_cache[rank], ori_slot_mapping, ori_block_table,
            cmp_kv[rank], cmp_block_table,
            position_ids, position_ids_local[rank], cmp_slot_mapping, state_slot_mapping,
            attn_sink, wo_a[rank], wo_b[rank], wo_b_scale,
            x_norm_window, ready, o_window, a2a_ready, rs_window, rs_ready,
            x_out[rank], num_tokens_full, num_tokens_local, rank,
            device=rank,
        )


def _quant_w_per_output_channel(w):
    import torch

    amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.view(1, -1)
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    w_i8 = w_i32.to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def _golden_local_norm(tensors, x_hc_local):
    """hc_pre + rms_norm over one rank's slice, the chunk it contributes to the gather."""
    import torch

    x_mixed = torch.zeros(T_LOC, D, dtype=torch.bfloat16)
    post = torch.zeros(T_LOC, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(T_LOC, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x": x_hc_local,
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post,
        "comb": comb,
    })
    return golden_rms_norm(x_mixed, tensors["attn_norm_w"])


def _golden_hca_cp_rank(tensors, x_hc_local, position_ids_local, kv_cache_in, compress_state_in,
                        cmp_kv_in, x_normed_full, wo_a_full, wo_b_full):
    """Torch reference for one rank: q on its token slice; kv, writeback and the compressor
    on the full run.

    Returns this rank's ``x_out`` slice; the three cache tensors are updated in place.
    """
    import torch

    from utils import cache_row_from_table

    num_tokens_full = int(tensors["num_tokens_full"])
    num_tokens_local = int(tensors["num_tokens_local"])

    x_mixed = torch.zeros(T_LOC, D, dtype=torch.bfloat16)
    post = torch.zeros(T_LOC, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(T_LOC, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x": x_hc_local,
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post,
        "comb": comb,
    })
    x_normed_local = golden_rms_norm(x_mixed, tensors["attn_norm_w"])

    positions_local = position_ids_local.to(torch.long)
    rope_cos_loc = tensors["freqs_cos"].index_select(0, positions_local).contiguous()
    rope_sin_loc = tensors["freqs_sin"].index_select(0, positions_local).contiguous()
    q = torch.zeros(T_LOC, H, HEAD_DIM, dtype=torch.bfloat16)
    kv_local_unused = torch.zeros(T_LOC, HEAD_DIM, dtype=torch.bfloat16)
    qr_local = torch.zeros(T_LOC, Q_LORA, dtype=torch.int8)
    qr_scale_local = torch.zeros(T_LOC, 1, dtype=torch.float32)
    golden_qkv_proj_rope({
        "x": x_normed_local,
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_loc,
        "rope_sin": rope_sin_loc,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv_local_unused,
        "qr": qr_local,
        "qr_scale": qr_scale_local,
    })

    positions_full = tensors["position_ids"].to(torch.long)
    rope_cos_full = tensors["freqs_cos"].index_select(0, positions_full).contiguous()
    rope_sin_full = tensors["freqs_sin"].index_select(0, positions_full).contiguous()
    q_full_unused = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr_full = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale_full = torch.zeros(T, 1, dtype=torch.float32)
    golden_qkv_proj_rope({
        "x": x_normed_full,
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_full,
        "rope_sin": rope_sin_full,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q_full_unused,
        "kv": kv,
        "qr": qr_full,
        "qr_scale": qr_scale_full,
    })

    kv_cache_flat = kv_cache_in.view(HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    for t in range(num_tokens_full):
        dst_row = int(tensors["ori_slot_mapping"][t].item())
        if dst_row >= 0:
            kv_cache_flat[dst_row, :] = kv[t]

    golden_prefill_compressor_ratio128({
        "x": x_normed_full,
        "compress_state": compress_state_in,
        "compress_state_block_table": tensors["compress_state_block_table"],
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "cmp_kv": cmp_kv_in,
        "position_ids": tensors["position_ids"],
        "num_tokens": tensors["num_tokens_full"],
        "cmp_slot_mapping": tensors["cmp_slot_mapping"],
        "state_slot_mapping": tensors["state_slot_mapping"],
    })

    def build_sparse_metadata():
        swa_idx = torch.full((T_LOC, WIN), -1, dtype=torch.int32)
        cmp_idx = torch.full((T_LOC, IDX_TOPK), -1, dtype=torch.int32)
        ori_table = tensors["ori_block_table"]
        cmp_cap = SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE
        for t in range(num_tokens_local):
            abs_pos = int(position_ids_local[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for k, key_abs in enumerate(range(key_start_abs, abs_pos + 1)):
                row = cache_row_from_table(ori_table, key_abs)
                if row >= 0:
                    swa_idx[t, k] = row
            visible_cmp = min((abs_pos + 1) // COMPRESS_RATIO, IDX_TOPK, cmp_cap)
            if visible_cmp > 0:
                cmp_idx[t, :visible_cmp] = torch.arange(visible_cmp, dtype=torch.int32)
        return swa_idx, cmp_idx

    swa_indices, cmp_indices = build_sparse_metadata()
    attn_out = torch.zeros(T_LOC, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn({
        "q": q,
        "ori_kv": kv_cache_in,
        "swa_indices": swa_indices,
        "cmp_kv": cmp_kv_in,
        "cmp_block_table": tensors["cmp_block_table"],
        "cmp_indices": cmp_indices,
        "attn_sink": tensors["attn_sink"],
        "num_tokens": tensors["num_tokens_local"],
        "freqs_cos": rope_cos_loc,
        "freqs_sin": rope_sin_loc,
        "wo_a": wo_a_full,
        "wo_b": wo_b_full,
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    y = torch.zeros(T_LOC, HC_MULT, D, dtype=torch.float32)
    golden_hc_post_prefill({
        "x": attn_out.view(T_LOC, D),
        "residual": x_hc_local,
        "post": post,
        "comb": comb,
        "y": y,
        "num_tokens": tensors["num_tokens_local"],
    })
    return y


def golden_prefill_hca_cp(tensors):
    """Torch reference: the all_gather is the rank-ordered concat of the local norms, and
    the sharded o_proj is the full-weight projection the reduce-scatter reassembles."""
    import torch

    slices = [_golden_local_norm(tensors, tensors["x_hc"][rank].view(T_LOC, HC_MULT, D)) for rank in range(CP)]
    x_normed_full = torch.cat(slices, dim=0).to(torch.bfloat16)
    wo_a_full = tensors["wo_a"].reshape(O_GROUPS, O_LORA, O_GROUP_IN)
    wo_b_full = tensors["wo_b"].permute(1, 0, 2).reshape(D, O_GROUPS * O_LORA)
    for rank in range(CP):
        kv_cache_in = tensors["kv_cache"][rank].clone()
        compress_state_in = tensors["compress_state"][rank].clone()
        cmp_kv_in = tensors["cmp_kv"][rank].clone()
        x_hc_local = tensors["x_hc"][rank].view(T_LOC, HC_MULT, D)
        y = _golden_hca_cp_rank(
            tensors, x_hc_local, tensors["position_ids_local"][rank],
            kv_cache_in, compress_state_in, cmp_kv_in,
            x_normed_full, wo_a_full, wo_b_full,
        )
        tensors["kv_cache"][rank] = kv_cache_in
        tensors["compress_state"][rank] = compress_state_in
        tensors["cmp_kv"][rank] = cmp_kv_in
        tensors["x_out"][rank] = y


@functools.lru_cache(maxsize=None)
def _state_block_table(max_blocks, physical_blocks):
    """Constant scrambled state block table [max_blocks]."""
    import torch
    blocks = torch.arange(max_blocks, dtype=torch.int32)
    return (blocks * 17 + 3) % physical_blocks


def build_tensor_specs(start_pos: int = START_POS, num_tokens: int = T):
    """Every rank's slice stacked on a leading CP axis."""
    import torch
    from golden import ScalarSpec, TensorSpec
    from utils import build_rope_tables, cache_row_from_table, quant_w_per_channel

    if num_tokens != T:
        raise ValueError(f"every slice must be active: num_tokens must be {T}")
    context_len = start_pos
    if context_len < 0:
        raise ValueError(f"context_len must be non-negative, got {context_len}")
    max_position = context_len + num_tokens
    if max_position > MAX_SEQ_LEN:
        raise ValueError(f"position_ids exceed MAX_SEQ_LEN={MAX_SEQ_LEN}: got {max_position}")

    num_tokens_local = T_LOC
    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)
    state_table = _state_block_table(HCA_STATE_MAX_BLOCKS, HCA_STATE_PHYSICAL_BLOCKS)

    def token_pos():
        # Global absolute positions; padding rows keep their arange default.
        pos = torch.arange(T, dtype=torch.int32)
        for local_s in range(num_tokens):
            pos[local_s] = context_len + local_s
        return pos

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        return int(state_table[block].item()) * HCA_STATE_BLOCK_SIZE + intra

    def init_x_hc_full():
        x = torch.empty(T, HC_MULT, D).uniform_(-1, 1)
        x[num_tokens:] = 0
        return x
    # Real layer-9 (HCA, ratio-128) hc_attn scale/base, fn synthetic at real magnitude. A
    # synthetic scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling
    # attn_out and the hc residual to near-zero in x_out where W8A8 noise blows up the tail.
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.0495
    def init_hc_attn_scale():
        return torch.tensor([0.079046, 0.04213, 0.121901])
    def init_hc_attn_base():
        return torch.tensor([
            -3.3004, 2.5553, -2.2787, -3.4925,
            -3.8197, -3.4161, -2.7144, -2.9181,
            2.362, -2.4746, -2.1352, -3.2216,
            -4.474, 2.2488, -2.1053, -3.1675,
            -2.8362, -1.9042, 2.0432, -3.062,
            -2.7902, -3.0908, -3.002, 3.1161,
        ])
    def init_attn_norm_w():
        return torch.ones(D)
    def init_wq_a():
        return (torch.rand(D, Q_LORA) - 0.5) * D ** -0.5
    def init_wq_b():
        return (torch.rand(Q_LORA, H * HEAD_DIM) - 0.5) * Q_LORA ** -0.5
    def init_wkv():
        return (torch.rand(D, HEAD_DIM) - 0.5) * D ** -0.5
    # Quant-faithful HCA (ratio-128) main compressor fixtures (mean l7/l9 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std;
    # RMSNorm gamma near the measured mean.
    def init_cmp_wkv():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0246
    def init_cmp_wgate():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0316
    def init_cmp_ape():
        return torch.randn(COMPRESS_RATIO, MAIN_OUT_DIM) * 0.0340
    def init_cmp_norm_w():
        return 0.1001 + torch.randn(HEAD_DIM,) * 0.0549
    def init_ori_block_table():
        table = torch.full((SPARSE_ORI_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(SPARSE_ORI_MAX_BLOCKS):
            table[block] = block
        return table
    def init_cmp_block_table():
        # Single-request paged table: one compressed page mapped to physical block 0.
        table = torch.full((SPARSE_CMP_MAX_BLOCKS,), -1, dtype=torch.int32)
        table[0] = 0
        return table
    def init_compress_state_block_table():
        return state_table.clone()
    def init_kv_cache():
        cache = torch.zeros(HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        table = init_ori_block_table()
        start = max(0, context_len - WIN)
        for abs_pos in range(start, context_len):
            row = cache_row_from_table(table, abs_pos)
            if row >= 0:
                cache_flat[row] = ((torch.rand(HEAD_DIM,) - 0.5) * 0.1).to(torch.bfloat16)
        # Every rank starts from the same replicated cache.
        return cache.expand(CP, *cache.shape).contiguous()
    def init_compress_state():
        state = torch.zeros(HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM)
        flat = state.view(-1, MAIN_COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, context_len - COMPRESS_RATIO), context_len):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(MAIN_COMPRESS_STATE_DIM,) - 0.5) * 0.05
        return state.expand(CP, *state.shape).contiguous()
    def init_cmp_kv():
        cache = torch.zeros(HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(HCA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        table = init_cmp_block_table()
        completed = context_len // COMPRESS_RATIO
        for cmp_slot in range(completed):
            row = cache_row_from_table(table, cmp_slot)
            if row >= 0:
                cache_flat[row] = ((torch.rand(HEAD_DIM,) - 0.5) * 0.1).to(torch.bfloat16)
        return cache.to(torch.bfloat16).expand(CP, *cache.shape).contiguous()
    def init_ori_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        pos = token_pos()
        table = init_ori_block_table()
        for t in range(num_tokens):
            mapping[t] = cache_row_from_table(table, int(pos[t].item()))
        return mapping
    def init_cmp_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        table = init_cmp_block_table()
        for local_s in range(num_tokens):
            abs_len = context_len + local_s + 1
            if abs_len >= COMPRESS_RATIO and abs_len % COMPRESS_RATIO == 0:
                mapping[local_s] = cache_row_from_table(table, abs_len // COMPRESS_RATIO - 1)
        return mapping
    def init_state_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        pos = token_pos()
        for t in range(num_tokens):
            mapping[t] = state_row(int(pos[t].item()))
        return mapping

    x_hc_full = init_x_hc_full()
    attn_norm_w = init_attn_norm_w().to(torch.bfloat16)

    position_ids = token_pos()
    x_hc_local = x_hc_full.view(CP, T_LOC, HC_MULT, D).contiguous()
    position_ids_local = position_ids.view(CP, T_LOC).contiguous()

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = ((torch.rand(D, O_GROUPS * O_LORA) - 0.5) * (O_GROUPS * O_LORA) ** -0.5).to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_channel(wo_b_bf16)
    wo_a_full = ((torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) * O_GROUP_IN ** -0.5).to(torch.bfloat16)
    # o_proj is TP-sharded: wo_a splits on the group axis, wo_b on the matching K band.
    wo_a_local = wo_a_full.view(CP, O_GROUPS_LOC, O_LORA, O_GROUP_IN).contiguous()
    wo_b_local = wo_b_i8.view(D, CP, O_GROUPS_LOC * O_LORA).permute(1, 0, 2).contiguous()

    return num_tokens_local, [
        TensorSpec("x_hc", [CP, T_LOC, HC_MULT, D], torch.float32, init_value=lambda: x_hc_local),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.bfloat16, init_value=lambda: attn_norm_w),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=lambda: torch.ones(Q_LORA)),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=lambda: torch.ones(HEAD_DIM)),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=shared_freqs_cos.clone),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=shared_freqs_sin.clone),
        TensorSpec("cmp_wkv", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_cmp_norm_w),
        TensorSpec("compress_state",
                   [CP, HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM],
                   torch.float32, init_value=init_compress_state, is_output=True),
        TensorSpec("compress_state_block_table", [HCA_STATE_MAX_BLOCKS], torch.int32,
                   init_value=init_compress_state_block_table),
        TensorSpec("kv_cache", [CP, HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=init_kv_cache, is_output=True),
        TensorSpec("ori_slot_mapping", [T], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec("ori_block_table", [SPARSE_ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("cmp_kv", [CP, HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=init_cmp_kv, is_output=True),
        TensorSpec("cmp_block_table", [SPARSE_CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("position_ids", [T], torch.int32, init_value=lambda: position_ids),
        TensorSpec("position_ids_local", [CP, T_LOC], torch.int32, init_value=lambda: position_ids_local),
        TensorSpec("cmp_slot_mapping", [T], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [T], torch.int64, init_value=init_state_slot_mapping),
        TensorSpec("attn_sink", [H], torch.float32, init_value=lambda: torch.zeros(H)),
        TensorSpec("wo_a", [CP, O_GROUPS_LOC, O_LORA, O_GROUP_IN], torch.bfloat16,
                   init_value=lambda: wo_a_local),
        TensorSpec("wo_b", [CP, D, O_GROUPS_LOC * O_LORA], torch.int8, init_value=lambda: wo_b_local),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [CP, T_LOC, HC_MULT, D], torch.float32, is_output=True),
        ScalarSpec("num_tokens_full", torch.int32, num_tokens),
        ScalarSpec("num_tokens_local", torch.int32, num_tokens_local),
    ]


def cp_replica_allclose(name: str, atol: float, rtol: float):
    """Replicated-cache comparator: the CP copies agree with each other, then match the
    reference.

    The projections accumulate their split-K partials with FP32 atomic adds, so the
    completion order -- and with it the last bf16 bit -- is not reproducible across cards.
    The replicas are therefore held to the same bar as the reference comparison rather than
    to bit equality.
    """
    from golden import ratio_allclose

    base_cmp = ratio_allclose(atol=atol, rtol=rtol)

    def cmp(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        for replica in range(1, actual.shape[0]):
            same, detail = base_cmp(
                actual[replica], actual[0],
                actual_outputs=actual_outputs, expected_outputs=expected_outputs,
                inputs=inputs, rtol=rtol, atol=atol,
            )
            if not same:
                return False, f"    {name} replica {replica} diverges from replica 0:\n{detail}"
        return base_cmp(
            actual, expected,
            actual_outputs=actual_outputs, expected_outputs=expected_outputs,
            inputs=inputs, rtol=rtol, atol=atol,
        )

    cmp.__name__ = f"cp_replica_allclose({name}, replicas={CP})"
    return cmp


if __name__ == "__main__":
    import argparse
    from pypto.ir.distributed_compiled_program import DistributedConfig

    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser(description="DeepSeek V4 context-parallel prefill HCA test.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(CP)),
                        help="Comma-separated device ids; at least CP of them.")
    parser.add_argument("--cp", type=int, default=_CP_DEFAULT, choices=list(_CP_CHOICES),
                        help="CP world size; read from argv at import time to freeze the shapes.")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="context_len (multiple of WIN); fixture-only, lowered into token metadata.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) < CP:
        raise SystemExit(f"CP{CP} needs {CP} devices, got {device_ids}")

    compare_tokens, specs = build_tensor_specs(args.start_pos)
    print(f"--- prefill_hca_cp: {CP} ranks x {T_LOC} rows, full={T}, devices={device_ids[:CP]} ---")

    result = run_jit(
        fn=l3_prefill_hca_cp,
        specs=specs,
        golden_fn=golden_prefill_hca_cp,
        compile_cfg=dict(
            distributed_config=DistributedConfig(device_ids=device_ids[:CP], num_sub_workers=0),
            dump_passes=args.dump_passes,
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_dep_gen=args.enable_dep_gen,
        ),
        compile_only=args.compile_only,
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.005, max_diff_hd=1,
                                   valid_rows=compare_tokens, valid_axis=1, zero_tail=True),
            "kv_cache": cp_replica_allclose("kv_cache", atol=1e-4, rtol=1e-2),
            "compress_state": cp_replica_allclose("compress_state", atol=1e-3, rtol=1e-3),
            "cmp_kv": cp_replica_allclose("cmp_kv", atol=1e-4, rtol=1e-2),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
