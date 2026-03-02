"""Qwen3-Next Gated Delta Rule — ported to new pypto coding style.

Implements chunk-wise gated delta rule linear attention with O(n)
complexity.  All functions are opaque and call tensor_functions
primitives.
"""

import pypto.language as pl

from tensor_functions import (
    tensor_add,
    tensor_div,
    tensor_exp,
    tensor_full,
    tensor_matmul,
    tensor_mul,
    tensor_sqrt,
    tensor_sub,
    tensor_sum,
    tensor_view,
)


# ===================================================================
# l2norm
# ===================================================================

@pl.function
def l2norm(query: pl.Tensor, key: pl.Tensor,
           q_out: pl.Out[pl.Tensor],
           k_out: pl.Out[pl.Tensor],
           eps: float = 1e-6) -> None:
    """L2-normalise query [L, D] and key [L, D]."""
    q_sq = tensor_mul(query, query)
    q_ss = tensor_sum(q_sq, axis=-1, keepdim=True)
    q_ss_eps = tensor_add(q_ss, eps)
    q_norm = tensor_sqrt(q_ss_eps)
    q_normed = tensor_div(query, q_norm)
    pl.assemble(q_out, q_normed, [0, 0])

    k_sq = tensor_mul(key, key)
    k_ss = tensor_sum(k_sq, axis=-1, keepdim=True)
    k_ss_eps = tensor_add(k_ss, eps)
    k_norm = tensor_sqrt(k_ss_eps)
    k_normed = tensor_div(key, k_norm)
    pl.assemble(k_out, k_normed, [0, 0])


# ===================================================================
# pre_attn
# ===================================================================

@pl.function
def pre_attn(
    gate_view: pl.Tensor,
    key_view_2d: pl.Tensor,
    beta_view: pl.Tensor,
    tril: pl.Tensor,
    mask: pl.Tensor,
    gate_cum_out: pl.Out[pl.Tensor],
    decay_mask_out: pl.Out[pl.Tensor],
    a_out: pl.Out[pl.Tensor],
    key_beta_out: pl.Out[pl.Tensor],
) -> None:
    """Compute gate_cumsum, decay_mask, A matrix, and key*beta.

    gate_view: [L, 1], key_view_2d: [L, D], beta_view: [L, 1],
    tril: [L, L], mask: [L, L].
    """
    # cumsum via tril matmul
    gate_cum = tensor_matmul(tril, gate_view)
    pl.assemble(gate_cum_out, gate_cum, [0, 0])

    # decay_mask = exp((gate_cum - gate_cum^T) * tril)
    gate_cum_t = pl.transpose(gate_cum)
    diff = tensor_sub(gate_cum, gate_cum_t)
    diff_masked = tensor_mul(diff, tril)
    decay_mask = tensor_exp(diff_masked)
    pl.assemble(decay_mask_out, decay_mask, [0, 0])

    # key_beta = key * beta
    key_beta = tensor_mul(key_view_2d, beta_view)
    pl.assemble(key_beta_out, key_beta, [0, 0])

    # kkt = key_beta @ key^T
    key_t = pl.transpose(key_view_2d)
    kkt = tensor_matmul(key_beta, key_t)

    # A = kkt * decay_mask * mask
    a = tensor_mul(kkt, decay_mask)
    a = tensor_mul(a, mask)
    pl.assemble(a_out, a, [0, 0])


# ===================================================================
# inverse_pto (block-wise matrix inversion placeholder)
# ===================================================================

@pl.function
def inverse_pto(a_matrix: pl.Tensor,
                inv_out: pl.Out[pl.Tensor],
                n: int = 0) -> None:
    """Block-wise matrix inversion: (I - A)^{-1}.

    Uses iterative Neumann series: X = I + A + A^2 + A^3 + ...
    truncated at the matrix dimension.

    TODO: full implementation with proper tiling.
    """
    L = a_matrix.shape[0]

    # Start with identity
    tensor_full(inv_out, 0.0)
    # Diagonal = 1 (conceptual; needs scatter_update for diagonal)
    # Iterative: inv = I + A + A@A + A@A@A ...
    accum = a_matrix  # A^1
    result = tensor_add(inv_out, a_matrix)

    for _ in pl.range(1, L, 1):
        accum = tensor_matmul(accum, a_matrix)
        result = tensor_add(result, accum)

    pl.assemble(inv_out, result, [0, 0])


# ===================================================================
# chunk_gated_delta_rule (main entry)
# ===================================================================

@pl.function
def chunk_gated_delta_rule(
    query: pl.Tensor,
    key: pl.Tensor,
    value: pl.Tensor,
    gate: pl.Tensor,
    beta: pl.Tensor,
    tril: pl.Tensor,
    mask: pl.Tensor,
    output: pl.Out[pl.Tensor],
    eps: float = 1e-6,
) -> pl.Tensor:
    """Main entry: chunk gated delta rule linear attention.

    All inputs are 2-D [L, D] or [L, 1] or [L, L].
    Output is [L, D].

    Steps:
      1. L2-normalise query and key.
      2. pre_attn: compute gate_cum, decay_mask, A, key_beta.
      3. Compute attention output from the fused operator.
    """
    L, D = query.shape[0], query.shape[1]

    # Step 1: L2 norm
    q_norm = pl.create_tensor([L, D], dtype=query.dtype)
    k_norm = pl.create_tensor([L, D], dtype=key.dtype)
    l2norm(query, key, q_norm, k_norm, eps)

    # Step 2: pre_attn
    gate_cum = pl.create_tensor([L, 1], dtype=pl.FP32)
    decay_mask = pl.create_tensor([L, L], dtype=pl.FP32)
    a_matrix = pl.create_tensor([L, L], dtype=pl.FP32)
    key_beta = pl.create_tensor([L, D], dtype=pl.FP32)

    pre_attn(gate, k_norm, beta, tril, mask,
             gate_cum, decay_mask, a_matrix, key_beta)

    # Step 3: (I - A)^{-1}
    inv_matrix = pl.create_tensor([L, L], dtype=pl.FP32)
    inverse_pto(a_matrix, inv_matrix)

    # Step 4: output = inv_matrix @ (decay_mask * (q_norm @ value))
    qv = tensor_matmul(q_norm, value)
    qv_decay = tensor_mul(qv, decay_mask)
    result = tensor_matmul(inv_matrix, qv_decay)

    pl.assemble(output, result, [0, 0])
    return output
