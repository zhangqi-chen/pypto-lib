"""GLM-4.5 Fused Attention — RMSNorm + QKV + RoPE + Flash Attention.

End-to-end fused attention: from raw hidden states to attention output.
"""

import pypto.language as pl
from tensor_functions import (
    tensor_add,
    tensor_cast,
    tensor_matmul,
    tensor_mul,
    tensor_rmsnorm,
    tensor_rope,
    tensor_view,
)

from .glm_attention import attention as flash_attention
from .glm_attention_pre_quant import attention_pre_quant


@pl.function
def attention_fusion(
    hidden_states: pl.Tensor,
    residual: pl.Tensor,
    layernorm_weight: pl.Tensor,
    layernorm_bias: pl.Tensor,
    qkv_weight: pl.Tensor,
    qkv_deq_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    key_cache: pl.Tensor,
    value_cache: pl.Tensor,
    block_tables: pl.Tensor,
    actual_seq_lens: pl.Tensor,
    attn_out: pl.Out[pl.Tensor],
    residual_out: pl.Out[pl.Tensor],
    eps: float = 1e-5,
    softmax_scale: float = 1.0,
    head_dim: int = 128,
    block_size: int = 128,
) -> None:
    """Fused attention: pre-quant → flash attention.

    hidden_states: [bs, hidden_size]
    """
    bs = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    q_size = qkv_weight.shape[0] // 3

    query = pl.create_tensor([bs, q_size], dtype=hidden_states.dtype)
    key = pl.create_tensor([bs, q_size], dtype=hidden_states.dtype)
    value = pl.create_tensor([bs, q_size], dtype=hidden_states.dtype)

    # Step 1: Pre-processing (RMSNorm + QKV + RoPE)
    attention_pre_quant(
        hidden_states, residual,
        layernorm_weight, layernorm_bias,
        qkv_weight, qkv_deq_scale,
        q_norm_weight, k_norm_weight,
        cos, sin,
        query, key, value, residual_out,
        eps, head_dim,
    )

    # TODO: scatter key/value into KV cache

    # Step 2: Flash Attention
    flash_attention(
        query, key_cache, value_cache,
        block_tables, actual_seq_lens,
        attn_out, softmax_scale, block_size,
    )
