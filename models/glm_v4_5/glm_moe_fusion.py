"""GLM-4.5 MoE Fusion — gate + expert selection + shared expert FFN.

Fuses the gating, expert selection, and shared-expert quantised FFN
into a single logical operator.
"""

import pypto.language as pl
from tensor_functions import tensor_add, tensor_mul, tensor_view

from .glm_ffn_shared_expert_quant import ffn_shared_expert_quant
from .glm_gate import gate
from .glm_select_experts import select_experts


@pl.function
def moe_fusion(
    gate_weight: pl.Tensor,
    hidden_states: pl.Tensor,
    e_score_bias: pl.Tensor,
    w13: pl.Tensor,
    w13_scale: pl.Tensor,
    w2: pl.Tensor,
    w2_scale: pl.Tensor,
    topk_weights_out: pl.Out[pl.Tensor],
    topk_ids_out: pl.Out[pl.Tensor],
    ffn_res: pl.Out[pl.Tensor],
    top_k: int = 4,
    renormalize: bool = True,
    topk_group: int = 4,
    num_expert_group: int = 4,
) -> None:
    """Fused MoE: gate → select experts → shared expert FFN.

    gate_weight:    [num_experts, hidden_size]
    hidden_states:  [bs, hidden_size]
    e_score_bias:   [1, num_experts]
    w13, w2:        shared expert weights (INT8)
    w13_scale, w2_scale: dequantization scales
    """
    bs = hidden_states.shape[0]
    ne = gate_weight.shape[0]

    # Step 1: Gate
    router_logits = pl.create_tensor([bs, ne], dtype=pl.FP32)
    gate(hidden_states, gate_weight, router_logits)

    # Step 2: Select experts
    select_experts(
        router_logits, e_score_bias,
        topk_weights_out, topk_ids_out,
        top_k, renormalize, topk_group, num_expert_group,
    )

    # Step 3: Shared expert FFN
    ffn_shared_expert_quant(
        hidden_states, w13, w13_scale, w2, w2_scale, ffn_res,
    )
