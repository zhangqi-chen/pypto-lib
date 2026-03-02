"""GLM-4.5 Expert Selection — MoE top-k expert routing.

Implements grouped top-k expert selection with optional renormalization:
  sigmoid → add bias → group top-k → mask → global top-k → renormalize.
"""

import pypto.language as pl
from tensor_functions import (
    tensor_add,
    tensor_div,
    tensor_mul,
    tensor_sigmoid,
    tensor_sum,
    tensor_topk,
)


@pl.function
def select_experts(
    router_logits: pl.Tensor,
    e_score_bias: pl.Tensor,
    topk_weights_out: pl.Out[pl.Tensor],
    topk_ids_out: pl.Out[pl.Tensor],
    top_k: int = 4,
    renormalize: bool = True,
    topk_group: int = 4,
    num_expert_group: int = 4,
) -> None:
    """Select top-k experts per token from router logits.

    router_logits:  [bs, num_experts]
    e_score_bias:   [1, num_experts]
    topk_weights:   [bs, top_k]
    topk_ids:       [bs, top_k]

    Steps per token:
      1. sigmoid(router_logits)
      2. Add e_score_bias
      3. Grouped top-k within each expert group
      4. Mask and global top-k
      5. Optional renormalization (weights sum to 1)
    """
    bs = router_logits.shape[0]
    ne = router_logits.shape[1]

    # Step 1: sigmoid
    scores = tensor_sigmoid(router_logits)

    # Step 2: add bias
    scores_biased = tensor_add(scores, e_score_bias)

    # Step 3-4: top-k selection
    # TODO: full grouped top-k implementation requires tensor_topk
    # Placeholder: direct top-k on biased scores
    topk_vals = pl.create_tensor([bs, top_k], dtype=pl.FP32)
    topk_idx = pl.create_tensor([bs, top_k], dtype=pl.INT32)
    tensor_topk(scores_biased, topk_vals, topk_idx, k=top_k, axis=-1)

    # Step 5: renormalize
    if renormalize:
        weight_sum = tensor_sum(topk_vals, axis=-1, keepdim=True)
        topk_vals = tensor_div(topk_vals, weight_sum)

    pl.assemble(topk_weights_out, topk_vals, [0, 0])
    pl.assemble(topk_ids_out, topk_idx, [0, 0])
