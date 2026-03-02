"""GLM-4.5 Gate — router logits computation for MoE.

router_logits = hidden_states @ gate_weight^T
"""

import pypto.language as pl
from tensor_functions import tensor_matmul


@pl.function
def gate(hidden_states: pl.Tensor, gate_weight: pl.Tensor,
         router_logits_out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Compute router logits for MoE gating.

    hidden_states: [bs, hidden_size]
    gate_weight:   [num_experts, hidden_size]
    router_logits: [bs, num_experts]
    """
    gate_weight_t = pl.transpose(gate_weight)
    result = tensor_matmul(hidden_states, gate_weight_t)
    pl.assemble(router_logits_out, result, [0, 0])
    return router_logits_out
