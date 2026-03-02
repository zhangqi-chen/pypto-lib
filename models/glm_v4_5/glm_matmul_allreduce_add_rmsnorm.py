"""GLM-4.5 Fused matmul + all-reduce + add + RMSNorm.

In distributed settings the matmul output is all-reduced before adding
the residual and applying RMSNorm.  The all-reduce step is a placeholder
here (single-device semantics).
"""

import pypto.language as pl
from tensor_functions import (
    tensor_add,
    tensor_matmul,
    tensor_rmsnorm,
    tensor_view,
)


@pl.function
def matmul_allreduce_add_rmsnorm(
    hidden_states: pl.Tensor,
    matmul_weight: pl.Tensor,
    residual: pl.Tensor,
    gamma: pl.Tensor,
    bias: pl.Tensor,
    normed_out: pl.Out[pl.Tensor],
    residual_out: pl.Out[pl.Tensor],
    eps: float = 1e-5,
) -> None:
    """Fused matmul → (all-reduce) → add residual → RMSNorm.

    hidden_states: [bs, in_dim]
    matmul_weight: [out_dim, in_dim]
    residual:      [bs, out_dim]
    gamma:         [1, out_dim]
    bias:          [1, out_dim]
    """
    weight_t = pl.transpose(matmul_weight)
    mm_result = tensor_matmul(hidden_states, weight_t)

    # all-reduce placeholder (identity in single-device mode)
    reduced = mm_result

    # add residual
    fused = tensor_add(reduced, residual)
    pl.assemble(residual_out, fused, [0, 0])

    # RMSNorm
    normed = tensor_rmsnorm(fused, gamma, eps)
    normed = tensor_add(normed, bias)
    pl.assemble(normed_out, normed, [0, 0])
