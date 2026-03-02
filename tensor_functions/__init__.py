"""PyPTO-Lib Tensor Functions.

A library of tensor-level primitive functions implemented with explicit
tiling, cast_tensor_to_tile / cast_tile_to_tensor, and tile-level
operations.  All functions are opaque (no incore boundary specified).
"""

# Cast primitives
from .cast_primitives import cast_tensor_to_tile, cast_tile_to_tensor

# Elementwise binary
from .elementwise_binary import (
    tensor_add,
    tensor_div,
    tensor_maximum,
    tensor_minimum,
    tensor_mul,
    tensor_sub,
)

# Elementwise unary
from .elementwise_unary import (
    tensor_abs,
    tensor_exp,
    tensor_log,
    tensor_logical_not,
    tensor_neg,
    tensor_recip,
    tensor_relu,
    tensor_rsqrt,
    tensor_sigmoid,
    tensor_sqrt,
)

# Reductions
from .reduction import tensor_amax, tensor_amin, tensor_sum

# Linear algebra
from .linalg import tensor_batch_matmul, tensor_matmul

# Type and layout
from .type_layout import (
    tensor_cast,
    tensor_clone,
    tensor_concat,
    tensor_expand_clone,
    tensor_full,
    tensor_reshape,
    tensor_transpose,
    tensor_view,
)

# Indexing
from .indexing import tensor_gather, tensor_scatter_update, tensor_where

# Composite (manual-fused)
from .composite import (
    tensor_dequant,
    tensor_gelu,
    tensor_layernorm,
    tensor_rmsnorm,
    tensor_rope,
    tensor_softmax,
    tensor_swiglu,
    tensor_symmetric_quant,
    tensor_topk,
)
