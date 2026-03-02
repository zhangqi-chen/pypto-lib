"""DeepSeek-V3 model family (MLA, Sparse Attention, Lightning Indexer)."""

from .lightning_indexer_prolog_quant import lightning_indexer_prolog_quant_compute
from .lightning_indexer_quant import lightning_indexer_decode_compute
from .mla_indexer_prolog_quant import mla_indexer_prolog_quant_compute
from .mla_prolog_quant import mla_prolog_quant_compute, rms_norm, rope_2d
from .sparse_attention_antiquant import sparse_attention_antiquant_compute
from .sparse_flash_attention_quant import sparse_flash_attention_quant_compute
