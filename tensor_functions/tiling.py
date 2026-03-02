"""Common tiling helpers for tensor functions.

Provides default tile-size constants and utility functions used by all
tensor function implementations.
"""

# ---------------------------------------------------------------------------
# Default tile sizes (can be overridden per-function when needed)
# ---------------------------------------------------------------------------

DEFAULT_VEC_TILE_M = 16
DEFAULT_VEC_TILE_N = 128

DEFAULT_MAT_TILE_M = 16
DEFAULT_MAT_TILE_N = 16
DEFAULT_MAT_TILE_K = 16


def compute_actual_size(total: int, offset: int, tile_size: int) -> int:
    """Return the actual tile extent, handling the tail block.

    For the last tile along a dimension the remaining elements may be
    fewer than ``tile_size``.  This function clamps the tile extent so
    it does not exceed the tensor boundary::

        actual = min(tile_size, total - offset)
    """
    return min(tile_size, total - offset)
