# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401 -- Imported symbols are also used by the appended generated helper.
"""META-owned zero initialization and token-owned diagonal coefficient writes.

The offset allocation has T additional INT32 words at its end. META initializes
them for this invocation before publishing; quantization only reads this tail.
All coefficient rows are zeroed before any token owner writes its 64-column
diagonal segment. Quantization publication covers that segment as well as its
original INT8/FP32 outputs. No token can clear another token's segment.
"""

import pypto.language as pl

from .indexer_group_stages import (
    COEFF_COLS_DYN,
    COEFF_ROWS_DYN,
    COEFFICIENT_ROWS,
    HEADS,
    OFFSET_ROWS_DYN,
    TOKENS_DYN,
    WEIGHT_ROWS_DYN,
    indexer_decode_group,
    indexer_query_visibility,
)
from .indexer_query_stages import (
    HADAMARD_ROWS_DYN,
    HADAMARD_SCALE,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    QH_HEAD_DIM_TILE,
    QH_QUANT_TILE,
    QUANT_ROWS_DYN,
    SCALE_ROWS_DYN,
)
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN

assert HEADS == IDX_N_HEADS == QH_QUANT_TILE == 64


@pl.jit.inline(auto_scope=False)
def prepare_fused_query_coefficients(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    offsets: pl.InOut[pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32]],
    coefficients: pl.InOut[pl.Tensor[[COEFF_ROWS_DYN, COEFF_COLS_DYN], pl.FP16]],
    tokens: pl.Scalar[pl.INDEX],
    GROUP_SIZE: pl.constexpr,
):
    """Only the existing META owner calls this, after the original group plan."""
    map_base = pl.tensor.dim(offsets, 0) - tokens
    for query in pl.range(tokens):
        pl.write(offsets, [map_base + query], pl.cast(-1, pl.INT32))
    requests = pl.cast(pl.tensor.dim(lengths, 0), pl.INDEX)
    groups = pl.cast(pl.read(offsets, [requests]), pl.INDEX)
    zero_rows = pl.tile.full([COEFFICIENT_ROWS, GROUP_SIZE * HEADS], dtype=pl.FP16, value=0.0)
    for group in pl.range(groups):
        _request, first, live, _visible = indexer_decode_group(bounds, lengths, positions, offsets, group, GROUP_SIZE)
        coefficients = pl.store(zero_rows, [group * COEFFICIENT_ROWS, 0], coefficients)
        for lane in pl.range(live):
            query = first + lane
            if query >= 0 and query < tokens:
                pl.write(offsets, [map_base + query], pl.cast(group * GROUP_SIZE + lane, pl.INT32))
    return offsets, coefficients
