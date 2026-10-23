# SPDX-License-Identifier: Apache-2.0
"""Read-only per-invocation proof, published by the existing META writer."""

HELPER_SOURCE = '''# SPDX-License-Identifier: Apache-2.0
"""Equal-six-query metadata lookup; no task, workspace allocation or notification.

The META producer writes a 0/1 proof into the last alignment word before group
descriptors, only when that word is outside the B+1 live prefix. Readers must
acquire the same invocation's META phase. The flag is immutable for the rest
of the invocation and never replaces a synchronization primitive. A packed
prefix with no spare word always takes the original request lookup.
"""

import pypto.language as pl

from .indexer_group_stages import GROUP_DESCRIPTOR_ALIGNMENT, OFFSET_ROWS_DYN
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN, query_request


@pl.jit.inline(auto_scope=False)
def read_uniform_six(
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    requests: pl.Scalar[pl.INDEX],
) -> pl.Scalar[pl.INT32]:
    proof = pl.cast(0, pl.INT32)
    descriptor_base = (
        (requests + 1 + GROUP_DESCRIPTOR_ALIGNMENT - 1) // GROUP_DESCRIPTOR_ALIGNMENT * GROUP_DESCRIPTOR_ALIGNMENT
    )
    if descriptor_base > requests + 1:
        proof = pl.read(offsets, [descriptor_base - 1])
    return proof


@pl.jit.inline(auto_scope=False)
def query_request_uniform_six(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    query: pl.Scalar[pl.INDEX],
    tokens: pl.Scalar[pl.INDEX],
    proof: pl.Scalar[pl.INT32],
) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]]:
    owner = pl.cast(-1, pl.INDEX)
    begin = pl.cast(0, pl.INDEX)
    end = pl.cast(0, pl.INDEX)
    active = pl.cast(0, pl.INT32)
    if proof == 1:
        if query >= 0 and query < tokens:
            owner = query // 6
            begin = owner * 6
            end = begin + 6
            active = pl.cast(1, pl.INT32)
    else:
        owner, begin, end, active = query_request(bounds, lengths, query, tokens)
    return owner, begin, end, active
'''
