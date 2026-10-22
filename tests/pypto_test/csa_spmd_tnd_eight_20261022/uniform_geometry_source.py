# SPDX-License-Identifier: Apache-2.0
"""Device-only proof of a dense, equal-query/equal-KV TND batch."""

UNIFORM_HELPERS = '''

@pl.jit.inline(auto_scope=False)
def uniform_query_width(
    query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    token_capacity: pl.Scalar[pl.INDEX],
) -> pl.Scalar[pl.INDEX]:
    """Prove every Q/L entry this invocation; zero selects the original scan.

    Full capacity must be active and densely partitioned into equal queries,
    with equal positive KV lengths. No inference from Host shape alone, no
    retained proof across graph replays, and no GM scratch/publication.
    Positions and page addresses remain checked by the existing consumers.
    """
    width = pl.cast(0, pl.INDEX)
    requests = pl.cast(pl.tensor.dim(seq_lens, 0), pl.INDEX)
    if requests > 1 and pl.tensor.dim(query_start_loc, 0) >= requests + 1:
        first = pl.cast(pl.read(query_start_loc, [0]), pl.INDEX)
        step = pl.cast(pl.read(query_start_loc, [1]), pl.INDEX)
        length = pl.cast(pl.read(seq_lens, [0]), pl.INDEX)
        if first == 0 and step > 0 and length >= step and requests * step == token_capacity:
            valid = pl.cast(1, pl.INT32)
            for request in pl.range(requests):
                actual_end = pl.cast(pl.read(query_start_loc, [request + 1]), pl.INDEX)
                actual_length = pl.cast(pl.read(seq_lens, [request]), pl.INDEX)
                if actual_end != (request + 1) * step or actual_length != length:
                    valid = pl.cast(0, pl.INT32)
            if valid != 0:
                width = step
    return width


@pl.jit.inline(auto_scope=False)
def query_request_with_uniform(
    query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    query: pl.Scalar[pl.INDEX],
    token_capacity: pl.Scalar[pl.INDEX],
    uniform_width: pl.Scalar[pl.INDEX],
) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]]:
    """Use only a proof from the same immutable Q/L and invocation."""
    owner = pl.cast(-1, pl.INDEX)
    begin = pl.cast(0, pl.INDEX)
    end = pl.cast(0, pl.INDEX)
    active = pl.cast(0, pl.INT32)
    if uniform_width > 0:
        if query >= 0 and query < token_capacity:
            owner = query // uniform_width
            begin = owner * uniform_width
            end = begin + uniform_width
            active = pl.cast(1, pl.INT32)
    else:
        owner, begin, end, active = query_request(query_start_loc, seq_lens, query, token_capacity)
    return owner, begin, end, active
'''
