# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CSA ACLGraph replay eligibility for DSpark decode.

DSpark uses 5 speculative tokens plus 1 draft token per request, so every
decode batch has a fixed per-request token count of 6.  ACLGraph capture sizes
must be aligned to multiples of 6 so that the BSH-layout kernel (B requests,
S=6 tokens each) sees a cleanly divisible token axis.

This module owns the one constant that ties the two concerns together and the
predicate that decides whether a padded batch can safely replay a captured graph.
"""

# DSpark decode: 5 speculative tokens + 1 draft = 6 tokens per request.
QUERY_TOKENS: int = 6


def can_replay_csa_graph(
    *,
    padded_tokens: int,
    num_tokens: int,
    num_reqs: int,
    uniform_decode: bool,
    max_batch_size: int,
) -> bool:
    """Return True when the current batch may replay a captured CSA graph.

    Parameters
    ----------
    padded_tokens:
        The graph bucket size (batch_descriptor.num_tokens) chosen by
        dispatch_cudagraph.  Must be a multiple of QUERY_TOKENS for a CSA
        bucket.
    num_tokens:
        Actual scheduled token count for this step (before padding).
    num_reqs:
        Number of live (non-dummy) requests in the batch.
    uniform_decode:
        True when every live request contributes exactly QUERY_TOKENS tokens.
    max_batch_size:
        The maximum batch the CSA kernel was compiled for (from VllmConfig).
    """
    # Bucket outside the CSA capture range: not a CSA graph, pass through.
    in_csa_range = (
        0 < padded_tokens <= max_batch_size * QUERY_TOKENS
        and padded_tokens % QUERY_TOKENS == 0
    )
    if not in_csa_range:
        return True

    # Inside the CSA range: every live request must contribute exactly
    # QUERY_TOKENS tokens, and actual tokens must not exceed the bucket.
    # Dummy padding rows are gated by seq_lens == 0 inside the kernel.
    return (
        uniform_decode
        and num_tokens <= padded_tokens
        and num_tokens == num_reqs * QUERY_TOKENS
    )
