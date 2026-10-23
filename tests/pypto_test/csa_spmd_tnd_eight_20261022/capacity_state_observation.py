# SPDX-License-Identifier: Apache-2.0
"""Test-only compressed-page holes and physical-plan readback; no timed work."""

import numpy as np


def physical_mapping(arrays, compressed_rows):
    """Integer address oracle including inactive, padded and mixed request rows."""
    logical = arrays["logical_topk"].astype(np.int64)
    bounds = arrays["idx_bounds"].astype(np.int64)
    lengths = arrays["idx_lengths"].astype(np.int64)
    positions = arrays["positions"].astype(np.int64)
    table = arrays["compressed_table"].astype(np.int64)
    tokens = len(positions)
    assert logical.shape == (tokens, 512)
    assert np.all((positions >= 0) & (positions < 2**24 - 1))
    expected = np.full_like(logical, -1)
    for token in range(tokens):
        owners = [
            r
            for r, length in enumerate(lengths)
            if r + 1 < len(bounds)
            and 0 <= bounds[r] <= token < bounds[r + 1] <= tokens
            and 0 < bounds[r + 1] - bounds[r] <= length
        ]
        if len(owners) != 1:
            continue
        request = owners[0]
        if positions[token] != lengths[request] - (bounds[request + 1] - bounds[request]) + token - bounds[request]:
            continue
        upper = (int(positions[token]) + 1) // 4
        for slot, index in enumerate(logical[token]):
            if 0 <= index < upper and index // 32 < table.shape[1]:
                page = table[request, index // 32]
                if 0 <= page < compressed_rows // 32:
                    expected[token, slot] = page * 32 + index % 32
    return expected


def invalidate_selected_pages(metadata, topk, histories, physical_pages, invalid_page):
    """Invalidate historical pages selected by each request's first query."""
    assert invalid_page == -1 or invalid_page == physical_pages
    bounds = metadata["idx_query_start_loc"].tolist()
    table = metadata["cmp_block_table"]
    details = []
    for request, history in enumerate(histories):
        token = bounds[request]
        if token == bounds[request + 1]:
            continue
        logical = topk[token].tolist()
        # The first query cannot attend the newly compressed write row.
        assert len(logical) == 512 and all(0 <= row < history // 4 for row in logical)
        columns = sorted({row // 32 for row in logical})
        assert columns and max(columns) < (history // 4) // 32
        assert all(0 <= int(table[request, col]) < physical_pages for col in columns)
        table[request, columns] = invalid_page
        details.append(dict(request=request, first_token=token, columns=columns))
    assert details
    return dict(invalid_page=invalid_page, physical_pages=physical_pages, selected=details)


def inspect_plan(entry, root, hole=None):
    """Independently map all selected rows after each candidate replay."""
    arrays = {
        name: entry.buffers[name].detach().cpu().numpy().copy()
        for name in ("positions", "idx_bounds", "idx_lengths", "compressed_table")
    }
    tokens = root["idx_topk"].shape[0]
    arrays.update(
        logical_topk=root["idx_topk"].detach().cpu().numpy().copy(),
        physical_indices=entry.buffers["attention_indices"][:tokens].detach().cpu().numpy().copy(),
        valid_blocks=entry.buffers["attention_mask"][:tokens, :5].detach().cpu().numpy().copy(),
    )
    expected = physical_mapping(arrays, entry.buffers["main_cache"].shape[0])
    np.testing.assert_array_equal(arrays["physical_indices"], expected)
    np.testing.assert_array_equal(
        arrays["valid_blocks"][:, 1:] > 0, np.any(expected.reshape(tokens, 4, 128) >= 0, axis=2)
    )
    empty_queries = []
    if hole is not None:
        for request in hole["selected"]:
            token = request["first_token"]
            assert np.all(arrays["physical_indices"][token] == -1)
            assert np.all(arrays["valid_blocks"][token, 1:] == 0)
            assert arrays["valid_blocks"][token, 0] == 1
            empty_queries.append(token)
    return dict(
        passed=True,
        independent_physical_mapping=True,
        table_columns=arrays["compressed_table"].shape[1],
        selected_capacity=entry.policy.plan_table_columns,
        valid_compressed_rows=int(np.sum(expected >= 0)),
        all_compressed_blocks_invalid_but_swa_valid=empty_queries,
    )
