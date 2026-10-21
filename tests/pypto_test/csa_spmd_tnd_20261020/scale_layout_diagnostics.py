# SPDX-License-Identifier: Apache-2.0
"""Check the completed device scale-page proof outside measured execution."""


def inspect_scale_layout(entry, workspace, base):
    """Demand the exact proof or fallback implied by every actual Native page.

    CPU readbacks here are test diagnostics after the kernel completes. They do
    not feed metadata or routing decisions into production execution.
    """
    bounds = entry.buffers["idx_bounds"].cpu().tolist()
    lengths = entry.buffers["idx_lengths"].cpu().tolist()
    table = entry.buffers["indexer_table"].cpu()
    physical_pages = entry.buffers["idx_cache"].shape[0]
    tokens = entry.buffers["positions"].shape[0]
    assert len(bounds) == len(lengths) + 1 and table.shape[0] == len(lengths)
    assert workspace.numel() == base + 2 * len(lengths), "unexpected layout proof extent"
    exact_integer_limit = 1 << 24
    rows = []
    for request, length in enumerate(lengths):
        begin, end = bounds[request:request + 2]
        active = length > 0 and 0 <= begin < end <= tokens and end - begin <= length
        pages = ((length // 4) + 31) // 32 if active else 0
        expected = (0, 0)
        if active and 0 < pages <= table.shape[1] and 0 < physical_pages <= exact_integer_limit:
            ids = table[request, :pages].tolist()
            if all(0 <= page < physical_pages for page in ids):
                differences = [right - left for left, right in zip(ids, ids[1:])]
                if all(delta == 1 for delta in differences):
                    expected = (ids[0], 1)
                elif all(delta == -1 for delta in differences):
                    expected = (ids[0], -1)
        actual = tuple(workspace[base + 2 * request:base + 2 * request + 2].tolist())
        assert actual == expected, f"request{request}: proof {actual} != complete-page evidence {expected}"
        rows.append({"request": request, "active": active, "pages": pages,
                     "first_page": actual[0], "direction": actual[1]})
    return {
        "passed": True, "requests": rows,
        "ascending": [row["request"] for row in rows if row["direction"] == 1],
        "descending": [row["request"] for row in rows if row["direction"] == -1],
        "active_fallback": [row["request"] for row in rows if row["active"] and row["direction"] == 0],
        "inactive": [row["request"] for row in rows if not row["active"]],
        "scope": "Exact post-completion page-layout proof; outside L1 and production dispatch",
    }
