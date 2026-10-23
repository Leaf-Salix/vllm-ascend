# SPDX-License-Identifier: Apache-2.0
"""Keep locality metrics distinct from set reuse and reject stale physical plans."""

import numpy as np
import pytest
from analyze_kv_access import analyze_arrays


def fixture(order=None, bad_page=False):
    logical = np.tile(np.arange(512, dtype=np.int32), (2, 1))
    if order is not None:
        logical[1] = logical[1, order]
    table = np.arange(64, dtype=np.int32)[None, :]
    physical = logical.copy()
    if bad_page:
        table[0, 0] = -1
        physical[logical < 32] = -1
    return dict(
        logical_topk=logical,
        physical_indices=physical,
        idx_bounds=np.array([0, 2], dtype=np.int32),
        idx_lengths=np.array([4098], dtype=np.int32),
        positions=np.array([4096, 4097], dtype=np.int64),
        compressed_table=table,
        valid_blocks=np.ones((2, 5), dtype=np.int32),
    )


def test_contiguous_and_identical_queries():
    result = analyze_arrays(fixture(), 64 * 32)
    assert result["valid_compressed_rows"] == 1024
    assert result["contiguous_runs_within_existing_64_row_lane_segments"] == 16
    assert result["whole_page_overfetch_ratio_per_existing_lane_segment"] == 1
    assert result["exact_64_segments"] == 8 and result["exact_128_blocks"] == 4
    assert result["consecutive_query_row_set_reuse"]["mean"] == 1


def test_same_set_different_order_is_not_tile_reuse():
    result = analyze_arrays(fixture(np.arange(511, -1, -1)), 64 * 32)
    assert result["consecutive_query_row_set_reuse"]["mean"] == 1
    assert result["exact_64_segments"] == result["exact_128_blocks"] == 0
    assert result["consecutive_query_same_slot_fraction"]["mean"] == 0
    assert result["contiguous_runs_within_existing_64_row_lane_segments"] == 8 + 512


def test_holes_are_not_counted_as_payload_or_contiguous_runs():
    result = analyze_arrays(fixture(bad_page=True), 64 * 32)
    assert result["valid_compressed_rows"] == 960
    assert result["contiguous_runs_within_existing_64_row_lane_segments"] == 16


def test_physical_plan_must_match_logical_topk_and_current_pages():
    data = fixture()
    data["physical_indices"][0, 30] += 1
    with pytest.raises(AssertionError):
        analyze_arrays(data, 64 * 32)
