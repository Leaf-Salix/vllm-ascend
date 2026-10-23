# SPDX-License-Identifier: Apache-2.0
"""Precheck the new state validator's selected holes and padding oracle."""

import numpy as np
import pytest
import torch
from capacity_state_observation import invalidate_selected_pages, physical_mapping


@pytest.mark.parametrize("invalid", [-1, 4096])
def test_selected_holes_exclude_compressor_write_page_and_restore(invalid):
    table = torch.arange(2051, dtype=torch.int32).repeat(2, 1)
    metadata = dict(idx_query_start_loc=torch.tensor([0, 6, 12]), cmp_block_table=table)
    topk = torch.arange(512, dtype=torch.int32).repeat(12, 1)
    # Add a selected historical row near the boundary, but keep column 2048 intact.
    topk[0, 511] = 65535
    original = table.clone()
    witness = invalidate_selected_pages(metadata, topk, [262144, 262144], 4096, invalid)
    assert len(witness["selected"]) == 2
    assert torch.equal(table[:, 2048:], original[:, 2048:])
    assert table[0, 2047] == invalid and table[1, 2047] == original[1, 2047]
    arrays = dict(
        logical_topk=topk.numpy(),
        idx_bounds=np.array([0, 6, 12]),
        idx_lengths=np.array([262150, 262150]),
        positions=np.tile(np.arange(262144, 262150), 2),
        compressed_table=table.numpy(),
    )
    assert np.all(physical_mapping(arrays, 4096 * 32)[[0, 6]] == -1)
    arrays["compressed_table"] = original.numpy()
    np.testing.assert_array_equal(physical_mapping(arrays, 4096 * 32), topk.numpy())


def test_oracle_does_not_assign_dummy_or_stale_rows_to_active_requests():
    arrays = dict(
        logical_topk=np.zeros((6, 512), dtype=np.int32),
        idx_bounds=np.array([0, 2, 6, 6]),
        idx_lengths=np.array([132, 0, 0]),
        positions=np.array([130, 999, 0, 0, 0, 0]),
        compressed_table=np.array([[7], [8], [9]]),
    )
    expected = physical_mapping(arrays, 10 * 32)
    assert np.all(expected[0] == 224) and np.all(expected[1:] == -1)
