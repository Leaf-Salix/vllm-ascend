from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend.ops import dsv4_csa_attention


def test_native_forward_fallback_preserves_attention_interface():
    positions = torch.tensor([3], dtype=torch.int64)
    hidden_states = torch.randn(1, 4)
    scaling = torch.tensor([1.0])
    native_output = torch.randn_like(hidden_states)
    attention = Mock()
    attention.dsa_attn.return_value = native_output

    with patch.object(dsv4_csa_attention, "_try_pypto", return_value=None):
        output = dsv4_csa_attention.forward(attention, positions, hidden_states, scaling)

    assert output is native_output
    attention.dsa_attn.assert_called_once_with(positions, hidden_states, scaling)


def test_compressed_row_mapping_ignores_rectangular_padding():
    # The old adapter counted each replicated boundary as a new compressed row.
    positions = torch.tensor([3, 7, 8], dtype=torch.int64)
    src, _ = dsv4_csa_attention.rectangular(len(positions), 8, positions.device)

    boundary, row = dsv4_csa_attention._expand_compressed_rows(positions, src)

    assert row[::8].tolist() == [0, 1, 1]
    assert row[8:16].tolist() == [1] * 8
    assert boundary[::8].tolist() == [True, True, False]


def test_state_writeback_preserves_null_page_and_padding():
    block_table = torch.arange(1, 17, dtype=torch.int32).unsqueeze(0)
    positions = torch.full((8,), 0, dtype=torch.int64)
    cache = torch.arange(20 * 2 * 4, dtype=torch.float32).reshape(20, 2, 1, 4)
    expected = cache.clone()
    ring = torch.arange(16 * 4, dtype=torch.float32).reshape(8, 2, 4) + 1000

    with patch.object(dsv4_csa_attention, "kernel", return_value=(SimpleNamespace(MAIN_STATE_STORAGE_LEN=16), None)):
        plan = dsv4_csa_attention.state_ring_plan(positions, 8, block_table)
        block, intra, row, valid = plan
        for i in range(valid.numel()):
            if valid[i]:
                expected[block[i], intra[i], 0] = ring.reshape(-1, 4)[row[i]]
        dsv4_csa_attention.write_state_ring(cache, ring, plan, 8, positions, 4)

    torch.testing.assert_close(cache, expected)


def test_shared_indexer_page_commits_key_and_scale_independently():
    # A3 stores key and scale as two strided views of the same physical page.
    physical = torch.zeros((3, 32, 1, 129), dtype=torch.float32)
    block_table = torch.tensor([[1, 2]], dtype=torch.int32)
    slot = torch.tensor([37], dtype=torch.int64)
    request = torch.tensor([0], dtype=torch.int64)

    key = dsv4_csa_attention.repage_kv(physical[..., :128], block_table, 4)
    scale = dsv4_csa_attention.repage_kv(physical[..., 128:], block_table, 4)
    key_slot = key.remap(slot, request)
    scale.track(slot, request)
    scale_slot, _ = scale._located(slot, request)
    key.view.reshape(-1, 1, 128)[key_slot[0], 0, 0] = 7
    scale.view.reshape(-1, 1, 1)[scale_slot[0], 0, 0] = 3

    key.commit()
    scale.commit()

    assert physical[1, 5, 0, 0] == 7
    assert physical[1, 5, 0, 128] == 3
