# SPDX-License-Identifier: Apache-2.0
"""Request-major cache: permuted pages, padding, refresh, and physical slot commits."""

import torch

from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.indexer_cache import SplitIndexerCache


def cache_views():
    page_bytes, prefix, pages = 4224, 64, 6
    raw = torch.full((prefix + pages * page_bytes + 64,), 37, dtype=torch.int8)
    key = raw.as_strided((pages, 32, 1, 128), (page_bytes, 128, 128, 1), prefix)
    scale = raw.view(torch.float16).as_strided((pages, 32, 1, 1), (page_bytes // 2, 1, 1, 1),
                                             (prefix + 4096) // 2)
    key.copy_((torch.arange(key.numel()).reshape(key.shape) % 127 - 63).to(torch.int8))
    scale.copy_((torch.arange(scale.numel()).reshape(scale.shape) / 128 + 0.125).half())
    return raw, key, scale


def metadata():
    table = torch.tensor([[2, 4, 0], [1, 3, 5], [-1, -1, -1]], dtype=torch.int32)
    lengths = torch.tensor([132, 262, 0], dtype=torch.int32)
    bounds = torch.tensor([0, 6, 12, 12], dtype=torch.int32)
    return table, lengths, bounds


def test_split_cache_orders_logical_pages_and_refreshes_without_aliasing():
    raw, key, scale = cache_views()
    before = raw.clone()
    table, lengths, _ = metadata()
    cache = SplitIndexerCache(key, scale, batch_capacity=3, table_columns=3)
    cache.load(table, lengths)
    assert cache.key.is_contiguous() and cache.scale.is_contiguous()
    # Even a 33-candidate request issues a whole 384-candidate Score tile.
    assert cache.key.view(3, cache.request_rows, 128)[-1, :384].shape[0] == 384
    # Physical pages [2,4] and [1,3,5] become contiguous request-local ranges.
    # Tail pages repeat the last visible page; the padding request is safe.
    expected_pages = ([2] + [4] * (cache.request_pages - 1)
                      + [1, 3] + [5] * (cache.request_pages - 2) + [0] * cache.request_pages)
    assert torch.equal(cache.key, key[expected_pages])
    assert torch.equal(cache.scale, scale[expected_pages])
    assert torch.equal(raw, before)
    cache.key[0, 31].fill_(9)
    assert not torch.equal(cache.key[0], key[2])
    key[1, 7].fill_(23)
    scale[1, 7].fill_(0.75)
    table[1, 0] = 0
    cache.load(table, lengths)
    expected_pages[cache.request_pages] = 0
    assert torch.equal(cache.key, key[expected_pages])
    assert torch.equal(cache.scale, scale[expected_pages])
    assert cache.matches(key, scale)
    assert not cache.matches(key.clone(), scale.clone())


def test_split_cache_commits_only_current_valid_slots(monkeypatch):
    raw, key, scale = cache_views()
    table, lengths, bounds = metadata()
    cache = SplitIndexerCache(key, scale, batch_capacity=3, table_columns=3)
    cache.load(table, lengths)
    slots = torch.tensor([[2, 31], [4, 0], [5, 0], [-1, -1]], dtype=torch.int32)
    expected = raw.clone()
    for logical, page, row, value in ((31, 2, 31, 11), (32, 4, 0, 17), (cache.request_rows + 64, 5, 0, 23)):
        cache.key.view(-1, 128)[logical].fill_(value)
        cache.scale.view(-1)[logical] = value / 32
        start = key.storage_offset() + page * key.stride(0) + row * 128
        expected[start:start + 128].fill_(value)
        offset = scale.storage_offset() + page * scale.stride(0) + row
        expected.view(torch.float16)[offset] = value / 32

    def reference_scatter(target, indices, values):
        for index, (page, row) in enumerate(indices.tolist()):
            if page >= 0 and row >= 0:
                target[page, row].copy_(values[index])

    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_v2", reference_scatter, raising=False)
    cache.commit(slots, lengths, bounds)
    assert torch.equal(raw, expected)
