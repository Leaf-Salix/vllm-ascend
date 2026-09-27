# SPDX-License-Identifier: Apache-2.0
"""Torch bridge from Native pages to request-major contiguous key/scale rows.

Copies execute on the caller's stream and are captured in ACL Graph. Only the
current compact slots are committed back; history is never copied back in full.
FP16 scales retain the Native rounding contract. Score tiles widen to FP32.
"""

import torch

from .config import INDEXER_NATIVE_CUBE_MIN_ROWS

INDEXER_PAGE_ROWS = 32
SCORE_DIRECT_TILE_ROWS = 384
SCORE_BUFFERED_TILE_ROWS = 1024  # 覆盖长历史Score的最大物理读取范围。


class SplitIndexerCache:
    def __init__(self, key: torch.Tensor, scale: torch.Tensor, *, batch_capacity: int, table_columns: int):
        if key.dtype != torch.int8 or tuple(key.shape[1:]) != (32, 1, 128):
            raise ValueError("Expected INT8 indexer keys [pages, 32, 1, 128]")
        if scale.dtype != torch.float16 or tuple(scale.shape) != (key.shape[0], 32, 1, 1):
            raise ValueError("Expected FP16 indexer scales [pages, 32, 1, 1]")
        if key.device != scale.device:
            raise ValueError("Indexer key and scale must be on the same device")
        self.native_key, self.native_scale = key, scale
        self.batch_capacity = batch_capacity
        self.table_columns = table_columns
        # The last tile can have only one valid candidate. Reserve a full tile
        # beyond the table width so both lanes remain inside the allocation.
        score_tile = (SCORE_BUFFERED_TILE_ROWS if table_columns * INDEXER_PAGE_ROWS >= INDEXER_NATIVE_CUBE_MIN_ROWS
                      else SCORE_DIRECT_TILE_ROWS)
        tail_pages = (score_tile - 1 + INDEXER_PAGE_ROWS - 1) // INDEXER_PAGE_ROWS
        self.request_pages = table_columns + tail_pages
        self.request_rows = self.request_pages * 32
        pages = batch_capacity * self.request_pages
        self.key = torch.empty((pages, 32, 1, 128), dtype=key.dtype, device=key.device)
        self.scale = torch.empty((pages, 32, 1, 1), dtype=scale.dtype, device=scale.device)
        self.page_indices = torch.arange(self.request_pages, dtype=torch.int64, device=key.device).view(1, -1)

    def matches(self, key: torch.Tensor, scale: torch.Tensor) -> bool:
        return all(
            source.device == target.device and source.dtype == target.dtype
            and source.shape == target.shape and source.stride() == target.stride()
            and source.data_ptr() == target.data_ptr()
            for source, target in ((key, self.native_key), (scale, self.native_scale))
        )

    def views(self, batch: int):
        if not 1 <= batch <= self.batch_capacity:
            raise ValueError("Logical indexer cache exceeds its warmed batch capacity")
        pages = batch * self.request_pages
        return self.key[:pages], self.scale[:pages]

    def load(self, block_table: torch.Tensor, seq_lens: torch.Tensor):
        if block_table.shape[1] != self.table_columns:
            raise ValueError("Logical indexer cache page-table width changed after allocation")
        key, scale = self.views(seq_lens.numel())
        last_page = ((seq_lens.to(torch.int64) // 4 - 1).clamp_min(0) // 32).view(-1, 1)
        logical_pages = torch.minimum(self.page_indices, last_page)
        physical_pages = torch.gather(block_table, 1, logical_pages).clamp_min(0).reshape(-1)
        torch.index_select(self.native_key, 0, physical_pages, out=key)
        torch.index_select(self.native_scale, 0, physical_pages, out=scale)

    def slot_updates(self, slots: torch.Tensor, seq_lens: torch.Tensor, query_bounds: torch.Tensor):
        # Reproduce Native compact-row ownership, including zero-token padding
        # requests. Preserve Native physical slots for the final scatter.
        starts = seq_lens - (query_bounds[1:] - query_bounds[:-1])
        counts = seq_lens // 4 - starts // 4
        ends = counts.cumsum(0)
        metadata_rows = torch.arange(slots.shape[0], device=slots.device, dtype=torch.int64)
        requests = torch.searchsorted(ends, metadata_rows, right=True).clamp_max(seq_lens.numel() - 1)
        prefix = ends - counts
        rows = requests * self.request_rows + (starts // 4)[requests] + metadata_rows - prefix[requests]
        valid = (slots[:, 0] >= 0) & (slots[:, 1] >= 0)
        rows = torch.where(valid, rows, 0)
        keys = torch.index_select(self.key.view(-1, 128), 0, rows).view(-1, 1, 128)
        scales = torch.index_select(self.scale.view(-1, 1), 0, rows).view(-1, 1, 1)
        return keys, scales

    def commit(self, slots: torch.Tensor, seq_lens: torch.Tensor, query_bounds: torch.Tensor):
        keys, scales = self.slot_updates(slots, seq_lens, query_bounds)
        torch.ops._C_ascend.npu_scatter_nd_update_v2(self.native_key, slots, keys)
        torch.ops._C_ascend.npu_scatter_nd_update_v2(self.native_scale, slots, scales)
