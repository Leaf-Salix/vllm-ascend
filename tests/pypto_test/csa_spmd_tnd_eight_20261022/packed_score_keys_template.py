# SPDX-License-Identifier: Apache-2.0
"""AIV supplies one half of a proven full key block through the idle G6 arena.

Only leaves with a complete number of 1024-row blocks enter this helper. The
caller proves all page IDs contiguous and in bounds for this invocation. Scale
remains private and is read through the original path at the consumption step.
"""

import pypto.language as pl

PACK_CACHE_PAGES = pl.dynamic("CSA_PACK_CACHE_PAGES")
PACK_PAGE_BYTES = pl.dynamic("CSA_PACK_PAGE_BYTES")
PACK_KEY_ROWS = pl.dynamic("CSA_PACK_KEY_ROWS")
PACK_TABLE_REQUESTS = pl.dynamic("CSA_PACK_TABLE_REQUESTS")
PACK_TABLE_COLS = pl.dynamic("CSA_PACK_TABLE_COLS")
PACK_READY_EVENT = 6
PACK_PAGE_KEY_BYTES = 4096
PACK_PAGES_PER_LANE = 16
PACK_ROWS_PER_STEP = 1024
PACK_HEAD_DIM = 128
PACK_RING_SLOTS = 2
PACK_LANE_BYTES = PACK_PAGES_PER_LANE * PACK_PAGE_KEY_BYTES
PACK_DMA_BLOCK_BYTES = 32
PACK_INDEX_BLOCKS = PACK_LANE_BYTES // PACK_DMA_BLOCK_BYTES
PACK_BLOCKS_PER_PAGE = PACK_PAGE_KEY_BYTES // PACK_DMA_BLOCK_BYTES
PACK_PROOF_MAX_PAGES = 256
PACK_PROOF_INTEGER_LIMIT = 16777216


@pl.jit.inline(auto_scope=False)
def packed_leaf_layout_scalar(
    table: pl.Tensor[[PACK_TABLE_REQUESTS, PACK_TABLE_COLS], pl.INT32],
    request: pl.Scalar[pl.INDEX],
    begin: pl.Scalar[pl.INDEX],
    pages: pl.Scalar[pl.INDEX],
    capacity: pl.Scalar[pl.INDEX],
):
    """AIC-only integer equivalent of the existing AIV leaf layout proof."""
    first_page = pl.cast(0, pl.INDEX)
    direction = pl.cast(0, pl.INT32)
    if request >= 0 and request < pl.tensor.dim(table, 0):
        if begin >= 0 and pages > 0 and pages <= PACK_PROOF_MAX_PAGES:
            if begin + pages <= pl.tensor.dim(table, 1):
                if capacity > 0 and capacity <= PACK_PROOF_INTEGER_LIMIT:
                    first = pl.cast(pl.read(table, [request, begin]), pl.INDEX)
                    last = pl.cast(pl.read(table, [request, begin + pages - 1]), pl.INDEX)
                    trial = pl.cast(1, pl.INT32)
                    if last < first:
                        trial = pl.cast(-1, pl.INT32)
                    expected_last = first + pl.cast(trial, pl.INDEX) * (pages - 1)
                    if last == expected_last:
                        if pl.min(first, expected_last) >= 0 and pl.max(first, expected_last) < capacity:
                            direction = trial
                            for page in pl.range(pages):
                                if direction != 0:
                                    actual = pl.cast(pl.read(table, [request, begin + page]), pl.INDEX)
                                    if actual != first + pl.cast(trial, pl.INDEX) * page:
                                        direction = pl.cast(0, pl.INT32)
                            if direction != 0:
                                first_page = first
    return first_page, direction


@pl.jit.inline(auto_scope=False)
def supply_packed_score_keys(
    native: pl.Tensor[[PACK_CACHE_PAGES, PACK_PAGE_BYTES], pl.INT8],
    ring: pl.InOut[pl.Tensor[[PACK_KEY_ROWS, PACK_HEAD_DIM], pl.INT8]],
    worker: pl.Scalar[pl.INDEX],
    lane: pl.Scalar[pl.INDEX],
    step: pl.Scalar[pl.INDEX],
    first_page: pl.Scalar[pl.INDEX],
    direction: pl.Scalar[pl.INT32],
):
    first = first_page + (step * 2 + lane) * PACK_PAGES_PER_LANE * pl.cast(direction, pl.INDEX)
    lowest = first
    if direction < 0:
        lowest = first - PACK_PAGES_PER_LANE + 1
    # The Native row stride includes the 64 scale bytes. This two-dimensional
    # DMA copies only the key prefix, with 16 bursts per lane.
    raw = pl.load(native, [lowest, 0], [PACK_PAGES_PER_LANE, PACK_PAGE_KEY_BYTES])
    flat = pl.reshape(pl.tile.reinterpret_view(raw, pl.FP16), [1, PACK_LANE_BYTES // 2])
    if direction < 0:
        blocks = pl.tile.ci(0, [1, PACK_INDEX_BLOCKS], dtype=pl.INT32)
        pages = pl.tile.shrs(blocks, 7)
        in_page = pl.sub(blocks, pl.mul(pages, PACK_BLOCKS_PER_PAGE))
        reverse_pages = pl.add(pl.neg(pages), PACK_PAGES_PER_LANE - 1)
        offsets = pl.add(pl.mul(reverse_pages, PACK_PAGE_KEY_BYTES), pl.mul(in_page, PACK_DMA_BLOCK_BYTES))
        reordered = pl.tile.gatherb(flat, pl.tile.reinterpret_view(offsets, pl.UINT32))
        ordered = pl.yield_(reordered)
    else:
        ordered = pl.yield_(flat)
    key = pl.reshape(pl.tile.reinterpret_view(ordered, pl.INT8), [PACK_ROWS_PER_STEP // 2, PACK_HEAD_DIM])
    row = (worker * PACK_RING_SLOTS + step % PACK_RING_SLOTS) * PACK_ROWS_PER_STEP + lane * (PACK_ROWS_PER_STEP // 2)
    pl.store(key, [row, 0], ring)
    pl.system.sync_set(PACK_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
    return ring
