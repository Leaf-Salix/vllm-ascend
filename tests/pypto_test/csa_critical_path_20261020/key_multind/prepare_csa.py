# SPDX-License-Identifier: Apache-2.0
"""在当前正式Score中接入多ND直接供数候选；数学/事件/形状策略保持。"""

import ast
import json
from pathlib import Path
import shutil
import textwrap

HERE = Path(__file__).resolve().parent

PAIR_HELPER = '''@pl.jit.inline(auto_scope=False)
def _indexer_key_pair_mat(
    cache: pl.Tensor[[CACHE_PAGES_DYN, PAGE_BYTES_DYN], pl.INT8],
    key0: pl.Tensor[[KEY0_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    key64: pl.Tensor[[KEY64_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    table: pl.Tensor[[REQUESTS_DYN, TABLE_COLUMNS_DYN], pl.INT32],
    request: pl.Scalar[pl.INDEX],
    key_row: pl.Scalar[pl.INDEX],
    cache_length: pl.Scalar[pl.INDEX],
):
    """两页正序相邻且stride可编码时一次多ND；其余保留原逐页读取。

    两个N32页构成既有N64面板，与B/S具体值无关。物理页及Native页尾布局
    均来自本轮device数据；尾页重复钳制不能走两页快路。Mat直接送原Right
    预取槽，不再组装到128行父Mat以免产生A3不支持的Mat→Mat复制。
    """
    native_pages = pl.tensor.dim(cache, 0)
    native_pitch = pl.tensor.dim(cache, 1)
    table_columns = pl.tensor.dim(table, 1)
    logical0 = pl.min(key_row // BLOCK_SIZE, pl.max((cache_length - 1) // BLOCK_SIZE, 0))
    logical1 = pl.min(key_row // BLOCK_SIZE + 1, pl.max((cache_length - 1) // BLOCK_SIZE, 0))
    physical0 = pl.cast(-1, pl.INDEX)
    physical1 = pl.cast(-1, pl.INDEX)
    if logical0 >= 0 and logical0 < table_columns:
        physical0 = pl.cast(pl.read(table, [request, logical0]), pl.INDEX)
    if logical1 >= 0 and logical1 < table_columns:
        physical1 = pl.cast(pl.read(table, [request, logical1]), pl.INDEX)
    pair_mat = pl.tile.create([64, IDX_HEAD_DIM], dtype=pl.INT8, target_memory=pl.MemorySpace.Mat)
    if logical1 == logical0 + 1 and physical0 >= 0 and physical1 == physical0 + 1 and physical1 < native_pages and native_pitch >= 4160 and native_pitch <= 65535:
        # 保留原连续根，核内metadata视图跳过每页scale尾部。这里不做任何GM复制。
        pair_view = pl.tensor.view(cache, [native_pages, BLOCK_SIZE, IDX_HEAD_DIM],
                                   strides=[native_pitch, IDX_HEAD_DIM, 1])
        pair_nd = pl.load(pair_view, [physical0, 0, 0], [2, BLOCK_SIZE, IDX_HEAD_DIM],
                          target_memory=pl.MemorySpace.Mat)
        pair_mat = pl.tile.reshape(pair_nd, [64, IDX_HEAD_DIM])
    else:
        # 不连续、逆序、重复末页或超编码范围时，保持原两页次序和容量保护。
        # 已读页号直接复用，避免fallback再次读取Native页表。
        for page_lane in pl.unroll(2):
            physical = physical0
            if page_lane == 1:
                physical = physical1
            if physical >= 0 and physical < native_pages:
                native_byte = physical * native_pitch
                if native_byte % IDX_HEAD_DIM == 0:
                    pair_mat = pl.gather_row(pair_mat, key0, [page_lane * BLOCK_SIZE, 0],
                                              [native_byte // IDX_HEAD_DIM, 0], [BLOCK_SIZE, IDX_HEAD_DIM])
                else:
                    pair_mat = pl.gather_row(pair_mat, key64, [page_lane * BLOCK_SIZE, 0],
                                              [native_byte // IDX_HEAD_DIM, 0], [BLOCK_SIZE, IDX_HEAD_DIM])
    return pair_mat


'''


def main():
    source = HERE / "source_candidate"
    source.mkdir(exist_ok=False)
    for path in (HERE.parent / "source_after_score_split").glob("*.py"):
        shutil.copyfile(path, source / path.name)
    path = source / "indexer_score_stages.py"
    text = path.read_text()
    first = text.index("def _indexer_score_cube_split_store(")
    last = text.index("@pl.jit.inline(auto_scope=False)\ndef indexer_score_cube_stage", first)
    body = text[first:last]
    a = body.index("                if key_prefetch_panels > 0:\n                    buf_prime_col = 0")
    b = body.index("                # 固定panel顺序：", a)
    original = body[a:b]
    replacement = '''                if key_prefetch_panels == 1 and key_prefetch_to_l0:
                    prime_pair = _indexer_key_pair_mat(
                        idx_native_kv_cache, idx_kv_cache, idx_kv_cache_shift64, idx_block_table,
                        buf_batch_idx, buf_logical_begin + buf_score_begin, buf_cache_len,
                    )
                    buf_current_key = pl.tile.move(pl.tile.transpose_view(prime_pair), target_memory=pl.MemorySpace.Right)
                else:
'''
    body = body[:a] + replacement + textwrap.indent(original, "    ") + body[b:]
    a = body.index("                    if buf_qk_panel + key_prefetch_panels < score_tile // qk_panel_cols:")
    start = body.index("\n", a) + 1
    end = body.index("                    if key_prefetch_panels > 0:\n                        buf_key_panel", start)
    original = body[start:end]
    replacement = '''                        if key_prefetch_panels == 1 and key_prefetch_to_l0:
                            next_pair = _indexer_key_pair_mat(
                                idx_native_kv_cache, idx_kv_cache, idx_kv_cache_shift64, idx_block_table,
                                buf_batch_idx, buf_logical_begin + buf_score_begin + buf_load_col, buf_cache_len,
                            )
                            buf_prefetched_key = pl.tile.move(pl.tile.transpose_view(next_pair), target_memory=pl.MemorySpace.Right)
                        else:
'''
    body = body[:start] + replacement + textwrap.indent(original, "    ") + body[end:]
    text = text[:first] + body + text[last:]
    marker = "@pl.jit.inline(auto_scope=False)\ndef _indexer_score_cube_split_store"
    assert text.count(marker) == 1
    text = text.replace(marker, PAIR_HELPER + marker)
    ast.parse(text)
    path.write_text(text)
    for batch in (16, 28):
        shutil.copyfile(HERE.parent / "score_split_pool" / f"controls_b{batch}.json", HERE / f"controls_b{batch}.json")
    (HERE / "CSA_PREPARATION.json").write_text(json.dumps({
        "reference": "source_after_score_split/13259f92", "files": [path.name],
        "scope": "existing long G6 split-score N64 with one-panel L0 prefetch",
        "fast_path": "device positive adjacent pages and 4160<=page_bytes<=65535",
        "math_events_policy_unchanged": True, "mat_parent_assembly": False,
        "device_or_performance_proven": False,
    }, ensure_ascii=False, indent=2) + "\n")
    print("complete 37-source candidate prepared; only key supply in split helper changes")


if __name__ == "__main__":
    main()
