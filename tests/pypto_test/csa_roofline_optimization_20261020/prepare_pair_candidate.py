# SPDX-License-Identifier: Apache-2.0
"""保留v5历史，创建使用显式三输出成对压缩的完整动态SPMD候选。"""
import ast
from pathlib import Path
import shutil


PAIRED_SORT = '''
@pl.jit.inline(auto_scope=False)
def sort_compacted_prefix(
    values: pl.Tile[[1, 1024], pl.FP32],
    indices: pl.Tile[[1, 1024], pl.INT32],
    kept: pl.Scalar[pl.INT32],
):
    """只排序有效压缩前缀；索引已含全局基址，无需随机gather或再加偏移。"""
    if kept <= 512:
        tiny_v = pl.tile.extract(values, 0, 0, [1, 512], target_memory=pl.MemorySpace.Vec)
        tiny_v = pl.set_validshape(tiny_v, 1, pl.cast(kept, pl.INDEX))
        tiny_v = pl.maximum(pl.fillpad(tiny_v, pad_value=pl.PadValue.min),
                            NEGATIVE_SENTINEL)
        tiny_i = pl.tile.slice(indices, [1, 512], [0, 0])
        tiny_pairs = pl.sort32(tiny_v, pl.reinterpret_view(tiny_i, pl.UINT32))
        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=64)
        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=256)
        root = pl.tile.extract(tiny_pairs, 0, 0, [1, PAIR_WIDTH],
                               target_memory=pl.MemorySpace.Vec)
    else:
        small_v = pl.set_validshape(values, 1, pl.cast(kept, pl.INDEX))
        small_v = pl.maximum(pl.fillpad(small_v, pad_value=pl.PadValue.min),
                             NEGATIVE_SENTINEL)
        small_pairs = pl.sort32(small_v, pl.reinterpret_view(indices, pl.UINT32))
        small_pairs = pl.mrgsort(small_pairs, block_len=64)
        small_pairs = pl.mrgsort(small_pairs, block_len=256)
        left = pl.tile.slice(small_pairs, [1, PAIR_WIDTH], [0, 0])
        right = pl.tile.slice(small_pairs, [1, PAIR_WIDTH], [0, PAIR_WIDTH])
        tmp = pl.tile.create([1, 2 * PAIR_WIDTH], dtype=pl.FP32)
        merged = pl.tile.mrgsort(left, right, tmp=tmp)
        root = pl.tile.extract(merged, 0, 0, [1, PAIR_WIDTH],
                               target_memory=pl.MemorySpace.Vec)
    return root


'''


def main():
    root = Path(__file__).resolve().parent
    target = root / "source_topk_paired"
    target.mkdir(exist_ok=False)
    for source in (root / "source_topk_payload").glob("*.py"):
        shutil.copy2(source, target / source.name)
    helper = target / "topk_threshold_stages.py"
    text = helper.read_text()
    first = text.index("@pl.jit.inline(auto_scope=False)\ndef gather_sorted_prefix(")
    last = text.index("@pl.jit.inline(auto_scope=False)\ndef threshold_merge_top512(", first)
    text = text[:first] + PAIRED_SORT + text[last:]
    original = '''        selected, count = pl.tile.gather_compare(
            valid_values, threshold, compare_tmp, cmp_mode="ge", out_cols=MAX_SEGMENT,
        )'''
    replacement = '''        # 同一掩码保序压缩score和INT32索引，容量1024但count不截断。
        # 超容量时保留原完整排序；所有压缩成本纳入整层测量。
        packed_values, selected, count = pl.tile.compact_compare(
            valid_values, threshold, compare_tmp, logical_begin,
            cmp_mode="ge", out_cols=1024,
        )'''
    assert original in text
    text = text.replace(original, replacement)
    text = text.replace(
        "selected = pl.tile.full([1, MAX_SEGMENT], dtype=pl.INT32, value=0)",
        "selected = pl.tile.full([1, 1024], dtype=pl.INT32, value=0)\n"
        "        packed_values = pl.tile.full([1, 1024], dtype=pl.FP32, value=0)",
    )
    text = text.replace(
        "new_root = gather_sorted_prefix(values, selected, logical_begin, kept)",
        "new_root = sort_compacted_prefix(packed_values, selected, kept)",
    )
    text = text.replace(
        "    首版完整B16/B28精度通过但未达性能门槛；本版缩短gather容量，\n"
        "    需要重新验证。正式算子未改，不能沿用首版PASS作为本版证据。",
        "    本版直接保序压缩score与索引，移除索引补齐、物化和随机gather。\n"
        "    新增一遍连续score压缩，是否获益待完整实测，不能沿用v5的PASS。",
    )
    ast.parse(text)
    helper.write_text(text)


if __name__ == "__main__":
    main()
