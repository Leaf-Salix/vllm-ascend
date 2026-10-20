# SPDX-License-Identifier: Apache-2.0
"""从精度已通过的首版创建独立候选，只缩短筛选后实际gather的容量。

不改变GE、计数、原始score、同分顺序、首段或完整排序回退。512/1024是
保留数量对应的数学容量，不是batch单点白名单，kept仍从device获取。
"""
import ast
import json
from pathlib import Path
import shutil

PREFIX_HELPER = '''
@pl.jit.inline(auto_scope=False)
def gather_sorted_prefix(
    values: pl.Tile[[1, MAX_SEGMENT], pl.FP32],
    selected: pl.Tile[[1, MAX_SEGMENT], pl.INT32],
    logical_begin: pl.Scalar[pl.INT32],
    kept: pl.Scalar[pl.INT32],
):
    """只搬运将被实际排序的前缀容量，避免先gather2048项再丢弃尾部。"""
    # A3的INT32窄TEXTRACT不支持；窄view直接fillpad也有后端容量约束。
    # 因此保留原2048索引补零，只缩短随后真正耗时的score gather和全局索引偏移。
    # 短slice保留母块物理跨度，整数加零将其物化为512/1024容量的独立Tile；
    # 索引不经过浮点转换，额外Vector复制的成本也必须计入完整性能。
    selected_valid = pl.set_validshape(selected, 1, pl.cast(kept, pl.INDEX))
    selected_clean = pl.fillpad(selected_valid, pad_value=pl.PadValue.zero)
    selected_clean = pl.set_validshape(selected_clean, 1, MAX_SEGMENT)
    if kept <= 512:
        # 只消费压缩结果的有效前缀；补零的索引安全指向原段第0项。
        # 排序前再把无效score置最小值，旧根已满512项，伪尾不能进入结果。
        tiny_view = pl.tile.slice(selected_clean, [1, 512], [0, 0])
        tiny_clean = pl.add(tiny_view, pl.const(0, pl.INT32))
        tiny_tmp = pl.tile.create([1, 512], dtype=pl.INT32)
        tiny_values = pl.tile.gather(values, tiny_clean, tiny_tmp)
        tiny_global = pl.add(tiny_clean, logical_begin)
        tiny_values = pl.set_validshape(tiny_values, 1, pl.cast(kept, pl.INDEX))
        tiny_values = pl.maximum(pl.fillpad(tiny_values, pad_value=pl.PadValue.min),
                                 NEGATIVE_SENTINEL)
        tiny_pairs = pl.sort32(tiny_values, pl.reinterpret_view(tiny_global, pl.UINT32))
        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=64)
        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=256)
        root = pl.tile.extract(tiny_pairs, 0, 0, [1, PAIR_WIDTH],
                               target_memory=pl.MemorySpace.Vec)
    else:
        # 513～1024项沿用原1024排序树及左段优先的归并顺序。
        small_view = pl.tile.slice(selected_clean, [1, 1024], [0, 0])
        small_clean = pl.add(small_view, pl.const(0, pl.INT32))
        small_tmp = pl.tile.create([1, 1024], dtype=pl.INT32)
        small_values = pl.tile.gather(values, small_clean, small_tmp)
        small_global = pl.add(small_clean, logical_begin)
        small_values = pl.set_validshape(small_values, 1, pl.cast(kept, pl.INDEX))
        small_values = pl.maximum(pl.fillpad(small_values, pad_value=pl.PadValue.min),
                                  NEGATIVE_SENTINEL)
        small_pairs = pl.sort32(small_values, pl.reinterpret_view(small_global, pl.UINT32))
        small_pairs = pl.mrgsort(small_pairs, block_len=64)
        small_pairs = pl.mrgsort(small_pairs, block_len=256)
        small_left = pl.tile.slice(small_pairs, [1, PAIR_WIDTH], [0, 0])
        small_right = pl.tile.slice(small_pairs, [1, PAIR_WIDTH], [0, PAIR_WIDTH])
        small_merge_tmp = pl.tile.create([1, 2 * PAIR_WIDTH], dtype=pl.FP32)
        small_merged = pl.tile.mrgsort(small_left, small_right, tmp=small_merge_tmp)
        root = pl.tile.extract(small_merged, 0, 0, [1, PAIR_WIDTH],
                               target_memory=pl.MemorySpace.Vec)
    return root


'''


def main():
    here = Path(__file__).resolve().parent
    target = here / "source_topk_payload"
    if target.exists():
        raise RuntimeError("不覆盖已经记录的候选")
    target.mkdir()
    for source in (here / "source_topk_candidate").glob("*.py"):
        shutil.copy2(source, target / source.name)
    helper = target / "topk_threshold_stages.py"
    text = helper.read_text()
    anchor = "@pl.jit.inline(auto_scope=False)\ndef threshold_merge_top512("
    assert text.count(anchor) == 1
    text = text.replace(anchor, PREFIX_HELPER + anchor)
    start = text.index("            # 只读 gather_compare 写入的有效前缀")
    end = text.index("        else:\n            # 只在原排序回退时", start)
    text = text[:start] + (
        "            new_root = gather_sorted_prefix(values, selected, logical_begin, kept)\n"
    ) + text[end:]
    text = text.replace(
        "    46组 A3 排序 probe 支持保序压缩的同分顺序，尚不代替本函数/完整\n"
        "    CSA 的设备验证。本文件是测试候选，未采用到正式 ops。",
        "    首版完整B16/B28精度通过但未达性能门槛；本版缩短gather容量，\n"
        "    需要重新验证。正式算子未改，不能沿用首版PASS作为本版证据。"
    )
    ast.parse(text)
    helper.write_text(text)
    (here / "PAYLOAD_PREPARATION.json").write_text(json.dumps({
        "candidate": str(target),
        "based_on": "source_topk_candidate",
        "changed_files": ["topk_threshold_stages.py"],
        "formal_changed": False,
        "previous_b16_net_saved_us": -0.46,
        "previous_b28_net_saved_us": 9.10,
        "new_mechanism": "压缩后gather、索引偏移及fillpad处理512/1024项，原版均处理2048项",
        "new_component_change": False,
        "cpu_compiled": False, "device_validated": False, "performance_measured": False,
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
