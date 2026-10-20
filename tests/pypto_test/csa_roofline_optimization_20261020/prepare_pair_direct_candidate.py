# SPDX-License-Identifier: Apache-2.0
"""从已验证成对压缩候选派生直接输出版本，保留旧版证据不覆盖。

只改变压缩输出和scratch容量，以及短排序前必要的1024列视图。
分支条件、同分归并、score计算、状态写入和设备metadata均沿用旧版。
"""

import ast
from pathlib import Path
import shutil


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"候选源码上下文改变：{old[:90]}")
    return text.replace(old, new)


def main():
    here = Path(__file__).resolve().parent
    target = here / "source_topk_pair_direct"
    shutil.copytree(here / "source_topk_paired", target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    helper = target / "topk_threshold_stages.py"
    text = helper.read_text()
    text = replace_once(text, "COMPARE_SCRATCH_BYTES = 18432", "COMPARE_SCRATCH_BYTES = 9216")
    text = replace_once(
        text,
        "    values: pl.Tile[[1, 1024], pl.FP32],\n"
        "    indices: pl.Tile[[1, 1024], pl.INT32],",
        "    values: pl.Tile[[1, MAX_SEGMENT], pl.FP32],\n"
        "    indices: pl.Tile[[1, MAX_SEGMENT], pl.INT32],",
    )
    text = replace_once(
        text,
        "        small_v = pl.set_validshape(values, 1, pl.cast(kept, pl.INDEX))",
        "        # 输出容量扩大只为直接接收压缩结果；短排序仍限定1024列。\n"
        "        prefix = pl.tile.extract(values, 0, 0, [1, 1024], target_memory=pl.MemorySpace.Vec)\n"
        "        small_indices = pl.tile.slice(indices, [1, 1024], [0, 0])\n"
        "        small_v = pl.set_validshape(prefix, 1, pl.cast(kept, pl.INDEX))",
    )
    text = replace_once(
        text, "small_pairs = pl.sort32(small_v, pl.reinterpret_view(indices, pl.UINT32))",
        "small_pairs = pl.sort32(small_v, pl.reinterpret_view(small_indices, pl.UINT32))",
    )
    text = replace_once(text, 'cmp_mode="ge", out_cols=1024,', 'cmp_mode="ge", out_cols=MAX_SEGMENT,')
    text = text.replace(
        "pl.tile.full([1, 1024], dtype=", "pl.tile.full([1, MAX_SEGMENT], dtype="
    )
    text = text.replace(
        "        # 同一掩码保序压缩score和INT32索引，容量1024但count不截断。\n"
        "        # 超容量时保留原完整排序；所有压缩成本纳入整层测量。",
        "        # 输出容量等于源容量，后端可直接写DPS结果，消除两次暂存复制。\n"
        "        # kept>1024仍沿原完整排序回退，不改变短排序的选择或同分规则。",
    )
    text = text.replace(
        "    新增一遍连续score压缩，是否获益待完整实测，不能沿用v5的PASS。",
        "    本次扩大输出并缩小scratch，使压缩直接落到DPS输出；\n"
        "    索引块倍增由PTOAS实现，设备和完整L1收益必须重新验证。",
    )
    ast.parse(text)
    helper.write_text(text)
    # 下一轮同轮参考/候选统一150，不改变旧pair的100tick历史快照。
    shutil.copyfile(here / "source_after_wait150/wait_codegen.py", target / "wait_codegen.py")
    print(target)


if __name__ == "__main__":
    main()
