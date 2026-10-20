# SPDX-License-Identifier: Apache-2.0
"""在已修复同分的原半叶候选上，取消分数的 GM 写回和重读。

完整保留六条长度分支、Sort32/MrgSort 顺序、输入优先顺序及空根规则。
新 helper 只把原 GM load 改成等宽 UB extract + 相同 valid_shape；
不把不同长度统一补齐到4096后排序，不恢复已失败的流式归并。
"""

import ast
import json
from pathlib import Path
import re
import shutil


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"上下文数量不唯一：{old[:100]}")
    return text.replace(old, new)


def main():
    here = Path(__file__).resolve().parent
    source = here / "source_b1_legacy_half"
    target = here / "source_b1_half_score_resident"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "indexer_topk_stages.py"
    text = path.read_text()
    start = text.index("@pl.jit.inline(auto_scope=False)\ndef indexer_topk_half_leaf(")
    end = text.index("\n\n@pl.jit.inline", start + 1)
    helper = text[start:end]
    helper = once(helper, "def indexer_topk_half_leaf(", "def indexer_topk_half_leaf_resident(")
    helper = once(
        helper,
        "    score_arena: pl.Tensor[[SCORE_ROWS_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32],\n",
        "    half_scores: pl.Tile[[1, 4096], pl.FP32],\n",
    )
    helper = once(helper, "    score_row: pl.Scalar[pl.INDEX],\n", "")
    helper = once(
        helper,
        '    """连续半叶1..4096候选转成512项(score,index)根，保持原分支同分顺序。"""',
        '    """UB内原半叶：六条长度分支、物理排序宽度及同分顺序均复现GM版。"""',
    )
    pattern = (
        r"(?m)^(        )(\w+) = pl\.load\(score_arena, \[score_row, 0\], "
        r"\[1, (\d+)\], valid_shape=\[1, valid_count\]\)$"
    )
    widths = []

    def replace_load(match):
        indent, name, width = match.groups()
        widths.append(int(width))
        return (
            f"{indent}{name} = pl.tile.extract(\n"
            f"{indent}    half_scores, 0, 0, [1, {width}], target_memory=pl.MemorySpace.Vec\n"
            f"{indent})\n"
            f"{indent}{name} = pl.set_validshape({name}, 1, valid_count)"
        )

    helper, count = re.subn(pattern, replace_load, helper)
    if count != 6 or widths != [512, 1024, 2048, 2560, 3072, 4096]:
        raise ValueError(f"六条原load分支不匹配：{count}, {widths}")
    text = text[:end] + "\n\n" + helper + text[end:]
    ast.parse(text)
    path.write_text(text)

    path = target / "indexer_score_stages.py"
    text = path.read_text()
    text = once(
        text,
        "from .indexer_topk_stages import PAIR_ROWS_DYN, SCORE_ROWS_DYN, indexer_topk_half_leaf",
        "from .indexer_topk_stages import (\n"
        "    PAIR_ROWS_DYN, SCORE_ROWS_DYN, indexer_topk_half_leaf,\n"
        "    indexer_topk_half_leaf_resident,\n"
        ")",
    )
    text = once(
        text,
        "                # balanced每AIV持G/2条完整query；连续两轮凑2048候选，尾轮仍按实际valid排序。",
        "                # B1每个AIV只负责一条query的原半叶，最多4096项。\n"
        "                # 在共同作用域声明，避免分支局部定义穿过循环的SSA作用域。\n"
        "                # G6不消费这个Tile，由constexpr消除其无用定义。\n"
        "                # 保留完整半叶后按原长度分支排序；不用流式2048归并。\n"
        "                legacy_half_scores = pl.tile.full(\n"
        "                    [TOPK_CANDIDATES_PER_LEAF // (2 * score_tile), score_tile], dtype=pl.FP32, value=FP32_NEG_INF\n"
        "                )\n"
        "                # G6仍按原流式2048候选处理，B1分支在下方单独保留整半叶。",
    )
    text = once(
        text,
        "                            # 复用旧score scratch，完整保留原半叶排序分支及同分顺序。\n"
        "                            # 每个AIV只负责一个query；row与其它worker/lane不相交。\n"
        "                            legacy_score_row = buf_worker * 2 * query_group_size + buf_score_lane * 3\n"
        "                            pl.store(single_scaled, [legacy_score_row, single_step * score_tile], score_arena)",
        "                            # 同一AIV连续收集1024分数，叶内无跨核消费者，\n"
        "                            # UB槽仅在本叶排序结束后复用，不需要GM发布。\n"
        "                            legacy_half_scores = pl.tile.assemble(\n"
        "                                legacy_half_scores, single_scaled, [single_step, 0]\n"
        "                            )",
    )
    text = once(
        text,
        "                                indexer_topk_half_leaf(\n"
        "                                    score_arena, pair_arena,\n"
        "                                    buf_worker * 2 * query_group_size + buf_score_lane * 3,\n"
        "                                    single_begin, legacy_valid, legacy_slot,\n"
        "                                )",
        "                                legacy_half_flat = pl.tile.reshape(legacy_half_scores, [1, TOPK_CANDIDATES_PER_LEAF // 2])\n"
        "                                indexer_topk_half_leaf_resident(\n"
        "                                    legacy_half_flat, pair_arena,\n"
        "                                    single_begin, legacy_valid, legacy_slot,\n"
        "                                )",
    )
    ast.parse(text)
    path.write_text(text)
    (here / "B1_HALF_SCORE_RESIDENT_SOURCE.json").write_text(json.dumps({
        "source": str(source),
        "candidate": str(target),
        "changed_files": ["indexer_score_stages.py", "indexer_topk_stages.py"],
        "replaced_load_widths": widths,
        "math_order_changed": False,
        "old_helper_preserved": True,
        "extra_live_half_score_bytes": 16384,
        "actual_ub_peak": "待完整编译检查，不能仅由新增Tile大小推导",
        "adopted": False,
    }, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
