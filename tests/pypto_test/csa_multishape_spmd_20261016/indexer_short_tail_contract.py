# SPDX-License-Identifier: Apache-2.0
"""CPU核对未写score_transfer尾部不会进入原AIV TopK的有效前缀。"""

import argparse
import ast
import json
from pathlib import Path


def check(body_source):
    module = ast.parse(Path(body_source).read_text())
    vector = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)
                  and n.name == "indexer_score_topk_native_pair_aiv")
    expressions = {ast.unparse(n) for n in ast.walk(vector)}
    for query in range(3):
        for tile, prefix in [(512, "tiny"), (1024, "small"), (2048, "large")]:
            valid = f"single_segment_valid_inline8659__ssa_v{query}"
            expected = [x for x in expressions if x.startswith("pl.tile.set_validshape(")
                        and x.endswith(f", 1, {valid})") and x.startswith(f"pl.tile.set_validshape({prefix}_tile")]
            if len(expected) != 1:
                raise ValueError(f"query{query}的{tile}项TopK尾部mask合同改变")
        formula = (f"pl.max(pl.min(single_query_visible_inline8891__ssa_v{query} - "
                   "single_segment_begin_inline8760__ssa_v0, single_segment_span_inline8784__ssa_v0), 0)")
        if formula not in expressions:
            raise ValueError("各query的有效前缀不再按原公式裁剪")
    if sum(isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.tile.fillpad"
           for n in ast.walk(vector)) != 9:
        raise ValueError("原九条排序分支没有保持fillpad")
    checked = 0
    fast_cases = 0
    for group_valid in range(1, 8193):
        steps = (group_valid + 1023) // 1024
        # 每个query不晚于组内最后一个query；这是原S6分组的前提。
        # 0/1与可见末端邻域同时覆盖完全不可见query和常规S6压缩因果差。
        for query_valid in {0, 1, max(0, group_valid - 2), max(0, group_valid - 1), group_valid}:
            query_valid = min(query_valid, group_valid)
            for step in range(steps):
                remaining = group_valid - step * 1024
                short = remaining <= 64
                if short:
                    assert 1 <= remaining <= 64 and step == steps - 1
                    # 该步未写[64,1024)，排序前缀不能包含其中任何元素。
                    invalid_begin = step * 1024 + 64
                    assert query_valid <= invalid_begin
                    fast_cases += 1
                if step % 2 == 1 or step == steps - 1:
                    segment_begin = step // 2 * 2048
                    span = (step % 2 + 1) * 1024
                    segment_valid = max(min(query_valid - segment_begin, span), 0)
                    if short:
                        short_offset = (step % 2) * 1024 + 64
                        assert segment_valid <= short_offset
                checked += 1
    return {"leaf_valid_counts": [1, 8192], "query_step_cases_checked": checked,
            "short_tail_cases_checked": fast_cases, "aiv_fillpad_branches_checked": 9,
            "unwritten_tail_excluded_from_effective_sort_prefix": True,
            "scope": "CPU区间与原源码合同；不代替FP数学、设备同步或性能验证",
            "precondition": "沿用原S6分组有序位置，组内最后query具有最大可见候选数"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.source)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
