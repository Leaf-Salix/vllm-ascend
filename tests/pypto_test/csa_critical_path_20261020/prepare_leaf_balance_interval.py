# SPDX-License-Identifier: Apache-2.0
"""按已有Host group上界裁掉不可能启用的叶平衡路径；实际group仍由device计算。"""
import ast
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FLAG = "ROUND_LEAF_BALANCE"


def once(source, old, new):
    assert source.count(old) == 1, old[:100]
    return source.replace(old, new)


def append_flag(source, definitions, calls):
    """按AST位置插入，保留原注释、空白及长签名；不整文件ast.unparse。"""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets, position = [], 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    edits = []
    seen_defs, seen_calls = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in definitions:
            last = node.args.defaults[-1] if node.args.defaults else node.args.args[-1]
            insert = offsets[last.end_lineno]
            edits.append((insert, f"    {FLAG}: pl.constexpr = False,\n"))
            seen_defs[node.name] = seen_defs.get(node.name, 0) + 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in calls:
            close = offsets[node.end_lineno - 1] + node.end_col_offset - 1
            assert source[close] == ")"
            if node.lineno == node.end_lineno:
                edits.append((close, f", {FLAG}"))
            else:
                closing_line_start = offsets[node.end_lineno - 1]
                before = source[:closing_line_start].rstrip()
                assert before.endswith(","), node.func.id
                indent = lines[node.args[-1].lineno - 1]
                indent = indent[:len(indent) - len(indent.lstrip())]
                edits.append((closing_line_start, f"{indent}{FLAG},\n"))
            seen_calls[node.func.id] = seen_calls.get(node.func.id, 0) + 1
    assert set(seen_defs) == set(definitions), (seen_defs, definitions)
    assert set(seen_calls) == set(calls), (seen_calls, calls)
    for at, text in sorted(edits, reverse=True):
        source = source[:at] + text + source[at:]
    ast.parse(source)
    return source, dict(definitions=seen_defs, calls=seen_calls)


def main():
    base = HERE / "source_leaf_balanced_fast"
    target = HERE / "source_leaf_balanced_interval"
    target.mkdir(exist_ok=False)
    for path in base.glob("*.py"):
        (target / path.name).write_bytes(path.read_bytes())
    wiring = {
        "entry.py": (("_csa_core_group", "decode_csa_spmd"), ("_csa_core_group", "cube_schedule", "vector_schedule")),
        "cube_schedule.py": (("cube_schedule",), ("indexer_score_cube_stage",)),
        "vector_schedule.py": (("vector_schedule",), ("indexer_group_plan_stage", "indexer_score_vector_lane_stage", "indexer_pair_publish_stage")),
        "indexer_group_stages.py": (("indexer_long_leaf_plan", "indexer_group_plan_stage"), ("indexer_long_leaf_plan",)),
        "indexer_score_stages.py": (("indexer_score_cube_stage", "indexer_score_vector_lane_stage"), ()),
        "indexer_publish_stages.py": (("indexer_pair_publish_stage",), ("wait_indexer_query_roots",)),
        "attention_dependencies.py": (("wait_indexer_query_roots",), ()),
    }
    audit = {}
    for filename, (definitions, calls) in wiring.items():
        path = target / filename
        source, audit[filename] = append_flag(path.read_text(), definitions, calls)
        path.write_text(source)
    path = target / "indexer_group_stages.py"
    source = path.read_text()
    source = once(source, "    if query_groups >= SCORE_WORKERS and leaf_count <= SCORE_WORKERS:\n",
                  "    if ROUND_LEAF_BALANCE and query_groups >= SCORE_WORKERS and leaf_count <= SCORE_WORKERS:\n")
    # 两个只负责owner旋转的helper改名，数学分组函数的BALANCED含义保持。
    tree = ast.parse(source)
    spans = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ("indexer_group_leaf_rotation", "indexer_work_item")]
    source_lines = source.splitlines(keepends=True)
    for node in sorted(spans, key=lambda n:n.lineno, reverse=True):
        segment = "".join(source_lines[node.lineno - 1:node.end_lineno])
        segment = segment.replace("BALANCED", FLAG)
        source_lines[node.lineno - 1:node.end_lineno] = segment.splitlines(keepends=True)
    source = "".join(source_lines)
    ast.parse(source)
    path.write_text(source)

    path = target / "indexer_score_stages.py"
    source = path.read_text()
    old = "indexer_work_item(buf_item, groups, buf_max_leaves, balance_leaves)"
    assert source.count(old) == 2
    source = source.replace(old, "indexer_work_item(buf_item, groups, buf_max_leaves, ROUND_LEAF_BALANCE)")
    path.write_text(source)

    path = target / "attention_dependencies.py"
    source = path.read_text()
    source = once(source, "rotation = indexer_group_leaf_rotation(group, groups, leaves, BALANCED)",
                  "rotation = indexer_group_leaf_rotation(group, groups, leaves, ROUND_LEAF_BALANCE)")
    source = once(source, """            if BALANCED:
                if rotation != 0:
""", """            if ROUND_LEAF_BALANCE:
                if rotation != 0:
""")
    path.write_text(source)

    path = target / "policy.py"
    source = path.read_text()
    source = once(source, "    plan_physical_rows: bool = False\n",
                  "    plan_physical_rows: bool = False\n    round_leaf_balance: bool = False\n")
    source = once(source, '            "PLAN_PHYSICAL_ROWS": self.plan_physical_rows,\n',
                  '            "PLAN_PHYSICAL_ROWS": self.plan_physical_rows,\n            "ROUND_LEAF_BALANCE": self.round_leaf_balance,\n')
    source = once(source, "        plan_physical_rows=tokens >= EARLY_GEOMETRY_MIN_TOKENS,\n", """        plan_physical_rows=tokens >= EARLY_GEOMETRY_MIN_TOKENS,
        # max_query_tokens为Native已有真实query上界。若总group容量不足24，
        # 编译掉新叶平衡/旋转路径；不为它在device每个消费者重复做无效判断。
        # 开启后仍检查本轮实际groups，padding可落回旧计划，不把上界当实际值。
        round_leaf_balance=(long_history and group == 6 and
                            min(tokens, requests * ((max_query_tokens + group - 1) // group)) >= 24),
""")
    doc_end = source.index('"""', source.index('"""') + 3)
    source = source[:doc_end] + """ROUND_LEAF_BALANCE按已有Host描述符推导的query-group容量区间选择；容量
不足24时编译删除整条新增映射，避免小区间支付无效Scalar代价。不是B16/B28
白名单；档内实际group/历史仍读取device，正反映射必须同时使用本开关。
当前仅候选，收益须以完整同轮对照决定，不因分档已实现而声称通过。
""" + source[doc_end:]
    path.write_text(source)
    for path in target.glob("*.py"):
        ast.parse(path.read_text())
    spec = importlib.util.spec_from_file_location("leaf_interval_policy", path.parent / "policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    controls = {}
    for batch in (16, 28):
        values = module.select_policy(tokens=batch * 6, requests=batch, max_query_tokens=6,
                                      max_seq_len=131078).constexprs()
        assert values[FLAG] == (batch == 28)
        controls[str(batch)] = values
        (HERE / f"leaf_interval_controls_b{batch}.json").write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n")
    report = dict(wiring=audit, controls=controls, formal_source_changed=False,
                  added_tensor_roots=0, host_descriptor_only=True, device_validated=False)
    (HERE / "LEAF_BALANCE_INTERVAL_PREPARE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print({"files_wired":len(audit), "B16_round":controls['16'][FLAG], "B28_round":controls['28'][FLAG]})


if __name__ == "__main__":
    main()
