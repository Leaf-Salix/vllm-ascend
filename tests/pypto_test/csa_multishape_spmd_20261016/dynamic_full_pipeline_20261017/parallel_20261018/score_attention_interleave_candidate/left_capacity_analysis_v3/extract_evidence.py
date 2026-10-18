"""只读提取 v3 已有 IR 的实际 allocation 与 metadata-view 别名证据。"""

import ast
import json
from pathlib import Path


EVIDENCE = Path(__file__).resolve().parent
CANDIDATE = EVIDENCE.parent
PASSES = CANDIDATE / "cpu_long_v3/kernel/passes_dump"
FILENAMES = [
    "35_after_InitMemRef.py",
    "36_after_MaterializeSemanticAliases.py",
    "37_after_MemoryReuse.py",
]


def memref(annotation):
    for node in ast.walk(annotation):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.MemRef":
            return {
                "base": ast.unparse(node.args[0]),
                "offset": ast.unparse(node.args[1]),
                "bytes": ast.literal_eval(node.args[2]),
            }
    return None


def main():
    data = {}
    excerpts = []
    for filename in FILENAMES:
        path = PASSES / filename
        source = path.read_text()
        module = ast.parse(source)
        function = next(
            node for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef) and node.name == "cube_schedule"
        )
        definitions = {
            node.target.id: node for node in ast.walk(function)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        selected = {}
        names = [
            "qk_l1_local_inline5359__ssa_v1", "qk_l1_t_inline5348__ssa_v0",
            "score_query_inline2896__ssa_v0", "score_coeff_inline2934__ssa_v0",
            "query_left_inline2917__ssa_v0", "buf_coefficient_pair_inline5311__ssa_v0",
            "query_iter_mv", "coeff_iter_mv",
            "query_after_inline2884__rv_v0_mv", "coeff_after_inline2925__rv_v0_mv",
        ]
        for name in names:
            node = definitions.get(name)
            if node is None:
                continue
            selected[name] = {"line": node.lineno, "memref": memref(node.annotation)}
            if filename == FILENAMES[-1] or name.startswith("qk_l1_"):
                excerpts.append(
                    f"### {filename}:{node.lineno} — {name}\n\n```python\n"
                    + ast.get_source_segment(source, node) + "\n```\n"
                )
        allocations = []
        for node in ast.walk(function):
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.value, ast.Call):
                continue
            if ast.unparse(node.value.func) != "pl.tile.alloc":
                continue
            allocations.append({
                "name": ast.unparse(node.target), "line": node.lineno,
                "space": ast.unparse(node.value.args[0]).rsplit(".", 1)[-1],
                "bytes": ast.literal_eval(node.value.args[1]),
            })
        totals = {}
        for allocation in allocations:
            totals[allocation["space"]] = totals.get(allocation["space"], 0) + allocation["bytes"]
        data[filename] = {
            "path": str(path), "size_bytes": path.stat().st_size,
            "selected": selected, "allocation_totals": totals,
            "mat_left_allocations": [a for a in allocations if a["space"] in ("Mat", "Left")],
        }
    final = data[FILENAMES[-1]]
    assert final["allocation_totals"]["Mat"] == 753664
    assert final["allocation_totals"]["Left"] == 77824
    for filename, expected_same in zip(FILENAMES, [True, False, False]):
        selected = data[filename]["selected"]
        same = (
            selected["qk_l1_local_inline5359__ssa_v1"]["memref"]["base"]
            == selected["qk_l1_t_inline5348__ssa_v0"]["memref"]["base"]
        )
        assert same == expected_same
    result = {
        "method": "已有 IR 的标准库 AST/直接文本提取；未编译、未执行设备、无哈希",
        "pass_evidence": data,
        "conclusions": {
            "left_bytes": 77824, "left_capacity_bytes": 65536,
            "mat_bytes": 753664, "mat_capacity_bytes": 524288,
            "query_and_attention_left_already_share": "mem_left_502@0",
            "kv_transpose_view_first_split": "35 -> 36 MaterializeSemanticAliases",
            "kv_view_orphan_bytes": 262144,
            "query_init_branch_copy_bytes_each": 49152,
            "coeff_extra_pool_bytes": 12288,
            "not_a_valid_allocation": "view失配是正确性缺陷，不可把736KiB当数学必要峰值",
        },
    }
    (EVIDENCE / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (EVIDENCE / "IR_EXCERPTS.md").write_text("# v3 已有 IR 小片段\n\n" + "\n".join(excerpts))
    (EVIDENCE / "LOCAL_ARTIFACTS.json").write_text(json.dumps([
        {"path": item["path"], "size_bytes": item["size_bytes"], "retained_locally": True}
        for item in data.values()
    ], ensure_ascii=False, indent=2) + "\n")
    print("PASS: Left=77824, Mat=753664; transpose_view split first occurs in pass36")


if __name__ == "__main__":
    main()
