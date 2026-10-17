# SPDX-License-Identifier: Apache-2.0
"""审计编译产物中的轮询状态和提交结构；不把 CPU 审计当作设备执行 PASS。"""

import argparse
import ast
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_file = args.compiled / "kernel/passes_dump/25_after_ExpandMixedKernel.py"
    source = source_file.read_text()
    checks = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Call) or ast.unparse(node.iter.func) != "pl.while_":
            continue
        names = [v.id for v in ast.walk(node.target) if isinstance(v, ast.Name)]
        if not any(v.startswith("ready_") for v in names):
            continue
        initial = next((v.value for v in node.iter.keywords if v.arg == "init_values"), None)
        calls = [v for v in ast.walk(node) if isinstance(v, ast.Call)]
        conditions = [v for v in calls if ast.unparse(v.func) == "pl.cond"]
        yields = [v for v in calls if ast.unparse(v.func) == "pl.yield_"]
        if initial is None or not conditions or not yields:
            raise AssertionError("轮询的初值/条件/迭代值不完整")
        if not set(names) & {v.id for v in ast.walk(conditions[0]) if isinstance(v, ast.Name)}:
            raise AssertionError("轮询条件没有使用 loop carry")
        checks.append({"line": node.lineno, "carry": names, "initial": ast.unparse(initial),
                       "condition": ast.unparse(conditions[0]), "yield": ast.unparse(yields[-1])})
    layout = json.loads(args.layout.read_text())
    expected = sum(len(v) for v in layout["batches"].values())
    if "__FREE_VAR" in source or len(checks) != expected:
        raise AssertionError("轮询状态或数量与调度表不符")
    orch = (args.compiled / "kernel/orchestration/entry.cpp").read_text()
    submits = re.findall(r"\brt_submit_\w+\s*\(", orch)
    if len(submits) != 1 or "set_block_num(24)" not in orch or "MixedKernels" not in orch:
        raise AssertionError("提交结构不是预期的一次 24 组 MIX")
    result = {"scope": "CPU IR/orchestration 审计；codegen 状态从独立结果读取，无设备执行",
              "source_ir": str(source_file), "pass25_polling_loops": len(checks),
              "expected_loops": expected, "free_var_occurrences": 0, "polling_state_preserved": True,
              "submission_calls": submits, "block_num": 24,
              "sync_start": "set_require_sync_start(true)" in orch, "mixed_task": True,
              "full_codegen_passed": json.loads((args.compiled / "result.json").read_text())["full_codegen_passed"],
              "checks": checks}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
