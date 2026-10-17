# SPDX-License-Identifier: Apache-2.0
"""只在 CPU 核对分阶段内存规划；不改变生产编译器，不执行设备代码。"""

import argparse
import ast
import copy
import json
from pathlib import Path


def isolate(source, role, index):
    module = ast.parse(source)
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef))
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "csa_static_" + role)
    stages = [node for node in fn.body if isinstance(node, ast.If)
              and "mapped_block_" in ast.unparse(node.test)]
    stage = stages[index]
    # 所有顶层非 stage 语句只包含参数别名、Scalar 调度描述符及 alloc。
    # 保留它们保证 SSA 定义完整，再移除当前数学阶段从未引用的 tile.alloc。
    body = [copy.deepcopy(node) for node in fn.body if node is stage or node not in stages]
    live = {node.id for node in ast.walk(stage) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    retained = []
    for node in body:
        if (isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "pl.tile.alloc"
                and node.target.id not in live):
            continue
        retained.append(node)
    fn.body = retained
    cls.body = [fn]
    return ast.unparse(module) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--role", choices=("aic", "aiv"), required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codegen", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    text = isolate(args.source.read_text(), args.role, args.stage)
    source = args.output / "isolated.py"
    source.write_text(text)
    import pypto.language as pl
    from pypto import passes
    from pypto.backend import BackendType, set_backend_type

    set_backend_type(BackendType.Ascend910B)
    program = pl.parse(text, filename=str(source))
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        reused = passes.memory_reuse()(program)
        planned = passes.allocate_memory_addr()(reused)
    (args.output / "memory_reuse.py").write_text(reused.as_python("pl"))
    (args.output / "allocated.py").write_text(planned.as_python("pl"))
    if args.codegen:
        from pypto.pypto_core import codegen
        from pypto.backend.pto_backend import _compile_pto_module

        codes = codegen.PTOCodegen().generate(planned)
        text = codes if isinstance(codes, str) else "\n".join(codes.values())
        cpp = _compile_pto_module(text, "isolated_stage", str(args.output))
        (args.output / "generated.cpp").write_text(cpp)
    (args.output / "result.json").write_text(json.dumps({
        "passed": True, "scope": "只验证此阶段独立内存规划，不代表完整融合正确或性能",
        "role": args.role, "stage": args.stage, "device_execution": False}, ensure_ascii=False, indent=2) + "\n")
    print("PASS", args.role, args.stage, flush=True)


if __name__ == "__main__":
    main()
