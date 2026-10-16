# SPDX-License-Identifier: Apache-2.0
"""抽取已定位的 CSA 阶段作分配诊断；绝非完整 CSA 或可运行的设备测试。"""

import argparse
import ast
import copy
import json
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', choices=('aic', 'aiv'), default='aic')
    parser.add_argument('--stages', nargs='+', type=int)
    parser.add_argument('--output', type=Path, default=here / 'static/memory_probe')
    args = parser.parse_args()
    supported = {'aic': {13: 'indexer_score_topk_native_pair_aic', 15: 'proj_a_mm'},
                 'aiv': {32: 'qk_pv_aiv'}}
    if args.stages is None:
        args.stages = list(supported[args.core])
    if any(i not in supported[args.core] for i in args.stages):
        raise ValueError('此诊断只覆盖已定位的 Indexer、O-proj 与 QK/PV 阶段')
    source = here / "static/h131072_b4/entry.py"
    module = ast.parse(source.read_text())
    cls = next(v for v in module.body if isinstance(v, ast.ClassDef))
    functions = {v.name: v for v in cls.body if isinstance(v, ast.FunctionDef)}
    aic = functions[f"csa_static_{args.core}"]
    original = copy.deepcopy(aic.body)
    aic.body = original[:3 if args.core == 'aic' else 4]
    for index in args.stages:
        body_index = next(i for i, n in enumerate(original) if isinstance(n, ast.If)
                          and ast.unparse(n.test) == f"mapped_block_{index} >= 0")
        aic.body.extend(copy.deepcopy(original[body_index - 3:body_index]))
        branch = copy.deepcopy(original[body_index])
        dependency = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.For)
                          and ast.unparse(n.target) == f"dependency_{index}")
        assert ast.unparse(branch.body[dependency + 1]) == "pl.system.fence()"
        assert ast.unparse(branch.body[dependency + 2]) == "pl.system.cacheinvalid()"
        assert isinstance(branch.body[-1], ast.If) and ast.unparse(branch.body[-1].test) == f"publish_{index} != 0"
        # 保留全部原数学、FFTS 和本地完成边界；只去掉跨任务进度协议。
        branch.body = [branch.body[0]] + branch.body[dependency + 3:-1]
        aic.body.append(branch)
    aic.body.append(original[-1])
    # 去掉与所选阶段无关的根描述符，使组件只需检查实际用到的参数。
    used = {n.id for statement in aic.body[:-1] for n in ast.walk(statement)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    aic.args.args = [p for p in aic.args.args if p.arg == 'self' or p.arg in used]
    outputs = [n for n in aic.body[-1].value.elts if n.id in used]
    aic.body[-1].value.elts = outputs
    types = {p.arg: p.annotation for p in aic.args.args if p.arg != 'self'}
    aic.returns.slice.elts = [copy.deepcopy(types[n.id].slice) for n in outputs]
    entry = functions["entry"]
    entry.args = copy.deepcopy(aic.args)
    entry.returns = copy.deepcopy(aic.returns)
    spmd = next(n for n in entry.body if isinstance(n, ast.With))
    call = spmd.body[0].value
    call.func.attr = aic.name
    call.args = [ast.Name(id=p.arg, ctx=ast.Load()) for p in aic.args.args if p.arg != 'self']
    spmd.body[0].targets[0].elts = [ast.Name(id=n.id, ctx=ast.Store()) for n in outputs]
    entry.body[-1].value.elts = copy.deepcopy(outputs)
    cls.body = [aic, entry]
    cls.name = "CSAAllocatorFocusedProbe"
    ast.fix_missing_locations(module)
    directory = args.output
    directory.mkdir(exist_ok=True, parents=True)
    output = directory / "entry.py"
    output.write_text("# 只供 CPU 编译的阶段分配诊断；不允许单独上卡。\n" + ast.unparse(module) + "\n")
    record = {"source": str(source), "output": str(output), "selected_batches": args.stages,
              "core": args.core,
              "kernels": [supported[args.core][i] for i in args.stages],
              "tensor_parameters": len(aic.args.args) - 1,
              "arithmetic_changed": False, "pipeline_changed": False, "cross_task_progress_removed": True,
              "device_execution_allowed": False,
              "reason": "未包含依赖生产者与 MIX 的另一半，只为定位片上规划问题"}
    (directory / "scope.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
