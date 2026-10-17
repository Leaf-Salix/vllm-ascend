# SPDX-License-Identifier: Apache-2.0
"""解释正式输出helper的Scalar循环，核对分组区间覆盖、原核归属和数学体。

本检查只擦除数学tile块并记录其全局任务编号，保留源码的索引计算、循环
和分支。期望值独立从全局任务集合筛选owner和group，不复制首项公式。
同时核对组内数学AST；编译简化后的表达及设备缓存一致性仍须另验。
"""

import argparse
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def body_loop(fn):
    return next(n for n in fn.body if isinstance(n, ast.For))


def interpreter(name, file, is_ob=False):
    fn = copy.deepcopy(function(PKG / file, name))
    fn.decorator_list = []
    for arg in fn.args.args:
        arg.annotation = None
    fn.returns = None
    outer = body_loop(fn)
    record = ast.parse("record(work)" if is_ob else "record(logical)").body
    if is_ob:
        branch = next(n for n in outer.body if isinstance(n, ast.If))
        branch.body = record
    else:
        outer.body = record
    fn.body = [node for node in fn.body if not isinstance(node, ast.Return)]
    names = [a.arg for a in fn.args.args]
    tree = ast.fix_missing_locations(ast.Module([fn], type_ignores=[]))
    namespace = dict(
        pl=SimpleNamespace(range=range, max=max, tensor=SimpleNamespace(dim=lambda _t, _axis: 8 * 2048)),
        O_GROUPS=8,
        O_GROUP_IN=4096,
        O_LORA=1024,
        PROJ_A_ROW_TILE=128,
        QUANT_TASK_T_TILE=32,
        PROJ_B_D_TILE=512,
        D=4096,
    )
    exec(compile(tree, str(PKG / file), "exec"), namespace)

    def run(*, tokens, worker, workers, cols, rows, begin, end):
        seen = []
        namespace["record"] = seen.append
        values = dict(
            tokens=tokens,
            worker_id=worker,
            WORKERS=workers,
            A_COL_TILE=cols,
            ROW_TILE=rows,
            group_begin=begin,
            group_end=end,
        )
        namespace[name](*(values.get(key) for key in names))
        return seen

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = HERE / "output_group_pipeline_v1/before"
    configurations = [
        ("oa", "output_projection_a_stages.py", "o_projection_a_stage_nz", "o_projection_a_groups_stage_nz", 24),
        (
            "quant",
            "output_projection_a_stages.py",
            "o_projection_group_quant_stage",
            "o_projection_groups_quant_stage",
            48,
        ),
        ("ob", "output_projection_b_stages.py", "_o_b_nz_stage", "_o_b_nz_groups_stage", 24),
    ]
    math_checks = []
    cases = 0
    for kind, file, old, new, workers in configurations:
        original = body_loop(function(before / file, old)).body
        candidate = body_loop(function(PKG / file, new)).body
        if kind == "ob":
            original = original[1:]  # 首条work赋值仅属于任务选取，不属于数学。
            candidate = next(n for n in candidate if isinstance(n, ast.If)).body
        equal = ast.dump(ast.Module(original, [])) == ast.dump(ast.Module(candidate, []))
        assert equal, (kind, "math body changed")
        math_checks.append(dict(stage=kind, loop_math_body_ast_identical=True))
        run = interpreter(new, file, is_ob=kind == "ob")
        # T=0及各tile两侧，含比当前七点更大的动态循环范围；不是新增支持承诺。
        for tokens in (
            0,
            1,
            7,
            8,
            9,
            23,
            24,
            25,
            31,
            32,
            33,
            47,
            48,
            95,
            96,
            97,
            127,
            128,
            129,
            143,
            144,
            167,
            168,
            169,
            191,
            192,
            193,
            383,
            384,
            385,
            511,
            512,
            513,
            1025,
        ):
            rows = 32 if tokens <= 32 else 96 if tokens <= 96 else 128
            for cols in (128, 256):
                per_group = (
                    ((tokens + 127) // 128) * (1024 // cols)
                    if kind == "oa"
                    else (tokens + 31) // 32
                    if kind == "quant"
                    else 8
                )
                for worker in range(workers):
                    for begin, end in [(0, 8), *[(g, g + 1) for g in range(8)], (0, 0), (8, 8), (2, 6)]:
                        actual = run(
                            tokens=tokens, worker=worker, workers=workers, cols=cols, rows=rows, begin=begin, end=end
                        )
                        expected = [
                            n
                            for n in range(8 * per_group)
                            if n % workers == worker and begin * per_group <= n < end * per_group
                        ]
                        assert actual == expected, (kind, tokens, cols, worker, begin, end, actual, expected)
                        cases += 1
    report = dict(
        passed=True,
        cases=cases,
        source="formal helper AST",
        math_checks=math_checks,
        scope="任务覆盖与组内数学一致；非设备精度或编译后有符号算术证明",
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
