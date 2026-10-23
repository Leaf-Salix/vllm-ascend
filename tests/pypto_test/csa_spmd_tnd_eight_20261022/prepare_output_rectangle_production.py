# SPDX-License-Identifier: Apache-2.0
"""Integrate the validated NZ rectangle; leave the existing ND interface intact."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def span(function):
    return min([function.lineno] + [d.lineno for d in function.decorator_list]) - 1, function.end_lineno


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    parent = json.loads((here / "physical_capacity_production_v1.json").read_text())
    candidate = json.loads((here / "output_rectangle_v3.json").read_text())
    assert json.loads((here / "output_rectangle_v3_codegen_validation.json").read_text())["passed"]
    assert json.loads((here / "output_rectangle_v3_runtime_validation.json").read_text())["passed"]
    original = {p.name: p.read_text() for p in root.glob("*.py")}
    assert {name: hashlib.sha256(text.encode()).hexdigest() for name, text in original.items()} == parent["files"]
    source = {p.name: p.read_text() for p in Path(candidate["source"]).glob("*.py")}
    name = "output_projection_a_stages.py"
    tree = ast.parse(original[name])
    nd = next(fn for fn in tree.body if isinstance(fn, ast.FunctionDef) and fn.name == "o_projection_a_stage_nd")
    first, last = span(nd)
    nd_source = "".join(original[name].splitlines(keepends=True)[first:last])
    # The current CSA entry imports only the NZ helper. Do not expand the unused
    # ND API based on tests that validated the native NZ2 path.
    lines = source[name].splitlines(keepends=True)
    nd_functions = [
        fn for fn in ast.parse(source[name]).body if isinstance(fn, ast.FunctionDef) and fn.name.endswith("_stage_nd")
    ]
    assert len(nd_functions) == 3
    first, last = span(nd_functions[0])[0], span(nd_functions[-1])[1]
    lines[first:last] = [nd_source + "\n"]
    source[name] = "".join(lines)
    source[name] = replace_once(
        source[name],
        "    所有容量轴均 pl.dynamic；固定8组、模型维度和128/256列tile不是B/S白名单。\n",
        "    所有容量轴均 pl.dynamic；固定8组、模型维度及核内tile不代表固定B/S。\n"
        "    NZ路径允许M128/N128或N256，以及容量128<T<=160的M160/N192区间；\n"
        "    后者按五个N192和一个静态N64覆盖1024列。ND接口仍保持原M128路径。\n",
    )
    source[name] = replace_once(
        source[name],
        "    O_A 以(group,ceil(T/128)行块,N列块)展平，logical%WORKERS 为AIC owner；\n",
        "    NZ O_A以(group,ceil(T/A_ROW_TILE)行块,ceil(1024/A_COL_TILE)列块)展平，\n"
        "    logical%WORKERS为AIC owner；AIC发布和量化获取使用同一组内任务数。\n",
    )
    source[name] = replace_once(
        source[name],
        "        if nf < 5:\n",
        "        # NZ末维必须静态；第六块只算/写64列，不越过本组的1024列。\n        if nf < 5:\n",
    )
    policy = source["policy.py"]
    policy = replace_once(
        policy,
        "DEFAULT_PHYSICAL_PLAN_COLUMNS = 2048\n",
        "DEFAULT_OA_ROWS = 128\nRECTANGLE_OA_ROWS = 160\nRECTANGLE_OA_COLS = 192\n"
        "DEFAULT_PHYSICAL_PLAN_COLUMNS = 2048\n",
    )
    policy = replace_once(policy, "    oa_rows: int = 128\n", "    oa_rows: int = DEFAULT_OA_ROWS\n")
    policy = replace_once(
        policy,
        "    output_rectangle = 128 < tokens <= 160\n",
        "    # One M tile removes repeated weight loads; actual rows remain device-side.\n"
        "    output_rectangle = DEFAULT_OA_ROWS < tokens <= RECTANGLE_OA_ROWS\n",
    )
    policy = replace_once(
        policy,
        "        oa_cols=192 if output_rectangle else (128 if tokens <= 96 else 256),\n"
        "        oa_rows=160 if output_rectangle else 128,\n",
        "        oa_cols=RECTANGLE_OA_COLS if output_rectangle else (128 if tokens <= 96 else 256),\n"
        "        oa_rows=RECTANGLE_OA_ROWS if output_rectangle else DEFAULT_OA_ROWS,\n",
    )
    source["policy.py"] = policy
    changes = [name for name, body in source.items() if body != original[name]]
    assert set(changes) == {
        "policy.py",
        "entry.py",
        "cube_schedule.py",
        "vector_schedule.py",
        "output_projection_a_stages.py",
    }
    for name in changes:
        ast.parse(source[name])
    for name in changes:
        (root / name).write_text(source[name])
    print("Integrated five source modules; production freeze and code equivalence still required")


if __name__ == "__main__":
    main()
