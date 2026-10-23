# SPDX-License-Identifier: Apache-2.0
"""Joint O_A geometry, quant dependencies and complementary O_B ownership trial."""

import ast
import hashlib
import json
import runpy
import subprocess
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    parent = subprocess.check_output(["git", "rev-parse", "79fae6db0"], cwd=repo, text=True).strip()
    package = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    paths = subprocess.check_output(["git", "ls-tree", "--name-only", parent, package + "/"], cwd=repo, text=True)
    original = {
        Path(path).name: subprocess.check_output(["git", "show", f"{parent}:{path}"], cwd=repo, text=True)
        for path in paths.splitlines()
        if path.endswith(".py")
    }
    source = dict(original)
    name = "policy.py"
    source[name] = replace_once(source[name], "    ob_rows: int\n", "    ob_rows: int\n    oa_rows: int = 128\n")
    source[name] = replace_once(
        source[name],
        '            "OA_COLS": self.oa_cols,\n',
        '            "OA_COLS": self.oa_cols,\n            "OA_ROWS": self.oa_rows,\n',
    )
    source[name] = replace_once(
        source[name],
        "    return CSASPMDPolicy(\n",
        "    # One M192 tile; actual rows still come from the device tensor.\n"
        "    output_tall = 128 < tokens <= 192\n    return CSASPMDPolicy(\n",
    )
    source[name] = replace_once(
        source[name],
        "        oa_cols=128 if tokens <= 96 else 256,\n",
        "        oa_cols=128 if tokens <= 96 or output_tall else 256,\n        oa_rows=192 if output_tall else 128,\n",
    )
    source[name] = replace_once(
        source[name],
        "        output_work_balance=(\n            single_request_leaves\n",
        "        output_work_balance=(\n            output_tall\n            or single_request_leaves\n",
    )
    for name in ["entry.py", "cube_schedule.py", "vector_schedule.py"]:
        declaration = "    OA_COLS: pl.constexpr,\n"
        assert source[name].count(declaration) == (2 if name == "entry.py" else 1)
        source[name] = source[name].replace(declaration, declaration + "    OA_ROWS: pl.constexpr,\n")
    name = "entry.py"
    call = "            OA_COLS,\n"
    assert source[name].count(call) == 3
    source[name] = source[name].replace(call, call + "            OA_ROWS,\n")
    for name in ["cube_schedule.py", "vector_schedule.py"]:
        old = "((tokens + 127) // 128) * (1024 // OA_COLS)"
        assert source[name].count(old) == (1 if name == "cube_schedule.py" else 2)
        source[name] = source[name].replace(old, "((tokens + OA_ROWS - 1) // OA_ROWS) * (1024 // OA_COLS)")
    name = "cube_schedule.py"
    source[name] = replace_once(
        source[name], "GROUPS, OA_COLS, oa_group, oa_group + 1", "GROUPS, OA_COLS, OA_ROWS, oa_group, oa_group + 1"
    )
    source[name] = replace_once(source[name], "GROUPS, OA_COLS)\n", "GROUPS, OA_COLS, OA_ROWS)\n")
    name = "output_projection_a_stages.py"
    tree = ast.parse(source[name])
    lines = source[name].splitlines(keepends=True)
    names = {"o_projection_a_stage_nd", "o_projection_a_groups_stage_nz", "o_projection_a_stage_nz"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == 3
    for node in reversed(functions):
        text = "".join(lines[node.lineno - 1 : node.end_lineno])
        text = text.replace("PROJ_A_ROW_TILE", "A_ROW_TILE")
        text = replace_once(
            text, "    A_COL_TILE: pl.constexpr,\n", "    A_COL_TILE: pl.constexpr,\n    A_ROW_TILE: pl.constexpr,\n"
        )
        if node.name == "o_projection_a_stage_nz":
            text = replace_once(
                text, "WORKERS, A_COL_TILE, 0, O_GROUPS", "WORKERS, A_COL_TILE, A_ROW_TILE, 0, O_GROUPS"
            )
        lines[node.lineno - 1 : node.end_lineno] = [text]
    source[name] = "".join(lines)
    target = here / "runs/output_tall_v1"
    target.mkdir(exist_ok=False)
    for name, text in source.items():
        ast.parse(text)
        (target / name).write_text(text)
    select = runpy.run_path(str(target / "policy.py"))["select_policy"]
    baseline = json.loads((here / "baseline.json").read_text())
    for case_id in ["C04", "C05", "C06", "C07"]:
        case = baseline["cases"][case_id]["case"]
        descriptor_path = here / f"descriptors/{case_id}_leaf_scale_guard_v1.json"
        if descriptor_path.exists():
            descriptors = json.loads(descriptor_path.read_text())
        else:
            assert case_id == "C06"
            previous = json.loads(Path(baseline["cases"][case_id]["source_report"]).read_text())
            descriptors = dict(buffers=previous["descriptors"]["spmd"])
        descriptors["policy"] = select(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=max(case["query_lengths"]),
            max_seq_len=max(case["seq_lens"]),
        ).constexprs()
        (here / f"descriptors/{case_id}_output_tall_v1.json").write_text(json.dumps(descriptors, indent=2) + "\n")
    report = dict(
        source=str(target),
        parent_commit=parent,
        adopted=False,
        changes=[name for name in source if source[name] != original[name]],
        files={name: hashlib.sha256(text.encode()).hexdigest() for name, text in source.items()},
        guard="Existing Host T capacity satisfies 128 < T <= 192; no assumption about actual Q/L or page order",
        mechanism=[
            "O_A single M192/N128 grid instead of M128+tail/N256; preserve K256 accumulation and stage=2",
            "Both AIC publication and AIV per-group quant dependencies use the same OA_ROWS geometry",
            "Reuse existing 32-rank complementary O_B ownership on the same 24 physical AICs",
        ],
        unchanged=(
            "All earlier stages, original quant/O_B math, full output checks, 512B signal slots and phase protocol"
        ),
        macro_evidence="macro_schedule_analysis.json; theoretical balance and logical weight bytes only",
        differentiation="Not old smaller-M extra-weight-load trial or two-accumulator cross-M row-pair restoration",
        next_gate="Full compile and actual O_A memory/weight-load audit before any NPU measurement",
    )
    with (here / "output_tall_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
