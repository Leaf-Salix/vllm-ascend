# SPDX-License-Identifier: Apache-2.0
"""Keep the Indexer chain early; share the remaining main/KV work on Q_B cores."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

from prepare_scale_gather import replace_once


def prepare():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    parent = json.loads((here / "joint_prefix_v2.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    package = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    original_cube = subprocess.check_output(
        ["git", "show", f"{parent['parent_commit']}:{package}/cube_schedule.py"], cwd=repo, text=True
    )
    cube = source["cube_schedule.py"]
    start = cube.index("    if JOINT_BF16_PREFIX and COOPERATIVE_PREFIX:\n")
    end = cube.index("    signal = wait_row_owners(\n", start)
    old_start = original_cube.index("    idx_value, idx_score = indexer_compressor_project_stage(\n")
    old_end = original_cube.index("    signal = wait_row_owners(\n", old_start)
    cube = cube[:start] + original_cube[old_start:old_end] + cube[end:]
    marker = "    signal = publish_phase(signal, PHASE_IDX_QUERY_PROJECT, worker, CORES, generation)\n"
    cube = replace_once(
        cube,
        marker,
        marker
        + """    if JOINT_BF16_PREFIX and COOPERATIVE_PREFIX:
        # Leave the existing four Hadamard cores free to advance the Indexer chain.
        # Twenty independent main/KV column blocks run on the Q_B core group.
        main_value, main_score, kv_fp32 = joint_bf16_prefix_stage(
            normed, main_wkv, main_wgate, main_value, main_score, wkv, kv_fp32,
            worker, GROUPS, PREFIX_PROJECTION_ROWS,
        )
        signal = publish_phase(signal, PHASE_MAIN_PROJECT, worker, CORES, generation)
        signal = publish_phase(signal, PHASE_KV_PROJECT, worker, CORES, generation)
""",
    )
    source["cube_schedule.py"] = cube
    stage = source["joint_prefix_stages.py"]
    tree = ast.parse(stage)
    indexer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_project_indexer")
    lines = stage.splitlines(keepends=True)
    del lines[indexer.decorator_list[0].lineno - 1 : indexer.end_lineno]
    stage = "".join(lines)
    stage = replace_once(
        stage,
        "One row block has 24 N128 jobs: two Indexer value, two Indexer gate,\n"
        "eight main value, eight main gate, four original KV. No weights are packed\n",
        "One row block has 20 N128 jobs: eight main value, eight main gate,\n"
        "four original KV, on physical cores 4..23. No weights are packed\n",
    )
    stage = replace_once(stage, "JOBS_PER_ROW_BLOCK = 24", "JOBS_PER_ROW_BLOCK = 20\nINDEXER_BRANCH_CORES = 4")
    stage = replace_once(
        stage,
        "    idx_wkv: pl.Tensor[[256, 4096], pl.BF16],\n"
        "    idx_wgate: pl.Tensor[[256, 4096], pl.BF16],\n"
        "    idx_value: pl.InOut[pl.Tensor[[IPROJ, 256], pl.FP32]],\n"
        "    idx_score: pl.InOut[pl.Tensor[[IPROJ, 256], pl.FP32]],\n",
        "",
    )
    stage = replace_once(
        stage,
        "    for job in pl.range(worker, row_blocks * JOBS_PER_ROW_BLOCK, WORKERS):\n",
        "    projection_worker = (worker + WORKERS - INDEXER_BRANCH_CORES) % WORKERS\n"
        "    for job in pl.range(projection_worker, row_blocks * JOBS_PER_ROW_BLOCK, WORKERS):\n",
    )
    begin = stage.index("        if projection_column < 2:\n")
    stage = (
        stage[:begin]
        + """        if projection_column < 8:
            column = projection_column * N_TILE
            main_value = _project_main(normed, main_wkv, main_value, row, column, rows, ROW_TILE)
        elif projection_column < 16:
            column = (projection_column - 8) * N_TILE
            main_score = _project_main(normed, main_wgate, main_score, row, column, rows, ROW_TILE)
        else:
            column = (projection_column - 16) * N_TILE
            kv_fp32 = _project_forward(normed, wkv, kv_fp32, row, column, rows, ROW_TILE)
    return main_value, main_score, kv_fp32
"""
    )
    stage = stage.replace(
        "from .runtime_abi import IPROJ, KV_ROWS, MPROJ, T", "from .runtime_abi import KV_ROWS, MPROJ, T"
    )
    source["joint_prefix_stages.py"] = stage
    target = here / "runs/critical_prefix_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        ast.parse(content)
        (target / name).write_text(content)
    for case in ("C03", "C04", "C05"):
        descriptor = json.loads((here / f"descriptors/{case}_joint_prefix_v2.json").read_text())
        (here / f"descriptors/{case}_critical_prefix_v1.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    report = dict(
        source=str(target),
        parent="joint_prefix_v2.json",
        original_commit=parent["parent_commit"],
        adopted=False,
        mechanism="Original early Indexer projection; 20 main/KV jobs on cores 4..23 after Indexer query publication",
        critical_branch="Cores 0..3 can advance the original Hadamard branch without main/KV jobs",
        unchanged="Native roots, vector schedule, policy domain, math helpers, downstream math and signal protocol",
        codegen_gate="Full resources/root ABI; disabled case CPP; only then strict full L1 on C04/C05",
        files={name: hashlib.sha256(content.encode()).hexdigest() for name, content in source.items()},
    )
    with (here / "critical_prefix_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    prepare()
