# SPDX-License-Identifier: Apache-2.0
"""Isolate finer quant ownership from removal of unread large-NZ padding writes."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    parent = subprocess.check_output(["git", "rev-parse", "79fae6db0"], cwd=repo, text=True).strip()
    package = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    paths = subprocess.check_output(["git", "ls-tree", "--name-only", parent, package + "/"], cwd=repo, text=True)
    original = {Path(p).name: subprocess.check_output(["git", "show", f"{parent}:{p}"], cwd=repo, text=True)
                for p in paths.splitlines() if p.endswith(".py")}
    for variant in ["output_quant_task16_v1", "output_quant_no_unused_pad_v1"]:
        source = dict(original)
        name = "output_projection_a_stages.py"
        if variant == "output_quant_task16_v1":
            source[name] = replace_once(source[name], "QUANT_TASK_T_TILE = 32", "QUANT_TASK_T_TILE = 16")
            for schedule in ["cube_schedule.py", "vector_schedule.py"]:
                source[schedule] = replace_once(
                    source[schedule], "import pypto.language as pl\n",
                    "import pypto.language as pl\n\nfrom .output_projection_a_stages import QUANT_TASK_T_TILE\n",
                )
                sites = source[schedule].count("(tokens + 31) // 32")
                assert sites == (2 if schedule == "cube_schedule.py" else 1)
                source[schedule] = source[schedule].replace(
                    "(tokens + 31) // 32", "(tokens + QUANT_TASK_T_TILE - 1) // QUANT_TASK_T_TILE"
                )
            mechanism = "Split 32-row quant work units into 16 rows; preserve 8-row mathematical tiles and pipeline"
            guard = "Experimental all tiers; first measure C04 only, then determine retained resource domain"
            invariants = "All real rows, last-block zero writer, and matching per-group AIC dependencies"
        else:
            source[name] = replace_once(
                source[name], "import pypto.language as pl\n",
                "import pypto.language as pl\n\n"
                "from .output_projection_b_stages import PROJ_B_MEDIUM_T_TILE, QUANT_WEIGHT_NZ\n",
            )
            source[name] = replace_once(
                source[name], "    padded_rows = (tokens + ROW_TILE - 1) // ROW_TILE * ROW_TILE\n",
                "    # NZ large-M O_B already loads and stores only its actual valid rows.\n"
                "    # Keep padding for the full-M small/medium NZ and ND consumers.\n"
                "    if QUANT_WEIGHT_NZ and ROW_TILE > PROJ_B_MEDIUM_T_TILE:\n"
                "        padded_rows = tokens\n"
                "    else:\n"
                "        padded_rows = (tokens + ROW_TILE - 1) // ROW_TILE * ROW_TILE\n",
            )
            mechanism = "Do not clear quant activation rows beyond T when the selected NZ O_B never reads them"
            guard = "QUANT_WEIGHT_NZ and ROW_TILE > PROJ_B_MEDIUM_T_TILE (96)"
            invariants = (
                "Every row inside actual T, including inactive requests; all scale writes, math and sync unchanged"
            )
        target = here / "runs" / variant
        target.mkdir(exist_ok=False)
        for name, text in source.items():
            ast.parse(text)
            (target / name).write_text(text)
        report = dict(
            source=str(target), parent_commit=parent, baseline="baseline.json", adopted=False,
            files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
            changes=[n for n in source if source[n] != original[n]],
            mechanism=mechanism, guard=guard, invariants=invariants,
            signal_layout="Original 512B slots and all publish/wait/fence retained",
            source_reference="Primary MPMD scheduling_v17_v18_results: quant parallelism improved local stage only",
        )
        with (here / f"{variant}.json").open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
        print(target)


if __name__ == "__main__":
    main()
