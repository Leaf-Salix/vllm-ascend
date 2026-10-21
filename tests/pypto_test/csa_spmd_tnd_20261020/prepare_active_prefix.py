# SPDX-License-Identifier: Apache-2.0
"""Prepare a source-only trial that skips waits for empty producer loops."""

import ast
import difflib
import shutil
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    original = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    destination = here / "runs/active_prefix_wait_candidate_v1"
    shutil.copytree(original, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = destination / "vector_schedule.py"
    before = path.read_text()
    source = before
    begin = source.index("    # Q_A 每个 M 组")
    end = source.index("    qr, qr_scale, qr_matmul, qr_scale_pad = q_r_stage(", begin)
    qa_waits = source[begin:end]
    source = source[:begin] + (
        "    # Empty QR workers do not read QA; keep all notifications unchanged.\n"
        "    if worker < (tokens + 7) // 8:\n"
        + "".join("    " + line if line.strip() else line for line in qa_waits.splitlines(keepends=True))
    ) + source[end:]
    guards = {
        "PHASE_IDX_QUERY_PROJECT": "worker < ((tokens + 7) // 8) * 16",
        "PHASE_WEIGHTS_PROJECT": "worker < (tokens + 15) // 16",
        "PHASE_IDX_QUERY_HAD": "worker < tokens",
        "PHASE_IDX_HADAMARD": "worker < idx_requests",
    }
    for phase, condition in guards.items():
        matching = [
            line
            for line in source.splitlines(keepends=True)
            if line.startswith("    signal = wait_row_owners(") and f", {phase}," in line
        ]
        assert len(matching) == 1, (phase, matching)
        line = matching[0]
        source = source.replace(line, f"    if {condition}:\n    {line}", 1)
    weights_wait = (
        "    signal = wait_row_owners(signal, PHASE_WEIGHTS, CORES, generation, "
        "0, tokens, 16, VECTORS, GROUPS, 1)\n"
        "    signal = wait_row_owners(signal, PHASE_IDX_QUERY_QUANT, CORES, generation, "
        "0, tokens, 1, VECTORS, GROUPS, 1)\n"
    )
    assert source.count(weights_wait) == 1
    source = source.replace(
        weights_wait,
        "    coefficient_groups = pl.cast(pl.read(group_offsets, [idx_requests]), pl.INDEX)\n"
        "    if worker < coefficient_groups:\n"
        + "".join("    " + line for line in weights_wait.splitlines(keepends=True)),
    )
    ast.parse(source)
    path.write_text(source)
    (here / "active_prefix_wait_v1.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                source.splitlines(keepends=True),
                fromfile="a/vector_schedule.py",
                tofile="b/vector_schedule.py",
            )
        )
    )
    print(destination)


if __name__ == "__main__":
    main()
