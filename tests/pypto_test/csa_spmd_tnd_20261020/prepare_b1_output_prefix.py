# SPDX-License-Identifier: Apache-2.0
"""Combine the validated B1 output mapping with prior empty-consumer wait guards."""

import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path


def guarded_wait(block, predicate, setup=""):
    """Only the existing balanced-G2 B1 domain gets the narrower waits."""
    body = "".join("        " + line if line.strip() else line for line in block.splitlines(True))
    original = "".join("    " + line if line.strip() else line for line in block.splitlines(True))
    return (
        "    if BALANCED and QUERY_GROUP == 2:\n"
        + setup
        + f"        if {predicate}:\n"
        + body
        + "    else:\n"
        + original
    )


def main():
    here = Path(__file__).resolve().parent
    origin = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    target = here / "runs/b1_output_prefix_candidate_v80"
    shutil.copytree(origin, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "vector_schedule.py"
    before = path.read_text()
    text = before
    start = text.index("    # Q_A 每个 M 组")
    end = text.index("    qr, qr_scale, qr_matmul, qr_scale_pad = q_r_stage(", start)
    text = text[:start] + guarded_wait(text[start:end], "worker < (tokens + 7) // 8") + text[end:]
    for phase, predicate in {
        "PHASE_IDX_QUERY_PROJECT": "worker < ((tokens + 7) // 8) * 16",
        "PHASE_WEIGHTS_PROJECT": "worker < (tokens + 15) // 16",
        "PHASE_IDX_QUERY_HAD": "worker < tokens",
        "PHASE_IDX_HADAMARD": "worker < idx_requests",
    }.items():
        matches = [line for line in text.splitlines(True)
                   if line.startswith("    signal = wait_row_owners(") and f", {phase}," in line]
        assert len(matches) == 1, phase
        text = text.replace(matches[0], guarded_wait(matches[0], predicate), 1)
    coefficients = (
        "    signal = wait_row_owners(signal, PHASE_WEIGHTS, CORES, generation, "
        "0, tokens, 16, VECTORS, GROUPS, 1)\n"
        "    signal = wait_row_owners(signal, PHASE_IDX_QUERY_QUANT, CORES, generation, "
        "0, tokens, 1, VECTORS, GROUPS, 1)\n"
    )
    assert text.count(coefficients) == 1
    text = text.replace(coefficients, guarded_wait(
        coefficients, "worker < coefficient_groups",
        "        coefficient_groups = pl.cast(pl.read(group_offsets, [idx_requests]), pl.INDEX)\n",
    ))
    ast.parse(text)
    path.write_text(text)
    changed = [x.name for x in origin.glob("*.py") if x.read_bytes() != (target / x.name).read_bytes()]
    assert changed == ["vector_schedule.py"]
    (here / "b1_output_prefix_v80.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), text.splitlines(True),
        fromfile="a/vector_schedule.py", tofile="b/vector_schedule.py",
    )))
    descriptor_source = here / "runs/output_work_balance_b1_s6_and_s1_s6_v67/buffer_descriptors.json"
    description = json.loads(descriptor_source.read_text())
    assert description["policy"]["BALANCED"] and description["policy"]["QUERY_GROUP"] == 2
    assert description["policy"]["OUTPUT_WORK_BALANCE"]
    descriptor_path = here / "compile_descriptors/b1_output_prefix_v80.json"
    descriptor_path.write_text(json.dumps(description, indent=2) + "\n")
    report = {
        "source": str(target), "descriptors": str(descriptor_path), "changed_files": changed,
        "vector_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "device_execution": False, "adopted": False,
        "scope": "Existing B1 balanced-G2 domain only; preserve all math and notifications",
        "prior_evidence": "v38/v47 active-prefix trials plus v67 output ownership; combination not yet measured",
    }
    (here / "b1_output_prefix_v80.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
