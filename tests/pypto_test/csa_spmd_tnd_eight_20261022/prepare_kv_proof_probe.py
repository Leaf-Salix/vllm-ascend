# SPDX-License-Identifier: Apache-2.0
"""Freeze one eager readback of actual proof fields without timing or instrumentation."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/kv_access_harness_v1"
    target = here / "runs/kv_proof_harness_v1"
    export = (here / "kv_access_probe.py").read_text()
    export = replace_once(
        export,
        '        valid_blocks=entry.buffers["attention_mask"][:tokens, :5].detach().cpu().numpy().copy(),\n',
        '        valid_blocks=entry.buffers["attention_mask"][:tokens, :5].detach().cpu().numpy().copy(),\n'
        '        all_valid_proof=entry.buffers["attention_mask"][:tokens, 7:11].detach().cpu().numpy().copy(),\n',
    )
    export = replace_once(
        export,
        '    path = directory / "kv_access.npz"\n',
        """    compressed_rows = entry.buffers["main_cache"].shape[0]
    columns = entry.buffers["compressed_table"].shape[1]
    assert 0 < columns <= 2048 and 0 < compressed_rows // 32 <= 16777216
    indices = arrays["physical_indices"].reshape(tokens, 4, 128)
    expected = ((indices >= 0) & (indices < compressed_rows)).all(axis=2).astype(np.int32)
    np.testing.assert_array_equal(arrays["all_valid_proof"], expected)
    eligible = (arrays["all_valid_proof"] == 1)
    eligible &= arrays["valid_blocks"][:, 1:5] > 0
    eligible &= arrays["valid_blocks"][:, 0:4] > 0
    proof_observation = dict(
        device_proof_matches_actual_physical_rows=True,
        physical_mode_supported=True,
        proof_values=np.unique(arrays["all_valid_proof"]).tolist(),
        all_valid_blocks=int(expected.sum()),
        eligible_fast_blocks=int(eligible.sum()),
        possible_compressed_blocks=tokens * 4,
        note="Actual device-produced inputs satisfy the fast-branch predicates; not an instruction-count trace",
    )
    path = directory / "kv_access.npz"
""",
    )
    export = replace_once(
        export,
        '            "merged_schedule_v1; bind actual emitted C++ to retained measurement before using this probe"\n',
        '            "Measured candidate; bind actual emitted C++ '
        'to its strict full-L1 report before using this probe"\n',
    )
    export = replace_once(
        export, "        case=case,\n", "        case=case,\n        proof_observation=proof_observation,\n"
    )
    ast.parse(export)
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "kv_access_probe.py").write_text(export)
    runner = (
        (target / "run_pto.py")
        .read_text()
        .replace(
            "Retained TND-SPMD selected-KV metadata observation only", "Measured TND-SPMD candidate proof readback only"
        )
        .replace("already-validated retained implementation", "separately-validated candidate implementation")
    )
    (target / "run_pto.py").write_text(runner)
    launcher = replace_once(
        (here / "launch_kv_access_probe.sh").read_text(),
        'harness="$here/runs/kv_access_harness_v1"',
        'harness="$here/runs/kv_proof_harness_v1"',
    )
    with (here / "launch_kv_proof_probe.sh").open("x") as stream:
        stream.write(launcher)
    report = dict(
        source=str(target),
        parent="kv_access_harness_v1",
        operator_source_unchanged=True,
        purpose="One eager readback; verify proof columns against physical indices "
        "and record expected branch eligibility",
        performance_eligible=False,
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    with (here / "kv_proof_harness_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
