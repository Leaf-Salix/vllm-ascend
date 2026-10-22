# SPDX-License-Identifier: Apache-2.0
"""Restrict the critical-prefix candidate to the existing single-M192 resource domain."""

import ast
import hashlib
import json
import runpy
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/critical_prefix_v1"
    target = here / "runs/critical_prefix_v2"
    source = {p.name: p.read_text() for p in parent.glob("*.py")}
    source["policy.py"] = replace_once(
        source["policy.py"], "joint_bf16_prefix=64 < tokens <= 192", "joint_bf16_prefix=128 < tokens <= 192"
    )
    target.mkdir(exist_ok=False)
    for name, code in source.items():
        ast.parse(code)
        (target / name).write_text(code)
    select = runpy.run_path(str(target / "policy.py"))["select_policy"]
    baseline = json.loads((here / "baseline.json").read_text())
    policies = {}
    for case_id, row in baseline["cases"].items():
        case = row["case"]
        prepared = here / f"descriptors/{case_id}_leaf_scale_guard_v1.json"
        if prepared.exists():
            descriptor = json.loads(prepared.read_text())
        else:
            original = json.loads(Path(row["source_report"]).read_text())
            descriptor = dict(buffers=original["descriptors"]["spmd"])
        descriptor["policy"] = select(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=max(case["query_lengths"]),
            max_seq_len=max(case["seq_lens"]),
        ).constexprs()
        policies[case_id] = descriptor["policy"]
        (here / f"descriptors/{case_id}_critical_prefix_v2.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    report = dict(
        source=str(target),
        parent="critical_prefix_v1.json",
        adopted=False,
        changed_files=["policy.py"],
        reason="C04 776.74us regresses against retained 763.65us; C05 1003.02us is provisionally positive",
        domain="128 < Host token capacity <= 192, with cooperative prefix; no equal-Q/L assumption",
        enabled_eight_cases=[k for k, v in policies.items() if v["JOINT_BF16_PREFIX"]],
        files={name: hashlib.sha256(code.encode()).hexdigest() for name, code in source.items()},
    )
    with (here / "critical_prefix_v2.json").open("x") as out:
        out.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
