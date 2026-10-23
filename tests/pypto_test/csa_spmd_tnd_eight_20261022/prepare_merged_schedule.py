# SPDX-License-Identifier: Apache-2.0
"""Combine disjoint validated domains while preserving the retained leaf optimization."""

import ast
import hashlib
import json
import re
import runpy
import subprocess
import tempfile
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    base, ours, theirs = (here / "runs" / n for n in ("baseline", "critical_prefix_v2", "score_group_order_v5"))
    target = here / "runs/merged_schedule_v1"
    source, resolutions = {}, []
    expected_conflicts = {"cube_schedule.py": 2, "entry.py": 4, "policy.py": 3}
    with tempfile.TemporaryDirectory(prefix="csa_schedule_merge_") as temporary:
        temp = Path(temporary)
        names = sorted({p.name for p in ours.glob("*.py")} | {p.name for p in theirs.glob("*.py")})
        for name in names:
            texts = [(root / name).read_text() if (root / name).exists() else "" for root in (ours, base, theirs)]
            if not texts[2]:
                source[name] = texts[0]
                continue
            if not texts[0]:
                source[name] = texts[2]
                continue
            for side, code in zip(("ours", "base", "theirs"), texts):
                (temp / side).write_text(code)
            merge = subprocess.run(
                [
                    "git",
                    "merge-file",
                    "-p",
                    "-L",
                    "retained_and_critical",
                    "-L",
                    "frozen_original",
                    "-L",
                    "group_order",
                    str(temp / "ours"),
                    str(temp / "base"),
                    str(temp / "theirs"),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            assert merge.returncode == expected_conflicts.get(name, 0), (name, merge.returncode, merge.stderr)

            def resolve(match, filename=name):
                left, right = match.group(1), match.group(2)
                if "main_wkv" in left:
                    assert filename == "cube_schedule.py" and "JOINT_BF16_PREFIX" in left
                    assert "JOINT_BF16_PREFIX" not in right
                    result, decision = left, "Keep full critical-prefix argument list; other side only reformats it"
                else:
                    assert "JOINT_BF16_PREFIX" in left or "joint_bf16_prefix" in left
                    assert "SCORE_GROUP_ORDER" in right or "score_group_order" in right
                    result, decision = left + right, "Preserve both independent constexpr fields and call arguments"
                resolutions.append(dict(file=filename, decision=decision, ours=left, theirs=right, resolved=result))
                return result

            source[name] = re.sub(
                r"^<<<<<<< retained_and_critical\n(.*?)^=======\n(.*?)^>>>>>>> group_order\n",
                resolve,
                merge.stdout,
                flags=re.M | re.S,
            )
    source["policy.py"] = replace_once(
        source["policy.py"],
        "joint_bf16_prefix=128 < tokens <= 192",
        "joint_bf16_prefix=long_history and group == 6 and 128 < tokens <= 192",
    )
    for code in source.values():
        assert "<<<<<<<" not in code
        ast.parse(code)
    target.mkdir(exist_ok=False)
    for name, code in source.items():
        (target / name).write_text(code)
    select = runpy.run_path(str(target / "policy.py"))["select_policy"]
    baseline = json.loads((here / "baseline.json").read_text())
    domains = {}
    for case_id, row in baseline["cases"].items():
        case = row["case"]
        descriptor = json.loads((here / f"descriptors/{case_id}_critical_prefix_v2.json").read_text())
        descriptor["policy"] = select(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=max(case["query_lengths"]),
            max_seq_len=max(case["seq_lens"]),
        ).constexprs()
        domains[case_id] = {
            k: descriptor["policy"][k] for k in ("LEAF_SCALE_LAYOUT", "JOINT_BF16_PREFIX", "SCORE_GROUP_ORDER")
        }
        with (here / f"descriptors/{case_id}_merged_schedule_v1.json").open("x") as out:
            out.write(json.dumps(descriptor, indent=2) + "\n")
    report = dict(
        source=str(target),
        parents=["critical_prefix_v2.json", "score_group_order_v5.json"],
        original_algorithm="baseline.json",
        adopted=False,
        device_execution=False,
        resolutions=resolutions,
        domains=domains,
        files={name: hashlib.sha256(code.encode()).hexdigest() for name, code in source.items()},
        next_gate="Exact generated code per active/off class; retain source only after all validation gates",
    )
    with (here / "merged_schedule_v1.json").open("x") as out:
        out.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(domains, indent=2))


if __name__ == "__main__":
    main()
