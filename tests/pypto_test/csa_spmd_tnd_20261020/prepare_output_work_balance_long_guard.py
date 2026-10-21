# SPDX-License-Identifier: Apache-2.0
"""Keep output redistribution in the B1/long-G6 resource domains with evidence."""

import json
import shutil
from pathlib import Path

from prepare_output_work_balance_guard import load_policy


def main():
    here = Path(__file__).resolve().parent
    original = here / "runs/output_work_balance_native_candidate_v2"
    target = here / "runs/output_work_balance_native_candidate_v3"
    shutil.copytree(original, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "policy.py"
    before = path.read_text()
    old = """        # B1 has an explicitly allowed separate policy. For multiple requests,
        # retain only the existing >=64-token resource domain below the N128
        # upper boundary. B4 was flat and B8 regressed; neither supports broad
        # T<=96 adoption. This is an empirical interval, not a bottleneck proof.
        output_work_balance=(
            (requests == 1 and max_query_tokens <= 6)
            or EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS
        ),
"""
    new = """        # B1 reuses its existing balanced half-leaf domain. Multiple-request
        # redistribution is limited to long G6 and the existing 64..96-token
        # resource interval. Short G2/B16 regressed despite exact outputs;
        # neither equal tile counts nor source-level balance proves speedup.
        output_work_balance=(
            single_request_leaves
            or (long_history and group == 6
                and EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS)
        ),
"""
    assert before.count(old) == 1
    path.write_text(before.replace(old, new))
    first = load_policy(original / "policy.py", "output_balance_previous_policy")
    second = load_policy(path, "output_balance_long_guard_policy")
    cases = []
    for tokens, requests, history, expected in (
        (6, 1, 131072, True), (24, 4, 131072, False), (48, 8, 131072, False),
        (65, 16, 131072, True), (66, 11, 131072, True), (96, 16, 131072, True),
        (96, 16, 8192, False), (168, 28, 131072, False), (192, 32, 8192, False),
        (6, 1, 8192, False), (6, 1, 1048570, False),
    ):
        kwargs = dict(tokens=tokens, requests=requests, max_query_tokens=6, max_seq_len=history + 6)
        a, b = first(**kwargs).constexprs(), second(**kwargs).constexprs()
        assert b["OUTPUT_WORK_BALANCE"] == expected
        assert {k: v for k, v in a.items() if k != "OUTPUT_WORK_BALANCE"} == {
            k: v for k, v in b.items() if k != "OUTPUT_WORK_BALANCE"
        }
        cases.append(dict(tokens=tokens, requests=requests, history=history, enabled=expected,
                          old_and_new_policy_equal=a == b))
    assert all((target / p.name).read_bytes() == p.read_bytes()
               for p in original.glob("*.py") if p.name != "policy.py")
    description = json.loads((here / "runs/output_work_balance_guard_h8192_b16_s6_v70"
                              / "buffer_descriptors.json").read_text())
    description["policy"]["OUTPUT_WORK_BALANCE"] = False
    for name in ("disabled", "control"):
        if name == "control":
            del description["policy"]["OUTPUT_WORK_BALANCE"]
        (here / f"compile_descriptors/output_balance_short_{name}.json").write_text(
            json.dumps(description, indent=2) + "\n"
        )
    report = dict(candidate=str(target), parent=str(original), changed_files=["policy.py"], cases=cases,
                  adopted=False, device_execution=False, remaining="Corrected mixed fixture; boundary; full8 gate")
    (here / "output_work_balance_v3.json").write_text(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
