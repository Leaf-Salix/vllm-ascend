# SPDX-License-Identifier: Apache-2.0
"""Narrow the output redistribution after the B4/B8 paired measurements."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path


def load_policy(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.select_policy


def main():
    here = Path(__file__).resolve().parent
    original = here / "runs/output_work_balance_native_candidate_v1"
    target = here / "runs/output_work_balance_native_candidate_v2"
    shutil.copytree(original, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "policy.py"
    before = path.read_text()
    old = "        output_work_balance=tokens <= LOCAL_STATS_MAX_TOKENS,\n"
    new = """        # B1 has an explicitly allowed separate policy. For multiple requests,
        # retain only the existing >=64-token resource domain below the N128
        # upper boundary. B4 was flat and B8 regressed; neither supports broad
        # T<=96 adoption. This is an empirical interval, not a bottleneck proof.
        output_work_balance=(
            (requests == 1 and max_query_tokens <= 6)
            or EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS
        ),
"""
    assert before.count(old) == 1
    path.write_text(before.replace(old, new))
    first = load_policy(original / "policy.py", "output_balance_v1_policy")
    second = load_policy(path, "output_balance_v2_policy")
    cases = []
    for tokens, requests, expected in ((1, 1, True), (6, 1, True), (12, 2, False), (24, 4, False),
                                       (48, 8, False), (63, 11, False), (64, 11, True), (65, 12, True),
                                       (66, 11, True), (96, 16, True), (97, 17, False), (168, 28, False)):
        kwargs = dict(tokens=tokens, requests=requests, max_query_tokens=6, max_seq_len=131078)
        a, b = first(**kwargs).constexprs(), second(**kwargs).constexprs()
        assert b["OUTPUT_WORK_BALANCE"] == expected
        assert {k: v for k, v in a.items() if k != "OUTPUT_WORK_BALANCE"} == {
            k: v for k, v in b.items() if k != "OUTPUT_WORK_BALANCE"
        }
        cases.append(dict(tokens=tokens, requests=requests, enabled=expected, old_and_new_policy_equal=a == b))
    assert all((target / f.name).read_bytes() == f.read_bytes()
               for f in original.glob("*.py") if f.name != "policy.py")
    desc = json.loads((here / "runs/output_work_balance_h131072_b8_s6_v68/buffer_descriptors.json").read_text())
    desc["policy"]["OUTPUT_WORK_BALANCE"] = False
    disabled = here / "compile_descriptors/output_work_balance_disabled_b8.json"
    disabled.write_text(json.dumps(desc, indent=2) + "\n")
    del desc["policy"]["OUTPUT_WORK_BALANCE"]
    control = here / "compile_descriptors/output_work_balance_control_b8.json"
    control.write_text(json.dumps(desc, indent=2) + "\n")
    report = dict(candidate=str(target), parent=str(original), changed_files=["policy.py"], cases=cases,
                  disabled_descriptors=str(disabled), control_descriptors=str(control),
                  adopted=False, device_execution=False,
                  remaining="Short history, mixed/capacity tail, disabled generated code, coherent full8 cohort")
    (here / "output_work_balance_v2.json").write_text(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
