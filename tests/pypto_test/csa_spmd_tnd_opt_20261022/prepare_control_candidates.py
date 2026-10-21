# SPDX-License-Identifier: Apache-2.0
"""Create independent output-balance and short-half-UB ablations of frozen v122."""

import ast
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Source precondition failed: {before!r}")
    return text.replace(before, after)


def main():
    here = Path(__file__).resolve().parent
    old = here.parent / "csa_spmd_tnd_20261020"
    reference = here / "runs/frozen_v122"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reference.glob("*.py")}
    if actual != expected:
        raise ValueError("Candidate must start from exact frozen v122")
    original = (reference / "policy.py").read_text()
    variants = {
        "output_balance_off": replace_once(
            original,
            "        output_work_balance=(\n"
            "            single_request_leaves\n"
            "            or (long_history and group == 6 and "
            "EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS)\n"
            "        ),",
            "        output_work_balance=False,",
        ),
        "short_half_ub96": replace_once(
            original,
            "# At most two query-group work items per each of the 24 physical groups.\n"
            "SHORT_HALF_SCORE_GROUP_LIMIT = 48",
            "# Candidate: four sequential group waves reuse each AIV's private half-leaf UB.\n"
            "# Lifecycle/address reuse must be verified after compilation and on device.\n"
            "SHORT_HALF_SCORE_GROUP_LIMIT = 96",
        ),
    }
    report = {"reference_kind": "frozen_tnd_v122", "adopted": False, "device_execution": False, "variants": {}}
    for name, text in variants.items():
        ast.parse(text)
        target = here / "runs" / name
        shutil.copytree(reference, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (target / "policy.py").write_text(text)
        spec = importlib.util.spec_from_file_location(name, target / "policy.py")
        policy = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = policy
        spec.loader.exec_module(policy)
        cases = ((131072, 1), (131072, 16)) if name == "output_balance_off" else ((8192, 32),)
        descriptors = []
        for history, batch in cases:
            desc = json.loads((old / f"compile_descriptors/shared_combined_v122_h{history}_b{batch}.json").read_text())
            selected = policy.select_policy(
                tokens=batch * 6, requests=batch, max_query_tokens=6, max_seq_len=history + 6
            )
            previous = desc["policy"]
            desc["policy"] = selected.constexprs()
            changed = {k: [previous[k], v] for k, v in desc["policy"].items() if previous[k] != v}
            required = (
                {"OUTPUT_WORK_BALANCE": [True, False]}
                if name == "output_balance_off"
                else {"SHORT_HALF_SCORE_UB": [False, True]}
            )
            if changed != required:
                raise ValueError(f"Unexpected non-isolated policy change: {changed}")
            path = here / "descriptors" / f"{name}_h{history}_b{batch}.json"
            path.parent.mkdir(exist_ok=True)
            with path.open("x") as stream:
                stream.write(json.dumps(desc, indent=2) + "\n")
            descriptors.append({"path": str(path), "history": history, "batch": batch, "changes": changed})
        report["variants"][name] = {
            "source": str(target),
            "descriptors": descriptors,
            "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
        }
    with (here / "control_candidates.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"prepared": list(variants), "device_execution": False}))


if __name__ == "__main__":
    main()
