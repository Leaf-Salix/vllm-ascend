# SPDX-License-Identifier: Apache-2.0
"""Freeze independent current-baseline trials for newly permitted finer tiers."""

import ast
import collections
import difflib
import hashlib
import json
import shutil
from pathlib import Path


def verify_large_output_grid():
    """Check both projection grids including every row tile and column owner."""
    cases = []
    for tokens in range(129, 257):
        oa_jobs = 8 * ((tokens + 127) // 128) * (1024 // 256)
        assert oa_jobs == 64
        ob_rows = (tokens + 127) // 128
        writes = collections.Counter()
        counts = [0] * 24
        for physical in range(24):
            for group in range(8):
                rank = physical + 16 if physical < 16 else physical - 16 + (group % 2) * 8
                first = group * 8
                owned = rank + ((first + 31 - rank) // 32) * 32
                if owned >= first + 8:
                    continue
                # These are the original NZ helper's loop bounds. Each owner
                # keeps all of its row tiles and its two N256 column halves.
                for wave in range(first // 32, (first + 8 - rank + 31) // 32):
                    work = rank + wave * 32
                    if work < first:
                        continue
                    counts[physical] += 1
                    for row in range(ob_rows):
                        for half in range(2):
                            writes[(work // 8, row, (work % 8) * 2 + half)] += 1
        expected = {(group, row, column) for group in range(8)
                    for row in range(ob_rows) for column in range(16)}
        assert set(writes) == expected and set(writes.values()) == {1}
        assert counts == [2] * 16 + [4] * 8
        cases.append({"tokens": tokens, "oa_jobs": oa_jobs, "output_tiles": len(writes)})
    return {"passed": True, "cases": cases, "device_execution": False}


def main():
    here = Path(__file__).resolve().parent
    origin = here / "runs/output_work_balance_native_candidate_v4"
    variants = {
        "large_output_balance": {
            "anchor": "            single_request_leaves\n",
            "replacement": (
                "            single_request_leaves\n"
                "            or (long_history and group == 6 and 128 < tokens <= 256)\n"
            ),
            "run": "cohort_output_balance_h131072_b28_s6_v75",
            "flag": "OUTPUT_WORK_BALANCE",
        },
        "short_hc_consumer": {
            "anchor": "        cooperative_prefix=True,\n",
            "replacement": (
                "        cooperative_prefix=True,\n"
                "        hc_consumer_reduce=(\n"
                "            not long_history and EARLY_GEOMETRY_MIN_TOKENS <= tokens <= 256\n"
                "        ),\n"
            ),
            "run": "cohort_output_balance_h8192_b32_s6_v76",
            "flag": "HC_CONSUMER_REDUCE",
        },
    }
    report = {"adopted": False, "device_execution": False,
              "origin": str(origin), "large_output_ownership": verify_large_output_grid(), "variants": {}}
    for name, setting in variants.items():
        target = here / "runs" / (name + "_candidate_v78")
        shutil.copytree(origin, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        path = target / "policy.py"
        before = path.read_text()
        assert before.count(setting["anchor"]) == 1
        after = before.replace(setting["anchor"], setting["replacement"])
        ast.parse(after)
        path.write_text(after)
        (here / (name + "_v78.patch")).write_text("".join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), fromfile="a/policy.py", tofile="b/policy.py"
        )))
        description = json.loads((here / "runs" / setting["run"] / "buffer_descriptors.json").read_text())
        assert description["policy"][setting["flag"]] is False
        description["policy"][setting["flag"]] = True
        descriptor_path = here / "compile_descriptors" / (name + "_v78.json")
        descriptor_path.write_text(json.dumps(description, indent=2) + "\n")
        changed = [x.name for x in origin.glob("*.py") if x.read_bytes() != (target / x.name).read_bytes()]
        assert changed == ["policy.py"]
        report["variants"][name] = {
            "source": str(target), "descriptors": str(descriptor_path), "changed_files": changed,
            "policy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (here / "fine_tier_candidates_v78.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["variants"], indent=2))


if __name__ == "__main__":
    main()
