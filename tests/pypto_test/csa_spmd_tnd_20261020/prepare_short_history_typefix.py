# SPDX-License-Identifier: Apache-2.0
"""Fix the B1-only temporary type without changing either Score algorithm.

The parser checks the inactive balanced-G2 branch at short SCORE_TILE=768.
Using floor(8192 / (2*768))*768 describes3840 elements, but the branch later
reshapes to4096. The B1 algorithm is a fixed4096 half-leaf with1024-wide
chunks, so its temporary must use BUFFERED_LONG_SCORE_TILE independently of
the inactive short-history tile. The long-history shape remains4x1024.
"""

import difflib
import hashlib
import json
import shutil
from pathlib import Path


def correct(path):
    before = path.read_text()
    marker = "[TOPK_CANDIDATES_PER_LEAF // (2 * score_tile), score_tile]"
    assert before.count(marker) == 2
    replacement = "[TOPK_CANDIDATES_PER_LEAF // (2 * BUFFERED_LONG_SCORE_TILE), BUFFERED_LONG_SCORE_TILE]"
    after = before.replace(marker, replacement)
    path.write_text(after)
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="a/indexer_score_stages.py", tofile="b/indexer_score_stages.py",
    ))


def main():
    here = Path(__file__).resolve().parent
    baseline_source = here / "bsh_baseline_a241fa61"
    baseline = here / "bsh_baseline_a241fa61_short_typefix"
    candidate_source = here / "runs/active_prefix_qa_policy_native_candidate_v2"
    candidate = here / "runs/prefix_policy_short_typefix_native_candidate_v3"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(baseline_source, baseline, ignore=ignore)
    shutil.copytree(candidate_source, candidate, ignore=ignore)
    baseline_patch = correct(baseline / "deepseek_v4_flash_csa_spmd/indexer_score_stages.py")
    candidate_patch = correct(candidate / "indexer_score_stages.py")
    assert baseline_patch == candidate_patch
    patch_path = here / "short_history_inactive_b1_typefix.patch"
    patch_path.write_text(baseline_patch)
    manifest = json.loads((here / "bsh_baseline_a241fa61_manifest.json").read_text())
    manifest["bsh"]["snapshot"] = str(baseline)
    manifest["bsh"]["files"] = {
        str(path.relative_to(baseline)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in baseline.rglob("*.py")
    }
    manifest["bsh"]["local_compile_fix"] = {
        "patch": str(patch_path), "sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
        "description": (
            "B1-only temporary has fixed4x1024 shape; short inactive branch can parse. No short math changed."
        ),
        "original_failure": "runs/bsh_a241_short_original_cpu_v44/result.json",
        "upstream_session_modified": False,
    }
    (here / "bsh_baseline_a241fa61_short_typefix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (here / "short_history_typefix.json").write_text(json.dumps({
        "candidate": str(candidate), "baseline": str(baseline), "production_adopted": False,
        "original_bsh_fails_on_accepted_tools": True,
        "precision_or_performance_claim": False,
        "required": "Full short-history compile and NPU validation; unchanged long-history generated code proof",
    }, indent=2) + "\n")
    descriptors = here / "compile_descriptors"
    descriptors.mkdir(exist_ok=True)
    for batch in (16, 32):
        shutil.copyfile(
            here / f"runs/cohort_prefix_a241_h8192_b{batch}_s6_v41/buffer_descriptors.json",
            descriptors / f"short_history_b{batch}.json",
        )
    print(candidate)


if __name__ == "__main__":
    main()
