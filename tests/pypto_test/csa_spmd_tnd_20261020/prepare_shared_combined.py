# SPDX-License-Identifier: Apache-2.0
"""Combine only the shared-SDK-supported wide-N domain and masked RoPE tail."""

import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_score_owner_harness import replace_once


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    source = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    target = here / "runs/shared_combined_candidate_v122"
    evidence = json.loads((here / "runs/query_rope_tail_helper_device_v121/result.json").read_text())
    assert evidence["passed"] and len(evidence["cases"]) == 39 and all(row["passed"] for row in evidence["cases"])
    tail = json.loads((here / "query_rope_tail_results_v118.json").read_text())
    assert tail["passed"] and tail["incremental_percent"] < 0 and tail["b1_s1_s6_matrix_passed"]
    three_way = json.loads((here / "three_way_shared_results_v117.json").read_text())
    wide = next(case for case in three_way["cases"] if "b8" in case["case"])
    assert wide["passed"] and wide["candidate_vs_current_percent"] < 0
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(
        here / "runs/query_rope_tail_candidate_v118/indexer_query_stages.py", target / "indexer_query_stages.py"
    )
    text = (target / "policy.py").read_text()
    tree = ast.parse(text)
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "CSASPMDPolicy"
    )
    replacements = {
        "qb_wide_n": "long_history and group == 6 and SCORE_SPLIT_MIN_TOKENS <= tokens < EARLY_GEOMETRY_MIN_TOKENS",
        "prefix_branch_groups": "4",
    }
    for keyword in call.keywords:
        if keyword.arg in replacements:
            text = replace_once(
                text,
                ast.get_source_segment((target / "policy.py").read_text(), keyword.value),
                replacements[keyword.arg],
            )
    text = replace_once(
        text,
        "# Two M256 query-Hadamard tiles cover the existing first M8 token block.\nTWO_HADAMARD_MAX_TOKENS = 8\n",
        "",
    )
    text = replace_once(
        text,
        "        # Keep the original Q_B in the small-M16 QA domain, which regressed in the paired test.\n",
        "        # Shared-SDK three-way evidence supports the existing long-G6 [32,64) domain.\n"
        "        # G2 wide-N/two-core prefix showed no benefit; keep the original four producers.\n",
    )
    (target / "policy.py").write_text(text)
    workspace = next(parent for parent in here.parents if (parent / "env-dsv4-0251rc1.sh").is_file())
    subprocess.run([str(workspace / ".venv/bin/ruff"), "format", str(target / "policy.py")], check=True)
    spec = importlib.util.spec_from_file_location("shared_combined_policy_v122", target / "policy.py")
    policy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = policy
    spec.loader.exec_module(policy)
    cases = []
    for batch, history in (
        (1, 131072),
        (4, 131072),
        (8, 131072),
        (16, 131072),
        (24, 131072),
        (28, 131072),
        (16, 8192),
        (32, 8192),
    ):
        selected = policy.select_policy(tokens=batch * 6, requests=batch, max_query_tokens=6, max_seq_len=history + 6)
        assert selected.qb_wide_n == (batch == 8 and history == 131072)
        assert selected.prefix_branch_groups == 4
        path = here / "compile_descriptors" / f"shared_combined_v122_h{history}_b{batch}.json"
        desc = json.loads((here / "compile_descriptors" / f"qb_prefix_v114_h{history}_b{batch}.json").read_text())
        desc["policy"] = selected.constexprs()
        path.write_text(json.dumps(desc, indent=2) + "\n")
        cases.append(dict(batch=batch, history=history, descriptor=str(path), policy=desc["policy"]))
    report = dict(
        source=str(target),
        base=str(source),
        adopted=False,
        device_execution=False,
        release=three_way["release"],
        cases=cases,
        evidence=[
            "three_way_shared_results_v117.json",
            "query_rope_tail_results_v118.json",
            "runs/query_rope_tail_helper_device_v121/result.json",
        ],
        historical_b1_wide_and_prefix="Retained in v114; not enabled by this shared-SDK default",
        source_manifest={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    (here / "shared_combined_candidate_v122.json").write_text(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
