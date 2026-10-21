# SPDX-License-Identifier: Apache-2.0
"""Audit active UB residency and exact inactive code before a device trial."""

import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

from audit_borrowed_score_stream import normalized_cpp


def load_policy(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def integer(node):
    return ast.literal_eval(node.args[0] if isinstance(node, ast.Call) else node)


def main():
    here = Path(__file__).resolve().parent
    origin = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    candidate = here / "runs/short_half_ub_candidate_v95"
    checks, proofs = {}, []
    for case, control in (
        ("short_b32", here / "runs/short_half_ub_control_short_b32_cpu_v95/kernel"),
        ("long_b16", here / "runs/output_balance_reuse_v73/b16_candidate/kernel"),
    ):
        kernel = here / f"runs/short_half_ub_{case}_cpu_v95/kernel"
        files = [
            root / name
            for root in (control, kernel)
            for name in ("ptoas/dynamic_csa_full.cpp", "orchestration/decode_csa_spmd.cpp")
        ]
        checks[case + "_whole_cpp_equal"] = normalized_cpp(files[0]) == normalized_cpp(files[2])
        checks[case + "_host_byte_equal"] = files[1].read_bytes() == files[3].read_bytes()
        proofs.append({"case": case, "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    kernel = here / "runs/short_half_ub_short_b16_cpu_v95/kernel"
    passes = list((kernel / "passes_dump").glob("*after_AllocateMemoryAddr.py"))
    assert len(passes) == 1
    tree = ast.parse(passes[0].read_text())
    vector = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "vector_schedule"
    )
    allocations = []
    for node in ast.walk(vector):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if "pl.Mem.Vec" not in ast.unparse(node.annotation):
            continue
        refs = [n for n in ast.walk(node.annotation) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.MemRef"]
        assert len(refs) == 1
        ref = refs[0]
        allocations.append(
            {
                "name": node.target.id,
                "root": ast.unparse(ref.args[0]),
                "offset": integer(ref.args[1]),
                "bytes": integer(ref.args[2]),
            }
        )
    cache = [row for row in allocations if row["name"].startswith("buf_half_scores_")]
    checks["active_half_cache_33792_bytes"] = bool(cache) and all(row["bytes"] == 33792 for row in cache)
    checks["active_half_cache_same_storage"] = len({(r["root"], r["offset"]) for r in cache}) == 1
    peak = max(row["offset"] + row["bytes"] for row in allocations)
    checks["active_ub_fits"] = peak <= 188416
    score_writes = [
        node
        for node in ast.walk(vector)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) in ("pl.store", "pl.tile.store")
        and any(isinstance(n, ast.Name) and n.id.startswith("score_arena") for n in ast.walk(node))
    ]
    checks["active_score_arena_has_no_stores"] = not score_writes
    launch = (kernel / "orchestration/decode_csa_spmd.cpp").read_text()
    checks["single_mix24"] = launch.count("rt_submit_task(") == 1 and re.findall(r"set_block_num\((\d+)\)", launch) == [
        "24"
    ]
    cohort = json.loads((here / "cohort_output_balance_v4.json").read_text())
    task = next(row["task_id"] for row in cohort["cases"] if (row["history"], row["batch"]) == (8192, 16))
    job = next(row for row in json.loads((here / "jobs.json").read_text()) if row["task_id"] == task)
    old_paths = [
        p
        for p in (Path(job["output"]) / "build_output").glob("decode_csa_spmd_*/ptoas/dynamic_csa_full.cpp")
        if "PTOAS__WAIT_GE_BACKOFF<150>" in p.read_text()
    ]
    assert len(old_paths) == 1
    new_path = kernel / "ptoas/dynamic_csa_full.cpp"
    cube_pattern = r"AICORE void cube_schedule\([\s\S]*?^\}"
    cube_texts = [re.search(cube_pattern, normalized_cpp(p), re.MULTILINE).group() for p in (old_paths[0], new_path)]
    checks["active_cube_cpp_equal"] = cube_texts[0] == cube_texts[1]
    policies = [
        load_policy(root / "policy.py", name)
        for root, name in ((origin, "original_half_policy"), (candidate, "candidate_half_policy"))
    ]
    cases = 0
    for requests in (1, 8, 15, 16, 17, 23, 24, 25, 32, 47, 48, 49):
        for query in range(1, 7):
            for history in (8192, 15360, 32760, 131072):
                for tokens in sorted({1, requests, requests * query}):
                    kwargs = dict(tokens=tokens, requests=requests, max_query_tokens=query, max_seq_len=history + query)
                    old, new = [p.select_policy(**kwargs).constexprs() for p in policies]
                    enabled = new.pop("SHORT_HALF_SCORE_UB")
                    assert new == old
                    expected = (
                        kwargs["max_seq_len"] // 4 <= 8192
                        and old["QUERY_GROUP"] == 2
                        and requests * ((query + 1) // 2) <= 48
                    )
                    assert enabled == expected
                    cases += 1
    checks["host_policy_other_controls_unchanged"] = True
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "disabled_proofs": proofs,
        "active_cube_files": [str(old_paths[0]), str(new_path)],
        "active_allocation_pass": str(passes[0]),
        "active_allocation_pass_sha256": hashlib.sha256(passes[0].read_bytes()).hexdigest(),
        "active_half_cache": cache,
        "vector_peak_end_bytes": peak,
        "host_policy_cases": cases,
        "device_execution": False,
        "scope": "CPU code/resource proof only; exact device state, tie and L1 evidence still required",
    }
    (here / "short_half_ub_codegen_audit_v95.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": checks, "peak": peak, "policy_cases": cases}, indent=2))
    assert report["passed"], checks


if __name__ == "__main__":
    main()
