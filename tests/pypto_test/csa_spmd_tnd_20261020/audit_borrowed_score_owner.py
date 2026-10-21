# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU ownership checks plus generated-code evidence; no NPU claim."""

import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace


def normalized_cpp(path):
    return "\n".join(line for line in path.read_text().splitlines()
                     if not line.lstrip().startswith("// pto:"))


def load_scalar_functions(path, names, namespace):
    """Execute the actual scalar bodies with bounds-checked list primitives.

    Only decorators and type annotations are removed. This does not model NPU
    concurrency, cache visibility, compiler lowering, or device integer widths.
    """
    selected = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            node.returns = None
            for argument in node.args.args:
                argument.annotation = None
            selected.append(node)
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def scalar_namespace(source):
    def read(buffer, indices):
        assert len(indices) == 1 and 0 <= indices[0] < len(buffer)
        return buffer[indices[0]]

    def write(buffer, indices, value):
        assert len(indices) == 1 and 0 <= indices[0] < len(buffer)
        assert -(1 << 31) <= value < (1 << 31)
        buffer[indices[0]] = value

    def wait(signals, offset, generation):
        signals.append((offset, generation))
        return signals

    namespace = {
        "GROUPS": 24, "CORES": 72, "VECTORS": 48, "HEAD_DIM": 512,
        "KV_DENSE_M_TILE": 64, "KV_FIXED_SMALL_ROWS": 128, "KV_M_TILE": 16,
        "KV_N_TILE": 128, "KV_OM": 3, "QPROJ_N_BLOCKS": 128, "QPROJ_PIPE_M_TILE": 128,
        "GROUP_DESCRIPTOR_ALIGNMENT": 16, "GROUP_DESCRIPTOR_WORDS": 4,
        "SCORE_TILE": 1024, "MAX_LEAVES": 32, "CANDIDATES_PER_LEAF": 8192, "SCORE_WORKERS": 24,
        "SCALE_PLAN_ALIGNMENT": 16, "GROUP_WORDS_PER_TOKEN": 4, "SCALE_JOB_WORDS": 4,
        "LAYOUT_WORDS_PER_REQUEST": 2, "SIGNAL_SLOT_WORDS": 128, "PHASE_SCORE_VECTOR": 35,
        "SCORE_BALANCE_BRANCH_GROUPS": 4, "SCORE_BALANCE_WORDS": 96,
        "SCORE_MIN_IMBALANCE": 2, "SCORE_COST_SENTINEL": 1 << 30,
        "csa_signal_wait": wait,
        "pl": SimpleNamespace(
            INDEX="index", INT32="int32", cast=lambda value, dtype: int(value),
            range=range, read=read, write=write, min=min, max=max,
            system=SimpleNamespace(fence=lambda: None, cacheinvalid=lambda: None),
        ),
    }
    for filename, names in (
        ("indexer_group_stages.py", ["indexer_work_item", "indexer_leaf_range"]),
        ("indexer_scale_plan_stages.py", ["indexer_scale_plan_base"]),
        ("indexer_scale_layout_stages.py", ["indexer_scale_layout_base"]),
        ("score_owner_stages.py", ["score_balance_base", "score_balance_plan_stage",
                                   "score_logical_worker", "wait_balanced_score_roots"]),
    ):
        load_scalar_functions(source / filename, names, namespace)
    return namespace


def verify_ownership(namespace, workspace, plan, requests, tokens):
    base = namespace["score_balance_base"](requests, tokens)
    mapping = workspace[base + 48:base + 72]
    inverse = workspace[base + 72:base + 96]
    assert sorted(mapping) == sorted(inverse) == list(range(24))
    assert all(inverse[logical] == physical for physical, logical in enumerate(mapping))
    groups, leaves = plan[0], plan[2]
    writers = {}
    # Enumerate the actual producer work-item traversal, independently of the
    # consumer's logical-owner formula and independently for both vector lanes.
    for physical, logical in enumerate(mapping):
        assert namespace["score_logical_worker"](workspace, requests, tokens, physical) == logical
        for item in range(logical, groups * leaves, 24):
            group, leaf = namespace["indexer_work_item"](item, groups, leaves)
            for lane in range(2):
                key = (group, leaf, lane)
                assert key not in writers
                writers[key] = 24 + 2 * physical + lane
    assert len(writers) == 2 * groups * leaves
    for group in range(groups):
        for local_query in range(6):
            signals = namespace["wait_balanced_score_roots"](
                [], workspace, plan, requests, tokens, group, local_query, leaves, 7, 6,
            )
            expected = [((35 * 72 + writers[group, leaf, local_query // 3]) * 128, 7)
                        for leaf in range(leaves)]
            assert signals == expected
    return mapping


def check_scalar_cases(source):
    ns = scalar_namespace(source)
    requests, tokens = 48, 168
    base = ns["score_balance_base"](requests, tokens)
    workspace = [37] * (base + 96 + 16)
    descriptor = (requests + 1 + 15) // 16 * 16
    for group in range(48):
        workspace[descriptor + 4 * group + 3] = 32769
    prefix = workspace[:base]
    cases = []
    specifications = [
        ("b28_active", [28, 0, 4, 8, 1, 6, 1, 0], False),
        ("negative_bool_true", [28, 0, 4, 8, 1, 6, -1, 0], False),
        ("empty_after_active", [0, 0, 0, 0, 0, 6, 1, 0], True),
        ("few_groups_identity", [23, 0, 4, 8, 1, 6, 1, 0], True),
        ("balanced_24_identity", [24, 0, 4, 8, 0, 6, 1, 0], True),
        ("balanced_48_identity", [48, 0, 4, 8, 0, 6, 1, 0], True),
        ("too_many_groups", [49, 0, 4, 8, 1, 6, 1, 0], True),
        ("disabled_plan", [28, 0, 4, 8, 1, 6, 0, 0], True),
        ("invalid_leaf_count", [28, 0, 0, 8, 1, 6, 1, 0], True),
        ("invalid_extra", [28, 0, 4, 8, 4, 6, 1, 0], True),
        ("active_again", [28, 0, 4, 8, 1, 6, 1, 0], False),
    ]
    for name, plan, identity in specifications:
        ns["score_balance_plan_stage"](workspace, plan, requests, tokens)
        mapping = verify_ownership(ns, workspace, plan, requests, tokens)
        assert (mapping == list(range(24))) is identity
        assert workspace[:base] == prefix and workspace[base + 96:] == [37] * 16
        cases.append({"name": name, "identity": identity, "mapping": mapping})
    # The checker must reject a corrupted inverse; otherwise the wait check
    # could pass while missing the exact synchronization regression of interest.
    workspace[base + 72], workspace[base + 73] = workspace[base + 73], workspace[base + 72]
    try:
        verify_ownership(ns, workspace, specifications[-1][1], requests, tokens)
    except AssertionError:
        rejected_corrupt_inverse = True
    else:
        raise AssertionError("corrupted inverse accepted")
    return {"passed": True, "cases": cases, "rejected_corrupt_inverse": rejected_corrupt_inverse,
            "scope": "Actual scalar source, bounded CPU emulation; not device or memory-order proof"}


def check_codegen(here):
    runs = here / "runs"
    old = runs / "output_balance_reuse_v73/b16_candidate/kernel"
    disabled = runs / "borrowed_score_owner_b16_cpu_v88/kernel"
    active = runs / "borrowed_score_owner_b28_cpu_v88/kernel"
    proofs = [root / relative for root in (old, disabled)
              for relative in ("ptoas/dynamic_csa_full.cpp", "orchestration/decode_csa_spmd.cpp")]
    checks = {
        "disabled_b16_cpp_equal": normalized_cpp(proofs[0]) == normalized_cpp(proofs[2]),
        "disabled_b16_host_byte_equal": proofs[1].read_bytes() == proofs[3].read_bytes(),
    }
    launch = (active / "orchestration/decode_csa_spmd.cpp").read_text()
    checks["one_mix_24_groups"] = (launch.count("rt_submit_task(") == 1
                                   and re.findall(r"set_block_num\((\d+)\)", launch) == ["24"])
    allocation_pass = next((active / "passes_dump").glob("*after_AllocateMemoryAddr.py"))
    tree = ast.parse(allocation_pass.read_text())
    parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    # Locate the unique allocation of the 96-word Score table by its initialized
    # identity loop, without depending on compiler-assigned inline suffixes.
    writes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and ast.unparse(node.func) == "pl.tensor.write"
              and len(node.args) > 1 and "best_physical_inline" in ast.unparse(node.args[1])]
    assert len(writes) == 1
    bases = [node.id for node in ast.walk(writes[0].args[1])
             if isinstance(node, ast.Name) and node.id.startswith("base_inline")]
    assert len(bases) == 1
    table_writes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "pl.tensor.write" and len(node.args) > 1
                    and bases[0] in ast.unparse(node.args[1])]
    guards = []
    for node in table_writes:
        path = []
        current = node
        while current in parent:
            current = parent[current]
            if isinstance(current, ast.If):
                path.append(ast.unparse(current.test))
        assert "worker__ssa_v0 == planner__ssa_v0" in path
        guards.append({"line": node.lineno, "guards": path})
    checks["all_ten_table_write_sites_have_unique_planner_guard"] = len(guards) == 10
    assert all(checks.values()), checks
    return {"passed": True, "checks": checks, "scalar_write_guards": guards,
            "proof_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in proofs + [allocation_pass]}}


def check_policy(here, source):
    spec = importlib.util.spec_from_file_location("score_owner_candidate_policy", source / "policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cases = 0
    for tokens in (1, 16, 24, 64, 96, 128, 144, 145, 168, 192, 256, 257):
        for query in (5, 6):
            for maximum in (8198, 32768, 32772, 131078):
                selected = module.select_policy(
                    tokens=tokens, requests=(tokens + query - 1) // query,
                    max_query_tokens=query, max_seq_len=maximum,
                )
                assert selected.score_prefix_balance == (query == 6 and maximum >= 32772 and 144 < tokens <= 256)
                cases += 1
    for label in ("b28", "b16"):
        description = json.loads((here / "compile_descriptors" / f"borrowed_score_owner_{label}_v88.json").read_text())
        tokens = description["buffers"]["positions"]["shape"][0]
        requests = description["buffers"]["idx_lengths"]["shape"][0]
        policy = module.select_policy(tokens=tokens, requests=requests, max_query_tokens=6, max_seq_len=131078)
        assert description["policy"] == policy.constexprs()
        size = ((requests + 16) // 16 * 16 + 4 * tokens + 15) // 16 * 16 + 48 + 4 * tokens
        if policy.scale_page_coalescing or policy.score_prefix_balance:
            size += 2 * requests
        if policy.score_prefix_balance:
            size += 96
        assert description["buffers"]["group_offsets"]["shape"] == [size]
    return {"passed": True, "boundary_cases": cases, "descriptor_policy_and_capacity_matches": ["b28", "b16"]}


def main():
    here = Path(__file__).resolve().parent
    source = here / "runs/borrowed_score_owner_candidate_v88"
    report = {"scalar": check_scalar_cases(source), "codegen": check_codegen(here),
              "policy": check_policy(here, source),
              "device_execution": False, "adopted": False}
    report["passed"] = all(report[name]["passed"] for name in ("scalar", "codegen", "policy"))
    (here / "borrowed_score_owner_audit_v88.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "scalar_cases": len(report["scalar"]["cases"]),
                      "codegen": report["codegen"]["checks"]}, indent=2))


if __name__ == "__main__":
    main()
