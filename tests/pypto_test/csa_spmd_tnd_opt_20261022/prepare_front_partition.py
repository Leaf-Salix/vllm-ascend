# SPDX-License-Identifier: Apache-2.0
"""Partition the single-SPMD front into disjoint Indexer and main QKV groups.

The mathematical helpers remain unchanged. Only worker domains, their actual
phase owners, and stage order change. The original whole schedule remains the
constexpr fallback. Scale-page ownership keeps its original 48 logical writers.
"""

import argparse
import ast
import copy
import hashlib
import json
import shutil
import textwrap
from pathlib import Path

GUARD = "COOPERATIVE_PREFIX and QUERY_GROUP == 6"


def function(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)


class Names(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            return ast.copy_location(ast.parse(self.mapping[node.id], mode="eval").body, node)
        return node


def call_statement(source, name, mapping=None, rename=None):
    node = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, (ast.Assign, ast.Expr)) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == name
    )
    result = Names(mapping or {}).visit(copy.deepcopy(node))
    if rename:
        result.value.func = ast.Name(id=rename, ctx=ast.Load())
    return ast.unparse(ast.fix_missing_locations(result)) + "\n"


def publish(phase, cube=False):
    return f"signal = publish_phase(signal, {phase}, {'worker' if cube else 'core'}, CORES, generation)\n"


def wait(phase, count, workers, first="0", unit="1"):
    return f"signal = wait_row_owners(signal, {phase}, CORES, generation, 0, {count}, {unit}, {workers}, {first}, 1)\n"


def forked_helpers(source, groups):
    """Clone only scheduling wrappers, preserving their math and state lifetimes."""
    clones = []
    for label in ("indexer", "main"):
        old_name = label + "_compressor_vector_stage"
        fn = copy.deepcopy(function(source, old_name))
        fn.name = "front_" + old_name
        vector_count = "FRONT_INDEX_VECTORS" if label == "indexer" else "FRONT_MAIN_VECTORS"
        fn = Names({"VECTORS": vector_count}).visit(fn)
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "wait_row_owners":
                phase = ast.unparse(node.args[1])
                if phase == "PHASE_IDX_PROJECT":
                    node.args[7] = ast.Name(id="FRONT_INDEX_GROUPS", ctx=ast.Load())
                elif phase == "PHASE_MAIN_PROJECT":
                    node.args[7] = ast.Name(id="FRONT_MAIN_GROUPS", ctx=ast.Load())
                    node.args[8] = ast.Name(id="FRONT_INDEX_GROUPS", ctx=ast.Load())
                elif phase == "PHASE_MAIN_POOL":
                    node.args[8] = ast.parse("GROUPS + FRONT_INDEX_VECTORS", mode="eval").body
        clones.append(ast.unparse(ast.fix_missing_locations(fn)))
    constants = (
        f"\nFRONT_INDEX_GROUPS = {groups}\nFRONT_INDEX_VECTORS = {2 * groups}\n"
        f"FRONT_MAIN_GROUPS = {24 - groups}\nFRONT_MAIN_VECTORS = {2 * (24 - groups)}\n"
    )
    return source + constants + "\n\n" + "\n\n".join(clones) + "\n"


def cube_region(source, nonnegative_qb=False):
    idx = {"GROUPS": "FRONT_INDEX_GROUPS"}
    main = {"GROUPS": "FRONT_MAIN_GROUPS", "worker": "worker - FRONT_INDEX_GROUPS"}
    a = publish("PHASE_MAIN_PROJECT", True) + publish("PHASE_QB", True)
    a += call_statement(source, "indexer_compressor_project_stage", idx) + publish("PHASE_IDX_PROJECT", True)
    a += wait("PHASE_QR", "tokens", "VECTORS", "GROUPS", "8")
    a += call_statement(source, "indexer_query_project_stage", idx) + publish("PHASE_IDX_QUERY_PROJECT", True)
    a += call_statement(source, "indexer_weights_project_stage", idx) + publish("PHASE_WEIGHTS_PROJECT", True)
    a += "planner = pl.min((tokens + 7) // 8, VECTORS - 1)\n"
    a += "signal = wait_producer(signal, PHASE_META, GROUPS + planner, CORES, generation)\n"
    a += "requests = pl.tensor.dim(idx_lengths, 0)\n"
    a += wait("PHASE_IDX_NORM", "requests", "FRONT_INDEX_VECTORS", "GROUPS")
    a += "compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)\n"
    a += call_statement(source, "indexer_compressor_hadamard_stage", idx) + publish("PHASE_IDX_HADAMARD", True)
    a += "query_rope_tasks = ((tokens + 7) // 8) * 16\n"
    a += wait("PHASE_IDX_QUERY_ROPE", "query_rope_tasks", "FRONT_INDEX_VECTORS", "GROUPS")
    a += (
        "idx_query_had = _query_hadamard_four_tokens(idx_query_rope_heads, hadamard, "
        "idx_query_had, tokens, worker, FRONT_INDEX_GROUPS)\n"
    )
    a += publish("PHASE_IDX_QUERY_HAD", True)
    b = "".join(
        publish(p, True)
        for p in (
            "PHASE_IDX_PROJECT",
            "PHASE_IDX_QUERY_PROJECT",
            "PHASE_WEIGHTS_PROJECT",
            "PHASE_IDX_HADAMARD",
            "PHASE_IDX_QUERY_HAD",
        )
    )
    b += call_statement(source, "compressor_project_stage", main) + publish("PHASE_MAIN_PROJECT", True)
    b += wait("PHASE_QR", "tokens", "VECTORS", "GROUPS", "8")
    qb_mapping = dict(main)
    if nonnegative_qb:
        # The remaining consecutive physical IDs form a complete modulo-domain.
        # This avoids an NZ offset whose non-negativity relies on a branch guard.
        qb_mapping["worker"] = "worker % FRONT_MAIN_GROUPS"
    b += call_statement(source, "q_b_stage", qb_mapping) + publish("PHASE_QB", True)
    result = "if worker < FRONT_INDEX_GROUPS:\n" + textwrap.indent(a, "    ")
    result += "else:\n" + textwrap.indent(b, "    ")
    result += "front_kv_worker = (worker + GROUPS - FRONT_INDEX_GROUPS) % GROUPS\n"
    result += call_statement(source, "kv_project_stage", {"worker": "front_kv_worker"})
    result += publish("PHASE_KV_PROJECT", True)
    return result


def vector_region(source):
    idx = {"VECTORS": "FRONT_INDEX_VECTORS", "COOPERATIVE_PREFIX": "True"}
    main = {"VECTORS": "FRONT_MAIN_VECTORS", "worker": "worker - FRONT_INDEX_VECTORS", "COOPERATIVE_PREFIX": "True"}
    a = "".join(publish(p) for p in ("PHASE_MAIN_POOL", "PHASE_MAIN_COMMIT", "PHASE_MAIN_OUTPUT", "PHASE_QUERY"))
    a += call_statement(source, "indexer_compressor_vector_stage", idx, "front_indexer_compressor_vector_stage")
    a += wait("PHASE_IDX_QUERY_PROJECT", "32", "FRONT_INDEX_GROUPS")
    a += call_statement(source, "indexer_query_dequant_rope_stage", idx) + publish("PHASE_IDX_QUERY_ROPE")
    a += "weights_tasks = ((tokens + 15) // 16) * 4\n"
    a += wait("PHASE_WEIGHTS_PROJECT", "weights_tasks", "FRONT_INDEX_GROUPS")
    a += call_statement(source, "indexer_weights_reduce_stage", idx) + publish("PHASE_WEIGHTS")
    a += wait("PHASE_IDX_QUERY_HAD", "(tokens + 3) // 4", "FRONT_INDEX_GROUPS")
    a += call_statement(source, "indexer_query_quant_stage", idx) + publish("PHASE_IDX_QUERY_QUANT")
    a += wait("PHASE_WEIGHTS", "tokens", "FRONT_INDEX_VECTORS", "GROUPS", "16")
    a += wait("PHASE_IDX_QUERY_QUANT", "tokens", "FRONT_INDEX_VECTORS", "GROUPS")
    a += call_statement(source, "indexer_head_coefficients_stage", idx) + publish("PHASE_COEFF")
    a += "compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)\nhad_tasks = ((compact_count + 15) // 16) * 2\n"
    a += wait("PHASE_IDX_HADAMARD", "had_tasks", "FRONT_INDEX_GROUPS")
    a += call_statement(source, "indexer_compact_quant_cache_stage", idx) + publish("PHASE_IDX_QUANT")
    a += wait("PHASE_IDX_QUANT", "idx_requests", "FRONT_INDEX_VECTORS", "GROUPS")
    # The metadata page-owner lists still have exactly 48 heads. One physical AIV
    # processes several whole logical owners; it never splits a page's 64B RMW.
    a += "for front_scale_owner in pl.range(worker, VECTORS, FRONT_INDEX_VECTORS):\n"
    a += textwrap.indent(
        call_statement(source, "indexer_scale_page_commit_stage", {"worker": "front_scale_owner"}), "    "
    )
    a += publish("PHASE_IDX_SCALE")
    b = "".join(
        publish(p)
        for p in (
            "PHASE_IDX_POOL",
            "PHASE_IDX_COMMIT",
            "PHASE_IDX_NORM",
            "PHASE_IDX_QUERY_ROPE",
            "PHASE_WEIGHTS",
            "PHASE_IDX_QUERY_QUANT",
            "PHASE_COEFF",
            "PHASE_IDX_QUANT",
            "PHASE_IDX_SCALE",
        )
    )
    # Main roles no longer enter the Indexer wrapper, which previously acquired
    # META for every AIV. Acquire its offsets/counts explicitly before main work.
    b += "signal = wait_row_owners(signal, PHASE_META, CORES, generation, planner, 1, 1, VECTORS, GROUPS, 1)\n"
    b += call_statement(source, "main_compressor_vector_stage", main, "front_main_compressor_vector_stage")
    b += wait("PHASE_QB", "128", "FRONT_MAIN_GROUPS", "FRONT_INDEX_GROUPS")
    b += call_statement(source, "q_dequant_rope_stage", main) + publish("PHASE_QUERY")
    return "if worker < FRONT_INDEX_VECTORS:\n" + textwrap.indent(a, "    ") + "else:\n" + textwrap.indent(b, "    ")


def replace_region(source, start, end, replacement):
    lines = source.splitlines(keepends=True)
    old = "".join(lines[start:end])
    lines[start:end] = [
        "    if "
        + GUARD
        + ":\n"
        + textwrap.indent(replacement, "        ")
        + "    else:\n"
        + textwrap.indent(old, "    ")
    ]
    return "".join(lines)


def add_imports(source, imports):
    existing = {a.asname or a.name for n in ast.parse(source).body if isinstance(n, ast.ImportFrom) for a in n.names}
    additions = []
    for module, names in imports.items():
        missing = sorted(set(names) - existing)
        if missing:
            additions.append(f"from .{module} import ({', '.join(missing)})\n")
    # Insert before the first function, after every original module declaration.
    lines = source.splitlines(keepends=True)
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    first = min([fn.lineno] + [n.lineno for n in fn.decorator_list]) - 1
    lines[first:first] = ["".join(additions) + "\n"]
    return "".join(lines)


def build(reference, groups, nonnegative_qb=False):
    cube = (reference / "cube_schedule.py").read_text()
    vector = (reference / "vector_schedule.py").read_text()
    cfn, vfn = function(cube, "cube_schedule"), function(vector, "vector_schedule")
    cstart = next(
        n.lineno - 1
        for n in cfn.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "indexer_compressor_project_stage"
    )
    cend = next(
        n.lineno - 1
        for n in cfn.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "wait_row_owners"
        and ast.unparse(n.value.args[1]) == "PHASE_IDX_QUERY_QUANT"
    )
    vstart = next(
        n.lineno - 1 for n in vfn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "COOPERATIVE_PREFIX"
    )
    vend = next(
        n.lineno - 1
        for n in vfn.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "wait_row_owners"
        and ast.unparse(n.value.args[1]) == "PHASE_IDX_SCALE"
    )
    # The later Attention consumer needs actual main-writer IDs, not the earlier
    # role numbers. Its fallback stays byte-for-byte intact.
    main_wait = next(
        n
        for n in vfn.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "wait_row_owners"
        and ast.unparse(n.value.args[1]) == "PHASE_MAIN_OUTPUT"
    )
    lines = vector.splitlines(keepends=True)
    old = "".join(lines[main_wait.lineno - 1 : main_wait.end_lineno])
    new = wait("PHASE_MAIN_OUTPUT", "main_requests", "FRONT_MAIN_VECTORS", "GROUPS + FRONT_INDEX_VECTORS")
    lines[main_wait.lineno - 1 : main_wait.end_lineno] = [
        "    if " + GUARD + ":\n" + textwrap.indent(new, "        ") + "    else:\n" + textwrap.indent(old, "    ")
    ]
    new_vector = replace_region("".join(lines), vstart, vend, vector_region(vector))
    new_cube = replace_region(cube, cstart, cend, cube_region(cube, nonnegative_qb))
    constants = ["FRONT_INDEX_GROUPS", "FRONT_INDEX_VECTORS", "FRONT_MAIN_GROUPS", "FRONT_MAIN_VECTORS"]
    new_cube = add_imports(
        new_cube, {"vector_prefix_stages": constants, "projection_overlap_stages": ["_query_hadamard_four_tokens"]}
    )
    new_vector = add_imports(
        new_vector,
        {
            "vector_prefix_stages": constants
            + ["front_indexer_compressor_vector_stage", "front_main_compressor_vector_stage"],
            "runtime_abi": [
                "PHASE_IDX_POOL",
                "PHASE_IDX_COMMIT",
                "PHASE_IDX_NORM",
                "PHASE_MAIN_POOL",
                "PHASE_MAIN_COMMIT",
            ],
        },
    )
    return {
        "cube_schedule.py": new_cube,
        "vector_schedule.py": new_vector,
        "vector_prefix_stages.py": forked_helpers((reference / "vector_prefix_stages.py").read_text(), groups),
    }


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-groups", type=int, choices=(4, 8, 12), default=8)
    parser.add_argument("--nonnegative-qb", action="store_true")
    args = parser.parse_args()
    reference = here / "runs/frozen_v122"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reference.glob("*.py")} == expected
    revised = build(reference, args.index_groups, args.nonnegative_qb)
    name = f"front_partition_i{args.index_groups}"
    if args.nonnegative_qb:
        name += "_v2"
    target = here / "runs" / name
    shutil.copytree(reference, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for file, text in revised.items():
        ast.parse(text)
        (target / file).write_text(text)
    report = {
        "source": str(target),
        "reference_kind": "frozen_tnd_v122",
        "adopted": False,
        "device_execution": False,
        "index_groups": args.index_groups,
        "nonnegative_qb": args.nonnegative_qb,
        "guard": GUARD,
        "change": "single MIX, disjoint front roles after shared QA/QR; original downstream Score/Attention/output",
        "preserved": [
            "all math helpers",
            "all Tensor ABI and workspace layouts",
            "48 logical scale page writers",
            "per-core generation",
            "empty-phase publication",
            "shared-state pool-before-commit",
            "Score FIFO",
        ],
        "changed_files": list(revised),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    }
    with (here / f"{name}.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
