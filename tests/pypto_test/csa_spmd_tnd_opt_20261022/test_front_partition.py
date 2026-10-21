# SPDX-License-Identifier: Apache-2.0
"""Execute actual front schedules with math stubs; audit physical dependencies.

Checks phase ownership, absence of schedule cycles, pool-reader lifetimes, and
all 48 original scale-page owners. Device accuracy and performance are separate.
"""

import ast
from collections import Counter, defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import pytest
from prepare_front_partition import GUARD, build, cube_region, vector_region
from test_mixed_candidates import plain_function

REFERENCE = Path(__file__).resolve().parent / "runs/frozen_v122"
CUBE_PHASES = {
    "PHASE_IDX_PROJECT",
    "PHASE_IDX_QUERY_PROJECT",
    "PHASE_WEIGHTS_PROJECT",
    "PHASE_IDX_HADAMARD",
    "PHASE_IDX_QUERY_HAD",
    "PHASE_MAIN_PROJECT",
    "PHASE_QB",
    "PHASE_KV_PROJECT",
}
VECTOR_PHASES = {
    "PHASE_IDX_POOL",
    "PHASE_IDX_COMMIT",
    "PHASE_IDX_NORM",
    "PHASE_IDX_QUERY_ROPE",
    "PHASE_WEIGHTS",
    "PHASE_IDX_QUERY_QUANT",
    "PHASE_COEFF",
    "PHASE_IDX_QUANT",
    "PHASE_IDX_SCALE",
    "PHASE_MAIN_POOL",
    "PHASE_MAIN_COMMIT",
    "PHASE_MAIN_OUTPUT",
    "PHASE_QUERY",
}
MATH_ARITY = {
    "indexer_compressor_project_stage": 2,
    "indexer_query_project_stage": 1,
    "indexer_weights_project_stage": 1,
    "indexer_compressor_hadamard_stage": 1,
    "_query_hadamard_four_tokens": 1,
    "compressor_project_stage": 2,
    "q_b_stage": 1,
    "kv_project_stage": 1,
    "indexer_compressor_pool_stage": 1,
    "indexer_compressor_state_commit_stage": 1,
    "indexer_compressor_norm_rope_stage": 1,
    "compressor_pool_stage": 1,
    "compressor_state_commit_stage": 1,
    "compressor_norm_rope_cache_stage": 3,
    "indexer_query_dequant_rope_stage": 1,
    "indexer_weights_reduce_stage": 1,
    "indexer_query_quant_stage": 2,
    "indexer_head_coefficients_stage": 1,
    "indexer_compact_quant_cache_stage": 3,
    "indexer_scale_page_commit_stage": 1,
    "q_dequant_rope_stage": 1,
}


def simulate(groups, tokens, requests):
    cube_text = (REFERENCE / "cube_schedule.py").read_text()
    vector_text = (REFERENCE / "vector_schedule.py").read_text()
    revised = build(REFERENCE, groups, nonnegative_qb=True)
    cube, vector = cube_region(cube_text, nonnegative_qb=True), vector_region(vector_text)
    signatures = {}
    for path in REFERENCE.glob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in MATH_ARITY:
                signatures[node.name] = [a.arg for a in node.args.args]
    query = ast.parse((REFERENCE / "query_projection_stages.py").read_text())
    signature = next(n for n in query.body if isinstance(n, ast.FunctionDef) and n.name == "_q_b_nz_stage")
    signatures["q_b_stage"] = [a.arg for a in signature.args.args]
    assert signatures.keys() == MATH_ARITY.keys()
    events, by_core, pubs, owners = [], defaultdict(list), {}, defaultdict(list)
    current_core = 0

    def event(kind, detail):
        index = len(events)
        events.append((current_core, kind, detail))
        by_core[current_core].append(index)
        return index

    def publish(signal, phase, core, cores, generation):
        assert core == current_core and cores == 72 and generation == 7
        assert (phase, core) not in pubs
        pubs[phase, core] = event("publish", (phase, core))
        return signal

    def wait(signal, phase, cores, generation, first, count, tile, workers, base, stride):
        assert workers > 0 and tile > 0
        units = range(first // tile, (first + count + tile - 1) // tile)
        expected = {(phase, base + (unit % workers) * stride) for unit in units}
        event("wait", expected)
        return signal

    def wait_one(signal, phase, producer, cores, generation):
        event("wait", {(phase, producer)})
        return signal

    def math(name):
        def run(*args):
            values = dict(zip(signatures[name], args))
            worker = values.get("worker", values.get("worker_id"))
            workers = values["WORKERS"]
            assert worker is not None and 0 <= worker < workers, (name, worker, workers)
            owners[name].append((worker, workers, current_core))
            event("math", name)
            return None if MATH_ARITY[name] == 1 else (None,) * MATH_ARITY[name]

        return run

    for core in range(72):
        current_core = core
        text = cube if core < 24 else vector
        all_text = text + revised["vector_prefix_stages.py"]
        ns = {n.id: n.id for n in ast.walk(ast.parse(all_text)) if isinstance(n, ast.Name)}
        ns.update(
            pl=SimpleNamespace(
                tensor=SimpleNamespace(
                    dim=lambda tensor, axis: requests if tensor in ("idx_lengths", "main_lengths") else tokens
                ),
                cast=lambda value, dtype: value,
                INDEX=None,
                min=min,
                max=max,
                range=range,
                read=lambda tensor, index: (tokens + 3) // 4,
            ),
            GROUPS=24,
            VECTORS=48,
            CORES=72,
            FRONT_INDEX_GROUPS=groups,
            FRONT_INDEX_VECTORS=2 * groups,
            FRONT_MAIN_GROUPS=24 - groups,
            FRONT_MAIN_VECTORS=2 * (24 - groups),
            COOPERATIVE_PREFIX=True,
            QUERY_GROUP=6,
            QB_WIDE_N=tokens < 64,
            tokens=tokens,
            idx_requests=requests,
            main_requests=requests,
            planner=min((tokens + 7) // 8, 47),
            worker=core if core < 24 else core - 24,
            core=core,
            generation=7,
            signal=None,
            publish_phase=publish,
            wait_row_owners=wait,
            wait_producer=wait_one,
        )
        ns.update({name: math(name) for name in MATH_ARITY})
        for name in ("front_indexer_compressor_vector_stage", "front_main_compressor_vector_stage"):
            plain_function(revised["vector_prefix_stages.py"], name, ns)
        exec(compile(text, "actual_front_schedule", "exec"), ns)

    # Every original front phase is published exactly once by all its roles.
    for phase in CUBE_PHASES:
        assert {core for p, core in pubs if p == phase} == set(range(24)), phase
    for phase in VECTOR_PHASES:
        assert {core for p, core in pubs if p == phase} == set(range(24, 72)), phase

    # Resolve waits to actual publications and combine with per-core program order.
    after, degree = defaultdict(set), [0] * len(events)

    def edge(before, following):
        if following not in after[before]:
            after[before].add(following)
            degree[following] += 1

    for seq in by_core.values():
        for before, following in zip(seq, seq[1:]):
            edge(before, following)
    for index, (_, kind, dependencies) in enumerate(events):
        if kind == "wait":
            for phase, core in dependencies:
                if phase not in ("PHASE_META", "PHASE_QR"):
                    assert (phase, core) in pubs, (phase, core)
                    edge(pubs[phase, core], index)
    ready = deque(index for index, d in enumerate(degree) if d == 0)
    complete = 0
    while ready:
        before = ready.popleft()
        complete += 1
        for following in after[before]:
            degree[following] -= 1
            if degree[following] == 0:
                ready.append(following)
    assert complete == len(events), "The physical front dependency graph contains a wait cycle"

    # Request-level state writers must acquire every real reader of shared prefix.
    for label, count, first, helper in (
        ("PHASE_IDX_POOL", 2 * groups, 24, "indexer_compressor_state_commit_stage"),
        ("PHASE_MAIN_POOL", 2 * (24 - groups), 24 + 2 * groups, "compressor_state_commit_stage"),
    ):
        readers = {(label, first + i) for i in range(min(requests, count))}
        for worker in range(min(requests, count)):
            seen = set()
            for index in by_core[first + worker]:
                _, kind, detail = events[index]
                if kind == "wait":
                    seen.update(detail)
                if kind == "math" and detail == helper:
                    assert readers <= seen, (label, worker, readers - seen)

    # Main consumers explicitly acquire metadata after losing the Indexer wrapper.
    for core in range(24 + 2 * groups, 72):
        seen = set()
        for index in by_core[core]:
            _, kind, detail = events[index]
            if kind == "wait":
                seen.update(detail)
            if kind == "math" and detail == "compressor_pool_stage":
                assert ("PHASE_META", 24 + min((tokens + 7) // 8, 47)) in seen

    for name, assigned in owners.items():
        counts = {w for _, w, _ in assigned}
        assert len(counts) == 1
        total = counts.pop()
        assert Counter(w for w, _, _ in assigned) == Counter(range(total)), name
    scale = owners["indexer_scale_page_commit_stage"]
    assert {w for w, _, _ in scale} == set(range(48))
    assert all(total == 48 and core == 24 + owner % (2 * groups) for owner, total, core in scale)


@pytest.mark.parametrize("groups", (4, 8, 12))
@pytest.mark.parametrize("tokens,requests", ((24, 4), (41, 12), (56, 16), (96, 16), (168, 28), (192, 32), (385, 65)))
def test_actual_schedule_owners_lifetimes_and_dependency_graph(groups, tokens, requests):
    simulate(groups, tokens, requests)


def test_original_math_helpers_and_tensor_abi_are_preserved():
    revised = build(REFERENCE, 8)
    for name in ("cube_schedule", "vector_schedule"):
        old = next(
            n
            for n in ast.parse((REFERENCE / f"{name}.py").read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        new = next(
            n for n in ast.parse(revised[f"{name}.py"]).body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        assert ast.dump(old.args, include_attributes=False) == ast.dump(new.args, include_attributes=False)
    source = (REFERENCE / "vector_prefix_stages.py").read_text()
    assert revised["vector_prefix_stages.py"].startswith(source)


@pytest.mark.parametrize("groups", (4, 8, 12))
@pytest.mark.parametrize("requests", (1, 16, 28, 65))
def test_attention_wait_uses_actual_main_writers_not_empty_indexer_publications(groups, requests):
    revised = build(REFERENCE, groups, nonnegative_qb=True)
    node = next(
        n
        for n in ast.walk(ast.parse(revised["vector_schedule.py"]))
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == GUARD
        and any(
            isinstance(v, ast.Call)
            and ast.unparse(v.func) == "wait_row_owners"
            and ast.unparse(v.args[1]) == "PHASE_MAIN_OUTPUT"
            for v in ast.walk(n)
        )
    )
    seen = set()

    def wait(signal, phase, cores, generation, first, count, tile, workers, base, stride):
        seen.update(base + unit % workers * stride for unit in range(first, first + count))

    ns = dict(
        COOPERATIVE_PREFIX=True,
        QUERY_GROUP=6,
        signal=None,
        PHASE_MAIN_OUTPUT=None,
        CORES=72,
        generation=7,
        main_requests=requests,
        FRONT_MAIN_VECTORS=2 * (24 - groups),
        FRONT_INDEX_VECTORS=2 * groups,
        GROUPS=24,
        wait_row_owners=wait,
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "actual_attention_wait", "exec"),
        ns,
    )
    first = 24 + 2 * groups
    assert seen == set(range(first, first + min(requests, 2 * (24 - groups))))


def test_closed_front_branch_restores_complete_original_schedule_ast():
    class CloseFront(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.unparse(node.test) == GUARD:
                return node.orelse
            return self.generic_visit(node)

    revised = build(REFERENCE, 8, nonnegative_qb=True)
    for name in ("cube_schedule", "vector_schedule"):
        original = next(
            n
            for n in ast.parse((REFERENCE / f"{name}.py").read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        candidate = next(
            n for n in ast.parse(revised[f"{name}.py"]).body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        assert ast.dump(CloseFront().visit(candidate), include_attributes=False) == ast.dump(
            original, include_attributes=False
        )
