# SPDX-License-Identifier: Apache-2.0
"""Check actual-source event projections, math and fallback preservation."""

import ast
import copy
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
BASE = HERE / "runs/merged_schedule_v1/attention_stream_stages.py"
SOURCE = HERE / "runs/attention_overlap_v1/attention_stream_stages.py"


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def project(body, needed):
    """Backward-slice scalar control needed by actual sync_set/sync_wait calls."""
    result = []
    for node in reversed(body):
        keep = None
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            name = ast.unparse(call.func)
            if name in ("pl.system.sync_set", "pl.system.sync_wait"):
                needed.update(names(call.args[0]))
                needed.add("qk_tick")
                keep = ast.parse(
                    f"trace.append(('{name.rsplit('_', 1)[1]}', {ast.unparse(call.args[0])}, qk_tick))"
                ).body[0]
        elif isinstance(node, ast.If):
            yes_needed, no_needed = set(needed), set(needed)
            yes, no = project(node.body, yes_needed), project(node.orelse, no_needed)
            if yes or no:
                needed.update(yes_needed | no_needed | names(node.test))
                keep = ast.If(test=copy.deepcopy(node.test), body=yes or [ast.Pass()], orelse=no)
        elif isinstance(node, ast.For):
            inner_needed = set(needed)
            inner = project(node.body, inner_needed)
            if inner:
                iterator = copy.deepcopy(node.iter)
                assert ast.unparse(iterator.func) in ("pl.range", "pl.unroll")
                iterator.func = ast.Name(id="range", ctx=ast.Load())
                iterator.keywords = []
                target = copy.deepcopy(node.target)
                if isinstance(target, ast.Tuple):
                    assert not (names(target.elts[1]) & inner_needed), "Unexpected carried event state"
                    target = target.elts[0]
                needed.update(inner_needed | names(iterator))
                keep = ast.For(target=target, iter=iterator, body=inner, orelse=[])
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(names(target) & needed for target in targets):
                needed.update(names(node.value))
                keep = ast.Assign(targets=copy.deepcopy(targets), value=copy.deepcopy(node.value))
        if keep is not None:
            result.append(keep)
    return list(reversed(result))


def compiled_projection(path, name):
    original = function(path, name)
    body = project(original.body, set())
    projected = copy.deepcopy(original)
    projected.decorator_list = []
    projected.returns = None
    projected.args.defaults = []
    for arg in projected.args.args:
        arg.annotation = None
    projected.body = ast.parse("trace = []").body + body + ast.parse("return trace").body
    module = ast.fix_missing_locations(ast.Module(body=[projected], type_ignores=[]))
    constants = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            constants.update({target.id: node.value.value for target in node.targets if isinstance(target, ast.Name)})
    namespace = dict(
        constants,
        pl=SimpleNamespace(
            cast=lambda value, dtype: int(value),
            INT32=int,
            read=lambda array, index: array[index[0]][index[1]],
            tensor=SimpleNamespace(dim=lambda shape, axis: shape[axis]),
        ),
        physical_plan_supported=lambda *args: 1,
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name], [arg.arg for arg in projected.args.args]


def event_trace(compiled, tokens, mask, lane=0):
    invoke, params = compiled
    kwargs = {name: None for name in params}
    kwargs.update(
        valid_block_mask=mask,
        position_ids=(tokens,),
        q_flat=(tokens * 64, 512),
        cmp_block_table=(tokens, 64),
        cmp_kv_flat=(8192, 512),
        core_id=0,
        CORE_GROUPS=24,
        EVENT_BASE=2,
    )
    if "lane_id" in params:
        kwargs.update(lane_id=lane, PLAN_PHYSICAL_ROWS=True)
    return invoke(**{name: kwargs[name] for name in params})


def assert_event_dag(traces):
    """Program order plus paired producer/consumer edges, not a pipe simulator."""
    outgoing = defaultdict(set)
    incoming = defaultdict(int)
    occurrences = defaultdict(list)
    nodes = []

    def edge(src, dst):
        if dst not in outgoing[src]:
            outgoing[src].add(dst)
            incoming[dst] += 1

    for actor, trace in enumerate(traces):
        for index, (kind, event, tick) in enumerate(trace):
            node = (actor, index)
            nodes.append(node)
            if index:
                edge((actor, index - 1), node)
            offset = (0 if event in (2, 3) else 2) if actor == 0 else {2: 0, 3: 1, 4: 1, 5: 3}[event]
            occurrences[(kind, event, tick - offset)].append(node)
    for (kind, event, work), consumers in occurrences.items():
        if kind != "wait":
            continue
        producers = occurrences[("set", event, work)]
        assert len(producers) == (2 if event in (2, 4) else 1)
        assert len(consumers) == (1 if event in (2, 4) else 2)
        for producer in producers:
            for consumer in consumers:
                edge(producer, consumer)
    assert {key[1:] for key in occurrences if key[0] == "set"} == {key[1:] for key in occurrences if key[0] == "wait"}
    queue = deque(node for node in nodes if incoming[node] == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for successor in outgoing[node]:
            incoming[successor] -= 1
            if incoming[successor] == 0:
                queue.append(successor)
    assert visited == len(nodes), "Event dependency cycle"


@pytest.mark.parametrize("tokens", [96, 144])
def test_source_event_sequences_and_liveness_over_all_two_query_masks(tokens):
    cube = compiled_projection(BASE, "attention_qk_pv_cube_stage")
    old = compiled_projection(BASE, "attention_qk_pv_vector_lane_stage")
    new = compiled_projection(SOURCE, "attention_qk_pv_vector_overlap_stage")
    for pattern in range(1 << 10):
        mask = [[0] * 16 for _ in range(tokens)]
        for work in range(10):
            mask[(work // 5) * 24][work % 5] = (pattern >> work) & 1
        traces = [event_trace(cube, tokens, mask)]
        for lane in (0, 1):
            before = event_trace(old, tokens, mask, lane)
            after = event_trace(new, tokens, mask, lane)
            assert before == after
            traces.append(after)
        assert_event_dag(traces)


def test_event_verifier_rejects_missing_lane_notification():
    with pytest.raises(AssertionError):
        assert_event_dag([[("wait", 2, 0)], [("set", 2, 0)], []])


def test_original_function_and_cube_remain_identical():
    for name in ("attention_qk_pv_vector_lane_stage", "attention_qk_pv_cube_stage"):
        assert ast.dump(function(BASE, name)) == ast.dump(function(SOURCE, name))


def test_fast_softmax_and_fallback_are_original_math():
    original = function(BASE, "attention_qk_pv_vector_lane_stage")
    candidate = function(SOURCE, "attention_qk_pv_vector_overlap_stage")
    outer_old = next(n for n in original.body if isinstance(n, ast.For))
    outer_new = next(n for n in candidate.body if isinstance(n, ast.For))
    branch = next(n for n in outer_new.body if isinstance(n, ast.If) and ast.unparse(n.test) == "overlap_ready != 0")
    original_begin = next(i for i, n in enumerate(outer_old.body) if isinstance(n, ast.If))
    original_end = next(
        i
        for i, n in enumerate(outer_old.body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "qk_tick >= QK_PRE_LAUNCH + 1"
    )
    assert [ast.dump(n) for n in branch.orelse[:-1]] == [
        ast.dump(n) for n in outer_old.body[original_begin:original_end]
    ]
    assignments = {}
    for node in ast.walk(original):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = ast.dump(node.value)
    for node in branch.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("qk_scores_half", "qk_bias", "qk_masked", "qk_mi", "qk_exp", "qk_li", "qk_probability"):
                assert ast.dump(node.value) == assignments[name]


class ExecuteCarriedLoops(ast.NodeTransformer):
    """Give pl.range/pl.yield_ their sequential CPU dataflow for the reader."""

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node.func) == "pl.yield_":
            return node.args[0] if len(node.args) == 1 else ast.Tuple(elts=node.args, ctx=ast.Load())
        return node

    def visit_For(self, node):
        assert ast.unparse(node.iter.func) == "pl.range"
        init = next((kw.value for kw in node.iter.keywords if kw.arg == "init_values"), None)
        node.iter.func = ast.Name(id="range", ctx=ast.Load())
        node.iter.keywords = []
        if init is None:
            return self.generic_visit(node)
        assert isinstance(node.target, ast.Tuple) and isinstance(init, ast.Tuple)
        index, carry = node.target.elts
        terminal = node.body[-1]
        assert isinstance(terminal, ast.Assign) and ast.unparse(terminal.value.func) == "pl.yield_"
        results = copy.deepcopy(terminal.targets[0])
        result_load = ast.parse(ast.unparse(results), mode="eval").body
        initial = init.elts[0] if len(init.elts) == 1 else init
        carry_target = carry.elts[0] if len(carry.elts) == 1 else carry
        node.target = index
        node.body.insert(0, ast.Assign(targets=[carry_target], value=result_load))
        node = self.generic_visit(node)
        return [ast.Assign(targets=[results], value=initial), node]


@pytest.mark.parametrize("bad_row", [None, 0, 31, 32, 63])
@pytest.mark.parametrize("bad_value", [-1, 256, 2**31 - 1])
def test_actual_two_half_reader_holes_and_reuse(bad_row, bad_value):
    fn = function(SOURCE, "attention_qk_pv_vector_overlap_stage")
    outer = next(n for n in fn.body if isinstance(n, ast.For))
    branch = next(n for n in outer.body if isinstance(n, ast.If) and ast.unparse(n.test) == "overlap_ready != 0")
    reader = copy.deepcopy(next(n for n in branch.body if isinstance(n, ast.For)))
    module = ast.Module(body=[reader], type_ignores=[])
    code = compile(ast.fix_missing_locations(ExecuteCarriedLoops().visit(module)), "actual_two_half_reader", "exec")
    rng = np.random.default_rng(20261023)
    cache = rng.integers(0, 65536, (256, 512), dtype=np.uint16)

    def gather(dst, src, dst_offset, src_offset, shape):
        dr, dc = dst_offset
        sr, sc = src_offset
        nr, nc = shape
        assert 0 <= sr < len(src)
        dst[dr : dr + nr, dc : dc + nc] = src[sr : sr + nr, sc : sc + nc]
        return dst

    def assemble(dst, src, offset):
        return gather(dst, src, offset, [0, 0], list(src.shape))

    def store(tile, offset, dst):
        gather(dst, tile, offset, [0, 0], list(tile.shape))

    pl = SimpleNamespace(
        cast=lambda value, dtype: int(value),
        INT32=int,
        INDEX=int,
        read=lambda array, offset: int(array[tuple(offset)]),
        tensor=SimpleNamespace(dim=lambda array, axis: array.shape[axis]),
        gather_row=gather,
        store=store,
        tile=SimpleNamespace(assemble=assemble),
    )
    for lane in (0, 1):
        state = rng.integers(0, 65536, (32, 512), dtype=np.uint16)
        valid_indices = rng.integers(0, len(cache), 64, dtype=np.int32)
        with_hole = valid_indices.copy()
        if bad_row is not None:
            with_hole[bad_row] = bad_value
        for indices in (valid_indices, with_hole, np.full(64, -1, np.int32), valid_indices):
            output = np.full((160, 512), 0xA55A, np.uint16)
            expected = output.copy()
            rows = np.zeros((64, 512), np.uint16)
            valid = (indices >= 0) & (indices < len(cache))
            rows[valid] = cache[indices[valid]]
            first = 8 + lane * 64
            expected[first : first + 64] = rows
            namespace = dict(
                pl=pl,
                dma_kv_iter=state,
                dma_zero_row=np.zeros((1, 512), np.uint16),
                KV_DMA_ROWS=32,
                HEAD_DIM=512,
                physical_rows=indices[None, :],
                cmp_kv_flat=cache,
                qk_kv_row=8,
                qk_lane_kv=lane * 64,
                kv_transfer=output,
            )
            exec(code, namespace)
            state = namespace["dma_finished"]
            np.testing.assert_array_equal(output, expected)
