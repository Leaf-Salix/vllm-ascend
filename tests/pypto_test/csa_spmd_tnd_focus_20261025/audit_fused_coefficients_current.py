# SPDX-License-Identifier: Apache-2.0
"""执行实际helper的CPU数值/写者模型；不声称模拟Ascend指令或设备精度。"""

import ast
import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "frozen/pypto/deepseek_v4_flash_csa_spmd_tnd"
CANDIDATE = HERE / "candidates/fused_coefficients_current_v2"


class PL:
    FP32, FP16, INT32, INT64, INT8, INDEX = np.float32, np.float16, np.int32, np.int64, np.int8, int
    range = staticmethod(range)
    unroll = staticmethod(range)
    min = staticmethod(min)
    max = staticmethod(max)
    mul = staticmethod(np.multiply)
    maximum = staticmethod(np.maximum)
    neg = staticmethod(np.negative)
    div = staticmethod(np.divide)
    recip = staticmethod(np.reciprocal)
    reshape = staticmethod(np.reshape)
    row_max = staticmethod(lambda a: np.max(a, axis=1, keepdims=True))
    row_expand_mul = staticmethod(np.multiply)
    full = staticmethod(lambda shape, dtype, value: np.full(shape, value, dtype=dtype))
    dim = staticmethod(lambda a, axis: a.shape[axis])
    read = staticmethod(lambda a, at: a[tuple(at)])
    owner = -1
    writes = []

    @staticmethod
    def cast(value, target_type, mode=None):
        if target_type in (np.int8, np.int32, np.int64, int) and mode == "rint":
            value = np.rint(value)
        a = np.asarray(value).astype(target_type)
        return a.item() if a.ndim == 0 else a

    @staticmethod
    def write(a, at, value):
        a[tuple(at)] = value

    @staticmethod
    def load(a, at, shape):
        return a[tuple(slice(start, start + count) for start, count in zip(at, shape))].copy()

    @staticmethod
    def store(value, at, target):
        target[tuple(slice(start, start + count) for start, count in zip(at, value.shape))] = value
        return target

    @staticmethod
    def assemble(target, value, at):
        PL.writes.append((PL.owner, tuple(at), value.shape))
        return PL.store(value, at, target)


PL.tensor = PL
PL.tile = PL


def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node = copy.deepcopy(node)
            node.decorator_list = []
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            exec(
                compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"),
                namespace,
            )


def namespace(fused_path):
    ns = dict(
        pl=PL,
        HEADS=64,
        IDX_N_HEADS=64,
        IDX_HEAD_DIM=128,
        QH_QUANT_TILE=64,
        QH_HEAD_DIM_TILE=64,
        HADAMARD_SCALE=128**-0.5,
        INT8_AMAX_EPS=1e-4,
        INT8_SCALE_MAX=127.0,
        COMPRESS_RATIO=4,
        COEFFICIENT_ROWS=16,
        GROUP_DESCRIPTOR_ALIGNMENT=16,
        GROUP_DESCRIPTOR_WORDS=4,
        MAX_CANDIDATES=262144,
        CANDIDATES_PER_LEAF=8192,
        SCORE_TILE=1024,
        MAX_LEAVES=32,
        SCORE_WORKERS=24,
        SCALE_PLAN_ALIGNMENT=16,
        GROUP_WORDS_PER_TOKEN=4,
        SCALE_JOB_WORDS=4,
    )
    functions(BASE / "request_geometry.py", {"request_bounds"}, ns)
    functions(
        BASE / "indexer_group_stages.py",
        {
            "indexer_long_leaf_plan",
            "indexer_group_plan_stage",
            "indexer_decode_group",
            "indexer_query_visibility",
            "indexer_head_coefficients_stage",
        },
        ns,
    )
    functions(BASE / "indexer_query_stages.py", {"indexer_query_quant_stage"}, ns)
    functions(BASE / "indexer_scale_plan_stages.py", {"indexer_scale_plan_base"}, ns)
    functions(BASE / "indexer_scale_layout_stages.py", {"indexer_scale_layout_base"}, ns)
    functions(fused_path, {"prepare_fused_query_coefficients", "indexer_query_quant_coeff_stage"}, ns)
    return ns


def one_case(ns, name, tokens, bounds, lengths, positions, seed):
    b = len(lengths)
    bounds, lengths, positions = (
        np.asarray(bounds, np.int32),
        np.asarray(lengths, np.int32),
        np.asarray(positions, np.int64),
    )
    old_words = ns["indexer_scale_layout_base"](b, tokens, 48) + 2 * b
    offsets = np.full(old_words + tokens, -91, np.int32)
    plan = np.full(8, -91, np.int32)
    ns["indexer_group_plan_stage"](bounds, lengths, positions, offsets, plan, tokens, 0, 6, True, 1)
    groups = int(plan[0])
    assert groups <= tokens
    rng = np.random.default_rng(seed)
    had = rng.standard_normal((tokens * 64, 128), dtype=np.float32)
    had[0:64] = 0  # amax epsilon及全零量化
    weights = rng.standard_normal((tokens, 64), dtype=np.float32)
    weights[0] = 0
    q0, q1 = [np.full((tokens * 64, 128), -99, np.int8) for _ in range(2)]
    s0, s1 = [np.full((tokens * 64, 1), -99, np.float32) for _ in range(2)]
    c0, c1 = [np.full((tokens * 16, 384), np.float16(13)) for _ in range(2)]
    prefix_before = offsets[:old_words].copy()
    ns["prepare_fused_query_coefficients"](bounds, lengths, positions, offsets, c1, tokens, 6)
    assert np.array_equal(prefix_before, offsets[:old_words])
    assert np.all(c1[: groups * 16] == 0) and np.all(c1[groups * 16 :] == 13)
    PL.writes = []
    for worker in range(48):
        ns["indexer_query_quant_stage"](had, q0, s0, tokens, worker, 48)
    for worker in range(48):
        ns["indexer_head_coefficients_stage"](
            s0.reshape(tokens, 64), weights, positions, bounds, lengths, offsets, c0, worker, 48, 6
        )
    PL.writes = []
    for worker in range(47, -1, -1):  # 不依赖token owner的执行次序
        PL.owner = worker
        ns["indexer_query_quant_coeff_stage"](
            had, q1, s1, tokens, worker, 48, weights, bounds, lengths, positions, offsets, c1, 6
        )
    assert np.array_equal(q0, q1)
    assert np.array_equal(s0.view(np.uint32), s1.view(np.uint32))
    assert np.array_equal(c0.view(np.uint16), c1.view(np.uint16)), name
    seen = set()
    for owner, (row, col), shape in PL.writes:
        assert shape == (1, 64)
        group, lane = row // 16, row % 16
        _, first, live, _ = ns["indexer_decode_group"](bounds, lengths, positions, offsets, group, 6)
        assert lane < live and col == lane * 64 and owner == (first + lane) % 48
        assert (row, col) not in seen
        seen.add((row, col))
    return dict(
        name=name,
        tokens=tokens,
        requests=b,
        groups=groups,
        overlapping_tokens=int(np.count_nonzero(offsets[old_words:] == -2)),
        diagonal_writes=len(seen),
        exact_cpu_quant_scale_coeff=True,
        old_offset_prefix_untouched=True,
        inactive_coefficient_capacity_untouched=True,
    )


def static_contracts():
    source = (CANDIDATE / "vector_schedule.py").read_text()
    tree = ast.parse(source)
    meta = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "worker == planner")
    calls = [n for n in ast.walk(meta) if isinstance(n, ast.Call)]
    prep = next(n for n in calls if ast.unparse(n.func) == "prepare_fused_query_coefficients")
    publish_meta = next(
        n for n in calls if ast.unparse(n.func) == "publish_phase" and ast.unparse(n.args[1]) == "PHASE_META"
    )
    assert prep.lineno < publish_meta.lineno
    publish_weights = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "publish_phase"
        and ast.unparse(n.args[1]) == "PHASE_WEIGHTS"
    )
    fused_if = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "FUSED_INDEXER_COEFF"
        and any(
            isinstance(x, ast.Call) and ast.unparse(x.func) == "indexer_query_quant_coeff_stage" for x in ast.walk(n)
        )
    )
    assert publish_weights.lineno < fused_if.lineno
    prefix = (CANDIDATE / "vector_prefix_stages.py").read_text()
    assert "signal = wait_producer(signal, PHASE_META, GROUPS + planner, CORES, generation)" in prefix
    cube = (CANDIDATE / "cube_schedule.py").read_text()
    assert "wait_row_owners(signal, PHASE_IDX_QUERY_QUANT, CORES, generation, 0, tokens, 1, VECTORS, GROUPS, 1)" in cube
    assert "if not FUSED_INDEXER_COEFF:\n        groups =" in cube
    # 执行实际wait_row_owners，确保任意T的获取集包含所有token%48生产者。
    acquired = []
    wait_ns = dict(
        pl=PL,
        SIGNAL_SLOT_WORDS=128,
        csa_signal_wait=lambda signal, offset, generation: acquired.append(offset // 128) or signal,
    )
    PL.system = SimpleNamespace(fence=lambda: None, cacheinvalid=lambda: None)
    functions(BASE / "phase_sync.py", {"wait_row_owners"}, wait_ns)
    for tokens in range(1, 257):
        acquired.clear()
        wait_ns["wait_row_owners"](None, 0, 72, 1, 0, tokens, 1, 48, 24, 1)
        assert set(acquired) == {24 + token % 48 for token in range(tokens)}
    select = runpy.run_path(str(CANDIDATE / "policy.py"))["select_policy"]
    for tokens in (1, 6, 32, 63, 64, 96, 128, 144, 192, 193, 256):
        policy = select(tokens=tokens, requests=(tokens + 5) // 6, max_query_tokens=6, max_seq_len=131078)
        assert policy.fused_indexer_coeff == (64 <= tokens <= 192)
    for filename in ("binding.py", "graph_contract.py", "indexer_query_stages.py", "indexer_group_stages.py"):
        assert (BASE / filename).read_bytes() == (CANDIDATE / filename).read_bytes()
    old_quant = next(
        n
        for n in ast.parse((BASE / "indexer_query_stages.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "indexer_query_quant_stage"
    )
    new_quant = next(
        n
        for n in ast.parse((CANDIDATE / "fused_coefficients.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "indexer_query_quant_coeff_stage"
    )
    old_loop = next(n for n in old_quant.body if isinstance(n, ast.For))
    new_loop = next(n for n in new_quant.body if isinstance(n, ast.For))
    assert ast.dump(old_loop.iter) == ast.dump(new_loop.iter)
    assert [ast.dump(n) for n in old_loop.body] == [ast.dump(n) for n in new_loop.body[: len(old_loop.body)]]
    ns = namespace(CANDIDATE / "fused_coefficients.py")
    entry = ast.parse((CANDIDATE / "entry.py").read_text())
    checked_allocations = 0
    for fn in entry.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in ("decode_csa_spmd", "decode_csa_spmd_debug"):
            body = [
                copy.deepcopy(n)
                for n in fn.body
                if (
                    isinstance(n, ast.Assign)
                    and any(isinstance(x, ast.Name) and x.id == "group_offset_words" for x in n.targets)
                )
                or (isinstance(n, ast.If) and ast.unparse(n.test) == "FUSED_INDEXER_COEFF")
            ]
            model = ast.parse("def allocation(b, t, FUSED_INDEXER_COEFF):\n    pass\n").body[0]
            model.body = body + [ast.Return(value=ast.Name(id="group_offset_words", ctx=ast.Load()))]
            alloc_ns = {}
            exec(
                compile(ast.fix_missing_locations(ast.Module(body=[model], type_ignores=[])), "allocation", "exec"),
                alloc_ns,
            )
            for b in range(1, 65):
                for t in range(1, 257):
                    old_end = ns["indexer_scale_layout_base"](b, t, 48) + 2 * b
                    assert alloc_ns["allocation"](b, t, False) == old_end
                    assert alloc_ns["allocation"](b, t, True) - t == old_end
            checked_allocations += 1
    assert checked_allocations == 2
    return dict(
        meta_before_quant=True,
        weights_wait_acyclic=True,
        quant_acquire_tokens_1_to_256=True,
        quant_arithmetic_ast_identical=True,
        binding_and_original_math_helpers_unchanged=True,
        internal_workspace_both_entries_32768_capacity_checks=True,
    )


def main():
    ns = namespace(CANDIDATE / "fused_coefficients.py")
    cases = []
    for tokens in (64, 96, 144, 192):
        lengths_q = [6] * (tokens // 6) + ([tokens % 6] if tokens % 6 else [])
        bounds = np.cumsum([0] + lengths_q).tolist()
        lengths = [131072 + q for q in lengths_q]
        positions = [131072 + i for q in lengths_q for i in range(q)]
        cases.append(one_case(ns, f"uniform_T{tokens}", tokens, bounds, lengths, positions, 1024))
    for name, bounds, lengths, positions in (
        (
            "mixed_inactive_padding",
            [0, 6, 6, 9, 16, 20],
            [131078, 0, 8195, 131079, 0],
            list(range(131072, 131078)) + list(range(8192, 8195)) + list(range(131072, 131079)) + [-1] * 48,
        ),
        ("overlap_same_causal", [0, 6, 2, 8], [106, 0, 108], list(range(100, 108)) + [-1] * 56),
        ("overlap_distinct_causal", [0, 6, 2, 8], [106, 0, 208], list(range(100, 106)) + [206, 207] + [-1] * 56),
        ("bad_positions", [0, 6, 12], [106, 106], [100, 101, -1, 103, 104, 105] + [100] * 6 + [-1] * 52),
        ("all_inactive", [0, 64], [0], [-1] * 64),
        ("invalid_bounds", [-1, 6, 70], [106, 164], [-1] * 64),
    ):
        cases.append(one_case(ns, name, 64, bounds, lengths, positions, 99))
    # 负对照：证明旧单值映射确实不能复刻前级仍允许的重复区间语义。
    old_ns = namespace(HERE / "candidates/fused_coefficients_current_v1/fused_coefficients.py")
    rejected = False
    try:
        one_case(
            old_ns,
            "v1_overlap_negative_control",
            64,
            [0, 6, 2, 8],
            [106, 0, 108],
            list(range(100, 108)) + [-1] * 56,
            99,
        )
    except AssertionError:
        rejected = True
    assert rejected
    report = dict(
        passed=True,
        candidate=str(CANDIDATE),
        cases=cases,
        contracts=static_contracts(),
        old_single_map_negative_control_rejected=rejected,
        device_execution=False,
        limitation="CPU执行真实helper数学/地址；不是CCE或NPU精度证明，也不证明设备重叠和性能",
    )
    (HERE / "fused_coefficients_current_v2_contracts.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(dict(passed=True, scenarios=len(cases), negative_control_rejected=rejected)))


if __name__ == "__main__":
    main()
