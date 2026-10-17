# SPDX-License-Identifier: Apache-2.0
"""动态 C4 pool/state 的有界 CPU 同源控制流与独立窗口 oracle。

被测侧只删除正式源码的 import、DSL 装饰器和类型注解，执行原 Python AST；
load/store/read 用带完整边界检查的 NumPy shim。所有在线 softmax 运算显式
保持 FP32。独立 oracle 按八个逻辑位置直接构造 value/score，再以 FP64 稳定
softmax 求结果，不复用生产者的 mi/li/oi 循环；因此仅允许 CPU 求和树误差。
另从当前 canonical 两个 compressor 提取六条在线更新，逐节点核对 AST 顺序。

输入覆盖短/odd/mixed query、H=0/1/7、跨页、dummy、causal 不一致与非法
页/offset；inactive、Q 未覆盖区及容量尾部保留 poison 是正确合同。先执行
所有 pool 再 commit；单独的错误交错反例只证明这个阶段依赖不可删。
本脚本不导入 PyPTO、不编译、不执行 NPU，不证明硬件浮点、流水或缓存一致性。
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PACKAGE = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
CANONICAL = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa"
SOURCE = PACKAGE / "compressor_pool_stages.py"
POISON = np.float32(-931.125)
GUARD = np.float32(-1987.25)
ATOL, RTOL = 3e-6, 3e-6


class PlainPython(ast.NodeTransformer):
    """只处理 DSL 导入/类型/装饰器，不改任何被测条件、循环、索引或运算。"""

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_FunctionDef(self, node):
        node.decorator_list = []
        node.returns = None
        for argument in node.args.args:
            argument.annotation = None
        return self.generic_visit(node)


class Interpreter:
    def __init__(self):
        self.events, self.names, self.worker = [], {}, -1
        self.language = SimpleNamespace(
            dynamic=lambda name: name, cast=lambda value, *args, **kw: int(value),
            INDEX="INDEX", INT32="INT32", FP32="FP32", range=range,
            read=self.read, load=self.load, store=self.store,
            add=np.add, sub=np.subtract, mul=np.multiply, div=np.divide,
            maximum=np.maximum, exp=np.exp,
            tensor=SimpleNamespace(dim=lambda value, axis: value.shape[axis], read=self.read),
            tile=SimpleNamespace(full=lambda shape, dtype, value: np.full(shape, value, dtype=np.float32)),
        )
        config = ast.parse((CANONICAL / "config.py").read_text())
        constants = {node.targets[0].id: ast.literal_eval(node.value) for node in config.body
                     if isinstance(node, ast.Assign) and len(node.targets) == 1
                     and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in ("C4A_COMPRESSOR_BLOCK_SIZE", "FP32_NEG_INF")}
        assert constants["C4A_COMPRESSOR_BLOCK_SIZE"] == 2
        self.namespace = {"pl": self.language, **constants}
        for path in (PACKAGE / "request_geometry.py", SOURCE):
            tree = ast.fix_missing_locations(PlainPython().visit(ast.parse(path.read_text())))
            exec(compile(tree, str(path), "exec"), self.namespace)

    def bind(self, fixture):
        self.events.clear()
        self.names = {id(value): name for name, value in fixture.items() if isinstance(value, np.ndarray)}

    def region(self, tensor, origin, shape, operation):
        assert len(origin) == len(shape) == tensor.ndim
        assert all(0 <= start and 0 <= size and start + size <= extent
                   for start, size, extent in zip(origin, shape, tensor.shape)), (operation, origin, shape, tensor.shape)
        self.events.append((operation, self.names.get(id(tensor), "temporary"), tuple(origin), tuple(shape), self.worker))
        return tuple(slice(start, start + size) for start, size in zip(origin, shape))

    def read(self, tensor, indices):
        slices = self.region(tensor, indices, [1] * tensor.ndim, "read")
        return tensor[slices].reshape(()).item()

    def load(self, tensor, origin, shape):
        return tensor[self.region(tensor, origin, shape, "load")].copy()

    def store(self, value, origin, tensor):
        tensor[self.region(tensor, origin, value.shape, "store")] = value

    def run(self, fixture, head, workers, phase, order=None):
        prefix = "indexer_" if head == 128 else ""
        suffix = "pool_stage" if phase == "pool" else "state_commit_stage"
        function = self.namespace[prefix + "compressor_" + suffix]
        for self.worker in (order if order is not None else reversed(range(workers))):
            if phase == "pool":
                function(fixture["kv"], fixture["score"], fixture["state"], fixture["table"], fixture["ape"],
                         fixture["positions"], fixture["bounds"], fixture["lengths"], fixture["pooled"], self.worker, workers)
            else:
                function(fixture["kv"], fixture["score"], fixture["ape"], fixture["positions"], fixture["bounds"],
                         fixture["lengths"], fixture["slots"], fixture["state"], self.worker, workers)


def requests(fixture):
    """独立定义 Q/L 合法区间；不调用正式 request_bounds 或其解释后的版本。"""
    q, lengths, tokens = fixture["bounds"], fixture["lengths"], len(fixture["positions"])
    for request, length in enumerate(lengths.tolist()):
        if request + 1 >= len(q) or length <= 0:
            continue
        begin, end = int(q[request]), int(q[request + 1])
        if 0 <= begin < end <= tokens and end - begin <= length:
            yield request, begin, end, length - (end - begin)


def make_fixture(spec, head):
    widths, histories = spec["widths"], spec["histories"]
    leading, trailing = spec.get("leading", 0), spec.get("trailing", 0)
    boundaries = np.array([leading, *[leading + value for value in np.cumsum(widths).tolist()]], np.int32)
    tokens = int(boundaries[-1]) + trailing
    lengths = np.array([width + history if history >= 0 else 0 for width, history in zip(widths, histories)], np.int32)
    columns = max([1, *((max(0, history) + width + 1) // 2 for width, history in zip(widths, histories))])
    pages = len(widths) * columns + 3
    outdim, elements = 2 * head, 8 * head
    rng = np.random.default_rng(8127 + head + tokens)
    # 每页增加前后保护区。中间原始 state 未使用位置保留 poison，历史有效槽单独填入。
    guarded_state = np.full((pages + 2, elements), GUARD, np.float32)
    state = guarded_state[1:-1]
    state.fill(POISON)
    guarded_pool = np.full((tokens + 5 + 2, head), GUARD, np.float32)
    pooled = guarded_pool[1:-1]
    pooled.fill(POISON)
    table = np.arange(len(widths) * columns, dtype=np.int32).reshape(len(widths), columns)
    if table.size:
        table = (len(widths) * columns - 1 - table).copy()
    ape = rng.normal(0, 0.2, (4, outdim)).astype(np.float32)
    positions = np.full(tokens, 2**31 + 3, np.int64)
    slots = np.zeros((tokens, 2), np.int32)
    kv = np.full((tokens + 4, outdim), POISON, np.float32)
    score = np.full_like(kv, POISON)
    kv[:tokens] = rng.normal(0, 0.6, (tokens, outdim))
    score[:tokens] = rng.normal(0, 0.9, (tokens, outdim))
    if spec.get("shared_history"):
        # 请求0本轮写pos7的槽，同时是请求1的旧历史pos7：需要所有pool先完成。
        table[1, :4] = table[0, :4]
    for request, (width, history) in enumerate(zip(widths, histories)):
        if history < 0:
            continue
        for position in range(history):
            page = int(table[request, position // 2])
            column = position % 2 * 2 * outdim
            state[page, column:column + outdim] = rng.normal(0, 0.6, outdim)
            # 历史 score 已经包含 APE；oracle 后面不能给它再次加 APE。
            state[page, column + outdim:column + 2 * outdim] = (
                rng.normal(0, 0.9, outdim).astype(np.float32) + ape[position % 4])
        for local in range(width):
            token, position = int(boundaries[request]) + local, history + local
            positions[token] = position
            slots[token] = (table[request, position // 2], position % 2)
    fixture = dict(kv=kv, score=score, state=state, table=table, ape=ape, positions=positions,
                   bounds=boundaries, lengths=lengths, slots=slots, pooled=pooled,
                   guarded_state=guarded_state, guarded_pool=guarded_pool)
    mutation = spec.get("mutation")
    if mutation == "negative_page":
        fixture["table"].fill(-1)
        fixture["slots"][:, 0] = -1
    elif mutation == "page_overflow":
        fixture["table"].fill(pages)
        fixture["slots"][:, 0] = pages
    elif mutation == "negative_offset":
        fixture["slots"][:, 1] = -1
    elif mutation == "offset_overflow":
        fixture["slots"][:, 1] = 2
    elif mutation == "empty_table":
        fixture["table"] = table[:, :0].copy()
    elif mutation == "short_page":
        fixture["state"] = state[:, :elements - 32]
    elif mutation == "causal_mismatch":
        # 仍保留 C4 关闭余数，确保是 causal equality 拒绝，不是 modulo 恰巧拒绝。
        fixture["positions"][::2] += 4
    elif mutation == "length_too_short":
        fixture["lengths"].fill(1)
    elif mutation == "negative_bound":
        fixture["bounds"][0] = -1
    elif mutation == "bound_overflow":
        fixture["bounds"][-1] = tokens + 1
    elif mutation == "truncated_bounds":
        fixture["bounds"] = fixture["bounds"][:1].copy()
    return fixture


def independent_oracle(fixture, head):
    """直接枚举完整八位置矩阵，与生产代码的在线更新循环独立。"""
    state, table, positions = fixture["state"], fixture["table"], fixture["positions"]
    expected_pool, expected_state = fixture["pooled"].copy(), state.copy()
    pool_writes = np.zeros(expected_pool.shape, bool)
    state_writes = np.zeros(expected_state.shape, bool)
    output_dim, tile_width = 2 * head, 512 if head == 512 else 64
    closed = []
    for request, begin, end, history in requests(fixture):
        for token in range(begin, end):
            expected_pool[token] = 0
            pool_writes[token] = True
            position = int(positions[token])
            causal = position == history + token - begin
            if causal and position % 4 == 3:
                values = np.zeros((8, head), np.float64)
                scores = np.full((8, head), np.finfo(np.float32).min, np.float64)
                for slot, logical_position in enumerate(range(position - 7, position + 1)):
                    half = 0 if slot < 4 else head
                    if logical_position < 0:
                        continue
                    if logical_position < history:
                        scores[slot] = 0
                        logical_page = logical_position // 2
                        if logical_page >= table.shape[1]:
                            continue
                        page = int(table[request, logical_page])
                        if not 0 <= page < state.shape[0]:
                            continue
                        # 非法截断页只验证按现有tile级容量guard安全回退，不声称是有效输入。
                        for h0 in range(0, head, tile_width):
                            column = logical_position % 2 * 2 * output_dim + half + h0
                            if column + output_dim + tile_width <= state.shape[1]:
                                values[slot, h0:h0 + tile_width] = state[page, column:column + tile_width]
                                scores[slot, h0:h0 + tile_width] = state[page, column + output_dim:column + output_dim + tile_width]
                    else:
                        overlay = begin + logical_position - history
                        assert begin <= overlay <= token < end
                        values[slot] = fixture["kv"][overlay, half:half + head]
                        # 输入两项先按实际FP32加法形成score，再用FP64参考softmax。
                        scores[slot] = (fixture["score"][overlay, half:half + head]
                                        + fixture["ape"][logical_position % 4, half:half + head]).astype(np.float32)
                probability = np.exp(scores - scores.max(axis=0, keepdims=True))
                expected_pool[token] = ((values * probability).sum(axis=0) / probability.sum(axis=0)).astype(np.float32)
                closed.append(token)
            if causal:
                page, offset = map(int, fixture["slots"][token])
                column = offset * 2 * output_dim
                if 0 <= page < state.shape[0] and 0 <= offset < 2 and column + 2 * output_dim <= state.shape[1]:
                    assert not state_writes[page, column:column + 2 * output_dim].any(), "fixture违反有效写槽唯一合同"
                    expected_state[page, column:column + output_dim] = fixture["kv"][token]
                    expected_state[page, column + output_dim:column + 2 * output_dim] = (
                        fixture["score"][token] + fixture["ape"][position % 4])
                    state_writes[page, column:column + 2 * output_dim] = True
    return expected_pool, expected_state, pool_writes, state_writes, closed


def assignment_name(node):
    return node.targets[0].id if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else None


def numerical_ast_audit():
    """在线更新六条语句必须与两份 canonical AST 完全一致，不能只看算式名称。"""
    production = ast.parse(SOURCE.read_text())
    reports = []
    for old_file, old_function, new_function in (
        ("decode_compressor_ratio4.py", "compressor_ratio4_pool_projected", "compressor_pool_stage"),
        ("decode_indexer_compressor.py", "indexer_compressor_pool_projected", "indexer_compressor_pool_stage"),
    ):
        original = next(node for node in ast.parse((CANONICAL / old_file).read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == old_function)
        candidate = next(node for node in production.body if isinstance(node, ast.FunctionDef) and node.name == new_function)
        old_loop = next(node for node in ast.walk(original) if isinstance(node, ast.For)
                        and isinstance(node.target, ast.Name) and node.target.id == "state_idx")
        new_loop = next(node for node in ast.walk(candidate) if isinstance(node, ast.For)
                        and isinstance(node.target, ast.Name) and node.target.id == "state_index")
        initial_orders = []
        for function, loop in ((original, old_loop), (candidate, new_loop)):
            parent = next(node for node in ast.walk(function)
                          if hasattr(node, "body") and isinstance(node.body, list) and loop in node.body)
            initial = [assignment_name(node) for node in parent.body[:parent.body.index(loop)]
                       if assignment_name(node) in ("mi", "li", "oi")]
            assert initial == ["mi", "li", "oi"]
            initial_orders.append(initial)
        names = ["mi_next", "alpha", "beta", "li", "oi", "mi"]
        old_updates = [node for node in old_loop.body if assignment_name(node) in names]
        new_updates = [node for node in new_loop.body if assignment_name(node) in names]
        assert [assignment_name(node) for node in old_updates] == names
        assert [assignment_name(node) for node in new_updates] == names
        assert [ast.dump(node, include_attributes=False) for node in old_updates] == [ast.dump(node, include_attributes=False) for node in new_updates]
        old_li = next(node for node in ast.walk(original) if assignment_name(node) == "li")
        new_li = next(node for node in ast.walk(candidate) if assignment_name(node) == "li")
        assert ast.dump(old_li, include_attributes=False) == ast.dump(new_li, include_attributes=False)
        assert ast.dump(old_loop.iter, include_attributes=False) == ast.dump(new_loop.iter, include_attributes=False)
        old_div = [node for node in ast.walk(original) if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.div"
                   and [ast.unparse(arg) for arg in node.args] == ["oi", "li"]]
        new_div = [node for node in ast.walk(candidate) if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.div"
                   and [ast.unparse(arg) for arg in node.args] == ["oi", "li"]]
        assert len(old_div) == len(new_div) == 1
        reports.append({"canonical_file": str(CANONICAL / old_file), "canonical_function": old_function,
                        "new_function": new_function, "online_update_ast_exact": True,
                        "update_order": names, "updates": [ast.unparse(node) for node in new_updates],
                        "initialization_order": initial_orders[0], "initialization_order_ast_exact": True,
                        "initial_li_ast_exact": True, "seven_position_loop_ast_exact": True,
                        "final_div_oi_li_preserved": True,
                        "initial_current_second_half": "两源码读取当前token后半区，score加本位置APE，li=exp(mi-mi)，oi为value；地址由独立窗口oracle覆盖"})
    return reports


def specs():
    base = dict(widths=[3, 5, 1], histories=[1, 7, 0], leading=1, trailing=2)
    result = [dict(name="short_h0_no_close", widths=[1], histories=[0]),
              dict(name="short_h1_close", widths=[3], histories=[1]),
              dict(name="odd_h7_cross_pages", widths=[9], histories=[7]),
              dict(name="mixed_odd_and_uncovered", widths=[1, 3, 5, 2], histories=[0, 1, 7, 15], leading=2, trailing=3),
              dict(name="uniform_s6", widths=[6, 6], histories=[7, 1]),
              dict(name="dummy_stale_positions", widths=[3, 5, 2], histories=[1, -1, 7], trailing=3),
              dict(name="all_dummy", widths=[1, 5], histories=[-1, -1]),
              dict(name="t0_b0", widths=[], histories=[]),
              dict(name="t0_b1_zero_query", widths=[0], histories=[0]),
              dict(name="no_closing_groups", widths=[1, 2], histories=[0, 0]),
              dict(name="shared_history_requires_phase", widths=[1, 4], histories=[7, 8], shared_history=True)]
    for mutation in ("negative_page", "page_overflow", "negative_offset", "offset_overflow", "empty_table",
                     "short_page", "causal_mismatch", "length_too_short", "negative_bound", "bound_overflow", "truncated_bounds"):
        result.append(dict(base, name=mutation, mutation=mutation))
    return result


def check_case(interpreter, spec, head, workers):
    fixture = make_fixture(spec, head)
    initial = {name: value.copy() for name, value in fixture.items() if isinstance(value, np.ndarray)}
    expected_pool, expected_state, pool_mask, state_mask, closed = independent_oracle(fixture, head)
    interpreter.bind(fixture)
    interpreter.run(fixture, head, workers, "pool")
    pool_events = list(interpreter.events)
    active_tokens = {token for _, begin, end, _ in requests(fixture) for token in range(begin, end)}
    owners = {token: request % workers for request, begin, end, _ in requests(fixture) for token in range(begin, end)}
    for kind, name, origin, shape, worker in pool_events:
        if name in ("positions", "kv", "score"):
            assert origin[0] in active_tokens, (spec["name"], "inactive payload read", name, origin)
        if kind == "store" and name == "pooled":
            assert origin[0] in active_tokens and worker == owners[origin[0]]
    assert np.all(np.isfinite(fixture["pooled"][pool_mask]))
    assert np.allclose(fixture["pooled"][pool_mask], expected_pool[pool_mask], atol=ATOL, rtol=RTOL)
    assert np.array_equal(fixture["pooled"][~pool_mask], initial["pooled"][~pool_mask])
    assert np.array_equal(fixture["state"], initial["state"]), "pool不能写历史state"
    interpreter.run(fixture, head, workers, "commit")
    assert np.array_equal(fixture["state"], expected_state), (spec["name"], head, "state values")
    assert np.array_equal(fixture["state"][~state_mask], initial["state"][~state_mask])
    assert np.all(fixture["guarded_state"][[0, -1]] == GUARD)
    assert np.all(fixture["guarded_pool"][[0, -1]] == GUARD)
    if spec.get("mutation") == "short_page":
        assert np.array_equal(fixture["guarded_state"][1:-1, -32:], initial["guarded_state"][1:-1, -32:])
    for name in ("kv", "score", "table", "ape", "positions", "bounds", "lengths", "slots"):
        assert np.array_equal(fixture[name], initial[name]), name
    for kind, name, origin, _, _ in interpreter.events:
        if name in ("positions", "kv", "score", "slots"):
            assert origin[0] in active_tokens, (spec["name"], "inactive payload access", name, origin)
    error = np.abs(fixture["pooled"][pool_mask].astype(np.float64) - expected_pool[pool_mask].astype(np.float64))
    return {"case": spec["name"], "head": head, "workers": workers, "tokens": len(fixture["positions"]),
            "query_widths": spec["widths"], "histories": spec["histories"], "closed_tokens": closed,
            "max_abs": float(error.max()) if error.size else 0.0, "memory_accesses": len(interpreter.events),
            "active_rows_initialized": len(active_tokens), "inactive_and_tail_poison_preserved": True,
            "state_unwritten_preserved": True, "guards_passed": True, "input_unchanged": True, "passed": True}


def ordering_counterexample(interpreter, head):
    spec = dict(name="shared_history_requires_phase", widths=[1, 4], histories=[7, 8], shared_history=True)
    fixture = make_fixture(spec, head)
    expected, _, mask, _, _ = independent_oracle(fixture, head)
    interpreter.bind(fixture)
    # 这是故意错误的阶段交错，不是建议实现：请求0先commit，污染请求1尚未读的历史。
    interpreter.run(fixture, head, 2, "pool", [0])
    interpreter.run(fixture, head, 2, "commit", [0])
    interpreter.run(fixture, head, 2, "pool", [1])
    difference = float(np.max(np.abs(fixture["pooled"][mask] - expected[mask])))
    assert difference > 1e-3
    return {"head": head, "incorrect_local_pool_commit_order_max_abs": difference,
            "global_pool_before_commit_dependency_required": True,
            "note": "仅CPU反例；正式代码未改、正常oracle全部pool先于commit"}


def main():
    interpreter = Interpreter()
    ast_audit = numerical_ast_audit()
    cases = [check_case(interpreter, spec, head, workers)
             for spec in specs() for head in (512, 128) for workers in (1, 3, 48)]
    counterexamples = [ordering_counterexample(interpreter, head) for head in (512, 128)]
    report = {"passed": True, "source": str(SOURCE), "source_ast_interpreted": True,
              "canonical_online_ast": ast_audit, "cases": cases, "case_count": len(cases),
              "case_shapes": len(specs()), "oracle": "独立八位置FP64稳定softmax，输入score保持FP32加法边界",
              "atol": ATOL, "rtol": RTOL, "max_abs": max(case["max_abs"] for case in cases),
              "state_commit_exact": True, "ordering_counterexamples": counterexamples,
              "device_execution": False, "new_compilation": False, "hardware_precision_proved": False,
              "limitations": ["未模拟A3浮点指令、DMA完成或Scalar cache一致性",
                              "非法metadata用例只核对越界安全与当前回退语义，不宣称有效模型数学",
                              "Q全局单调/活动区间不重叠、有效写槽唯一、投影与pooled容量至少T由调用者保证",
                              "causal检查拒绝当前不一致行；不宣称把此前不一致位置传播为整个请求失活",
                              "inactive pooled poison必须由后续消费者的active/闭合判定隔离，不需要本阶段全T清零"]}
    (HERE / "oracle.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    review = {"passed": True, "source": str(SOURCE), "blocking_defects_found": [],
              "address_review": ["主state每逻辑位置2048 FP32、每页4096；Indexer每位置512、每页1024",
                                 "历史地址按logical_position//2查本请求页表，页内offset乘2*OUT_DIM；score再偏移OUT_DIM",
                                 "本轮overlay按Q/H映射回同请求TND，不依赖固定S，历史score不重复加APE",
                                 "pool只写active请求行；未覆盖区保留poison是正确合同",
                                 "commit在所有pool完成后才可覆盖state；共享历史反例证明确有跨请求读后写依赖"],
              "numerical_review": ast_audit,
              "readability_review": {"module_overview": True, "layout_and_dynamic_axes": True,
                                     "role_and_ownership": True, "workspace_and_sync": True,
                                     "reading_order": True, "online_order_and_tail": True,
                                     "nonblocking_clarifications": ["文件头可再显式写明kv_proj/score_proj行容量均须>=T，当前仅pooled的>=T最明确",
                                                                     "当前causal判断针对被处理行；此前位置异常是否使整请求失活不属于本helper合同"]},
              "evidence": "oracle.json", "source_modified": False, "dependencies_modified": False,
              "new_compilation": False, "device_execution": False,
              "not_covered": report["limitations"]}
    (HERE / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(cases), "max_abs": report["max_abs"],
                      "online_ast_exact": True, "state_exact": True, "device_execution": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
