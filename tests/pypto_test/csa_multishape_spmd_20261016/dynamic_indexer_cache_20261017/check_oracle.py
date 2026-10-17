# SPDX-License-Identifier: Apache-2.0
"""Indexer 紧凑 cache 的 CPU 地址/掩码、量化 DAG 与串行 scale RMW 审阅。

被测侧解释当前正式 Python AST，仅删除 DSL import/装饰器/类型；load/store
检查真实容量，valid_shape 限制物理读写区域，BF16 用 FP32 位模式模拟 RNE。
参考侧独立逐请求枚举关闭 token，逐行以 frexp 构造 BF16 舍入，再计算 FP32
量化与预期 INT8/FP16 字节。该数值模型不代表 A3 div/recip 或 cast 的设备误差。

metadata 的 count/offset 直接执行当前同源 helper，并与独立枚举结果核对；
不把请求起点强行补齐到16。cache期望只改合法key的128B及对应scale的2B，
被测scale阶段必须在worker0串行保留其余64B邻槽。额外CPU反例说明若两个
写者各自读旧64B再覆盖，slot不同也会丢失更新。仅写本目录三个审阅文件。
"""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PACKAGE = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
CANONICAL = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa"
SOURCE = PACKAGE / "indexer_cache_stages.py"
KEY_BYTES, PAGE_TOKENS, HEAD = 4096, 32, 128
POISON = np.float32(-731.25)
BYTE_GUARD = np.int8(-59)


class PlainPython(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_FunctionDef(self, node):
        node.decorator_list, node.returns = [], None
        for argument in node.args.args:
            argument.annotation = None
        return self.generic_visit(node)


class ValidTile:
    def __init__(self, array, valid):
        self.array, self.valid = array, tuple(map(int, valid))


def data(value):
    return value.array if isinstance(value, ValidTile) else value


def bf16_bits(value):
    array = np.asarray(value, dtype=np.float32)
    bits = array.copy().view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def bf16_frexp(value):
    """参考侧用尾数/指数舍入，避免复用被测侧位运算实现。用例仅含有限普通值。"""
    array = np.asarray(value, np.float32)
    fraction, exponent = np.frexp(np.abs(array).astype(np.float64))
    rounded = np.ldexp(np.rint(fraction * 256.0) / 256.0, exponent)
    return np.copysign(rounded, array).astype(np.float32)


class Interpreter:
    def __init__(self):
        self.events, self.names, self.worker = [], {}, -1
        unary = lambda fn: lambda value, *args: fn(data(value))
        binary = lambda fn: lambda left, right: fn(data(left), data(right))
        self.pl = SimpleNamespace(
            dynamic=lambda name: name, range=range, min=min, INDEX="INDEX", INT32="INT32", INT8="INT8",
            FP32="FP32", FP16="FP16", BF16="BF16", cast=self.cast, read=self.read, write=self.write,
            load=self.load, store=self.store, slice=self.slice, create_tile=lambda shape, **kw: np.empty(shape, np.float32),
            set_validshape=lambda value, *valid: ValidTile(data(value), valid),
            reshape=lambda value, shape: data(value).reshape(shape), mul=binary(np.multiply),
            div=binary(np.divide), maximum=binary(np.maximum), abs=unary(np.abs),
            recip=lambda value: np.divide(np.float32(1.0), data(value)),
            row_max=lambda value, *args: np.max(data(value), axis=-1, keepdims=True),
            row_expand_mul=binary(np.multiply),
            tensor=SimpleNamespace(dim=lambda value, axis: value.shape[axis], read=self.read),
        )
        self.pl.tile = SimpleNamespace(
            full=lambda shape, dtype, value: np.full(shape, value, np.float32), load=self.load,
            store=self.store, write=self.tile_write, reinterpret_view=self.reinterpret,
        )
        config = ast.parse((CANONICAL / "config.py").read_text())
        constants = {node.targets[0].id: ast.literal_eval(node.value) for node in config.body
                     if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in ("INT8_AMAX_EPS", "INT8_SCALE_MAX")}
        self.constants = constants
        self.namespace = {"pl": self.pl, **constants}
        for path in (PACKAGE / "request_geometry.py", PACKAGE / "metadata_stages.py", SOURCE):
            tree = ast.fix_missing_locations(PlainPython().visit(ast.parse(path.read_text())))
            exec(compile(tree, str(path), "exec"), self.namespace)

    def bind(self, fixture):
        self.events.clear()
        self.names = {id(value): name for name, value in fixture.items() if isinstance(value, np.ndarray)}

    def region(self, tensor, origin, shape, kind):
        origin, shape = tuple(map(int, origin)), tuple(map(int, shape))
        assert len(origin) == len(shape) == tensor.ndim
        assert all(0 <= start and 0 <= width and start + width <= extent
                   for start, width, extent in zip(origin, shape, tensor.shape)), (kind, origin, shape, tensor.shape)
        self.events.append((kind, self.names.get(id(tensor), "temporary"), origin, shape, self.worker))
        return tuple(slice(start, start + width) for start, width in zip(origin, shape))

    def cast(self, value, target_type=None, mode=None, *args):
        value = data(value)
        if target_type == "INDEX":
            return int(value)
        if target_type == "BF16":
            return bf16_bits(value)
        if target_type == "FP32":
            return np.asarray(value, np.float32)
        if target_type == "FP16":
            return np.asarray(value, np.float16)
        if target_type in ("INT32", "INT8"):
            converted = np.rint(value) if mode == "rint" else np.trunc(value)
            return np.asarray(converted, np.int32 if target_type == "INT32" else np.int8)
        raise ValueError(target_type)

    def read(self, tensor, indices):
        return tensor[self.region(tensor, indices, [1] * tensor.ndim, "read")].reshape(()).item()

    def write(self, tensor, indices, value):
        tensor[self.region(tensor, indices, [1] * tensor.ndim, "write")] = value

    def load(self, tensor, origin, shape, valid_shape=None, **kwargs):
        valid = list(shape if valid_shape is None else valid_shape)
        region = self.region(tensor, origin, valid, "load")
        tile = np.zeros(shape, tensor.dtype)
        tile[tuple(slice(0, size) for size in valid)] = tensor[region]
        return tile

    def slice(self, tensor, shape, origin, valid_shape=None, **kwargs):
        array = data(tensor)
        valid = list(shape if valid_shape is None else valid_shape)
        tile = np.zeros(shape, array.dtype)
        assert all(0 <= start and start + width <= extent for start, width, extent in zip(origin, valid, array.shape))
        tile[tuple(slice(0, size) for size in valid)] = array[tuple(slice(start, start + width) for start, width in zip(origin, valid))]
        return ValidTile(tile, valid)

    def store(self, value, origin, tensor):
        array = data(value)
        valid = value.valid if isinstance(value, ValidTile) else array.shape
        tensor[self.region(tensor, origin, valid, "store")] = array[tuple(slice(0, size) for size in valid)]

    def tile_write(self, tile, indices, value):
        self.events.append(("tile_write", "scale_half", tuple(indices), (1, 1), self.worker))
        tile[tuple(map(int, indices))] = value

    def reinterpret(self, tile, dtype):
        return data(tile).view(np.float16 if dtype == "FP16" else np.int8)

    def metadata(self, fixture):
        self.worker = 0
        self.namespace["compact_offsets_and_count_stage"](
            fixture["bounds"], fixture["lengths"], fixture["offsets"], fixture["count"], len(fixture["positions"]), 0)

    def quant(self, fixture, workers):
        for self.worker in reversed(range(workers)):
            self.namespace["indexer_compact_quant_cache_stage"](
                fixture["values"], fixture["positions"], fixture["bounds"], fixture["lengths"], fixture["offsets"],
                fixture["slots"], fixture["scale_values"], fixture["debug"], fixture["cache"], self.worker, workers)

    def commit(self, fixture, worker):
        self.worker = worker
        self.namespace["indexer_compact_scale_commit_stage"](
            fixture["scale_values"], fixture["positions"], fixture["bounds"], fixture["lengths"], fixture["offsets"],
            fixture["slots"], fixture["cache"], worker)


def requests(fixture):
    q, lengths, tokens = fixture["bounds"], fixture["lengths"], len(fixture["positions"])
    for request, length in enumerate(lengths.tolist()):
        if length <= 0 or request + 1 >= len(q):
            continue
        begin, end = map(int, q[request:request + 2])
        if 0 <= begin < end <= tokens and end - begin <= length:
            yield request, begin, end, length - (end - begin)


def enumerate_compact(fixture):
    offsets = np.zeros_like(fixture["lengths"])
    rows = []
    request_rows = {}
    for request, begin, end, history in requests(fixture):
        # 独立逐位置数出历史关闭数，不复用生产 floor 差公式。
        historical_groups = len(range(3, history, 4))
        offsets[request] = len(rows) - historical_groups - 1
        current = [(request, token, history + token - begin) for token in range(begin, end)
                   if (history + token - begin) % 4 == 3]
        request_rows[request] = (len(rows), current)
        rows.extend(current)
    return offsets, rows, request_rows


def fixture_for(spec):
    widths, histories = spec["widths"], spec["histories"]
    leading, trailing = spec.get("leading", 0), spec.get("trailing", 0)
    bounds = np.array([leading, *[leading + x for x in np.cumsum(widths).tolist()]], np.int32)
    tokens = int(bounds[-1]) + trailing
    lengths = np.array([width + history if history >= 0 else 0 for width, history in zip(widths, histories)], np.int32)
    positions = np.full(tokens, 2**31 + 3, np.int64)
    for request, (width, history) in enumerate(zip(widths, histories)):
        if history >= 0:
            positions[bounds[request]:bounds[request + 1]] = np.arange(history, history + width)
    fixture = dict(bounds=bounds, lengths=lengths, positions=positions,
                   offsets=np.full(len(widths), -999, np.int32), count=np.array([-999], np.int32))
    _, closed, _ = enumerate_compact(fixture)
    count = len(closed)
    capacity = 0 if spec.get("zero_capacity") else count + 5
    rng = np.random.default_rng(20261017 + tokens + count)
    values = np.full((capacity, HEAD), POISON, np.float32)
    if count and capacity:
        values[:count] = rng.normal(0, 1.7, (count, HEAD))
        # 含BF16正负ties和零行，区分先BF16舍入与先Hadamard缩放的数值边界。
        values[0, :6] = [1.00390625, 1.01171875, -1.00390625, -1.01171875, 0.0, -0.0]
        if count > 1:
            values[1] = 0
    scales = np.full((capacity + (2 if capacity else 0), 1), POISON, np.float32)
    slots = np.full((capacity, 2), -1, np.int32)
    pages, page_bytes = max(1, (count + 31) // 32) + 1, 4160 + 32
    for row in range(min(count, capacity)):
        slots[row] = (row // 32, (row * 7 + 3) % 32)
    guarded = np.full((pages + 2, page_bytes), BYTE_GUARD, np.int8)
    cache = guarded[1:-1]
    cache[:] = ((np.arange(cache.size).reshape(cache.shape) * 37 + 19) % 256).astype(np.uint8).view(np.int8)
    for page in range(pages):
        old_scales = (0.125 + np.arange(32) / 256 + page).astype(np.float16)
        cache[page, KEY_BYTES:KEY_BYTES + 64] = old_scales.view(np.int8)
    guarded_debug = np.full((tokens + 2, HEAD), POISON, np.float32)
    fixture.update(values=values, scale_values=scales, slots=slots, cache=cache, guarded_cache=guarded,
                   debug=guarded_debug[1:-1], guarded_debug=guarded_debug)
    mutation = spec.get("mutation")
    if mutation == "causal_stale":
        for _, token, _ in closed[::2]:
            positions[token] += 4
    elif mutation == "negative_slot_offset":
        slots[:, 1] = -1
    elif mutation == "large_slot_offset":
        slots[:, 1] = 32
    elif mutation == "negative_page":
        slots[:, 0] = -1
    elif mutation == "large_page":
        slots[:, 0] = pages
    elif mutation == "short_page":
        fixture["cache"] = cache[:, :4159]
    return fixture


def quantize_independent(value, constants):
    rounded = bf16_frexp(value)
    scaled = np.multiply(rounded, np.float32(128**-0.5), dtype=np.float32)
    maximum = np.float32(max(float(np.max(np.abs(scaled))), constants["INT8_AMAX_EPS"]))
    multiplier = np.float32(constants["INT8_SCALE_MAX"]) / maximum
    scale = np.float32(1.0) / multiplier
    quantized_input = np.multiply(scaled, multiplier, dtype=np.float32)
    # CPU独立整数ties-to-even；这个范围内后续FP16转换可精确表示整数。
    quantized = np.array([round(float(number)) for number in quantized_input], np.int8)
    return scaled, quantized, scale


def expected(fixture, constants):
    original_cache = fixture["cache"].copy()
    key_cache, final_cache = original_cache.copy(), original_cache.copy()
    debug, scales = fixture["debug"].copy(), fixture["scale_values"].copy()
    key_mask, final_mask = np.zeros_like(original_cache, bool), np.zeros_like(original_cache, bool)
    debug_mask, scale_mask = np.zeros_like(debug, bool), np.zeros_like(scales, bool)
    _, _, groups = enumerate_compact(fixture)
    committed = []
    for request, _, _, history in requests(fixture):
        _, current = groups[request]
        # 当前源码允许negative offset，只检查它组合出的prefix；坏prefix单独作为非法metadata用例。
        prefix = int(fixture["offsets"][request]) + len(range(3, history, 4)) + 1
        count = len(current)
        if not (0 <= prefix and prefix + count <= len(fixture["values"])
                and prefix + count <= len(scales) and prefix + count <= len(fixture["slots"])):
            continue
        for local, (_, token, logical_position) in enumerate(current):
            compact = prefix + local
            value, key, scale = quantize_independent(fixture["values"][compact], constants)
            scales[compact, 0], scale_mask[compact, 0] = scale, True
            if int(fixture["positions"][token]) != logical_position:
                continue
            page, offset = map(int, fixture["slots"][compact])
            if not (0 <= page < len(original_cache) and 0 <= offset < 32 and original_cache.shape[1] >= 4160):
                continue
            start = offset * 128
            assert not key_mask[page, start:start + 128].any(), "有效写slot必须唯一"
            key_cache[page, start:start + 128] = key
            final_cache[page, start:start + 128] = key
            key_mask[page, start:start + 128] = True
            final_mask[page, start:start + 128] = True
            debug[token], debug_mask[token] = value, True
            scale_bytes = np.asarray([scale], np.float16).view(np.int8)
            final_cache[page, KEY_BYTES + offset * 2:KEY_BYTES + offset * 2 + 2] = scale_bytes
            final_mask[page, KEY_BYTES + offset * 2:KEY_BYTES + offset * 2 + 2] = True
            committed.append((page, offset, compact))
    return dict(key_cache=key_cache, final_cache=final_cache, debug=debug, scales=scales,
                key_mask=key_mask, final_mask=final_mask, debug_mask=debug_mask, scale_mask=scale_mask,
                committed=committed)


def specs():
    result = [dict(name="mixed_odd_non16_prefix", widths=[4, 1, 69, 3], histories=[0, 0, 7, 1], leading=2, trailing=3),
              dict(name="same_page_different_requests", widths=[4, 4], histories=[0, 0]),
              dict(name="negative_legal_compact_offset", widths=[1, 5], histories=[7, 1]),
              dict(name="uniform_s6", widths=[6, 6], histories=[7, 0]),
              dict(name="odd_short", widths=[1, 3, 5], histories=[0, 1, 7]),
              dict(name="zero_close", widths=[1, 2], histories=[0, 0]),
              dict(name="zero_capacity_no_close", widths=[1, 2], histories=[0, 0], zero_capacity=True),
              dict(name="zero_capacity_with_close_invalid", widths=[4], histories=[0], zero_capacity=True),
              dict(name="dummy_stale", widths=[4, 5, 3], histories=[0, -1, 1], trailing=2),
              dict(name="t0", widths=[], histories=[], zero_capacity=True),
              dict(name="all_dummy", widths=[3, 5], histories=[-1, -1])]
    for mutation in ("causal_stale", "negative_slot_offset", "large_slot_offset", "negative_page", "large_page",
                     "short_page", "negative_prefix", "large_prefix"):
        result.append(dict(name=mutation, widths=[4, 7, 3], histories=[0, 1, 7], mutation=mutation))
    return result


def normalized_dag():
    old_tree = ast.parse((CANONICAL / "decode_indexer_compressor.py").read_text())
    old = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == "indexer_compressor_write")
    new_tree = ast.parse(SOURCE.read_text())
    new = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == "indexer_compact_quant_cache_stage")
    mapping = dict(kv_blk_f32="kv_f32", kv_amax="amax", kv_scale_q_row="scale_q_row",
                   kv_scale_dq_col="scale_dq_col", kv_scale_q_col="scale_q_col", kv_scaled="scaled",
                   kv_i32="quant_i32", kv_half="quant_half", kv_i8_blk="quant_i8", RMS_PAD_TILE="ROW_TILE")
    targets = ["kv_f32", "amax", "scale_q_row", "scale_dq_col", "scale_q_col", "scaled", "quant_i32", "quant_half", "quant_i8"]
    constants = {}
    for node in ast.walk(new):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("floor", "numerator"):
                assert node.targets[0].id not in constants
                constants[node.targets[0].id] = node.value
    assert set(constants) == {"floor", "numerator"}

    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            # CPU lowering 将两个 full 提取成带 valid_shape 的命名常量。
            # 仅内联这两个唯一赋值；不把任意中间量展开，以免掩盖算术顺序变化。
            if isinstance(node.ctx, ast.Load) and node.id in constants:
                return self.visit(copy.deepcopy(constants[node.id]))
            return ast.copy_location(ast.Name(id=mapping.get(node.id, node.id), ctx=node.ctx), node)

        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "kv_final":
                return ast.copy_location(ast.Name(id="raw", ctx=ast.Load()), node)
            return self.generic_visit(node)

        def visit_Call(self, node):
            node = self.generic_visit(node)
            name = ast.unparse(node.func)
            if name in ("pl.set_validshape", "pl.tile.set_validshape"):
                return node.args[0]
            if name == "pl.tile.full":
                node.func = ast.parse("pl.full", mode="eval").body
            if name in ("pl.row_max", "pl.tile.row_max"):
                node.func = ast.parse("pl.row_max", mode="eval").body
                node.args = node.args[:1]  # UB scratch只是存储参数，未改变归约。
            return node

    def select(function):
        selected = []
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if mapping.get(node.targets[0].id, node.targets[0].id) in targets:
                    selected.append(node)
        return sorted(selected, key=lambda node: node.lineno)

    old_nodes = [Normalize().visit(node) for node in select(old)]
    new_nodes = [Normalize().visit(node) for node in select(new)]
    assert len(old_nodes) == len(new_nodes) == 11
    assert [ast.dump(node, include_attributes=False) for node in old_nodes] == [ast.dump(node, include_attributes=False) for node in new_nodes]
    commit = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == "indexer_compact_scale_commit_stage")
    names = []
    for node in sorted((node for node in ast.walk(commit) if isinstance(node, ast.Call)), key=lambda node: (node.lineno, node.col_offset)):
        name = ast.unparse(node.func)
        if name in ("pl.tile.load", "pl.tile.reinterpret_view", "pl.tile.write", "pl.tile.store"):
            names.append(name)
    assert names == ["pl.tile.load", "pl.tile.reinterpret_view", "pl.tile.write", "pl.tile.reinterpret_view", "pl.tile.store"]
    outer_if = next(node for node in commit.body if isinstance(node, ast.If))
    assert ast.unparse(outer_if.test) == "worker_id == 0"
    return {"canonical": str(CANONICAL / "decode_indexer_compressor.py") + ":indexer_compressor_write",
            "quantization_assignments": [ast.unparse(node) for node in new_nodes], "normalized_ast_exact": True,
            "normalization": ["变量重命名", "输入kv_final切片映射到raw", "pl.tile.full与pl.full等价",
                              "唯一赋值floor/numerator常量内联，保留full原始数值与dtype",
                              "移除新增valid_shape描述和row_max UB scratch参数，不改变数值DAG"],
            "rmw_call_order": names, "single_worker_zero_guard": True,
            "note": "AST等价只覆盖数值表达式；动态有效范围另由同源控制流及独立枚举oracle检查"}


def check_case(interpreter, spec, workers):
    fixture = fixture_for(spec)
    interpreter.bind(fixture)
    expected_offsets, rows, groups = enumerate_compact(fixture)
    interpreter.metadata(fixture)
    assert np.array_equal(fixture["offsets"], expected_offsets)
    assert fixture["count"][0] == len(rows)
    mutation = spec.get("mutation")
    if mutation == "negative_prefix":
        fixture["offsets"][:] -= 1000
    elif mutation == "large_prefix":
        fixture["offsets"][:] += 1000
    initial = {name: array.copy() for name, array in fixture.items() if isinstance(array, np.ndarray)}
    golden = expected(fixture, interpreter.constants)
    interpreter.events.clear()
    interpreter.quant(fixture, workers)
    assert np.array_equal(fixture["cache"], golden["key_cache"])
    assert np.array_equal(fixture["debug"].view(np.uint32), golden["debug"].view(np.uint32))
    assert np.array_equal(fixture["scale_values"].view(np.uint32), golden["scales"].view(np.uint32))
    quant_events = list(interpreter.events)
    for kind, name, origin, shape, worker in quant_events:
        if kind == "store" and name == "scale_values":
            assert shape[0] <= 16
            containing = [request for request, (prefix, current) in groups.items()
                          if prefix <= origin[0] and origin[0] + shape[0] <= prefix + len(current)]
            assert len(containing) == 1 and worker == containing[0] % workers
    before_nonowner = fixture["cache"].copy()
    interpreter.commit(fixture, 1)
    assert np.array_equal(fixture["cache"], before_nonowner)
    start = len(interpreter.events)
    interpreter.commit(fixture, 0)
    commit_events = interpreter.events[start:]
    assert np.array_equal(fixture["cache"], golden["final_cache"])
    assert np.array_equal(fixture["cache"][~golden["final_mask"]], initial["cache"][~golden["final_mask"]])
    rmw = [(kind, origin, shape, worker) for kind, name, origin, shape, worker in commit_events
           if name == "cache" and kind in ("load", "store")]
    assert len(rmw) == 2 * len(golden["committed"])
    for index, (page, offset, compact) in enumerate(golden["committed"]):
        assert rmw[2 * index] == ("load", (page, 4096), (1, 64), 0)
        assert rmw[2 * index + 1] == ("store", (page, 4096), (1, 64), 0)
    assert np.all(fixture["guarded_cache"][[0, -1]] == BYTE_GUARD)
    assert np.all(fixture["guarded_debug"][[0, -1]] == POISON)
    if mutation == "short_page":
        assert np.array_equal(fixture["guarded_cache"][1:-1, 4159:], initial["guarded_cache"][1:-1, 4159:])
    active_tokens = {token for _, begin, end, _ in requests(fixture) for token in range(begin, end)}
    candidate_tokens = {token for _, token, _ in rows}
    for kind, name, origin, _, _ in interpreter.events:
        if name == "positions":
            assert origin[0] in active_tokens and origin[0] in candidate_tokens
    for name in ("values", "positions", "bounds", "lengths", "offsets", "slots"):
        assert np.array_equal(fixture[name], initial[name]), name
    return {"case": spec["name"], "workers": workers, "tokens": len(fixture["positions"]),
            "compact_count": len(rows), "request_prefixes": [prefix for prefix, _ in groups.values()],
            "unaligned_request_prefix": any(prefix % 16 for prefix, current in groups.values() if current),
            "updated_slots": len(golden["committed"]), "rmw_pairs": len(rmw) // 2,
            "cpu_key_scale_debug_exact": True, "neighbor_bytes_preserved": True,
            "padding_poison_guards_preserved": True, "read_only_inputs_unchanged": True, "passed": True}


def parallel_rmw_counterexample():
    page = np.arange(64, dtype=np.uint8).view(np.int8).copy()
    first, second = page.copy(), page.copy()
    first.view(np.float16)[3] = np.float16(0.125)
    second.view(np.float16)[10] = np.float16(0.25)
    expected = page.copy()
    expected.view(np.float16)[3] = np.float16(0.125)
    expected.view(np.float16)[10] = np.float16(0.25)
    actual = first.copy()
    actual[:] = second
    assert not np.array_equal(actual, expected)
    return {"different_slots": [3, 10], "same_rmw_region_bytes": 64, "lost_update_if_both_read_old_page": True,
            "different_bytes": int(np.count_nonzero(actual != expected)),
            "note": "故意错误的CPU交错反例；正式阶段只让worker0串行RMW"}


def main():
    interpreter = Interpreter()
    audit = normalized_dag()
    cases = [check_case(interpreter, spec, workers) for spec in specs() for workers in (1, 3, 48)]
    report = {"passed": True, "source": str(SOURCE), "source_ast_interpreted": True,
              "metadata_source": str(PACKAGE / "metadata_stages.py"), "case_count": len(cases), "cases": cases,
              "canonical_dag": audit, "rmw_counterexample": parallel_rmw_counterexample(),
              "cpu_numerical_model": "BF16位模式RNE对独立frexp RNE；逐行FP32量化对独立numpy/Python round；FP16 bytes",
              "device_execution": False, "new_compilation": False, "a3_bitwise_proved": False,
              "preparation_attempts": ["首次AST审计比较失败：新源码将floor/numerator提取为带valid_shape的常量，而审计尚未内联；仅修正审计归一化，未改生产数学",
                                       "适配生产尾块修正：full常量显式valid_shape；动态单行Tile改用pl.slice。CPU解释器执行当前源码接口"],
              "limits": ["不模拟A3 div/recip指令精度、Scalar FP16默认cast模式、DMA或cache一致性",
                         "values、scale_values、slots容量与同轮Q/L/offset必须由组装入口统一验证",
                         "quant因values不足跳过时，commit没有values参数，不得仍把旧scale作为有效结果提交",
                         "Q全局无重叠、紧凑offset同源、有效slot唯一仍是外层合同；非法metadata只测试越界保护",
                         "未执行Hadamard、完整Indexer或完整CSA，不报告性能"]}
    (HERE / "oracle.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    review = {"passed": True, "blocking_defects_found": [], "source_modified": False,
              "address_review": ["Q/L决定每请求关闭数与first_token，offset与历史组数合成prefix；未使用固定S",
                                 "16行只作UB分块，请求compact起点不对齐16也只写rows行scale",
                                 "C=0与负prefix先于slot读取，不沿用旧safe_row=-1读取",
                                 "合法negative compact_offset不可当无效；应检查合成prefix",
                                 "stale causal行可写中间scale，不得写key/debug/Native scale"],
              "numerical_dag_review": audit,
              "rmw_review": ["唯一worker0按请求/关闭行顺序读取64B、reinterpret FP16、改一个元素、转回并写64B",
                             "不同请求同页不同slot仍共享同一64B；不能凭slot唯一改多核并发",
                             "key区域、历史邻槽scale及页尾额外字节均按独立掩码原样保留"],
              "readability_review": {"module_overview": True, "inputs_outputs_layout": True,
                                     "dynamic_capacity": True, "worker_and_phase_dependency": True,
                                     "numerical_order": True, "tail_valid_rows": True,
                                     "nonblocking_note": "最终组装入口必须统一拒绝values容量不足，避免仅quant跳过而commit读旧scale"},
              "evidence": "oracle.json", "new_compilation": False, "device_execution": False,
              "not_covered": report["limits"]}
    (HERE / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(cases), "canonical_dag_exact": True,
                      "key_scale_debug_cpu_exact": True, "device_execution": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
