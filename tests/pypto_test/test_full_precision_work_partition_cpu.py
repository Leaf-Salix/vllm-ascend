# SPDX-License-Identifier: Apache-2.0
"""精度版实际DSL AST的工作划分与K序CPU检查；不执行浮点数值或NPU。"""

import ast
import copy
import runpy
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/pypto"
FULL = "deepseek_v4_flash_csa_tnd_full_precision"
PREC = "deepseek_v4_flash_csa_tnd_precision"
parse = lambda p: ast.parse(p.read_text())
dump = lambda n: ast.dump(n, include_attributes=False)
fn = lambda t, name: next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name)


def named_with(t, name):
    return next(
        n
        for n in ast.walk(t)
        if isinstance(n, ast.With)
        and any(
            isinstance(x.context_expr, ast.Call)
            and any(
                k.arg == "name_hint" and isinstance(k.value, ast.Constant) and k.value.value == name
                for k in x.context_expr.keywords
            )
            for x in n.items
        )
    )


# Test-only address/K-order simulators. No production numerical implementation.
class View:
    def __init__(self, kind, shape, offset, valid=None):
        self.kind = kind
        self.shape = shape
        self.offset = offset
        self.valid = valid or shape


class Acc:
    def __init__(self, shape):
        self.shape = shape
        self.row = 0
        self.valid = shape[0]
        self.seq = []


class PL:
    FP32 = "fp32"

    @staticmethod
    def spmd(n, **kw):
        return range(n)

    range = staticmethod(range)
    min = staticmethod(min)

    @staticmethod
    def create_tensor(shape, **kw):
        return Acc(shape)

    @staticmethod
    def slice(source, shape, offset, valid_shape=None):
        out = View(source, shape, offset, valid_shape)
        if source == "flat":
            assert offset[0] >= 0 and offset[0] + out.valid[0] <= PL.padded and (offset[1] + shape[1] <= 32768)
        else:
            assert source == "weight" and offset[0] + shape[0] <= 32768
        return out

    @staticmethod
    def matmul(x, w, **kw):
        a = Acc([x.shape[0], 32])
        return PL.matmul_acc(a, x, w, init_cond=True)

    @staticmethod
    def matmul_acc(a, x, w, init_cond=False):
        assert x.offset[1] == w.offset[0] and x.shape[1] == w.shape[0] == 128
        if init_cond:
            a.seq = []
        a.seq.extend(range(x.offset[1] // 16, x.offset[1] // 16 + 8))
        a.row = x.offset[0]
        a.valid = x.valid[0]
        return a

    @staticmethod
    def set_validshape(a, m, n):
        assert 0 < m <= a.shape[0] and n == 32
        a.valid = m
        return a

    @staticmethod
    def assemble(dst, a, offset):
        base, col = offset
        split = base // PL.padded
        assert col == 0 and base % PL.padded == a.row
        assert a.row + a.valid <= PL.padded, "cross-split store"
        for i in range(a.valid):
            key = (split, a.row + i)
            assert key not in dst
            dst[key] = list(a.seq)
        return dst


class Weight:
    def __getitem__(self, key):
        return key


class Trace:
    FP32 = "FP32"
    rows = 0
    wide = 0
    narrow = 0
    range = staticmethod(range)
    parallel = staticmethod(range)

    @staticmethod
    def pipeline(*args, stage):
        assert stage == 2
        return range(*args)

    min = staticmethod(min)
    max = staticmethod(max)

    @staticmethod
    def spmd(n, **kwargs):
        return range(n)

    @staticmethod
    def read(tensor, index):
        return int(tensor[tuple(index)])

    @staticmethod
    def create_tensor(shape, dtype):
        assert dtype == "FP32"
        if len(shape) == 2:
            assert shape[1] == 8192
            return {"shape": shape, "values": {}}
        assert shape in ([1, 16, 128], [1, 128, 128])
        if shape[1] == 128:
            Trace.wide += 1
        else:
            Trace.narrow += 1
        return {"shape": shape, "sequence": [], "instructions": 0}

    @staticmethod
    def slice(tensor, shape, offset, valid_shape):
        assert shape[0] in (16, 128) and shape[1] in (16, 256)
        assert 0 < valid_shape[0] <= shape[0] and valid_shape[1] == shape[1]
        assert offset[0] >= 0 and offset[0] + valid_shape[0] <= Trace.rows
        assert offset[1] % 4096 + shape[1] <= 4096
        return (shape, offset, valid_shape)

    @staticmethod
    def matmul(x, w, out_dtype):
        assert out_dtype == "FP32"
        acc = Trace.create_tensor([1, x[0][0], 128], dtype="FP32")
        return Trace.matmul_acc(acc, x, w, True)

    @staticmethod
    def matmul_acc(acc, x, w, init_cond):
        shape, offset, valid = x
        assert acc["shape"] == [1, shape[0], 128]
        group = offset[1] // 4096
        k = offset[1] % 4096
        assert w[0] == slice(group, group + 1) and w[1] == slice(k, k + shape[1])
        assert bool(init_cond) == (acc["instructions"] == 0)
        acc["sequence"] += list(range(k // 16, (k + shape[1]) // 16))
        acc["instructions"] += 1
        acc["valid_rows"] = valid[0]
        if init_cond:
            acc["seed_K"] = shape[1]
        return acc

    @staticmethod
    def assemble(output, acc, offset):
        row, column = offset
        assert row + acc["shape"][1] <= output["shape"][0], "physical M128 store exceeds internal capacity"
        assert column + 128 <= 8192
        assert len(acc["sequence"]) == 256 and len(set(acc["sequence"])) == 256
        group = column // 1024
        block = column % 1024 // 128
        for rb in range(row // 16, (row + acc["valid_rows"] + 15) // 16):
            key = (rb, group, block)
            assert key not in output["values"], "overlapping real output rows"
            output["values"][key] = acc["sequence"].copy()
            output.setdefault("instructions", {})[key] = acc["instructions"]
            output.setdefault("seed_K", {})[key] = acc["seed_K"]
        return output


def test_checked48_actual_range_covers_all_tokens_once():
    block = named_with(parse(P / PREC / "decode_sparse_attn_csa.py"), "csa_checked_indices")
    workers = ast.literal_eval(block.items[0].context_expr.args[0])
    assert workers == 48
    loop = next(n for n in block.body if isinstance(n, ast.For))
    assert ast.unparse(loop.target) == "checked_t"
    call = compile(ast.Expression(loop.iter), "actual_checked_range", "eval")

    class PL:
        range = staticmethod(range)

    for t in range(1, 257):
        owned = [q for w in range(workers) for q in eval(call, {"pl": PL, "checked_worker": w, "t_dim": t})]
        assert len(owned) == t and sorted(owned) == list(range(t))
    assert 512 * 4 % 64 == 0  # Each independently-owned row is cacheline aligned.


def test_hc_actual_production_AST_all_tokens_K16_order_and_tail():
    production = fn(parse(P / FULL / "hc_pre.py"), "hc_pre_gates_from_rms")
    branch = next(
        n
        for n in production.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "t_dim >= 64" and isinstance(n.body[0], ast.For)
    )
    # Both paths write parent partials; do not consume a helper's obsolete alias.
    parent_alloc = next(
        n
        for n in production.body
        if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "mixes_partials" for x in n.targets)
    )
    assert production.body.index(parent_alloc) < production.body.index(branch)
    for arm in (branch.body, branch.orelse):
        stores = [
            n
            for root in arm
            for n in ast.walk(root)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.assemble"
        ]
        assert len(stores) == 1 and ast.unparse(stores[0].args[0]) == "mixes_partials"
    PL.tensor = PL
    ns = dict(
        pl=PL,
        LINEAR_OK=22,
        LINEAR_K_PER_SPLIT=768,
        HC_DIM=16384,
        MIX_PAD=32,
        panel64_packed_flat="flat",
        panel64_packed_weight="weight",
    )
    args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=n) for n in ("t_dim", "t_linear", "mixes_partials")],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )
    trace = ast.FunctionDef(
        name="trace",
        args=args,
        body=[copy.deepcopy(branch), ast.Return(ast.Name("mixes_partials", ast.Load()))],
        decorator_list=[],
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[trace], type_ignores=[])), "actual_HC_caller_AST", "exec"),
        ns,
    )
    for t in range(1, 257):
        PL.padded = (t + 15) // 16 * 16
        actual = ns["trace"](t, PL.padded, {})
        assert len(actual) == 22 * PL.padded
        for (split, row), seq in actual.items():
            count = min(768, 16384 - split * 768) // 64
            assert seq == list(range(split * 96, split * 96 + count * 8))
            assert 0 <= row < PL.padded
    assert 240 - 3 * 64 == 48


def test_oproj_actual_dispatch_all_T_order_and_fallbacks():
    root = fn(parse(P / FULL / "o_proj.py"), "_o_proj_test")
    dispatch = next(n for n in root.body if isinstance(n, ast.If) and ast.unparse(n.test) == "rows >= 192")
    trace_pl = Trace
    ns = dict(pl=trace_pl, heads=object(), wo_a=Weight())
    prefix = [
        n
        for n in root.body
        if isinstance(n, ast.Assign)
        and any(isinstance(x, ast.Name) and x.id in ("row_tiles", "accum") for x in n.targets)
    ]
    args = ast.arguments(
        posonlyargs=[], args=[ast.arg(arg=n) for n in ("rows", "start")], kwonlyargs=[], kw_defaults=[], defaults=[]
    )
    trace = ast.FunctionDef(
        name="trace",
        args=args,
        body=copy.deepcopy(prefix + [dispatch]) + [ast.Return(ast.Name("accum", ast.Load()))],
        decorator_list=[],
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[trace], type_ignores=[])), "actual_oproj_AST", "exec"), ns)
    data = runpy.run_path(str(P / FULL / "oproj_order_data.py"))

    def check(t, plane):
        trace_pl.rows = t
        trace_pl.wide = trace_pl.narrow = 0
        out = ns["trace"](t, plane)
        assert len(out["values"]) == ((t + 15) // 16) * 64
        for (rb, group, block), seq in out["values"].items():
            first = max(int(plane[rb, group, block]), 0)
            assert seq == [(first + i) % 256 for i in range(256)]
        return trace_pl.wide, trace_pl.narrow

    for t in range(1, 257):
        plane = np.array(data["OPROJ_TILE_STARTS"][data["OPROJ_TOKEN_MODES"][t - 1]], dtype=np.int32)
        w, n = check(t, plane)
        if t == 144:
            assert w == 128 and n == 0  # Low-T remains V4 M128, not global M16 fallback.
    for t in (6, 144, 191, 192, 240, 256):
        for first in range(256):
            check(t, np.full((16, 8, 8), first, dtype=np.int32))
        for peer in range(1, min(16, (t + 15) // 16)):
            plane = np.zeros((16, 8, 8), dtype=np.int32)
            plane[peer] = 1
            check(t, plane)
        check(t, -np.arange(1, 1025, dtype=np.int32).reshape(16, 8, 8))


def test_kv_actual_slot_expressions_lifetimes_and_L1_capacity():
    tree = parse(P / PREC / "decode_sparse_attn_csa.py")
    actual = named_with(tree, "qk_pv")
    byname = {}
    for node in ast.walk(actual):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            byname.setdefault(node.targets[0].id, node.value)
    constants = dict(H=64, HEAD_DIM=512, ATTN_CUBE_KV_TILE=128, ATTN_K_TILE=512)
    # Read shapes and slot arithmetic from live production AST, not a duplicated kernel.
    evaluate = lambda n, **kw: eval(
        compile(ast.Expression(n), "actual_KV_expression", "eval"),
        {"pl": type("PL", (), {"max": staticmethod(max)}), **constants, **kw},
    )
    kvshape = evaluate(byname["qk_l1"].args[0])
    qshape = evaluate(byname["qk_q"].args[2])
    pshape = evaluate(byname["pv_probability"].args[2])
    assert kvshape == [384, 512] and qshape == [64, 512] and pshape == [64, 128]
    sizes = [shape[0] * shape[1] * 2 for shape in (kvshape, qshape, pshape)]
    ends = np.cumsum([0] + sizes).tolist()
    assert ends == [0, 393216, 458752, 475136] and ends[-1] < 512 * 1024
    qloop = next(n for n in ast.walk(actual) if isinstance(n, ast.For) and ast.unparse(n.target) == "qk_part")
    ploop = next(n for n in ast.walk(actual) if isinstance(n, ast.For) and ast.unparse(n.target) == "pv_part")
    assert evaluate(qloop.iter.args[0]) == evaluate(ploop.iter.args[0]) == 4
    reload_if = next(n for n in ploop.body if isinstance(n, ast.If))
    for token in range(256):
        slots = [None] * 3
        for part in range(4):
            row = evaluate(byname["qk_slot_row"], qk_part=part)
            assert row % 128 == 0 and row + 128 <= kvshape[0]
            slots[row // 128] = (token, part)
        assert slots[1:] == [(token, 2), (token, 3)]
        for part in range(4):
            row = evaluate(byname["pv_slot_row"], pv_part=part)
            if evaluate(reload_if.test, pv_part=part):
                slots[row // 128] = (token, part)
            assert slots[row // 128] == (token, part)
    # Logical memory budget and ownership only; backend addresses/fences require
    # source-bound generated-code review and NPU evidence, not this CPU mock.
