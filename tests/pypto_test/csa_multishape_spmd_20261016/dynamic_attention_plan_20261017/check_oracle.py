# SPDX-License-Identifier: Apache-2.0
"""直接解释正式helper源的CPU地址/尾块/mask oracle；不模拟NPU同步或指令精度。"""

import argparse
import ast
import copy
import json
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class Tile:
    def __init__(self, data, valid=None):
        self.data = np.asarray(data)
        self.valid = tuple(valid or self.data.shape)


class Plain(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.decorator_list = []
        node.returns = None
        for argument in node.args.args:
            argument.annotation = None
        return self.generic_visit(node)


class MathNames(ast.NodeTransformer):
    def visit_Name(self, node):
        names = {"BIAS_T_TILE": "ROWS", "ROPE_CS_T_TILE": "ROWS", "IDX_TOPK": "TOPK"}
        return ast.copy_location(ast.Name(id=names.get(node.id, node.id), ctx=node.ctx), node)

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node) in ("pl.tile.full", "pl.tile.arange"):
            return ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr=node.attr, ctx=node.ctx)
        return node


def verify_math(old, new):
    source = ast.parse(old.read_text())
    destination = ast.parse(new.read_text())
    names = [
        "c_pos_q",
        "c_upper_b",
        "c_ge_tile",
        "c_lt_tile",
        "c_mask_tile",
        "c_out_tile",
        "v_columns",
        "v_col_tile",
        "v_valid_tile",
        "v_lo_fp32",
        "v_hi_fp32",
        "v_ge",
        "v_lt",
        "cs_ones",
        "cs_idx_f",
        "cs_col",
        "cs_dup_i32",
        "cs_dup_f",
        "cs_lane",
        "cs_sign",
    ]
    for name in names:
        assignments = []
        for tree in (source, destination):
            item = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == name)
            assignments.append(MathNames().visit(copy.deepcopy(item.value)))
        assert ast.dump(assignments[0], include_attributes=False) == ast.dump(
            assignments[1], include_attributes=False
        ), name
    prohibited = {"S", "DECODE_SEQ", "DECODE_BATCH", "DECODE_TOKENS"}
    assert not prohibited.intersection(n.id for n in ast.walk(destination) if isinstance(n, ast.Name))
    calls = {ast.unparse(n.func) for n in ast.walk(destination) if isinstance(n, ast.Call)}
    assert not {"pl.spmd", "pl.at", "pl.system.submit", "pl.system.syncall"}.intersection(calls)
    return names


def source_namespace(package, accesses):
    gm_names, allowed_rows, writes = {}, {}, []

    def data(value):
        return value.data if isinstance(value, Tile) else np.asarray(value)

    def checked(tensor, offsets, widths, operation):
        assert len(offsets) == tensor.ndim == len(widths)
        assert all(
            start >= 0 and start + width <= extent for start, width, extent in zip(offsets, widths, tensor.shape)
        ), (operation, offsets, widths, tensor.shape)
        if operation == "read" and id(tensor) in allowed_rows:
            assert all(row in allowed_rows[id(tensor)] for row in range(offsets[0], offsets[0] + widths[0])), (
                gm_names[id(tensor)],
                offsets,
            )
        accesses.append((gm_names.get(id(tensor), "unknown"), operation, list(offsets), list(widths)))

    def scalar_read(tensor, indices):
        if isinstance(tensor, Tile):
            assert all(0 <= i < n for i, n in zip(indices, tensor.data.shape))
            return tensor.data[tuple(indices)]
        checked(tensor, indices, [1] * tensor.ndim, "read")
        return tensor[tuple(indices)]

    def scalar_write(tensor, indices, value):
        if isinstance(tensor, Tile):
            assert all(0 <= i < n for i, n in zip(indices, tensor.data.shape))
            tensor.data[tuple(indices)] = value
        else:
            checked(tensor, indices, [1] * tensor.ndim, "write")
            tensor[tuple(indices)] = value
            writes.append((gm_names.get(id(tensor)), tuple(indices), (1,) * tensor.ndim))

    def load(tensor, offsets, shape, valid_shape=None):
        valid = shape if valid_shape is None else valid_shape
        checked(tensor, offsets, valid, "read")
        array = np.zeros(shape, dtype=tensor.dtype)
        array[tuple(slice(0, n) for n in valid)] = tensor[tuple(slice(s, s + n) for s, n in zip(offsets, valid))]
        return Tile(array, valid)

    def store(tile, offsets, tensor):
        assert isinstance(tile, Tile)
        checked(tensor, offsets, tile.valid, "write")
        tensor[tuple(slice(s, s + n) for s, n in zip(offsets, tile.valid))] = tile.data[
            tuple(slice(0, n) for n in tile.valid)
        ]
        writes.append((gm_names.get(id(tensor)), tuple(offsets), tile.valid))

    def binary(operation, left, right):
        tiles = [v for v in (left, right) if isinstance(v, Tile)]
        # PyPTO把1.0等立即数绑定为操作Tile的FP32类型；NumPy 2的0D FP64
        # ndarray会反而提升整个表达式。显式模拟DSL类型，避免大整数边界假失败。
        dtype = tiles[0].data.dtype if tiles else None
        lhs = data(left) if isinstance(left, Tile) else np.asarray(left, dtype=dtype)
        rhs = data(right) if isinstance(right, Tile) else np.asarray(right, dtype=dtype)
        result = operation(lhs, rhs)
        if not tiles:
            return result
        valid = [
            min([t.valid[i] if t.data.shape[i] != 1 else n for t in tiles] + [n]) for i, n in enumerate(result.shape)
        ]
        return Tile(result, valid)

    def cast(value, target_type=None, mode=None):
        result = data(value).astype(target_type)
        return Tile(result, value.valid) if isinstance(value, Tile) else result[()]

    def tile_slice(value, shape, offsets, valid_shape=None, **unused):
        assert isinstance(value, Tile)
        valid = shape if valid_shape is None else valid_shape
        assert all(
            start >= 0 and start + width <= extent for start, width, extent in zip(offsets, valid, value.data.shape)
        )
        result = np.zeros(shape, dtype=value.data.dtype)
        result[tuple(slice(0, n) for n in valid)] = value.data[tuple(slice(s, s + n) for s, n in zip(offsets, valid))]
        return Tile(result, valid)

    def row_max(tile, _temporary=None):
        result = tile.data.max(axis=1, keepdims=True)
        return Tile(result, [tile.valid[0], 1])

    def row_min(tile, _temporary=None):
        result = tile.data[:, : tile.valid[1]].min(axis=1, keepdims=True)
        return Tile(result, [tile.valid[0], 1])

    def tile_assemble(target, value, offsets):
        target.data[tuple(slice(s, s + n) for s, n in zip(offsets, value.valid))] = value.data[
            tuple(slice(0, n) for n in value.valid)
        ]
        return target

    full = lambda shape, dtype, value: Tile(np.full(shape, value, dtype=dtype))
    arange = lambda start, shape, dtype: Tile(np.arange(start, start + np.prod(shape), dtype=dtype).reshape(shape))
    pl = SimpleNamespace(
        INT32=np.int32,
        INT64=np.int64,
        INDEX=np.int64,
        FP32=np.float32,
        range=lambda *args: range(*(int(x) for x in args)),
        create_tile=lambda shape, dtype: Tile(np.zeros(shape, dtype=dtype)),
        reshape=lambda tile, shape: Tile(tile.data.reshape(shape), shape),
        min=min,
        max=max,
        read=scalar_read,
        write=scalar_write,
        load=load,
        store=store,
        slice=tile_slice,
        cast=cast,
        set_validshape=lambda t, *v: Tile(t.data, v),
        row_max=row_max,
        row_min=row_min,
        neg=lambda t: Tile(-t.data, t.valid),
        tensor=SimpleNamespace(read=scalar_read, dim=lambda x, a: x.shape[a]),
        tile=SimpleNamespace(full=full, arange=arange, write=scalar_write, assemble=tile_assemble),
    )
    for name, operation in [
        ("add", np.add),
        ("sub", np.subtract),
        ("mul", np.multiply),
        ("minimum", np.minimum),
        ("maximum", np.maximum),
        ("row_expand_mul", np.multiply),
        ("col_expand_mul", np.multiply),
        ("row_expand_sub", np.subtract),
    ]:
        setattr(pl, name, lambda a, b, operation=operation: binary(operation, a, b))
    namespace = {"pl": pl}
    nodes = []
    for filename in ("request_geometry.py", "attention_plan_stages.py"):
        tree = ast.parse((package / filename).read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                with suppress(ValueError, TypeError):
                    namespace[ast.unparse(node.targets[0])] = ast.literal_eval(node.value)
            elif isinstance(node, ast.FunctionDef):
                nodes.append(Plain().visit(copy.deepcopy(node)))
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<formal-source-cpu-oracle>", "exec"
        ),
        namespace,
    )
    return namespace, gm_names, allowed_rows, writes


def geometry(q, lengths, tokens):
    owners = []
    for token in range(tokens):
        matches = []
        for r, length in enumerate(lengths):
            if length > 0 and r + 1 < len(q):
                start, end = int(q[r]), int(q[r + 1])
                if 0 <= start < end <= tokens and end - start <= length and start <= token < end:
                    matches.append((r, start, end))
        owners.append(matches[0] if len(matches) == 1 else None)
    return owners


def expected(topk, positions, q, lengths, original_table, compressed_table, opages, cpages):
    tokens = len(positions)
    out_i = np.full((tokens, 512), -1, np.int32)
    out_b = np.full((tokens, 640), np.float32(-1e20), np.float32)
    out_m = np.zeros((tokens, 5), np.int32)
    owners = geometry(q, lengths, tokens)
    for token, owner in enumerate(owners):
        if owner is None:
            continue
        r, start, end = owner
        pos = int(positions[token])
        if pos != int(lengths[r]) - (end - start) + token - start:
            continue
        upper = np.float32(np.int32(np.float32(np.float32(pos) + np.float32(1)) * np.float32(0.25)))
        raw = topk[token].astype(np.float32)
        mask = np.minimum(np.maximum(raw + np.float32(1), np.float32(0)), np.float32(1)) * np.minimum(
            np.maximum(upper - raw, np.float32(0)), np.float32(1)
        )
        page_mask = np.zeros(512, np.float32)
        emitted = (mask * (raw + np.float32(1)) - np.float32(1)).astype(np.int32)
        for k, candidate in enumerate(emitted):
            if 0 <= candidate < int(upper):
                column = int(candidate) // 32
                if column < compressed_table.shape[1] and 0 <= compressed_table[r, column] < cpages:
                    page_mask[k] = 1
        mask = mask * page_mask
        index = mask * (raw + np.float32(1)) - np.float32(1)
        out_i[token] = index.astype(np.int32)
        out_b[token, 128:] = np.minimum(index, np.float32(0)) * np.float32(1e20)
        for block in range(4):
            out_m[token, 1 + block] = np.int32(mask[block * 128 : (block + 1) * 128].max())
        length = min(pos + 1, 128)
        first = pos - length + 1
        for k in range(length):
            column = (first + k) // 32
            if 0 <= column < original_table.shape[1] and 0 <= original_table[r, column] < opages:
                out_b[token, k] = np.float32(0)
                out_m[token, 0] = 1
    return out_i, out_b, out_m, owners


def audit(repo):
    package = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    names = verify_math(
        package.parent / "deepseek_v4_flash_csa/decode_sparse_attn_csa.py", package / "attention_plan_stages.py"
    )
    accesses = []
    ns, gm_names, allowed, writes = source_namespace(package, accesses)
    cases = []
    for batch, history in [
        (1, 0),
        (2, 29),
        (4, 131072),
        (8, 131072),
        (16, 131072),
        (24, 131072),
        (28, 131072),
        (16, 8192),
        (32, 8192),
    ]:
        cases.append(
            {
                "name": f"uniform_B{batch}_H{history}",
                "q": list(range(0, batch * 6 + 1, 6)),
                "lengths": [history + 6] * batch,
                "T": batch * 6,
                "canonical": True,
            }
        )
    for sizes in ([1], [2, 4], [2, 2, 2], [1, 3, 5], [5, 1, 7, 3], [0, 1, 0, 8, 2]):
        q = np.cumsum([0, *sizes]).tolist()
        cases.append({"name": f"mixed_{sizes}", "q": q, "lengths": [131 + size for size in sizes], "T": q[-1] + 3})
    for spec in [
        {"name": "fp32_index_page_boundary", "q": [0, 1], "lengths": [67108928], "T": 1, "huge_boundary": True},
        {"name": "empty", "q": [0], "lengths": [], "T": 0},
        {"name": "no_requests", "q": [0], "lengths": [], "T": 7},
        {"name": "all_dummy", "q": [99, -9, 999], "lengths": [0, -1], "T": 9},
        {"name": "mixed_dummy", "q": [0, 3, 6, 9], "lengths": [135, 0, 137], "T": 10},
        {"name": "reverse", "q": [0, 7, 3], "lengths": [135, 136], "T": 9},
        {"name": "overlap", "q": [0, 6, 2, 8], "lengths": [132, 133, 134], "T": 9},
        {"name": "short_Q", "q": [0, 3], "lengths": [134, 135, 136], "T": 9},
        {"name": "negative_Q", "q": [-2, 3, 7], "lengths": [132, 133], "T": 8},
        {"name": "oversize_Q", "q": [0, 20], "lengths": [141], "T": 9},
        {"name": "L_less_query", "q": [0, 9], "lengths": [2], "T": 9},
        {"name": "stale_positions", "q": [0, 3, 9], "lengths": [134, 139], "T": 11, "stale": True},
        {"name": "holes", "q": [0, 5, 9], "lengths": [133, 139], "T": 9, "holes": True},
        {"name": "positive_oob_pages", "q": [0, 5, 9], "lengths": [133, 139], "T": 9, "outpages": True},
        {"name": "short_tables", "q": [0, 5, 9], "lengths": [133, 139], "T": 9, "shorttables": True},
        {"name": "zero_tables", "q": [0, 5, 9], "lengths": [133, 139], "T": 9, "zerotables": True},
        {"name": "zero_cache_pages", "q": [0, 5, 9], "lengths": [133, 139], "T": 9, "zeropages": True},
    ]:
        cases.append(spec)
    rng = np.random.default_rng(20261017)
    records = []
    for number, spec in enumerate(cases):
        q = np.array(spec["q"], np.int32)
        lengths = np.array(spec["lengths"], np.int32)
        tokens = spec["T"]
        batch = len(lengths)
        owners = geometry(q, lengths, tokens)
        positions = np.full(tokens, 2**50, np.int64)
        for t, owner in enumerate(owners):
            if owner:
                r, start, end = owner
                positions[t] = int(lengths[r]) - (end - start) + t - start
        if spec.get("stale"):
            positions[::2] += 1
        max_length = max([int(v) for v in lengths] + [1])
        oc = (max_length + 31) // 32
        cc = (max_length // 4 + 31) // 32
        if spec.get("shorttables"):
            oc, cc = 2, 1
        if spec.get("zerotables"):
            oc, cc = 0, 0
        if spec.get("huge_boundary"):
            oc, cc = 0, 524288
        opages, cpages = 23, 19
        ot = np.arange(batch * oc, dtype=np.int32).reshape(batch, oc) % opages
        ct = np.arange(batch * cc, dtype=np.int32).reshape(batch, cc) % cpages
        if spec.get("holes"):
            ot[:, ::2] = -1
            ct[:, ::2] = -1
        if spec.get("outpages"):
            ot[:, ::2] = opages
            ct[:, ::2] = cpages + 1
        if spec.get("zeropages"):
            opages = cpages = 0
        topk = rng.integers(-4, max(4, max_length // 4 + 20), size=(tokens, 512), dtype=np.int32)
        if tokens:
            topk[:, 0:8] = [-2147483648, -1, 0, 1, 31, 32, 2147483647, 127]
        if spec.get("huge_boundary"):
            topk[:] = 16777216
            topk[0, :8] = [-1, 0, 16777215, 16777216, 16777217, 16777218, 16777231, 16777232]
        cap = tokens + 11
        indices = np.full((cap, 512), -777, np.int32)
        bias = np.full((cap, 640), 777.0, np.float32)
        mask = np.full((cap, 16), -777, np.int32)
        sin = rng.normal(size=(tokens, 64)).astype(np.float32)
        signed = np.full((tokens + 13, 64), 777.0, np.float32)
        for name, value in [
            ("topk", topk),
            ("positions", positions),
            ("bounds", q),
            ("lengths", lengths),
            ("original_table", ot),
            ("compressed_table", ct),
            ("indices", indices),
            ("bias", bias),
            ("mask", mask),
            ("sin", sin),
            ("signed", signed),
        ]:
            gm_names[id(value)] = name
        allowed.clear()
        allowed[id(positions)] = {t for t, o in enumerate(owners) if o is not None}
        valid_requests = {
            o[0] for t, o in enumerate(owners) if o and positions[t] == int(lengths[o[0]]) - (o[2] - o[1]) + t - o[1]
        }
        allowed[id(ot)] = valid_requests
        allowed[id(ct)] = valid_requests
        workers = (1, 3, 24)[number % 3]
        accesses.clear()
        writes.clear()
        for worker in range(workers):
            ns["attention_slot_plan_stage"](
                topk, positions, q, lengths, ot, ct, indices, bias, mask, opages, cpages, worker, workers
            )
            ns["attention_inverse_rope_sign_stage"](sin, signed, worker, workers)
        ei, eb, em, unused = expected(topk, positions, q, lengths, ot, ct, opages, cpages)
        np.testing.assert_array_equal(indices[:tokens], ei)
        np.testing.assert_array_equal(bias[:tokens], eb)
        np.testing.assert_array_equal(mask[:tokens, :5], em)
        np.testing.assert_array_equal(signed[:tokens], sin * np.array([1.0, -1.0] * 32, np.float32))
        assert np.all(indices[tokens:] == -777) and np.all(bias[tokens:] == 777) and np.all(mask[tokens:] == -777)
        assert np.all(mask[:tokens, 5:] == -777) and np.all(signed[tokens:] == 777)
        # 除SWA有空洞时允许覆盖该行bias，indices/mask/signed每元素必须唯一写入。
        for name, array, columns in [("indices", indices, 512), ("mask", mask, 5), ("signed", signed, 64)]:
            counts = np.zeros(array.shape, np.int32)
            for item, offset, shape in writes:
                if item == name:
                    counts[tuple(slice(s, s + n) for s, n in zip(offset, shape))] += 1
            assert np.all(counts[:tokens, :columns] == 1) and np.all(counts[tokens:] == 0), name
            if name == "mask":
                assert np.all(counts[:tokens, 5:] == 0)
        records.append(
            {
                "name": spec["name"],
                "T": tokens,
                "B": batch,
                "workers": workers,
                "Q": spec["q"],
                "canonical_legal": spec.get("canonical", False),
                "accesses": len(accesses),
                "passed": True,
            }
        )
    return {
        "passed": True,
        "source_interpreted": True,
        "cases": records,
        "case_count": len(records),
        "canonical_math_AST_equal": names,
        "mixed_same_T_cases": ["mixed_[2, 4]", "mixed_[2, 2, 2]"],
        "poison_pad_and_mask_last_11_untouched": True,
        "actual_T_and_workers_coverage": True,
        "inactive_positions_and_page_reads_rejected": True,
        "invalid_pages_mask_index_bias_consistent": True,
        "device_execution": False,
        "limits": [
            "NumPy FP32基础算术解释源，不模拟NPU指令和缓存。",
            "canonical数学AST/合法输入oracle对照，不导入带静态shape断言的canonical模块。",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = audit(Path(__file__).resolve().parents[4])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": True, "cases": result["case_count"], "math_assignments": len(result["canonical_math_AST_equal"])}
        )
    )


if __name__ == "__main__":
    main()
