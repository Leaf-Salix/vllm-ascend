# SPDX-License-Identifier: Apache-2.0
"""同源动态请求控制流/访存oracle及canonical数学AST审计，不模拟硬件浮点。"""

import ast
import copy
from collections import Counter
from types import SimpleNamespace

import numpy as np


class Plain(ast.NodeTransformer):
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


class MathNames(ast.NodeTransformer):
    def visit_Name(self, node):
        names = {
            "kv_norm_chunk": "kv_rms_low",
            "kv_rope_norm": "kv_rms_high",
            "normed_kv": "normed",
            "kv_final": "output",
            "compact_rows": "valid_count",
        }
        return ast.copy_location(ast.Name(id=names.get(node.id, node.id), ctx=node.ctx), node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node.func) == "pl.row_sum":
            node.args = node.args[:1]
        return node


def assignment(function, name):
    return next(n for n in ast.walk(function) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == name)


def bf16(value):
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def audit(repo):
    old = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_indexer_compressor.py"
    source = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    new = source / "indexer_compressor_output_stages.py"
    old_tree, new_tree = ast.parse(old.read_text()), ast.parse(new.read_text())
    old_functions = {n.name: n for n in old_tree.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: n for n in new_tree.body if isinstance(n, ast.FunctionDef)}
    norm_names = [
        "folded_sq",
        "square_sum",
        "variance",
        "inv_rms",
        "normed_chunk",
        "normed_nope",
        "rope_normed",
        "sin_signed",
        "rope_rot",
        "normed_rope",
    ]
    for name in norm_names:
        a = MathNames().visit(copy.deepcopy(assignment(old_functions["indexer_compressor_pool_projected"], name)))
        b = MathNames().visit(copy.deepcopy(assignment(new_functions["indexer_compressor_norm_rope_stage"], name)))
        assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False), name
    had_names = ["hadamard_tile", "had_b0", "had_rows", "kv_proj_tile", "kv_hadamard_acc"]
    for name in had_names:
        a = MathNames().visit(copy.deepcopy(assignment(old_functions["indexer_compressor_write"], name)))
        b = MathNames().visit(copy.deepcopy(assignment(new_functions["indexer_compressor_hadamard_stage"], name)))
        assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False), name
    forbidden_names = {"S", "DECODE_SEQ", "DECODE_BATCH", "BS_PAD", "BOUNDARY_ROWS_PER_REQUEST"}
    assert not forbidden_names.intersection(n.id for n in ast.walk(new_tree) if isinstance(n, ast.Name))
    calls = {ast.unparse(n.func) for n in ast.walk(new_tree) if isinstance(n, ast.Call)}
    assert not {"pl.spmd", "pl.at", "pl.system.submit", "pl.system.syncall"}.intersection(calls)

    config = ast.parse((old.parent / "config.py").read_text())
    flash = next(n.value for n in config.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "FLASH")
    model = {
        k.arg: ast.literal_eval(k.value)
        for k in flash.keywords
        if k.arg in ("index_head_dim", "qk_rope_head_dim", "rms_norm_eps")
    }
    model["index_nope_head_dim"] = model["index_head_dim"] - model["qk_rope_head_dim"]
    assert model["index_head_dim"] == 128 and model["qk_rope_head_dim"] == 64
    accesses, stores, row_masks = [], [], {}

    def check(tensor, offsets, shape):
        assert len(offsets) == tensor.ndim == len(shape)
        assert all(start >= 0 and start + width <= extent for start, width, extent in zip(offsets, shape, tensor.shape))
        if id(tensor) in row_masks:
            assert all(row_masks[id(tensor)][offsets[0] : offsets[0] + shape[0]])
        accesses.append((id(tensor), tuple(offsets), tuple(shape)))

    def load(tensor, offsets, shape, valid_shape=None, **kwargs):
        valid = shape if valid_shape is None else valid_shape
        check(tensor, offsets, valid)
        value = np.zeros(shape, dtype=tensor.dtype)
        value[tuple(slice(0, n) for n in valid)] = tensor[tuple(slice(s, s + n) for s, n in zip(offsets, valid))]
        return value

    def store(value, offsets, tensor):
        check(tensor, offsets, value.shape)
        tensor[tuple(slice(s, s + n) for s, n in zip(offsets, value.shape))] = value
        stores.append((id(tensor), tuple(offsets), tuple(value.shape)))
        return tensor

    def read(tensor, indices):
        check(tensor, indices, [1] * tensor.ndim)
        return tensor[tuple(indices)].item()

    def write(tensor, indices, value):
        check(tensor, indices, [1] * tensor.ndim)
        tensor[tuple(indices)] = value

    def cast(value, target_type, **kwargs):
        if target_type == "BF16":
            return bf16(value)
        if np.isscalar(value):
            return int(value) if target_type in ("INDEX", "INT32") else np.float32(value)
        return np.asarray(value).astype(np.int32 if target_type == "INT32" else np.float32)

    def gather_row(dest, tensor, dst, src, shape):
        result = dest.copy()
        result[dst[0] : dst[0] + shape[0], dst[1] : dst[1] + shape[1]] = load(tensor, src, shape)
        return result

    pl = SimpleNamespace(
        dynamic=lambda name: name,
        INDEX="INDEX",
        INT32="INT32",
        FP32="FP32",
        BF16="BF16",
        MemorySpace=SimpleNamespace(Vec="Vec"),
        range=range,
        min=min,
        cast=cast,
        load=load,
        store=store,
        read=read,
        write=write,
        gather_row=gather_row,
        slice=lambda x, shape, offsets, **kwargs: load(x, offsets, shape, **kwargs),
        add=np.add,
        sub=np.subtract,
        mul=np.multiply,
        sqrt=np.sqrt,
        recip=np.reciprocal,
        row_sum=lambda x, tmp: np.sum(x, axis=1, keepdims=True, dtype=np.float32),
        col_expand_mul=np.multiply,
        row_expand_mul=np.multiply,
        reshape=np.reshape,
        transpose=lambda x, axis1, axis2: np.swapaxes(x, axis1, axis2),
        create_tile=lambda shape, **kwargs: np.zeros(shape, np.float32),
        tensor=SimpleNamespace(dim=lambda x, axis: x.shape[axis], read=read),
    )
    pl.tile = SimpleNamespace(
        full=lambda shape, dtype, value: np.full(shape, value, dtype=np.float32),
        arange=lambda start, shape, dtype: np.arange(start, start + np.prod(shape), dtype=np.int32).reshape(shape),
        gather=lambda x, index, tmp: x.reshape(-1)[index],
    )
    namespace = {"pl": pl, "M": SimpleNamespace(**model)}
    for filename in ("request_geometry.py", "metadata_stages.py", "indexer_compressor_output_stages.py"):
        path = source / filename
        tree = ast.fix_missing_locations(Plain().visit(ast.parse(path.read_text())))
        exec(compile(tree, str(path), "exec"), namespace)
    norm = namespace["indexer_compressor_norm_rope_stage"]
    rng = np.random.default_rng(20261017)
    cases = [
        ([], []),
        ([6, 6], [-1, -1]),
        ([1, 1], [0, 1]),
        ([2, 2, 2], [128, 256, 64]),
        ([4, 2], [128, 256]),
        ([1, 13, 9], [0, 18, 17]),
        ([3, 2, 6], [127, -1, 257]),
        ([17, 32, 53], [127, 1, 8191]),
        ([6] * 28, [131072] * 28),
    ]
    oracle_cases = 0
    for widths, histories in cases:
        for stale in (False, True):
            bounds = np.asarray([0, *np.cumsum(widths)], np.int32)
            lengths = np.asarray([h + w if h >= 0 else 0 for w, h in zip(widths, histories)], np.int32)
            tokens = int(bounds[-1])
            positions = np.full(tokens + 3, 127, np.int64)
            active = np.zeros(tokens + 3, bool)
            closed = []
            for request, (width, history) in enumerate(zip(widths, histories)):
                if history >= 0:
                    begin, end = bounds[request : request + 2]
                    positions[begin:end] = np.arange(history, history + width) + (1000 if stale else 0)
                    active[begin:end] = True
                    closed.extend(
                        (request, int(begin) + i, p)
                        for i, p in enumerate(range(history, history + width))
                        if (p + 1) % 4 == 0
                    )
            count = len(closed)
            pad = ((count + 15) // 16) * 16 + 16
            offsets = np.zeros(len(widths), np.int32)
            namespace["compact_offsets_stage"](bounds, lengths, offsets, len(positions), 0)
            pooled = np.full((tokens + 5, 128), np.nan, np.float32)
            pooled[: len(active)][active] = rng.normal(size=(int(active.sum()), 128)).astype(np.float32)
            gamma = bf16(rng.normal(size=128))
            cos = rng.normal(size=(count, 64)).astype(np.float32)
            sin = rng.normal(size=(count, 64)).astype(np.float32)
            expected = np.full((pad, 128), -777.0, np.float32)
            expected[:count] = 0
            if not stale:
                for c, (_, token, _) in enumerate(closed):
                    row = pooled[token]
                    square = row[:64] * row[:64] + row[64:] * row[64:]
                    variance = np.sum(square, dtype=np.float32) * np.float32(1.0 / 128) + np.float32(
                        model["rms_norm_eps"]
                    )
                    inv = np.reciprocal(np.sqrt(variance))
                    nope = (row[:64] * inv) * gamma[:64]
                    rope = (row[64:] * inv) * gamma[64:]
                    rotated = rope * cos[c] + rope[np.arange(64) ^ 1] * (
                        sin[c] * (np.arange(64) % 2 * 2 - 1).astype(np.float32)
                    )
                    expected[c] = np.concatenate((bf16(nope), bf16(rotated)))
            for workers in (1, 3, 24):
                output = np.full((pad, 128), -777.0, np.float32)
                accesses.clear()
                stores.clear()
                row_masks.clear()
                row_masks[id(pooled)] = np.r_[active, np.zeros(len(pooled) - len(active), bool)]
                row_masks[id(positions)] = active
                for worker in range(workers):
                    norm(pooled, gamma, cos, sin, positions, bounds, lengths, offsets, output, worker, workers)
                np.testing.assert_array_equal(output, expected)
                assert np.isfinite(output[:count]).all()
                assert all(start[0] < count for identity, start, _ in stores if identity == id(output))
                oracle_cases += 1
    had_geometry = 0
    for count in (0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 55, 56, 57, 127, 128, 129):
        blocks = (count + 15) // 16
        for workers in (1, 3, 24):
            covered = Counter()
            for worker in range(workers):
                for logical in range(worker, 2 * blocks, workers):
                    column, block = logical // blocks, logical % blocks
                    rows = min(16, count - block * 16)
                    assert rows > 0 and block * 16 + rows <= count
                    covered.update((row, column) for row in range(block * 16, block * 16 + rows))
            assert len(covered) == count * 2 and all(n == 1 for n in covered.values())
            had_geometry += 1
    return {
        "norm_math_assignments_ast_equal": norm_names,
        "hadamard_assignments_ast_equal": had_names,
        "normalization_source_ast_executed": True,
        "oracle_cases": oracle_cases,
        "hadamard_geometry_cases": had_geometry,
        "dummy_and_uncovered_pooled_not_read": True,
        "compact_padding_unchanged": True,
        "stale_positions_owned_compact_rows_zero": True,
        "floating_point_model": "NumPy FP32及显式BF16 RNE，仅CPU数学/地址oracle，不是Ascend精度结论",
        "device_execution": False,
    }
