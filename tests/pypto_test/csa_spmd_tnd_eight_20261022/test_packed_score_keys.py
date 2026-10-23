# SPDX-License-Identifier: Apache-2.0
"""Execute the actual packing helper on bytes and model slot lifetime."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/packed_score_keys_v6"
PARENT = HERE / "runs/output_rectangle_production_v1"


class StripTypes(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.decorator_list = []
        node.returns = None
        return self.generic_visit(node)

    def visit_arg(self, node):
        node.annotation = None
        return node


def packing_helper(name="supply_packed_score_keys"):
    events = []

    def load(array, offsets, shape):
        row, col = offsets
        height, width = shape
        assert row >= 0 and row + height <= array.shape[0]
        assert col + width <= array.shape[1]
        return array[row : row + height, col : col + width].copy()

    def store(value, offsets, target):
        row, col = offsets
        target[row : row + value.shape[0], col : col + value.shape[1]] = value
        return target

    def gatherb(value, offsets):
        raw = value.view(np.uint8).reshape(-1, 32)
        assert np.all(offsets % 32 == 0)
        return raw[offsets.ravel() // 32].copy().reshape(1, -1).view(value.dtype)

    pl = SimpleNamespace(
        INT8=np.int8,
        FP16=np.float16,
        INT32=np.int32,
        UINT32=np.uint32,
        INDEX=int,
        cast=lambda value, dtype: dtype(value),
        min=min,
        max=max,
        range=range,
        read=lambda value, indices: value[tuple(indices)],
        load=load,
        store=store,
        reshape=np.reshape,
        sub=np.subtract,
        mul=np.multiply,
        add=np.add,
        neg=np.negative,
        yield_=lambda value: value,
        tile=SimpleNamespace(
            reinterpret_view=lambda value, dtype: value.view(dtype),
            ci=lambda start, shape, dtype: np.arange(start, start + np.prod(shape), dtype=dtype).reshape(shape),
            shrs=np.right_shift,
            gatherb=gatherb,
        ),
        tensor=SimpleNamespace(
            dim=lambda value, axis: value.shape[axis], reinterpret_view=lambda value, dtype: value.view(dtype)
        ),
        system=SimpleNamespace(sync_set=lambda *args, **kwargs: events.append((args, kwargs))),
        PipeType=SimpleNamespace(MTE3="MTE3"),
        KernelType=SimpleNamespace(AIV="AIV"),
    )
    tree = ast.parse((SOURCE / "packed_score_keys.py").read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) or (isinstance(node, ast.Assign) and not isinstance(node.value, ast.Call))
    ]
    tree = ast.fix_missing_locations(StripTypes().visit(tree))
    scope = {"pl": pl}
    exec(compile(tree, str(SOURCE / "packed_score_keys.py"), "exec"), scope)
    return scope[name], events


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("worker,step,lane", [(0, 0, 0), (0, 0, 1), (13, 1, 0), (23, 3, 1)])
def test_actual_helper_preserves_key_bytes_and_only_own_half(direction, worker, step, lane):
    fn, events = packing_helper()
    rng = np.random.default_rng(1024)
    native = rng.integers(-128, 128, size=(256, 4160), dtype=np.int8)
    arena = np.full((288, 8192), 0xA5A5A5A5, dtype=np.uint32).view(np.float32)
    before = arena.view(np.uint8).copy()
    first_page = 200 if direction < 0 else 20
    ring = arena.view(np.int8).reshape(-1, 128)
    result = fn(native, ring, worker, lane, step, first_page, direction)
    indices = first_page + direction * ((step * 2 + lane) * 16 + np.arange(16))
    expected = native[indices, :4096].reshape(512, 128)
    first = (worker * 2 + step % 2) * 1024 + lane * 512
    np.testing.assert_array_equal(ring[first : first + 512], expected)
    before.view(np.int8).reshape(-1, 128)[first : first + 512] = expected
    np.testing.assert_array_equal(arena.view(np.uint8), before)
    assert result is ring and len(events) == 1
    assert events[0] == ((6,), {"pipe": "MTE3", "ffts_mode": 2, "core_type": "AIV"})


def calls(source, function, names):
    tree = ast.parse(source)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
    selected = [node for node in ast.walk(fn) if isinstance(node, ast.Call) and ast.unparse(node.func) in names]
    return [
        ast.dump(node, include_attributes=False)
        for node in sorted(selected, key=lambda node: (node.lineno, node.col_offset))
    ]


def test_original_cube_arithmetic_and_vector_order_unchanged():
    parent = (PARENT / "indexer_score_stages.py").read_text()
    candidate = (SOURCE / "indexer_score_stages.py").read_text()
    for fn, names in (
        ("_indexer_score_cube_split_store", {"pl.tile.matmul_acc", "pl.tile.assemble", "pl.store"}),
        (
            "indexer_score_vector_lane_stage",
            {"pl.mul", "pl.cast", "pl.sort32", "pl.mrgsort", "threshold_merge_top512", "pl.store"},
        ),
    ):
        # Extra Scalar casts only select the new path; numerical casts retain
        # original expressions and order after the two predicate casts are removed.
        actual = calls(candidate, fn, names)
        if fn == "indexer_score_vector_lane_stage":
            actual = [
                call
                for call in actual
                if call
                not in {
                    ast.dump(ast.parse("pl.cast(0, pl.INT32)", mode="eval").body, include_attributes=False),
                    ast.dump(ast.parse("pl.cast(1, pl.INT32)", mode="eval").body, include_attributes=False),
                }
            ]
            expected = [
                call
                for call in calls(parent, fn, names)
                if call
                not in {
                    ast.dump(ast.parse("pl.cast(0, pl.INT32)", mode="eval").body, include_attributes=False),
                    ast.dump(ast.parse("pl.cast(1, pl.INT32)", mode="eval").body, include_attributes=False),
                }
            ]
        else:
            expected = calls(parent, fn, names)
        assert actual == expected


def test_business_roots_unchanged_and_alias_is_byte_equivalent():
    original = json.loads((HERE / "descriptors/C04_output_rectangle_v1.json").read_text())
    packed = json.loads((HERE / "descriptors/C04_packed_score_keys_v6.json").read_text())
    alias = packed["buffers"].pop("packed_keys")
    assert original["buffers"] == packed["buffers"]
    assert np.prod(alias["shape"]) == np.prod(original["buffers"]["score_arena"]["shape"]) * 4
    assert alias["dtype"] == "torch.int8"
    assert 'self.buffers["score_arena"].view(torch.int8).view(-1, 128)' in (SOURCE / "binding.py").read_text()
    assert original["policy"] == {k: v for k, v in packed["policy"].items() if k != "PACKED_SCORE_KEYS"}


def test_reuse_existing_exhaustive_key_slot_model():
    directory = (
        HERE.parent
        / "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/key_prefetch_design"
    )
    sys.path.insert(0, str(directory))
    import key_prefetch_protocol_model as model

    # The real helper does not share scale at all. Reuse the established two-
    # lane key/Score state machine; this is protocol evidence, not DMA evidence.
    cases = [[count] for count in range(1, 9)] + [[1, 1], [1, 2], [2, 1], [3, 3], [1, 3, 1], [0, 2, 0, 3], [6, 5]]
    results = [model.explore(case, shared_scale=False) for case in cases]
    assert all(row["passed"] and row["terminal_states"] == 1 for row in results)
    assert all(max(row["max_pending_key_per_lane"]) <= 2 for row in results)


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("pages", [1, 32, 160, 192, 256])
def test_scalar_proof_agrees_with_full_page_integer_oracle(direction, pages):
    proof, _ = packing_helper("packed_leaf_layout_scalar")
    values = (300 + direction * np.arange(pages)).astype(np.int32)
    table = np.pad(values, (7, 9), constant_values=-1)[None, :]
    expected_direction = direction if pages > 1 else 1
    assert proof(table, 0, 7, pages, 1024) == (300, expected_direction)
    assert proof(table, -1, 7, pages, 1024) == (0, 0)
    assert proof(table, 0, -1, pages, 1024) == (0, 0)
    assert proof(table, 0, 7, pages, 16777217) == (0, 0)
    if pages > 2:
        table[0, 7 + pages // 2] += 1
        assert proof(table, 0, 7, pages, 1024) == (0, 0)
