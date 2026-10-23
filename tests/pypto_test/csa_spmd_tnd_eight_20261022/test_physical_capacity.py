# SPDX-License-Identifier: Apache-2.0
"""Execute the actual extended physical-plan address body at table/page boundaries."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
BASE = HERE / "runs/merged_schedule_v1"
SOURCE = HERE / "runs/physical_capacity_2080_v1"


def module():
    return ast.parse((SOURCE / "attention_plan_stages.py").read_text())


def function(name):
    return next(node for node in module().body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_only_capacity_changes_in_the_frozen_operator():
    old = {p.name: p.read_text() for p in BASE.glob("*.py")}
    new = {p.name: p.read_text() for p in SOURCE.glob("*.py")}
    assert old.keys() == new.keys()
    name = "attention_plan_stages.py"
    assert new[name].replace("PHYSICAL_PLAN_COLUMNS = 2080", "PHYSICAL_PLAN_COLUMNS = 2048", 1) == old[name]
    assert all(new[key] == value for key, value in old.items() if key != name)


@pytest.mark.parametrize("columns", [0, 1, 2048, 2049, 2050, 2080, 2081])
@pytest.mark.parametrize("pages", [0, 1, 32801, 16777216, 16777217])
def test_shared_producer_consumer_support_guard(columns, pages):
    fn = copy.deepcopy(function("physical_plan_supported"))
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    namespace = dict(
        pl=SimpleNamespace(cast=lambda value, dtype: int(value), INT32=int),
        PHYSICAL_PLAN_COLUMNS=2080,
        EXACT_FP32_INT=16777216,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "support_guard", "exec"), namespace)
    for enabled in (False, True):
        actual = namespace["physical_plan_supported"](columns, pages, enabled)
        assert actual == int(enabled and 0 < columns <= 2080 and 0 < pages <= 16777216)


@pytest.mark.parametrize("columns", [2048, 2049, 2050, 2080])
@pytest.mark.parametrize("pages", [32801, 16777216])
@pytest.mark.parametrize("causal_tail", [False, True])
def test_actual_address_math_and_selected_tail_pages(columns, pages, causal_tail):
    fn = function("attention_slot_plan_stage")
    branch = next(
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "p_raw" for t in child.targets)
            for child in node.body
        )
    )
    code = compile(ast.Module(body=copy.deepcopy(branch.body), type_ignores=[]), "actual_physical_plan", "exec")
    rng = np.random.default_rng(2080 + columns)
    table = rng.integers(0, pages, (1, columns), dtype=np.int32)
    # Exercise the final supported physical page and both invalid page sentinels.
    table[0, 0], table[0, 1], table[0, 2] = pages - 1, -1, pages
    upper = min(65537, columns * 32) if causal_tail else columns * 32
    logical = rng.integers(-10, columns * 32 + 64, 512, dtype=np.int32)
    logical[:17] = [
        -1,
        0,
        31,
        32,
        63,
        64,
        95,
        65503,
        65504,
        65535,
        65536,
        65537,
        65567,
        65568,
        columns * 32 - 1,
        columns * 32,
        2**31 - 1,
    ]
    output = np.full((3, 512), 0x12345678, np.int32)
    validity = np.full((8, 512), -7, np.float32)

    def load(array, offset, shape, valid_shape, **kwargs):
        assert shape == [1, 2080] and valid_shape == [1, columns]
        result = np.full(shape, -999999, array.dtype)
        result[0, :columns] = array[offset[0], :columns]
        return result

    def assemble(dst, src, offset):
        result = dst.copy()
        result[offset[0] : offset[0] + src.shape[0], offset[1] : offset[1] + src.shape[1]] = src
        return result

    def store(src, offset, dst):
        dst[offset[0] : offset[0] + src.shape[0], offset[1] : offset[1] + src.shape[1]] = src

    def cast(value, target_type, **kwargs):
        return np.asarray(value).astype(target_type)

    tile = SimpleNamespace(
        maximums=np.maximum,
        minimums=np.minimum,
        shrs=np.right_shift,
        shls=np.left_shift,
        sub=np.subtract,
        add=np.add,
        assemble=assemble,
        gather=lambda data, indices, scratch: data.ravel()[indices],
    )
    pl = SimpleNamespace(
        tile=tile,
        tensor=SimpleNamespace(dim=lambda array, axis: array.shape[axis]),
        slice=lambda array, shape, offset, **kwargs: array[
            offset[0] : offset[0] + shape[0], offset[1] : offset[1] + shape[1]
        ],
        load=load,
        store=store,
        cast=cast,
        INT32=np.int32,
        FP32=np.float32,
        MemorySpace=SimpleNamespace(Vec=None),
        create_tile=lambda shape, **kwargs: np.empty(shape, np.int32),
        add=np.add,
        sub=np.subtract,
        mul=np.multiply,
        neg=np.negative,
        minimum=np.minimum,
        maximum=np.maximum,
    )
    namespace = dict(
        pl=pl,
        TOPK=512,
        PHYSICAL_PLAN_COLUMNS=2080,
        c_page_indices=logical[None, :],
        row=0,
        token=1,
        request=0,
        upper=upper,
        compressed_table=table,
        compressed_pages=pages,
        c_pages_valid=validity,
        indices=output,
    )
    exec(code, namespace)
    expected = np.full(512, -1, np.int32)
    for i, raw in enumerate(logical):
        raw = int(raw)
        if 0 <= raw < upper and raw // 32 < columns:
            page = int(table[0, raw // 32])
            if 0 <= page < pages:
                expected[i] = page * 32 + raw % 32
    np.testing.assert_array_equal(output[1], expected)
    np.testing.assert_array_equal(namespace["c_pages_valid"][0], (expected >= 0).astype(np.float32))
    assert np.all(output[[0, 2]] == 0x12345678)
    assert np.all(namespace["c_pages_valid"][1:] == -7)
