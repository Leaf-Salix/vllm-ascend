# SPDX-License-Identifier: Apache-2.0
"""Execute frozen HC source over CPU tensors and audit its actual wait expressions.

This validates arithmetic ordering, addresses and ownership. NumPy row sums and
rsqrt are not an A3 numerical oracle; device bit-equality remains mandatory.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/hc_cooperative_v2"
PARENT = HERE / "runs/physical_capacity_production_v1"


class Tensor:
    def __init__(self, data, valid=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.valid = tuple(valid or self.data.shape)

    def __setitem__(self, index, value):
        self.data[index] = value.data


def tile_slice(source, shape, offsets, valid_shape=None, **_kwargs):
    valid = valid_shape or shape
    result = np.zeros(shape, np.float32)
    src = tuple(slice(start, start + length) for start, length in zip(offsets, valid))
    dst = tuple(slice(0, length) for length in valid)
    result[dst] = source.data[src]
    return Tensor(result, valid)


def assemble(destination, value, offsets):
    target = tuple(slice(start, start + length) for start, length in zip(offsets, value.valid))
    valid = tuple(slice(0, length) for length in value.valid)
    destination.data[target] = value.data[valid]
    return destination


def fillpad(value, **_kwargs):
    result = np.zeros_like(value.data)
    valid = tuple(slice(0, length) for length in value.valid)
    result[valid] = value.data[valid]
    return Tensor(result)


def scalar_or_tensor(value):
    return value.data if isinstance(value, Tensor) else np.float32(value)


def cpu_language():
    return SimpleNamespace(
        tensor=SimpleNamespace(dim=lambda value, axis: value.data.shape[axis]),
        range=range,
        pipeline=lambda *args, **_kw: range(*args),
        min=min,
        full=lambda shape, value, **_kw: Tensor(np.full(shape, value, np.float32)),
        slice=tile_slice,
        load=lambda source, offsets, shape, **kw: tile_slice(source, shape, offsets, **kw),
        store=lambda value, offsets, dest: assemble(dest, value, offsets),
        assemble=assemble,
        cast=lambda value, _dtype: Tensor(value.data.copy()),
        mul=lambda left, right: Tensor(scalar_or_tensor(left) * scalar_or_tensor(right)),
        add=lambda left, right: Tensor(scalar_or_tensor(left) + scalar_or_tensor(right)),
        row_sum=lambda value: Tensor(np.sum(value.data, axis=1, keepdims=True, dtype=np.float32)),
        reshape=lambda value, shape: Tensor(value.data.reshape(shape)),
        set_validshape=lambda value, *shape: Tensor(value.data, shape),
        fillpad=fillpad,
        rsqrt=lambda value, **_kw: Tensor(np.float32(1) / np.sqrt(value.data)),
        FP32=np.float32,
        PadValue=SimpleNamespace(zero=0),
        MemorySpace=SimpleNamespace(Vec="Vec"),
    )


def functions(path, names):
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            nodes.append(node)
    assert {node.name for node in nodes} == set(names)
    scope = dict(
        pl=cpu_language(),
        T_TILE=8,
        HC_MULT=4,
        D=4096,
        HC_DIM=16384,
        HC_DIM_INV=1.0 / 16384,
        NORM_EPS=1e-6,
        RMS_K_TILE=512,
        LINEAR_OK=4,
        LINEAR_K_PER_SPLIT=4096,
        RMS_SLOT_WORDS=128,
        RMS_CHUNKS_PER_SPLIT=8,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return [scope[name] for name in names]


@pytest.mark.parametrize("tokens", [1, 6, 8, 9, 16, 48, 96, 144, 192])
def test_source_math_preserves_order_and_tail(tokens):
    (original,) = functions(PARENT / "hc_stages.py", ["hc_widen_rms_range"])
    widen, combine = functions(SOURCE / "hc_cooperative_stages.py", ["hc_widen_columns", "hc_combine_rms"])
    rng = np.random.default_rng(tokens)
    data = rng.standard_normal((tokens, 16384), dtype=np.float32)
    # Values spanning exponents make reassociated partial sums observable.
    data *= np.exp2(rng.integers(-8, 8, size=data.shape)).astype(np.float32)
    data.view(np.uint32)[:] &= np.uint32(0xFFFF0000)  # Exact BF16-representable input.
    x = Tensor(data)
    reference = Tensor(np.full_like(data, np.nan))
    candidate = Tensor(np.full_like(data, np.nan))
    padded_rows = (tokens + 15) // 16 * 16
    reference_inv = Tensor(np.full((padded_rows, 1), 37, np.float32))
    candidate_inv = Tensor(reference_inv.data.copy())
    old_scratch = Tensor(np.full((8, 16384), 37, np.float32))
    new_scratch = Tensor(old_scratch.data.copy())
    for worker in rng.permutation(48):
        original(x, reference, reference_inv, old_scratch, int(worker), 48)
        widen(x, candidate, new_scratch, int(worker), 48)
    for worker in rng.permutation(48):
        combine(new_scratch, candidate_inv, tokens, int(worker), 48)
    assert np.array_equal(reference.data.view(np.uint32), candidate.data.view(np.uint32))
    assert np.array_equal(reference_inv.data.view(np.uint32), candidate_inv.data.view(np.uint32))
    written = np.zeros(new_scratch.data.shape, dtype=bool)
    for row_tile in range((tokens + 7) // 8):
        written[:4, row_tile * 128 : row_tile * 128 + 64] = True
    assert np.all(new_scratch.data[~written] == 37)


def schedule_prefixes():
    cube = (SOURCE / "cube_schedule.py").read_text()
    cube = cube[
        cube.index("    hc_tasks = ((tokens + 15) // 16) * 4\n") : cube.index("    partials = hc_project_range(")
    ]
    vector = (SOURCE / "vector_schedule.py").read_text()
    vector = vector[vector.index("    # INPUT now publishes") : vector.index("    # 公共metadata")]
    return [
        compile("\n".join(line[4:] for line in code.splitlines()), "frozen_schedule", "exec") for code in (cube, vector)
    ]


def run_prefix(code, tokens, worker):
    waited, published = set(), set()

    def wait(*args):
        _signal, _phase, _cores, _gen, first, rows, tile_rows, workers, _base, _stride = args
        assert tile_rows == 1
        waited.update(row % workers for row in range(first, first + min(rows, workers)))

    scope = dict(
        tokens=tokens,
        worker=worker,
        core=24 + worker,
        VECTORS=48,
        CORES=72,
        GROUPS=24,
        signal=None,
        generation=1,
        PHASE_INPUT=1,
        pl=SimpleNamespace(min=min),
        wait_row_owners=wait,
        publish_phase=lambda *_args: published.add(worker),
        hc_combine_rms=lambda *_args: None,
        widen_tail=None,
        inv=None,
    )
    exec(code, scope)
    return waited, published


def test_actual_schedule_dependency_coverage_and_slots():
    cube, vector = schedule_prefixes()
    for tokens in range(1, 1025):
        row_tiles = (tokens + 7) // 8
        writers = {work % 48 for work in range(4 * row_tiles)}
        all_published = set()
        for worker in range(48):
            expected_rms = {(block * 4 + split) % 48 for block in range(worker, row_tiles, 48) for split in range(4)}
            waited, published = run_prefix(vector, tokens, worker)
            assert waited == expected_rms <= writers
            all_published.update(published)
        assert all_published == writers
        for worker in range(24):
            expected = set()
            for task in range(worker, 4 * ((tokens + 15) // 16), 24):
                rb, split = divmod(task, 4)
                for block in (2 * rb, 2 * rb + 1):
                    if block < row_tiles:
                        expected.add((block * 4 + split) % 48)
            waited, _ = run_prefix(cube, tokens, worker)
            assert waited == expected <= writers
        # Scratch slots are 512-byte aligned and never shared by two owners.
        slots = {(split * 16384 + block * 128) * 4 for block in range(row_tiles) for split in range(4)}
        assert len(slots) == row_tiles * 4
        assert all(slot % 512 == 0 for slot in slots)
        assert max(slots) + 512 <= 8 * 16384 * 4
    # The very next row tile does not fit within one split's physical row.
    assert ((1025 + 7) // 8 - 1) * 128 + 64 > 16384


def test_unrelated_modules_and_original_math_unchanged():
    manifest = json.loads((HERE / "hc_cooperative_v2.json").read_text())
    for path in PARENT.glob("*.py"):
        if path.name not in manifest["changed_modules"]:
            assert path.read_bytes() == (SOURCE / path.name).read_bytes()
    assert (SOURCE / "hc_stages.py").read_bytes() == (PARENT / "hc_stages.py").read_bytes()
