# SPDX-License-Identifier: Apache-2.0
"""Execute the actual proof/reader branches and preserve all original math AST."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/plan_gather_valid_v1"
BASE = HERE / "runs/merged_schedule_v1"


def function(root, file, name):
    return next(
        n for n in ast.parse((root / file).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def proof_node():
    fn = function(SOURCE, "attention_plan_stages.py", "attention_slot_plan_stage")
    return next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.If)
        and any(isinstance(child, ast.For) and ast.unparse(child.target) == "proof_block" for child in n.body)
    )


def reader_node():
    fn = function(SOURCE, "attention_stream_stages.py", "attention_qk_pv_vector_lane_stage")
    return next(n for n in ast.walk(fn) if isinstance(n, ast.If) and "COMPRESSED_ALL_VALID_COL0" in ast.unparse(n.test))


def apply_proof(mask, metadata, first, rows, *, enabled=True, physical=1):
    pl = SimpleNamespace(
        range=range,
        INT32=np.int32,
        slice=lambda a, shape, offset, valid_shape: a[
            offset[0] : offset[0] + shape[0], offset[1] : offset[1] + shape[1]
        ],
        row_min=lambda a, tmp: np.min(a, axis=1, keepdims=True),
        cast=lambda value, dtype: dtype(value),
        read=lambda a, offset: a[tuple(offset)],
        write=lambda a, offset, value: a.__setitem__(tuple(offset), value),
    )
    namespace = dict(
        pl=pl,
        c_mask_tile=mask,
        valid_blocks=metadata,
        first=first,
        rows=rows,
        PLAN_PHYSICAL_ROWS=enabled,
        physical_mode=physical,
        ROWS=8,
        ATTN_K_TILE=128,
        SPARSE_BLOCKS=5,
        COMPRESSED_ALL_VALID_COL0=7,
        reduce_tmp=None,
    )
    exec(compile(ast.Module(body=[proof_node()], type_ignores=[]), "actual_plan_proof", "exec"), namespace)


@pytest.mark.parametrize("rows", range(1, 9))
def test_proof_soundness_overwrite_and_tail_guards(rows):
    rng = np.random.default_rng(333 + rows)
    metadata = rng.integers(11, 99, size=(24, 16), dtype=np.int32)
    original = metadata.copy()
    patterns = [
        np.ones((8, 512), np.float32),
        rng.integers(0, 2, size=(8, 512)).astype(np.float32),
        np.zeros((8, 512), np.float32),
        np.ones((8, 512), np.float32),
    ]
    for mask in patterns:
        apply_proof(mask, metadata, 8, rows)
        expected = original.copy()
        expected[8 : 8 + rows, 7:11] = mask[:rows].reshape(rows, 4, 128).min(axis=2).astype(np.int32)
        np.testing.assert_array_equal(metadata, expected)


@pytest.mark.parametrize("enabled,physical", [(False, 0), (False, 1), (True, 0)])
def test_disabled_or_unsupported_plan_has_no_new_writes(enabled, physical):
    metadata = np.full((8, 16), 37, np.int32)
    apply_proof(np.ones((8, 512), np.float32), metadata, 0, 8, enabled=enabled, physical=physical)
    assert np.all(metadata == 37)


@pytest.mark.parametrize("bad_row", [None, 0, 63, 64, 127])
@pytest.mark.parametrize("bad_index", [-1, 256, 2**31 - 1])
def test_actual_reader_proof_and_bitwise_fallback(bad_row, bad_index):
    rng = np.random.default_rng(144)
    cache = rng.integers(0, 65535, size=(256, 512), dtype=np.uint16)
    indices = rng.integers(0, 256, size=128, dtype=np.int32)
    if bad_row is not None:
        indices[bad_row] = bad_index
    valid = (indices >= 0) & (indices < len(cache))
    mask = np.ones((8, 512), np.float32)
    mask[0, :128] = valid.astype(np.float32)
    metadata = np.full((8, 16), 37, np.int32)
    apply_proof(mask, metadata, 0, 1)
    modes = []

    def mgather(mem, idx, *, coalesce, gather_oob):
        assert coalesce == "row"
        modes.append(gather_oob)
        idx = idx.ravel()
        ok = (idx >= 0) & (idx < len(mem))
        if gather_oob == "undefined":
            assert np.all(ok), "Unproven address reached the unchecked reader"
        out = np.zeros((len(idx), 512), dtype=np.uint16)
        out[ok] = mem[idx[ok]]
        return out

    pl = SimpleNamespace(read=lambda a, ix: a[tuple(ix)], tile=SimpleNamespace(mgather=mgather))
    compiled = compile(ast.Module(body=[reader_node()], type_ignores=[]), "actual_proven_reader", "exec")
    outputs = []
    for lane in range(2):
        namespace = dict(
            pl=pl,
            valid_block_mask=metadata,
            qk_t=0,
            qk_sb=1,
            COMPRESSED_ALL_VALID_COL0=7,
            cmp_kv_flat=cache,
            physical_rows=indices[lane * 64 : (lane + 1) * 64][None, :],
        )
        exec(compiled, namespace)
        outputs.append(namespace["qk_kv_half"])
    expected = np.zeros((128, 512), np.uint16)
    expected[valid] = cache[indices[valid]]
    np.testing.assert_array_equal(np.concatenate(outputs), expected)
    assert modes == (["undefined", "undefined"] if bad_row is None else ["zero", "zero"])


def test_all_existing_math_and_fifo_are_unchanged():
    class RemoveProof(ast.NodeTransformer):
        def visit_If(self, node):
            if any(isinstance(n, ast.For) and ast.unparse(n.target) == "proof_block" for n in node.body):
                return None
            if "COMPRESSED_ALL_VALID_COL0" in ast.unparse(node.test):
                return node.orelse
            return self.generic_visit(node)

    for file, name in (
        ("attention_plan_stages.py", "attention_slot_plan_stage"),
        ("attention_stream_stages.py", "attention_qk_pv_vector_lane_stage"),
        ("attention_stream_stages.py", "attention_qk_pv_cube_stage"),
    ):
        actual = RemoveProof().visit(function(SOURCE, file, name))
        assert ast.dump(actual) == ast.dump(function(BASE, file, name))
