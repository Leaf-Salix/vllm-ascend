# SPDX-License-Identifier: Apache-2.0
"""Independent TND precision selection and real variable request eligibility."""

import pytest
from test_dsv4_csa_service_perf_tnd import _context, _hidden

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision.service import CSAServiceRuntime
from vllm_ascend.ops.pypto.variant import selected_variant, variant_package


@pytest.mark.parametrize("lengths", [[6, 3, 1, 1], [1, 2, 3, 4, 5, 6], [6] * 4])
def test_precision_tnd_keeps_real_packed_requests(lengths):
    runtime, context = _context(lengths, sum(lengths))
    runtime.__class__ = CSAServiceRuntime
    assert runtime.eligible(context, *_hidden(sum(lengths)))


@pytest.mark.parametrize(
    "variant,package",
    [
        ("performance", "deepseek_v4_flash_csa"),
        ("precision", "deepseek_v4_flash_dspark"),
        ("tnd_precision", "deepseek_v4_flash_csa_tnd_precision"),
    ],
)
def test_variants_have_independent_packages(monkeypatch, variant, package):
    monkeypatch.setenv("PTO_CSA_VARIANT", variant)
    assert selected_variant() == variant
    assert variant_package().endswith("." + package)


def test_tnd_precision_defaults_to_fixed_reduction(monkeypatch):
    from vllm_ascend.envs import _pto_csa_atomic_add

    monkeypatch.setenv("PTO_CSA_VARIANT", "tnd_precision")
    monkeypatch.delenv("VLLM_ASCEND_PTO_CSA_ATOMIC_ADD", raising=False)
    assert _pto_csa_atomic_add() == 0


def test_precision_kernel_returns_external_output():
    """Catch registration alias errors before hardware allocation."""
    from pypto.ir._kernel_compile import kernel_signature_for_program

    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision.decode_csa import decode_csa_tp1_layer_test

    params, aliases = kernel_signature_for_program(decode_csa_tp1_layer_test.specialize())
    assert aliases
    assert all(params[index].name == "native_heads" for index in aliases)


def test_precision_rejects_atomic_reduction(monkeypatch):
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision import reduction

    monkeypatch.setattr(reduction, "ATOMIC_ADD", 1)
    with pytest.raises(ValueError, match="ATOMIC_ADD=0"):
        reduction.validate_reduction_mode()


def test_precision_requires_native_hc_layer():
    with pytest.raises(ValueError, match="Native decoder layer"):
        CSAServiceRuntime(None, None, 4)


@pytest.mark.parametrize(
    "candidate,request_index,position,expected",
    [
        (3, 0, 15, 3),
        (-1, 0, 15, -1),
        (3, -1, 15, -1),
        (3, 2, 15, -1),
        (3, 1, 15, -1),
        (6, 0, 23, -1),
        (32, 0, 255, -1),
        (96, 0, 511, -1),
    ],
)
def test_compressed_index_rejects_inactive_and_negative_pages(candidate, request_index, position, expected):
    """Execute the actual Scalar helper with CPU table reads, without a device."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision import decode_sparse_attn_csa

    source = Path(decode_sparse_attn_csa.__file__).read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "checked_compressed_index"
    )
    function.decorator_list = []
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    scalar_api = SimpleNamespace(
        cast=lambda value, dtype: int(value),
        read=lambda tensor, indices: int(tensor[tuple(indices)]),
        tensor=SimpleNamespace(dim=lambda tensor, axis: tensor.shape[axis]),
        INT32=None,
        INDEX=None,
    )
    namespace = {"pl": scalar_api, "COMPRESS_RATIO": 4, "BLOCK_SIZE": 32}
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, "compressed_index_scalar.py", "exec"), namespace)
    lengths = np.array([16, 0], dtype=np.int32)
    table = np.array([[0, -1, 2], [2, 3, 4]], dtype=np.int32)
    assert namespace["checked_compressed_index"](candidate, request_index, position, lengths, table) == expected


def test_precision_operator_namespace_is_independent(monkeypatch):
    import pypto.torch

    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision import native_adapter, reduction

    names = []
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_ATOMIC_ADD", "0")
    monkeypatch.setattr(reduction, "ATOMIC_ADD", 0)
    monkeypatch.setattr(pypto.torch, "register", lambda kernel, name: names.append(name) or kernel)
    kernel = object()
    operators = native_adapter.CSAOperators.register(kernel)
    assert operators.attention is kernel
    assert names == ["dsv4_csa_tnd_precision::attention"]


def test_precision_rejects_uncalibrated_device(monkeypatch):
    from vllm_ascend import utils

    monkeypatch.setattr(utils, "get_ascend_device_type", lambda: utils.AscendDeviceType.A2)
    with pytest.raises(ValueError, match="requires A3"):
        CSAServiceRuntime(None, None, 4, layer=object())


@pytest.mark.parametrize(
    "lengths,head_dim,expected",
    [([6, 6, 6, 6], 512, 32), ([6, 3, 1, 1], 512, 64), ([6, 6], 128, 16), ([1, 2], 128, 64)],
)
def test_compressor_native_column_group_uses_live_bounds(lengths, head_dim, expected):
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision import compact_metadata

    function = next(
        node
        for node in ast.parse(Path(compact_metadata.__file__).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "native_compressor_column_group"
    )
    function.decorator_list = []
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    scalar_api = SimpleNamespace(
        cast=lambda value, dtype: int(value),
        read=lambda tensor, indices: int(tensor[tuple(indices)]),
        range=range,
        tensor=SimpleNamespace(dim=lambda tensor, axis: tensor.shape[axis]),
        INT32=None,
        INDEX=None,
    )
    namespace = {"pl": scalar_api}
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, "native_compressor_column_group.py", "exec"), namespace)
    bounds = np.array([0, *np.cumsum(lengths)], dtype=np.int32)
    assert namespace["native_compressor_column_group"](bounds, sum(lengths), head_dim) == expected


@pytest.mark.parametrize(
    "block,positions",
    [
        (0, [62, 63, 64, 126, 127, 62]),
        (1, [251, 255, 259, 503, 507, 511, 515, 1019, 1023, 1027, 2043, 2047, 251]),
    ],
)
def test_native_sum_reads_live_actual_key_count(block, positions):
    """Replay the actual AST across all length tiers, including backward updates."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision import decode_sparse_attn_csa

    tree = ast.parse(Path(decode_sparse_attn_csa.__file__).read_text())
    body = next(
        node.body
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "sm_part"
    )
    start = next(
        i for i, node in enumerate(body) if isinstance(node, ast.Assign) and node.targets[0].id == "sm_position"
    )
    end = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign) and node.targets[0].id == "sm_sum")
    code = compile(ast.fix_missing_locations(ast.Module(body=body[start:end], type_ignores=[])), "live_sum.py", "exec")
    reductions, masks, folds = [], [], []

    def row_sum(value, temporary):
        reductions.append(value.shape)
        return value.sum(axis=1, keepdims=True, dtype=np.float32)

    def tile_slice(value, shape, offset, valid_shape=None):
        if valid_shape is not None:
            masks.append(valid_shape[1])
        cols = valid_shape[1] if valid_shape is not None else shape[1]
        return value[:, offset[1] : offset[1] + cols]

    def store(value, offset, target):
        target[:] = value
        return target

    def add(left, right):
        folds.append(right.copy())
        return np.add(left, right)

    scalar_api = SimpleNamespace(
        read=lambda tensor, indices: int(tensor[tuple(indices)]),
        cast=lambda value, dtype: value.astype(dtype)
        if isinstance(value, np.ndarray)
        else dtype(value)
        if dtype
        else int(value),
        min=min,
        add=add,
        reshape=np.reshape,
        row_sum=row_sum,
        unroll=range,
        col_expand_add=np.add,
        store=store,
        load=lambda tensor, offset, shape: tensor.copy(),
        arange=lambda start, shape, dtype: np.arange(start, start + shape[1], dtype=dtype).reshape(shape),
        create_tile=lambda shape, dtype: np.empty(shape, dtype=dtype),
        tile=SimpleNamespace(
            arange=lambda start, shape, dtype: np.arange(start, start + shape[1], dtype=dtype).reshape(shape),
            slice=tile_slice,
            assemble=lambda target, source, offset: source.copy(),
            full=lambda shape, dtype, value: np.full(shape, value, dtype=dtype),
            cmps=lambda value, scalar, cmp_type: value < scalar,
            sel=lambda mask, left, right, tmp: np.where(mask, left, right),
        ),
        MemorySpace=SimpleNamespace(Vec=None),
        FP32=np.float32,
        INT32=np.int32,
        UINT32=np.uint32,
        INDEX=None,
    )
    live_positions = np.zeros((1, 1), dtype=np.int64)
    namespace = {
        "pl": scalar_api,
        "WIN": 128,
        "CMP_TOPK": 512,
        "COMPRESS_RATIO": 4,
        "SOFTMAX_HEAD_TILE": 8,
        "position_ids": live_positions,
        "qk_t": 0,
        "qk_sb": block,
        "sm_row": 0,
        "softmax_copy64": np.zeros((8, 64), dtype=np.float32),
    }
    for position in positions:
        count = min(position + 1, 128) if block == 0 else min((position + 1) // 4, 512)
        values = np.full((8, 512), np.nan, dtype=np.float32)
        values[:, :count] = 1
        namespace["sm_exp"] = values
        live_positions[0, 0] = position
        reductions.clear()
        masks.clear()
        folds.clear()
        exec(code, namespace)
        if count < 64:
            assert masks == [count]
            assert reductions == [(8, count)]
            assert not folds
        elif count < 512:
            assert reductions == [(64, 8), (8, 8)]
            assert len(folds) == (count - 1) // 64
            for i, folded in enumerate(folds, 1):
                np.testing.assert_array_equal(folded, values[:, i * 64 : (i + 1) * 64])
        else:
            assert reductions == [(512, 8), (64, 8), (8, 8)]
            assert not folds
        np.testing.assert_array_equal(namespace["sm_block_sum"], np.full((8, 1), count, dtype=np.float32))


@pytest.mark.parametrize(
    "table,first,second,limit,expected",
    [
        ([1, 2], 7, 2, 64, True),
        ([1, 2], 2, 7, 64, False),
        ([6, 5, 4, 3], 23, 59, 127, True),
        ([6, 5, 4, 3], 59, 23, 127, False),
        ([6, 5, 4, 3], 23, 126, 127, False),
        ([6, 5, 4, 3], 126, 23, 127, False),
        ([1, 1], 39, 7, 64, False),
        ([1, 2], 7, 7, 64, False),
        ([1, 2], -1, 7, 64, False),
        ([1, 2], 7, -1, 64, False),
        ([1, 2], -1, -1, 0, False),
        ([65537, 1], 0, 32, 64, True),
        ([65538, 1], 0, 32, 64, False),
    ],
)
def test_native_compressed_pair_order_uses_physical_pages(table, first, second, limit, expected):
    """Native CopyInKv's merge order includes strict last-index/stride fallbacks."""
    import ast
    import copy
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision import decode_sparse_attn_csa

    tree = ast.parse(Path(decode_sparse_attn_csa.__file__).read_text())
    function = copy.deepcopy(
        next(node for node in tree.body if getattr(node, "name", None) == "native_compressed_pair_swap")
    )
    function.decorator_list = []
    for arg in function.args.args:
        arg.annotation = None
    namespace = {
        "pl": SimpleNamespace(
            cast=lambda value, dtype: int(value),
            read=lambda value, index: int(value[tuple(index)]),
            INT32=int,
            INDEX=int,
        ),
        "BLOCK_SIZE": decode_sparse_attn_csa.BLOCK_SIZE,
        "HEAD_DIM": decode_sparse_attn_csa.HEAD_DIM,
        "KV_ELEMENT_BYTES": decode_sparse_attn_csa.KV_ELEMENT_BYTES,
        "KV_PAIR_MAX_STRIDE_BYTES": decode_sparse_attn_csa.KV_PAIR_MAX_STRIDE_BYTES,
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "native_pair.py", "exec"),
        namespace,
    )
    actual = namespace["native_compressed_pair_swap"](first, second, 0, limit, np.asarray([table], dtype=np.int32))
    assert bool(actual) is expected
