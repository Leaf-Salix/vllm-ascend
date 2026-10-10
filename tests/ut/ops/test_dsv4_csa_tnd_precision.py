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
    "block,positions,expected_shapes",
    [
        (0, [126, 127, 126], [[(8, 512)], [(64, 8), (8, 8)], [(8, 512)]]),
        (1, [126, 127, 126], [[(8, 512)], [(512, 8), (64, 8), (8, 8)], [(8, 512)]]),
    ],
)
def test_native_sum_guard_reads_live_window_boundary(block, positions, expected_shapes):
    """Execute the real guard across updates; hardware validates reduction bits."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision import decode_sparse_attn_csa

    tree = ast.parse(Path(decode_sparse_attn_csa.__file__).read_text())
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Call)
        and isinstance(node.test.left.func, ast.Attribute)
        and node.test.left.func.attr == "read"
        and isinstance(node.test.left.args[0], ast.Name)
        and node.test.left.args[0].id == "position_ids"
    ]
    assert len(guards) == 1
    code = compile(ast.fix_missing_locations(ast.Module(body=guards, type_ignores=[])), "live_sum_guard.py", "exec")
    reductions = []

    def row_sum(value, temporary):
        reductions.append(value.shape)
        return value.sum(axis=1, keepdims=True, dtype=np.float32)

    scalar_api = SimpleNamespace(
        read=lambda tensor, indices: int(tensor[tuple(indices)]),
        add=np.add,
        reshape=np.reshape,
        row_sum=row_sum,
        create_tile=lambda shape, dtype: np.empty(shape, dtype=dtype),
        FP32=np.float32,
    )
    values = np.zeros((8, 512), dtype=np.float32)
    values[:, : 128 if block == 0 else 512] = 1
    live_positions = np.zeros((1, 1), dtype=np.int64)
    namespace = {
        "pl": scalar_api,
        "WIN": 128,
        "SOFTMAX_HEAD_TILE": 8,
        "position_ids": live_positions,
        "qk_t": 0,
        "qk_sb": block,
        "sm_exp": values,
        "qk_reduce_tmp": np.empty_like(values),
    }
    for position, shapes in zip(positions, expected_shapes):
        live_positions[0, 0] = position
        reductions.clear()
        exec(code, namespace)
        assert reductions == shapes
        np.testing.assert_array_equal(namespace["sm_block_sum"], values.sum(axis=1, keepdims=True))
