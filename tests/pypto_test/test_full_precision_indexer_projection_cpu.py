# SPDX-License-Identifier: Apache-2.0
"""将正式Indexer投影与已完成的Native三边界/Graph资格绑定，CPU无需vLLM。"""

import ast
import hashlib
import json
import runpy
import tarfile
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_PACKAGE = _ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_full_precision"
_EVIDENCE = Path(__file__).parent / "evidence_tnd_precision/full_pto_20261016"


def test_all_indexer_token_orders_match_native_qualification():
    data = runpy.run_path(str(_PACKAGE / "indexer_order_data.py"))
    raw = (_EVIDENCE / "indexer-native-order-map.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == data["NATIVE_INDEXER_ORDER_MAP_SHA256"]
    reference = json.loads(raw)
    assert len(data["INDEXER_TOKEN_MODES"]) == 256
    assert len(data["INDEXER_TILE_ORDERS"]) == 20
    assert len(data["INDEXER_K_ORDERS"]) == 30
    for order in data["INDEXER_K_ORDERS"]:
        assert sorted(order) == list(range(16))
    for tokens in range(1, 257):
        mode = data["INDEXER_TOKEN_MODES"][tokens - 1]
        indices = data["INDEXER_TILE_ORDERS"][mode]
        assert len(indices) == 16 and all(len(row) == 4 for row in indices)
        actual = [[list(data["INDEXER_K_ORDERS"][index]) for index in row] for row in indices]
        assert actual == reference["modes"][reference["tokens_to_mode"][tokens - 1]]


def test_formal_indexer_kernel_matches_npu_qualified_program():
    archive = _EVIDENCE / "indexer-evidence-v1.tar.gz"
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        == "8787c67d3d7f9b04caa3a596521b1aee94d732b130796d8bbaef3a238af3acab"
    )
    with tarfile.open(archive) as bundle:
        source = bundle.extractfile("indexer-evidence-v1/experiments/v6/indexer_projection.py").read().decode()

    # Immutable raw source remains the historical NPU evidence anchor.
    assert hashlib.sha256(source.encode()).hexdigest() == (
        "8b4246005af95de875a368885b4fad967223644d7dc6046f09f3c33bcfb6d4c2"
    )
    historical = ast.parse(source)
    current = ast.parse((_PACKAGE / "indexer_projection.py").read_text())
    old_fn = next(node for node in historical.body if isinstance(node, ast.FunctionDef))
    new_fn = next(node for node in current.body if isinstance(node, ast.FunctionDef))
    assert old_fn.name == "indexer_projection"
    assert new_fn.name == "_indexer_projection"
    assert [ast.unparse(node) for node in old_fn.decorator_list] == ["pl.jit"]
    assert new_fn.decorator_list == []
    # Only the decorator/name/explicit inline+test aliases changed. All arguments,
    # annotations, numerical constants, K-order reads and BF16/FP16 stores remain.
    old_fn.name = new_fn.name
    old_fn.decorator_list = []
    assert ast.dump(old_fn, include_attributes=False) == ast.dump(new_fn, include_attributes=False)
    assert isinstance(historical.body[0], ast.Expr) and isinstance(historical.body[0].value, ast.Constant)
    assert isinstance(historical.body[0].value.value, str)
    assert isinstance(current.body[0], ast.Expr) and isinstance(current.body[0].value, ast.Constant)
    assert isinstance(current.body[0].value.value, str)
    old_nodes = historical.body[1:]
    new_nodes = current.body[1:]
    assert [ast.dump(node, include_attributes=False) for node in old_nodes] == [
        ast.dump(node, include_attributes=False) for node in new_nodes[:-2]
    ]
    expected_aliases = ast.parse(
        "indexer_projection_entry = pl.jit.inline(auto_scope=False)(_indexer_projection)\n"
        "indexer_projection = pl.jit(_indexer_projection)\n"
    ).body
    assert [ast.dump(node, include_attributes=False) for node in new_nodes[-2:]] == [
        ast.dump(node, include_attributes=False) for node in expected_aliases
    ]


def test_service_replaces_native_indexer_projection_boundaries():
    # Structural routing only. Final source/NPU identity and tokens are checked
    # separately by test_full_precision_current_evidence_cpu.py.
    tree = ast.parse((_PACKAGE / "service.py").read_text())
    runtime = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    forward = next(node for node in runtime.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
    calls = {ast.unparse(node.func) for node in ast.walk(forward) if isinstance(node, ast.Call)}
    root_calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "self.operators.attention"
    ]
    assert len(root_calls) == 1
    assert ast.unparse(root_calls[0]) == "self.operators.attention(*call.core_args)"
    forbidden = {"impl.weights_proj", "DeviceOperator.prepare_dsa_indexer_weights", "impl._forward_o_proj"}
    assert calls.isdisjoint(forbidden)

    root = ast.parse((_PACKAGE / "decode_csa.py").read_text())
    function = next(node for node in root.body if isinstance(node, ast.FunctionDef) and node.name == "_full_csa_tnd")
    assert len(function.args.args) == 62 and function.args.defaults == []
    math_calls = {ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)}
    assert {"indexer_projection_entry", "o_proj_entry"} <= math_calls
    assert not any(name.startswith(("torch.", "torch_npu.", "self.layer.", "impl.")) for name in math_calls)
    assert math_calls.isdisjoint(forbidden)
    bindings = {
        (node.level, node.module, alias.name)
        for node in root.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert (1, "indexer_projection", "indexer_projection_entry") in bindings
    assert (1, "o_proj", "o_proj_entry") in bindings
    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(name, ast.Name) and name.id == "decode_csa_tp1_layer_test" for name in node.targets)
        and ast.unparse(node.value) == "pl.jit(auto_scope=False)(_full_csa_tnd)"
        for node in root.body
    )
    adapter = ast.parse((_PACKAGE / "native_adapter.py").read_text())
    registrations = [
        node
        for node in ast.walk(adapter)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "pypto.torch.register"
    ]
    assert len(registrations) == 1
    assert ast.unparse(registrations[0].args[0]) == "decode_csa_tp1_layer_test"
    assert ast.literal_eval(registrations[0].args[1]) == "dsv4_csa_tnd_full_precision::attention"
