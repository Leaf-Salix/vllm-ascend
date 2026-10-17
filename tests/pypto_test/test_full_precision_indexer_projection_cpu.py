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

    def program(text):
        tree = ast.parse(text)
        return ast.dump(
            ast.Module(
                body=[
                    node
                    for node in tree.body
                    if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef))
                ],
                type_ignores=[],
            )
        )

    assert program((_PACKAGE / "indexer_projection.py").read_text()) == program(source)


def test_service_replaces_native_indexer_projection_boundaries():
    tree = ast.parse((_PACKAGE / "service.py").read_text())
    runtime = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    forward = next(node for node in runtime.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
    calls = {ast.unparse(node.func) for node in ast.walk(forward) if isinstance(node, ast.Call)}
    assert "self.pto_indexer_projection" in calls
    assert "impl.weights_proj" not in calls
    assert "DeviceOperator.prepare_dsa_indexer_weights" not in calls
    assert "impl._forward_o_proj" in calls
