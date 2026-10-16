# SPDX-License-Identifier: Apache-2.0
"""将正式WKV及次序表绑定到已完成的NPU对拍，CPU机无需导入vLLM/NPU。"""

import ast
import hashlib
import json
import runpy
import tarfile
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_PACKAGE = _ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_full_precision"
_EVIDENCE = Path(__file__).parent / "evidence_tnd_precision/full_pto_20261016"


def test_all_token_orders_match_captured_native_qualification():
    data = runpy.run_path(str(_PACKAGE / "wkv_order_data.py"))
    raw = (_EVIDENCE / "wkv-native-order-map.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == data["NATIVE_ORDER_MAP_SHA256"]
    reference = json.loads(raw)
    assert len(data["WKV_TOKEN_MODES"]) == 256
    assert len(data["WKV_COLUMN_ORDERS"]) == 9
    for order in data["WKV_K_ORDERS"]:
        assert sorted(order) == list(range(16))
    for tokens in range(1, 257):
        mode = data["WKV_TOKEN_MODES"][tokens - 1]
        indices = data["WKV_COLUMN_ORDERS"][mode]
        assert len(indices) == 32
        actual = [list(data["WKV_K_ORDERS"][index]) for index in indices]
        assert actual == reference["modes"][reference["tokens_to_mode"][tokens - 1]]


def test_formal_kernel_matches_npu_qualified_program():
    archive = _EVIDENCE / "wkv-evidence-v3.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
        "f8fcd4ae5603690b2401e1cf9f50dda5531f5f5d16c088b110bcb5d50e9de0c4"
    )
    with tarfile.open(archive) as bundle:
        source = bundle.extractfile("wkv-evidence-v3/experiments/wkv-module-v17-all-shapes/wkv.py").read().decode()

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

    assert program((_PACKAGE / "wkv.py").read_text()) == program(source)


def test_service_replaces_native_wkv_call():
    tree = ast.parse((_PACKAGE / "service.py").read_text())
    runtime = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    forward = next(node for node in runtime.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
    calls = {ast.unparse(node.func) for node in ast.walk(forward) if isinstance(node, ast.Call)}
    assert "self.pto_wkv" in calls
    assert "self.layer.self_attn.wkv" not in calls
