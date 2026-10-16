# SPDX-License-Identifier: Apache-2.0
"""HC-pre的Native split-K随token档位变化；未实现的mDim2不得误用mDim1核。"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_full_precision/service.py"


def _runtime():
    # 加载实际准入方法，隔离CPU机上不存在的vLLM/NPU依赖。
    node = next(n for n in ast.parse(_PATH.read_text()).body if isinstance(n, ast.ClassDef))
    node.body = [
        item
        for item in node.body
        if (isinstance(item, ast.FunctionDef) and item.name == "eligible")
        or (
            isinstance(item, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "max_hc_pre_tokens" for target in item.targets)
        )
    ]

    class Baseline:
        def eligible(self, context, hidden, positions):
            self.delegated = (context, hidden, positions)
            return context.accept

    namespace = {"BaselineRuntime": Baseline}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(_PATH), "exec"), namespace)
    return namespace["CSAServiceRuntime"]()


@pytest.mark.parametrize("tokens", [1, 11, 24, 240, 256])
def test_supported_rows_preserve_native_metadata_eligibility(tokens):
    runtime = _runtime()
    hidden, positions = SimpleNamespace(shape=(tokens, 4, 4096)), object()
    for accept in (True, False):
        context = SimpleNamespace(accept=accept)
        assert runtime.eligible(context, hidden, positions) is accept
        assert runtime.delegated == (context, hidden, positions)


@pytest.mark.parametrize("tokens", [257, 384])
def test_native_mdim2_rows_do_not_use_mdim1_hc_pre(tokens):
    runtime = _runtime()
    assert not runtime.eligible(SimpleNamespace(accept=True), SimpleNamespace(shape=(tokens, 4, 4096)), None)
    assert not hasattr(runtime, "delegated")
