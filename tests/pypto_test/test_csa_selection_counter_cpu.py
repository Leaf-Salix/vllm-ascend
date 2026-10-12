# SPDX-License-Identifier: Apache-2.0
"""无需 Torch/NPU：验证可选 HCA 不影响 CSA，且真实依赖错误向上传播。"""

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace

path = Path(__file__).parent / "offline_pd/observer.py"
node = next(
    n
    for n in ast.parse(path.read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name == "_enable_selection_counter"
)
code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")


def test_selection_counter_without_npu():
    for mode in ("absent", "inherited", "broken"):

        class CSA:
            layer_name = "layer2"

            def eligible(self, context, hidden, positions):
                return context

        class HCA(CSA):
            layer_name = "layer3"

        def importer(name, *args, _mode=mode, **kwargs):
            if name.endswith(".variant"):
                return SimpleNamespace(variant_package=lambda: "test.csa")
            if name == "test.csa.service":
                return SimpleNamespace(CSAServiceRuntime=CSA)
            if name.endswith("deepseek_v4_flash_hca.service"):
                if _mode == "absent":
                    raise ModuleNotFoundError(name=name.rsplit(".", 1)[0])
                if _mode == "broken":
                    raise ModuleNotFoundError(name="real_missing_dependency")
                return SimpleNamespace(HCAServiceRuntime=HCA)
            raise AssertionError(name)

        env = {"__builtins__": dict(vars(builtins), __import__=importer), "_CSA_SELECTION": {}}
        exec(code, env)
        try:
            env["_enable_selection_counter"]()
        except ModuleNotFoundError as error:
            assert mode == "broken" and error.name == "real_missing_dependency"
            continue
        assert mode != "broken"
        hidden = SimpleNamespace(shape=(240,))
        assert CSA().eligible(True, hidden, None)
        assert not CSA().eligible(False, hidden, None)
        assert env["_CSA_SELECTION"]["layer2"] == {"pto_tokens240": 1, "native_tokens240": 1}
        if mode == "inherited":
            HCA().eligible(True, hidden, None)
            assert env["_CSA_SELECTION"]["layer3"] == {"pto_tokens240": 1}
        print(mode, "PASS")
    print("dependency error propagation PASS")
