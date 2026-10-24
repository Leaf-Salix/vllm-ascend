# SPDX-License-Identifier: Apache-2.0
"""加载固定 PTO 源码/SDK；不执行推理插件的 __init__，禁止导入引擎和 Native 扩展。"""

import argparse
import importlib.abc
import importlib.machinery
import json
import os
import runpy
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-toolchain", action="store_true", help="显式使用仓库公共工具链清单")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("script", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    workspace = args.workspace_root.resolve()
    manifest = json.loads((repo / "tests/pypto_test/csa_shared_toolchain/manifest.json").read_text())
    if manifest["status"] != "validated":
        raise RuntimeError("公共工具链尚未验收，不能作为复现版本")
    pypto = workspace / manifest["pypto"]["source"]
    simpler = workspace / manifest["simpler"]["source"]
    ptoas = workspace / manifest["ptoas"]["build"]
    site = simpler / f".venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    roots = {
        "pypto": pypto / "python",
        "simpler": simpler / "python",
        "simpler_setup": simpler,
        "_task_interface": site,
    }
    if not list((pypto / "python/pypto").glob("pypto_core*.so")) or not list(site.glob("_task_interface*.so")):
        raise RuntimeError("固定 PyPTO/Simpler 尚未构建")
    if not (ptoas / "bin/ptoas").is_file():
        raise RuntimeError("固定 PTOAS 尚未构建")

    class Sources(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            top = fullname.partition(".")[0]
            if top == "vllm" or fullname.startswith(
                ("vllm_ascend._C", "vllm_ascend.utils", "vllm_ascend.models", "vllm_ascend.attention")
            ):
                raise ImportError(f"独立 PyTorch 入口禁止加载推理引擎/Native 模块：{fullname}")
            if top not in roots:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path or [str(roots[top])])
            if spec is None:
                raise ImportError(f"固定 SDK 路径找不到 {fullname}")
            return spec

    sys.meta_path.insert(0, Sources())
    # 命名空间只提供算子源码的包路径，不执行插件注册、logger 或 Native op 初始化。
    for name, path in (
        ("vllm_ascend", repo / "vllm_ascend"),
        ("vllm_ascend.ops", repo / "vllm_ascend/ops"),
        ("vllm_ascend.ops.pypto", repo / "vllm_ascend/ops/pypto"),
    ):
        if name in sys.modules:
            raise RuntimeError(f"{name} 已被加载，不能混用运行环境")
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
    os.environ.update(
        PTOAS_ROOT=str(ptoas),
        HCA_SIMPLER_ROOT=str(simpler),
        HCA_SIMPLER_ROOT_ACTIVE=str(simpler),
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        VLLM_ASCEND_ENABLE_NZ="2",
        VLLM_ASCEND_PTO_CSA_ATOMIC_ADD="0",
        CSA_SHARED_TOOLCHAIN_MANIFEST=str(repo / "tests/pypto_test/csa_shared_toolchain/manifest.json"),
    )
    os.environ["PATH"] = str(ptoas / "bin") + os.pathsep + os.environ.get("PATH", "")
    # L1 转换器子进程只需 Simpler 的 Python 工具。
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, (simpler, simpler / "python", site)))
    script = args.script.resolve()
    sys.path[:0] = [str(script.parent), str(repo / "tests/pypto_test")]
    sys.argv = [str(script), *args.arguments]
    print("TORCH_ONLY_SDK " + json.dumps({key: str(path) for key, path in roots.items()}), flush=True)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
