# SPDX-License-Identifier: Apache-2.0
"""CSA 复现环境入口：显式绑定源码与本地构建产物，然后运行已有测试。

该文件仅用于 tests，不参与服务或算子执行。它替代历史测试依赖本机
editable hook 的环境初始化，保留原测试的计算、精度门槛和计时方式。
PyPTO 必须在指定源码树构建；Simpler 必须安装到自己目录下的 .venv。
所有绑定仅作用于当前 Python 进程，不修改共享安装和其他工作区。
"""

import argparse
import importlib.abc
import importlib.machinery
import json
import os
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-toolchain", action="store_true")
    parser.add_argument("--pypto-root", type=Path)
    parser.add_argument("--simpler-root", type=Path)
    parser.add_argument("--ptoas-root", type=Path)
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--native-plugin-root", type=Path, help="数值对拍时冻结Native插件源码")
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("script", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    workspace = args.workspace_root.resolve() if args.workspace_root else repo.parent
    if args.shared_toolchain:
        if any((args.pypto_root, args.simpler_root, args.ptoas_root)):
            parser.error("--shared-toolchain 与单独指定 SDK 路径互斥")
        sys.path.insert(0, str(repo / "tests/pypto_test"))
        from csa_shared_toolchain.selection import selected_toolchain

        selected, paths, manifest = selected_toolchain()
        args.pypto_root, args.simpler_root, args.ptoas_root = paths["pypto"], paths["simpler"], paths["ptoas"]
        print(
            "CSA_SHARED_TOOLCHAIN "
            + json.dumps(
                {
                    "release": selected["release"],
                    "status": selected["status"],
                    "manifest": str(manifest),
                    **{key: str(path) for key, path in paths.items()},
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    elif not all((args.pypto_root, args.simpler_root, args.ptoas_root)):
        parser.error("请选择 --shared-toolchain 或完整指定三个 SDK 路径")
    pypto = args.pypto_root.resolve()
    simpler = args.simpler_root.resolve()
    ptoas = args.ptoas_root.resolve()
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    simpler_site = simpler / ".venv/lib" / version / "site-packages"
    roots = {
        "pypto": pypto / "python",
        "simpler": simpler / "python",
        "simpler_setup": simpler,
        "_task_interface": simpler_site,
        "vllm_ascend": args.native_plugin_root.resolve() if args.native_plugin_root else repo,
    }
    if args.vllm_root:
        roots["vllm"] = args.vllm_root.resolve()
    if not list((pypto / "python/pypto").glob("pypto_core*.so")):
        raise RuntimeError("请先在 --pypto-root 中执行原生 CMake 构建")
    if not list(simpler_site.glob("_task_interface*.so")):
        raise RuntimeError("请先在 --simpler-root/.venv 中安装对应 Simpler")
    if not (ptoas / "bin/ptoas").is_file():
        raise RuntimeError("--ptoas-root 下缺少 bin/ptoas")

    class SourceFinder(importlib.abc.MetaPathFinder):
        """优先定位指定顶层包，子模块沿该包自己的路径继续加载。"""

        def find_spec(self, fullname, path=None, target=None):
            top = fullname.partition(".")[0]
            if top not in roots:
                return None
            search = path if "." in fullname else [str(roots[top])]
            spec = importlib.machinery.PathFinder.find_spec(fullname, search)
            if spec is None:
                raise ImportError(f"指定源码路径无法加载 {fullname}: {roots[top]}")
            return spec

    for name, root in roots.items():
        loaded = sys.modules.get(name)
        if loaded is not None and root not in Path(loaded.__file__).resolve().parents:
            raise RuntimeError(f"{name} 已从其他路径加载，拒绝混用工具链")
    sys.meta_path.insert(0, SourceFinder())
    sys.path[:0] = [str(repo / "tests/pypto_test"), str(simpler_site)]
    os.environ.update(
        PTO_EAGER_ROOT=str(workspace),
        CSA_STATIC_PYPTO_ROOT=str(pypto),
        CSA_STATIC_PYPTO_IMPORT_MODE="filesystem",
        HCA_SIMPLER_ROOT=str(simpler),
        HCA_SIMPLER_ROOT_ACTIVE=str(simpler),
        PTOAS_ROOT=str(ptoas),
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        DYNAMIC_EPLB="false",
        EXPERT_MAP_RECORD="false",
    )
    os.environ["PATH"] = str(ptoas / "bin") + os.pathsep + os.environ.get("PATH", "")
    # vLLM 的模型注册检查会另起进程，继续使用相同的模型及插件源码。
    child_sources = [str(args.native_plugin_root.resolve() if args.native_plugin_root else repo)]
    if args.shared_toolchain:
        os.environ.update(
            CSA_SHARED_TOOLCHAIN_RELEASE=selected["release"],
            CSA_SHARED_TOOLCHAIN_MANIFEST=str(manifest),
            CSA_SHARED_PYPTO_ROOT=str(pypto),
            CSA_SHARED_SIMPLER_ROOT=str(simpler),
            CSA_SHARED_PTOAS_ROOT=str(ptoas),
            CSA_SDMA_PYPTO_ROOT=str(pypto),
            CSA_SDMA_SIMPLER_ROOT=str(simpler),
        )
        child_sources.insert(0, str(repo / "tests/pypto_test/csa_shared_toolchain/python_startup"))
    if args.vllm_root:
        child_sources.insert(0, str(args.vllm_root.resolve()))
    inherited = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(child_sources + inherited))
    # 测试仍调用同一个环境入口；无需伪造或改写旧 editable 安装中的模块表。
    import dsv4_csa_env

    dsv4_csa_env.activate = lambda: repo
    target = args.script.resolve()
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *args.arguments]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
