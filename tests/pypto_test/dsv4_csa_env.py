"""Source selection for standalone CSA tests; never imported by production code."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def activate() -> Path:
    """选择当前 release 的模型源码与指定调试分支的 editable 安装。"""
    # 所有 CSA 功能/性能测试关闭 EPLB，不继承调用者的重平衡或热度采集开关。
    os.environ["DYNAMIC_EPLB"] = "false"
    os.environ["EXPERT_MAP_RECORD"] = "false"
    root_text = os.environ.get("PTO_EAGER_ROOT")
    if not root_text:
        raise RuntimeError("source ../env-dsv4-0251rc1.sh before running the CSA tests")
    root = Path(root_text).resolve()
    repo = Path(__file__).resolve().parents[2]
    vllm_source = root / ".cache/migration-v0.25.1rc1/vllm"
    expected_sources = [("vllm_ascend", repo), ("pypto", root / "pypto")]
    if vllm_source.is_dir():
        expected_sources.append(("vllm", vllm_source))
    for name, expected in expected_sources:
        loaded = sys.modules.get(name)
        if loaded is not None and expected not in Path(loaded.__file__).resolve().parents:
            raise RuntimeError(f"{name} already loaded from {loaded.__file__}, expected {expected}")
    sys.path.insert(0, str(repo))
    if vllm_source.is_dir():
        sys.path.insert(0, str(vllm_source))
    # vLLM inspects model classes in a fresh Python subprocess. sys.path alone
    # would leave that subprocess using the old editable vLLM checkout.
    source_paths = [str(repo)]
    if vllm_source.is_dir():
        source_paths.insert(0, str(vllm_source))
    inherited_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(source_paths + inherited_paths))

    # Match the Qwen example's process-local editable-hook repair. The global
    # user site also has PyPTO/Simpler checkouts; their hooks must not win.
    def is_other_editable(finder: object) -> bool:
        sources = getattr(finder, "known_source_files", {})
        return type(finder).__name__ == "ScikitBuildRedirectingFinder" and any(
            "/pypto/" in str(source) or "/simpler/" in str(source) for source in sources.values()
        )

    sys.meta_path[:] = [finder for finder in sys.meta_path if not is_other_editable(finder)]
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = root / ".venv-dsv4-0251rc1/lib" / version / "site-packages"
    # HCA_SIMPLER_ROOT：把 simpler 指到一个独立 worktree，用于在不影响共用 checkout
    # （CSA 会话正在用它）的前提下试改运行时。不设置时行为与以前完全一致。
    # 必须在这里处理：本函数下面会重新执行 venv 里的 editable 钩子，把 simpler 钉回
    # 共用目录，因此任何在 site 阶段（sitecustomize）做的重定向都会被覆盖。
    # simpler_setup 也要一起指过去——它的 PROJECT_ROOT 按模块位置解析，
    # get_pto_isa_clone_path() 与 .pto-isa.lock 都挂在其下，只重定向 simpler 会让锁
    # 仍然写进共用的 simpler/build。
    simpler_root_text = os.environ.get("HCA_SIMPLER_ROOT")
    simpler_root = Path(simpler_root_text).resolve() if simpler_root_text else None
    if simpler_root is not None:
        if not (simpler_root / "python/simpler").is_dir() or not (simpler_root / "simpler_setup").is_dir():
            raise RuntimeError(f"HCA_SIMPLER_ROOT 不是 simpler 仓库：{simpler_root}")
        for loaded_name in ("simpler", "simpler_setup"):
            loaded = sys.modules.get(loaded_name)
            if loaded is not None and simpler_root not in Path(loaded.__file__).resolve().parents:
                raise RuntimeError(f"{loaded_name} 已从 {loaded.__file__} 加载，无法改指 {simpler_root}")
        sys.path.insert(0, str(simpler_root))
        sys.path.insert(0, str(simpler_root / "python"))
    for name in ("pypto", "simpler"):
        if name == "simpler" and simpler_root is not None:
            # 不重装 simpler 的 editable 钩子，否则它会把 simpler 钉回共用目录。
            continue
        candidates = [site / f"_{name}_editable.py", site / f"_editable_skbc_{name}.py"]
        installed = [candidate for candidate in candidates if candidate.is_file()]
        if len(installed) != 1:
            raise ImportError(f"expected one editable hook for {name}, found {installed} in {site}")
        path = installed[0]
        spec = importlib.util.spec_from_file_location(f"_csa_{name}_editable", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    if simpler_root is not None:
        # 钩子重装后再摘一次：上面的 exec_module 可能又插回 simpler 的 finder。
        sys.meta_path[:] = [
            finder
            for finder in sys.meta_path
            if not (
                type(finder).__name__ == "ScikitBuildRedirectingFinder"
                and any("/simpler/" in str(source) for source in getattr(finder, "known_source_files", {}).values())
            )
        ]
        os.environ["HCA_SIMPLER_ROOT_ACTIVE"] = str(simpler_root)
    return repo


def write_json(path: Path, value: object) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
