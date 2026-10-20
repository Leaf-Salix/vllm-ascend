# SPDX-License-Identifier: Apache-2.0
"""只在测试进程绑定冻结公共源码，再复用原完整精度及配对 L1 测试。

不改测试容差、图重放检查或调度开销口径；数值候选可能在原逐 bit 比较处
停止，这也是要保留的诊断结果，不能为了拿性能把 FAIL 改成 PASS。
"""
import argparse
import importlib
import json
import runpy
import sys
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    frozen = here / "frozen_pypto"
    if not (frozen / "__init__.py").is_file():
        raise RuntimeError("请先运行 freeze_shared_sources.py")
    prefix = "vllm_ascend.ops.pypto"
    already = [name for name in sys.modules if name.startswith(prefix + ".")]
    if already:
        raise RuntimeError("已有公共子模块导入，拒绝混用源码：" + str(already))
    package = importlib.import_module(prefix)
    package.__path__ = [str(frozen)]
    package.__spec__.submodule_search_locations = [str(frozen)]
    runner = here.parent / "csa_critical_path_20261020/run_full.py"
    sys.path.insert(0, str(runner.parent))
    sys.argv[0] = str(runner)
    try:
        runpy.run_path(str(runner), run_name="__main__")
    finally:
        if args.output.is_dir():
            modules = {
                name: str(Path(module.__file__).resolve())
                for name, module in tuple(sys.modules.items())
                if name.startswith(prefix + ".") and getattr(module, "__file__", None)
            }
            (args.output / "imported_source_paths.json").write_text(
                json.dumps(modules, ensure_ascii=False, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
