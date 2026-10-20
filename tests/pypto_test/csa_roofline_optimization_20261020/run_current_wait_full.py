# SPDX-License-Identifier: Apache-2.0
"""冻结公共数学源码，并显式绑定本轮选定的等待策略，再跑原完整配对测试。

旧run_frozen_full及其frozen_pypto仍复现100tick，不改写历史基线。
新的候选/参考共用--wait-policy-source指定文件，实际常量须在CCEC输入核对。
只改变测试进程的模块绑定，不修改共享安装或在测量热路径增加Host调用。
"""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

from run_wait_backoff import audit_actual_sources


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--wait-policy-source", type=Path, required=True)
    args, forwarded = ap.parse_known_args()
    if not args.wait_policy_source.is_file():
        raise FileNotFoundError(args.wait_policy_source)
    if "--wait-backoff" not in forwarded:
        raise ValueError("必须启用现有编译回调，不能将仅绑定模块当作策略已生效")
    prefix = "vllm_ascend.ops.pypto"
    if any(name.startswith(prefix + ".") for name in sys.modules):
        raise RuntimeError("冻结前已有公共子模块导入")
    package = importlib.import_module(prefix)
    package.__path__ = [str(here / "frozen_pypto")]
    package.__spec__.submodule_search_locations = package.__path__
    parent = prefix + ".deepseek_v4_flash_csa_spmd"
    parent_module = importlib.import_module(parent)
    name = parent + ".wait_codegen"
    spec = importlib.util.spec_from_file_location(name, args.wait_policy_source.resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    parent_module.wait_codegen = module
    ticks = module.WAIT_BACKOFF_TICKS
    import pypto

    pypto.configure_cache(pypto.CacheConfig(enabled=False))
    runner = here.parent / "csa_critical_path_20261020/run_full.py"
    sys.path.insert(0, str(runner.parent))
    sys.argv = [str(runner), "--output", str(args.output), *forwarded]
    try:
        runpy.run_path(str(runner), run_name="__main__")
        result = json.loads((args.output / "result.json").read_text())
        if not result["passed"]:
            raise AssertionError("完整CSA验证尚未通过")
        matrix = result["source_matrix"]
        audits = {}
        for role in ("spmd_candidate", "spmd_reference"):
            folder = Path(matrix["kernel_configs"][role]).parent
            audits[role] = audit_actual_sources(set(folder.glob("kernels/*/*.cpp")), ticks)
        (args.output / "CURRENT_WAIT_POLICY_AUDIT.json").write_text(
            json.dumps(dict(passed=True, ticks=ticks, roles=audits,
                            source=str(args.wait_policy_source.resolve())),
                       ensure_ascii=False, indent=2) + "\n"
        )
    finally:
        if args.output.is_dir():
            (args.output / "selected_wait_codegen.py").write_text(args.wait_policy_source.read_text())
            (args.output / "imported_source_paths.json").write_text(
                json.dumps({
                    name: str(Path(mod.__file__).resolve())
                    for name, mod in tuple(sys.modules.items())
                    if name.startswith(prefix + ".") and getattr(mod, "__file__", None)
                }, ensure_ascii=False, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
