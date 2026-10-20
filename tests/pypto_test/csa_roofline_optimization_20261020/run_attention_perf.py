# SPDX-License-Identifier: Apache-2.0
"""绑定冻结公共源码后运行已授权的Attention性能诊断。"""
import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output", type=Path, required=True)
args, _ = parser.parse_known_args()
output = args.output.resolve()
os.environ["PYPTO_PROG_BUILD_DIR"] = str(output / "build_output")
os.environ["PYPTO_CACHE_DIR"] = str(output / "cache")
logs = output.parent / (output.name + "_device_logs")
logs.mkdir(exist_ok=True)
os.environ["ASCEND_PROCESS_LOG_PATH"] = str(logs)
prefix = "vllm_ascend.ops.pypto"
if any(name.startswith(prefix + ".") for name in sys.modules):
    raise RuntimeError("公共子模块已加载，拒绝混用冻结与活动源码")
package = importlib.import_module(prefix)
package.__path__ = [str(here / "frozen_pypto")]
package.__spec__.submodule_search_locations = package.__path__
original = (here.parent / "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
            "/parallel_20261018/mgather_physical_plan_retest_20261019")
sys.path[:0] = [str(here), str(original)]
runner = here / "attention_perf_runner.py"
sys.argv[0] = str(runner)
try:
    runpy.run_path(str(runner), run_name="__main__")
finally:
    if output.is_dir():
        (output / "diagnostic_runner.py").write_text(runner.read_text())
        (output / "diagnostic_matrix.py").write_text((here / "attention_perf_matrix.py").read_text())
        modules = {name: str(Path(module.__file__).resolve())
                   for name, module in tuple(sys.modules.items())
                   if name.startswith(prefix + ".") and getattr(module, "__file__", None)}
        (output / "imported_source_paths.json").write_text(
            json.dumps(modules, ensure_ascii=False, indent=2) + "\n")
