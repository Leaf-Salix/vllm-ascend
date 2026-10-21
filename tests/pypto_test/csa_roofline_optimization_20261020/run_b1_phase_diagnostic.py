# SPDX-License-Identifier: Apache-2.0
"""观察当前B1单SPMD阶段端点；保留原数学及同步，不计入性能对照表。

复用已有通知/等待时戳探针，只写每角色独占信号槽的保留缓存行。
不注入FFTS等待计数，避免旧探针的全24核非空工作假设。所有角色都观察，
没有工作也保留其原phase。探针扰动后的相邻端点差不是引擎独占时间。
"""

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

from run_wait_backoff import audit_actual_sources


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    output = args.output.resolve()
    frozen = here / "frozen_pypto_goal10_b1update"
    source = here / "source_after_b1_half_v2"
    prefix = "vllm_ascend.ops.pypto"
    if any(name.startswith(prefix + ".") for name in sys.modules):
        raise RuntimeError("冻结前已有公共子模块导入")
    package = importlib.import_module(prefix)
    package.__path__ = [str(frozen)]
    package.__spec__.submodule_search_locations = package.__path__
    parent = importlib.import_module(prefix + ".deepseek_v4_flash_csa_spmd")
    name = parent.__name__ + ".wait_codegen"
    spec = importlib.util.spec_from_file_location(name, source / "wait_codegen.py")
    wait = importlib.util.module_from_spec(spec)
    sys.modules[name] = wait
    spec.loader.exec_module(wait)
    parent.wait_codegen = wait

    base = here.parent / "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
    sys.path.insert(0, str(base))
    diagnostic = importlib.import_module("compile_diagnostics")
    original_export = diagnostic.export_clock

    def export(signal, destination, phases, *, mask, tokens):
        result = original_export(signal, destination, phases)
        # 此探针不写32..46项；扩大padding白名单的旧供数runner仍须核对这些项。
        assert bool((signal[:, :, 32:47] == 0).all())
        result.update(eligible_for_performance=False, all_roles=72,
                      scope="原phase通知端点和累计GM wait；不含FFTS细分，不是纯计算耗时")
        return result

    diagnostic.export_clock = export
    import pypto

    pypto.configure_cache(pypto.CacheConfig(enabled=False))
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(output / "build_output")
    os.environ["PYPTO_CACHE_DIR"] = str(output / "cache")
    runner = here.parent / "csa_critical_path_20261020/run_supply_device.py"
    sys.argv = [str(runner), "--output", str(output), "--batch", "1", "--history", "131072",
                "--launches", "1", "--phase-clock", "--wait-backoff", "--spmd-source", str(source)]
    try:
        runpy.run_path(str(runner), run_name="__main__")
        result = json.loads((output / "result.json").read_text())
        assert result["passed"]
        artifacts = set((output / "build_output").glob("decode_csa_spmd_*/kernels/*/*.cpp"))
        audit = audit_actual_sources(artifacts, wait.WAIT_BACKOFF_TICKS)
        (output / "B1_PHASE_AUDIT.json").write_text(json.dumps(
            dict(passed=True, source=str(source), frozen=str(frozen),
                 wait_ticks=wait.WAIT_BACKOFF_TICKS, cpp=audit,
                 eligible_for_performance=False), ensure_ascii=False, indent=2) + "\n")
    finally:
        diagnostic.export_clock = original_export


if __name__ == "__main__":
    main()
