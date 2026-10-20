# SPDX-License-Identifier: Apache-2.0
"""复用原真实S1/S5精确回归，核对100/150退避的图重放及信号代际。

两边同冻结正式数学与工具链；reference为100，candidate为150。
S5可在容量6的策略下执行真实5个query，不把输入补成S6。
本入口不测性能，不修改正式源码或共享安装。
"""
import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

from run_wait_backoff import audit_actual_sources, rewrite_ticks


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--output", type=Path, required=True)
    args, _ = ap.parse_known_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(output / "build_output")
    os.environ["PYPTO_CACHE_DIR"] = str(output / "cache")
    logs = output.parent / (output.name + "_device_logs")
    logs.mkdir(exist_ok=False)
    os.environ["ASCEND_PROCESS_LOG_PATH"] = str(logs)

    package = importlib.import_module("vllm_ascend.ops.pypto")
    if any(name.startswith(package.__name__ + ".") for name in sys.modules):
        raise RuntimeError("冻结公共源码前已有子模块导入")
    package.__path__ = [str(here / "frozen_pypto")]
    package.__spec__.submodule_search_locations = package.__path__
    harness = here / "wait150_query_regression"
    sys.path.insert(0, str(harness))
    diagnostics = importlib.import_module("compile_diagnostics")
    old_probe = diagnostics.compilation_probe
    audits = []

    @contextlib.contextmanager
    def probe(directory, *, wait_backoff=False, phase_clock=False):
        directory = Path(directory)
        role = directory.name.removeprefix("compile_")
        if role not in ("reference", "candidate") or not wait_backoff or phase_clock:
            raise AssertionError("只允许真实query两侧的等待策略编译")
        ticks = 100 if role == "reference" else 150
        wait = importlib.import_module(
            "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.wait_codegen"
        )
        original = wait.rewrite_wait_source
        before = set((output / "build_output").glob("*decode_csa_spmd_*/kernels/*/*.cpp"))
        wait.rewrite_wait_source = lambda text: rewrite_ticks(text, ticks, original)
        try:
            with old_probe(directory, wait_backoff=True):
                yield
        finally:
            wait.rewrite_wait_source = original
        after = set((output / "build_output").glob("*decode_csa_spmd_*/kernels/*/*.cpp"))
        actual = audit_actual_sources(after - before, ticks)
        audits.append(dict(role=role, ticks=ticks, actual_cc_ec_inputs=actual))

    import pypto

    pypto.configure_cache(pypto.CacheConfig(enabled=False))
    diagnostics.compilation_probe = probe
    sys.argv[0] = str(harness / "query_shape_pair.py")
    try:
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        result = json.loads((output / "result.json").read_text())
        if not result["passed"] or sorted(a["role"] for a in audits) != ["candidate", "reference"]:
            raise AssertionError("真实query回归或两侧实际源码审计没有完成")
    finally:
        diagnostics.compilation_probe = old_probe
        if output.exists():
            (output / "WAIT_BACKOFF_QUERY_AUDIT.json").write_text(
                json.dumps(dict(roles=audits, candidate_ticks=150, reference_ticks=100),
                           ensure_ascii=False, indent=2) + "\n"
            )
            (output / Path(__file__).name).write_text(Path(__file__).read_text())


if __name__ == "__main__":
    main()
