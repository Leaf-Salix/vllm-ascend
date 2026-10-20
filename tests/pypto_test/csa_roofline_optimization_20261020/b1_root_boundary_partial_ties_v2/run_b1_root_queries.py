# SPDX-License-Identifier: Apache-2.0
"""B1分叶及UB常驻根的真实S1/S5回归；两侧统一150tick，不测性能。"""

import argparse
import contextlib
import importlib
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
    args, forwarded = parser.parse_known_args()
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
        raise RuntimeError("冻结前已有公共子模块导入")
    package.__path__ = [str(here / "frozen_pypto_goal10")]
    package.__spec__.submodule_search_locations = package.__path__
    harness = here / "b1_root_query_regression"
    sys.path.insert(0, str(harness))
    diagnostics = importlib.import_module("compile_diagnostics")
    original = diagnostics.compilation_probe
    audits = []

    @contextlib.contextmanager
    def probe(directory, *, wait_backoff=False, phase_clock=False):
        role = Path(directory).name.removeprefix("compile_")
        if role not in ("reference", "candidate") or not wait_backoff or phase_clock:
            raise ValueError("只核对真实query的两侧无插桩编译")
        before = set((output / "build_output").glob("*decode_csa_spmd_*/kernels/*/*.cpp"))
        with original(directory, wait_backoff=True):
            yield
        after = set((output / "build_output").glob("*decode_csa_spmd_*/kernels/*/*.cpp"))
        audits.append(dict(role=role, ticks=150, actual_cc_ec_inputs=audit_actual_sources(after - before, 150)))

    import pypto

    pypto.configure_cache(pypto.CacheConfig(enabled=False))
    diagnostics.compilation_probe = probe
    sys.argv = [str(harness / "query_shape_pair.py"), "--output", str(args.output), *forwarded]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
        result = json.loads((output / "result.json").read_text())
        if not result["passed"] or sorted(row["role"] for row in audits) != ["candidate", "reference"]:
            raise AssertionError("真实query回归或实际编译审计没有完成")
    finally:
        diagnostics.compilation_probe = original
        if output.exists():
            (output / "QUERY_WAIT_AUDIT.json").write_text(
                json.dumps(dict(roles=audits, ticks=150), ensure_ascii=False, indent=2) + "\n"
            )
            (output / Path(__file__).name).write_text(Path(__file__).read_text())


if __name__ == "__main__":
    main()
