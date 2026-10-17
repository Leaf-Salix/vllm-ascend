# SPDX-License-Identifier: Apache-2.0
"""只编译动态 HC→Q_A→QR→QB→query RMS/RoPE 前段，记录真实失败，不执行 NPU。"""

import argparse
import json
import os
import re
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontend-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False, "scope": "源级动态HC至query输出，一个NZ2 MIX；不是完整CSA",
        "device_execution": False, "device_precision_validated": False,
        "device_sync_validated": False, "performance_measured": False,
        "weight_layout_mode": 2, "cache_policy_override": False,
        "core_groups": 24, "participant_cores": 72, "signal_phases_including_generation": 12,
    }
    try:
        os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"
        import dsv4_csa_env

        dsv4_csa_env.activate()
        import probe
        from pypto import passes
        from pypto.ir.printer import python_print
        from pypto.runtime import RunConfig

        here = Path(__file__).resolve().parent
        package = here.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
        for source in (here / "probe.py", here / "check_cpu.py"):
            (args.output / source.name).write_text(source.read_text())
        for name in ("hc_stages.py", "qkv_stages.py", "query_projection_stages.py", "phase_sync.py", "signals.py"):
            (args.output / name).write_text((package / name).read_text())
        logical = python_print(probe.entry.specialize())
        (args.output / "symbolic_entry.py").write_text(logical)
        expected_dims = {"QUERY_FRONT_T", "QUERY_FRONT_PAD16", "QUERY_FRONT_PAD8", "QUERY_FRONT_PARTIAL_ROWS",
                         "QUERY_FRONT_QR_PAD128", "QUERY_FRONT_PROJ_PAD128", "QUERY_FRONT_SIGNAL_WORDS"}
        for symbol in expected_dims:
            assert symbol in logical, symbol
        report["symbolic_frontend_passed"] = True
        if args.frontend_only:
            lowered = probe.entry.lower(config=RunConfig(platform="a2a3"))
            (args.output / "lowered.py").write_text(str(lowered))
            report["frontend_lowering_passed"] = True
        else:
            artifact = probe.entry.compile(config=RunConfig(
                platform="a2a3", save_kernels=True, save_kernels_dir=str(args.output / "kernel"),
                dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO,
            ))
            output = Path(artifact.output_dir)
            orchestration = "\n".join(path.read_text() for path in (output / "orchestration").glob("*.cpp"))
            calls = re.findall(r"\brt_submit_\w+\s*\(", orchestration)
            kernels = list((output / "kernels").rglob("*.cpp"))
            dimensions = {}
            for path in kernels:
                dimensions[str(path.relative_to(output))] = re.findall(
                    r"int64_t (QUERY_FRONT_\w+) = .*?->shapes\[\d+\]\);", path.read_text(),
                )
            report.update(
                output_dir=str(output), submission_calls=calls, dynamic_cpp_dimensions=dimensions,
                single_mix_submission=len(calls) == 1 and "MixedKernels" in orchestration,
                sync_start="set_require_sync_start(true)" in orchestration,
                full_codegen_passed=True,
            )
            assert report["single_mix_submission"] and report["sync_start"]
            all_dims = {name for names in dimensions.values() for name in names}
            assert expected_dims <= all_dims
        report["passed"] = True
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        summary = {key: value for key, value in report.items() if key != "traceback"}
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
