# SPDX-License-Identifier: Apache-2.0
"""只编译两路压缩单MIX组合，不执行NPU；保留失败、源码快照及真实动态ABI。"""

import argparse
import json
import re
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    package = here.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    report = {
        "passed": False,
        "device_execution": False,
        "device_sync_validated": False,
        "device_precision_validated": False,
        "performance_measured": False,
        "sample_shapes_supplied": False,
        "core_groups": 24,
        "aic": 24,
        "aiv": 48,
        "phase_count": 14,
        "signal_minimum_int32_words": 14 * 72 * 128,
        "scope": "两路压缩组合，非完整CSA",
    }
    try:
        for path in here.glob("*.py"):
            (output / path.name).write_text(path.read_text())
        for name in (
            "compressor_projection_stages",
            "compressor_pool_stages",
            "compressor_output_stages",
            "indexer_compressor_output_stages",
            "indexer_cache_stages",
            "metadata_stages",
            "request_geometry",
            "phase_sync",
            "signals",
        ):
            (output / (name + ".py")).write_text((package / (name + ".py")).read_text())
        (output / "model_config.py").write_text((package.parent / "deepseek_v4_flash_csa/config.py").read_text())
        from check_phase_model import check

        report["phase_model"] = check(here)
        import dsv4_csa_env

        dsv4_csa_env.activate()
        import probe
        from pypto import passes
        from pypto.ir.printer import python_print
        from pypto.runtime import RunConfig

        logical = python_print(probe.entry.specialize())
        (output / "symbolic_entry.py").write_text(logical)
        dimensions = set(re.findall(r'pl.dynamic\("(COMP_PIPE_\w+)"\)', logical))
        assert dimensions
        report["symbolic_parse_passed"] = True
        report["symbolic_dynamic_dimensions"] = sorted(dimensions)
        artifact = probe.entry.compile(
            config=RunConfig(
                platform="a2a3",
                save_kernels=True,
                save_kernels_dir=str(output / "kernel"),
                dump_passes=True,
                memory_planner=passes.MemoryPlanner.PYPTO,
            )
        )
        kernel = Path(artifact.output_dir)
        report["artifact_output"] = str(kernel)
        report["full_codegen_passed"] = True
        orchestration = "\n".join(p.read_text() for p in (kernel / "orchestration").glob("*.cpp"))
        submissions = re.findall(r"\brt_submit_\w+\s*\(", orchestration)
        report["submission_calls"] = submissions
        report["single_mix_submission"] = len(submissions) == 1 and "MixedKernels" in orchestration
        report["sync_start"] = "set_require_sync_start(true)" in orchestration
        assert report["single_mix_submission"] and report["sync_start"]
        cpp_dims = {}
        for path in (kernel / "kernels").rglob("*.cpp"):
            cpp_dims[str(path.relative_to(kernel))] = re.findall(
                r"// Extract dynamic dim: (COMP_PIPE_\w+)", path.read_text()
            )
        report["cpp_dynamic_dimensions"] = cpp_dims
        assert dimensions <= {dim for values in cpp_dims.values() for dim in values}
        report["compiled_abi"] = json.loads((kernel / "compiled_meta.json").read_text())["params"]
        assert len(report["compiled_abi"]) == 48
        report["passed"] = True
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    finally:
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
