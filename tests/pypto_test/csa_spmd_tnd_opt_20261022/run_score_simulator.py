# SPDX-License-Identifier: Apache-2.0
"""Run one immutable producer diagnostic through the incore-profiling workflow."""

import argparse
import hashlib
import json
from pathlib import Path

from generate_score_sim_case import module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    assert 0 <= args.device < 16
    job = args.job.resolve()
    manifest = json.loads((job / "job_manifest.json").read_text())
    for relative, expected in manifest["immutable_files"].items():
        assert hashlib.sha256((job / relative).read_bytes()).hexdigest() == expected, relative
    workspace = Path(manifest["workspace"])
    shared_path = workspace / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/manifest.json"
    shared = json.loads(shared_path.read_text())
    assert shared["status"] == "validated" and shared == manifest["shared_toolchain"]
    isa = workspace / shared["simpler"]["source"] / "build/pto-isa"
    cann = "/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2/set_env.sh"
    profiler = module(job / "scripts/tools/incore_profile.py", "score_incore_profiler")
    config, case_args = profiler.parse_args(
        [
            "--build-dir",
            str(job / "input"),
            "--target",
            "a2a3",
            "--func",
            "score_producer",
            "--cann-set-env",
            cann,
            "--pto-isa-root",
            str(isa),
            "--aicore-arch",
            "dav-c220-cube",
            "--soc-version",
            "Ascend910_9391",
            "--no-auto-msopprof",
            "--msprof-timeout",
            "600",
        ]
    )
    assert not case_args and not config.auto_msopprof and config.msopprof is None
    # Only testcase generation is extended: exact per-axis allocations, scalar
    # bindings, control inputs, and an output check. The skill owns build/export.
    config.generate_testcase = job / "scripts/generate_score_sim_case.py"
    env = profiler.validate_toolchain(config)
    profiler.assert_bisheng_supports_tl(env, config.aicore_arch)
    profiler.ensure_msopprof_worker(env, config)
    assert config.soc_version in profiler.discover_camodel_socs(env)
    source = job / "input/ptoas/score_producer.cpp"
    assert source.is_file() and source.with_suffix(".pto").is_file()
    output = job / "profile"
    output.mkdir(exist_ok=False)
    result = profiler.export_one(source, config, env, output, 1, 1)
    profiler.write_outputs(output, [result], list(result))
    check = Path(result["case_dir"]) / "score_output_check.txt"
    output_verified = check.exists() and check.read_text().strip() == "passed elements=12288 value=8"
    report = dict(
        scope=manifest["scope"],
        formal_gate_eligible=False,
        physical_queue_device=args.device,
        simulator_virtual_device=0,
        simulator_soc=config.soc_version,
        target="a2a3",
        cann_set_env=cann,
        pto_isa_root=str(isa),
        shared_toolchain=shared,
        output_verified=output_verified,
        output_check=str(check),
        collection=result,
        passed=result["status"] == "exported" and output_verified,
        limitations=[
            "single AIC producer without FIFO consumer",
            "synthetic inputs",
            "not a 72-role schedule or hardware latency prediction",
            "simulator model pinned for relative diagnosis; hardware SoC identity not asserted",
        ],
    )
    (job / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "output_verified", "simulator_soc")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
