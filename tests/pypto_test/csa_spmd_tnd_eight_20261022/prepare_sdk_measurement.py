# SPDX-License-Identifier: Apache-2.0
"""Freeze a version-gated successor without changing completed measurement artifacts."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def transform(source):
    source = replace_once(
        source,
        "from pathlib import Path\n",
        "from pathlib import Path\n\n"
        "from measurement_provenance import capture_toolchain, validate_sources, validated_reference\n",
    )
    source = replace_once(
        source,
        "    args = parser.parse_args()\n",
        "    comparison = parser.add_mutually_exclusive_group(required=True)\n"
        '    comparison.add_argument("--calibrate-baseline", action="store_true")\n'
        '    comparison.add_argument("--reference-result", type=Path)\n'
        "    args = parser.parse_args()\n"
        "    import pypto\n"
        "    pypto.configure_cache(pypto.CacheConfig(enabled=False))\n",
    )
    source = replace_once(
        source,
        "    args.output.mkdir(parents=True, exist_ok=False)\n",
        '    expected_sdk = json.loads(Path(__file__).with_name("expected_sdk.json").read_text())\n'
        "    sdk = capture_toolchain(expected_sdk)\n"
        '    source_hashes = validate_sources(args.frozen / "pypto/deepseek_v4_flash_csa_spmd_tnd",\n'
        "        args.candidate, frozen_baseline, args.calibrate_baseline)\n"
        "    args.output.mkdir(parents=True, exist_ok=False)\n",
    )
    source = replace_once(
        source,
        "    check_case(case)\n",
        "    check_case(case)\n"
        "    reference_median = None\n"
        "    if args.reference_result:\n"
        "        reference_median = validated_reference(\n"
        "            json.loads(args.reference_result.read_text()), case, sdk, frozen_baseline)\n",
    )
    source = replace_once(
        source,
        '        "scope": "TND-SPMD full CSA L1; frozen reference medians reused",\n',
        '        "scope": "TND-SPMD complete 72-role single-MIX L1; SDK-matched comparison",\n'
        '        "calibration": args.calibrate_baseline,\n'
        '        "shared_toolchain": sdk,\n'
        '        "source_hashes": source_hashes,\n'
        '        "reference_result": str(args.reference_result) if args.reference_result else None,\n'
        '        "reference_median_us": reference_median,\n'
        '        "sampling": {"warmup": 3, "valid_windows": 10, "replays_per_window": 1},\n',
    )
    source = replace_once(
        source,
        '        "historical_reference": frozen_baseline["cases"][case["id"]]["cell"],\n',
        '        "original_baseline_archive": "baseline.json; original SDK data retained unchanged",\n',
    )
    source = replace_once(
        source,
        '            report["ratio_to_frozen"] = (report["statistics_us"]["candidate"]["median"]\n'
        '                / report["historical_reference"]["median_us"])\n',
        '            report["ratio_to_frozen"] = (\n'
        '                report["statistics_us"]["candidate"]["median"] / reference_median\n'
        "                if reference_median is not None else None)\n",
    )
    ast.parse(source)
    return source


def main():
    here = Path(__file__).resolve().parent
    original = here / "runs/measurement_harness"
    target = here / "runs/sdk_measurement_harness_v1"
    shutil.copytree(original, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source = transform((original / "run_pto.py").read_text())
    (target / "run_pto.py").write_text(source)
    shutil.copy2(here / "measurement_provenance.py", target / "measurement_provenance.py")
    workspace = next(p for p in here.parents if (p / "env-dsv4-0251rc1.sh").is_file())
    sdk = workspace / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/manifest.json"
    shutil.copy2(sdk, target / "expected_sdk.json")
    report = dict(
        target=str(target),
        predecessor="measurement_harness.json",
        shared_toolchain=json.loads(sdk.read_text()),
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
        changed_files=["run_pto.py", "measurement_provenance.py", "expected_sdk.json"],
        measurement_boundary_unchanged=True,
        old_raw_reports_unchanged=True,
        candidate_windows=10,
        unmeasured_reference_replays=10,
        reject_cross_sdk_comparison=True,
    )
    with (here / "sdk_measurement_harness_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
