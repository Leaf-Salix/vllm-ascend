# SPDX-License-Identifier: Apache-2.0
"""Run a frozen correctness-only helper with source and toolchain checks."""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

from bsh_probe import check_toolchain, runtime_artifacts


def check_sources(directory):
    manifest = json.loads((directory / "source_manifest.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen helper source changed: {name}")
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--probe-directory", type=Path, required=True)
    parser.add_argument("--toolchain-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    check_sources(args.probe_directory)
    check_toolchain(args.toolchain_manifest)
    import pypto

    pypto.configure_cache(pypto.CacheConfig(enabled=False))
    artifacts = runtime_artifacts()
    probe = args.probe_directory / "probe.py"
    sys.path.insert(0, str(args.probe_directory))
    sys.argv = [str(probe), "--output", str(args.output), *forwarded]
    try:
        runpy.run_path(str(probe), run_name="__main__")
    finally:
        path = args.output / "result.json"
        if path.exists():
            report = json.loads(path.read_text())
            passed = report["passed"]
            report["passed"] = False
            try:
                report["frozen_source_files_verified"] = check_sources(args.probe_directory)
                report["toolchain_post_run_audit"] = check_toolchain(args.toolchain_manifest)
                report["runtime_artifacts"] = runtime_artifacts(require_loaded="--cpu-only" not in forwarded)
                if artifacts != report["runtime_artifacts"]:
                    raise ValueError("Compiler/runtime extensions changed during helper validation")
                report["passed"] = passed
                report["stage"] = "complete"
                report["full_csa_validated"] = False
                report["performance_measured"] = False
            except BaseException as exc:
                report["audit_error"] = str(exc)
                raise
            finally:
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
