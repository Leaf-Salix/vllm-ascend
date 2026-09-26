"""Lower the complete integration CSA chain on CPU; no device or numerical claim."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dsv4_csa_env import activate, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    activate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    argv_before = tuple(sys.argv)
    report = {"scope": "CPU full-chain lowering; no device execution"}
    try:
        from pypto.runtime import RunConfig

        from vllm_ascend.ops.pypto.variant import selected_variant, variant_package
        package = variant_package()
        config = __import__(f"{package}.config", fromlist=["config"])
        report["variant"] = selected_variant()

        assert tuple(sys.argv) == argv_before, "kernel import changed service argv"
        assert config.TP == 1 and config.DECODE_SEQ == 6
        decode_csa_tp1_layer_test = __import__(
            f"{package}.decode_csa", fromlist=["decode_csa_tp1_layer_test"]
        ).decode_csa_tp1_layer_test

        kernels = [decode_csa_tp1_layer_test]
        report["kernels"] = []
        for kernel in kernels:
            program = kernel.lower(config=RunConfig(platform="a2a3"))
            name = kernel.__name__
            (args.output_dir / f"{name}_lowered.py").write_text(str(program))
            report["kernels"].append(
                {"name": name, "parameters": kernel.param_names, "mutable_and_output": kernel.output_param_names}
            )
        report.update(
            status="PASS",
            tp=config.TP,
            query=config.DECODE_SEQ,
            argv_unchanged=True,
        )
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        write_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
