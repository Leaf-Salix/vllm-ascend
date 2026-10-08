"""Lower the complete integration CSA chain on CPU; no device or numerical claim."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dsv4_csa_env import activate, write_json
from dsv4_csa_replay import argument_roles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build", action="store_true", help="同时完成 PTOAS/CCE 编译；不初始化 NPU")
    args = parser.parse_args()
    activate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    argv_before = tuple(sys.argv)
    report = {
        "scope": "CPU full-chain compilation; no device execution"
        if args.build
        else "CPU full-chain lowering; no device execution"
    }
    try:
        from pypto.runtime import RunConfig

        from vllm_ascend.ops.pypto.variant import selected_variant, variant_package

        package = variant_package()
        config = __import__(f"{package}.config", fromlist=["config"])
        report["variant"] = selected_variant()

        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.nz_mode import root_weight_layouts
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.reduction import ATOMIC_ADD

        assert tuple(sys.argv) == argv_before, "kernel import changed service argv"
        assert config.TP == 1 and config.DECODE_SEQ == 6
        module = __import__(f"{package}.decode_csa", fromlist=["decode_csa_tp1_layer_test"])
        decode_csa_tp1_layer_test = module.decode_csa_tp1_layer_test
        report.update(root_layouts=root_weight_layouts(module._decode_csa_tp1_layer), atomic_add=ATOMIC_ADD)

        kernels = [decode_csa_tp1_layer_test]
        report["kernels"] = []
        for kernel in kernels:
            if args.build:
                compiled = kernel.warmup(
                    config=RunConfig(
                        platform="a2a3", save_kernels=True, save_kernels_dir=str((args.output_dir / "build").resolve())
                    )
                )
                program = compiled.program
                report["device_binaries"] = "built_without_device_execution"
            else:
                program = kernel.lower(config=RunConfig(platform="a2a3"))
            name = kernel.__name__
            (args.output_dir / f"{name}_lowered.py").write_text(str(program))
            report["kernels"].append(
                {
                    "name": name,
                    "parameters": list(argument_roles(getattr(module, name))),
                    "mutable_and_output": kernel.output_param_names,
                }
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
