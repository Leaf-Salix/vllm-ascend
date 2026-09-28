# SPDX-License-Identifier: Apache-2.0
"""Use the frozen reference host harness to test the relocated BSH kernel.

This tests kernel relocation, not the Leaf production model integration.
Source the reference environment first. All remaining arguments are forwarded
unchanged to its dsv4_csa_single_layer.py diagnostic.
"""

import argparse
import importlib
import importlib.util
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkout", type=Path, required=True)
    parser.add_argument(
        "--candidate-package",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "vllm_ascend/attention/pto_kernels/dspark_layer",
    )
    parser.add_argument("--variant", choices=("performance",), default="performance")
    args, forwarded = parser.parse_known_args()
    tests = args.reference_checkout.resolve() / "tests/pypto_test"
    package = args.candidate_package.resolve()
    script = tests / "dsv4_csa_single_layer.py"
    if not script.is_file() or not (package / "decode_csa.py").is_file():
        parser.error("reference diagnostic and candidate decode_csa.py must exist")
    sys.path.insert(0, str(tests))
    environment = importlib.import_module("dsv4_csa_env")
    original_activate = environment.activate

    def activate_then_load():
        # Preserve source-selection checks before importing any PyPTO module.
        selected = original_activate()
        spec = importlib.util.spec_from_file_location(
            "leaf_dspark_layer", package / "__init__.py", submodule_search_locations=[str(package)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        root = importlib.import_module("leaf_dspark_layer.decode_csa")
        if Path(root.__file__).resolve() != package / "decode_csa.py":
            raise RuntimeError(f"Candidate module was shadowed: {root.__file__}")
        reference_module = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_csa"
        sys.modules[reference_module] = root
        print("CANDIDATE_KERNEL_SOURCE", root.__file__, flush=True)
        return selected

    environment.activate = activate_then_load
    sys.argv = [str(script), "--variant", args.variant, *forwarded]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        environment.activate = original_activate


if __name__ == "__main__":
    main()
