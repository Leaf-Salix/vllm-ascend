# SPDX-License-Identifier: Apache-2.0
"""Compare prearmed and GM-written completion markers on one matching SDK."""

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

import torch
from pypto.backend import pto_backend
from tmr_csa_full_128k_b16_compile import PACKAGES, compile_variant

PACKAGES["previous_all"] = PACKAGES["all"]


def compile_comparison(output):
    # Keep the same current runtime/headers for every path. Only the compiler
    # submission capability and local Set adapter reproduce the previous ABI.
    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(Path(pto_backend.__file__).resolve().parents[3]),
            "show",
            "3d88caef9:python/pypto/backend/pto_backend.py",
        ],
        text=True,
    )
    previous_adapter = next(
        ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_LOCAL_TASK_BOUNDARY_ADAPTER" for target in node.targets)
    )
    original_codegen = pto_backend._codegen_core.generate_orchestration
    original_adapter = pto_backend._LOCAL_TASK_BOUNDARY_ADAPTER

    class PreviousSubmission:
        def __init__(self, result):
            self.result = result
            self.code = re.sub(r"^\s*\w+\.set_prearmed_async_tail\(true\);\n", "", result.code, flags=re.M)
            assert "set_prearmed_async_tail" not in self.code

        def __getattr__(self, key):
            return getattr(self.result, key)

    def previous_codegen(*args, **kwargs):
        return PreviousSubmission(original_codegen(*args, **kwargs))

    torch.set_num_threads(2)
    for name in ("baseline", "previous_all", "all"):
        try:
            if name == "previous_all":
                pto_backend._LOCAL_TASK_BOUNDARY_ADAPTER = previous_adapter
                pto_backend._codegen_core.generate_orchestration = previous_codegen
            compile_variant(name, output / name)
        finally:
            pto_backend._LOCAL_TASK_BOUNDARY_ADAPTER = original_adapter
            pto_backend._codegen_core.generate_orchestration = original_codegen


if __name__ == "__main__":
    if "--compile" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--compile", action="store_true")
        parser.add_argument("--output", type=Path, required=True)
        compile_comparison(parser.parse_args().output)
    else:
        from tmr_csa_full_128k_b16_demo import main

        main()
