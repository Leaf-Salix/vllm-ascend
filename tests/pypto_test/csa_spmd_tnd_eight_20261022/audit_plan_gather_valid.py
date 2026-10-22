# SPDX-License-Identifier: Apache-2.0
"""Gate full-codegen/ABI/resources and retain original FFTS sites."""

import argparse
import ast
import json
import re
import runpy
from pathlib import Path

from analyze_priority_phase import executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-dir", type=Path, required=True)
    parser.add_argument("--reference-kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disabled", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = args.compile_dir
    assert json.loads((root / "result.json").read_text())["full_codegen_passed"]
    kernel = root / "kernel"
    abi = lambda p: json.loads((p / "compiled_meta.json").read_text())["kernel_abi"]
    assert abi(kernel) == abi(args.reference_kernel)
    report = dict(
        candidate="plan_gather_valid_v1",
        full_codegen_passed=True,
        device_execution=False,
        root_and_return_abi_unchanged=True,
        kernel=str(kernel),
        reference=str(args.reference_kernel),
    )
    if args.disabled:
        files = (
            "orchestration/decode_csa_spmd.cpp",
            "ptoas/dynamic_csa_full.cpp",
            "kernels/aic/cube_schedule.cpp",
            "kernels/aiv/vector_schedule.cpp",
        )
        assert all(
            executable((kernel / name).read_text()) == executable((args.reference_kernel / name).read_text())
            for name in files
        )
        report["disabled_four_cpp_equal"] = True
    else:
        before = (args.reference_kernel / "ptoas/dynamic_csa_full.cpp").read_text()
        after = (kernel / "ptoas/dynamic_csa_full.cpp").read_text()
        assert after.count("MGATHER<pto::Coalesce::Row>(") == 1
        assert after.count("MGATHER<pto::Coalesce::Row, pto::GatherOOB::Zero>(") == 1
        assert after.count("TROWMIN(") == before.count("TROWMIN(") + 1
        pattern = r"pto\.sync\.(set|wait) <(PIPE_\w+)>, %c(\d+)_index(?: \{ffts_mode = (\d+) : i32\})?"
        old = (args.reference_kernel / "ptoas/dynamic_csa_full.pto").read_text()
        new = (kernel / "ptoas/dynamic_csa_full.pto").read_text()
        assert sorted(re.findall(pattern, old)) == sorted(re.findall(pattern, new))
        memory = runpy.run_path(str(here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"))["memory"]
        (ir,) = (kernel / "passes_dump").glob("*after_AllocateMemoryAddr.py")
        limits = dict(vec=188416, mat=524288, acc=131072, left=65536, right=65536)
        resources = {}
        for fn in ast.walk(ast.parse(ir.read_text())):
            if not isinstance(fn, ast.FunctionDef) or fn.name not in ("cube_schedule", "vector_schedule"):
                continue
            maxima = {}
            for node in ast.walk(fn):
                if isinstance(node, ast.AnnAssign) and (region := memory(node.annotation)):
                    kind = region[0].strip("'").removeprefix("mem_").split("_")[0]
                    if kind in limits:
                        maxima[kind] = max(maxima.get(kind, 0), region[1] + region[2])
            assert all(value <= limits[kind] for kind, value in maxima.items())
            resources[fn.name] = maxima
        assert set(resources) == {"cube_schedule", "vector_schedule"}
        report.update(
            resources=resources,
            existing_ffts_sites_unchanged=True,
            undefined_reader_and_zero_fallback_emitted=True,
            additional_row_min_sites=1,
        )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
