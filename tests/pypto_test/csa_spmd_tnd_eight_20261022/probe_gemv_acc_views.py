# SPDX-License-Identifier: Apache-2.0
"""Check public-API workarounds for the GEMV Acc view mismatch, without NPU."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import pypto
from pypto import passes
from pypto.runtime import CompileOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "static_slice",
            "set_validshape",
            "inferred_acc",
            "parent_validshape",
            "parent_narrowed",
            "parent_narrowed_named",
        ),
        default=("static_slice", "set_validshape"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    original = (here / "score_gemv_alias_repro.py").read_text()
    original = original.split('if __name__ == "__main__":', 1)[0]
    assemble = "        pool = pl.tile.assemble(pool, result, [0, i * 64])"
    assert original.count(assemble) == 1
    variants = {
        "static_slice": original.replace(
            assemble, "        result = pl.tile.slice(result, [16, 64], [0, 0], valid_shape=[1, 64])\n" + assemble
        ),
        "set_validshape": original.replace(
            assemble, "        result = pl.tile.set_validshape(result, 1, 64)\n" + assemble
        ),
        "inferred_acc": original.replace(
            "pl.tile.create([16, 128], pl.FP32, target_memory=pl.MemorySpace.Acc)",
            "pl.tile.create([16, 128], pl.FP32)",
        ),
        "parent_validshape": original.replace(
            "    for i in pl.unroll(2):",
            "    pool = pl.tile.set_validshape(pool, 16, 128)\n    for i in pl.unroll(2):",
        ),
        "parent_narrowed": original.replace(
            "    for i in pl.unroll(2):",
            "    for i in pl.unroll(2):\n        pool = pl.tile.set_validshape(pool, 1, 128)",
        ),
        "parent_narrowed_named": original.replace(
            "        slot = pl.tile.slice(pool,",
            "        pool_live = pl.tile.set_validshape(pool, 1, 128)\n        slot = pl.tile.slice(pool_live,",
        ),
    }
    pypto.configure_cache(pypto.CacheConfig(enabled=False))
    report = dict(
        device_execution=False,
        sdk_modified=False,
        original_repro="score_gemv_alias_repro.py",
        original_failure="runs/score_gemv_alias_repro_v2/report/codegen_errors.txt",
        variants={},
    )
    for name in args.variants:
        source = variants[name]
        script = args.output / f"{name}.py"
        script.write_text(source)
        row = dict(source=str(script), source_sha256=hashlib.sha256(source.encode()).hexdigest())
        report["variants"][name] = row
        try:
            entry = runpy.run_path(str(script))["entry"]
            config = CompileOptions(
                platform="a2a3",
                output_dir=str(args.output / name),
                dump_passes=True,
                memory_planner=passes.MemoryPlanner.PYPTO,
            )
            artifact = entry._resolve_kernel_artifact((), {"config": config}, allow_signature_mode=True)
            root = Path(artifact.output_dir)
            errors = list((root / "report").glob("*errors*"))
            assert not errors, "\n".join(p.read_text() for p in errors)
            pto_files = list((root / "ptoas").glob("*.pto"))
            cpp_files = list((root / "ptoas").glob("*.cpp"))
            assert pto_files and cpp_files, "Missing generated PTO/CPP"
            pto = "\n".join(p.read_text() for p in pto_files)
            moves = [s.split(" loc(")[0] for s in pto.splitlines() if "pto.tmov" in s]
            acc_moves = [s for s in moves if "loc=acc" in s]
            assert not acc_moves, acc_moves
            from pypto.jit._artifact_manifest import BuildKind
            from pypto.runtime._prebuilt import prepare_prebuilt

            prepare_prebuilt(root, "a2a3", BuildKind.SINGLE_CHIP, kernel_abi=artifact.kernel_abi)
            row.update(compiled=True, binary_compiled=True, acc_moves=acc_moves, artifact=str(root))
        except Exception as exc:
            row.update(compiled=False, binary_compiled=False, error=str(exc))
        print(json.dumps({name: row}, ensure_ascii=False), flush=True)
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
