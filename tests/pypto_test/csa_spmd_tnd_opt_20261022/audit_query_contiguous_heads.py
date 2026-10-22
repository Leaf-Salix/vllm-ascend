# SPDX-License-Identifier: Apache-2.0
"""Check GM view roots/strides, unchanged ABI and the emitted vector resources."""

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "csa_spmd_tnd_20261020"))
from audit_score_wide_k128 import memory  # noqa: E402


def inspect(root):
    assert json.loads((root / "result.json").read_text())["full_codegen_passed"]
    (path,) = (root / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py")
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "vector_schedule")
    annotations = {
        n.target.id: n.annotation
        for n in ast.walk(fn)
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }
    annotations.update({n.arg: n.annotation for n in fn.args.args})
    regions = {name: memory(annotation) for name, annotation in annotations.items()}
    vec = [r for r in regions.values() if r and r[0].startswith("mem_vec_")]
    return path, fn, annotations, regions, max(r[1] + r[2] for r in vec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compile_dir", type=Path)
    parser.add_argument(
        "--baseline-compile",
        type=Path,
        required=True,
        help="Any v122 compile with unchanged Query body and identical descriptor",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, baseline = args.compile_dir.resolve(), args.baseline_compile.resolve()
    path, fn, annotations, regions, maximum = inspect(root)
    base_path, _, _, _, base_maximum = inspect(baseline)
    assert maximum <= 196608
    meta = json.loads((root / "kernel/compiled_meta.json").read_text())
    base_meta = json.loads((baseline / "kernel/compiled_meta.json").read_text())
    assert meta == base_meta, "GM views must not add roots, parameters, returns or execution modes"
    views = []
    for n in ast.walk(fn):
        if not isinstance(n, ast.AnnAssign) or not isinstance(n.target, ast.Name):
            continue
        if not n.target.id.startswith(("q_head_input_", "q_head_output_", "q_head_weights_")):
            continue
        assert ast.unparse(n.value.func) == "pl.tensor.view"
        origin = n.value.args[0].id
        assert regions[n.target.id] == regions[origin]
        (layout,) = [
            x for x in ast.walk(n.annotation) if isinstance(x, ast.Call) and ast.unparse(x.func) == "pl.TensorView"
        ]
        stride = next(k.value for k in layout.keywords if k.arg == "stride")
        assert ast.literal_eval(stride) == [512, 1]
        views.append(dict(name=n.target.id, origin=origin, region=regions[origin], stride=[512, 1], line=n.lineno))
    assert len(views) == 3
    scalar = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id.startswith("qr_scale_scalar_")
    ]
    assert scalar and all(ast.unparse(n.annotation) == "pl.Scalar[pl.FP32]" for n in scalar)
    assert all(ast.unparse(n.value.func) == "pl.tensor.read" for n in scalar)
    loads = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id.startswith(("q_head_acc_", "q_head_scale_"))
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.tile.load"
        and n.value.args[0].id.startswith(("q_head_input_", "q_head_weights_"))
    ]
    assert loads and all(ast.literal_eval(n.value.args[2]) == [8, 512] for n in loads)
    report = dict(
        passed=True,
        device_execution=False,
        ir=str(path),
        baseline_ir=str(base_path),
        exact_compiled_meta_equal=True,
        gm_views=views,
        vector_address_max=maximum,
        baseline_vector_address_max=base_maximum,
        emitted_contiguous_load_sites=len(loads),
        device_fp32_scale_read_sites=len(scalar),
        precision="Not yet device validated; activation MULS then per-element weight MUL preserve FP32 order",
        scope="Static ABI/address/operation audit; not a performance claim",
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "exact_compiled_meta_equal", "vector_address_max")}))


if __name__ == "__main__":
    main()
