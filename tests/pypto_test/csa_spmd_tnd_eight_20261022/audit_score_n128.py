# SPDX-License-Identifier: Apache-2.0
"""Audit N128 generated allocation, ABI and critical accumulator reuse edges.

This checks concrete generated dependencies, not general correctness of every
event in the kernel. Device output, repeated graph replay and timing are still
required before adoption.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path


def paired_flag(text, source, target):
    sets = re.findall(rf"set_flag\({source}, {target}, (EVENT_ID\d+)\);", text)
    return any(f"wait_flag({source}, {target}, {event});" in text for event in sets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compile_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = args.compile_dir.resolve()
    assert json.loads((root / "result.json").read_text())["full_codegen_passed"]
    helper = here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"
    spec = importlib.util.spec_from_file_location("existing_memory_audit", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (ir_path,) = (root / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py")
    limits = {"vec": 196608, "mat": 524288, "left": 65536, "right": 65536, "acc": 131072}
    resources = {}
    for fn in ast.walk(ast.parse(ir_path.read_text())):
        if not isinstance(fn, ast.FunctionDef) or fn.name not in ("cube_schedule", "vector_schedule"):
            continue
        maxima = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.AnnAssign) and (memory := module.memory(node.annotation)):
                kind = memory[0].removeprefix("mem_").split("_")[0]
                if kind in limits:
                    maxima[kind] = max(maxima.get(kind, 0), memory[1] + memory[2])
        for kind, maximum in maxima.items():
            assert maximum <= limits[kind], (fn.name, kind, maximum)
        resources[fn.name] = maxima
    cpp = root / "kernel/ptoas/dynamic_csa_full.cpp"
    code = cpp.read_text()
    lines = code.splitlines()
    accs = set(re.findall(r"Tile<TileType::Acc, int32_t, 192, 128,[^;\n]+> (v\d+);", code))
    assert len(accs) == 2, accs  # live <= 3 branch and live > 3 branch
    evidence = []
    for acc in sorted(accs):
        writes = [i for i, line in enumerate(lines) if re.search(rf"TMATMUL\({acc},", line)]
        assert len(writes) in (8, 16), (acc, writes)
        previous_insert = None
        for line_no in writes:
            insert = next(
                i for i in range(line_no + 1, len(lines)) if "TINSERT<" in lines[i] and f", {acc}," in lines[i]
            )
            assert paired_flag("\n".join(lines[line_no:insert]), "PIPE_M", "PIPE_FIX"), line_no
            if previous_insert is not None:
                assert paired_flag("\n".join(lines[previous_insert:line_no]), "PIPE_FIX", "PIPE_M"), line_no
            evidence.append(dict(acc=acc, matmul_line=line_no + 1, fix_insert_line=insert + 1))
            previous_insert = insert
    assert len(evidence) == 24
    # The reused WS Right storage must wait for the preceding first-half MMAD.
    right_edges = []
    for i, line in enumerate(lines):
        if not re.search(r"// pto: %split_right1(?:_last)?_inline", line):
            continue
        # Each declaration has duplicate comments; only retain the one before TASSIGN.
        stop = next(j for j in range(i + 1, len(lines)) if "TEXTRACT(" in lines[j])
        if any("// pto: %split_right1" in s for s in lines[i + 1 : stop]):
            continue
        previous_mmad = next(j for j in range(i - 1, -1, -1) if "TMATMUL(" in lines[j])
        segment = "\n".join(lines[previous_mmad:stop])
        assert paired_flag(segment, "PIPE_M", "PIPE_MTE1"), (i, stop)
        right_edges.append(dict(first_half_matmul_line=previous_mmad + 1, next_extract_line=stop + 1))
    assert len(right_edges) == 8, right_edges
    baseline = json.loads((here / "baseline.json").read_text())
    original = Path(baseline["cases"]["C04"]["source_report"]).parent / "build_output"
    (metadata,) = original.glob("decode_csa_spmd_*/compiled_meta.json")
    actual = json.loads((root / "kernel/compiled_meta.json").read_text())["kernel_abi"]
    assert actual == json.loads(metadata.read_text())["kernel_abi"]
    report = dict(
        passed=True,
        device_execution=False,
        packed_abi_unchanged=True,
        resources=resources,
        qk_reuse_edges=evidence,
        ws_right_reuse_edges=right_edges,
        ir=str(ir_path),
        cpp=str(cpp),
        cpp_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
        limitations=__doc__,
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=True, resources=resources, qk_writes=len(evidence), ws_reuses=len(right_edges))))


if __name__ == "__main__":
    main()
