# SPDX-License-Identifier: Apache-2.0
"""Audit active M64/N256 allocations and last-consumer reuse events."""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "csa_spmd_tnd_20261020"))
from audit_score_wide_k128 import memory, overlap  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compile_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.compile_dir
    assert json.loads((root / "result.json").read_text())["full_codegen_passed"]
    (ir_path,) = (root / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py")
    cube = next(
        n
        for n in ast.walk(ast.parse(ir_path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "cube_schedule"
    )
    records = {}
    for node in ast.walk(cube):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            region = memory(node.annotation)
            if region:
                records[node.target.id] = dict(region=region, line=node.lineno, node=node)

    def selected(prefix, suffix=None):
        return sorted(
            [
                (name, row)
                for name, row in records.items()
                if name.startswith(prefix) and (suffix is None or name.endswith(suffix))
            ],
            key=lambda item: item[1]["line"],
        )

    def one(prefix):
        rows = selected(prefix, "__ssa_v0")
        assert len(rows) == 1, (prefix, [name for name, _ in rows])
        return rows[0]

    qname, qrow = one("query_qk_acc_")
    wname, wrow = one("query_ws_acc_")
    qk, ws = qrow["region"], wrow["region"]
    assert (qk[2], ws[2]) == (65536, 16384)
    assert not overlap(qk, ws) and max(qk[1] + qk[2], ws[1] + ws[2]) <= 131072
    left = [one(f"query_left_{q}_")[1]["region"] for q in range(6)]
    left += [one(f"coefficient_left_{q}_")[1]["region"] for q in range(6)]
    assert [r[2] for r in left] == [8192] * 6 + [2048] * 6
    assert max(r[1] + r[2] for r in left) <= 65536
    assert not any(overlap(a, b) for i, a in enumerate(left) for b in left[i + 1 :])
    keys = selected("buf_current_key_")
    rights = [selected(f"query_right_{q}_") for q in range(6)]
    assert len(keys) == 4 and all(len(rows) == 4 for rows in rights)
    for panel in range(4):
        key = keys[panel][1]["region"]
        assert key[2] == 32768
        for q in range(6):
            right = rights[q][panel][1]["region"]
            assert right[2] == 32768 and right[1] + right[2] <= 65536
            if q < 5:
                assert not overlap(key, right), (panel, q)
            # The final WS may reuse this panel's key only after its final QK.

    cpp = (root / "kernel/ptoas/dynamic_csa_full.cpp").read_text().split("\nAICORE void vector_schedule", 1)[0]

    def variable(name, kind):
        start = cpp.index("// pto: %" + name + "\n")
        found = re.search(r"\b(v\d+) = Tile<TileType::" + kind, cpp[start : start + 1000])
        assert found, name
        return found.group(1)

    qvar, wvar = variable(qname, "Acc"), variable(wname, "Acc")
    qmm = list(re.finditer(r"\bTMATMUL\(" + qvar + r", [^;]+;", cpp))
    fixes = list(re.finditer(r"\bTINSERT[^\n]+\(v\d+, " + qvar + r", [^;]+;", cpp))
    winit = list(re.finditer(r"\bTMATMUL\(" + wvar + r", [^;]+;", cpp))
    wacc = list(re.finditer(r"\bTMATMUL_ACC\(" + wvar + r", [^;]+;", cpp))
    ws_ops = sorted(winit + wacc, key=lambda m: m.start())
    assert (len(qmm), len(fixes), len(winit), len(wacc)) == (24, 24, 4, 20)

    def extract(name):
        var = variable(name, "Right")
        matches = list(re.finditer(r"\bTEXTRACT\(" + var + r", [^;]+;", cpp))
        assert len(matches) == 1, name
        return matches[0]

    def reuse(begin, end, producer, consumer):
        assert begin < end
        between = cpp[begin:end]
        sets = set(re.findall(rf"set_flag\({producer}, {consumer}, (EVENT_ID\d+)\)", between))
        waits = set(re.findall(rf"wait_flag\({producer}, {consumer}, (EVENT_ID\d+)\)", between))
        assert sets & waits, (begin, end, producer, consumer)
        return sorted(sets & waits)

    reuse_records = []
    for p in range(4):
        for q in range(1, 6):
            current = p * 6 + q
            # Current FIX is issued before the previous query's weighted reduction.
            assert qmm[current].start() < fixes[current].start() < ws_ops[current - 1].start()
            reuse_records.append(
                dict(
                    resource="qk_acc",
                    panel=p,
                    query=q,
                    events=reuse(fixes[current - 1].end(), qmm[current].start(), "PIPE_FIX", "PIPE_M"),
                )
            )
        for q in range(1, 6):
            before = ws_ops[p * 6 + q - 1]
            next_read = extract(rights[q][p][0])
            previous = rights[q - 1][p][1]["region"]
            current = rights[q][p][1]["region"]
            if overlap(previous, current):
                reuse_records.append(
                    dict(
                        resource="ws_right",
                        panel=p,
                        query=q,
                        events=reuse(before.end(), next_read.start(), "PIPE_M", "PIPE_MTE1"),
                    )
                )
        if overlap(keys[p][1]["region"], rights[5][p][1]["region"]):
            reuse_records.append(
                dict(
                    resource="retired_key",
                    panel=p,
                    # The final Right is loaded from this QK's own FIX output:
                    # M -> FIX -> MTE1 also retires the old INT8 key read. Do not
                    # require a redundant direct M -> MTE1 edge on that path.
                    transitive_events=[
                        dict(
                            edge="M->FIX",
                            ids=reuse(qmm[p * 6 + 5].end(), fixes[p * 6 + 5].start(), "PIPE_M", "PIPE_FIX"),
                        ),
                        dict(
                            edge="FIX->MTE1",
                            ids=reuse(
                                fixes[p * 6 + 5].end(), extract(rights[5][p][0]).start(), "PIPE_FIX", "PIPE_MTE1"
                            ),
                        ),
                    ],
                )
            )
        if p and overlap(rights[5][p - 1][1]["region"], keys[p][1]["region"]):
            reuse_records.append(
                dict(
                    resource="next_key",
                    panel=p,
                    events=reuse(ws_ops[p * 6 - 1].end(), extract(keys[p][0]).start(), "PIPE_M", "PIPE_MTE1"),
                )
            )

    report = dict(
        passed=True,
        device_execution=False,
        ir=str(ir_path),
        qk_acc=qk,
        ws_acc=ws,
        left=left,
        right_keys=[r["region"] for _, r in keys],
        right_ws=[[r["region"] for _, r in rows] for rows in rights],
        counts=dict(qk=24, qk_fix=24, ws_init=4, ws_acc=20, panels=4),
        reuse_events=reuse_records,
        limitation="Allocation/ordering/event audit only; no measured overlap, precision or L1 speedup claim",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "device_execution", "qk_acc", "ws_acc", "counts")}))


if __name__ == "__main__":
    main()
