# SPDX-License-Identifier: Apache-2.0
"""Reassign eight O_B jobs using full/tail O_A row costs in the one-short-tail tier."""

import ast
import hashlib
import json
import runpy
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    baseline = json.loads((here / "baseline.json").read_text())
    source = {p.name: p.read_text() for p in (here / "runs/baseline").glob("*.py")}
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()} == baseline["files"]
    filename = "policy.py"
    source[filename] = replace_once(
        source[filename],
        "    output_work_balance: bool = False\n",
        "    output_work_balance: bool = False\n    output_tail_balance: bool = False\n",
    )
    source[filename] = replace_once(
        source[filename],
        '            "OUTPUT_WORK_BALANCE": self.output_work_balance,\n',
        '            "OUTPUT_WORK_BALANCE": self.output_work_balance,\n'
        '            "OUTPUT_TAIL_BALANCE": self.output_tail_balance,\n',
    )
    source[filename] = replace_once(
        source[filename],
        "    return CSASPMDPolicy(\n",
        "    return CSASPMDPolicy(\n"
        "        output_tail_balance=(long_history and group == 6 and 128 < tokens <= 160),\n",
    )
    for name in ("entry.py", "cube_schedule.py"):
        declaration = "    OUTPUT_WORK_BALANCE: pl.constexpr = False,\n"
        assert source[name].count(declaration) == (2 if name == "entry.py" else 1)
        source[name] = source[name].replace(
            declaration, declaration + "    OUTPUT_TAIL_BALANCE: pl.constexpr = False,\n"
        )
    call = "            OUTPUT_WORK_BALANCE,\n"
    assert source["entry.py"].count(call) == 2
    source["entry.py"] = source["entry.py"].replace(call, call + "            OUTPUT_TAIL_BALANCE,\n")
    filename = "cube_schedule.py"
    tree = ast.parse(source[filename])
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "OUTPUT_WORK_BALANCE")
    assert len(branch.orelse) == 1 and isinstance(branch.orelse[0], ast.For)
    loop = branch.orelse[0]
    lines = source[filename].splitlines(keepends=True)
    previous = lines[loop.lineno - 1 : loop.end_lineno]
    new_loop = """for ob_group in pl.range(8):
    ob_first = ob_group * 8
    ob_owned = worker + ((ob_first + GROUPS - 1 - worker) // GROUPS) * GROUPS
    if ob_group < 2:
        if worker >= 20:
            ob_owned = ob_first + worker - 20
        elif ob_owned < ob_first + 4:
            ob_owned = ob_first + 8
    if ob_owned < ob_first + 8:
        signal = wait_row_owners(
            signal, PHASE_O_QUANT_GROUP_BASE + ob_group, CORES, generation,
            ob_group * quant_per_group, quant_per_group, 1, VECTORS, GROUPS, 1,
        )
        o_partials = o_b_groups_stage(
            o_quant, wo_b, o_partials, tokens, ob_owned, 64, OB_ROWS, ob_group, ob_group + 1,
        )
"""
    nested = [" " * loop.col_offset + "if OUTPUT_TAIL_BALANCE:\n"]
    nested += [" " * (loop.col_offset + 4) + line + "\n" for line in new_loop.splitlines()]
    nested += [" " * loop.col_offset + "else:\n"]
    nested += ["    " + line for line in previous]
    lines[loop.lineno - 1 : loop.end_lineno] = nested
    source[filename] = "".join(lines)
    target = here / "runs/output_tail_balance_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        ast.parse(content)
        (target / name).write_text(content)
    select = runpy.run_path(str(target / "policy.py"))["select_policy"]
    case = baseline["cases"]["C05"]["case"]
    policy = select(
        tokens=case["token_capacity"],
        requests=case["request_capacity"],
        max_query_tokens=6,
        max_seq_len=max(case["seq_lens"]),
    ).constexprs()
    assert policy["OUTPUT_TAIL_BALANCE"] and not policy["OUTPUT_WORK_BALANCE"]
    original = json.loads(Path(baseline["cases"]["C05"]["source_report"]).read_text())
    (here / "descriptors/C05_output_tail_balance_v1.json").write_text(
        json.dumps(dict(buffers=original["descriptors"]["spmd"], policy=policy), indent=2) + "\n"
    )
    report = dict(
        source=str(target),
        baseline="baseline.json",
        adopted=False,
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        changes=["policy.py", "entry.py", "cube_schedule.py"],
        guard="Long G6, one full M128 plus 1..32 tail rows (128<T<=160)",
        mechanism="Move group0/1 dc0..3 O_B jobs from eight full-row O_A owners to four short-tail owners 20..23",
        unchanged="All 64 output jobs, per-group quant waits, O_A/quant/math helpers and all-24 PHASE_OB publication",
        difference_from_old_v78="Old 16-heavy/8-light scheme ignored full/tail row costs; this moves only eight jobs",
        hypothesis="Reduce final O_B stragglers; rank64 helper overhead may offset the change",
    )
    with (here / "output_tail_balance_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
