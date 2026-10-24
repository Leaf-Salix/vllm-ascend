# SPDX-License-Identifier: Apache-2.0
"""Give four completed Hadamard owners the Q_B remainder columns."""

import ast
import copy
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "output_rectangle_production_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    name = "query_projection_stages.py"
    original = next(
        n for n in ast.parse(source[name]).body if isinstance(n, ast.FunctionDef) and n.name == "_q_b_nz_stage"
    )
    helper = copy.deepcopy(original)
    helper.name = "q_b_nz_column_range"
    helper.args.args = [arg for arg in helper.args.args if arg.arg != "QB_WIDE_N"]
    helper.args.defaults = []
    helper.args.args += [
        ast.arg(arg=n, annotation=ast.parse("pl.constexpr", mode="eval").body) for n in ("COLUMN_BEGIN", "COLUMN_END")
    ]
    branch = next(n for n in helper.body if isinstance(n, ast.If))
    helper.body = branch.orelse + [helper.body[-1]]
    loop = next(n for n in helper.body if isinstance(n, ast.For))
    loop.iter = ast.parse(
        "pl.range(0, (COLUMN_END - COLUMN_BEGIN - qproj_worker + WORKERS - 1) // WORKERS)", mode="eval"
    ).body
    assignment = next(n for n in loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "qproj_n_idx")
    assignment.value = ast.parse("COLUMN_BEGIN + qproj_worker + qproj_round * WORKERS", mode="eval").body
    source[name] += "\n\n" + ast.unparse(ast.fix_missing_locations(helper)) + "\n"
    name = "projection_overlap_stages.py"
    text = replace_once(
        source[name],
        "from .query_projection_stages import QUANT_WEIGHT_LAYOUT, q_b_stage\n",
        "from .query_projection_stages import (\n"
        "    QUANT_WEIGHT_LAYOUT, QUANT_WEIGHT_NZ, QPROJ_N_BLOCKS, q_b_stage, q_b_nz_column_range,\n)\n",
    )
    condition = "QUANT_WEIGHT_NZ and not QB_WIDE_N and BRANCH_GROUPS == 4"
    early = "        signal = publish_phase(signal, PHASE_QB, worker, CORES, generation)\n"
    position = text.index("    if worker < BRANCH_GROUPS:\n")
    after = text[position:]
    first = after.index(early)
    after = after[:first] + f"        if not ({condition}):\n    " + early + after[first + len(early) :]
    text = text[:position] + after
    marker = "        signal = publish_phase(signal, PHASE_IDX_QUERY_HAD, worker, CORES, generation)\n    else:\n"
    tail = (
        "        # Hadamard output is published before this independent remainder work.\n"
        f"        if {condition}:\n"
        "            q_proj = q_b_nz_column_range(\n"
        "                wq_b, qr_matmul, q_proj, tokens, worker, BRANCH_GROUPS,\n"
        "                (QPROJ_N_BLOCKS // (GROUPS - BRANCH_GROUPS)) * (GROUPS - BRANCH_GROUPS),\n"
        "                QPROJ_N_BLOCKS,\n"
        "            )\n"
        "            signal = publish_phase(signal, PHASE_QB, worker, CORES, generation)\n"
    )
    text = replace_once(text, marker, marker.removesuffix("    else:\n") + tail + "    else:\n")
    call = "        q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, qb_worker, GROUPS - BRANCH_GROUPS, QB_WIDE_N)\n"
    text = replace_once(
        text,
        call,
        f"        if {condition}:\n"
        "            q_proj = q_b_nz_column_range(\n"
        "                wq_b, qr_matmul, q_proj, tokens, qb_worker, GROUPS - BRANCH_GROUPS, 0,\n"
        "                (QPROJ_N_BLOCKS // (GROUPS - BRANCH_GROUPS)) * (GROUPS - BRANCH_GROUPS),\n"
        "            )\n"
        "        else:\n    " + call,
    )
    source[name] = text
    target = here / "runs/qb_tail_helpers_v1"
    target.mkdir(exist_ok=False)
    for name, text in source.items():
        ast.parse(text)
        (target / name).write_text(text)
    report = dict(
        source=str(target.resolve()),
        parent="output_rectangle_production_v1.json",
        adopted=False,
        device_execution=False,
        guard="Native NZ, N256 Q_B, four existing Hadamard owners; wide-N and other branch counts unchanged",
        mechanism="Main twenty AICs compute columns 0..119, six per core; four Hadamard AICs "
        "publish Indexer results then compute columns 120..127, two each",
        invariant="All 128 columns exactly once, original dynamic M loop and K256 accumulation, "
        "same existing all-24 QB consumer wait; helper QB publication moves after real output",
        risks="Helpers may start too late; their later KV/Score arrival may offset shorter Q_B tail; "
        "no new hardware participant or task is created",
        files={name: hashlib.sha256(text.encode()).hexdigest() for name, text in source.items()},
    )
    with (here / "qb_tail_helpers_v1.json").open("x") as out:
        out.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
