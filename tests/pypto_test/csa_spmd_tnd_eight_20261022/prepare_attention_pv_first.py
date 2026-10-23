# SPDX-License-Identifier: Apache-2.0
"""Run the two-ticks-old PV before acquiring the current KV block on AIC."""

import ast
import hashlib
import json
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "output_rectangle_production_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    name = "attention_stream_stages.py"
    tree = ast.parse(source[name])
    cube = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "attention_qk_pv_cube_stage")
    loop = next(n for n in cube.body if isinstance(n, ast.For))
    assert len(loop.body) == 2 and all(isinstance(n, ast.If) for n in loop.body)
    qk, pv = loop.body
    assert ast.unparse(qk.test) == "qk_tick < qk_block_count"
    assert ast.unparse(pv.test) == "qk_tick >= QK_PRE_LAUNCH"
    lines = source[name].splitlines(keepends=True)
    first = lines[qk.lineno - 1 : qk.end_lineno]
    second = lines[pv.lineno - 1 : pv.end_lineno]
    lines[qk.lineno - 1 : pv.end_lineno] = [
        "        # Both operations retain their original work IDs and events.\n"
        "        # Consume older P/KV before waiting for the current KV producer.\n",
        *second,
        *first,
    ]
    source[name] = "".join(lines)
    ast.parse(source[name])
    target = here / "runs/attention_pv_first_v1"
    target.mkdir(exist_ok=False)
    for name, text in source.items():
        (target / name).write_text(text)
    report = dict(
        source=str(target.resolve()),
        parent="output_rectangle_production_v1.json",
        adopted=False,
        device_execution=False,
        mechanism="AIC executes PV(tick-2) before KV_READY/QK(tick); AIV is unchanged",
        unchanged="Both branch bodies, all operands/math/events/masks, three slots, work numbering, warmup and drain",
        possible_benefit="Expose useful Cube work before waiting for the current AIV gather",
        risks=[
            "P(tick-2) may be late; moving its wait can delay QK",
            "Program order plus paired events must remain acyclic for holes and cross-query reuse",
            "Compiler L1/L0 dependencies and actual overlap require emitted-code and runtime checks",
        ],
        files={name: hashlib.sha256(text.encode()).hexdigest() for name, text in source.items()},
    )
    with (here / "attention_pv_first_v1.json").open("x") as out:
        out.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
