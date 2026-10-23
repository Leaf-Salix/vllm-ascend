# SPDX-License-Identifier: Apache-2.0
"""Exercise the frozen measurement loop without a device or real graph."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_only_candidate_gets_ten_complete_windows(tmp_path):
    source = Path(__file__).with_name("runs") / "measurement_harness/run_pto.py"
    if not source.is_file():
        pytest.skip("Materialize the frozen measurement harness before running this artifact integration check")
    tree = ast.parse(source.read_text())
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and ast.unparse(n.iter) == "range(10)")
    state = dict(active=False, candidate=0, reference=0, measured=0, begins=0, ends=0)

    def replay(side):
        state[side] += 1
        if side == "reference":
            assert not state["active"]
        else:
            assert state["active"]
            state["measured"] += 1

    def begin():
        assert not state["active"]
        state["active"] = True
        state["begins"] += 1

    def end():
        assert state["active"]
        state["active"] = False
        state["ends"] += 1

    def export(source, destination, side, kernel_config):
        assert side == "candidate" and not state["active"]
        assert kernel_config == "candidate_config"
        return dict(raw=str(source), destination=str(destination), side=side)

    records = []
    scope = dict(
        graphs={s: SimpleNamespace(replay=lambda side=s: replay(side)) for s in ("reference", "candidate")},
        fixture=object(),
        restore=lambda _: None,
        records=records,
        stream=SimpleNamespace(synchronize=lambda: None),
        args=SimpleNamespace(output=tmp_path),
        pypto=SimpleNamespace(torch=SimpleNamespace(begin_dfx=begin, end_dfx=end)),
        export_l1=export,
        tables={"candidate": "candidate_config"},
    )
    exec(compile(ast.Module(body=[loop], type_ignores=[]), "measurement_loop", "exec"), scope)
    assert state == dict(active=False, candidate=10, reference=10, measured=10, begins=10, ends=10)
    assert len(records) == len({r["raw"] for r in records}) == 10
    assert len({r["destination"] for r in records}) == 10
