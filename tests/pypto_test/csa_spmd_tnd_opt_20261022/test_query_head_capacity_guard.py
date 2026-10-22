# SPDX-License-Identifier: Apache-2.0
"""Capacity specialization does not alter request math or other policy choices."""

import hashlib
import runpy
from pathlib import Path

from evaluate_nine_cases import EXPECTED_CASES

HERE = Path(__file__).resolve().parent


def test_only_policy_source_changes_and_other_constexprs_are_preserved():
    parent = HERE / "runs/query_head_guard"
    candidate = HERE / "runs/query_head_capacity_guard"
    changed = [
        p.name
        for p in parent.glob("*.py")
        if hashlib.sha256(p.read_bytes()).digest() != hashlib.sha256((candidate / p.name).read_bytes()).digest()
    ]
    assert changed == ["policy.py"]
    old = runpy.run_path(str(parent / "policy.py"))["select_policy"]
    new = runpy.run_path(str(candidate / "policy.py"))["select_policy"]
    for query in (1, 2, 4, 5, 6):
        for requests in (1, 4, 8, 12, 14, 16, 17, 24, 28, 32):
            for tokens in sorted({1, requests * query, min(64, requests * query), min(80, requests * query)}):
                for history in (8192, 32768, 131072):
                    kw = dict(tokens=tokens, requests=requests, max_query_tokens=query, max_seq_len=history + query)
                    before, after = old(**kw).constexprs(), new(**kw).constexprs()
                    enabled = before.pop("QUERY_CONTIGUOUS_HEADS") and tokens == requests * query
                    assert after.pop("QUERY_CONTIGUOUS_HEADS") == enabled
                    assert after == before


def test_known_regressions_are_closed_and_non_b16_full_capacity_remains_testable():
    select = runpy.run_path(str(HERE / "runs/query_head_capacity_guard/policy.py"))["select_policy"]
    for tokens, requests, enabled in ((64, 16, False), (80, 16, False), (72, 12, True), (96, 16, True)):
        policy = select(tokens=tokens, requests=requests, max_query_tokens=6, max_seq_len=131078)
        assert policy.constexprs()["QUERY_CONTIGUOUS_HEADS"] is enabled
    activated = []
    for history, batch, queries in EXPECTED_CASES:
        policy = select(
            tokens=sum(queries), requests=batch, max_query_tokens=max(queries), max_seq_len=history + max(queries)
        )
        if policy.constexprs()["QUERY_CONTIGUOUS_HEADS"]:
            activated.append((history, batch, queries))
    assert activated == [(131072, 16, (6,) * 16)]
