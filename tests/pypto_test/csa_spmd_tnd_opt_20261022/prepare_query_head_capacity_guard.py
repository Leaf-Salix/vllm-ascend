# SPDX-License-Identifier: Apache-2.0
"""Stage a capacity-based refinement, pending the non-B16 full-S6 boundary."""

import hashlib
import json
import runpy
import shutil
from pathlib import Path


def transform(source):
    old = "            and EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS\n"
    assert source.count(old) == 1
    return source.replace(
        old,
        old + "            # Capacity resource choice only; device Q/L still governs actual requests.\n"
        "            and tokens == requests * max_query_tokens\n",
        1,
    )


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/query_head_guard"
    expected = json.loads((here / "query_head_guard.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.glob("*.py")} == expected
    target = here / "runs/query_head_capacity_guard"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "policy.py").write_text(transform((parent / "policy.py").read_text()))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")}
    assert [name for name in files if files[name] != expected[name]] == ["policy.py"]
    old_policy = runpy.run_path(str(parent / "policy.py"))["select_policy"]
    policy = runpy.run_path(str(target / "policy.py"))["select_policy"]
    for tokens, requests in ((72, 12), (96, 16)):
        kw = dict(tokens=tokens, requests=requests, max_query_tokens=6, max_seq_len=131078)
        assert policy(**kw).constexprs() == old_policy(**kw).constexprs()
    for tokens in (64, 80):
        kw = dict(tokens=tokens, requests=16, max_query_tokens=6, max_seq_len=131078)
        new, old = policy(**kw).constexprs(), old_policy(**kw).constexprs()
        assert new.pop("QUERY_CONTIGUOUS_HEADS") is False and old.pop("QUERY_CONTIGUOUS_HEADS") is True
        assert new == old
    descriptor = json.loads((here / "runs/query_head_guard_t80_g6/buffer_descriptors.json").read_text())
    actual = policy(tokens=80, requests=16, max_query_tokens=6, max_seq_len=131078).constexprs()
    before = dict(descriptor["policy"])
    before["QUERY_CONTIGUOUS_HEADS"] = False
    assert before == actual
    descriptor["policy"] = actual
    with (here / "descriptors/query_head_capacity_guard_t80.json").open("x") as stream:
        stream.write(json.dumps(descriptor, indent=2) + "\n")
    report = dict(
        source=str(target),
        parent="query_head_guard",
        modified_files=["policy.py"],
        files=files,
        adopted=False,
        device_submitted=False,
        domain="existing long G6 64..96 capacity tier and token_capacity=request_capacity*max_query_tokens",
        dynamic_request_math="unchanged; no actual query lengths are inferred from capacity",
        rationale="B16 S6 normal and same-address evidence; partial-capacity T64/T80 regressions; B12 S6 still pending",
        pending="Non-B16 full-capacity boundary and disabled T80 generated-code comparison",
    )
    with (here / "query_head_capacity_guard.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
