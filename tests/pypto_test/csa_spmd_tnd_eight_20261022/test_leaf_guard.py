# SPDX-License-Identifier: Apache-2.0
"""Check existing resource boundaries and preservation of all unrelated policies."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("tokens", [6, 31, 32, 48, 56, 63, 64, 65, 80, 95, 96, 97, 128, 144, 168, 192])
@pytest.mark.parametrize("history", [8192, 131072, 262144])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_guard_preserves_other_policy_and_large_t_baseline(tokens, history, version):
    here = Path(__file__).resolve().parent
    root = here / f"runs/leaf_scale_guard_{version}"
    if not root.is_dir():
        pytest.skip("Materialize the guarded candidate first")
    old = runpy.run_path(str(here / "runs/baseline/policy.py"))["select_policy"]
    new = runpy.run_path(str(root / "policy.py"))["select_policy"]
    kwargs = dict(tokens=tokens, requests=(tokens + 5) // 6, max_query_tokens=6, max_seq_len=history + 6)
    previous, current = old(**kwargs).constexprs(), new(**kwargs).constexprs()
    enabled = 64 <= tokens <= 96 and history > 32768
    if version == "v2":
        enabled = enabled and (history + 6) // 4 <= 6 * 8192
    assert current.pop("LEAF_SCALE_LAYOUT") == enabled
    assert current["SCALE_PAGE_COALESCING"] == (enabled or previous["SCALE_PAGE_COALESCING"])
    current["SCALE_PAGE_COALESCING"] = previous["SCALE_PAGE_COALESCING"]
    assert current == previous


@pytest.mark.parametrize("cache_rows", [8192, 8193, 32768, 32769, 49151, 49152, 49153, 65537])
def test_v2_base_leaf_history_boundary(cache_rows):
    root = Path(__file__).with_name("runs") / "leaf_scale_guard_v2"
    if not root.is_dir():
        pytest.skip("Materialize v2 first")
    select = runpy.run_path(str(root / "policy.py"))["select_policy"]
    policy = select(tokens=96, requests=16, max_query_tokens=6, max_seq_len=4 * cache_rows)
    assert policy.leaf_scale_layout == (8192 < cache_rows <= 49152)
