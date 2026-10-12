# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Reuse consumed static events without changing the author's local boundary."""

import json
import re
from pathlib import Path

import pytest
from pypto.backend import pto_backend
from pypto.runtime import RunConfig

from . import test_local_task_boundary_mix as mixed
from . import test_local_task_boundary_overlap as pure
from .task_boundary_explicit_event_probe import explicit_sync


def _drained_prefix(explicit: str, core: str) -> str:
    def add_events(match):
        # Exhaust the shared last edge of the direct and relay routes, then
        # consume the events before any original load/compute/store is issued.
        cube = core == "aic" or match["name"].endswith("_aic")
        source = "PIPE_FIX" if cube else "PIPE_V"
        events = "".join(
            f"    pto.{operation}_flag[<{source}>, <PIPE_MTE3>, <EVENT_ID{event_id}>]\n"
            for event_id in range(8)
            for operation in ("set", "wait")
        )
        return match[0] + events + "    pto.barrier <PIPE_ALL>\n"

    explicit, count = re.subn(
        r"^[ \t]*func.func @(?P<name>\w+)\(.*?"
        r"^[ \t]*(?:func\.)?call @pypto_wait_pre_task_end\([^\n]*\n",
        add_events,
        explicit,
        flags=re.M | re.S,
    )
    assert count == (2 if core == "mix" else 1)
    return explicit


@pytest.mark.platforms("a2a3")
@pytest.mark.parametrize("core", ["aic", "aiv", "mix"])
def test_drained_events_keep_nonblocking_tail(test_config: RunConfig, tmp_path: Path, monkeypatch, core: str):
    original_events = pto_backend._apply_task_boundary_events
    checked = set()

    def check_callback(func_name, helper, ptoas_code):
        helper, contract = original_events(func_name, helper, ptoas_code)
        assert "PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1" not in contract
        assert "PYPTO_TASK_BOUNDARY_RELAY_V1" not in contract
        assert "pipe_barrier(PIPE_ALL)" not in helper
        checked.add(func_name)
        return helper, contract

    monkeypatch.setattr(pto_backend, "_apply_task_boundary_events", check_callback)
    units = ("mixed_producer", "mixed_consumer") if core == "mix" else ("boundary_probe_producer",)
    with explicit_sync(units, lambda text: _drained_prefix(text, core)):
        if core == "mix":
            mixed.test_mixed_members_enter_prefix_before_group_completion(
                test_config, tmp_path, monkeypatch, relay_events=False, nested_wait=False
            )
        else:
            pure.test_local_boundary_overlap(
                test_config, tmp_path, cube=core == "aic", branch_wait=False, monkeypatch=monkeypatch
            )
    expected = {f"mixed_{task}_{worker}" for task in ("producer", "consumer") for worker in ("aic", "aiv")}
    assert checked == (expected if core == "mix" else {"boundary_probe_producer", "boundary_probe_successor"})
    if core == "mix" and not test_config.codegen_only:
        output = Path(test_config.save_kernels_dir or tmp_path) / "mixed_chain" / "mixed_chain_results.json"
        samples = json.loads(output.read_text())["samples"]
        assert all(any(s["notifications_by_member"][worker::2]) for worker in (0, 1) for s in samples), (
            "both physical core types must execute the callback using the reclaimed event"
        )
