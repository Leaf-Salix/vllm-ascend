# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Compiler-selected notification events must reach both sides of each flag."""

from types import SimpleNamespace

import pytest
from pypto import ir
from pypto.backend.pto_backend import _apply_task_boundary_events, _compiled_late_poll


def vector_helper():
    func = SimpleNamespace(attrs={"task_boundary_scratch_base": 0}, func_type=ir.FunctionType.AIV)
    return _compiled_late_poll(func)[0]


def test_no_allocation_keeps_existing_callback():
    helper = vector_helper()
    assert _apply_task_boundary_events("producer", helper, "") == (helper, "")


def test_allocated_ids_reach_both_notification_endpoints():
    metadata = "// PYPTO_TASK_BOUNDARY_EVENTS_V1 producer MTE2=6 MTE1=7 M=7 FIX=7 V=5"
    helper, contract = _apply_task_boundary_events("producer", vector_helper(), metadata)
    for pipe, event in (("MTE2", 6), ("V", 5)):
        for operation in ("set", "wait"):
            assert f"{operation}_flag(PIPE_{pipe}, PIPE_MTE3, EVENT_ID{event});" in helper
        assert f"PIPE_{pipe}, PIPE_MTE3, EVENT_ID7" not in helper
    assert contract == "#define PYPTO_TASK_BOUNDARY_EVENTS_V1 1\n"


@pytest.mark.parametrize(
    "metadata",
    [
        "// PYPTO_TASK_BOUNDARY_EVENTS_V1 producer MTE2=8 MTE1=7 M=7 FIX=7 V=5",
        "// PYPTO_TASK_BOUNDARY_EVENTS_V1 producer MTE2=6 MTE1=7 M=7 FIX=7 V=5\n" * 2,
    ],
)
def test_invalid_contract_is_not_silently_ignored(metadata):
    with pytest.raises(ValueError, match="invalid or duplicate"):
        _apply_task_boundary_events("producer", vector_helper(), metadata)


def test_metadata_requires_the_matching_callback():
    metadata = "// PYPTO_TASK_BOUNDARY_EVENTS_V1 producer MTE2=6 MTE1=7 M=7 FIX=7 V=5"
    with pytest.raises(ValueError, match="without a tail callback"):
        _apply_task_boundary_events("producer", "", metadata)


def test_other_member_allocation_does_not_change_current_callback():
    metadata = "// PYPTO_TASK_BOUNDARY_EVENTS_V1 other MTE2=6 MTE1=7 M=7 FIX=7 V=5"
    helper = vector_helper()
    updated, contract = _apply_task_boundary_events("producer", helper, metadata)
    assert updated == helper and contract
