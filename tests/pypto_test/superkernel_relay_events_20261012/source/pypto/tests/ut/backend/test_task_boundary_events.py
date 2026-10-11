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


def test_no_free_event_uses_event_free_drain_only_in_callback():
    marker = "// PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1 producer"
    helper, contract = _apply_task_boundary_events("producer", vector_helper(), marker)
    assert "pipe_barrier(PIPE_ALL)" in helper
    assert "_flag(" not in helper and "copy_" not in helper and "dcci(" not in helper
    assert "#define PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1 1" in contract


def test_blocking_poll_is_scoped_to_its_member():
    marker = "// PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1 other"
    original = vector_helper()
    helper, contract = _apply_task_boundary_events("producer", original, marker)
    assert helper == original and contract


@pytest.mark.parametrize(
    "metadata",
    [
        "// PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1 producer\n" * 2,
        "// PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1 producer extra",
        "// PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1 producer\n"
        "// PYPTO_TASK_BOUNDARY_EVENTS_V1 producer MTE2=6 MTE1=7 M=7 FIX=7 V=5",
    ],
)
def test_malformed_or_conflicting_blocking_contract_fails_closed(metadata):
    with pytest.raises(ValueError, match="invalid|duplicate|both"):
        _apply_task_boundary_events("producer", vector_helper(), metadata)


def test_blocking_contract_requires_callback():
    with pytest.raises(ValueError, match="without a tail callback"):
        _apply_task_boundary_events("producer", "", "// PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1 producer")


@pytest.mark.parametrize("cube", [False, True], ids=["vector", "cube"])
def test_relay_joins_every_pipe_before_notification(cube):
    kind = ir.FunctionType.AIC if cube else ir.FunctionType.AIV
    helper = _compiled_late_poll(SimpleNamespace(attrs={"task_boundary_scratch_base": 0}, func_type=kind))[0]
    pairs = (
        (("MTE2", "MTE1"), ("MTE1", "M"), ("M", "FIX"), ("FIX", "MTE3"))
        if cube
        else (
            ("MTE2", "V"),
            ("V", "MTE3"),
        )
    )
    marker = "// PYPTO_TASK_BOUNDARY_RELAY_V1 producer " + " ".join(
        f"{src}_TO_{dst}={i}" for i, (src, dst) in enumerate(pairs)
    )
    updated, contract = _apply_task_boundary_events("producer", helper, marker)
    previous = updated.index("create_cbuf_matrix" if cube else "vector_dup")
    for event, (src, dst) in enumerate(pairs):
        for op in ("set", "wait"):
            position = updated.index(f"{op}_flag(PIPE_{src}, PIPE_{dst}, EVENT_ID{event});")
            assert previous < position
            previous = position
    assert previous < updated.index("pipe_barrier(PIPE_MTE3)") < updated.index("copy_")
    assert "PIPE_ALL" not in updated
    assert updated.count("set_flag(") == updated.count("wait_flag(") == len(pairs)
    assert contract == "#define PYPTO_TASK_BOUNDARY_RELAY_V1 1\n"


@pytest.mark.parametrize(
    "metadata",
    [
        "// PYPTO_TASK_BOUNDARY_RELAY_V1 producer MTE2_TO_V=8 V_TO_MTE3=7",
        "// PYPTO_TASK_BOUNDARY_RELAY_V1 producer V_TO_MTE3=7",
        "// PYPTO_TASK_BOUNDARY_RELAY_V1 producer MTE2_TO_V=7 V_TO_MTE3=7\n" * 2,
        "// PYPTO_TASK_BOUNDARY_RELAY_V1 producer MTE2_TO_MTE1=7 MTE1_TO_M=7 M_TO_FIX=7 FIX_TO_MTE3=7",
        "// PYPTO_TASK_BOUNDARY_RELAY_V1 producer MTE2_TO_V=7 V_TO_MTE3=7\n"
        "// PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1 producer",
        "// PYPTO_TASK_BOUNDARY_RELAY_V1 producer MTE2_TO_V=7 V_TO_MTE3=7\n"
        "// PYPTO_TASK_BOUNDARY_EVENTS_V1 producer MTE2=6 MTE1=7 M=7 FIX=7 V=7",
    ],
)
def test_malformed_or_conflicting_relay_is_rejected(metadata):
    with pytest.raises(ValueError, match="invalid|duplicate|physical core|both"):
        _apply_task_boundary_events("producer", vector_helper(), metadata)


def test_relay_applies_only_to_its_member_and_requires_complete_callback():
    marker = "// PYPTO_TASK_BOUNDARY_RELAY_V1 producer MTE2_TO_V=6 V_TO_MTE3=7"
    original = vector_helper()
    updated, contract = _apply_task_boundary_events("other", original, marker)
    assert updated == original and contract
    with pytest.raises(ValueError, match="without a tail callback"):
        _apply_task_boundary_events("producer", "", marker)
    with pytest.raises(ValueError, match="incomplete direct"):
        _apply_task_boundary_events(
            "producer", original.replace("    set_flag(PIPE_V,", "    invalid("), marker
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
