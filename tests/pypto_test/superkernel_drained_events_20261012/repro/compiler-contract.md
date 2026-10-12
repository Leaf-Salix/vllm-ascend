# Local task-tail event reuse

The private TMR handoff prototype keeps the author's local
`wait_pre_task_end()` boundary. This allocation rule does not identify a
Scalar boundary or change dispatch, ACK, or `allow_early_resolve`.

For `pto.runtime_late_handoff` functions with explicit synchronization, tail
notification normally needs event IDs unused by the original function. A
textually matched Set/Wait alone does not make an ID reusable: the destination
pipeline may not have consumed its Wait yet.

An existing top-level `pto.barrier <PIPE_ALL>` can retire matched static events
in a single-block function. The allocator can reuse those IDs for the later
notification. It preserves the original barrier and adds no device instruction.
A later event use reserves its ID again. Unmatched, repeated-unmatched, dynamic,
nested, and macro-hidden uses remain conservatively reserved.

A single-pipe barrier is insufficient. A barrier marked
`pto.auto_sync_tail_barrier` is also excluded because the runtime can defer that
barrier until after the notification. If neither direct nor relay events are
available, the existing blocking callback fallback remains.

This follows the documented [SetFlag/WaitFlag semantics](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.html)
and [PIPE_ALL completion guarantee](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/PipeBarrier_ISASI.html).
The proof relies on both matching event consumption and the retained drain;
the barrier does not consume an unmatched Set.

Regression: `test/lit/pto/local_task_pollable_drained_events.pto`.
