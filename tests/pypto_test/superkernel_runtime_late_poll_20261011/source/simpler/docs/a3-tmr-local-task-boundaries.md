# Experimental A3 TMR local task boundaries

This private runtime supports compiler-checked local predecessor waits and
asynchronous task tails. It does not fuse tasks, release incomplete outputs,
or change an operator's tiling, arithmetic, or dependency edges. HBG is outside
this prototype. MIX uses the same physical-member completion protocol and
retains the existing group completion count.

## Dispatch and fallback

A checked Scalar prefix can overlap the tail of the task already running on
its normally selected pending core. The local task need not be an input
producer: local pipeline reuse and input readiness are separate conditions.
For example, an AIV consumer of an AIC result may overlap an unrelated AIV
task. Its Wait drains the local tail, publishes the local task's real FIN,
then awaits the consumer's full-fanin doorbell before ACK and computation.

Core selection, early-resolve marks, dispatch queues, and ACK handling are
unchanged. A gated idle core may execute its checked Scalar prefix before its
full-fanin doorbell; its local Wait has no predecessor to drain. A `sync_start`
successor also permits the prefix, while the unchanged cohort doorbell still
holds computation until every member owns its running slot and all inputs are
ready. Pending handoffs publish the local FIN before waiting for that doorbell,
so the original pending-to-running promotion can complete the rendezvous.

Predicated and unmarked entries retain their existing fallback. MIX capability
masks follow SubtaskSlot (AIC=1, AIV0=2, AIV1=4); one member's Wait or tail
never enables another member. Local boundaries are explicit; automatic prefix
recognition is not part of this demo. All gated tasks prepare arguments on the
receiving AICore as in ordinary early dispatch, including pending handoffs.
This Scalar preparation precedes the checked prefix and can overlap a local
predecessor's pipeline; it does not read the input tensors' data. Ready tasks
retain their CPU-prepared arguments. Neither idle prefix execution nor sync-start coverage
adds a dispatch trigger or makes an ACK visible ahead of the existing gate.

## Tail completion protocol

`GlobalContext.async_tail_kind` selects `AsyncTailKind` without changing the
context layout. Local Set uses RuntimeCompiledPoll when the compiler reserved its scratch and
event resources; otherwise it keeps RuntimeBarrierAll. Legacy event modes remain distinct. Event-based kinds use ID 7 and target Scalar:

| Kind | Value | Engines whose events must be consumed |
| --- | --- | --- |
| None | 0 | Ordinary synchronous return |
| AivJoined | 1 | V, MTE3; legacy compiler joins upstream pipelines |
| AicJoined | 2 | M, FIX; legacy compiler joins upstream pipelines |
| AivAllPipes | 3 | V, MTE3, MTE2 |
| AicAllPipes | 4 | M, FIX, MTE1, MTE2 |
| RuntimeBarrierAll | 5 | Runtime-owned PIPE_ALL after the successor's prefix, or on fallback |
| RuntimePollable | 6 | Retired runtime-DMA experiment; retains full-wait fallback |
| RuntimeCompiledPoll | 7 | Compiler-owned notification callback after the first handoff miss |

The all-pipes compiler tail does not reserve internal non-Scalar event pairs.
Earlier uses of the same engine-to-Scalar pair must already have been consumed
before Scalar reaches the tail. The executor consumes every listed event
before FIN, resource reuse, or dependent successor work. A missing successor
uses the same real completion wait; it never converts an early return to FIN.

Legacy kinds remain distinct so existing precompiled kernels are not made to
wait for events they never issued. AIV/AIC device probes and a full CSA run are
required to establish functionality and performance separately; successful
compilation or a capability marker alone proves neither overlap nor a gain.

For compiler-checked local Wait entries, the predecessor callback clears and
flushes the handoff fields before publishing completion. A synchronous successor
without tracing does not write back the same clean control line again. Legacy
entries and diagnostic writes retain their conservative writeback. The normal
tail path passes its already-read completion kind into the drain helper; the
completion waits, ACK flow, dispatch order, and early-resolve settings are unchanged.

Chip swimlane, PMU and argument dumping deliberately retain serial execution
and reporting. They are not used to measure the overlap path. The executor
first probes pending work once at producer return. A RuntimeCompiledPoll tail
that misses re-enters a compiler-generated tail callback using argument tag
bit 1. This branch bypasses the operator body, so it neither repeats arithmetic
nor resets predecessor pipeline state. It enqueues an ordered 64-byte DMA token
and observes normal late dispatches until that token completes. The helper never publishes FIN or ACK;
the unchanged local Wait or fallback drains PIPE_ALL first.

The compiler reserves scratch after existing cross-core FIFO storage and before
tiles, using the shared MemoryReuse/allocation resolver. PTOAS excludes event 7
from MTE2/MTE1/M/FIX/V-to-MTE3 allocations. Manually allocated pipe-event
functions are rejected for this experimental contract. The runtime owns three
cache lines per physical core (destination and immutable positive/negative
sources) in the DMA-visible HBM shared-memory allocation, before the first
ring segment. This adds 13.5 KiB on A3. Sources are flushed at cold start and
shared by the two dispatch slots. The first implementation stalled with both private SO storage and this HBM
allocation, so those failures do not establish an allocation-specific cause.
A separate kernel-side probe passed with allocator-selected scratch. The
compiled callback is the next isolated candidate; it requires device validation.
Alternating tokens need no per-task Scalar reset. DispatchPayload remains 576
bytes; neither argument preparation nor scheduler dispatch policy changes.

Diagnostic trace word 4 retains 7 in its low byte. Bit 8 records entry into
post-miss polling; bit 9 records a successful handoff observed during that
poll. Reported tail duration includes notification DMA and cannot by itself
prove overlap with the predecessor's original computation.
