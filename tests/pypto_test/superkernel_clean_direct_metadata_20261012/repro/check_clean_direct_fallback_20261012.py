"""Compile a real explicit 0/7 task and verify it retains its event-six callback."""
import json
import sys
from contextlib import contextmanager
from pathlib import Path

private = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(private / 'pypto/tests/st/runtime/scheduling'))
import task_boundary_allocated_event_probe as allocated
import task_boundary_late_arrival_probe as probe
from task_boundary_clean_direct_probe import clean_direct_notification

event_instrument = allocated.make_instrument('dynamic')

@contextmanager
def instrument(mode, cube):
    with event_instrument(mode, cube), clean_direct_notification():
        yield

probe.instrument = instrument
output = private / 'artifacts/clean_direct_fallback_20261012'
probe.artifact(False, 'late', output, True)
source = next((output / 'build/kernels').rglob('*late_producer*.cpp')).read_text()
orch = next((output / 'build/orchestration').glob('*.cpp')).read_text()
start = source.index('void pypto_poll_local_tail(')
end = source.index('\n}\n', start)
helper = source[start:end]
assert 'set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6)' in helper
assert 'wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6)' in helper
assert 'set_runtime_compiled_poll(true)' in orch
assert 'set_runtime_direct_poll_scratch(' not in orch
result = {'status': 'PASS', 'target': 'AIV', 'original_events': [0, 7], 'notification_mte2_event': 6,
          'compiled_callback_retained': True, 'direct_metadata_absent': True}
(output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
