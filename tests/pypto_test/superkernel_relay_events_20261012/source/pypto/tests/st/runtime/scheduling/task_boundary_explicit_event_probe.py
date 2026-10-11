# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Replay handoff producers whose existing synchronization is explicit PTO IR.

Serialize the normal sync pass output, remove only its generated tail/prefix
bookkeeping (the frontend regenerates it), then compile the explicit pipe flags.
This preserves the real load/compute/store dependencies and the original DSL.
"""

import re
from contextlib import contextmanager
from pathlib import Path

from pypto.backend import pto_backend

if __package__:
    from . import task_boundary_runtime_poll_probe as runtime_probe
else:
    import task_boundary_runtime_poll_probe as runtime_probe


@contextmanager
def explicit_sync(unit_names=("late_producer",), transform=None):
    original = pto_backend._compile_pto_module

    def compile_explicit(pto_code, unit_name, output_dir, memory_planner, dump_ptoas_passes=False):
        if unit_name not in unit_names:
            return original(pto_code, unit_name, output_dir, memory_planner, dump_ptoas_passes)
        root = Path(output_dir) / "explicit_sync" / unit_name
        root.mkdir(parents=True, exist_ok=True)
        source, scheduled = root / "input.pto", root / "scheduled.pto"
        source.write_text(pto_code)
        flags = [*pto_backend._get_ptoas_flags(memory_planner), "--emit-pto-ir"]
        pto_backend._run_ptoas(str(source), str(scheduled), flags)
        explicit = scheduled.read_text()
        assert "pto.runtime_late_handoff" in explicit and "pto.set_flag" in explicit
        explicit, tails = re.subn(
            r"^\s*pto.barrier <PIPE_ALL>[^\n]*pto.auto_sync_tail_barrier[^\n]*\n",
            "",
            explicit,
            flags=re.M,
        )
        assert tails == explicit.count("pto.runtime_late_handoff")
        explicit = re.sub(r"^\s*(?:func\.)?call @pypto_keep_prefix\([^\n]*\n", "", explicit, flags=re.M)
        explicit = re.sub(r"^\s*func.func private @pypto_keep_prefix\([^\n]*\n", "", explicit, flags=re.M)
        if transform is not None:
            explicit = transform(explicit)
        (root / "manual.pto").write_text(explicit)
        return original(explicit, unit_name, output_dir, memory_planner, dump_ptoas_passes)

    try:
        pto_backend._compile_pto_module = compile_explicit
        yield
    finally:
        pto_backend._compile_pto_module = original


@contextmanager
def instrument(mode, cube):
    with explicit_sync(), runtime_probe.instrument(mode, cube):
        yield


if __name__ == "__main__":
    runtime_probe.probe.instrument = instrument
    runtime_probe.probe.validate = runtime_probe.validate
    runtime_probe.probe.main()
