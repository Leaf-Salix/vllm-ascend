"""可选的forward入场诊断；仅主机时钟和GC回调，不同步或修改设备工作。"""

import gc
import threading
import time
from contextlib import contextmanager


class ForwardHostDiagnostics:
    def __init__(self):
        self.events = []
        self.gc_enabled = gc.isenabled()
        self.gc_threshold = gc.get_threshold()
        self.gc_freeze_count = gc.get_freeze_count()
        self.runner = None
        self.original_methods = []
        self.callback = self._gc_event
        gc.callbacks.append(self.callback)

    def _gc_event(self, phase, info):
        # Do not scan objects or log in the hot path: either could cause stalls.
        self.events.append({"phase": phase, "generation": info["generation"],
                            "monotonic_ns": time.monotonic_ns(), "thread_id": threading.get_native_id(),
                            "collected": info.get("collected", 0)})

    @staticmethod
    def mark(entry, name):
        entry.setdefault("host", {})[name] = {
            "monotonic_ns": time.monotonic_ns(), "thread_cpu_ns": time.thread_time_ns()}

    def attach_runner(self, runner, active_entry):
        """Separate existing waits from preparation without adding any sync."""
        self.runner = runner

        def install(name, wrapped, original):
            self.original_methods.append((name, name in vars(runner), original))
            setattr(runner, name, wrapped)

        def timed(original, label):
            def call(*args, **kwargs):
                entry = active_entry()
                if entry is None:
                    return original(*args, **kwargs)
                self.mark(entry, f"{label}_begin")
                try:
                    return original(*args, **kwargs)
                finally:
                    self.mark(entry, f"{label}_end")
            return call

        for name, label in (
            ("_prepare_inputs", "inputs"),
            ("_determine_batch_execution_and_padding", "batch_coordination"),
            ("_build_attention_metadata", "attention_metadata"),
            ("_preprocess", "preprocess"),
        ):
            original = getattr(runner, name, None)
            if callable(original):
                install(name, timed(original, label), original)

        original_sync = getattr(runner, "synchronize_input_prep", None)
        if callable(original_sync):
            @contextmanager
            def input_prep(*args, **kwargs):
                entry = active_entry()
                if entry is not None:
                    self.mark(entry, "input_sync_begin")
                with original_sync(*args, **kwargs):
                    if entry is not None:
                        self.mark(entry, "input_sync_end")
                    try:
                        yield
                    finally:
                        if entry is not None:
                            self.mark(entry, "input_prep_end")
            install("synchronize_input_prep", input_prep, original_sync)

    def finish(self, entries):
        for name, instance_attribute, original in reversed(self.original_methods):
            if instance_attribute:
                setattr(self.runner, name, original)
            else:
                delattr(self.runner, name)
        self.original_methods.clear()
        self.runner = None
        gc.callbacks.remove(self.callback)
        return {"scope": "主机观测单独列出，不从设备forward耗时中扣除；未改变GC配置、未新增设备同步。",
                "gc_enabled": self.gc_enabled, "gc_threshold": self.gc_threshold,
                "gc_freeze_count": self.gc_freeze_count, "gc_events": self.events,
                "steps": [{"step": e["index"], **e.get("host", {})} for e in entries]}
