"""可选的forward入场诊断；仅主机时钟和GC回调，不同步或修改设备工作。"""

import gc
import threading
import time


class ForwardHostDiagnostics:
    def __init__(self):
        self.events = []
        self.gc_enabled = gc.isenabled()
        self.gc_threshold = gc.get_threshold()
        self.gc_freeze_count = gc.get_freeze_count()
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

    def finish(self, entries):
        gc.callbacks.remove(self.callback)
        return {"scope": "主机观测单独列出，不从设备forward耗时中扣除；未改变GC配置、未新增设备同步。",
                "gc_enabled": self.gc_enabled, "gc_threshold": self.gc_threshold,
                "gc_freeze_count": self.gc_freeze_count, "gc_events": self.events,
                "steps": [{"step": e["index"], **e.get("host", {})} for e in entries]}
