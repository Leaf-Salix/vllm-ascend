# SPDX-License-Identifier: Apache-2.0
"""模型加载后读取Worker实际绑定的PTO容量，不读设备tensor或改forward。"""

from pypto.runtime.kernel.context import get_process_kernel_state

from .observer import TimedObserver


def per_ring(value):
    return [value] * 4 if isinstance(value, int) else list(value) if value is not None else None


class RuntimeAuditObserver(TimedObserver):
    def offline_pto_window_config(self):
        config = get_process_kernel_state().config
        if config is None:
            raise RuntimeError("PTO window audit requires an initialized Worker")
        result = {
            "platform": config.platform,
            "runtime": config.runtime,
            "device_id": config.device_id,
            "ring_heap": per_ring(config.ring_heap),
            "ring_task_window": per_ring(config.ring_task_window),
            "ring_dep_pool": per_ring(config.ring_dep_pool),
        }
        if result["ring_task_window"] != [2048] * 4:
            raise RuntimeError(f"PTO task window did not take effect: {result}")
        if result["ring_heap"] != [256 << 20, 512 << 20, 256 << 20, 256 << 20]:
            raise RuntimeError(f"PTO heap changed during task-window experiment: {result}")
        return result
