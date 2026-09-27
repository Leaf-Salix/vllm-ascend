"""源头分离对照：每个布局同次加载对比常规与计时外缓存压力。"""

import ctypes
import json
import statistics
import sys
from pathlib import Path

ACL_DEV_ATTR_L2_CACHE_SIZE = 302  # CANN 9.0 acl_rt.h；单位bytes。


def main():
    source = Path(sys.argv.pop(1)).resolve()
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    sys.path[:0] = [str(source / "tests/pypto_test"), str(source)]
    import dsv4_csa_single_layer as runner

    original = runner.measure_graph_interval
    measurements = {}
    allocation = {}

    def measure(fixture, *args, **kwargs):
        import torch

        side = ("native", "pto")[len(measurements)]
        if not allocation:
            library = ctypes.CDLL("libascendcl.so")
            query = library.aclrtGetDeviceInfo
            query.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
            query.restype = ctypes.c_int
            size = ctypes.c_int64()
            device = torch.npu.current_device()
            error = query(device, ACL_DEV_ATTR_L2_CACHE_SIZE, ctypes.byref(size))
            if error or size.value <= 0:
                raise RuntimeError(f"aclrtGetDeviceInfo(L2)失败：error={error}, size={size.value}")
            allocation.update(device=device, l2_bytes=size.value, pressure_bytes=2 * size.value,
                              writes=0, buffer=torch.empty(2 * size.value, dtype=torch.uint8,
                                                          device=fixture["hidden"].device))

        def pressure():
            allocation["writes"] += 1
            allocation["buffer"].fill_(allocation["writes"] % 256)

        warm = original(fixture, *args, **kwargs)
        pressured = original(fixture, *args, **kwargs, before_timing=pressure)
        measurements[side] = {"normal": warm, "pressure": pressured,
                              "normal_mean_us": statistics.mean(warm["samples_us"]),
                              "pressure_mean_us": statistics.mean(pressured["samples_us"])}
        return warm

    runner.measure_graph_interval = measure
    try:
        runner.main()
    finally:
        report = {
            "operator_base_revision": "2a740c1f", "source_checkout": str(source),
            "status": "MEASURED" if len(measurements) == 2 else "INCOMPLETE",
            "scope": "同次加载先常规20次、后压力20次；每组预热5次，两侧相同；压力写入在设备开始事件之前。",
            "limits": "压力buffer是API报告L2容量的2倍；未测硬件命中率，不能称作保证冷缓存，也不模拟完整EP16。",
            "allocation": {k: v for k, v in allocation.items() if k != "buffer"},
            "measurements": measurements,
        }
        (output / "cache_pressure.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
