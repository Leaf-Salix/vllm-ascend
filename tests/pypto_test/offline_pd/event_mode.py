"""离线诊断 CANN 的进程级 event 模式；不修改 PyPTO/Simpler。"""

import ctypes


def get_event_work_mode():
    runtime = ctypes.CDLL("libruntime.so")
    function = runtime.rtEventWorkModeGet
    function.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    function.restype = ctypes.c_int
    value = ctypes.c_uint8()
    status = function(ctypes.byref(value))
    if status:
        raise RuntimeError(f"rtEventWorkModeGet failed: {status}")
    return value.value


def set_event_work_mode(mode):
    if mode not in (0, 1):
        raise ValueError("CANN event mode 必须为 0（软件）或 1（硬件）")
    runtime = ctypes.CDLL("libruntime.so")
    function = runtime.rtEventWorkModeSet
    function.argtypes = [ctypes.c_uint8]
    function.restype = ctypes.c_int
    status = function(mode)
    if status:
        raise RuntimeError(f"rtEventWorkModeSet({mode}) failed: {status}")
    if get_event_work_mode() != mode:
        raise RuntimeError("实际 CANN event 模式与请求不符")


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    import pypto.torch
    import torch_npu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", type=int, choices=(0, 1))
    args = parser.parse_args()
    torch_npu.npu.set_device(args.device)
    report = {"before": get_event_work_mode(), "requested": args.mode}
    if args.mode is not None:
        set_event_work_mode(args.mode)
    report["before_pypto_init"] = get_event_work_mode()
    pypto.torch.init(device=args.device, platform="a2a3")
    report["after_pypto_init"] = get_event_work_mode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
