# SPDX-License-Identifier: Apache-2.0
"""单卡测量公开运行时配置的设备内存；不加载模型或提交CSA kernel。"""

import argparse
import json
import os
import time
from pathlib import Path

import pypto
import pypto.torch
import simpler
import torch
import torch_npu
from pypto.runtime.kernel.context import get_process_kernel_state


def memory():
    torch.npu.synchronize()
    free, total = torch.npu.mem_get_info()
    return {
        "free_bytes": free,
        "total_bytes": total,
        "torch_allocated_bytes": torch.npu.memory_allocated(),
        "torch_reserved_bytes": torch.npu.memory_reserved(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task-window", type=int)
    parser.add_argument("--dep-pool", type=int)
    args = parser.parse_args()
    assert not args.output.exists()
    assert os.environ["TASK_DEVICE"] in ("8", "9", "10", "11", "12", "13", "15")
    assert os.environ["PYTHONNOUSERSITE"] == "1"
    for module, variable in ((pypto, "CSA_SHARED_PYPTO_ROOT"), (simpler, "CSA_SHARED_SIMPLER_ROOT")):
        assert Path(os.environ[variable]).resolve() in Path(module.__file__).resolve().parents
    assert os.environ["PTOAS_ROOT"] == os.environ["CSA_SHARED_PTOAS_ROOT"]
    assert os.environ["ASCEND_HOME_PATH"] and get_process_kernel_state().config is None
    torch.npu.set_device(0)
    before = memory()
    start = time.monotonic()
    ring_heap = (256 << 20, 512 << 20, 256 << 20, 256 << 20)
    pypto.torch.init(
        device=0,
        platform="a2a3",
        runtime="tensormap_and_ringbuffer",
        ring_heap=ring_heap,
        ring_task_window=args.task_window,
        ring_dep_pool=args.dep_pool,
    )
    after = memory()
    result = {
        "complete": True,
        "scope": "runtime init only; no model, CSA launch, token or performance qualification",
        "physical_device": os.environ["TASK_DEVICE"],
        "sdk_release": os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"],
        "pypto": pypto.__file__,
        "simpler": simpler.__file__,
        "ring_heap_bytes": ring_heap,
        "ring_task_window": args.task_window,
        "ring_dep_pool": args.dep_pool,
        "before": before,
        "after": after,
        "device_delta_bytes": before["free_bytes"] - after["free_bytes"],
        "torch_delta_bytes": after["torch_allocated_bytes"] - before["torch_allocated_bytes"],
        "elapsed_s": time.monotonic() - start,
        "torch_npu": torch_npu.__version__,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
