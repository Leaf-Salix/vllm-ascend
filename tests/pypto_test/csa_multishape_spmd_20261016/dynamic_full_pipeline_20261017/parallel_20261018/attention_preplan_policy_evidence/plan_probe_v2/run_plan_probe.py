# SPDX-License-Identifier: Apache-2.0
"""修复后的独立plan探针接口回归；不加载权重，不计性能，不代替完整CSA。"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import dsv4_csa_env
    dsv4_csa_env.activate()
    import pypto.torch
    import torch
    import torch_npu
    from pypto import passes
    from pypto.runtime import CompileOptions
    device_id = int(os.environ["TASK_DEVICE"])
    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    pypto.torch.init(device=device_id, platform="a2a3", runtime="tensormap_and_ringbuffer")
    from plan_device_probe import validate
    report = validate(device, CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO),
                      args.output / "result.json")
    report["scope"] = "独立plan后置路径：页保护及descriptor精确回归，非完整CSA、非性能测量"
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"])}))


if __name__ == "__main__":
    main()
