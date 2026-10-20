# SPDX-License-Identifier: Apache-2.0
"""队列内复用上游完整 MIX 回归，记录本机版本与精确检查结果。

每个进程仅使用一种 runtime；不改生成代码，不替换 ISA，不改上游比较条件。
所有 24 AIC / 48 AIV 执行，业务 Tile 跨越预取提交与等待保持存活。
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", choices=("tensormap_and_ringbuffer", "host_build_graph"), required=True)
    parser.add_argument("--scenario", choices=("eager", "replay"), required=True)
    parser.add_argument("--issuers", choices=("none", "single", "all"), default="all")
    parser.add_argument("--post-count", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    if int(os.environ["TASK_DEVICE"]) != args.device:
        raise RuntimeError("设备参数必须与队列分配一致")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    logs = out / "ascend"
    logs.mkdir()
    os.environ["ASCEND_PROCESS_LOG_PATH"] = str(logs)
    import pypto
    import torch
    import torch_npu
    from pypto import _kernel_abi

    source = Path(pypto.__file__).resolve().parents[2] / "tests/st/runtime/kernel/test_mix_prefetch.py"
    spec = importlib.util.spec_from_file_location("csa_sdma_upstream_mix_probe", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = {
        "passed": False,
        "device_execution": True,
        "performance_measured": False,
        "aic_roles": 24,
        "aiv_roles": 48,
        "upstream_test": str(source),
        "runtime": args.runtime,
        "scenario": args.scenario,
        "issuers": args.issuers,
        "post_count": args.post_count,
        "rounds": args.rounds,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "ascend_home": os.environ.get("ASCEND_HOME_PATH"),
        "pypto": pypto.__file__,
        "simpler_abi": _kernel_abi.SIMPLER_KERNEL_REVISION,
        "generated_code_replaced": False,
    }
    try:
        module._run(args.device, str(out), args.runtime, args.scenario, args.issuers,
                    rounds=args.rounds, post_count=args.post_count)
        result["upstream_result"] = json.loads((out / "result.json").read_text())
        result["passed"] = True
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        (out / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
