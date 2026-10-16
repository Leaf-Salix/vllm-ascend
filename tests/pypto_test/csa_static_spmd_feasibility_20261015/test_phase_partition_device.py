# SPDX-License-Identifier: Apache-2.0
"""验证 48+12 / 32+32 KiB 分阶段 L0A 复用的设备数值，不是 CSA 测试。"""

import argparse
import json
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

import torch
from pypto.ir import CompiledProgram
from pypto.runtime import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"scope": "分阶段 L0A arena 扩展", "passed": False,
              "device_execution": False, "full_csa_validated": False}
    try:
        compiled = CompiledProgram.from_dir(args.kernel)
        # 小整数使 FP16 输入及 FP32 点积参考均精确可表示。
        x = ((torch.arange(496 * 128).reshape(496, 128) % 7) - 3).half()
        w = ((torch.arange(128 * 32).reshape(128, 32) % 5) - 2).half()
        out = torch.full((496, 32), float("nan"))
        reference = x.float() @ w.float()
        compiled(x, w, out, config=RunConfig(platform="a2a3", device_id=args.device,
                                            save_kernels=True,
                                            save_kernels_dir=str(args.output / "run")))
        result["device_execution"] = True
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        result.update(passed=True, rows_checked=[192, 48, 128, 128], max_abs=0.0)
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "traceback"},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
