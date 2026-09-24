"""单卡回放一次真实 CSA 调用：验正确性、量单次耗时、可选采泳道。

存在的理由：改一处 kernel 就起 16 卡整模型（75 个权重分片、约 4.5 分钟加载）去看
它崩不崩、快不快，代价太高。本脚本吃 `offline_pd/run.py argdump` 落盘的真实根入参，
在**一张卡**上直接调 `decode_csa_tp1_attention_test`，一轮几十秒。

口径限制：
- 只跑 CSA 这一个算子，不含 MoE 与通信，绝对耗时不能当端到端性能；
  它衡量的是"这次 kernel 改动让该算子本身快了还是慢了"。
- 输入是某一步、某一层的快照，不随步数变化；用于对照而非覆盖所有档位。
- aicore 故障、越界这类问题会当场暴露，这是它相对 in-core 模拟器的价值。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from dsv4_csa_env import activate, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args-dir", type=Path, required=True, help="run.py argdump 落盘目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20, help="计时轮数（另有 3 轮预热）")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--swimlane", type=int, default=0, choices=range(5),
                        help="非 0 时开 DFX 芯片泳道；4 才有逐任务 kernel 时长")
    args = parser.parse_args()
    activate()

    import numpy as np
    import torch
    import torch_npu  # noqa: F401  加载 NPU 后端

    from pypto.runtime import RunConfig
    from vllm_ascend.ops.pypto.variant import selected_variant, variant_package

    package = variant_package()
    kernel = __import__(f"{package}.decode_csa", fromlist=["decode_csa_tp1_attention_test"]).decode_csa_tp1_attention_test

    meta = json.loads((args.args_dir / "csa_args_meta.json").read_text())
    blob = np.load(args.args_dir / "csa_args.npz")
    names = list(kernel.param_names)
    if names != meta["param_names"]:
        raise ValueError("落盘入参的参数表与当前 kernel 不一致，需重新 argdump")

    device = f"npu:{args.device}"
    torch.npu.set_device(args.device)
    tensors = {name: torch.from_numpy(blob[name]).to(device) for name in names}
    call_args = tuple(tensors[name] for name in names)

    args.output.mkdir(parents=True, exist_ok=True)
    config = RunConfig(platform="a2a3", device_id=args.device,
                       enable_chip_swimlane=args.swimlane,
                       enable_dep_gen=args.swimlane >= 4)

    report = {"variant": selected_variant(), "package": package, "device": args.device,
              "layer_index": meta["layer_index"], "tokens": meta["tokens"],
              "iters": args.iters, "warmup": args.warmup, "swimlane_level": args.swimlane,
              "scope": "单算子单卡回放，不含 MoE 与通信；绝对耗时不代表端到端性能"}
    try:
        for _ in range(args.warmup):
            kernel(*call_args, config=config)
        torch.npu.synchronize()

        samples = []
        for _ in range(args.iters):
            torch.npu.synchronize()
            start = time.perf_counter()
            kernel(*call_args, config=config)
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1e6)
        samples.sort()
        report.update(status="PASS",
                      us_min=samples[0], us_p50=statistics.median(samples),
                      us_p90=samples[int(len(samples) * 0.9) - 1], us_max=samples[-1],
                      samples_us=samples)
        out = tensors["attn_out"].detach().float().cpu()
        report["attn_out"] = {"finite": bool(torch.isfinite(out).all()),
                              "absmax": float(out.abs().max()), "mean": float(out.mean())}
    except BaseException as exc:  # 故障也要留证据
        report.update(status="FAIL", error=repr(exc))
        write_json(args.output / "report.json", report)
        raise
    write_json(args.output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "samples_us"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
