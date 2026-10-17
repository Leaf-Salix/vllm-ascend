# SPDX-License-Identifier: Apache-2.0
"""隔离G2尾组：正式Cube/AIV helper在live=2与live=1时第一query应完全一致。

此探针不重写设备数学，直接复用已有动态score编译壳及正式实现。CPU构造两条
可区分的INT8 query、Native key/scale页和对角head系数；仅更新Q/L、描述符、
第二query的有效性，同址执行2→1→2。第一query的输入、可见历史与叶划分不变，
所以第一query的完整score与两个Top512根必须逐bit一致。另以精确INT32点积
和显式FP16舍入作辅助算术核对；不测试性能，不依赖Native整层的其他数值误差。
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, choices=(2, 6), default=2)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    args = parser.parse_args()
    if "TASK_DEVICE" not in os.environ or args.device != int(os.environ["TASK_DEVICE"]):
        raise RuntimeError("必须使用task-submit分配的设备")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, performance_measured=False, device_execution=False,
                  group_size=args.group_size, cases=[])
    try:
        import dsv4_csa_env
        dsv4_csa_env.activate()
        import torch
        import torch_npu
        import pypto.torch
        from pypto import passes
        from pypto.runtime import CompileOptions
        from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs

        source_dir = Path(__file__).resolve().parent.parent / "dynamic_indexer_score_20261017"
        sys.path.insert(0, str(source_dir))
        import cpu_probes

        source = source_dir.parents[2].parent / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_score_stages.py"
        (args.output / "indexer_score_stages.py").write_text(source.read_text())
        (args.output / "score_tail_probe.py").write_text(Path(__file__).read_text())
        (args.output / "cpu_probes.py").write_text((source_dir / "cpu_probes.py").read_text())

        torch.npu.set_device(args.device)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        device = torch.device(f"npu:{args.device}")
        pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer",
                         **ring_sizing_kwargs())
        options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        # 固定seed及二进制可精确表示的系数，避免参考归约本身制造末位噪声。
        generator = torch.Generator().manual_seed(20261017)
        group = args.group_size
        pages = 64 if group == 2 else 65
        query = torch.randint(-10, 11, (group * 64, 128), dtype=torch.int8, generator=generator)
        keys = torch.randint(-7, 8, (pages * 32, 128), dtype=torch.int8, generator=generator)
        scale = (torch.arange(pages * 32) % 7 + 1).to(torch.float16) / 64
        coeff = torch.zeros((16, group * 64), dtype=torch.float16)
        for lane in range(group):
            coeff[lane, lane * 64:(lane + 1) * 64] = (torch.arange(64) % (8 - lane) + 1).half() / (8 + 8 * lane)
        native = torch.zeros((pages, 4160), dtype=torch.int8)
        native[:, :4096] = keys.reshape(pages, 4096)
        native[:, 4096:] = scale.reshape(pages, 32).view(torch.int8)
        dots = query[:64].to(torch.int32) @ keys[:2048].to(torch.int32).T
        relu_half = torch.clamp_min(dots.float() / 1024, 0).half().float()
        expected = (relu_half * coeff[0, :64].float()[:, None]).sum(0) * scale[:2048].float()
        query_device, coefficients, native_device = (v.to(device) for v in (query, coeff, native))
        flat = native_device.flatten()
        key0 = flat[:flat.numel() // 128 * 128].view(-1, 128)
        key64 = flat[64:64 + (flat.numel() - 64) // 128 * 128].view(-1, 128)
        table = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)
        positions = torch.arange(8192, 8192 + group, dtype=torch.int64, device=device)
        lengths = torch.tensor([8192 + group], dtype=torch.int32, device=device)
        bounds = torch.tensor([0, group], dtype=torch.int32, device=device)
        offsets_cpu = torch.zeros(32, dtype=torch.int32)
        offsets_cpu[1] = 1
        offsets_cpu[16:20] = torch.tensor([0, 0, group, (8192 + group) // 4])
        offsets = offsets_cpu.to(device)
        plan = torch.tensor([1, (8192 + group) // 4, 1, 8, 0, group, 0, 2], dtype=torch.int32, device=device)
        saved = {"expected": expected}
        for tile in ((768, 1024) if group == 2 else (768,)):
            kernel = getattr(cpu_probes, f"g{group}_{tile}_probe")
            transfer = torch.full((24 * 2 * group, tile), 37.0, device=device)
            ffts = torch.zeros(256, dtype=torch.int64, device=device)
            scores = torch.full((24 * 2 * group, 8192), 37.0, device=device)
            pairs = torch.full((group * 64, 1024), 37.0, device=device)
            tensors = (query_device, coefficients, key0, key64, native_device, table, positions,
                       lengths, bounds, offsets, plan, transfer, ffts, scores, pairs)
            addresses = tuple(v.data_ptr() for v in tensors)
            baseline = None
            for iteration, live in enumerate((group, *range(1, group), group)):
                bounds.copy_(torch.tensor([0, live], dtype=torch.int32))
                lengths.copy_(torch.tensor([8192 + live], dtype=torch.int32))
                offsets_cpu[18] = live
                offsets_cpu[19] = (8192 + live) // 4
                offsets.copy_(offsets_cpu)
                current_coeff = coeff.clone()
                current_coeff[live:].zero_()
                coefficients.copy_(current_coeff)
                transfer.fill_(37)
                scores.fill_(37)
                pairs.fill_(37)
                kernel(*tensors, config=options)
                torch.npu.synchronize()
                report["device_execution"] = True
                values = scores.cpu()
                span = (((8192 + live) // 4 + tile - 1) // tile) * (tile // 2)
                observed = torch.cat((values[0, :span], values[1, :2048 - span]))
                roots = pairs[:2].cpu()
                if baseline is None:
                    baseline = observed.clone(), roots.clone()
                row = dict(tile=tile, live=live, iteration=iteration,
                           score_differences=int((observed != baseline[0]).sum()),
                           root_differences=int((roots != baseline[1]).sum()),
                           cpu_differences=int((observed != expected).sum()),
                           cpu_max_abs=float((observed - expected).abs().max()),
                           finite=bool(torch.isfinite(observed).all()),
                           same_addresses=addresses == tuple(v.data_ptr() for v in tensors))
                row["passed"] = (row["score_differences"] == row["root_differences"] == row["cpu_differences"] == 0
                                 and row["finite"] and row["same_addresses"])
                report["cases"].append(row)
                saved[f"tile{tile}_round{iteration}"] = dict(score=observed, pairs=roots, transfer=transfer.cpu())
                print(json.dumps(row), flush=True)
        report["passed"] = all(row["passed"] for row in report["cases"])
        torch.save(saved, args.output / "diagnostic.pt")
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
