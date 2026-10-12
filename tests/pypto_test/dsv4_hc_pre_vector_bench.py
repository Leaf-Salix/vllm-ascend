# SPDX-License-Identifier: Apache-2.0
"""只执行 PTO，验证 HC_pre 第二阶段分工；旧 PTO 保存输出，候选对照并采 L1。"""

import argparse
import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path

from dsv4_csa_env import activate, write_json
from dsv4_csa_single_card_bench import _export_swimlane
from dsv4_hc_pre_bench import capture, compare, load_weights, require_pass
from dsv4_hc_pre_timing import summarize_swimlane


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-file", type=Path, required=True)
    parser.add_argument("--pypto-core", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--probe", action="store_true", help="只对额外携带逐核诊断 Tensor 的快照启用")
    parser.add_argument("--expected-active-vectors", type=int)
    parser.add_argument("--batch-size", nargs="+", type=int, default=[16])
    parser.add_argument("--seqlen", nargs="+", type=int, default=[6])
    parser.add_argument("--checkpoint", type=Path, default=Path("/data/model/DeepSeek-V4-Flash-0731-w8a8"))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("TASK_DEVICE"):
        raise RuntimeError("通过 task-submit --device auto 执行")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        raise RuntimeError("使用新的结果目录，不覆盖历史记录")
    activate()
    finders = [finder for finder in sys.meta_path
               if "pypto.pypto_core" in getattr(finder, "known_wheel_files", {})]
    if len(finders) != 1:
        raise RuntimeError("无法唯一定位本进程 PyPTO 调试扩展映射")
    finders[0].known_wheel_files["pypto.pypto_core"] = str(args.pypto_core.resolve(strict=True))
    name = "vllm_ascend.ops.pypto.hc_pre"
    spec = importlib.util.spec_from_file_location(name, args.kernel_file.resolve(strict=True))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    import pypto.torch
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    weights, weight_records = load_weights(args.checkpoint, 2, "attn", device)
    pypto.torch.init(device=0, platform="a2a3", runtime="tensormap_and_ringbuffer",
                     enable_chip_swimlane=1, enable_dep_gen=True, output_dir=str(args.output / "dfx"))
    op = pypto.torch.register(module.hc_pre, "hc_pre_vectors::hc_pre")
    report = {"status": "RUNNING", "scope": "只运行 PTO；没有 Native 调用或 PyTorch profiler",
              "physical_device": os.environ["TASK_DEVICE"], "kernel_file": str(args.kernel_file.resolve()),
              "weights": weight_records, "probe": args.probe, "cases": []}
    try:
        for window, (batch, seqlen) in enumerate(itertools.product(args.batch_size, args.seqlen)):
            tokens = batch * seqlen
            case = f"B{batch:02d}_S{seqlen}_T{tokens:03d}"
            generator = torch.Generator().manual_seed(20261009 + batch * 100 + seqlen)
            x = torch.randn(tokens, 4, 4096, generator=generator, dtype=torch.bfloat16).to(device)
            allocations = (torch.full((tokens + 2, 4096), 37, dtype=torch.bfloat16, device=device),
                           torch.full((tokens + 2, 4), 37, dtype=torch.float32, device=device),
                           torch.full((tokens + 2, 4, 4), 37, dtype=torch.float32, device=device))
            outputs = tuple(value[1:-1] for value in allocations)
            trace = torch.full((48, 16), -1, dtype=torch.int64, device=device) if args.probe else None

            def run():
                op(x, *weights, *outputs, *([trace] if trace is not None else []))
                return outputs

            eager = tuple(value.cpu() for value in run())
            row = {"name": case, "tokens": tokens}
            reference = None
            if args.reference:
                reference = torch.load(args.reference / f"{case}_outputs.pt", weights_only=True)
                row["old_pto"] = compare(eager, reference["initial"])
                require_pass(row["old_pto"])
            graph, actual = capture(run)
            row["graph_eager"] = compare(actual, eager, exact=True)
            require_pass(row["graph_eager"])
            # 完成 CPU 核对后重新预热；只记录固定三个 step，不事后选择快样本。
            for _ in range(5):
                graph.replay()
                torch.npu.synchronize()
            pypto.torch.begin_dfx()
            try:
                for _ in range(3):
                    graph.replay()
                    torch.npu.synchronize()
            finally:
                pypto.torch.end_dfx()
            directory = args.output / "dfx"
            if window:
                directory /= f"window_{window}"
            exported = _export_swimlane(directory, kernel_pattern="_jit_hc_pre_*/kernel_config.py")
            if not exported["exported"]:
                raise RuntimeError(exported)
            row["l1"] = summarize_swimlane(exported["merged_swimlane"], 3)
            if any(sample["blocks"] != 72 for sample in row["l1"]["samples"]):
                raise RuntimeError("L1 缺少 24 AIC + 48 AIV 的完整记录")
            if trace is not None:
                raw = trace.cpu().tolist()
                # 记录第二阶段真实执行的逻辑 AIV、循环次数及写入区间；不推算内部耗时。
                if {item[0] for item in raw} != set(range(48)):
                    raise RuntimeError("逐核诊断记录不完整")
                row["tail_probe"] = raw
                row["active_vectors"] = sum(item[3] > 0 for item in raw)
                if args.expected_active_vectors is not None and row["active_vectors"] != args.expected_active_vectors:
                    raise AssertionError(f"第二阶段实际工作核数不符：{row['active_vectors']}")
                comb_coverage = [0] * tokens
                post_coverage = [0] * tokens
                mixed_coverage = [[0] * 8 for _ in range(tokens)]
                for item in raw:
                    if item[3] == 0:
                        continue
                    if item[3] != item[9] - item[8]:
                        raise AssertionError("设备实际处理行数与记录的分工区间不一致")
                    for token in range(item[8], item[9]):
                        if item[5] == 0:
                            comb_coverage[token] += 1
                        else:
                            post_coverage[token] += int(item[6] == 0)
                            for block in range(item[6], item[7]):
                                mixed_coverage[token][block] += 1
                if (set(comb_coverage) != {1} or set(post_coverage) != {1}
                        or any(set(blocks) != {1} for blocks in mixed_coverage)):
                    raise AssertionError("实际分工存在漏算或重复写入")
                row["tail_coverage"] = "全部 token、D 分片及 post 写入各覆盖一次"
                work = []
                for vector, item in enumerate(raw):
                    role = "Sinkhorn" if item[5] == 0 else "门控及残差混合"
                    work.append({"logical_aiv": vector, "group": vector // 2, "lane": vector % 2,
                                 "role": role, "processed_rows": item[3], "iterations": item[4],
                                 "token_begin": item[8], "token_end": item[9],
                                 "d_begin": item[6] * 512 if item[5] else None,
                                 "d_end": item[7] * 512 if item[5] else None})
                write_json(args.output / f"{case}_tail_work.json", {
                    "scope": "设备写回的第二阶段实际执行记录；逻辑编号，不是物理核号；不含内部时间片",
                    "active_vectors": row["active_vectors"], "work": work})
            # 同址换输入，核实每次真正重算，保留旧版对应输出供候选比较。
            x.copy_(torch.randn(tokens, 4, 4096, generator=generator, dtype=torch.bfloat16).to(device))
            for value in outputs:
                value.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            updated = tuple(value.cpu() for value in outputs)
            if reference:
                row["old_pto_updated_input"] = compare(updated, reference["updated"])
                require_pass(row["old_pto_updated_input"])
            else:
                torch.save({"initial": eager, "updated": updated}, args.output / f"{case}_outputs.pt")
            if not all(bool((value[0].cpu() == 37).all() and (value[-1].cpu() == 37).all()) for value in allocations):
                raise AssertionError("输出保护行被改写")
            row.update(status="PASS", output_guards=True)
            report["cases"].append(row)
            write_json(args.output / "report.json", report)
            print(json.dumps({"case": case, "status": "PASS", "incore_p50_us": row["l1"]["span_p50_us"],
                              "active_vectors": row.get("active_vectors")}), flush=True)
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
