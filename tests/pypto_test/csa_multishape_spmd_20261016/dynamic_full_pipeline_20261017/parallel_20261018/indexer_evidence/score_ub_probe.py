# SPDX-License-Identifier: Apache-2.0
"""正式G2/768 helper同源对照：六个半叶分支、并列分数、同址live=2→1→2。

不改生产数学，不消费候选已删除的私有score_arena写入。比较两个版本的全部
pair arena位模式、Cube transfer及未写区；图内设备metadata反复改变。无性能采样。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = Path(
    "/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
)
CANDIDATE = HERE.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
REFERENCE = D / "parallel_20261018/baseline_source"


def load_probe(source, label, output):
    from source_loader import load_package

    package = load_package(source, label)
    original = D.parent / "dynamic_indexer_score_20261017/cpu_probes.py"
    text = original.read_text().replace(
        "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_score_stages",
        package.name + ".indexer_score_stages",
    )
    path = output / f"{label}_probe.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(label + "_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.g2_768_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    args = parser.parse_args()
    if not args.cpu_only and ("TASK_DEVICE" not in os.environ or args.device != int(os.environ["TASK_DEVICE"])):
        raise RuntimeError("必须经task-submit分配NPU")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(D))
    report = dict(passed=False, performance_measured=False, device_execution=False, cases=[])
    try:
        import dsv4_csa_env

        dsv4_csa_env.activate()
        from pypto import passes
        from pypto.runtime import CompileOptions

        kernels = {
            label: load_probe(source, label, args.output)
            for label, source in [("reference", REFERENCE), ("candidate", CANDIDATE)]
        }
        options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        if args.cpu_only:
            for label, kernel in kernels.items():
                artifact = kernel._resolve_kernel_artifact(
                    (),
                    {
                        "config": CompileOptions(
                            platform="a2a3",
                            memory_planner=passes.MemoryPlanner.PYPTO,
                            output_dir=str(args.output / label),
                        )
                    },
                    allow_signature_mode=True,
                )
                report[label] = str(artifact.output_dir)
            report["passed"] = True
        else:
            import torch  # noqa: I001 - 保留已验证的torch/NPU/bridge延迟导入顺序
            import torch_npu
            import pypto.torch
            from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs

            torch.npu.set_device(args.device)
            torch.npu.config.allow_internal_format = True
            torch_npu.npu.set_deterministic_level(0)
            device = torch.device(f"npu:{args.device}")
            pypto.torch.init(
                device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer", **ring_sizing_kwargs()
            )
            generator = torch.Generator().manual_seed(20261018)
            random_query = torch.randint(-10, 11, (128, 128), dtype=torch.int8, generator=generator)
            query = random_query.to(device)
            keys = torch.randint(-7, 8, (8192, 128), dtype=torch.int8, generator=generator)
            scale = ((torch.arange(8192) % 7 + 1).half() / 64).reshape(256, 32)
            native_cpu = torch.zeros((256, 4160), dtype=torch.int8)
            native_cpu[:, :4096] = keys.reshape(256, 4096)
            native_cpu[:, 4096:] = scale.view(torch.int8)
            native = native_cpu.to(device)
            flat = native.flatten()
            key0 = flat[: flat.numel() // 128 * 128].view(-1, 128)
            key64 = flat[64 : 64 + (flat.numel() - 64) // 128 * 128].view(-1, 128)
            table = torch.arange(256, dtype=torch.int32, device=device).reshape(1, 256)
            positions = torch.tensor([32768, 32769], dtype=torch.int64, device=device)
            lengths = torch.tensor([32770], dtype=torch.int32, device=device)
            bounds = torch.tensor([0, 2], dtype=torch.int32, device=device)
            offsets_cpu = torch.zeros(32, dtype=torch.int32)
            offsets_cpu[1] = 1
            offsets_cpu[16:20] = torch.tensor([0, 0, 2, 8192])
            offsets = offsets_cpu.to(device)
            plan = torch.tensor([1, 8192, 1, 8, 0, 2, 0, 2], dtype=torch.int32, device=device)
            coeff_cpu = torch.zeros((16, 128), dtype=torch.float16)
            coeff_cpu[0, :64] = (torch.arange(64) % 8 + 1).half() / 8
            coeff_cpu[1, 64:] = (torch.arange(64) % 7 + 1).half() / 16
            coefficients = coeff_cpu.to(device)
            transfer = torch.full((96, 768), 37.0, device=device)
            scores = torch.full((96, 8192), 37.0, device=device)
            pairs = torch.full((128, 1024), 37.0, device=device)
            ffts = torch.zeros(256, dtype=torch.int64, device=device)
            tensors = (
                query,
                coefficients,
                key0,
                key64,
                native,
                table,
                positions,
                lengths,
                bounds,
                offsets,
                plan,
                transfer,
                ffts,
                scores,
                pairs,
            )
            addresses = tuple(v.data_ptr() for v in tensors)
            graphs = {}
            for label, kernel in kernels.items():
                kernel(*tensors, config=options)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    kernel(*tensors, config=options)
                torch.npu.synchronize()
                graphs[label] = graph
            # 这些N覆盖半叶六个分支及384/768分界和8192末轮物理补齐。
            lengths_to_check = (
                1,
                511,
                512,
                513,
                768,
                769,
                1024,
                1025,
                1536,
                1537,
                2048,
                2049,
                3072,
                3073,
                3840,
                3841,
                4096,
                4097,
                4608,
                4609,
                6144,
                6145,
                7680,
                7681,
                8192,
            )
            for ties in (False, True):
                query.copy_(torch.zeros_like(random_query) if ties else random_query)
                for visible in lengths_to_check:
                    for live in (2, 1, 2):
                        positions.copy_(torch.tensor([4 * visible, 4 * visible + 1], dtype=torch.int64))
                        bounds.copy_(torch.tensor([0, live], dtype=torch.int32))
                        lengths.copy_(torch.tensor([4 * visible + live], dtype=torch.int32))
                        offsets_cpu[18:20] = torch.tensor([live, visible], dtype=torch.int32)
                        offsets.copy_(offsets_cpu)
                        plan.copy_(torch.tensor([1, visible, 1, 8, 0, 2, 0, 2], dtype=torch.int32))
                        coeff = coeff_cpu.clone()
                        coeff[live:].zero_()
                        coefficients.copy_(coeff)
                        actual = {}
                        for label, graph in graphs.items():
                            transfer.fill_(37)
                            scores.fill_(37)
                            pairs.fill_(37)
                            graph.replay()
                            torch.npu.synchronize()
                            actual[label] = (pairs.cpu().view(torch.uint8), transfer.cpu().view(torch.uint8))
                            if label == "candidate":
                                assert bool((scores == 37).all()), "候选不得重新写GM半叶scratch"
                        root_diff = int((actual["reference"][0] != actual["candidate"][0]).sum())
                        transfer_diff = int((actual["reference"][1] != actual["candidate"][1]).sum())
                        row = dict(
                            ties=ties,
                            visible=visible,
                            live=live,
                            root_byte_differences=root_diff,
                            transfer_byte_differences=transfer_diff,
                            same_addresses=addresses == tuple(v.data_ptr() for v in tensors),
                        )
                        row["passed"] = root_diff == transfer_diff == 0 and row["same_addresses"]
                        report["cases"].append(row)
                        if not row["passed"]:
                            torch.save(actual, args.output / "failed_pairs.pt")
                            raise AssertionError(row)
                print(json.dumps(dict(ties=ties, passed_cases=len(report["cases"]))), flush=True)
            report["cache_unchanged"] = torch.equal(native.cpu(), native_cpu)
            report["device_execution"] = True
            report["passed"] = report["cache_unchanged"] and all(row["passed"] for row in report["cases"])
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("cases", "traceback")}, ensure_ascii=False), flush=True
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
