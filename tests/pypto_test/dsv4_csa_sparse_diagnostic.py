# SPDX-License-Identifier: Apache-2.0
"""固定 Native Q/cache/Top-K，隔离两版 QK、softmax 和 PV 的数值差异。"""

import argparse
import importlib
import os
from pathlib import Path

from dsv4_csa_env import activate, write_json
from dsv4_csa_validation import compare_tensor


def build_kernel(variant):
    import pypto.language as pl

    mod = importlib.import_module("vllm_ascend.ops.pypto.deepseek_v4_flash_csa.decode_sparse_attn_csa")
    sparse = mod.sparse_attn_csa_tp1
    batch = pl.dynamic("DIAG_BATCH")
    tokens = pl.dynamic("DIAG_TOKENS")
    ori_pages = pl.dynamic("DIAG_ORI_PAGES")
    cmp_pages = pl.dynamic("DIAG_CMP_PAGES")
    ori_cols = pl.dynamic("DIAG_ORI_COLS")
    cmp_cols = pl.dynamic("DIAG_CMP_COLS")
    heads, width, rope = mod.H, mod.HEAD_DIM, mod.ROPE_DIM
    groups, pad, group_width = mod.O_GROUPS, mod.T_PAD, mod.O_GROUP_IN

    def root(
        q: pl.Tensor[[tokens, heads, width], pl.BF16],
        ori_kv: pl.Tensor[[ori_pages, 32, 1, width], pl.BF16],
        ori_table: pl.Tensor[[batch, ori_cols], pl.INT32],
        cmp_kv: pl.Tensor[[cmp_pages, 32, 1, width], pl.BF16],
        cmp_table: pl.Tensor[[batch, cmp_cols], pl.INT32],
        topk: pl.Tensor[[tokens, 512], pl.INT32],
        positions: pl.Tensor[[tokens, 1], pl.INT64],
        seq_lens: pl.Tensor[[batch], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cos: pl.Tensor[[tokens, rope], pl.FP32],
        sin: pl.Tensor[[tokens, rope], pl.FP32],
        output: pl.Out[pl.Tensor[[groups * pad, group_width], pl.BF16]],
    ):
        output, _ = sparse(q, ori_kv, ori_table, cmp_kv, cmp_table, topk, positions, seq_lens, sink, cos, sin, output)
        return output

    return pl.jit(auto_scope=False)(root), mod


def uniform_case(batch):
    """Q/sink 为零，SWA=2、compressed=1；归一化结果可直接计算，无权重依赖。"""
    import torch

    positions = torch.arange(255, 261).repeat(batch)
    compressed = (positions + 1) // 4
    indices = torch.arange(512).expand(batch * 6, -1).clone()
    indices[indices >= compressed[:, None]] = -1
    expected = ((256 + compressed).float() / (129 + compressed)).to(torch.bfloat16)
    return {
        "batch": batch,
        "history": 255,
        "position_ids": positions,
        "q": torch.zeros((batch * 6, 64, 512), dtype=torch.bfloat16),
        "sinks": torch.zeros(64),
        "seqused_kv": torch.full((batch,), 261, dtype=torch.int32),
        "ori_kv": torch.full((10, 32, 1, 512), 2, dtype=torch.bfloat16),
        "ori_block_table": torch.arange(9, 0, -1, dtype=torch.int32).repeat(batch, 1),
        "cmp_kv": torch.ones((4, 32, 1, 512), dtype=torch.bfloat16),
        "cmp_block_table": torch.arange(3, 0, -1, dtype=torch.int32).repeat(batch, 1),
        "cmp_sparse_indices": indices.to(torch.int32),
        "expected": expected[:, None, None].expand(-1, 64, 512).contiguous(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument(
        "--synthetic-batch",
        type=int,
        choices=(1, 3, 4),
        help="零 Q 均匀 attention 回归：B1/3 覆盖尾块，B4 覆盖整块，输出要求 bit 一致",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("performance",), default="performance")
    parser.add_argument("--device", type=int, default=0, help="task-submit 分配的单卡编号；诊断不自行选择其他设备")
    parser.add_argument(
        "--pmu",
        type=int,
        choices=(0, 1, 2, 4, 5, 6, 7, 8),
        default=0,
        help="独立 program 模式采核内计数器；2=流水利用率，4=内存，不作稳态性能计时",
    )
    args = parser.parse_args()
    os.environ["PTO_CSA_VARIANT"] = args.variant
    activate()
    import pypto.torch
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    payload = (
        uniform_case(args.synthetic_batch)
        if args.synthetic_batch
        else torch.load(args.input, map_location="cpu", weights_only=True)
    )
    kernel, mod = build_kernel(args.variant)
    target = "cpu" if args.pmu else f"npu:{args.device}"
    names = (
        "q",
        "ori_kv",
        "ori_block_table",
        "cmp_kv",
        "cmp_block_table",
        "cmp_sparse_indices",
        "position_ids",
        "seqused_kv",
        "sinks",
    )
    tensors = [payload[n].contiguous().to(target) for n in names]
    tensors[5] = tensors[5].view(-1, 512)
    tensors[6] = tensors[6].view(-1, 1)
    tokens = tensors[0].shape[0]
    cos = torch.ones((tokens, mod.ROPE_DIM), dtype=torch.float32, device=target)
    sin = torch.zeros_like(cos)
    output = torch.full((mod.O_GROUPS * mod.T_PAD, mod.O_GROUP_IN), float("nan"), dtype=torch.bfloat16, device=target)
    if args.pmu:
        from pypto.runtime import RunConfig

        config = RunConfig(
            platform="a2a3",
            device_id=args.device,
            enable_pmu=args.pmu,
            save_kernels=True,
            save_kernels_dir=str((args.output / "build").resolve()),
        )
        compiled = kernel.compile(*tensors, cos, sin, output, config=config)
        compiled(*tensors, cos, sin, output, config=config)
    else:
        pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        kernel(*tensors, cos, sin, output)
    actual = output.view(mod.O_GROUPS, mod.T_PAD, mod.O_GROUP_IN)[:, :tokens]
    actual = actual.transpose(0, 1).contiguous().view(tokens, mod.H, mod.HEAD_DIM).cpu()
    expected = payload["expected"].view_as(actual)
    checks = compare_tensor(actual, expected, 0, 0)
    report = {
        "status": checks["status"] if args.synthetic_batch else "MEASURED",
        "variant": args.variant,
        "input": str(args.input) if args.input else None,
        "reference": "均匀 attention 解析值" if args.synthetic_batch else "Native sparse attention",
        "batch": payload["batch"],
        "history": payload["history"],
        "device": args.device,
        "scope": "固定 Q/cache/Top-K；逆 RoPE 用 cos=1/sin=0 隔离 QK/softmax/PV；零容差比较",
        "comparison": checks,
        "per_token": [compare_tensor(a, e, 0, 0) for a, e in zip(actual, expected)],
    }
    if args.pmu:
        pmu_path = args.output / "build/dfx_outputs/pmu.csv"
        if not pmu_path.is_file() or pmu_path.stat().st_size == 0:
            raise RuntimeError(f"PMU 未生成计数器记录：{pmu_path}")
        report.update(
            execution="standalone_program",
            pmu_event_type=args.pmu,
            pmu_csv=str(pmu_path.resolve()),
            measurement_limit="固定Native输入的独立核内诊断，不代表完整CSA稳态或调度性能",
        )
    if checks.get("nonfinite") != 0:
        report["status"] = "FAIL"
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "report.json", report)
    torch.save(actual, args.output / "output.pt")
    print({k: v for k, v in report.items() if k != "per_token"})
    if report["status"] == "FAIL":
        raise RuntimeError("sparse 回归失败或诊断产生非有限值；见 report.json")


if __name__ == "__main__":
    main()
