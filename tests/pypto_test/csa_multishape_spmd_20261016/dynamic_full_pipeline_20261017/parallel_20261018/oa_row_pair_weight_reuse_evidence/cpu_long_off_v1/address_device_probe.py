# SPDX-License-Identifier: Apache-2.0
"""独立验证正式Attention地址helper：真实页宽、空洞、负索引及INT32边界。"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
import torch
from qk_address_cache_v1.attention_address_candidate import (
    compressed_half_addresses,
)

PROBE_COLUMNS = pl.dynamic("CSA_ADDRESS_PROBE_COLUMNS")


@pl.jit.incore
def address_probe_core(
    indices: pl.Tensor[[1, 512], pl.INT32],
    table: pl.Tensor[[1, PROBE_COLUMNS], pl.INT32],
    out: pl.Out[pl.Tensor[[4, 64], pl.INT32]],
):
    raw, columns, pages, physical = compressed_half_addresses(indices, table, 0, 0, 64)
    pl.store(raw, [0, 0], out)
    pl.store(columns, [1, 0], out)
    pl.store(pages, [2, 0], out)
    pl.store(physical, [3, 0], out)
    return out


@pl.jit
def address_probe(
    indices: pl.Tensor[[1, 512], pl.INT32],
    table: pl.Tensor[[1, PROBE_COLUMNS], pl.INT32],
    out: pl.Out[pl.Tensor[[4, 64], pl.INT32]],
):
    # helper关闭自动scope，必须显式放在incore函数内，不能直接留在SPMD编排层。
    with pl.spmd(1, name_hint="attention_address_regression"):
        out = address_probe_core(indices, table, out)
    return out


def validate(device, config, output):
    cases = []
    # 实际列数包括非32B对齐尾部；另覆盖helper下界1和物理Tile满容量2048。
    # 1/255列是helper合法合同，完整Attention仍按自己的工作量阈值选择Scalar路径。
    for width in (1, 255, 1025, 2048):
        table_cpu = (torch.arange(width, dtype=torch.int64) * 7 % 19).to(torch.int32).view(1, width)
        table_cpu[0, 0] = -1
        if width > 1:
            table_cpu[0, 1] = 19
            table_cpu[0, -1] = 67108863  # 最大合法INT32物理行对应页号
        indices_cpu = torch.arange(512, dtype=torch.int32).view(1, 512)
        values = torch.tensor(
            [-2147483648, -1, 0, 31, 32, 63, width * 32 - 1, width * 32, 2147483647], dtype=torch.int32
        )
        indices_cpu[0, 64:73] = values
        raw = indices_cpu[0, 64:128].to(torch.int64)
        columns = torch.clamp_min(raw, 0) // 32
        pages = table_cpu[0, columns.clamp_max(width - 1)].to(torch.int64)
        physical = pages * 32 + torch.clamp_min(raw, 0) % 32
        expected = torch.stack([raw, columns, pages, physical]).to(torch.int32)
        indices, table = indices_cpu.to(device), table_cpu.to(device)
        guarded = torch.full((6, 64), 37, dtype=torch.int32, device=device)
        out = guarded[1:5]
        address_probe(indices, table, out, config=config)
        torch.npu.synchronize()
        actual = out.cpu()
        assert torch.equal(actual, expected), f"地址helper在C={width}出现差异"
        assert (guarded[0].cpu() == 37).all() and (guarded[-1].cpu() == 37).all()
        # 逐候选合同：只有原地址合法时才消费physical，夹紧不能使坏候选变好。
        for row, index in enumerate(raw.tolist()):
            valid = index >= 0 and index // 32 < width
            valid = valid and 0 <= int(pages[row]) < 19
            scalar_expected = None
            if index >= 0 and index // 32 < width:
                page = int(table_cpu[0, index // 32])
                if 0 <= page < 19:
                    scalar_expected = page * 32 + index % 32
            assert valid == (scalar_expected is not None)
            if valid:
                assert int(actual[3, row]) == scalar_expected
        cases.append({"table_columns": width, "passed": True, "guards_preserved": True})
    report = {"passed": True, "cases": cases, "performance_measured": False}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    from pypto.runtime import CompileOptions

    parser = argparse.ArgumentParser(description="仅CPU编译地址边界probe，不打开NPU")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    address_probe._resolve_kernel_artifact(
        (),
        {"config": CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel"))},
        allow_signature_mode=True,
    )
    (args.output / "result.json").write_text('{"compiled": true, "device_execution": false}\n')
