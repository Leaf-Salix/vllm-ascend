#!/usr/bin/env python3
"""从72核原始指令日志解读GM阶段槽写入，帮助判断模型是否实际推进。

只记录ST指令在日志中的cycle，不把它当成写后DCCI/DSB完成、消费者已观察到
信号或阶段墙钟时间。使用本轮首次调用的代际1、512字节槽及72核映射验证
地址。若日志轮转丢失初始代际写入则拒绝，不伪造缺少的阶段。
"""

import ast
import csv
import json
import re
from pathlib import Path

from prepare_standalone import HERE


STORE = re.compile(r"\[(\d+)\].*?\(PC: (0x[0-9a-f]+)\).*? (ST_XD_XN(?:_IMM)?)\s+dtype:B32, XD:X\d+=(0x[0-9a-f]+|\d+), XN:X\d+=(0x[0-9a-f]+|\d+), (?:IMM:(0x[0-9a-f]+|\d+)|XM:X\d+=(0x[0-9a-f]+|\d+))")


def stores(files):
    for path in files:
        with path.open(errors="replace") as stream:
            for line in stream:
                if "ST_XD_XN" not in line or "dtype:B32" not in line:
                    continue
                match = STORE.search(line)
                if not match or int(match[4], 0) != 1 or "#POST:1" in line:
                    continue
                # B32的XM/IMM索引按4字节元素计；core1首次代际XM=0x80，
                # 正好对应128个INT32，即512字节隔离槽。拒绝post-index形式。
                address = int(match[5], 0) + 4 * int(match[6] or match[7], 0)
                yield int(match[1]), match[2], address


def main():
    case = Path((HERE / "ACTIVE_CASE_PATH.txt").read_text().strip())
    out = case.parents[2] / "funcs/dynamic_csa_full/collect/out"
    # 模型运行时在tmp_dump，结束后CANN会迁移至kernel/0/dump。
    candidates = {path.parent for path in out.glob("**/core0.cubecore0.instr_log*.dump")}
    assert len(candidates) == 1, "dump正在迁移或路径不唯一，请等待导出状态稳定"
    dump = next(iter(candidates))
    # 当前此文件尚未轮转；跨多个文件按cycle排序后选最早的代际store。
    initial = sorted(stores(dump.glob("core0.cubecore0.instr_log*.dump")))
    assert initial
    base = initial[0][2]
    phase_names = {0: "GENERATION"}
    source = HERE.parents[5] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/runtime_abi.py"
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and type(node.value.value) is int:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("PHASE_") and target.id != "PHASE_COUNT":
                    phase_names[node.value.value] = target.id
    for group in range(8):
        phase_names[44 + group] = f"PHASE_OA_GROUP_{group}"
        phase_names[52 + group] = f"PHASE_O_QUANT_GROUP_{group}"
    records, epochs = [], set()
    for group in range(24):
        for kind, owner in (("cubecore0", group), ("veccore0", 24+2*group), ("veccore1", 25+2*group)):
            core = f"core{group}.{kind}"
            for cycle, pc, address in stores(dump.glob(core + ".instr_log*.dump")):
                offset = address - base
                if not 0 <= offset < 60*72*512 or offset % 512:
                    continue
                phase, writer = divmod(offset // 512, 72)
                assert writer == owner, (core, owner, phase, writer, hex(address))
                if phase == 0:
                    epochs.add(core)
                records.append(dict(core=core, owner=owner, phase=phase, phase_name=phase_names[phase],
                                    instruction_cycle=cycle, pc=pc, signal_address=hex(address)))
    assert len(epochs) == 72, "缺少初始代际记录，不能当完整记录使用"
    records.sort(key=lambda row: (row["instruction_cycle"], row["owner"]))
    dest = HERE / "build_output/model_phase_writes"
    dest.mkdir(exist_ok=True)
    with (dest / "phase_stores.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    summary = dict(cores_with_initial_epoch=72, signal_base=hex(base), records=len(records), raw_dump=str(dump),
                   interpretation="ST指令时刻，非DCCI/DSB发布完成时间；非正式性能", bounded_model_backoff=True,
                   latest_by_core={core:max((r for r in records if r["core"]==core), key=lambda r:r["instruction_cycle"]) for core in sorted(epochs)})
    (dest / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(cores=72, records=len(records), latest=[summary["latest_by_core"][c] for c in ("core0.cubecore0","core0.veccore0","core12.cubecore0")]),ensure_ascii=False))


if __name__ == "__main__":
    main()
