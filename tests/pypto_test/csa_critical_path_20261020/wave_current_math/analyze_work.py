# SPDX-License-Identifier: Apache-2.0
"""从已回读metadata统计wave工作分布与重复Left恢复，不新增设备测量。"""

import ast
import json
from pathlib import Path

from audit_codegen import annotated

HERE = Path(__file__).resolve().parent
LEFT_BYTES_PER_RESTORE = 60 * 1024


def main():
    reports = {}
    for batch in (16, 28):
        result = json.loads((HERE / f"device_b{batch}_v1/result.json").read_text())
        assert result["passed"] and result["stage"] == "complete"
        protocol = result["launches"][0]["wave_protocol"]
        assert protocol["passed"]
        groups, _, leaves, leaf_tiles, extra = protocol["plan"][:5]
        descriptor_base = (protocol["requests"] + 16) // 16 * 16
        counts = [[0] * 24 for _ in protocol["waves"]]
        active_leaves = [0] * 24
        for item in range(groups * leaves):
            group, leaf = ((item % groups, item // groups) if groups < 24
                           else (item // leaves, item % leaves))
            descriptor = descriptor_base + group * 4
            live, visible = protocol["offsets"][descriptor + 2:descriptor + 4]
            begin = (leaf * leaf_tiles + min(leaf, extra)) * 1024
            capacity = (leaf_tiles + int(leaf < extra)) * 1024
            valid = max(min(visible - begin, capacity), 0)
            if live <= 0 or valid <= 0:
                continue
            # 与next_score_leaf相同：每lane容量最多4096，整step处理1024候选。
            steps = min((valid + 1023) // 1024, 8)
            owner = item % 24
            active_leaves[owner] += 1
            wave, = [i for i, w in enumerate(protocol["waves"])
                     if w["group_begin"] <= group < w["group_end"]]
            counts[wave][owner] += steps
        full_counts = [sum(row[core] for row in counts) for core in range(24)]
        ir, = (HERE / f"cpu_b{batch}_v1/kernel/passes_dump").glob("*after_AllocateMemoryAddr.py")
        tree = ast.parse(ir.read_text())
        cube, = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name == "cube_schedule"]
        restores = [r for r in annotated(cube)
                    if r["name"].startswith(("split_query0_", "split_query1_",
                                             "split_coefficient0_", "split_coefficient1_"))]
        assert len(restores) == 12 and sum(r["in_wave"] for r in restores) == 8
        assert all("extract" in r["operation"] for r in restores)
        reports[str(batch)] = {
            "source_metadata": str(HERE / f"device_b{batch}_v1/result.json"),
            "wave_count": protocol["wave_count"],
            "score_steps_per_core": full_counts,
            "score_steps_per_wave_per_core": counts,
            "max_steps_whole": max(full_counts),
            "sum_wave_max_steps": sum(max(row) for row in counts),
            "active_leaves_per_core": active_leaves,
            "reference_left_restore_bytes_model": sum(active_leaves) * LEFT_BYTES_PER_RESTORE,
            "candidate_left_restore_bytes_model": sum(full_counts) * LEFT_BYTES_PER_RESTORE,
            "extra_left_restore_bytes_model":
                (sum(full_counts) - sum(active_leaves)) * LEFT_BYTES_PER_RESTORE,
            "actual_ir_restores": restores,
            "scope": "实际metadata的工作量模型与生成位置，非硬件流量计数或阶段耗时",
            "caveat": "sum_wave_max不是实测关键路径；wave采用必要前驱等待，并非全核集合。"
                      "额外Left恢复、波次启动排空、片上竞争及两槽Attention可能共同抵消重叠，"
                      "不能把完整回退定量归给任意一项。",
        }
    (HERE / "WAVE_WORK_ANALYSIS.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n"
    )
    for batch, row in reports.items():
        print(batch, "wave", row["wave_count"], "steps",
              row["max_steps_whole"], row["sum_wave_max_steps"],
              "extra Left MiB", row["extra_left_restore_bytes_model"] / 2**20)


if __name__ == "__main__":
    main()
