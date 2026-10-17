"""只读提取现有编译产物；不调用编译器、运行时或设备，不计算内容摘要。"""

import json
import re
from collections import defaultdict
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = Path("/data/pyptouser/qinchuanyu/pto-eager")
C = ROOT / "vllm-ascend-dsv4-pto-0251rc1"
D = C / "tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
BUILD = D / "parallel_20261018/current_clock_h131072_b16_v1/build_output"
F = BUILD / "decode_csa_spmd_jqlk5eey"
O = BUILD / "_decode_csa_tp1_layer_0leidxnz"
W = ROOT / ".worktrees/csa-ultra-sync-20261018"
CPU_LONG = W / "tests/pypto_test/ultra_sync_20261018/kernel_long_v1/kernel/passes_dump/32_after_LowerPipelineLoops.py"
CPU_MEDIUM = D / "kernel_medium_v1/kernel/passes_dump/32_after_LowerPipelineLoops.py"
INPUTS = set()


def read(path):
    INPUTS.add(path)
    return path.read_text().splitlines()


def allocations(path, cube_only=False):
    rows = []
    for n, line in enumerate(read(path), 1):
        if cube_only and "func.func @vector_schedule" in line:
            break
        if "pto.alloc_tile" not in line:
            continue
        m = re.search(r"(%\w+) = pto.alloc_tile addr = %c(\d+)_i64.*?loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", line)
        if not m or m[3] not in {"left", "right"}:
            continue
        loc = re.search(r' loc\("([^\"]+)":(\d+):(\d+)\)', line)
        assert loc, (path, n)
        rows.append({"ir_line": n, "variable": m[1], "address": int(m[2]),
                     "space": m[3], "dtype": m[4], "shape": [int(m[5]), int(m[6])],
                     "source_file": loc[1], "source_line": int(loc[2])})
    return rows


def compact(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (Path(row["source_file"]).name, row["source_line"], row["space"], row["dtype"], *row["shape"])
        groups[key].append(row)
    return [{"source_file": key[0], "source_line": key[1], "space": key[2],
             "dtype": key[3], "shape": list(key[4:]),
             "addresses": sorted({r["address"] for r in values}),
             "ir_lines": [r["ir_line"] for r in values]} for key, values in groups.items()]


def hints(path):
    result = []
    for n, line in enumerate(read(path), 1):
        if "PH-MR-001" not in line:
            continue
        m = re.search(r"requested depth (\d+) for pipeline group (\d+) in (\w+), but only (\d+) of (\d+) buffers fit \((\d+) B per stage, (\d+) B free", line)
        assert m, (path, n)
        result.append({"line": n, "requested": int(m[1]), "group": int(m[2]),
                       "space": m[3], "achieved": int(m[4]), "slot_bytes": int(m[6]),
                       "free_bytes": int(m[7]), "text": line})
    return result


def memberships(path):
    lines = read(path)
    groups = {}
    in_cube = False
    for i, line in enumerate(lines):
        if re.search(r"def cube_schedule\(", line):
            in_cube = True
        elif re.search(r"def vector_schedule\(", line):
            in_cube = False
        if not in_cube:
            continue
        match = re.search(r'pipeline_membership": "([^\"]+)"', line)
        if not match:
            continue
        for entry in match[1].split(";"):
            group = int(entry.split(":")[0])
            if group not in groups:
                groups[group] = {"first_line": i + 1, "context": lines[max(0, i - 3):i + 1]}
    return groups


def snippet(path, lo, hi):
    return {"path": str(path), "first_line": lo, "last_line": hi,
            "text": "\n".join(f"{i}: {line}" for i, line in enumerate(read(path), 1) if lo <= i <= hi)}


fused = allocations(F / "ptoas/dynamic_csa_full.pto", cube_only=True)
original_names = ["hc_pre_linear", "qr_proj_matmul", "qr_proj_matmul_0", "idx_qr_proj_matmul",
                  "weights_proj", "kv_score_proj", "kv_score_proj_0", "kv_proj_matmul", "kv_proj_matmul_0",
                  "qproj_matmul", "proj_a_mm_0", "_proj_b_mm_nz_kernel__2", "indexer_score_topk_native_pair"]
original = {name: allocations(O / f"ptoas/{name}.pto") for name in original_names}
result = {
    "scope": "现有 593 长 B16 诊断及同次原分 task 生成物；只读物理地址、源位置和 C++ 依赖，不编译/上卡。",
    "performance_claim": "无性能收益结论；地址和事件差异不等于关键路径节省。",
    "group_mapping_limit": "最终设备 PTO 不保留 group ID。0..31 用已有 long CPU LowerPipelineLoops 成员及实际设备循环树对应；32/33 用 medium 的相同 O-B M96 分支对应。两份辅助 CPU 不是本次设备产物，不能替代设备地址证据。",
    "fused_hints": hints(F / "report/perf_hints.log"),
    "original_hints": hints(O / "report/perf_hints.log"),
    "fused_allocations": fused,
    "fused_compact": compact(fused),
    "original_allocations": original,
    "original_compact": {name: compact(rows) for name, rows in original.items()},
    "cpu_long_memberships": memberships(CPU_LONG),
    "cpu_medium_memberships": memberships(CPU_MEDIUM),
    "cpp_evidence": [
        snippet(F / "kernels/aic/cube_schedule.cpp", 139, 155),
        snippet(F / "kernels/aic/cube_schedule.cpp", 6101, 6180),
        snippet(O / "kernels/aic/hc_pre_linear.cpp", 330, 409),
        snippet(F / "kernels/aic/cube_schedule.cpp", 8047, 8180),
        snippet(O / "kernels/aic/kv_score_proj_0.cpp", 437, 561),
        snippet(F / "kernels/aic/cube_schedule.cpp", 10742, 10788),
        snippet(O / "kernels/aic/qproj_matmul.cpp", 399, 445),
        snippet(F / "kernels/aic/cube_schedule.cpp", 18615, 18726),
    ],
}
result["inputs"] = [{"path": str(p), "size_bytes": p.stat().st_size} for p in sorted(INPUTS)]
(OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": str(OUT / "result.json"), "input_files": len(INPUTS),
                  "fused_aic_operand_allocations": len(fused),
                  "original_kernels": len(original)}, ensure_ascii=False))
