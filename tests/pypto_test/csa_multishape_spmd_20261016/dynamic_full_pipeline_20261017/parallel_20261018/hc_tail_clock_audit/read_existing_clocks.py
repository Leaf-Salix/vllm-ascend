"""只读已有完整 CSA 插桩产物，汇总 HC/输出尾部端点，不编译、不执行 NPU。

输入为 attention_tick_h131072_b16_v1 的三份 phase_clock.json、同作业源码和
PTO 文本；输出只是派生证据。端点差包含等待/发布等成本，不是分段纯计算时长，
更不能据此推导未实现候选收益。所有源路径及字节数保留，不生成 hash。
"""

import argparse
import json
import statistics
from pathlib import Path


def distribution(values):
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "max": round(max(values), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    par = args.canonical / "tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018"
    job = par / "attention_tick_h131072_b16_v1"
    inputs = []

    def read(path):
        text = path.read_text()
        inputs.append({"path": str(path), "bytes": path.stat().st_size})
        return text

    result = json.loads(read(job / "result.json"))
    phases = ["BEGIN", "INPUT", "PARTIAL", "RAW", "GATES", "NORM", "COMB", "QA", "QR", "ROPE", "ATTN_PLAN", "ATTN_CUBE", "ATTN_VECTOR", "OB", "OUT"]
    rounds = []
    for index in range(3):
        rows = json.loads(read(job / f"clock_round_{index}_spmd_reference/phase_clock.json"))
        groups = {name: [row for row in rows if row["name"] == name] for name in phases}
        assert all(groups.values()), "不得把缺失阶段默认为零耗时"
        stats = {
            name: {
                key: distribution([row[key] for row in group])
                for key in ("notify_start_us", "notify_end_us")
            }
            for name, group in groups.items()
        }
        # 同核区间保留核编号；其中仍有 publish、wait 和业务计算，不能改称纯算子耗时。
        intervals = {}
        for before, after in (("GATES", "NORM"), ("NORM", "COMB")):
            first = {row["core"]: row for row in groups[before]}
            intervals[f"{before}_start_to_{after}_start_us"] = [
                {"core": row["core"], "us": round(row["notify_start_us"] - first[row["core"]]["notify_start_us"], 3)}
                for row in groups[after] if row["core"] in first
            ]
        rounds.append({
            "round": index,
            "phase_endpoints": stats,
            "same_core_intervals": intervals,
            "comb_latest_end_before_qa_earliest_start_us": round(stats["QA"]["notify_start_us"]["min"] - stats["COMB"]["notify_end_us"]["max"], 3),
            "comb_latest_start_before_qa_latest_start_us": round(stats["QA"]["notify_start_us"]["max"] - stats["COMB"]["notify_start_us"]["max"], 3),
            "ob_latest_start_to_out_latest_start_us": round(stats["OUT"]["notify_start_us"]["max"] - stats["OB"]["notify_start_us"]["max"], 3),
            "attention_vector_latest_start_to_out_latest_start_us": round(stats["OUT"]["notify_start_us"]["max"] - stats["ATTN_VECTOR"]["notify_start_us"]["max"], 3),
        })
    source = job / "sources/spmd_reference"
    for name in ("hc_stages.py", "cube_schedule.py", "vector_schedule.py", "output_projection_b_stages.py"):
        read(source / name)
    pto_paths = list((job / "build_output").glob("*/ptoas/dynamic_csa_full.pto"))
    assert len(pto_paths) == 1, pto_paths
    pto_path = pto_paths[0]
    lines = read(pto_path).splitlines()
    allocations = []
    for line_number, line in enumerate(lines, 1):
        if "pto.alloc_tile" in line and ("%norm_sq_sum_" in line or "%y_inv_" in line):
            allocations.append({"line": line_number, "text": line.strip()})
    assert len(allocations) == 2, allocations
    assert all("loc=vec" in item["text"] for item in allocations)
    start, end = allocations[0]["line"] - 2, allocations[-1]["line"] + 12
    excerpt = "\n".join(f"{i+1}: {lines[i]}" for i in range(start, min(end, len(lines)))) + "\n"
    (out / "hc_inverse_local_tiles.pto.txt").write_text(excerpt)
    evidence = {
        "scope": "只读既有三轮诊断；没有新编译、设备执行或正式源码修改",
        "task_id": "task_20261018_032708_262386210870",
        "diagnostic_passed": result.get("passed"),
        "eligible_for_performance": False,
        "timing_caveats": [
            "Attention tick 插桩会扰动同轮运行以及下游端点；不得与无插桩 L1 作性能比较。",
            "端点为发布前后 Scalar 时钟；窗口包括等待、业务运算、缓存维护和发布。",
            "不同核的最大值差仅作阶段包络；不能当作某核纯计算耗时或优化收益。",
        ],
        "rounds": rounds,
        "hc_inverse_existing_vec_allocations": allocations,
        "source_inputs": inputs,
        "conclusion": "没有足够新证据支持本轮 HC/pre-QA/输出尾部候选；已有方向不重复实现。",
    }
    (out / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(out), "rounds": [{k: v for k, v in row.items() if k not in ("phase_endpoints", "same_core_intervals")} for row in rounds], "local_tile_count": len(allocations)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
