"""读取已授权七档的真实报告和队列终态；不编译、不提交或重试设备任务。"""

import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


E = Path(__file__).resolve().parent
C = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1")
D = C / "tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
PAR = D / "parallel_20261018"
SOURCE = Path("/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/ultra_compact_priority_20261018/source_snapshot")
SIDES = ("dynamic_pto", "spmd_reference", "spmd_candidate")
REPRESENTATIVES = [
    {"history": 131072, "batch": 16, "task_id": "task_20261018_013723_19521519782"},
    {"history": 8192, "batch": 32, "task_id": "task_20261018_014358_34138622015"},
]


def checks(node, trail=""):
    """只检查明确的数值/保护区/协议结果，不把原始数值字段中的零当布尔失败。"""
    if isinstance(node, dict):
        if "status" in node:
            assert node["status"] == "PASS", (trail, node)
        if "mismatches" in node:
            assert node["mismatches"] == 0, (trail, node)
        if "atol" in node:
            assert node["atol"] == 0 and node["rtol"] == 0, (trail, node)
        if "passed" in node:
            assert node["passed"] is True, (trail, node)
        for key, child in node.items():
            checks(child, trail + "/" + key)
    elif isinstance(node, list):
        for i, child in enumerate(node):
            checks(child, trail + f"/{i}")


def evaluate(job):
    history, batch, task_id = job["history"], job["batch"], job["task_id"]
    out = PAR / f"compact_priority_h{history}_b{batch}_v1"
    status = subprocess.run(["task-submit", "--status", task_id], cwd=C, text=True, capture_output=True)
    status_text = (status.stdout + status.stderr).strip()
    (E / f"status_h{history}_b{batch}.log").write_text(status_text + "\n")
    row = {"history": history, "batch": batch, "task_id": task_id, "task_status": status_text,
           "output": str(out), "result": str(out / "result.json"),
           "source_matrix_result": str(out / "source_matrix/result.json"), "collected": False}
    if (out / "result.json").exists():
        report = json.loads((out / "result.json").read_text())
        row["stage"] = report.get("stage")
    else:
        report = None
    if status_text in {"pending", "running"}:
        row["pending"] = True
        return row
    row["pending"] = False
    # 只有队列已终态才保存完整日志与判定；不会把构建中间 passed=false 当成失败。
    if not (E / f"task_h{history}_b{batch}.log").exists():
        log = subprocess.run(["task-submit", "--log", task_id], cwd=C, text=True, capture_output=True)
        (E / f"task_h{history}_b{batch}.log").write_text(log.stdout + log.stderr)
    try:
        assert status.returncode == 0 and status_text == "completed (exit=0)", status_text
        assert report is not None and report["stage"] == "complete" and report["passed"] is True
        assert report["history"] == history and report["batch"] == batch and report["query_tokens"] == 6
        assert report["device_execution"] is True and report["full_csa_validated"] is True
        assert report["performance_measured"] is True and report["graph_replay_validated"] is True
        assert report["wait_backoff"] is True and report["phase_clock"] is False
        assert report["source_package"] == str(SOURCE)
        assert report["native_attention_run"] is False
        assert report["post_l1_protocol"] is True
        assert all(value is True for value in report["post_l1_scratch_guards"].values())
        checks(report["post_l1_guards"], "post_l1_guards")
        assert len(report["launches"]) == 3
        for launch in report["launches"]:
            assert launch["passed"] is True and launch["protocol"] is True
            assert all(value is True for value in launch["scratch_guards"].values())
            checks(launch["against_dynamic"], "launch/against_dynamic")
            checks(launch["guards"], "launch/guards")
        matrix = json.loads((out / "source_matrix/result.json").read_text())
        assert matrix == report["source_matrix"]
        assert matrix["passed"] is True and matrix["same_process"] is True and matrix["same_root_addresses"] is True
        assert matrix["reference_source"] == str(PAR / "baseline_source")
        assert matrix["policies"]["spmd_candidate"] == matrix["policies"]["spmd_reference"]
        checks(matrix["checks"], "source_matrix/checks")
        checks(matrix["protocol"], "source_matrix/protocol")
        assert len(matrix["windows"]) == 15
        samples, medians = {}, {}
        for side in SIDES:
            windows = [window for window in matrix["windows"] if window["side"] == side]
            assert len(windows) == 5
            assert all(window["clock_freq_hz"] == 50000000 and window["cores"] == 72 for window in windows)
            if side.startswith("spmd_"):
                assert all(window["task_count"] == 1 for window in windows)
            samples[side] = [window["incore_span_us"] for window in windows]
            medians[side] = statistics.median(samples[side])
            assert samples[side] == matrix["statistics_us"][side]["samples_us"]
            assert abs(medians[side] - matrix["statistics_us"][side]["median_us"]) < 1e-9
        candidate = medians["spmd_candidate"]
        reduction_reference = (1 - candidate / medians["spmd_reference"]) * 100
        reduction_original = (1 - candidate / medians["dynamic_pto"]) * 100
        row.update(collected=True, passed=True, query_tokens=6, source_package=report["source_package"],
                   device=report["environment"]["device"], policy=report["policy"],
                   samples_us=samples, medians_us=medians, windows=matrix["windows"],
                   reduction_vs_593_pct=reduction_reference, reduction_vs_original_task_pct=reduction_original,
                   target_30pct_met=candidate <= medians["dynamic_pto"] * 0.7,
                   target_latency_us=medians["dynamic_pto"] * 0.7,
                   gap_above_target_us=candidate - medians["dynamic_pto"] * 0.7,
                   candidate_vs_reference_report=matrix["candidate_vs_reference"],
                   comparison_report=matrix["comparison"],
                   validation={"full_current_fixture": True, "three_generations": True,
                               "graphs": True, "post_l1_exact": True, "native_memory_guards": True,
                               "scratch_guards": True, "protocol": True, "single_mix_per_window": True},
                   protocol=matrix["protocol"])
    except (AssertionError, KeyError, ValueError, TypeError) as error:
        row.update(passed=False, error=repr(error), action="先定位真实失败或审计边界；不自动重测。")
        if report is not None:
            row["device_report_error"] = report.get("error")
    return row


def main():
    jobs = json.loads((PAR / "compact_priority_remaining_jobs.json").read_text())["jobs"]
    assert len(jobs) == 5
    rows = [evaluate(job) for job in sorted(jobs + REPRESENTATIVES, key=lambda j: (-j["history"], j["batch"]))]
    complete = all(row["collected"] for row in rows)
    report = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), "complete": complete,
        "passed": complete and all(row.get("passed") for row in rows),
        "candidate_source": str(SOURCE), "reference_source": str(PAR / "baseline_source"),
        "registry": str(PAR / "compact_priority_remaining_jobs.json"),
        "method": "每档同进程、同根地址的原分task/593旧SPMD/阶段前移候选，交替顺序各五窗完整incore首尾；中位数由原始窗口重新计算。不能把不同作业总时差算作收益。",
        "scope": "七档均S6/正式第4层权重/合成历史；当前fixture输出、TopK、cache/state、graph、代际和保护区。",
        "excluded": ["Native attention正确性/性能未运行，既有差异不在本次验收范围", "token生成或整模型端到端验收不在范围", "未覆盖全部动态metadata边界或连续decode"],
        "threshold": "每档候选完整L1相对同进程原分task降低至少30%；相对593改善不能替代该门槛。",
        "rows": rows,
    }
    if complete:
        report["seven_shape_30pct_target_met"] = all(row["target_30pct_met"] for row in rows)
        report["all_improve_vs_593"] = all(row["reduction_vs_593_pct"] > 0 for row in rows)
        report["all_faster_than_original_task"] = all(row["reduction_vs_original_task_pct"] > 0 for row in rows)
    (E / "remaining_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        print(json.dumps({k: row[k] for k in ["history", "batch", "task_id", "task_status", "stage", "passed", "medians_us", "reduction_vs_593_pct", "reduction_vs_original_task_pct", "error"] if k in row}, ensure_ascii=False), flush=True)
    print(json.dumps({"complete": complete, "output": str(E / "remaining_summary.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
