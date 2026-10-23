# SPDX-License-Identifier: Apache-2.0
"""收取v37两个已完成对照及核内阶段；不启动新测试。"""

import collections
import importlib.util
import json
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location(
        "csa_l1_mapper", HERE.parent / "csa_performance_20261022_aicpu/export_l1_download.py"
    )
    mapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapper)
    preparation = json.loads((HERE / "scheduling_v37_preparation.json").read_text())
    result = dict(candidate=preparation["candidate"], adopted=False, stop_after_round=True, cases=[])
    for job in preparation["jobs"]:
        status = subprocess.check_output(["task-submit", "--status", job["task_id"]], text=True).strip()
        assert status == "completed (exit=0)"
        path = HERE / job["output"] / "result.json"
        report = json.loads(path.read_text())
        assert report["passed"] and report["stage"] == "complete"
        checks = {}
        for name, check in report["checks"].items():
            assert all(check["guards"].values())
            assert all(x["status"] == "PASS" and x["bit_equal"] for x in check["numerical"].values())
            checks[name] = dict(
                all_guards_passed=True,
                guard_count=len(check["guards"]),
                numerical={
                    n: {k: v[k] for k in ("status", "bit_equal", "max_abs") if k in v}
                    for n, v in check["numerical"].items()
                },
            )
        builds = list((path.parent / "build_output").glob("decode_csa_tnd_*/orchestration/*.cpp"))
        cpp = {}
        for source in builds:
            vector = source.parents[1] / "kernels/aiv/index_score_topk_vector.cpp"
            reductions = vector.read_text().count("TROWMAX(")
            assert reductions in (0, 1)
            side = "mpmd" if reductions else "mpmd_control"
            assert side not in cpp
            cpp[side] = source
        assert set(cpp) == {"mpmd", "mpmd_control"}
        windows = []
        for window in report["windows"]:
            raw_path = Path(window["raw"])
            raw = json.loads(raw_path.read_text())
            rows = raw["aicore_tasks"]
            mapping = mapper.task_map(raw_path, generated_cpp=cpp[window["side"]])[1]
            assert set(mapping) == {r[1] for r in rows}
            origin = min(r[3] for r in rows)
            scale = 1e6 / raw["metadata"]["clock_freq_hz"]
            assert abs((max(r[4] for r in rows) - origin) * scale - window["incore_span_us"]) < 1e-8
            groups = collections.defaultdict(list)
            for core, task, _, start, end, _ in rows:
                groups[mapping[task]["name"]].append((core, (start - origin) * scale, (end - origin) * scale))
            stages = {}
            for name in ("indexer_score", "indexer_topk", "sparse_attention", "o_proj_a", "hc_post"):
                times = groups[name]
                stages[name] = dict(
                    start_us=min(r[1] for r in times),
                    end_us=max(r[2] for r in times),
                    span_us=max(r[2] for r in times) - min(r[1] for r in times),
                    incore_max_us=max(r[2] - r[1] for r in times),
                    incore_sum_us=sum(r[2] - r[1] for r in times),
                )
            windows.append(dict(side=window["side"], raw=window["raw"], l1_us=window["incore_span_us"], stages=stages))
        summary = {}
        for side in ("mpmd_control", "mpmd"):
            selected = [w for w in windows if w["side"] == side]
            assert len(selected) == 10
            samples = [w["l1_us"] for w in selected]
            summary[side] = dict(
                samples_us=samples,
                min_us=min(samples),
                median_us=statistics.median(samples),
                mean_us=statistics.mean(samples),
                max_us=max(samples),
                stage_medians={
                    name: {k: statistics.median(w["stages"][name][k] for w in selected) for k in metrics}
                    for name, metrics in selected[0]["stages"].items()
                },
            )
        log = subprocess.check_output(["task-submit", "--log", job["task_id"]], text=True)
        bindings = [
            json.loads(line.split(" ", 1)[1]) for line in log.splitlines() if line.startswith("CSA_SHARED_TOOLCHAIN ")
        ]
        assert len(bindings) == 1 and bindings[0]["status"] == "validated"
        result["cases"].append(
            dict(
                source=str(path),
                case=report["case"],
                scope=job["scope"],
                task_id=job["task_id"],
                status=status,
                actual_toolchain=bindings[0],
                checks=checks,
                summary=summary,
                all_windows=windows,
                cpp_by_side={side: str(source) for side, source in cpp.items()},
            )
        )
        print(report["case"]["name"], {s: {k: v[k] for k in ("median_us", "max_us")} for s, v in summary.items()})
        print("Score", {s: v["stage_medians"]["indexer_score"] for s, v in summary.items()})
    (HERE / "scheduling_v37_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
