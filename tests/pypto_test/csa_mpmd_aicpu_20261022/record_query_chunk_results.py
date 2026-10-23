# SPDX-License-Identifier: Apache-2.0
"""只读既有v30收据与L1，生成收尾报告；不启动设备测试。"""

import collections
import importlib.util
import json
import re
import statistics
from pathlib import Path

P = Path(__file__).resolve().parent
R = P.parents[2]
Q = P.parent / "csa_performance_20261022_aicpu"
spec = importlib.util.spec_from_file_location("old_mapper", Q / "export_l1_download.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def rel(p):
    return str(p.relative_to(R))


def check_summary(checks):
    return {
        k: {
            "guards": all(v["guards"].values()),
            "numerical": all(x["status"] == "PASS" for x in v["numerical"].values()),
            "bit_equal": all(x["bit_equal"] for x in v["numerical"].values()),
        }
        for k, v in checks.items()
    }


def map_true(raw_path, cpp):
    text = cpp.read_text()
    names = json.loads(raw_path.with_name("names.json").read_text())["callable_id_to_name"]
    mixed = {
        n: [int(x.strip()) for x in ids.split(",")]
        for n, ids in re.findall(r"MixedKernels (mixed_\d+) = \{([^}]+)\};", text)
    }
    scopes = []
    for ma in re.finditer(r"SIMPLER_SCOPE\(ScopeMode::MANUAL\)", text):
        a = text.index("{", ma.end())
        i = a + 1
        depth = 1
        while depth:
            depth += (text[i] == "{") - (text[i] == "}")
            i += 1
        scopes.append((a, i))
    assert len(scopes) == 3
    records = []
    for ma in re.finditer(
        r"(?m)^ +TaskOutputTensors task_(\d+)_outs = rt_submit_(aic_task|aiv_task|task)\((\w+), params_t\1\);", text
    ):
        site, kind, func = ma.groups()
        ids = mixed[func] if kind == "task" else [int(func)]
        src = [names[str(i)] for i in ids]
        if kind == "task":
            assert len(src) == 3 and src[1] == src[2]
            if src[0].startswith("index_score_topk_cube"):
                name = "indexer_score"
            else:
                assert src[0].startswith("sparse_attn_cube")
                name = "sparse_attention"
        else:
            name = re.sub(r"_\d+$", "", src[0])
            name = {"o_b_group_task": "o_proj_b"}.get(name, name)
            name = name.replace("_query_chunk", "")
        records.append(
            {"name": name, "kind": kind, "source_names": src, "submit_site": int(site), "position": ma.start()}
        )
    common = [x for x in records if x["position"] < scopes[0][0]]
    chunks = [x for x in records if scopes[0][0] < x["position"] < scopes[0][1]]
    output = [x for x in records if scopes[1][0] < x["position"] < scopes[1][1]]
    post = [x for x in records if scopes[1][1] < x["position"] < scopes[2][0] and x["name"] == "hc_post"]
    assert len(chunks) == 4 and len(output) == 3 and len(post) == 1
    assert [x["name"] for x in chunks] == ["indexer_score", "indexer_topk", "csa_attention_plan", "sparse_attention"]
    mapping = {(1 << 32) | i: dict(x) for i, x in enumerate(common)}
    mapping[2 << 32] = post[0]
    for chunk in range(2):
        for i, x in enumerate(chunks):
            mapping[(3 << 32) | (chunk * 4 + i)] = dict(x, query_chunk=chunk)
    for group in range(8):
        for i, x in enumerate(output):
            mapping[(3 << 32) | (8 + group * 3 + i)] = dict(x, output_group=group)
    return mapping


report = {
    "candidate": "aicpu_v30_query_chunk_pipeline",
    "adopted": False,
    "control": "aicpu_v26_uniform_score_geometry",
    "production_stays": "d86c95f69 v22",
    "shared_release": "csa-shared-20261022-logical-codegen",
    "patch_base": "d86c95f69",
    "patch": "scheduling_v30_query_chunks.patch",
    "cpu_attempts": [
        {
            "source": rel(P / "runs" / f"{n}_h131072_b24_s6/result.json"),
            "passed": json.loads((P / "runs" / f"{n}_h131072_b24_s6/result.json").read_text())["passed"],
        }
        for n in ["cpu_v30", "cpu_v30b", "cpu_v30c"]
    ],
    "performance": [],
}
for case in ["h131072_b24_s6", "h8192_b32_s6", "h131072_b16_mixed"]:
    f = P / "runs" / f"aicpu_v30_pair_{case}/result.json"
    d = json.loads(f.read_text())
    assert d["passed"] and d["stage"] == "complete"
    checks = check_summary(d["checks"])
    assert all(all(v.values()) for v in checks.values())
    stats = d["statistics_us"]
    for s in stats.values():
        s["median"] = statistics.median(s["samples"])
    row = {
        "case": case,
        "source": rel(f),
        "checks": checks,
        "statistics_us": stats,
        "l1_raw": [{"side": w["side"], "raw": w["raw"]} for w in d["windows"]],
        "query_chunks_executed": case == "h131072_b24_s6",
    }
    print(case, [(s, round(v["median"], 2), v["max"]) for s, v in stats.items()])
    if case == "h131072_b24_s6":
        cpp_candidates = list((f.parent / "build_output").glob("decode_csa_tnd_*/orchestration/*.cpp"))
        by_side = {
            s: next(c for c in cpp_candidates if ("query_chunk" in c.read_text()) == (s == "mpmd"))
            for s in ["mpmd", "mpmd_control"]
        }
        summaries = []
        for w in d["windows"]:
            raw_path = Path(w["raw"])
            raw = json.loads(raw_path.read_text())
            mapping = (
                map_true(raw_path, by_side[w["side"]])
                if w["side"] == "mpmd"
                else old.task_map(raw_path, generated_cpp=by_side[w["side"]])[1]
            )
            assert set(mapping) == {r[1] for r in raw["aicore_tasks"]}
            origin = min(r[3] for r in raw["aicore_tasks"])
            factor = 1e6 / raw["metadata"]["clock_freq_hz"]
            stages = collections.defaultdict(list)
            for core, tid, _, start, end, _ in raw["aicore_tasks"]:
                x = mapping[tid]
                assert x["kind"] == "task" or x["kind"] == ("aic_task" if core < 24 else "aiv_task")
                label = x["name"] + ("/chunk" + str(x["query_chunk"]) if "query_chunk" in x else "")
                stages[label].append(((start - origin) * factor, (end - origin) * factor))
            metrics = {
                k: {
                    "start_us": min(x[0] for x in v),
                    "end_us": max(x[1] for x in v),
                    "span_us": max(x[1] for x in v) - min(x[0] for x in v),
                    "incore_max_us": max(x[1] - x[0] for x in v),
                }
                for k, v in stages.items()
                if k.startswith(
                    ("indexer_score", "indexer_topk", "csa_attention_plan", "sparse_attention", "o_proj_a", "hc_post")
                )
            }
            summaries.append({"side": w["side"], "raw": w["raw"], "metrics": metrics})
        row["all_window_stages"] = summaries
        row["stage_medians_us"] = {
            side: {
                name: {k: statistics.median(w["metrics"][name][k] for w in summaries if w["side"] == side) for k in met}
                for name, met in next(w["metrics"] for w in summaries if w["side"] == side).items()
            }
            for side in ["mpmd_control", "mpmd"]
        }
        for side, ms in row["stage_medians_us"].items():
            print(
                side,
                {
                    n: {k: round(v, 2) for k, v in vs.items()}
                    for n, vs in ms.items()
                    if n.startswith(("indexer_score", "sparse_attention", "indexer_topk", "csa_attention_plan"))
                },
            )
    report["performance"].append(row)
f = P / "runs/aicpu_v30_state_h131072_b28/result.json"
d = json.loads(f.read_text())
s = d["state_validation"]
assert d["passed"] and s["passed"]
report["state_validation"] = {
    "source": rel(f),
    "passed": True,
    "metadata_cases": [
        {"name": x["name"], "passed": x["passed"], "checks": check_summary(x["checks"])} for x in s["metadata_cases"]
    ],
    "continuous_steps": [{"passed": x["passed"], "checks": check_summary(x["checks"])} for x in s["continuous_steps"]],
}
assert all(
    all(v.values())
    for x in report["state_validation"]["metadata_cases"] + report["state_validation"]["continuous_steps"]
    for v in x["checks"].values()
)
report["decision"] = (
    "两块独立业务流水功能及同址10+4状态逐bit通过；长B24实际67组任务，中位1116.55→1306.36us回退，最大1126.60→1516.52us。短G2与混长走fallback。按用户要求本轮到此停止，不追加候选或新档测试，不采用生产；保留完整源码补丁及失败证据。"
)
(P / "scheduling_v30_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
jobs = json.loads((P / "jobs.json").read_text())
for j in jobs:
    f = P / j.get("output", "none") / "result.json"
    if f.exists():
        d = json.loads(f.read_text())
        if d.get("stage") == "complete" and d.get("passed"):
            j["last_observed_status"] = "completed (exit=0)"
(P / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n")
