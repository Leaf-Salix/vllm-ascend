"""只读整模型forward样本；复用已有rank门禁，不依赖耗时的trace离线解析。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NATIVE_ROOT = ROOT.parent / "coalesced/model"
ARTIFACTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ARTIFACTS.parent))
from offline_pd.performance import compare_worker_configs, distribution, load_rank, require  # noqa: E402

CASES = ((131072, 16), (8192, 40))
OPERATOR_REVISION = "2a740c1f (original cache layout)"


def collect_case(history, batch):
    folder = ROOT / f"h{history}" / f"b{batch}"
    bank = ARTIFACTS / "release_offline_pd_20260923" / f"h{history}_bank"
    plan = json.loads((bank / "plan.json").read_text())
    row = {"history": history, "batch": batch, "operator_revision": OPERATOR_REVISION, "errors": [],
           "token_mismatches": 0, "dspark_mismatched_ranks": 0, "compared_tokens": 0,
           "ranks": [], "source": str(folder)}
    samples = {side: [] for side in ("native", "pto")}
    per_step = {side: [] for side in samples}
    for rank in range(16):
        try:
            values = {side: load_rank((NATIVE_ROOT / f"h{history}/b{batch}" if side == "native" else folder),
                                      side, rank, 2, plan, batch=batch, tokens=128,
                                      steps=3, max_num_seqs=40, steady_cycles=10) for side in samples}
            native, pto = (values[side][0] for side in samples)
            for key in ("key", "history", "capture_sizes", "max_num_seqs", "deterministic",
                        "hccl_deterministic", "atomic_add", "eplb_enabled", "dynamic_eplb_env",
                        "expert_map_record_env", "custom_opp_path", "requested_steady_cycles"):
                require(native[key] == pto[key], f"rank{rank}: 两侧{key}不同")
            require(native["history"] == history, f"rank{rank}: 历史长度与矩阵不同")
            event_modes = compare_worker_configs(native["worker_runtime_config"][0],
                                                 pto["worker_runtime_config"][0])
            mismatch = sum(a != b for name in ("steady_output_token_ids", "output_token_ids")
                           for left, right in zip(native[name], pto[name]) for a, b in zip(left, right))
            stats_equal = values["native"][1] == values["pto"][1]
            row["token_mismatches"] += mismatch
            row["dspark_mismatched_ranks"] += not stats_equal
            row["compared_tokens"] += 2 * batch * 128
            detail = {"rank": rank, "token_mismatches": mismatch, "dspark_equal": stats_equal,
                      "cann_event_work_mode": event_modes}
            for side, (value, _) in values.items():
                steady = value["steady_window"][0]
                require(steady["warmup_steps"] == 8, f"rank{rank}: warmup与本轮声明不同")
                device = steady["forward"]["samples_us"]
                samples[side].extend(device)
                per_step[side].append(device)
                detail[side] = {"samples_us": device, "distribution": distribution(device),
                                "worker_runtime_config": value["worker_runtime_config"][0]}
            row["ranks"].append(detail)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row["errors"].append(f"rank{rank}: {exc}")
    if len(row["ranks"]) == 16 and not row["errors"]:
        row["forward"] = {side: distribution(v) for side, v in samples.items()}
        row["slowest_rank_forward"] = {
            side: distribution([max(values) for values in zip(*ranks)]) for side, ranks in per_step.items()}
        row["change_pct"] = (row["forward"]["pto"]["mean_us"] / row["forward"]["native"]["mean_us"] - 1) * 100
    row["status"] = ("MEASURED_TOKEN_PASS" if not row["errors"] and not row["token_mismatches"]
                     and not row["dspark_mismatched_ranks"] else "FAIL")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available", action="store_true", help="仅读取32份rank记录都完成的档位")
    args = parser.parse_args()
    rows = []
    for history, batch in CASES:
        if args.available:
            paths = [(NATIVE_ROOT if side == "native" else ROOT) / f"h{history}/b{batch}" / side /
                     f"rank{rank}.performance.json"
                     for side in ("native", "pto") for rank in range(16)]
            try:
                if any(json.loads(p.read_text()).get("stage") != "measured" for p in paths):
                    continue
            except (OSError, ValueError):
                continue
        rows.append(collect_case(history, batch))
    result = {"operator_revision": "2a740c1f (original cache layout)", "complete": len(rows) == len(CASES),
              "scope": ("每rank前8步后连续10步纯decode _model_forward；"
                        "Native复用前一任务235803控制，PTO为原布局2a740c1f；入场仍未门控。"),
              "limits": ("P95由16rank×10步的160个相关样本计算，最慢rank分布另列；"
                         "不是CSA时间、完整decode周期或初始化耗时。10步不能证明罕见长尾已消失。"),
              "cases": rows}
    (ROOT / "forward.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [f"# 当前性能版整模型forward（{OPERATOR_REVISION}）", "", result["scope"], "", result["limits"], "",
             "单位ms，负变化表示PTO更快。", "",
             "| 档位 | Native均值 | PTO均值 | PTO变化 | Native/PTO P95 | Native/PTO最大值 | token/DSpark |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        if "forward" not in row:
            lines.append(f"| {row['history']//1024}K/B{row['batch']} | 数据不全 | — | — | — | — | FAIL |")
            continue
        n, p = (row["forward"][s] for s in ("native", "pto"))
        lines.append(f"| {row['history']//1024}K/B{row['batch']} | {n['mean_us']/1000:.3f} | {p['mean_us']/1000:.3f} | "
                     f"{row['change_pct']:+.2f}% | {n['p95_us']/1000:.3f}/{p['p95_us']/1000:.3f} | "
                     f"{n['max_us']/1000:.3f}/{p['max_us']/1000:.3f} | {row['status']} |")
    lines += ["", "每步取16rank中最大的forward耗时，再对10步求均值；用于观察EP16最慢rank的影响。", "",
              "| 档位 | Native最慢rank均值 | PTO最慢rank均值 | PTO变化 | Native/PTO P95÷P50 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if "forward" not in row:
            continue
        n, p = (row["slowest_rank_forward"][s] for s in ("native", "pto"))
        fn, fp = (row["forward"][s] for s in ("native", "pto"))
        lines.append(f"| {row['history']//1024}K/B{row['batch']} | {n['mean_us']/1000:.3f} | "
                     f"{p['mean_us']/1000:.3f} | {(p['mean_us']/n['mean_us']-1)*100:+.2f}% | "
                     f"{fn['p95_us']/fn['p50_us']:.3f}/{fp['p95_us']/fp['p50_us']:.3f} |")
    lines += ["", "P95÷P50使用全部rank样本；每步最慢rank序列共10个样本，其P95等于最大值。",
              "每步按相同稳态步编号对齐，最慢rank耗时不包含各rank起始时间偏差或步间等待。", "",
              f"已取得{len(rows)}/2档。逐rank样本、每步最慢rank分布、实际配置和错误详情"
              "见[forward.json](forward.json)。",
              "旧版本全模型和当前单层数据不混入本表。"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)}/2 cases")
    for row in rows:
        print(row["history"], row["batch"], row["status"], row.get("change_pct"), row["errors"])
    if any(r["status"] == "FAIL" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
