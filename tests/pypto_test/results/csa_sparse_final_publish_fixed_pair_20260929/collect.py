"""Reuse full-state/official DFX validation, then report the fused Sparse boundary."""

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-evidence", action="store_true",
                        help="只重排已有完整证据，不重复大张量检查或 DFX 转换")
    args = parser.parse_args()
    path = ROOT.parent / "csa_score_segment_ub_20260929/collect.py"
    spec = importlib.util.spec_from_file_location("sparse_publish_pair", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.SOURCE_PREFIX = ROOT.parents[4] / ".cache/csa-sparse-final-publish-fix-4ffccb7b"
    module.VARIANT = "pkg:dsv4_csa_sparse_final_publish_4ffccb7b"
    module.TITLE = "Sparse最后PV块融合归一化、逆RoPE及分组发布"
    module.CHANGE_DESCRIPTION = "Sparse最终输出在最后PV块内发布，现有算术、Indexer和cache布局保持。"
    original_load = module.load

    def load_with_publication(name, path):
        loaded = original_load(name, path)
        if name == "coefficient_worker":
            original_summary = loaded.summarize

            def summarize(window):
                side = window.relative_to(ROOT).parts[2]
                if side not in ("baseline", "candidate"):
                    raise ValueError("Unknown frozen side for publication boundary")
                publication = "qk_pv_aiv" if side == "candidate" else "merge_norm"
                return original_summary(window, publication_task=publication)

            loaded.summarize = summarize
        return loaded

    module.load = load_with_publication
    if not args.reuse_evidence:
        # Shared collection writes its complete evidence before refusing a bad
        # state comparison. Render that failure explicitly without relaxing it.
        try:
            module.main()
        except SystemExit:
            if read(ROOT / "evidence.json")["state_status"] != "FAIL":
                raise
    evidence = read(ROOT / "evidence.json")
    summary = read(ROOT / "summary.json")
    state_status = evidence["state_status"]
    lines = ["# Sparse最后PV块融合归一化、逆RoPE及分组发布", "",
             f"**跨版本状态检查：{state_status}。** " + (
                 "该候选不得采用，下列计时仅保存失败实验的原始观察。" if state_status != "PASS" else
                 "完整状态零容差通过；其他验收范围见下文。"), "",
             "同卡CANN9.2/mode2/atomic0/det0，4ffccb7b对私有候选；5预热20次正式图事件，单位μs。", "",
             "| 档位 | CSA基线→候选 | 变化 | P95 | 最大值 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for case in evidence["cases"]:
        a, b = (case["sides"][s] for s in ("baseline", "candidate"))
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {a['mean_us']:.3f}→{b['mean_us']:.3f} "
                     f"| {case['csa_change_pct']:+.3f}% | {a['us_p95']:.3f}→{b['us_p95']:.3f} "
                     f"| {a['us_max']:.3f}→{b['us_max']:.3f} |")
    lines += ["", f"长短8:2完整CSA变化：{evidence['weighted_8_2_change_pct']:+.3f}%。", "",
              "独立四窗口DFX：QK/PV核时包含核内流水等待；融合前后工作范围不同，另列完整AIV工作量。", "",
              "| 档位 | Sparse AIC核时 | Sparse AIV核时 | 独立merge核时 | 合计AIV核·μs | Sparse到发布设备跨度 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for case, compact in zip(evidence["cases"], summary["cases"]):
        values = {}
        for side, data in case["sides"].items():
            records = []
            for window in data["windows"]:
                tasks = window["tasks"]
                aic, aiv = (tasks[f"qk_pv_{core}"] for core in ("aic", "aiv"))
                merge = tasks.get("merge_norm")
                if (aic["blocks"], aiv["blocks"]) != (24, 48):
                    raise ValueError("Unexpected Sparse core coverage")
                if (merge is not None) != (side == "baseline"):
                    raise ValueError("The final publication was not fused as intended")
                if merge and merge["blocks"] != 48:
                    raise ValueError("Unexpected original merge coverage")
                records.append({
                    "aic_us": aic["kernel_mean_us"], "aiv_us": aiv["kernel_mean_us"],
                    "merge_us": merge["kernel_mean_us"] if merge else 0,
                    "aiv_core_us": aiv["kernel_mean_us"] * aiv["blocks"]
                    + (merge["kernel_mean_us"] * merge["blocks"] if merge else 0),
                    "publication_span_us": (merge["last_end_us"] if merge else
                                             max(aic["last_end_us"], aiv["last_end_us"]))
                    - min(aic["first_start_us"], aiv["first_start_us"]),
                })
            values[side] = {key: statistics.mean(v[key] for v in records) for key in records[0]}
            compact["sides"][side]["sparse_windows"] = records
            compact["sides"][side]["sparse_mean"] = values[side]
        change = 100 * (values["candidate"]["aiv_core_us"] / values["baseline"]["aiv_core_us"] - 1)
        compact["sparse_aiv_work_change_pct"] = change
        cells = [f"{values['baseline'][key]:.3f}→{values['candidate'][key]:.3f}"
                 for key in ("aic_us", "aiv_us", "merge_us", "aiv_core_us", "publication_span_us")]
        lines.append("| " + " | ".join([f"{case['history']//1024}K/B{case['batch']}", *cells]) + " |")
    summary["weighted_8_2_sparse_aiv_work_change_pct"] = sum(
        w * c["sparse_aiv_work_change_pct"] for w, c in zip((.8, .2), summary["cases"]))
    lines += ["", "合计AIV核时为Sparse 48个AIV加原merge 48个AIV，融合后merge缺失记为结构性0，非伪造样本。",
              "这不是将不同融合范围的单核均值直接比较，也不把核·μs或独立DFX跨度当完整CSA延迟。",
              "各自图/eager及保护区检查通过；这不能替代跨版本比较。未做Native逐元素或EP16 token/DSpark验收。",
              "[精简样本及检查](summary.json)、[Worker分项](TASKS.md)、[完整DFX证据](evidence.json)。"]
    for case in evidence["cases"]:
        failed = {name: value for name, value in case["state_checks"].items() if value["status"] != "PASS"}
        for name, value in failed.items():
            lines.append(f"\n{case['history']//1024}K/B{case['batch']}：`{name}` "
                         f"{value['mismatches']}/{value['elements']} 元素不同，"
                         f"max_abs={value['max_abs']}，RMSE={value['rmse']}；其余检查见完整证据。")
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if state_status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
