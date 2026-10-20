# SPDX-License-Identifier: Apache-2.0
"""离线核对静态 CSA 历史表、原始时间戳、源码及同卡二进制对照。

只读取已有记录，不编译、不启动设备、不计算文件 hash。历史绝对路径失效时，
仅按 tests/pypto_test 后缀定位迁移后的同一文件；不能用邻档结果补位。
"""

import ast
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIVE = HERE.parent
REPO = HERE.parents[3]
STATIC = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_static_reference"
OLD_TABLE = REPO / (
    "tests/pypto_test/csa_multishape_spmd_20261016/"
    "dynamic_full_pipeline_20261017/FIVE_WAY_PERFORMANCE_SPEEDUP_20261018.json"
)


def read(path):
    return json.loads(Path(path).read_text())


def relocated(path):
    path = Path(path)
    marker = "tests/pypto_test/"
    if marker in str(path):
        local = REPO / (marker + str(path).split(marker, 1)[1])
        if local.is_file():
            return local
    if path.is_file():
        return path
    raise FileNotFoundError(path)


def functions(path):
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(Path(path).read_text()).body
        if isinstance(node, ast.FunctionDef)
    }


def windows(report):
    return report["l1"]["windows"] if "l1" in report else report["l1_windows"]


def verify_windows(items):
    checked = []
    for item in items:
        path = relocated(item["raw"])
        raw = read(path)
        rows = raw["aicore_tasks"]
        frequency = raw["metadata"]["clock_freq_hz"]
        span = (max(row[4] for row in rows) - min(row[3] for row in rows)) * 1e6 / frequency
        assert abs(span - item["incore_span_us"]) < 1e-8
        assert raw["chip_swimlane_level"] == 1
        assert frequency == 50_000_000
        assert len({row[0] for row in rows}) == 72
        if item["side"] != "dynamic_pto":
            assert len(rows) == 72 and len({row[1] for row in rows}) == 1
        checked.append(dict(
            side=item["side"], us=span, raw=str(path),
            start_skew_us=(max(row[3] for row in rows) - min(row[3] for row in rows)) * 1e6 / frequency,
            max_core_elapsed_us=max(row[4] - row[3] for row in rows) * 1e6 / frequency,
        ))
    return checked


def stats(report):
    return report["l1"]["statistics_us"] if "l1" in report else report["l1_incore"]


def main():
    old_rows = read(OLD_TABLE)["rows"]
    new_rows = read(FIVE / "PERFORMANCE.json")["rows"]
    manifest = read(STATIC / "reference_manifest.json")["entries"]
    rows = []
    for new in new_rows:
        h, b = new["history"], new["batch"]
        if new["static_l1_us"] is None:
            rows.append(dict(history=h, batch=b, available=False,
                             reason="无已验收静态实现；未修复、未补值"))
            continue
        old = next(r for r in old_rows if (r["history"], r["batch"]) == (h, b))
        archive = next(r for r in manifest if (r["history"], r["batch"]) == (h, b))
        old_path = relocated(old["static_source"])
        new_path = FIVE / f"results/h{h}_b{b}/static_v1/result.json"
        old_report, new_report = read(old_path), read(new_path)
        old_stats, new_stats = stats(old_report), stats(new_report)
        old_side = "static_spmd" if "static_spmd" in old_stats else "single_spmd"
        old_us, new_us = old_stats[old_side]["median_us"], new_stats["static_spmd"]["median_us"]
        assert old_us == old["static_l1_us"] and new_us == new["static_l1_us"]
        old_entry = REPO / archive["source_entry"]
        new_entry = STATIC / archive["entry"]
        ast_equal = functions(old_entry) == functions(new_entry)
        workspace_equal = (
            read(REPO / archive["source_layout"])["workspace"]
            == read(new_entry.parent / "layout.json")["workspace"]
        )
        assert ast_equal and workspace_equal
        old_multi, new_multi = old_stats["dynamic_pto"]["median_us"], new_stats["dynamic_pto"]["median_us"]
        record = dict(
            history=h, batch=b, available=True,
            old_us=old_us, new_us=new_us, difference_us=new_us - old_us,
            change_pct=(new_us / old_us - 1) * 100,
            old_matched_multitask_us=old_multi, new_matched_multitask_us=new_multi,
            old_relative_gain_pct=(old_multi / old_us - 1) * 100,
            new_relative_gain_pct=(new_multi / new_us - 1) * 100,
            function_ast_equal=ast_equal, function_count=len(functions(old_entry)),
            workspace_schedule_equal=workspace_equal,
            old_result=str(old_path), new_result=str(new_path),
            old_entry=str(old_entry), new_entry=str(new_entry),
            old_environment=old_report.get("environment"),
            new_environment=new_report.get("environment"),
            old_windows=verify_windows(windows(old_report)),
            new_windows=verify_windows(windows(new_report)),
        )
        pair_path = HERE / f"results/h{h}_b{b}/pair_v1/result.json"
        if pair_path.exists():
            pair = read(pair_path)
            assert pair["passed"] and not pair["recompiled"]
            ss = pair["l1"]["statistics_us"]
            differences = [a - z for a, z in zip(ss["spmd_old"]["samples_us"], ss["spmd_new"]["samples_us"])]
            record["same_process_binary_pair"] = dict(
                result=str(pair_path), device=pair["environment"]["device"],
                statistics_us=ss,
                new_saving_us=ss["spmd_old"]["median_us"] - ss["spmd_new"]["median_us"],
                paired_savings_us=differences,
                paired_saving_median_us=statistics.median(differences),
                new_faster_windows=sum(x > 0 for x in differences),
                same_arguments_workspace=pair["identical_arguments_and_workspace"],
                protocol_generations=pair["total_protocol_generations"],
                raw_windows=verify_windows(pair["l1"]["windows"]),
            )
        rows.append(record)
    report = dict(
        date="2026-10-20", rows=rows,
        conclusion="数学与调度表未变；三档原二进制同卡对照未重现跨轮大幅差异。"
                   "跨轮绝对值不能作为算法收益；原环境中各因素的占比无法事后唯一识别。",
        unchanged_contracts=["NZ2", "cached100 真机等待", "72 核完整 L1 incore 首尾",
                             "静态旧、新采样本来都轮换次序"],
        attribution_limit="当前对照控制了运行时、地址、输入及卡；历史频率/缓存/温度/"
                          "邻卡负载/进程前序不完整，不能将残差唯一称为降频或编译器收益。",
    )
    (HERE / "ANALYSIS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        if row["available"]:
            pair = row.get("same_process_binary_pair", {})
            print(row["history"], row["batch"], row["old_us"], row["new_us"],
                  "配对净省", pair.get("new_saving_us"))


if __name__ == "__main__":
    main()
