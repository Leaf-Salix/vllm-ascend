"""以QA消费者及辅助stream的调用顺序匹配Native QA/KV，读取现有七档profile。"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from offline_pd.performance import device_tasks, distribution, require  # noqa: E402

FP_MM = "aclnnMatmulWeightNz_MatMulCommon_MatMulV2"


def main():
    model = json.loads((ROOT / "model/model_gap_rank0.json").read_text())
    workers = json.loads((ROOT / "worker_gap.json").read_text())
    result = {"scope": "Native为真实模型rank0三步63个CSA层；PTO为layer4合成输入单窗口DFX，非同输入A/B。",
              "mapping": "QA是RmsNormDynamicQuant的同stream最近前驱FP matmul；另外两个FP matmul必须在同一辅助stream，"
                         "按_mla_prolog_multistream→cv_indexer_select_qli的源码顺序依次为KV与head权重投影。"
                         "每层要求3个FP matmul、1个RmsNormDynamicQuant；小档trace未单列KV Norm，不用它强行匹配。",
              "limits": "Native整kernel设备时间与PTO单block核内均值范围不同，不直接计算加速比。",
              "cases": []}
    lines = ["# 当前固定规约 Q/KV 与 Native", "", result["scope"], "", result["limits"], "",
             "单位μs。Native第4层列仅3步均值，完整63层样本另保存在JSON。PTO未把seed及调度等待加到matmul列。", "",
             "| 档位 | Native第4层QA | PTO QA block均值/数量 | Native第4层KV | PTO KV block均值/数量 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for case in model["cases"]:
        history, batch = case["history"], case["batch"]
        source = ROOT / "model" / f"h{history}" / f"b{batch}" / "native"
        all_rows = device_tasks(source, 0)
        samples = []
        for interval in case["native"]["intervals"]:
            rows = [r for r in all_rows if interval["start_ns"] <= r["start_ns"]
                    and r["start_ns"] + r["duration_ns"] <= interval["end_ns"]]
            # CANN may report the ACL name or the compiled kernel name.
            mm = [r for r in rows if r["name"] == FP_MM or r["name"].startswith("MatMulV2_NDNZ_")]
            qr = [r for r in rows if r["name"] == "RmsNormDynamicQuant"
                  or r["name"].startswith("RmsNormDynamicQuant_")]
            require(len(mm) == 3 and len(qr) == 1, f"{history}/B{batch}: 调用数量不匹配")
            prior = [r for r in mm if r["stream"] == qr[0]["stream"]
                     and r["start_ns"] + r["duration_ns"] <= qr[0]["start_ns"]]
            require(len(prior) == 1, f"{history}/B{batch}: QA同stream前驱不唯一")
            auxiliary = sorted([r for r in mm if r != prior[0]], key=lambda r: r["start_ns"])
            require(auxiliary[0]["stream"] == auxiliary[1]["stream"] != prior[0]["stream"],
                    f"{history}/B{batch}: 辅助stream调用分工不同")
            matched = {"QA": prior[0], "KV": auxiliary[0]}
            samples.append({"step": interval["step"], "layer": interval["layer"], **matched})
        task = next(c["tasks"] for c in workers["current"] if c["history"] == history and c["batch"] == batch)
        row = {"history": history, "batch": batch, "native_source": str(source), "native_samples": samples,
               "native_all_csa": {name: distribution([s[name]["duration_ns"] / 1000 for s in samples])
                                  for name in ("QA", "KV")},
               "native_layer4": {name: statistics.mean(s[name]["duration_ns"] / 1000
                                                       for s in samples if s["layer"] == 4)
                                 for name in ("QA", "KV")},
               "pto": {name: task[key] for name, key in (("QA", "qr_proj_matmul"), ("KV", "kv_proj_matmul"))}}
        result["cases"].append(row)
        n, p = row["native_layer4"], row["pto"]
        lines.append(f"| {history//1024}K/B{batch} | {n['QA']:.2f} | "
                     f"{p['QA']['kernel_mean_us']:.2f}/{p['QA']['blocks']} | "
                     f"{n['KV']:.2f} | {p['KV']['kernel_mean_us']:.2f}/{p['KV']['blocks']} |")
    (ROOT / "native_projection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines += ["", result["mapping"], "", "当前优先检查固定K条件下的M分工。"
              "M分组减少每核行数，但重复读权重会增加总搬运，必须同时看总核工作量、本体和EP16。"]
    (ROOT / "NATIVE_PROJECTION.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
