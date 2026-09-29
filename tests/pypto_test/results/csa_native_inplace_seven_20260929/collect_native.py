"""Native七档完成即可发布，不等待PTO及核内诊断阶段；不触发设备工作。"""

import json
import subprocess
from pathlib import Path

from collect import CASES, ROOT, load, native_scope, read


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    # The matrix runs the final Native phase before PTO under set -e.
    # A started PTO phase proves the final Native process returned successfully.
    next_phase = ROOT / "h8192_b32/pto/run.log"
    if status != "completed (exit=0)" and not next_phase.exists():
        raise RuntimeError("最后Native子进程尚未成功返回，继续等待同一任务")
    source = read(ROOT / "source.json")
    pair = load("native_only_pair", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    result = {"task": task, "matrix_status_at_collection": status,
              "native_status": "COMPLETE", "source": source,
              "completion_evidence": str(next_phase), "cases": [],
              "scope": "Native单卡HC_pre+norm+CSA+HC_post；不代表EP16或整模型"}
    for history, batch in CASES:
        folder = ROOT / f"h{history}_b{batch}/native"
        report = read(folder / "report.json")
        if (report["source"], report["side"], report["history"], report["batch"]) != (
                source["source"], "native", history, batch):
            raise ValueError("Native来源或档位不一致")
        side = pair.analyze_side(folder)
        scope = native_scope(report, Path(side["profile_csv"]))
        if report["timing"]["topk_selection"]["structural_errors"]:
            raise ValueError("Native Top-K结构错误")
        result["cases"].append({"history": history, "batch": batch,
                                **{k: side[k] for k in ("device", "cann", "requested", "mean_us", "us_p50",
                                                       "us_p95", "us_max", "guards", "profile_json")},
                                "samples_us": report["timing"]["samples_us"],
                                "compile_options": report["torch_compile_options"],
                                "backend_options": scope["backend_options"],
                                "measurement": scope["measurement"],
                                "state_status": "PASS", "state_count": len(side["states"]),
                                "profile_streams": scope["streams"],
                                "report": str(folder / "report.json")})
    if len({c["device"] for c in result["cases"]}) != 1:
        raise ValueError("Native七档不在同一设备")
    (ROOT / "native_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 最新标准Native七档基线", "",
             "Native七档已完成；不等待PTO配套采集。CANN9.2、mode2、det0、EPLB关闭。",
             "显式torch.compile backend=npugraph_ex、dynamic=False/fullgraph=True/inplace_pass=True；",
             "static_kernel_compile和SuperKernel开启，多流由后端图捕获。",
             "固定shape/地址，使用后端创建的唯一NPUGraph直接replay，无主机更新节点。",
             "正式第二个CSA层真实权重、独立合成历史；完整HC_pre+norm+CSA+HC_post，单位μs。",
             "5次预热后20次设备事件，profile独立采集；不含编译/初始化，不是模型forward。", "",
             "| 档位 | Native均值 | P50 | P95 | 最大值 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for c in result["cases"]:
        lines.append(f"| {c['history']//1024}K/B{c['batch']} | {c['mean_us']:.3f} | {c['us_p50']:.3f} "
                     f"| {c['us_p95']:.3f} | {c['us_max']:.3f} |")
    lines += ["", "静态编译安装、实际SuperKernel/多流、八类同图状态及保护区均检查通过。",
              "这是Native自身图重放检查，不是Native/PTO跨实现精度或逐token/DSpark验收。",
              "PTO使用现有已验证路径和数据，不要求复刻Native编译配置；新HC_post收益不推算进旧版本读数。",
              "[原样本、实际配置与profile](native_summary.json)。"]
    (ROOT / "NATIVE_RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
