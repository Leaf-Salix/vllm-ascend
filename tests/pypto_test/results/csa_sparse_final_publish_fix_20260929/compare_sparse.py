"""固定输入上区分融合失败与统计量抽取修复，保留每组 head 的误差。"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor  # noqa: E402


def main():
    torch.set_num_threads(4)
    expected = torch.load(ROOT / "sparse/baseline/output.pt", map_location="cpu", weights_only=True)
    reports = {}
    devices = set()
    for side in ("baseline", "broken", "candidate"):
        report = json.loads((ROOT / f"sparse/{side}/report.json").read_text())
        if (report["batch"], report["history"]) != (16, 8192):
            raise ValueError("固定输入必须只取 B16，未恢复退役 B40 档")
        devices.add(report["device"])
        actual = torch.load(ROOT / f"sparse/{side}/output.pt", map_location="cpu", weights_only=True)
        reports[side] = {
            "comparison": compare_tensor(actual, expected, 0, 0),
            "head_groups": [compare_tensor(actual[:, h:h+16], expected[:, h:h+16], 0, 0)
                            for h in range(0, 64, 16)],
            "native_reference": report["comparison"],
        }
    if len(devices) != 1:
        raise ValueError("必须同一个队列任务同卡执行")
    status = reports["candidate"]["comparison"]["status"]
    result = {"status": status, "device": devices.pop(), "reports": reports,
              "scope": "固定历史 Q/cache/TopK 的前 B16 个序列，cos=1/sin=0；独立 Sparse，非完整 CSA/模型验收"}
    (ROOT / "sparse_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print({"status": status, "comparisons": {k: v["comparison"] for k, v in reports.items()}})
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
