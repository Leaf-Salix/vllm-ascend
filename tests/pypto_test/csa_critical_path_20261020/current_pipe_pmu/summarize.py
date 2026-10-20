# SPDX-License-Identifier: Apache-2.0
"""只汇总已结束的当前正式PMU数据，保留全样本，禁止替代完整L1验收。"""

import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    jobs = json.loads((HERE / "TASKS.json").read_text())
    result = {}
    for batch, job in jobs.items():
        status = subprocess.check_output(["task-submit", "--status", job["task"]], text=True).strip()
        assert status == "completed (exit=0)", status
        folder = HERE / job["folder"]
        root = json.loads((folder / "result.json").read_text())
        data = json.loads((folder / "pipe_profile/result.json").read_text())
        assert root["passed"] and root["full_csa_validated"] and root["hardware_profile_collected"]
        assert not root["performance_measured"]
        assert data["passed"] and data["same_process"] and data["same_root_addresses"]
        assert root["compiler_comparison"]["same_operator_source"]
        assert len(data["samples"]) == 6
        differences = {}
        for name, summary in data["summary"].items():
            for key, values in summary.items():
                assert len(values["samples"]) == 3
                assert all(math.isfinite(value) for value in values["samples"])
        for key, value in data["summary"]["spmd_current"].items():
            differences[key] = value["median"] - data["summary"]["dynamic_pto"][key]["median"]
        result[batch] = {
            "task": job["task"], "status": status, "passed": True,
            "performance_eligible": False, "reference": "13259f92/source_after_score_split",
            "summary": data["summary"], "current_minus_multitask": differences,
            "root_result": str(folder / "result.json"),
            "raw_result": str(folder / "pipe_profile/result.json"),
            "scope": "全72角色聚合外部PMU，三步轮转；不更新L1性能表",
        }
        job["terminal_status"] = status
        print(batch, "AIC MTE1/MTE2",
              data["summary"]["spmd_current"]["aic_mte1_time(us)"]["median"],
              data["summary"]["spmd_current"]["aic_mte2_time(us)"]["median"])
    (HERE / "PIPE_COUNTERS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (HERE / "TASKS.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
