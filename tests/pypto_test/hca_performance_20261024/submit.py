# SPDX-License-Identifier: Apache-2.0
"""显式提交选定档位并记录唯一task id；不重提已有任务。"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "hca_tnd_mpmd_20261022"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("side", choices=["native", "pto", "cpu"])
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--revision", type=int, default=1)
    args = parser.parse_args()
    path = HERE / "tasks.json"
    tasks = json.loads(path.read_text()) if path.exists() else []
    cases = json.loads((HERE / "cases.json").read_text())["cases"]
    selected = [c for c in cases if not args.cases or c["id"] in args.cases]
    if args.side == "cpu":
        selected = [{"id": "compile"}]
    for case in selected:
        key = f"{args.side}_{case['id']}_v{args.revision}"
        if any(row["key"] == key for row in tasks):
            print(f"已提交，跳过：{key}", flush=True)
            continue
        output = HERE / "runs" / key
        if args.side == "native":
            cmd = ["bash", str(HERE / "launch_native.sh"), str(output), "--case", case["name"],
                   "--batch", str(case["active_requests"]), "--history", str(case["history_label"])]
        else:
            cmd = ["bash", str(OLD / "launch.sh"), "--experiment",
                   str(OLD / "experiments/performance_spec_20261024"), "--output", str(output)]
            if args.side == "cpu":
                cmd += ["--cpu"]
            else:
                cmd += ["--control-source", str(OLD / "experiments/attention_pair_control_v1/operator"),
                        "--performance-spec", "--diagnostic-perf", "--windows", "10",
                        "--batch", str(case["active_requests"]), "--history", str(case["history_label"]),
                        "--queries", ",".join(map(str, case["query_lengths"]))]
        task = subprocess.run(["task-submit", *(["--no-device"] if args.side == "cpu" else ["--device", "auto"]),
                               "--max-time", "3600", shlex.join(cmd)], text=True, capture_output=True, check=True)
        match = re.search(r"task_\d+_\d+_\d+", task.stdout)
        if match is None:
            raise RuntimeError(task.stdout + task.stderr)
        tasks.append({"key": key, "case_id": case["id"], "side": args.side,
                      "task_id": match.group(), "command": cmd, "output": str(output)})
        path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
        print(key, match.group(), flush=True)


if __name__ == "__main__":
    main()
