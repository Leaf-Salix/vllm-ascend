#!/usr/bin/env python3
"""完整采集超时后复用已验证binary重提，不重编、不缩核、不更改输入。

本轮增加CANN dump落盘，采用分钟/秒一致的分层时限。CANN原始命令沿用
skill记录的symbol/app/SoC，结果再交由原版skill的artifact检查函数核验。
启动与完成分开，立即保存队列ID，避免漫长采集期间丢失任务关联。
"""

import argparse
import csv
import importlib.util
import json
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from prepare_standalone import HERE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    registry = HERE / "PROFILE_SUBMISSION_V3.json"
    if not args.finalize:
        assert not registry.exists(), "已有v3登记，不能重复提交"
        case = Path((HERE / "ACTIVE_CASE_PATH.txt").read_text().strip())
        old = case.parents[2]
        entry = list(csv.DictReader((old / "manifest_export.csv").open()))[0]
        assert entry["status"] == "collect_failed"
        # 必须先取得旧任务终态，不能把waiter退出误认作任务结束后重提。
        old_status = subprocess.check_output(["task-submit", "--status", "task_20261019_001135_28286531625"],text=True).strip()
        assert old_status == "completed (exit=143)", old_status
        run = HERE / "build_output" / ("full_csa_all72_retry_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        target_case = run / "cases/ptoas/dynamic_csa_full_msprof"
        target_case.mkdir(parents=True)
        for path in case.iterdir():
            if path.is_file() and (path.suffix in (".cpp", ".json", ".txt") or path.name == "CMakeLists.txt"):
                shutil.copy2(path, target_case / path.name)
        collect = run / "funcs/dynamic_csa_full/collect"
        collect.mkdir(parents=True)
        old_script = Path(entry["collect_dir"]) / "collect_queued.sh"
        script = old_script.read_text().splitlines()
        lines = []
        for line in script:
            if line.startswith("exec "):
                command = shlex.split(line[5:])
                command = ["--timeout=90" if arg.startswith("--timeout=") else
                           "--output=" + str(collect / "out") if arg.startswith("--output=") else arg for arg in command]
                command.append("--dump=on")
                lines.append("exec " + shlex.join(command))
            elif line.startswith("cd "):
                lines.append("cd " + shlex.quote(str(target_case)))
            else:
                lines.append(line)
        (collect / "collect_queued.sh").write_text("\n".join(lines)+"\n")
        command = ["task-submit", "--no-device", "--max-time", "7200", "bash " + shlex.quote(str(collect / "collect_queued.sh"))]
        submitted = subprocess.run(command,check=True,text=True,capture_output=True)
        task = submitted.stdout.strip()
        assert task.startswith("task_"), submitted.stdout
        entry.update(status="running", case_dir=str(target_case), collect_dir=str(collect), export_dir=str(collect / "out"),
                     export_src="", artifact_count=0, trace_json="", visualize_data_bin="", core_trace_count=0,
                     instr_csv_count=0, message="复用已真机完整验证的原binary；72核、dump=on、90分钟应用时限/7200秒队列时限")
        record = dict(task_id=task, submitted_time=time.time(), run_root=str(run), original_run=str(old),
                      command=command, entry=entry, mathematics_changed=False, rebuilt=False, state="running")
        registry.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
        (HERE / "ACTIVE_CASE_PATH.txt").write_text(str(target_case)+"\n")
        print(task)
        return 0
    record = json.loads(registry.read_text())
    status = subprocess.check_output(["task-submit", "--status", record["task_id"]],text=True).strip()
    if status != "completed (exit=0)":
        print(status)
        return 1
    log = subprocess.check_output(["task-submit", "--log", record["task_id"]],text=True)
    entry = record["entry"]
    (Path(entry["collect_dir"]) / "collect.log").write_text(log)
    spec = importlib.util.spec_from_file_location("original_skill_profile", HERE / "skill_snapshot/.claude/skills/incore-profiling/incore_profile.py")
    profiler = importlib.util.module_from_spec(spec); spec.loader.exec_module(profiler)
    artifacts = profiler.collect_artifacts(Path(entry["export_dir"]))
    entry.update(artifacts)
    assert artifacts["core_trace_count"] == 72 and artifacts["instr_csv_count"] == 72, artifacts
    assert artifacts["trace_json"] and artifacts["visualize_data_bin"]
    assert profiler.detect_degenerate_trace(Path(entry["export_dir"])) is None
    entry.update(status="exported", export_src=entry["export_dir"], duration_sec=time.time()-record["submitted_time"],
                 message="复用已验证binary，完整72核导出；模拟输出精度另由validate_replay检查")
    run = Path(record["run_root"])
    with (run / "manifest_export.csv").open("w") as output:
        writer = csv.DictWriter(output,fieldnames=list(entry)); writer.writeheader(); writer.writerow(entry)
    (run / "summary.txt").write_text(f"task={record['task_id']}\nexported=1/1\ncore_trace=72\ninstr_csv=72\nrun_root={run}\n")
    record.update(state="completed", terminal_status=status)
    registry.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
    print(str(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
