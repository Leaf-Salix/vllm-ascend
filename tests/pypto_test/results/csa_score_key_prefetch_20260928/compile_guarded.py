"""利用EP16排队时间编译；任务启动即挂起本脚本创建的整组编译进程。"""

import datetime
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_TASK = "task_20260928_162727_150252329060"
ENVIRONMENT = Path("/data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh")


def model_status():
    try:
        result = subprocess.run(
            ["task-submit", "--status", MODEL_TASK], capture_output=True, text=True, timeout=2
        )
    except subprocess.TimeoutExpired:
        return "unknown: status timeout"
    if result.returncode:
        return "unknown: " + result.stderr.strip()
    return result.stdout.strip()


def compilation_allowed(status):
    # A pending task is watched every 0.5s, before model load/warmup can reach
    # formal timing. Missing/unknown status never allows compilation.
    return status == "pending" or status.startswith("completed (exit=") or status in {"failed", "cancelled", "killed"}


def main():
    process = None
    paused = False
    with (ROOT / "compile_guard.jsonl").open("a") as audit:
        def record(event, **details):
            row = {"time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "event": event, "model_task": MODEL_TASK, **details}
            line = json.dumps(row, ensure_ascii=False)
            audit.write(line + "\n")
            audit.flush()
            print(line, flush=True)

        status = model_status()
        record("observe_before_start", status=status)
        while not compilation_allowed(status):
            time.sleep(5)
            status = model_status()
        with (ROOT / "compile.log").open("w") as output:
            try:
                # Positional parameters are quoted by bash; no shell interpolation
                # of user input, and all descendants inherit our new process group.
                process = subprocess.Popen(
                    ["bash", "-c", 'source "$1"; exec python "$2"', "csa-compile",
                     str(ENVIRONMENT), str(ROOT / "compile.py")],
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                )
                record("compiler_started", pid=process.pid, process_group=process.pid, status=status)
                while process.poll() is None:
                    status = model_status()
                    allowed = compilation_allowed(status)
                    if not allowed and not paused:
                        os.killpg(process.pid, signal.SIGSTOP)
                        paused = True
                        record("compiler_group_stopped", process_group=process.pid, status=status)
                    elif allowed and paused:
                        os.killpg(process.pid, signal.SIGCONT)
                        paused = False
                        record("compiler_group_resumed", process_group=process.pid, status=status)
                    time.sleep(5 if paused else 0.5)
                record("compiler_finished", exit_code=process.returncode, status=model_status())
            except BaseException:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    record("compiler_group_killed_after_monitor_error", process_group=process.pid)
                raise
        return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
