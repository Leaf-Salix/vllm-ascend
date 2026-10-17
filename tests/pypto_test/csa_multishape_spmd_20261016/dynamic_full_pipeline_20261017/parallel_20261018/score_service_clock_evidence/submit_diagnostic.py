"""独立审阅完成后，仅提交一次正式668长B16的Score服务诊断，不自动重试。"""

import json
import re
import shlex
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    registry = HERE / "registry.json"
    assert not registry.exists(), "已登记任务，不得重复提交"
    review = json.loads((HERE / "STATIC_REVIEW.json").read_text())
    assert review["ready"]
    output = HERE.parent / "score_service_h131072_b16_v1"
    assert not output.exists(), "输出必须为新目录"
    command = ["bash", str(HERE / "run_device.sh"), str(output), "--baseline-only"]
    submit = ["task-submit", "--device", "auto", "--max-time", "1200", shlex.join(command)]
    process = subprocess.run(submit, capture_output=True, text=True)
    row = dict(history=131072, batch=16, query_tokens=6, output=str(output),
               command=submit, stdout=process.stdout, stderr=process.stderr,
               returncode=process.returncode, eligible_for_performance=False)
    registry.write_text(json.dumps([row], ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)
    assert process.returncode == 0 and re.fullmatch(r"task_\d{8}_\d{6}_\d+\s*", process.stdout)


if __name__ == "__main__":
    main()
