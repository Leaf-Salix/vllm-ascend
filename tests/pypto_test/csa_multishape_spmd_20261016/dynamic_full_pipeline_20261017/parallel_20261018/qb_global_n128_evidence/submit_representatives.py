"""主 Q_B 全局 N128 列任务重分：CPU审阅完成后只提交长短历史代表，不自动重试。"""

import json
import re
import shlex
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAR = HERE.parent
DYNAMIC = PAR.parent
CANDIDATE = HERE / "source_candidate"
REFERENCE = HERE / "source_reference"
CASES = ((131072, 16), (8192, 32))


def main():
    registry = HERE / "registry.json"
    assert not registry.exists(), "拒绝重复提交已登记任务"
    approval = json.loads((HERE / "CPU_REVIEW.json").read_text())
    assert approval["ready_for_representative_device_validation"]
    for source in (CANDIDATE, REFERENCE):
        assert len(list(source.glob("*.py"))) == 33
    jobs = []
    registry.write_text("[]\n")
    for history, batch in CASES:
        output = PAR / f"qb_global_n128_h{history}_b{batch}_v1"
        assert not output.exists()
        cmd = ["bash", str(DYNAMIC / "run_device.sh"), str(output),
               "--history", str(history), "--batch", str(batch),
               "--spmd-source", str(CANDIDATE),
               "--source-matrix-reference", str(REFERENCE),
               "--wait-backoff", "--l1-windows", "5"]
        command = ["task-submit", "--device", "auto", "--max-time", "1200", shlex.join(cmd)]
        result = subprocess.run(command, text=True, capture_output=True)
        row = dict(history=history, batch=batch, output=str(output), command=command,
                   stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        jobs.append(row)
        registry.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        assert result.returncode == 0 and re.fullmatch(r"task_\d{8}_\d{6}_\d+\s*", result.stdout)


if __name__ == "__main__":
    main()
