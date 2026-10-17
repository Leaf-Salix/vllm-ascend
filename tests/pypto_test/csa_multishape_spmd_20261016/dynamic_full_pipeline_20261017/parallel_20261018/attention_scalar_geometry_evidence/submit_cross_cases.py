"""只补两个交叉代表，区分请求数与既有长短策略，不重跑已完成两档。"""

import json
import re
import shlex
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAR = HERE.parent
DYNAMIC = PAR.parent
CASES = ((8192, 16), (131072, 24))


def main():
    registry = HERE / "registry.json"
    jobs = json.loads(registry.read_text())
    assert len(jobs) == 2, "补测入口只允许原两档完成后执行一次"
    for job in jobs:
        result = json.loads((Path(job["output"]) / "result.json").read_text())
        assert result["stage"] == "complete" and result["passed"]
    for history, batch in CASES:
        output = PAR / f"attention_scalar_geometry_h{history}_b{batch}_v1"
        assert not output.exists()
        command = ["bash", str(DYNAMIC / "run_device.sh"), str(output),
                   "--history", str(history), "--batch", str(batch),
                   "--spmd-source", str(HERE / "source_candidate"),
                   "--source-matrix-reference", str(HERE / "source_reference"),
                   "--wait-backoff", "--l1-windows", "5"]
        submit = ["task-submit", "--device", "auto", "--max-time", "1200", shlex.join(command)]
        run = subprocess.run(submit, capture_output=True, text=True)
        row = dict(history=history, batch=batch, output=str(output), command=submit,
                   stdout=run.stdout, stderr=run.stderr, returncode=run.returncode)
        jobs.append(row)
        registry.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        assert run.returncode == 0 and re.fullmatch(r"task_\d{8}_\d{6}_\d+\s*", run.stdout)


if __name__ == "__main__":
    main()
