"""只补两个受影响S6目标及S1/S5功能边界；不重跑已完成两代表，无自动重试。"""

import json
import re
import shlex
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAR = HERE.parent
DYNAMIC = PAR.parent


def submit(history, batch, query_tokens, performance):
    name = (f"oa_pair_manual_h{history}_b{batch}_v1" if performance else
            f"oa_pair_manual_h{history}_b{batch}_s{query_tokens}_v1")
    out = PAR / name
    assert not out.exists()
    cmd = ["bash", str(DYNAMIC / ("run_device.sh" if performance else "run_native_query_probe.sh")),
           str(out), "--history", str(history), "--batch", str(batch),
           "--spmd-source", str(HERE / "source_candidate")]
    if performance:
        cmd += ["--source-matrix-reference", str(HERE / "source_reference"), "--wait-backoff", "--l1-windows", "5"]
    else:
        cmd += ["--query-tokens", str(query_tokens), "--causal-extension-reference"]
    args = ["task-submit", "--device", "auto", "--max-time", "1200", shlex.join(cmd)]
    result = subprocess.run(args, text=True, capture_output=True)
    record = dict(history=history, batch=batch, query_tokens=query_tokens, output=str(out),
                  command=args, stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
    assert result.returncode == 0 and re.fullmatch(r"task_\d{8}_\d{6}_\d+\s*", result.stdout), record
    return record


def main():
    registry = HERE / "registry.json"
    jobs = json.loads(registry.read_text())
    assert [(x["history"], x["batch"]) for x in jobs] == [(131072, 4), (8192, 16)]
    assert all(json.loads((Path(x["output"]) / "result.json").read_text())["passed"] for x in jobs)
    boundary_registry = HERE / "boundary_registry.json"
    assert not boundary_registry.exists()
    boundary_registry.write_text("[]\n")
    for batch in (8, 16):
        row = submit(131072, batch, 6, True)
        jobs.append(row)
        registry.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    boundaries = []
    for batch, query in ((16, 1), (7, 5)):
        row = submit(8192, batch, query, False)
        boundaries.append(row)
        boundary_registry.write_text(json.dumps(boundaries, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
