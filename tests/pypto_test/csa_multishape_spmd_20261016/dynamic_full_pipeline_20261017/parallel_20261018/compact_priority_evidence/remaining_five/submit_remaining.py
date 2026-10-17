"""仅提交已授权的五个缺失验收点；每档一次，不自动重试。"""

import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


E = Path(__file__).resolve().parent
C = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1")
D = C / "tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
PAR = D / "parallel_20261018"
SOURCE = Path("/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/ultra_compact_priority_20261018/source_snapshot")
REGISTRY = PAR / "compact_priority_remaining_jobs.json"
CASES = [(131072, 4), (131072, 8), (131072, 24), (131072, 28), (8192, 16)]


def main():
    assert SOURCE.is_dir() and (SOURCE / "entry.py").is_file()
    assert not REGISTRY.exists(), "拒绝重复提交：任务登记文件已存在"
    for history, batch in CASES:
        assert not (PAR / f"compact_priority_h{history}_b{batch}_v1").exists()
    registry = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_source": str(SOURCE),
        "reference_source": str(PAR / "baseline_source"),
        "scope": "补齐阶段前移同源码七档的五个缺失点；每档同进程原task/593/候选五窗完整L1，不自动重测。",
        "completed_representatives_not_resubmitted": [[131072, 16], [8192, 32]],
        "jobs": [],
    }
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    for history, batch in CASES:
        out = PAR / f"compact_priority_h{history}_b{batch}_v1"
        command = shlex.join(["bash", str(D / "run_device.sh"), str(out),
                              "--history", str(history), "--batch", str(batch),
                              "--spmd-source", str(SOURCE), "--source-matrix-reference", str(PAR / "baseline_source"),
                              "--wait-backoff", "--l1-windows", "5"])
        args = ["task-submit", "--device", "auto", "--max-time", "1200", command]
        result = subprocess.run(args, cwd=C, text=True, capture_output=True)
        log = E / f"submit_h{history}_b{batch}.log"
        log.write_text(result.stdout + result.stderr)
        ids = sorted(set(re.findall(r"task_\d{8}_\d{6}_\d+", result.stdout + result.stderr)))
        row = {"history": history, "batch": batch, "output": str(out), "command": command,
               "submit_arguments": args[:-1], "submit_returncode": result.returncode,
               "submit_log": str(log), "task_ids_in_response": ids}
        if len(ids) == 1:
            row["task_id"] = ids[0]
        registry["jobs"].append(row)
        REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        assert result.returncode == 0 and len(ids) == 1, "提交异常已记录，停止后续提交，不自动重试"


if __name__ == "__main__":
    main()
