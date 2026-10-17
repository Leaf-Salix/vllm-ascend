"""只收取已登记且完成的任务日志与本地大产物清单；不提交、不重测、不做hash。"""

import argparse
import json
import subprocess
from pathlib import Path


TEXT_SUFFIXES = {".py", ".md", ".json", ".log", ".txt", ".sh", ".csv", ".pto", ".cpp", ".hpp", ".h", ".patch", ".diff", ".stdout"}
LOCAL_DIRECTORIES = {"build_output", "cache", "__pycache__", "passes_dump"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=("qb_b_resident", "qb_qr_b_resident", "oa_pair_manual", "oa_pair_small_t", "idx_had_owner", "idx_query_quant_owner"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    evidence = root / f"{args.candidate}_evidence"
    jobs = json.loads((evidence / "registry.json").read_text())
    all_local, text_files = [], []
    for job in jobs:
        task = job["stdout"].strip()
        directory = Path(job["output"])
        result = json.loads((directory / "result.json").read_text())
        assert result["stage"] == "complete" and result["passed"], "未完成或未通过，拒绝写完成清单"
        for option, suffix in (("--status", "status.txt"), ("--log", "task.log")):
            process = subprocess.run(["task-submit", option, task], text=True, capture_output=True)
            assert process.returncode == 0, process.stderr
            (evidence / f"{task}.{suffix}").write_text(process.stdout + process.stderr)
        for file in sorted(directory.rglob("*")):
            if not file.is_file():
                continue
            relative = file.relative_to(directory)
            item = {"path": str(file), "bytes": file.stat().st_size}
            if (any(part in LOCAL_DIRECTORIES for part in relative.parts)
                    or file.suffix not in TEXT_SUFFIXES or file.stat().st_size > 2_000_000):
                all_local.append(item)
            else:
                text_files.append(item)
    (evidence / "DEVICE_ARTIFACTS.json").write_text(json.dumps(
        {"text_files_for_git": text_files, "local_only": all_local,
         "local_reason": "编译产物/缓存/数值快照/超过2MB文件保留本地，只有路径及大小；不计算hash"},
        ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate": args.candidate, "completed_jobs": len(jobs),
                      "text_files": len(text_files), "local_only_files": len(all_local)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
