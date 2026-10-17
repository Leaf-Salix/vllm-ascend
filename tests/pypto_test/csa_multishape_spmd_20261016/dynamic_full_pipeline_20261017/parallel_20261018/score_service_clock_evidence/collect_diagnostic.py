"""只归档已完成诊断的小型文本与本地大产物路径，不运行算子或计算hash。"""

import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEXT_SUFFIXES = {".py", ".md", ".json", ".log", ".txt", ".sh", ".csv", ".diff"}
LOCAL_DIRECTORIES = {"build_output", "cache", "compile_cache", "__pycache__", "passes_dump"}


def main():
    jobs = json.loads((HERE / "registry.json").read_text())
    assert len(jobs) == 1
    row = jobs[0]
    task = row["stdout"].strip()
    output = Path(row["output"])
    result = json.loads((output / "result.json").read_text())
    assert result["stage"] == "complete" and result["passed"]
    assert result["eligible_for_performance"] is False
    for option, suffix in (("--status", "status.txt"), ("--log", "task.log")):
        process = subprocess.run(["task-submit", option, task], text=True, capture_output=True)
        assert process.returncode == 0, process.stderr
        (HERE / f"{task}.{suffix}").write_text(process.stdout + process.stderr)
    text_files, local_only = [], []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output)
        item = {"path": str(path), "bytes": path.stat().st_size}
        if (any(part in LOCAL_DIRECTORIES for part in relative.parts)
                or path.suffix not in TEXT_SUFFIXES or item["bytes"] > 2_000_000):
            local_only.append(item)
        else:
            text_files.append(item)
    manifest = {"text_files_for_git": text_files, "local_only": local_only,
                "eligible_for_performance": False,
                "local_reason": "二进制、缓存、大快照仅保留本地路径与大小；不计算hash"}
    (HERE / "DEVICE_ARTIFACTS.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"text_files": len(text_files), "local_files": len(local_only)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
