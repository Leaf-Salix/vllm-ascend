# SPDX-License-Identifier: Apache-2.0
"""归档本会话现存证据；不执行测试，不计算 hash，不删除旧记录。"""

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

TEXT_SUFFIXES = {".py", ".cpp", ".h", ".md", ".json", ".log", ".txt", ".csv", ".sh", ".patch", ".diff"}
LARGE_BYTES = 8 * 1024 * 1024


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def copy_text(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # 设备日志原件可能只读；只开放归档副本的 owner 写权限以便刷新。
        target.chmod(target.stat().st_mode | 0o200)
    shutil.copy2(source, target)


def git_snapshot(repo, output, committed=False):
    output.mkdir(parents=True, exist_ok=True)
    commands = {
        "branch.txt": ["branch", "--show-current"],
        "base.txt": ["log", "-1", "--format=fuller"],
        "status.txt": ["status", "--short"],
        "changes.patch": ["show", "--format=fuller", "HEAD"] if committed else ["diff", "HEAD"],
    }
    for name, command in commands.items():
        with (output / name).open("w") as stream:
            subprocess.run(["git", "-C", str(repo), *command], stdout=stream, check=True)


def summary(data):
    if not isinstance(data, dict):
        return "结构化明细，见 JSON"
    keys = ("passed", "status", "full_codegen_passed", "binary_compile_passed", "callable_assembled",
            "parse_passed", "allocation_passed",
            "kernel_runtime_validated", "device_execution")
    flags = [f"{key}={data[key]}" for key in keys if key in data]
    error = str(data.get("error", "")).splitlines()
    if error:
        flags.append(error[0][:220])
    return "; ".join(flags) or "结果明细见 JSON，不推断整体 PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--pypto", type=Path, required=True)
    parser.add_argument("--simpler", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    results = source.parent / "results" / source.name
    destination = args.destination.resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("归档位置必须与工作目录互不包含")
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    logs = source / "session_logs"
    logs.mkdir(exist_ok=True)
    for pattern in ("csa-*.log", "pypto-csa-*.log", "simpler-core*.log", "csa-*.sh"):
        for path in Path("/tmp").glob(pattern):
            if path.is_file():
                copy_text(path, logs / path.name)
    upstream = source / "upstream_records"
    upstream.mkdir(exist_ok=True)
    for name in ("csa-pypto-issue-3009-current", "csa-pypto-pr-current", "csa-simpler-pr-current"):
        path = Path("/tmp") / (name + ".json")
        if path.exists():
            copy_text(path, upstream / path.name)
    git_snapshot(args.pypto, source / "component_changes" / "pypto")
    git_snapshot(args.simpler, source / "component_changes" / "simpler", committed=True)

    validation = (source.parent / "DSV4_FLASH_CSA_VALIDATION_LOG.md").read_text()
    marker = "## 603. "
    excerpt = validation[validation.index(marker):]
    excerpt = excerpt.replace("](" + source.name + "/", "](")
    excerpt = excerpt.replace("](results/", "](" + str(source.parent / "results") + "/")
    (source / "validation_log_excerpt.md").write_text(
        "以下为主验证日志第 603 节起的记录，仅调整相对链接以适应归档位置。\n\n" + excerpt
    )

    evidence = source / "evidence"
    evidence.mkdir(exist_ok=True)
    inventory = []
    for path in sorted(results.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(results)
        stat = path.stat()
        keep = path.suffix in TEXT_SUFFIXES and stat.st_size <= LARGE_BYTES
        if any(part in {"opp_root", "passes_dump", "__pycache__", ".cache", "runtime"}
               for part in relative.parts):
            keep = False
        if keep:
            # 二进制副产物不能仅因扩展名像文本而被归档。
            with path.open("rb") as stream:
                keep = b"\0" not in stream.read(4096)
        target = evidence / relative
        if keep:
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_text(path, target)
        inventory.append({"source": str(path), "relative": str(relative), "bytes": stat.st_size,
                          "modified_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
                          "retention": "text_copy" if keep else "original_local",
                          "archive": str(Path("evidence") / relative) if keep else None})
    save_json(source / "artifact_inventory.json",
              {"updated_at_utc": now, "original_results": str(results), "artifacts": inventory,
               "note": "只记录路径、大小和时间；未计算内容 hash。大产物与设备二进制本地保留。"})

    attempts = []
    for directory in sorted(results.iterdir(), key=lambda path: path.stat().st_mtime):
        if not directory.is_dir():
            continue
        result_files = [path for path in directory.rglob("*result*.json")
                        if "opp_root" not in path.parts]
        entries = []
        for path in sorted(result_files):
            try:
                data = json.loads(path.read_text())
                outcome = summary(data)
            except (ValueError, OSError) as exc:
                outcome = f"记录读取失败：{exc}"
            entries.append({"path": str(path), "summary": outcome})
        attempts.append({"attempt": directory.name, "original_directory": str(directory),
                         "results": entries, "missing_final_result": not entries})
    save_json(source / "attempts.json", {"updated_at_utc": now, "attempts": attempts})
    lines = ["# CSA 单 SPMD：逐次尝试清单", "",
             "按现存结果目录的时间顺序归档；目录时间不是精确任务开始时间。",
             "没有最终 JSON 的目录明确标记，不能据目录存在推断任务执行或通过。",
             "各条完整错误、源码快照和日志见 evidence；大 IR/二进制路径见 artifact_inventory.json。",
             "CPU 单测、构建、无效试验另见 session_logs 和 PROCESS_LOG.md。", "",
             "| 尝试 | 实际记录 |", "| --- | --- |"]
    for attempt in attempts:
        notes = []
        for record in attempt["results"]:
            path = Path(record["path"])
            link = Path("evidence") / path.relative_to(results)
            notes.append(f"[{path.name}]({link}): " + record["summary"].replace("|", "\\|"))
        outcome = "<br>".join(notes or ["没有最终 result；查原目录与日志，未记 PASS"])
        lines.append(f"| {attempt['attempt']} | {outcome} |")
    (source / "ALL_ATTEMPTS.md").write_text("\n".join(lines) + "\n")

    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_text(path, target)
        copied.append(str(relative))
    record = {"updated_at_utc": now, "source_worktree_directory": str(source),
              "destination": str(destination), "results_directory": str(results),
              "attempt_directories": len(attempts), "indexed_artifacts": len(inventory),
              "copied_text_files": len(copied), "files": copied,
              "pr_created_by_archive": False, "tests_run_by_archive": False}
    save_json(destination / "ARCHIVE_STATE.json", record)
    save_json(source / "ARCHIVE_STATE.json", record)
    print(json.dumps({key: value for key, value in record.items() if key != "files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
