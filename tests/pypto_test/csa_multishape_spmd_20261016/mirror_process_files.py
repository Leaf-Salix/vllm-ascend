# SPDX-License-Identifier: Apache-2.0
"""把本实验过程文本镜像到指定目录；大产物留本地列路径，不做hash、不删除文件。"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil


def mirror(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("源与目标必须独立，不能相互包含")
    suffixes = {".py", ".sh", ".md", ".json", ".csv", ".log", ".txt", ".yaml", ".yml", ".cpp", ".h"}
    skipped = {".git", "__pycache__", "ascend", "build_output", "cache", "compile_cache",
               "kernel", "opp_root", "passes_dump"}
    excluded, local, count, byte_count, copied = [], [], 0, 0, 0
    for parent, directories, files in os.walk(source):
        parent = Path(parent)
        retained = []
        for directory in sorted(directories):
            path = parent / directory
            if directory in skipped or path.is_symlink():
                excluded.append(str(path))
            else:
                retained.append(directory)
        directories[:] = retained
        for name in sorted(files):
            path = parent / name
            if path.is_symlink() or name == "LOCAL_ARTIFACTS.json":
                continue
            stat = path.stat()
            allowed = path.suffix in suffixes or name == ".gitignore"
            if not allowed or stat.st_size > 2 * 1024 * 1024:
                local.append({"path": str(path), "bytes": stat.st_size,
                              "reason": "大于2MiB" if allowed else "非过程文本类型"})
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            prior = target.stat() if target.exists() else None
            if prior is None or (prior.st_mtime_ns, prior.st_size) != (stat.st_mtime_ns, stat.st_size):
                shutil.copy2(path, target)
                copied += 1
            count += 1
            byte_count += stat.st_size
    manifest = {"updated_at": datetime.now().isoformat(timespec="seconds"),
                "source_root": str(source), "mirrored_root": str(destination),
                "text_files": count, "text_bytes": byte_count,
                "excluded_directories": excluded, "local_large_files": local,
                "note": "仅按mtime_ns和大小增量复制过程文本；不做hash、不删除目标文件。大产物与编译缓存留源路径。"}
    (destination / "LOCAL_ARTIFACTS.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {"copied": copied, "text_files": count, "text_mib": round(byte_count / 1048576, 2),
            "local_files": len(local), "excluded_directories": len(excluded)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(mirror(args.source, args.destination), ensure_ascii=False))
