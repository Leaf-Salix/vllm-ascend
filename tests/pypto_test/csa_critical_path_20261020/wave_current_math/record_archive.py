# SPDX-License-Identifier: Apache-2.0
"""列出本候选可提交的源/文本证据及仅本地保留的产物，不执行Git操作。"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
REPO = CASE.parents[2]
TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".sh", ".md", ".log", ".merge"}


def main():
    selected = set()

    def small_files(directory):
        for path in directory.iterdir():
            if (path.is_file() and path.suffix in TEXT_SUFFIXES
                    and path.name not in {"symbolic.py", "ARCHIVE_MANIFEST.json"}):
                assert path.stat().st_size < 8 * 1024 * 1024, path
                selected.add(path)

    small_files(HERE)
    for name in ("source_candidate", "merge_preview", "merge_preview_v1"):
        small_files(HERE / name)
    local = [HERE / "source_migration_partial_v1"]
    for batch in (16, 28):
        cpu = HERE / f"cpu_b{batch}_v1"
        device = HERE / f"device_b{batch}_v1"
        small_files(cpu)
        small_files(device)
        local.extend([cpu / "symbolic.py", cpu / "kernel", device / "build_output",
                      HERE / f"device_b{batch}_v1_device_logs"])
        matrix = device / "source_matrix"
        selected.add(matrix / "result.json")
        windows = sorted(matrix.glob("l1_*"))
        assert len(windows) == 15, (batch, len(windows))
        for window in windows:
            for name in ("chip_swimlane_records.json", "swimlane.json"):
                path = window / name
                assert path.is_file(), path
                selected.add(path)
        local.append(matrix / "reference_source")
        for directory in (device / "codegen_probe", matrix / "spmd_reference_codegen"):
            if directory.exists():
                selected.update(directory.glob("*.json"))
                local.extend(directory.glob("*.cpp"))
    manifest = {
        "selected_paths": [str(path.relative_to(REPO)) for path in sorted(selected)],
        "file_count": len(selected),
        "bytes": sum(path.stat().st_size for path in selected),
        "manifest_itself_also_selected": True,
        "note": "只列候选文本源及完整30窗；不执行stage/commit，不含二进制/权重，不计算hash",
    }
    (HERE / "ARCHIVE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    inventory_path = CASE / "LOCAL_ARTIFACTS.json"
    inventory = json.loads(inventory_path.read_text())
    inventory_by_path = {row["path"]: row for row in inventory}
    for path in local:
        assert path.exists(), path
        files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        key = str(path.relative_to(REPO))
        inventory_by_path[key] = {
            "path": key, "files": len(files),
            "bytes": sum(p.stat().st_size for p in files), "in_git": False,
            "purpose": "当前数学wave本地编译/重复副本/原日志；源、审计、完整文本结果入Git；不做hash",
        }
    inventory_path.write_text(json.dumps(list(inventory_by_path.values()),
                                        ensure_ascii=False, indent=2) + "\n")
    print("selected", manifest["file_count"], "bytes", manifest["bytes"],
          "local paths", len(local))


if __name__ == "__main__":
    main()
