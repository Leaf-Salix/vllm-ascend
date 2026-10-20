# SPDX-License-Identifier: Apache-2.0
"""归档本轮完整算子、组件补丁、失败记录与配对原始窗；大产物只登记路径。"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
REPO = CASE.parents[2]
TEXT = {".py", ".json", ".jsonl", ".sh", ".md", ".log", ".txt", ".xml", ".patch"}


def main():
    selected = set()
    local = []

    def roots(directory):
        for path in directory.iterdir():
            if path.is_file() and path.suffix in TEXT and path.name not in {"symbolic.py", "ARCHIVE_MANIFEST.json"}:
                assert path.stat().st_size < 8 * 1024 * 1024, path
                selected.add(path)

    roots(HERE)
    roots(HERE / "source_candidate")
    for snapshot in sorted(HERE.glob("pypto_source_v*")):
        for path in snapshot.rglob("*"):
            if path.is_file():
                assert path.suffix in {".py", ".cpp", ".h", ".md"}, path
                selected.add(path)
    cpu_folders = sorted(p for p in HERE.glob("cpu*") if p.is_dir())
    cpu_folders.append(HERE / "inline_dim_parent_v1")
    for cpu in cpu_folders:
        roots(cpu)
        local.extend(p for p in (cpu / "kernel", cpu / "symbolic.py") if p.exists())
    for kind, job in json.loads((HERE / "TASKS.json").read_text()).items():
        device = HERE / job["folder"]
        roots(device)
        local.extend(p for p in (
            device / "build_output", device / "source", device / "kernel",
            HERE / (job["folder"] + "_device_logs"),
        ) if p.exists())
        if kind == "probe":
            continue
        matrix = device / "source_matrix"
        selected.add(matrix / "result.json")
        windows = list(matrix.glob("l1_*"))
        assert len(windows) == 15
        for window in windows:
            for name in ("chip_swimlane_records.json", "swimlane.json"):
                assert (window / name).is_file()
                selected.add(window / name)
        if (matrix / "reference_source").exists():
            local.append(matrix / "reference_source")
        for folder in (device / "codegen_probe", matrix / "spmd_reference_codegen"):
            if folder.exists():
                selected.update(folder.glob("*.json"))
                local.extend(folder.glob("*.cpp"))
    manifest = {
        "selected_paths": [str(p.relative_to(REPO)) for p in sorted(selected)],
        "files": len(selected), "bytes": sum(p.stat().st_size for p in selected),
        "note": "完整37源、组件14文件补丁/快照、失败尝试及原始五窗入Git；权重/二进制/大编译产物本地保留，不做hash",
    }
    (HERE / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    path = CASE / "LOCAL_ARTIFACTS.json"
    inventory = {row["path"]: row for row in json.loads(path.read_text())}
    for artifact in local:
        files = [artifact] if artifact.is_file() else [p for p in artifact.rglob("*") if p.is_file()]
        key = str(artifact.relative_to(REPO))
        inventory[key] = {
            "path": key, "files": len(files), "bytes": sum(p.stat().st_size for p in files),
            "in_git": False, "purpose": "Key多ND的大编译产物/重复副本；源和必要原始记录另选入Git；不做hash",
        }
    path.write_text(json.dumps(list(inventory.values()), ensure_ascii=False, indent=2) + "\n")
    print("selected", len(selected), "bytes", manifest["bytes"], "local paths", len(local))


if __name__ == "__main__":
    main()
