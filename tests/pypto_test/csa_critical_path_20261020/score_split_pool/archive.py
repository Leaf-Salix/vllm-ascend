# SPDX-License-Identifier: Apache-2.0
"""WS列池本轮源、完整文本证据及本地大产物清单，不执行Git操作。"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
REPO = CASE.parents[2]
TEXT = {".py", ".json", ".jsonl", ".sh", ".md", ".log"}


def main():
    selected = {CASE / "prepare_split_pool.py"}
    local = []

    def roots(directory):
        for path in directory.iterdir():
            if (path.is_file() and path.suffix in TEXT
                    and path.name not in {"symbolic.py", "ARCHIVE_MANIFEST.json"}):
                assert path.stat().st_size < 8 * 1024 * 1024, path
                selected.add(path)

    roots(HERE)
    roots(HERE / "source_candidate")
    for batch in (16, 28):
        cpu = HERE / f"cpu_b{batch}_v1"
        roots(cpu)
        local.extend([cpu / "kernel", cpu / "symbolic.py"])
        device = HERE / f"device_b{batch}_v1"
        roots(device)
        local.extend([device / "build_output", device / "source",
                      HERE / f"device_b{batch}_v1_device_logs"])
        matrix = device / "source_matrix"
        selected.add(matrix / "result.json")
        windows = list(matrix.glob("l1_*"))
        assert len(windows) == 15
        for window in windows:
            for name in ("chip_swimlane_records.json", "swimlane.json"):
                assert (window / name).is_file()
                selected.add(window / name)
        local.append(matrix / "reference_source")
        for folder in (device / "codegen_probe", matrix / "spmd_reference_codegen"):
            selected.update(folder.glob("*.json"))
            local.extend(folder.glob("*.cpp"))
    manifest = {"selected_paths": [str(p.relative_to(REPO)) for p in sorted(selected)],
                "files": len(selected), "bytes": sum(p.stat().st_size for p in selected),
                "note": "完整源及三方30窗入Git；不提交权重/二进制/大编译快照，不做hash"}
    (HERE / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    path = CASE / "LOCAL_ARTIFACTS.json"
    inventory = {row["path"]: row for row in json.loads(path.read_text())}
    for artifact in local:
        assert artifact.exists()
        files = [artifact] if artifact.is_file() else [p for p in artifact.rglob("*") if p.is_file()]
        key = str(artifact.relative_to(REPO))
        inventory[key] = {"path": key, "files": len(files),
                          "bytes": sum(p.stat().st_size for p in files), "in_git": False,
                          "purpose": "当前双QK列池的大编译/重复副本；完整源及轻量证据入Git；不做hash"}
    path.write_text(json.dumps(list(inventory.values()), ensure_ascii=False, indent=2) + "\n")
    print("selected", len(selected), "bytes", manifest["bytes"], "local paths", len(local))


if __name__ == "__main__":
    main()
