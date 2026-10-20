# SPDX-License-Identifier: Apache-2.0
"""保存联合短尾的完整源/失败修正/原始配对证据，大编译产物只登记。"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
REPO = CASE.parents[2]
TEXT = {".py", ".json", ".jsonl", ".sh", ".md", ".log", ".txt"}


def main():
    selected = {CASE / "prepare_score_joint_tail.py"}
    local = []

    def roots(directory):
        for path in directory.iterdir():
            if path.is_file() and path.suffix in TEXT and path.name not in {"symbolic.py", "ARCHIVE_MANIFEST.json"}:
                assert path.stat().st_size < 8 * 1024 * 1024, path
                selected.add(path)

    roots(HERE)
    roots(HERE / "source_candidate")
    attempts = {}
    for directory in sorted(HERE.glob("source_candidate_v*")):
        changed = []
        for path in directory.glob("*.py"):
            if path.read_bytes() != (HERE / "source_candidate" / path.name).read_bytes():
                changed.append(path.name)
                selected.add(path)
        attempts[directory.name] = {"base": "source_candidate", "overrides": sorted(changed),
                                    "reproduce": "完整当前37源复制到新目录，再覆盖本版列出的源文件；未列文件逐字相同"}
        local.append(directory)
    attempt_file = HERE / "SOURCE_ATTEMPTS.json"
    attempt_file.write_text(json.dumps(attempts, ensure_ascii=False, indent=2) + "\n")
    selected.add(attempt_file)
    for cpu in sorted(HERE.glob("cpu_b*_v*")):
        if not cpu.is_dir():
            continue
        roots(cpu)
        local.extend(p for p in (cpu / "kernel", cpu / "symbolic.py") if p.exists())
    for job in json.loads((HERE / "TASKS.json").read_text()).values():
        device = HERE / job["folder"]
        roots(device)
        local.extend(p for p in (device / "build_output", device / "source", HERE / (job["folder"] + "_device_logs")) if p.exists())
        matrix = device / "source_matrix"
        if matrix.exists():
            selected.add(matrix / "result.json")
            windows = list(matrix.glob("l1_*"))
            assert len(windows) == 15
            for window in windows:
                for name in ("chip_swimlane_records.json", "swimlane.json"):
                    assert (window / name).is_file()
                    selected.add(window / name)
            local.append(matrix / "reference_source")
        for folder in (device / "codegen_probe", matrix / "spmd_reference_codegen"):
            if folder.exists():
                selected.update(folder.glob("*.json"))
                local.extend(folder.glob("*.cpp"))
    manifest = {"selected_paths": [str(p.relative_to(REPO)) for p in sorted(selected)],
                "files": len(selected), "bytes": sum(p.stat().st_size for p in selected),
                "note": "完整源、历次差异源及原始配对窗入Git；不提交权重/二进制/大编译快照，不做hash"}
    (HERE / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    path = CASE / "LOCAL_ARTIFACTS.json"
    inventory = {row["path"]: row for row in json.loads(path.read_text())}
    for artifact in local:
        assert artifact.exists()
        files = [artifact] if artifact.is_file() else [p for p in artifact.rglob("*") if p.is_file()]
        key = str(artifact.relative_to(REPO))
        inventory[key] = {"path": key, "files": len(files), "bytes": sum(p.stat().st_size for p in files),
                          "in_git": False, "purpose": "联合短尾的大编译/重复副本；完整源及必要差异已另选入Git；不做hash"}
    path.write_text(json.dumps(list(inventory.values()), ensure_ascii=False, indent=2) + "\n")
    print("selected", len(selected), "bytes", manifest["bytes"], "local paths", len(local))


if __name__ == "__main__":
    main()
