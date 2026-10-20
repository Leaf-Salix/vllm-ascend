# SPDX-License-Identifier: Apache-2.0
"""本轮PMU与主投影候选的精确归档清单；只选文本，不执行Git操作。"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TEXT = {".py", ".json", ".jsonl", ".sh", ".md", ".log", ".csv"}


def main():
    selected = {Path(__file__).resolve(), HERE / "prepare_main_shared_left.py"}
    local = []

    def roots(directory):
        for path in directory.iterdir():
            if path.is_file() and path.suffix in TEXT and path.name != "symbolic.py":
                assert path.stat().st_size < 8 * 1024 * 1024, path
                selected.add(path)

    pmu = HERE / "current_pipe_pmu"
    main_root = HERE / "main_shared_left"
    roots(pmu)
    roots(main_root)
    for name in ("source_candidate", "source_candidate_v2"):
        roots(main_root / name)
    local.append(main_root / "source_preparation_v1")
    for batch in (16, 28):
        for version in (1, 2):
            cpu = main_root / f"cpu_b{batch}_v{version}"
            roots(cpu)
            local.extend([cpu / "kernel", cpu / "symbolic.py"])
        device = main_root / f"device_b{batch}_v2"
        roots(device)
        local.extend([device / "build_output", device / "source",
                      main_root / f"device_b{batch}_v2_device_logs"])
        matrix = device / "source_matrix"
        selected.add(matrix / "result.json")
        windows = list(matrix.glob("l1_*"))
        assert len(windows) == 15
        for window in windows:
            for name in ("chip_swimlane_records.json", "swimlane.json"):
                assert (window / name).is_file()
                selected.add(window / name)
        local.append(matrix / "reference_source")
        for directory in (device / "codegen_probe", matrix / "spmd_reference_codegen"):
            selected.update(directory.glob("*.json"))
            local.extend(directory.glob("*.cpp"))
        device = pmu / f"device_b{batch}_v1"
        roots(device)
        local.extend([device / "build_output", device / "source",
                      pmu / f"device_b{batch}_v1_device_logs"])
        selected.add(device / "pipe_profile/result.json")
        profile, = (device / "pipe_profile").glob("*_ascend_pt")
        selected.add(profile / "ASCEND_PROFILER_OUTPUT/kernel_details.csv")
        selected.add(profile / "ASCEND_PROFILER_OUTPUT/trace_view.json")
        for path in profile.iterdir():
            if path.name == "ASCEND_PROFILER_OUTPUT":
                local.extend(p for p in path.iterdir() if p not in selected)
            else:
                local.append(path)
        probe = device / "codegen_probe"
        selected.update(probe.glob("*.json"))
        local.extend(probe.glob("*.cpp"))
    manifest = {
        "selected_paths": [str(p.relative_to(REPO)) for p in sorted(selected)],
        "files": len(selected), "bytes": sum(p.stat().st_size for p in selected),
        "note": "完整PMU十二样本、主投影三方30窗、源及轻量证据；不含权重/二进制/大编译快照，不做hash",
    }
    (HERE / "MAIN_ROUND_ARCHIVE.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    path = HERE / "LOCAL_ARTIFACTS.json"
    previous = json.loads(path.read_text())
    index = {row["path"]: row for row in previous}
    for artifact in local:
        assert artifact.exists(), artifact
        files = [artifact] if artifact.is_file() else [p for p in artifact.rglob("*") if p.is_file()]
        key = str(artifact.relative_to(REPO))
        index[key] = {"path": key, "files": len(files),
                      "bytes": sum(p.stat().st_size for p in files), "in_git": False,
                      "purpose": "本轮PMU/共享Left的大产物或重复源码；选定源与完整文本证据入Git；不做hash"}
    path.write_text(json.dumps(list(index.values()), ensure_ascii=False, indent=2) + "\n")
    print("selected", len(selected), "bytes", manifest["bytes"], "local entries", len(local))


if __name__ == "__main__":
    main()
