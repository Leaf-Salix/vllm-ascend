# SPDX-License-Identifier: Apache-2.0
"""选择复测代码/报告/原始计时文本；大编译、完整trace与二进制仅登记路径。"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TEXT = {".py", ".md", ".json", ".sh", ".log", ".txt", ".csv"}


def main():
    selected, local = set(), set()
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in TEXT and path.name not in {"ARCHIVE_MANIFEST.json", "LOCAL_ARTIFACTS.json"}:
            selected.add(path)
    for name in ("source_dynamic", "measurement_source"):
        selected.update((HERE / name).glob("*.py"))
    for case in sorted((HERE / "results").iterdir()):
        selected.update(path for path in case.iterdir() if path.is_file() and path.suffix in TEXT)
        for phase in ("native_v1", "multitask_outer_v1", "l1_v1", "static_v1"):
            folder = case / phase
            if not folder.exists():
                continue
            for path in folder.iterdir():
                if path.is_file() and path.suffix in TEXT and path.name != "symbolic.py":
                    selected.add(path)
                elif path.is_dir():
                    local.add(path)
            report = folder / "report.json"
            if report.exists():
                value = json.loads(report.read_text())
                selected.add(Path(value["device_profile"]["raw_csv"]))
                short_cwd = folder / "compile_workdir.txt"
                if short_cwd.exists():
                    local.add(Path(short_cwd.read_text().strip()))
            for window in folder.glob("l1_*"):
                if window.is_dir():
                    selected.update(window / name for name in ("chip_swimlane_records.json", "name_map.json")
                                    if (window / name).is_file())
                    selected.update(window.glob("*_kernel_config.py"))
        for path in case.iterdir():
            if path.is_dir() and path.name not in {"native_v1", "multitask_outer_v1", "l1_v1", "static_v1"}:
                local.add(path)
    for path in selected:
        assert path.suffix in TEXT and path.stat().st_size < 8 * 1024 * 1024, path
    artifacts = []
    for path in sorted(local):
        files = [p for p in path.rglob("*") if p.is_file()]
        artifacts.append({"path": str(path), "files": len(files),
                          "bytes": sum(p.stat().st_size for p in files),
                          "selected_text_files": [str(p) for p in files if p in selected],
                          "note": "目录大产物本地保留；只有selected_text_files进入Git，不做hash"})
    (HERE / "LOCAL_ARTIFACTS.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2) + "\n")
    selected.add(HERE / "LOCAL_ARTIFACTS.json")
    manifest = {"selected_paths": [str(p.relative_to(REPO)) for p in sorted(selected)],
                "files": len(selected), "bytes": sum(p.stat().st_size for p in selected),
                "scope": "复测源、所有报告/日志、原始CSV与L1计时入Git；完整trace/编译/二进制本地；无权重、无hash"}
    (HERE / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("selected",len(selected),"bytes",manifest["bytes"],"local directories",len(artifacts))


if __name__ == "__main__":
    main()
