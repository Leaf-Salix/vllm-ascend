# SPDX-License-Identifier: Apache-2.0
"""执行三项针对性 PTOAS RUN/FileCheck 检查，不依赖本机 lit 配置初始化。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptoas-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check = shutil.which("FileCheck")
    if not check:
        raise RuntimeError("PATH 中缺少 FileCheck，不把未执行记录为通过")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [
        ("tgather_scalar_compare_a3_emitc.pto", ["--enable-insert-sync"]),
        ("tgather_a2a3_index_emitc.pto", []),
        ("tgather_compare_macro_sync_model.pto", ["--enable-insert-sync", "--pto-insert-sync-debug=2"]),
    ]
    records = []
    for name, options in cases:
        source = args.ptoas_root / "test/lit/pto" / name
        command = [str(args.ptoas_root / "build/bin/ptoas"), "--pto-arch=a3", *options, str(source), "-o", "-"]
        run = subprocess.run(command, text=True, capture_output=True)
        combined = run.stdout + run.stderr
        (args.output / (name + ".output")).write_text(combined)
        verify = (subprocess.run([check, str(source)], input=combined, text=True, capture_output=True)
                  if run.returncode == 0 else None)
        records.append({
            "test": name, "command": command, "compiler_exit": run.returncode,
            "filecheck_exit": verify.returncode if verify else None,
            "filecheck_output": verify.stdout + verify.stderr if verify else "",
        })
    report = {
        "passed": all(r["compiler_exit"] == 0 and r["filecheck_exit"] == 0 for r in records),
        "cases": records, "filecheck": check, "device_validation": False,
    }
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
