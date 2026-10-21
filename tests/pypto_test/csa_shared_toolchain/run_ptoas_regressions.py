# SPDX-License-Identifier: Apache-2.0
"""用公共PTOAS执行本轮官方修复及保留补丁的定向RUN/FileCheck检查。"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from selection import selected_toolchain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, roots, manifest = selected_toolchain()
    binary = roots["ptoas"] / "bin/ptoas"
    tree = roots["ptoas"].parent
    check = shutil.which("FileCheck")
    if not binary.is_file() or check is None:
        raise RuntimeError("公共PTOAS或FileCheck尚不可用")
    args.output.mkdir(parents=True, exist_ok=False)
    directory = tree / "test/lit/pto"
    cases = [
        (
            directory / "issue1620_dynamic_alloc_alias_sync.pto",
            "a3",
            ["--pto-level=level3", "--enable-insert-sync"],
            "CHECK",
            False,
        ),
        (directory / "issue1622_cfg_parallel_copy.pto", "a3", [], "CHECK", False),
        (directory / "issue1622_cfg_parallel_copy.pto", "a5", [], "CHECK", False),
        (directory / "comm_twait_backoff.pto", "a3", [], "CHECK", False),
        (directory / "comm_twait_backoff.pto", "a5", [], "A5", True),
        (directory / "emitc_nested_bitwise_precedence.pto", "a3", [], "CHECK", False),
        (directory / "tgather_scalar_compare_a3_emitc.pto", "a3", ["--enable-insert-sync"], "CHECK", False),
    ]
    for name in (
        "test_acc_column_subview_sync.pto",
        "test_key_l1_regions_sync.pto",
    ):
        cases.append(
            (tree / "test/samples/Sync" / name, "a3", ["--enable-insert-sync", "--pto-level=level3"], "CHECK", False)
        )

    # 按官方split-file标记生成三份无效输入，不改动原始回归源码。
    negative_checks = directory / "comm_twait_backoff_invalid.pto"
    sections = {}
    current = None
    for line in negative_checks.read_text().splitlines(keepends=True):
        if line.startswith("//--- "):
            current = line[6:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    for name, lines in sections.items():
        path = args.output / name
        path.write_text("".join(lines))
        cases.append((path, "a3", ["--emit-pto-ir"], path.stem.upper(), True))

    records = []
    for source, arch, options, prefix, expected_error in cases:
        command = [str(binary), "--pto-arch=" + arch, *options, str(source), "-o", "-"]
        run = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        name = f"{source.stem}_{arch}_{prefix}"
        (args.output / (name + ".txt")).write_text(run.stdout)
        pattern_file = negative_checks if source.parent == args.output else source
        verify = subprocess.run(
            [check, str(pattern_file), "--check-prefix=" + prefix],
            input=run.stdout,
            text=True,
            capture_output=True,
        )
        passed = (run.returncode != 0 if expected_error else run.returncode == 0) and verify.returncode == 0
        records.append(
            {
                "case": name,
                "passed": passed,
                "command": command,
                "compiler_exit": run.returncode,
                "filecheck_exit": verify.returncode,
                "filecheck_output": verify.stdout + verify.stderr,
            }
        )
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
    report = {
        "passed": all(record["passed"] for record in records),
        "device_execution": False,
        "release": data["release"],
        "manifest": str(manifest),
        "not_adopted": {
            "test_gm_column_partition_sync.pto": (
                "GM column-disjoint optimization was explicitly reverted in the previous "
                "validated toolchain by e5c1c432; its historical test expects the rejected candidate."
            )
        },
        "cases": records,
    }
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
