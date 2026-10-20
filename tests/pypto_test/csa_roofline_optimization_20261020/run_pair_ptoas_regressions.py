# SPDX-License-Identifier: Apache-2.0
"""使用实际新PTOAS验证成对压缩的生成、同步模型和无效合同拒绝。"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    workspace = Path(__file__).resolve().parents[4]
    tree = workspace / ".worktrees/ptoas-csa-pair-compact-20261020"
    binary = tree / "build/bin/ptoas"
    source = tree / "test/lit/pto/tcompact_compare_a3.pto"
    report = {"passed": False, "binary": str(binary), "checks": []}
    for prefix, extra in (("CHECK", []), ("SYNC", ["--pto-insert-sync-debug=2"])):
        command = [str(binary), "--pto-arch=a3", "--enable-insert-sync",
                   *extra, str(source), "-o", "-"]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (args.output / (prefix.lower() + ".txt")).write_text(result.stdout)
        check = subprocess.run(
            ["/usr/bin/FileCheck", str(source), "--check-prefix=" + prefix],
            input=result.stdout, capture_output=True, text=True,
        )
        (args.output / (prefix.lower() + "_filecheck.txt")).write_text(check.stderr)
        report["checks"].append({"name": prefix, "passed": result.returncode == 0
                                 and check.returncode == 0, "command": command})
    text = source.read_text()
    negatives = [
        ("unsupported_mode", text.replace("#pto<cmp ge>", "#pto<cmp ne>"), "eq/gt/ge", "a3"),
        ("too_small_scratch", text.replace("cols=1280", "cols=1024").replace("v_col=1280", "v_col=1024"),
         "scratch is too small", "a3"),
        ("output_capacity", text.replace("cols=32", "cols=128").replace("v_col=32", "v_col=128"),
         "capacity <= src cols", "a3"),
        ("a5_rejected", text, "requires A3", "a5"),
    ]
    for name, content, expected, arch in negatives:
        case = args.output / (name + ".pto")
        case.write_text(content)
        result = subprocess.run(
            [str(binary), "--pto-arch=" + arch, str(case), "-o", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        (args.output / (name + ".txt")).write_text(result.stdout)
        report["checks"].append({"name": name, "passed": result.returncode != 0
                                 and expected in result.stdout, "expected": expected})
    report["passed"] = all(c["passed"] for c in report["checks"])
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
