# SPDX-License-Identifier: Apache-2.0
"""Check final generated task bodies for pipe setup before the manual Wait."""

import argparse
import json
import re
from pathlib import Path

PIPE_OP = re.compile(
    r"\b(?:set_flag|wait_flag|pipe_barrier|set_ffts_base_addr|"
    r"TLOAD|TSTORE|TPipe|TPUSH|TPOP|TFREE)\s*[<(]"
)


def inspect_kernels(build):
    rows = []
    for path in sorted((build / "kernels").rglob("*.cpp")):
        text = path.read_text()
        pattern = r"^static __aicore__ void " + re.escape(path.stem) + r"\([^\n]*\) \{"
        start = re.search(pattern, text, re.MULTILINE)
        assert start, f"task definition not found: {path}"
        end = text.find(" = pypto_wait_local_boundary(", start.end())
        assert end >= 0, f"local Wait missing: {path}"
        prefix = text[start.end():end]
        bad = [
            {"line": text[:start.end() + match.start()].count("\n") + 1, "op": match.group().strip()}
            for match in PIPE_OP.finditer(prefix)
        ]
        rows.append({
            "function": path.stem,
            "kernel": str(path.relative_to(build)),
            "keep_operands": prefix.count("pypto_keep_prefix("),
            "unsafe_pipe_ops": bad,
        })
    assert len(rows) == 73, f"expected frozen CSA's 73 members, got {len(rows)}"
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"scope": "final generated C++ task bodies, not an instruction-latency measurement"}
    report["old"] = inspect_kernels(args.old)
    for variant in ("current", "prefix_layout"):
        report[variant] = inspect_kernels(args.compiled / variant / "build")
        assert all(not row["unsafe_pipe_ops"] for row in report[variant]), variant
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for variant in ("old", "current", "prefix_layout"):
        findings = {row["function"]: row["unsafe_pipe_ops"] for row in report[variant] if row["unsafe_pipe_ops"]}
        print(variant, "members=73", json.dumps(findings))


if __name__ == "__main__":
    main()
