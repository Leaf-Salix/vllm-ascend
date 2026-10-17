# SPDX-License-Identifier: Apache-2.0
"""从已有L1原始记录修正入口名称映射；不测性能，不覆盖原始证据。"""

import argparse
import json
import os
from pathlib import Path

from l1_compare import export_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    report = json.loads((output / "result.json").read_text())
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(output / "build_output")
    rows = []
    for window in report["l1"]["windows"]:
        source = Path(window["raw"]).parent
        corrected = export_window(source, output / "corrected_names" / source.name, window["side"])
        assert corrected["incore_span_us"] == window["incore_span_us"]
        rows.append(corrected)
    (output / "trace_name_correction.json").write_text(
        json.dumps(
            {
                "reason": "独立plan probe的func_id=0误覆盖基线入口的func_id=0名称",
                "performance_rerun": False,
                "all_raw_spans_unchanged": True,
                "original_files_preserved": True,
                "windows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"corrected_windows": len(rows), "all_raw_spans_unchanged": True}))


if __name__ == "__main__":
    main()
