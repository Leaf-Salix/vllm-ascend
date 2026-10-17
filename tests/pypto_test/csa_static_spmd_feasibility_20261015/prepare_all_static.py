# SPDX-License-Identifier: Apache-2.0
"""一次生成/解析七档静态模板；不做设备执行，也不重复动态算子验证。"""

import json
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

import pypto.language as pl
from generate_static_entry import generate, jit_source


def main():
    here = Path(__file__).resolve().parent
    catalog = json.loads((here / "generated/core_body_catalog.json").read_text())
    reports = []
    for path in sorted((here / "bindings").glob("*_prepared.json")):
        case = path.name.removesuffix("_prepared.json")
        directory = here / "static" / case
        schedule = json.loads(path.read_text())
        layout = json.loads((directory / "layout.json").read_text())
        source, bindings = generate(schedule, layout, catalog)
        filename = directory / "entry.py"
        filename.write_text(source)
        (directory / "entry_jit.py").write_text(jit_source(source))
        (directory / "entry.bindings.json").write_text(json.dumps(bindings, ensure_ascii=False, indent=2) + "\n")
        program = pl.parse(source, filename=str(filename))
        report = {"case": case, "parse_passed": True, "functions": len(program.functions),
                  "specialized_compute_bodies": len(bindings) - 1, "source": str(filename),
                  "entry_contract": bindings["_entry_contract"],
                  "full_codegen_passed": False, "device_execution": False}
        (directory / "parse_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    (here / "static/parse_all_result.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
