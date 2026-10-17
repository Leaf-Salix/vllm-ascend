# SPDX-License-Identifier: Apache-2.0
"""CPU 解析独立核体集合，记录完整覆盖；不把解析通过当作融合编译通过。"""

import argparse
import json
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

import pypto.language as pl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    source = Path(catalog["body_bundle"])
    program = pl.parse(source.read_text(), filename=str(source))
    if len(program.functions) != catalog["physical_kernel_bodies"]:
        raise ValueError("解析后的核体数量不一致")
    result = {"body_count": len(program.functions), "pypto_parse": "PASS", "device_execution": False,
              "fused_codegen_completed": False, "source": str(source),
              "scope": "原计算体完整导出、逻辑 worker 显式化和独立 IR 解析通过"}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
