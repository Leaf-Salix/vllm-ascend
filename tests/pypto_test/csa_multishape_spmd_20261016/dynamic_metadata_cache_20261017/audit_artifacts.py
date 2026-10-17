# SPDX-License-Identifier: Apache-2.0
"""只读现有CPU产物，核对动态维度实际来自Tensor描述符或动态标量ABI。

offsets探针仅用positions的维度，所以编译器在AICPU读shape后以Scalar传入
AIV；不应只检索AIV的Extract dynamic dim注释而误认为T被实值化。这里同时
核对orchestration取shape、add_scalar与AIV unpack。其它维度直接从描述符取。
"""

import json
import re
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    artifacts = {"offsets": root / "cpu_v1/offsets", "rope": root / "cpu_v2/rope", "cache": root / "cpu_v2/cache"}
    required = {
        "offsets": {"CSA_GEOMETRY_QUERY_BOUNDS", "CSA_GEOMETRY_REQUESTS", "METADATA_CACHE_PROBE_T"},
        "rope": {"CSA_GEOMETRY_QUERY_BOUNDS", "CSA_GEOMETRY_REQUESTS", "CSA_METADATA_COMPACT_ROWS",
                 "METADATA_CACHE_PROBE_T"},
        "cache": {"CSA_ORIGINAL_CACHE_ROWS", "METADATA_CACHE_PROBE_T"},
    }
    report = {"device_execution": False, "recompiled": False, "stages": {},
              "initial_audit_correction": "初版只查AIV descriptor，漏掉offsets的动态T scalar ABI；并非编译失败"}
    for name, path in artifacts.items():
        entries, kernel_texts = [], []
        for source in path.glob("kernels/*/*.cpp"):
            lines = source.read_text().splitlines()
            kernel_texts.append("\n".join(lines))
            for index, line in enumerate(lines):
                match = re.search(r"Extract dynamic dim: (\w+)", line)
                if match:
                    statement = lines[index + 1]
                    assert "shapes[" in statement
                    entries.append({"file": str(source.relative_to(root)), "line": index + 2,
                                    "symbol": match.group(1), "statement": statement.strip()})
        # Inline重名消解可能添加_1，不把其误记为另一个固定维度。
        symbols = {re.sub(r"_\d+$", "", entry["symbol"]) for entry in entries}
        scalar_evidence = None
        if name == "offsets":
            orchestration = (path / "orchestration/offsets_entry.cpp").read_text()
            dimension = "int64_t METADATA_CACHE_PROBE_T = (int64_t)orch_args.tensor(0).ref().shapes[0];"
            assert dimension in orchestration
            assert "params_t0.add_scalar(METADATA_CACHE_PROBE_T);" in orchestration
            assert "// Unpack scalar: METADATA_CACHE_PROBE_T" in "\n".join(kernel_texts)
            symbols.add("METADATA_CACHE_PROBE_T")
            scalar_evidence = {"shape_source": dimension, "dispatch": "params_t0.add_scalar(METADATA_CACHE_PROBE_T)",
                               "kernel": "Unpack scalar: METADATA_CACHE_PROBE_T", "host_device_value_readback": False}
        assert required[name] <= symbols, (name, required[name], symbols)
        report["stages"][name] = {"required_axes": sorted(required[name]), "descriptor_reads": entries,
                                  "scalar_evidence": scalar_evidence, "passed": True}
    report["passed"] = True
    (root / "artifact_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "stages": list(report["stages"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
