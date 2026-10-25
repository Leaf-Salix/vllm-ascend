# SPDX-License-Identifier: Apache-2.0
"""审查两档融合候选的实际PTO、内部编排和完整CPU编译产物。"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMITS = dict(vec=188416, mat=524288, acc=131072, left=65536, right=65536)
BYTES = dict(i8=1, i16=2, f16=2, bf16=2, f32=4, i32=4, ui32=4, i64=8)


def resources(pto):
    result = {}
    for chunk in pto.split("  func.func @")[1:]:
        name = chunk.split("(", 1)[0]
        constants = dict(
            (key, int(value)) for key, value in re.findall(r"(%[\w]+) = arith.constant (\d+) : (?:i64|index)", chunk)
        )
        maxima = {}
        witnesses = {}
        for line_number, line in enumerate(chunk.splitlines(), 1):
            if "pto.alloc_tile addr" not in line:
                continue
            match = re.search(
                r"addr = (%[\w]+).*loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+),.*blayout=(\w+), slayout=(\w+)", line
            )
            assert match, line
            address, space, dtype, rows, cols, layout, storage = match.groups()
            assert address in constants and dtype in BYTES
            width = BYTES[dtype]
            rows, cols = int(rows), int(cols)
            size = rows * cols * width
            if space == "vec" and storage == "none_box":
                if layout == "row_major":
                    size = rows * ((cols * width + 31) // 32 * 32)
                else:
                    size = cols * ((rows * width + 31) // 32 * 32)
            end = constants[address] + size
            if end > maxima.get(space, 0):
                maxima[space] = end
                witnesses[space] = dict(
                    address=constants[address], bytes=size, line_in_function=line_number, declaration=line.strip()
                )
        assert all(maxima.get(space, 0) <= limit for space, limit in LIMITS.items()), maxima
        result[name] = dict(maximum_declared_extent_bytes=maxima, witnesses=witnesses)
    return result


def main():
    report = dict(
        candidate="fused_coefficients_current_v2", passed=True, device_execution=False, adopted=False, cases={}
    )
    for case in ("C04", "C05"):
        root = HERE / f"runs/codegen_fused_coefficients_current_{case}_v2"
        result = json.loads((root / "result.json").read_text())
        assert result["stage"] == "cpu_backend_compiled" and result["passed"]
        (build,) = (root / "build_output").iterdir()
        pto_path = build / "ptoas/dynamic_csa_full.pto"
        pto = pto_path.read_text()
        stores = [
            line
            for line in pto.splitlines()
            if "pto.tstore" in line and '/fused_coefficients.py"' in line and "outs(%coefficients" in line
        ]
        assert len(stores) == 3
        assert sum("partition_tensor_view<1x64xf16>" in line for line in stores) == 2
        assert sum("partition_tensor_view<16x384xf16>" in line for line in stores) == 1
        orch = (build / "orchestration/decode_csa_spmd.cpp").read_text()
        assert "group_offset_words + CSA_FULL_T" in orch
        assert "TensorCreateInfo group_offsets_ci" in orch
        meta = json.loads((build / "compiled_meta.json").read_text())
        assert not any(p["name"] == "group_offsets" for p in meta["params"])
        report["cases"][case] = dict(
            cpu_backend_compiled=True,
            compile_result=str(root / "result.json"),
            pto=str(pto_path),
            resources=resources(pto),
            coefficient_store_declarations=stores,
            internal_offset_T_word_extension=True,
            external_group_offsets_parameter=False,
            shared_toolchain=result["shared_toolchain"],
        )
    report["limits"] = LIMITS
    report["dependency_proof"] = [
        "META唯一planner执行group plan后清零全部实际group矩阵并填map，再发布PHASE_META。",
        "每个AIV在indexer_compressor_vector_stage先获取META；该helper在两种COOPERATIVE_PREFIX路径均先于量化。",
        "每个AIV先完成head-weight reduce并发布PHASE_WEIGHTS，再到新增前置获取；WEIGHTS生产者不等待QUANT。",
        "token idx只归idx%48；128B对角片由该owner唯一写入。重复区间只让同一owner写不同group，不产生共享片。",
        "QUANT发布在所有系数写出之后；AIC原tokens/rows_per_tile=1获取集覆盖所有实际token%48 owner，故可省COEFF获取。",
    ]
    report["limitations"] = [
        "资源为最终PTO声明地址上界并结合完整编译成功，不把地址总和当同时live峰值。",
        "数值/状态/服务和性能均待NPU；v1重叠Q负对照失败，设备只可选择v2。",
        "当前完整CPU编译覆盖正式入口；debug入口追加容量及参数转发有源码合同，设备壳仍将重新编译debug入口。",
    ]
    (HERE / "fused_coefficients_current_v2_codegen.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                case: {fn: row["maximum_declared_extent_bytes"] for fn, row in data["resources"].items()}
                for case, data in report["cases"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
