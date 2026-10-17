# SPDX-License-Identifier: Apache-2.0
"""只读既有attention计划CPU产物，审计动态ABI、真实写回及无cache payload访存。"""

import argparse
import json
import re
from pathlib import Path


def role_body(text, name):
    start = text.index(f"static __aicore__ void {name}(")
    end = text.index("\n}\n", start) + 3
    body = text[start:end]
    signature = re.search(rf"void {name}\(([^\n]+)\) {{", body)[1]
    formals = [re.search(r"(v\d+)$", item.strip())[1] for item in signature.split(",")]
    invocation = list(re.finditer(rf"^    {name}\(([^\n]+)\);", text, re.M))[-1]
    actuals = invocation[1].split(", ")
    assert len(formals) == len(actuals)
    origins = dict(zip(formals, (arg.split("__ssa_")[0] for arg in actuals)))
    accesses = {}
    waits, publishes = [], []
    for local_line, line in enumerate(body.splitlines(), 1):
        number = text.count("\n", 0, start) + local_line
        view = re.search(r"GlobalTensor<.*> (v\d+) = GlobalTensor<.*>\((v\d+)(?:,| \+)", line)
        pointer = re.search(r"__gm__ \w+\* (v\d+) = PTOAS__GLOBAL_TENSOR_DATA\((v\d+)\);", line)
        if view or pointer:
            match = view or pointer
            assert match[2] in origins, (name, line)
            origins[match[1]] = origins[match[2]]
        store = re.search(r"\bTSTORE\((v\d+), (v\d+)\);", line)
        load = re.search(r"\bTLOAD\((v\d+), (v\d+)\);", line)
        scalar_store = re.search(r"\((v\d+)\)\[[^\]]+\] = [^;]+;", line)
        scalar_load = re.search(r"\b(?:int32_t|int64_t|float) (v\d+) = \((v\d+)\)\[[^\]]+\];", line)
        if store or load or scalar_store or scalar_load:
            match = store or load or scalar_store or scalar_load
            variable = match[1] if store or scalar_store else match[2]
            op = "TSTORE" if store else "TLOAD" if load else "scalar_store" if scalar_store else "scalar_load"
            assert variable in origins, (name, line)
            accesses.setdefault(origins[variable], {}).setdefault(op, []).append(
                {"line": number, "statement": match[0]}
            )
        if "pto::comm::TWAIT(" in line:
            waits.append(number)
        if "pto::comm::TNOTIFY(" in line:
            publishes.append(number)
    return {
        "body": body,
        "base_line": text.count("\n", 0, start),
        "arguments": dict(zip(formals, actuals)),
        "accesses": accesses,
        "waits": waits,
        "publishes": publishes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    files = list((args.artifact / "kernels").rglob("*.cpp"))
    assert len(files) == 1 and "/aiv/" in str(files[0])
    text = files[0].read_text()
    name = re.search(r"static __aicore__ void (\w+)\(", text)[1]
    role = role_body(text, name)
    access = role["accesses"]
    assert len(access["topk"]["TLOAD"]) == 1 and "scalar_load" not in access["topk"]
    assert len(access["sin"]["TLOAD"]) == 1
    for output in ("indices", "bias", "signed_sin"):
        assert access[output]["TSTORE"], output
    assert len(access["mask"]["scalar_store"]) == 2
    assert not {"original_cache", "compressed_cache"}.intersection(access)
    assert not any("original_cache" in v or "compressed_cache" in v for v in role["arguments"].values())
    assert not role["waits"] and not role["publishes"]
    for key in ("original_table", "compressed_table", "positions", "bounds", "lengths"):
        assert access[key]["scalar_load"], key
    dims = re.findall(r"// Extract dynamic dim: (\w+)\n\s*int64_t \w+ = static_cast<int64_t>\(([^;]+)\);", text)
    assert len(dims) == 7 and all("->shapes[" in expr for _, expr in dims)
    scalar_caps = re.findall(r"// Unpack scalar: (CSA_ATTN_PROBE_\w+)", text)
    assert set(scalar_caps) == {"CSA_ATTN_PROBE_COMPRESSED_PAGES", "CSA_ATTN_PROBE_ORIGINAL_PAGES"}
    orch = "\n".join(p.read_text() for p in (args.artifact / "orchestration").glob("*.cpp"))
    cap_origins = re.findall(
        r"int64_t (CSA_ATTN_PROBE_\w+) = \(int64_t\)orch_args.tensor\((\d+)\).ref\(\).shapes\[0\];", orch
    )
    assert set(cap_origins) == {("CSA_ATTN_PROBE_ORIGINAL_PAGES", "6"), ("CSA_ATTN_PROBE_COMPRESSED_PAGES", "7")}
    submissions = re.findall(r"rt_submit_\w+\(", orch)
    assert submissions == ["rt_submit_aiv_task("]
    assert text.count("TROWMAX(") == 1
    column_vectors = re.findall(r"Tile<TileType::Vec, (?:float|int32_t), 8, 1, BLayout::ColMajor[^\n]+", role["body"])
    assert len(column_vectors) >= 3
    capacity_guards = {}
    for cap in scalar_caps:
        parameter = next(k for k, v in role["arguments"].items() if v == cap)
        guards = [m[0] for m in re.finditer(rf"if \([^\n]*< {parameter}\)", role["body"])]
        assert guards, cap
        capacity_guards[cap] = {"cpp_parameter": parameter, "guards": guards}
    # 两个动态8行循环分别覆盖plan和逆RoPE；真实上限来自T descriptor。
    token_parameter = next(k for k, v in role["arguments"].items() if v == "CSA_ATTN_PLAN_T")
    dynamic_loops = [
        m[0] for m in re.finditer(rf"for \(int64_t v\d+ = [^;]+; v\d+ < {token_parameter};[^\n]+", role["body"])
    ]
    assert len(dynamic_loops) == 2, dynamic_loops
    # 尾行数需真实进入indices与signed_sin的TSTORE视图，不只存在于源注释。
    store_context = {}
    lines = text.splitlines()
    for output in ("indices", "signed_sin"):
        line = access[output]["TSTORE"][0]["line"]
        store_context[output] = {"line": line, "context": lines[line - 35 : line]}
        columns = 512 if output == "indices" else 64
        assert f"pto::Shape<1, 1, 1, -1, {columns}>" in "\n".join(lines[line - 35 : line])
    meta = json.loads((args.artifact / "compiled_meta.json").read_text())
    assert len(meta["params"]) == 13 and meta["num_return_types"] == 4
    result = {
        "passed": True,
        "scope": "attention plan与inverse RoPE两个helper的既有CPU产物审计，非完整CSA",
        "artifact": str(args.artifact.resolve()),
        "kernel_cpp": str(files[0]),
        "single_aiv_submission": True,
        "public_tensor_count": 13,
        "incore_tensor_count": 11,
        "incore_scalar_page_counts": 2,
        "cpp_dynamic_descriptor_dimensions": dims,
        "page_count_scalar_origins": cap_origins,
        "page_capacity_guards": capacity_guards,
        "dynamic_T_loops": dynamic_loops,
        "column_major_8x1_tile_count": len(column_vectors),
        "row_max_instructions": 1,
        "real_GM_accesses": access,
        "tail_store_context": store_context,
        "cache_payload_pointer_absent_from_incore": True,
        "topk_page_validation_reads_local_tile": True,
        "no_embedded_wait_notify": True,
        "device_execution": False,
        "device_precision_validated": False,
        "performance_measured": False,
        "limits": [
            "CPP地址/顺序与CPU oracle不是A3设备指令/缓存验证。",
            "ScalarWriteLineShared静态警告保留，mask按64B整行和8行tile分配给唯一owner；入口须保证64B对齐。",
            "实际T、输入动态轴一致性和workspace容量需要未来整层入口检查。",
        ],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "dynamic_dims": 9, "public_tensors": 13, "cache_payload_access": False}))


if __name__ == "__main__":
    main()
