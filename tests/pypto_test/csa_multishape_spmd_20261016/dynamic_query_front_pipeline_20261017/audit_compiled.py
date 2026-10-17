# SPDX-License-Identifier: Apache-2.0
"""只读query前段既有CPU产物，检查NZ布局、动态ABI和跨阶段实际GM存取。

依据每个具名GM根视图追踪子视图及TLOAD/TSTORE，而不是仅查函数名是否存在。
此检查可以发现返回原GM别名时数学被DCE或错误缓冲绑定，但不模拟A3缓存、
不执行设备数值，也不把CPP文本顺序当成完整硬件同步证明。
"""
import argparse
import json
import re
from pathlib import Path


def line_at(text, offset):
    return text.count("\n", 0, offset) + 1


def side(text, name):
    start = text.index(f"static __aicore__ void {name}(")
    end = text.index("\n}\n", start) + 3
    return text[start:end], start


def root_view(body, name):
    pattern = rf"// pto: %{name}__ssa_v0_view\n\s*(GlobalTensor<[^\n]+> (v\d+) = [^\n]+\((v\d+),[^\n]+;)"
    match = re.search(pattern, body)
    assert match, (name, "没有同名GM根视图")
    return match[2], match[3], match[1]


def accesses(full, body, base, name, op):
    root, parameter, declaration = root_view(body, name)
    result = []
    pointers = rf"__gm__ \w+\* (v\d+) = PTOAS__GLOBAL_TENSOR_DATA\({root}\);"
    for pointer in re.finditer(pointers, body):
        region = body[pointer.end():]
        view = re.search(rf"GlobalTensor<[^\n]+> (v\d+) = GlobalTensor<[^\n]+>\({pointer[1]}(?: \+|,)", region)
        assert view, (name, pointer[1])
        call_pattern = rf"TSTORE\({view[1]}, (v\d+)\);" if op == "TSTORE" else rf"TLOAD\((v\d+), {view[1]}\);"
        call = re.search(call_pattern, region)
        if call:
            result.append({"line": line_at(full, base + pointer.end() + call.start()),
                           "instruction": call[0], "subview": view[1]})
    assert result, (name, op, "数学真实访存必须保留")
    return {"parameter": parameter, "gm_root": root, "root_declaration": declaration, "operations": result}


def bounded(access, first, last):
    assert all(first < item["line"] < last for item in access["operations"]), (access, first, last)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "不可覆盖已有证据"
    files = {kind: args.artifact / f"kernels/{kind}/{name}.cpp" for kind, name in (("aic", "cube"), ("aiv", "vector"))}
    texts = {kind: path.read_text() for kind, path in files.items()}
    cube, cube_base = side(texts["aic"], "cube")
    vector, vector_base = side(texts["aiv"], "vector")
    sync = {}
    for name, kind, body, base in (("cube", "aic", cube, cube_base), ("vector", "aiv", vector, vector_base)):
        sync[name] = {op: [line_at(texts[kind], base + m.start())
                          for m in re.finditer(r"pto::comm::" + op + r"\(", body)]
                      for op in ("TWAIT", "TNOTIFY")}
    cw, cn = sync["cube"]["TWAIT"], sync["cube"]["TNOTIFY"]
    vw, vn = sync["vector"]["TWAIT"], sync["vector"]["TNOTIFY"]
    assert len(cw) == 3 and len(cn) == 4
    assert len(vw) == 9 and len(vn) == 9
    evidence = {}
    for role, name, op in (("aic", "qa", "TSTORE"), ("aiv", "qa", "TLOAD"),
                           ("aiv", "qr_matmul", "TSTORE"), ("aic", "qr_matmul", "TLOAD"),
                           ("aic", "q_proj", "TSTORE"), ("aiv", "q_proj", "TLOAD"),
                           ("aiv", "rope_signed", "TSTORE"), ("aiv", "rope_signed", "TLOAD"),
                           ("aiv", "rope_swap", "TSTORE"), ("aiv", "rope_swap", "TLOAD"),
                           ("aiv", "q_flat", "TSTORE")):
        body, base = (cube, cube_base) if role == "aic" else (vector, vector_base)
        evidence[f"{role}_{name}_{op}"] = accesses(texts[role], body, base, name, op)
    assert len(evidence["aic_qa_TSTORE"]["operations"]) == 4
    bounded(evidence["aic_qa_TSTORE"], cw[1], cn[2])
    bounded(evidence["aiv_qa_TLOAD"], vw[5], vn[6])
    bounded(evidence["aiv_qr_matmul_TSTORE"], vw[5], vn[6])
    bounded(evidence["aic_qr_matmul_TLOAD"], cw[2], cn[3])
    bounded(evidence["aic_q_proj_TSTORE"], cw[2], cn[3])
    bounded(evidence["aiv_q_proj_TLOAD"], vw[-1], vn[-1])
    for name in ("rope_signed", "rope_swap"):
        bounded(evidence[f"aiv_{name}_TSTORE"], vn[6], vn[7])
        bounded(evidence[f"aiv_{name}_TLOAD"], vw[-1], vn[-1])
    bounded(evidence["aiv_q_flat_TSTORE"], vw[-1], vn[-1])
    for name in ("qa", "qr_matmul", "q_proj"):
        _, a, _ = root_view(cube, name)
        _, b, _ = root_view(vector, name)
        assert a == b, (name, a, b)
    weights = {}
    for name, dtype in (("wq_a", "bfloat16_t"), ("wq_b", "int8_t")):
        root, parameter, declaration = root_view(cube, name)
        assert "pto::Layout::NZ" in declaration and f"GlobalTensor<{dtype}," in declaration
        weights[name] = {"layout": "NZ", "dtype": dtype, "root": root, "parameter": parameter,
                         "declaration": declaration}
    dimensions = {}
    for kind, text in texts.items():
        dimensions[kind] = [{"name": m[1], "line": line_at(text, m.start()), "statement": m[0]}
                            for m in re.finditer(r"int64_t (QUERY_FRONT_\w+) = .*?->shapes\[\d+\]\);", text)]
        assert {row["name"] for row in dimensions[kind]} == {
            "QUERY_FRONT_T", "QUERY_FRONT_PAD16", "QUERY_FRONT_PAD8", "QUERY_FRONT_PARTIAL_ROWS",
            "QUERY_FRONT_QR_PAD128", "QUERY_FRONT_PROJ_PAD128", "QUERY_FRONT_SIGNAL_WORDS"}
        unpacked = re.findall(r"// Unpack tensor: (\w+)", text)
        assert len(unpacked) == 35 and len(set(unpacked)) == 35
    orch = "\n".join(path.read_text() for path in (args.artifact / "orchestration").glob("*.cpp"))
    submissions = re.findall(r"\brt_submit_\w+\s*\(", orch)
    assert len(submissions) == 1 and "MixedKernels" in orch and "set_require_sync_start(true)" in orch
    result = {"passed": True, "scope": "动态query前段已有完整CPU产物审计，无设备执行",
              "single_mix_submission": True, "submissions": submissions, "tensor_abi_count": 35,
              "phase_count_including_generation": 12, "participant_cores": 72,
              "cpp_files": {kind: str(path) for kind, path in files.items()},
              "actual_weight_layouts": weights, "dynamic_descriptor_dimensions": dimensions,
              "signal_instruction_lines": sync, "gm_accesses": evidence,
              "verified_order": ["HC_NORM等待→QA写回→QA发布→QR读取QA并写INT8暂存→QR发布",
                                 "QR等待→QB读取同一INT8暂存并写INT32投影→QB发布",
                                 "RoPE准备写回→ROPE发布；QB/ROPE/QR等待→query读取→query输出→QUERY发布"],
              "cache_policy_override": False, "device_execution": False,
              "device_precision_validated": False, "device_sync_validated": False, "performance_measured": False,
              "limits": ["保留qa/q_proj原GM别名的返回SSA未使用警告，已查实际存储未被DCE",
                         "CPP依赖顺序不等于已验证设备缓存一致性或数值", "不包含KV/Indexer/Attention，非完整CSA",
                         "BYPASS外层标记在full_v1被拒绝；本轮未设置缓存策略，不做性能结论"]}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "NZ2": True, "dynamic_dims": 7, "single_mix": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
