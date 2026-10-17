# SPDX-License-Identifier: Apache-2.0
"""只读已有 CPU 编译产物：动态 ABI、单 MIX 与 Q_A 真实写回顺序检查。"""
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


def qa_accesses(full, body, base, op):
    root = re.search(r"GlobalTensor<float,.*?> (v\d+) = GlobalTensor<float,.*?>\(v6,", body)
    assert root, "共享第6个GM参数必须构成QA视图"
    accesses = []
    pattern = rf"__gm__ float\* (v\d+) = PTOAS__GLOBAL_TENSOR_DATA\({root[1]}\);"
    for pointer in re.finditer(pattern, body):
        region = body[pointer.end():]
        view = re.search(rf"GlobalTensor<float,.*?> (v\d+) = GlobalTensor<float,.*?>\({pointer[1]}(?: \+|,)", region)
        assert view, pointer[1]
        call_pattern = rf"TSTORE\({view[1]}, (v\d+)\);" if op == "TSTORE" else rf"TLOAD\((v\d+), {view[1]}\);"
        call = re.search(call_pattern, region)
        assert call, (op, view[1])
        offset = base + pointer.end() + call.start()
        accesses.append({"line": line_at(full, offset), "instruction": call[0], "global_root": root[1],
                         "subview": view[1], "root_parameter": "v6 / qa / shared_tensor_arg[5]"})
    assert accesses
    return accesses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "不可覆盖旧审计"
    files = {kind: args.artifact / f"kernels/{kind}/{name}.cpp" for kind, name in (("aic", "cube"), ("aiv", "vector"))}
    texts = {kind: path.read_text() for kind, path in files.items()}
    cube, cube_start = side(texts["aic"], "cube")
    vector, vector_start = side(texts["aiv"], "vector")
    stores = qa_accesses(texts["aic"], cube, cube_start, "TSTORE")
    loads = qa_accesses(texts["aiv"], vector, vector_start, "TLOAD")
    waits = [line_at(texts["aic"], cube_start + m.start()) for m in re.finditer(r"pto::comm::TWAIT\(", cube)]
    publishes = [line_at(texts["aic"], cube_start + m.start()) for m in re.finditer(r"pto::comm::TNOTIFY\(", cube)]
    assert len(waits) == 2 and len(publishes) == 3
    assert len(stores) == 4, "Q_A的M32/M64各保留dense与M16尾块写回"
    assert all(waits[1] < store["line"] < publishes[-1] for store in stores)
    assert len(loads) == 1
    vector_waits = [line_at(texts["aiv"], vector_start + m.start()) for m in re.finditer(r"pto::comm::TWAIT\(", vector)]
    assert vector_waits[-1] < loads[0]["line"]
    # NORM/QA phase基址来自6/7×72×128个INT32，不是某个T的等待表。
    assert "= 55296;" in cube and "= 64512;" in cube
    dynamic_dims = {}
    for kind, text in texts.items():
        dynamic_dims[kind] = [{"name": m[1], "line": line_at(text, m.start()), "statement": m[0]}
                             for m in re.finditer(r"int64_t (FRONT_\w+) = .*?->shapes\[\d+\]\);", text)]
        assert {row["name"] for row in dynamic_dims[kind]} == {
            "FRONT_T", "FRONT_PAD16", "FRONT_PAD8", "FRONT_PARTIAL_ROWS", "FRONT_SIGNAL_WORDS"}
        assert 'qa__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor*>(args[5]);' in text
    orch = "\n".join(path.read_text() for path in (args.artifact / "orchestration").glob("*.cpp"))
    calls = re.findall(r"\brt_submit_\w+\s*\(", orch)
    assert len(calls) == 1 and "MixedKernels" in orch and "set_require_sync_start(true)" in orch
    report = {
        "passed": True, "scope": "已有full_v3编译产物，只读复核，无重编译或设备运行",
        "full_codegen_passed": True, "single_mix_submission": True, "submission_calls": calls,
        "device_precision_validated": False, "device_sync_validated": False, "performance_measured": False,
        "original_result_note": "full_v3/result.json末尾失败只因ABI正则漏右括号；编译本身成功。原记录保留。",
        "compiler_warning": "显式qa_gm原参数别名使qa_v1返回SSA未使用；以下实际存储证明QA未被删除",
        "cpp_files": {key: str(path) for key, path in files.items()},
        "dynamic_dimensions": dynamic_dims,
        "qa_alias": {"shared_tensor_slot": 5, "same_pointer_parameter": "v6", "aic_writebacks": stores,
                     "aiv_loads": loads, "aic_wait_lines": waits, "aic_publish_lines": publishes,
                     "norm_wait_before_all_qa_stores": True, "qa_publish_after_all_qa_stores": True,
                     "aiv_qa_wait_lines": vector_waits[-2:], "qa_wait_before_read": True},
        "limitations": ["顺序/别名为编译源码证据；不证明A3硬件同步或数值", "未设置权重BYPASS缓存策略",
                        "HC输出与QA/QR数学仍待设备验证；不是完整CSA", "phase完成通知没有ACK，不可逐tile覆盖工作区"],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary = {key: report[key] for key in ("passed", "full_codegen_passed", "single_mix_submission")}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
