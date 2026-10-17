# SPDX-License-Identifier: Apache-2.0
"""只读既有两路压缩CPU产物：按GM参数追踪访存，核对真实count依赖。

不再次编译、不使用NPU。TWAIT/CPP顺序和写回存在性只能证明生成代码保留了
源码协议，不能代替A3缓存一致性、设备精度或图重放验证。
"""

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


def require_access(role, name, operation):
    found = role["accesses"].get(name, {}).get(operation, [])
    assert found, (name, operation, "实际GM访存缺失")
    return found


def bounded(role, name, operation, first, last):
    rows = require_access(role, name, operation)
    assert all(first < item["line"] < last for item in rows), (name, operation, first, last, rows)


def closing_brace(body, opening):
    depth, offset = 1, opening + 1
    while depth:
        depth += (body[offset] == "{") - (body[offset] == "}")
        offset += 1
    return offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "不能覆盖历史证据"
    files = {"cube": args.artifact / "kernels/aic/cube.cpp", "vector": args.artifact / "kernels/aiv/vector.cpp"}
    texts = {name: path.read_text() for name, path in files.items()}
    roles = {name: role_body(texts[name], name) for name in files}
    cube, vector = roles["cube"], roles["vector"]
    cw, cp, vw, vp = cube["waits"], cube["publishes"], vector["waits"], vector["publishes"]
    assert len(cw) == 2 and len(cp) == 4, (cw, cp)
    assert len(vw) == 8 and len(vp) == 11, (vw, vp)
    for name in ("main_value", "main_score"):
        bounded(cube, name, "TSTORE", cp[0], cp[1])
        bounded(vector, name, "TLOAD", vw[0], vp[4])
    for name in ("idx_value", "idx_score"):
        bounded(cube, name, "TSTORE", cp[1], cp[2])
        bounded(vector, name, "TLOAD", vw[3], vp[7])
    bounded(cube, "idx_normed", "TLOAD", cw[-1], cp[-1])
    bounded(cube, "idx_had", "TSTORE", cw[-1], cp[-1])
    bounded(vector, "main_state", "TLOAD", vw[0], vp[3])
    bounded(vector, "main_state", "TSTORE", vw[1], vp[4])
    bounded(vector, "main_pooled", "TSTORE", vw[0], vp[3])
    bounded(vector, "main_pooled", "TLOAD", vw[2], vp[5])
    for name in ("main_normed", "main_debug", "main_cache"):
        bounded(vector, name, "TSTORE", vw[2], vp[5])
    bounded(vector, "idx_state", "TLOAD", vw[3], vp[6])
    bounded(vector, "idx_state", "TSTORE", vw[4], vp[7])
    bounded(vector, "idx_pooled", "TSTORE", vw[3], vp[6])
    bounded(vector, "idx_pooled", "TLOAD", vw[5], vp[8])
    bounded(vector, "idx_normed", "TSTORE", vw[5], vp[8])
    bounded(vector, "idx_had", "TLOAD", vw[6], vp[9])
    for name in ("idx_scales", "idx_debug"):
        bounded(vector, name, "TSTORE", vw[6], vp[9])
    cache_stores = require_access(vector, "idx_cache", "TSTORE")
    assert len(cache_stores) == 2
    assert vw[6] < cache_stores[0]["line"] < vp[9]
    assert vw[7] < cache_stores[1]["line"] < vp[10]
    # 两个独立metadata扫描及其发布均位于同一worker==0条件内。
    body = vector["body"]
    zero_match = re.search(r"const int64_t (v\d+) = 0;\n  // pto: %c63_index", body)
    if not zero_match:
        zero_variables = re.findall(r"const int64_t (v\d+) = 0;", body)
    else:
        zero_variables = [zero_match[1]]
    guard = None
    for zero in zero_variables:
        guard = re.search(rf"bool (v\d+) = (v\d+) == {zero};\n  if \(\1\) {{", body)
        if guard:
            break
    assert guard, "metadata必须被单worker guard覆盖"
    worker = guard[2]
    worker_definition = re.search(rf"int64_t {worker} = ([^;]+);", body)[0]
    assert "v76" in worker_definition and "v78" in worker_definition
    assert vector["arguments"]["v76"] == "__pypto_spmd_block_idx"
    assert vector["arguments"]["v78"] == "__pypto_spmd_subblock_idx"
    assert "v105" in worker_definition and "const int64_t v105 = 2;" in body
    opening = body.index("{", guard.start())
    closing = closing_brace(body, opening)
    guard_first = vector["base_line"] + body.count("\n", 0, opening) + 1
    guard_last = vector["base_line"] + body.count("\n", 0, closing) + 1
    for name in ("main_offsets", "main_count", "idx_offsets", "idx_count"):
        bounded(vector, name, "scalar_store", guard_first, guard_last)
        assert not cube["accesses"].get(name, {}).get("scalar_store")
    assert guard_first < vp[1] < vp[2] < guard_last
    # 找到phase4、producer24的确切signal地址，不能用任意一个TWAIT代替。
    count_order = {}
    for name, role in roles.items():
        b = role["body"]
        constant = re.search(r"unsigned (v\d+) = 39936;", b)
        assert constant, "(4*72+24)*128的idxmeta信号槽缺失"
        view = re.search(
            rf"GlobalTensor<[^\n]+> (v\d+) = GlobalTensor<[^\n]+>\(v\d+ \+ {constant[1]},", b[constant.end() :]
        )
        assert view
        wait = re.search(rf"pto::comm::TWAIT\({view[1]}, v\d+, v\d+\);", b)
        assert wait
        wait_line = role["base_line"] + b.count("\n", 0, wait.start()) + 1
        loads = require_access(role, "idx_count", "scalar_load")
        assert len(loads) == 1 and wait_line < loads[0]["line"]
        load_local = loads[0]["line"] - role["base_line"]
        between = "\n".join(b.splitlines()[wait_line - role["base_line"] : load_local - 1])
        assert "ENTIRE_DATA_CACHE" in between and "dsb(DSB_DDR);" in between
        count_order[name] = {
            "signal_offset_int32": 39936,
            "metadata_wait_line": wait_line,
            "count_load": loads[0],
            "post_wait_dcci_and_dsb_before_count_load": True,
        }
    # Scalar发布缓存维护存在，先写两个count，再publish两个独立phase。
    count_writes = [
        row["line"] for name in ("main_count", "idx_count") for row in require_access(vector, name, "scalar_store")
    ]
    publish_gap = "\n".join(texts["vector"].splitlines()[max(count_writes) : vp[1] - 1])
    assert "ENTIRE_DATA_CACHE" in publish_gap and "dsb(DSB_DDR);" in publish_gap
    # MIX两半的Tensor参数顺序一致；48个Tensor+27个动态维度+2/3个worker索引。
    c_args, v_args = list(cube["arguments"].values()), list(vector["arguments"].values())
    assert c_args[:75] == v_args[:75]
    assert len(c_args) == 77 and len(v_args) == 78
    assert len([arg for arg in c_args if arg.startswith("COMP_PIPE_")]) == 27
    assert len([arg for arg in c_args if "__ssa_" in arg]) == 48
    result = {
        "passed": True,
        "artifact": str(args.artifact.resolve()),
        "command": "python audit_compiled.py --artifact cpu_v2/kernel --output <new_json>",
        "scope": "既有CPU产物的CPP语义/ABI审计；不重新编译、不执行设备",
        "shared_tensor_arguments": 48,
        "dynamic_descriptor_dimensions": 27,
        "incore_argument_counts": {"cube": 77, "vector": 78},
        "signal_wait_counts": {"cube": len(cw), "vector": len(vw)},
        "signal_notify_counts": {"cube": len(cp), "vector": len(vp)},
        "metadata_single_writer": {
            "worker_expression": worker_definition,
            "guard": guard[0],
            "guard_lines": [guard_first, guard_last],
            "aic_metadata_write_absent": True,
        },
        "metadata_publish_dcci_dsb_after_scalar_writes": True,
        "device_count_order": count_order,
        "roles": {
            name: {key: value for key, value in role.items() if key not in ("body", "base_line")}
            for name, role in roles.items()
        },
        "verified_order": [
            "两路project真实GM写回→发布→对应pool读取",
            "所有pool历史读取owner完成→state commit真实GM写回",
            "主pool→主RMS/RoPE→主cache真实写回",
            "idxnorm真实compact GM写回→发布→Hadamard读取→真实GM写回",
            "Hadamard等待→量化/key写回→全quant owner等待→单worker scale读改写",
            "仅AIV0写metadata/count→DCCI+DSB→发布；AIC与AIV等待phase4→缓存维护→load count",
        ],
        "device_execution": False,
        "hardware_cache_visibility_proven": False,
        "device_precision_validated": False,
        "performance_measured": False,
        "limits": [
            "CPP中的等待/访存顺序不等于已经实测A3跨核缓存可见性",
            "无运行时容量/地址/Native slot唯一性检查，使用者必须满足探针头部合同",
            "未用返回SSA警告仍存在，但对应原GM参数的实际访存已保留",
            "ScalarWriteLineShared静态警告保留，实际metadata仅block_idx*2+subblock_idx==0写入",
            "77/78是CPP本体形参（48 Tensor+27 descriptor维度+2/3 worker），不是77/78个dispatch槽；未执行运行时",
        ],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "gm_writes_and_count_wait": True, "dynamic_dims": 27}))


if __name__ == "__main__":
    main()
