# SPDX-License-Identifier: Apache-2.0
"""只读已生成的成对MIX CPP；沿真实GM别名审计写回、动态ABI和事件。"""

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
    kernel = args.artifact.resolve()
    cpp = (kernel / "kernels/aiv/vector.cpp").read_text()
    cube_cpp = (kernel / "kernels/aic/cube.cpp").read_text()
    vector = role_body(cpp, "vector")
    cube = role_body(cube_cpp, "cube")
    expected = {
        "cube": {"score_transfer": 1, "pv_transfer": 4},
        "vector": {
            "kv_transfer": 1,
            "probability_transfer": 1,
            "mi_transfer": 1,
            "li_transfer": 1,
            "o_packed_heads": 4,
        },
    }
    results = {}
    for role, data in [("cube", cube), ("vector", vector)]:
        access = data["accesses"]
        for tensor, count in expected[role].items():
            assert len(access[tensor]["TSTORE"]) == count, (role, tensor)
        constants = {
            name: int(value) for name, value in re.findall(r"const int(?:32|64)_t (v\d+) = (-?\d+);", data["body"])
        }
        for target, origin in re.findall(r"int32_t (v\d+) = \(int32_t\) (v\d+);", data["body"]):
            if origin in constants:
                constants[target] = constants[origin]
        event_msgs = {
            target: constants[event]
            for target, event in re.findall(r"uint16_t (v\d+) = getFFTSMsg\(FFTS_MODE_VAL, (v\d+)\);", data["body"])
        }
        events = []
        for offset, line in enumerate(data["body"].splitlines(), 1):
            wait = re.search(r"wait_flag_dev\((v\d+)\)", line)
            notify = re.search(r"ffts_cross_core_sync\((PIPE_\w+), (v\d+)\)", line)
            if wait:
                events.append({"op": "wait", "event": constants[wait[1]], "line": data["base_line"] + offset})
            if notify:
                events.append(
                    {
                        "op": "notify",
                        "event": event_msgs[notify[2]],
                        "pipe": notify[1],
                        "line": data["base_line"] + offset,
                    }
                )
        expected_events = (
            [("wait", 0), ("notify", 1), ("wait", 2), ("notify", 3)]
            if role == "cube"
            else [("wait", 1), ("notify", 0), ("wait", 1), ("notify", 2), ("notify", 0), ("wait", 3)]
        )
        assert [(e["op"], e["event"]) for e in events] == expected_events
        data.pop("body")
        results[role] = {**data, "events": events}
    dims = re.findall(r"// Extract dynamic dim: (\w+)\n\s*int64_t \w+ = static_cast<int64_t>\(([^;]+)\);", cpp)
    assert len(dims) == 13 and all("->shapes[" in expr for _, expr in dims)
    assert {
        "CSA_GEOMETRY_QUERY_BOUNDS",
        "CSA_GEOMETRY_REQUESTS",
        "CSA_ATTN_STREAM_PACK_ROWS",
        "CSA_ATTN_PLAN_T",
        "CSA_ATTN_STREAM_Q_ROWS",
    } <= {n for n, _ in dims}
    vector_body = role_body(cpp, "vector")["body"]
    constants = {name: int(value) for name, value in re.findall(r"const int64_t (v\d+) = (-?\d+);", vector_body)}
    packed_formal = next(k for k, v in vector["arguments"].items() if v == "CSA_ATTN_STREAM_PACK_ROWS")
    pack_div = [
        {"statement": m[0], "divisor": constants[m[2]]}
        for m in re.finditer(r"int64_t (v\d+) = " + packed_formal + r" / (v\d+);", vector_body)
        if m[2] in constants
    ]
    assert any(item["divisor"] == 8 for item in pack_div)
    meta = json.loads((kernel / "compiled_meta.json").read_text())["params"]
    assert len(meta) == 22
    assert next(x for x in meta if x["name"] == "ffts_workspace")["shape"] == [256]
    assert next(x for x in meta if x["name"] == "o_packed_heads")["shape"] == [-1, 4096]
    orchestration = (kernel / "orchestration/entry.cpp").read_text()
    assert len(re.findall(r"\brt_submit_\w+\(", orchestration)) == 1
    assert "MixedKernels mixed_0 = {0, 1, 1}" in orchestration and "set_block_num(24)" in orchestration
    raw = (kernel / "ptoas/entry_cluster_0.cpp").read_text()
    assert raw.count("set_ffts_base_addr(") == 2
    assert "set_ffts_base_addr(" not in cpp and "set_ffts_base_addr(" not in cube_cpp
    report = {
        "passed": True,
        "artifact": str(kernel),
        "roles": results,
        "dynamic_dimensions": [{"name": n, "descriptor_read": v} for n, v in dims],
        "dynamic_dimension_count": len(dims),
        "packed_stride_division": pack_div,
        "public_abi_count": len(meta),
        "single_mix": True,
        "aic": 24,
        "aiv": 48,
        "ffts_setup": {
            "raw_ptoas_setup_calls": 2,
            "final_wrapper_setup_calls": 0,
            "reason": (
                "PyPTO _ptoas_preprocess.py removes set_ffts_base_addr; Simpler runtime kernel owns hardware FFTS base"
            ),
            "pypto_source": (
                "/data/pyptouser/qinchuanyu/pto-eager/.worktrees/"
                "pypto-csa-static-fixes-20261016/python/pypto/backend/_ptoas_preprocess.py:123"
            ),
            "simpler_source": (
                "/data/pyptouser/qinchuanyu/pto-eager/.worktrees/"
                "simpler-csa-core-args-256-20261016/src/a2a3/platform/onboard/aicore/kernel.cpp:99"
            ),
        },
        "device_sync_validated": False,
        "device_precision_validated": False,
        "performance_measured": False,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": True, "dynamic_dimension_count": len(dims), "store_counts": expected, "event_calls": 10},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
