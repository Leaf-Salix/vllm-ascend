#!/usr/bin/env python3
"""从已验证的完整CSA构建恢复独立72核启动ABI；不修改数学函数。

真实Simpler wrapper从Tensor描述符取得132个core指针和56个动态维度，
另给block/lane编号。本脚本读取那段调用和shape读取语句，绑定捕获的物理
storage、偏移、动态维度。只恢复standalone缺少的FFTS与24-block启动。
原数学函数移除的四个set_ffts与真实PyPTO wrapper相同，逐函数token核对。
"""

import json
import re
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
CAPTURE = HERE / "build_output/capture_v2"
BUILD = HERE / "build_output/source_build"
FUNCTION = "dynamic_csa_full"


def function_parts(text, name):
    match = re.search(r"\bvoid\s+" + name + r"\(([^\n]+)\)\s*\{", text)
    assert match, name
    start = match.end() - 1
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return match.group(1).split(", "), text[start:index + 1]
    raise AssertionError(f"未找到完整函数: {name}")


def tokens(text):
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    return re.findall(r"\w+|[^\s\w]", text)


def prepare_source():
    result = json.loads((CAPTURE / "result.json").read_text())
    capture = json.loads((CAPTURE / "replay/capture.json").read_text())
    assert result["passed"] and result["stage"] == "complete" and capture["capture_complete"]
    builds = list((CAPTURE / "build_output").glob("decode_csa_spmd_*"))
    assert len(builds) == 1
    original_build = builds[0]
    raw_path = CAPTURE / "codegen_probe/modified.cpp"
    raw = raw_path.read_text()
    # 与真实runtime wrapper保持一致：FFTS来自runtime，不用业务GM数组作硬件base。
    body, removed = re.subn(r"(?m)^\s*set_ffts_base_addr\(\w+\);[^\n]*\n", "\n", raw)
    assert removed == 4
    wrappers = {
        "cube_schedule": original_build / "kernels/aic/cube_schedule.cpp",
        "vector_schedule": original_build / "kernels/aiv/vector_schedule.cpp",
    }
    matches = {}
    for name, path in wrappers.items():
        _, a = function_parts(body, name)
        _, b = function_parts(path.read_text(), name)
        matches[name] = tokens(a) == tokens(b)
        assert matches[name], f"{name}数学体与真机wrapper不一致"
    params, _ = function_parts(body, "cube_schedule")
    vector_params, _ = function_parts(body, "vector_schedule")
    assert vector_params[:-1] == params and len(params) == 190
    names = [param.rsplit(" ", 1)[-1].lstrip("*") for param in params]
    wrapper = wrappers["cube_schedule"].read_text()
    tail = wrapper[wrapper.rfind("// Forward to ptoas-generated function"):]
    arguments = re.search(r"cube_schedule\((.*?)\);", tail, re.S).group(1).split(", ")
    assert len(arguments) == len(params)
    dimension_sources = {
        symbol: (tensor, int(axis))
        for symbol, tensor, axis in re.findall(
            r"int64_t (CSA_FULL_\w+) = static_cast<int64_t>\((\w+)__ssa_v\d+_tensor->shapes\[(\d+)\]\);",
            wrapper,
        )
    }
    bindings = []
    for param, name, argument in zip(params, names, arguments):
        if "__gm__" in param:
            root = re.sub(r"__ssa_v\d+$", "", argument)
            extra = 0
            lookup = root
            if root in ("idx_key0", "idx_key64"):
                lookup = "idx_cache"
                extra = 64 if root == "idx_key64" else 0
            value = capture["roots"][lookup]
            bindings.append(dict(name=name, kind="pointer", root=root, storage=value["storage"],
                                 offset_bytes=value["offset_bytes"] + extra, declaration=param))
        elif argument in dimension_sources:
            root, axis = dimension_sources[argument]
            bindings.append(dict(name=name, kind="scalar", symbol=argument, root=root, axis=axis,
                                 value=capture["roots"][root]["shape"][axis], declaration=param))
        else:
            assert argument in ("__pypto_spmd_block_idx", "__pypto_spmd_block_num")
            bindings.append(dict(name=name, kind="core", symbol=argument,
                                 value=0 if argument.endswith("idx") else 24, declaration=param))
    assert sum(row["kind"] == "pointer" for row in bindings) == 132
    assert sum(row["kind"] == "scalar" for row in bindings) == 56
    core_idx, core_count = names[-2:]
    body += (
        f'\nextern "C" __global__ AICORE void {FUNCTION}({", ".join(params)}, uint64_t csa_ffts) {{\n'
        "  set_ffts_base_addr(csa_ffts);\n"
        f"  {core_idx} = static_cast<int32_t>(get_block_idx());\n"
        f"  {core_count} = 24;\n"
        "  SYNCALL<SyncCoreType::Mix>();\n"
        "#if defined(__DAV_CUBE__)\n"
        f"  cube_schedule({', '.join(names)});\n"
        "#elif defined(__DAV_VEC__)\n"
        f"  vector_schedule({', '.join(names)}, static_cast<int32_t>(get_subblockid()));\n"
        "#endif\n}\n"
    )
    out = BUILD / "ptoas"
    out.mkdir(parents=True, exist_ok=True)
    (out / (FUNCTION + ".cpp")).write_text(body)
    shutil.copy2(original_build / "ptoas/dynamic_csa_full.pto", out / (FUNCTION + ".pto"))
    report = dict(function=FUNCTION, source_build=str(original_build), source_cpp=str(raw_path),
                  capture=str(CAPTURE / "replay/capture.json"), math_tokens_match_device=matches,
                  removed_ffts_statements=removed, physical_launch_blocks=24,
                  physical_cores="24 AIC + 48 AIV", arguments=bindings,
                  entry_changes="FFTS由rtGetC2cCtrlAddr传入；hardware core/lane id；MIX起始会合对应require_sync_start",
                  math_changed=False, all_storage_aliases_preserved=True)
    (HERE / "WORKLOAD_BINDINGS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Standalone source ready: 132 pointers, 56 shape scalars, 24 groups; math matches actual device wrapper")
    return report


if __name__ == "__main__":
    prepare_source()
