"""比较两轮已完成的私有 CPU 产物与同控制旧 N128/N256 真机编译产物。

只分析现成文本；不重编基线、不执行设备、不复制大二进制、不生成 hash。
核对编译ABI、完整 PTO 分配地址和原始 C++ 调用/同步顺序，另外保存精确文本
是否相同。位置字符串不属于数学；除此之外发现差异需保留，不隐去不匹配。
"""

import collections
import difflib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
CANONICAL = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1")
JOB = CANONICAL / "tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/pv_n256_h131072_b16_v1"
OLD_NAMES = {128: "decode_csa_spmd_q1m9wnpj", 256: "decode_csa_spmd_zpf_6y0x"}
INPUTS = set()


def read(path):
    INPUTS.add(path)
    return path.read_text()


def strip_pto_locations(text):
    return re.sub(r' loc\("[^"\n]*":\d+:\d+\)', "", text)


def collect(kernel):
    pto = read(kernel / "ptoas/dynamic_csa_full.pto")
    cpp = read(kernel / "ptoas/dynamic_csa_full.cpp")
    allocations, upper = {}, {}
    role = None
    for line in pto.splitlines():
        fn = re.search(r"func.func @(\w+)", line)
        if fn:
            role = fn[1]
        if "pto.alloc_tile" in line:
            match = re.search(r"addr = %c(\d+)_i64.*?loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", line)
            assert match, line
            address, space, dtype, rows, cols = match.groups()
            record = dict(address=int(address), space=space, dtype=dtype, rows=int(rows), cols=int(cols))
            allocations.setdefault(role, []).append(record)
            end = int(address) + int(rows) * int(cols) * int(re.search(r"\d+", dtype)[0]) // 8
            key = role + "/" + space
            upper[key] = max(upper.get(key, 0), end)
    role_stats = {}
    for role in ("cube_schedule", "vector_schedule"):
        start = cpp.index("AICORE void " + role + "(")
        end = cpp.find("AICORE void ", start + 20)
        body = cpp[start:end if end >= 0 else len(cpp)]
        # 保存实际调用行，包含事件号/pipe/Tile实参，不能只比较调用次数。
        calls = [line.strip() for line in body.splitlines() if re.match(
            r"\s*(?:T[A-Z]\w*|wait_flag_dev|ffts_cross_core_sync|wait_flag|set_flag|pipe_barrier|dsb|dcci)\(", line)]
        names = [line.split("(", 1)[0] for line in calls]
        role_stats[role] = dict(calls=calls, call_counts=dict(collections.Counter(names)),
            if_sites=len(re.findall(r"\bif \(", body)))
    return dict(pto=pto, cpp=cpp, allocations=allocations, upper=upper, roles=role_stats,
        meta=json.loads(read(kernel / "compiled_meta.json")))


def main():
    files = sorted((HERE / "source_before").glob("*.py"))
    ref_source = JOB / "source_matrix/reference_source"
    assert len(files) == 33
    assert all(read(path) == read(ref_source / path.name) for path in files)
    matrix = json.loads(read(JOB / "source_matrix/result.json"))
    comparisons = {}
    for cols in (128, 256):
        old_path = JOB / "build_output" / OLD_NAMES[cols]
        new_path = HERE / f"cpu_n{cols}_v1/kernel"
        result = json.loads(read(HERE / f"cpu_n{cols}_v1/result.json"))
        assert result["passed"] and result["single_mix"] and result["sync_start"]
        assert len(result["symbolic_dimensions"]) == 58
        expected = matrix["policies"]["spmd_reference" if cols == 128 else "spmd_candidate"]
        assert result["controls"] == {**expected, "PV_N_TILE": cols}
        old, new = collect(old_path), collect(new_path)
        comparison = dict(old_artifact=str(old_path), new_artifact=str(new_path),
            same_existing_controls=True, tensor_root_count=len(new["meta"]["params"]),
            symbolic_dimensions=len(result["symbolic_dimensions"]), single_mix=result["single_mix"],
            sync_start=result["sync_start"], sample_shapes_supplied=result["sample_shapes_supplied"],
            compiled_meta_equal=old["meta"] == new["meta"],
            allocation_sequence_equal=old["allocations"] == new["allocations"],
            address_upper_bounds_equal=old["upper"] == new["upper"],
            old_address_upper_bounds=old["upper"], new_address_upper_bounds=new["upper"],
            raw_cpp_byte_equal=old["cpp"] == new["cpp"],
            pto_equal_except_source_locations=strip_pto_locations(old["pto"]) == strip_pto_locations(new["pto"]),
            cpp_roles={role: dict(call_sequence_equal=old["roles"][role]["calls"] == new["roles"][role]["calls"],
                if_sites_equal=old["roles"][role]["if_sites"] == new["roles"][role]["if_sites"],
                old_call_counts=old["roles"][role]["call_counts"],
            new_call_counts=new["roles"][role]["call_counts"]) for role in old["roles"]})
        # 两个进程的inline计数起点不同，只体现在pto调试注释；保留原始不等事实，
        # 另核对去掉这种注释后的完整C++逐字一致，不放宽业务语句或变量名。
        strip_debug_comments = lambda text: re.sub(r"(?m)^\s*// pto:.*\n", "", text)
        comparison["cpp_equal_except_pto_debug_comments"] = (
            strip_debug_comments(old["cpp"]) == strip_debug_comments(new["cpp"])
        )
        assert comparison["tensor_root_count"] == 130
        assert comparison["compiled_meta_equal"]
        assert comparison["allocation_sequence_equal"] and comparison["address_upper_bounds_equal"]
        assert comparison["cpp_equal_except_pto_debug_comments"]
        assert all(value["call_sequence_equal"] and value["if_sites_equal"]
            for value in comparison["cpp_roles"].values())
        if not comparison["raw_cpp_byte_equal"]:
            delta = list(difflib.unified_diff(old["cpp"].splitlines(True), new["cpp"].splitlines(True),
                fromfile=str(old_path), tofile=str(new_path)))
            (HERE / f"cpp_n{cols}_difference.txt").write_text("".join(delta[:240]))
            comparison["cpp_diff_total_lines"] = len(delta)
        comparisons[str(cols)] = comparison
    report = dict(scope="仅接线等价性；非新增设备或性能结果", comparisons=comparisons,
        inputs=[dict(path=str(p), bytes=p.stat().st_size) for p in sorted(INPUTS)])
    (HERE / "generated_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: {k: v for k, v in value.items() if k not in (
        "cpp_roles", "old_address_upper_bounds", "new_address_upper_bounds")} for key, value in comparisons.items()}, indent=2))


if __name__ == "__main__":
    main()
