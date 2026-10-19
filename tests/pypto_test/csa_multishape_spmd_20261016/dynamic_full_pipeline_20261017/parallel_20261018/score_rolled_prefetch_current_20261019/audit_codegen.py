# SPDX-License-Identifier: Apache-2.0
"""复核两面板循环的真实分配、动态循环及关闭路径；不执行NPU、不做hash。"""
import json
import re
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    cpu = here / "cpu_v1"
    result = json.loads((cpu / "result.json").read_text())
    ir = (cpu / "kernel/ptoas/dynamic_csa_full.pto").read_text()
    names = (
        "key_pool|local_pool|score_pool|odd_key|even_key|previous_even|previous_odd|"
        "prime_key|prime_qk|odd_qk|even_qk|final_qk|previous_ws|odd_ws|final_ws|"
        "before_last_ws|final_right|before_last|final_key"
    )
    allocations = []
    for line in ir.splitlines():
        match = re.search(
            r"(%\w+) = pto.alloc_tile addr = %c(\d+)_i64.*"
            r"!pto.tile_buf<loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", line,
        )
        if match and re.search(r"%(?:" + names + ")_", match[1]):
            allocations.append(dict(name=match[1], address=int(match[2]), loc=match[3],
                                    dtype=match[4], rows=int(match[5]), cols=int(match[6])))
    key_addresses = {r["address"] for r in allocations
                     if re.match(r"%(key_pool|local_pool)", r["name"])}
    score_addresses = {r["address"] for r in allocations if r["name"].startswith("%score_pool")}
    loops = [line.strip() for line in ir.splitlines() if "scf.for %pair_inline" in line]
    # 当前生成物审计，不能把预期地址写成内存规划接口或覆盖新的分配结果。
    assert key_addresses == {163840} and score_addresses == {65536}
    assert len(loops) == 1 and "to %c7_index" in loops[0]
    report = dict(full_codegen_passed=result["full_codegen_passed"],
                  symbolic_axes=len(result["symbolic_dimensions"]), single_mix=result["single_mix"],
                  key_pool_addresses=sorted(key_addresses), score_pool_addresses=sorted(score_addresses),
                  runtime_pair_loops=loops, allocations=allocations,
                  scope="仅编译与分配审计，不替代设备功能和性能", device_validated=False)
    short = here / "cpu_short_v2"
    if (short / "result.json").exists():
        short_result = json.loads((short / "result.json").read_text())
        report["short_full_codegen_passed"] = short_result.get("full_codegen_passed", False)
        if report["short_full_codegen_passed"]:
            reference = here.parent / "mgather_physical_plan_retest_20261019/cpu_short_v1"
            report["short_cpp_equal_ignoring_comments"] = {}
            for name in ("aic/cube_schedule.cpp", "aiv/vector_schedule.cpp"):
                a = (short / "kernel/kernels" / name).read_text()
                b = (reference / "kernel/kernels" / name).read_text()
                report["short_cpp_equal_ignoring_comments"][name] = (
                    re.sub(r"//[^\n]*", "", a) == re.sub(r"//[^\n]*", "", b)
                )
    (here / "CODEGEN_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "allocations"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
