# SPDX-License-Identifier: Apache-2.0
"""CPU最小复现：正数补偿模被Simplify去补偿后，PTO仍使用有符号rem。

输出容量必须至少192个INT32；此脚本只编译不运行设备。两种写法均计算
每组8项在24个worker下的首项：原写法先补24再取模，修订写法使用始终
非负的整除分子。保存pass IR、PTO及C++，便于组件维护者独立核对。
"""

import argparse
import json
import subprocess
from pathlib import Path

import pypto.language as pl
from pypto.runtime import CompileOptions

ROWS = pl.dynamic("OWNER_REPRO_ROWS")


def verify_generated_cpp_on_host(directory):
    """只替换设备修饰符与缓存intrinsic，保留生成的Scalar函数体原文作宿主验证。"""
    code = """#include <cstdint>
#include <cstdio>
#define AICORE
#define __gm__
#define __DAV_VEC__ 1
#define set_mask_norm(...) ((void)0)
#define set_vector_mask(...) ((void)0)
#define pipe_barrier(...) ((void)0)
#define dcci(...) ((void)0)
#define dsb(...) ((void)0)
#define ptoas_auto_sync_tail(...) ((void)0)
"""
    for name in ("compensated_mod", "nonnegative_div"):
        source = next((directory / name / "ptoas").glob("*.cpp")).read_text()
        body = source[source.index("AICORE void owner_core(") :]
        code += f"namespace {name} {{\n" + body + "\n}\n"
    code += r"""
int main() {
    int32_t old_values[192] = {}, new_values[192] = {};
    compensated_mod::owner_core(old_values, 192);
    nonnegative_div::owner_core(new_values, 192);
    int old_errors = 0, new_errors = 0;
    for (int g = 0; g < 8; ++g) {
        for (int w = 0; w < 24; ++w) {
            int expected = g * 8;
            while (expected % 24 != w) ++expected;
            old_errors += old_values[g * 24 + w] != expected;
            new_errors += new_values[g * 24 + w] != expected;
        }
    }
    std::printf("{\"cases\":192,\"compensated_mod_mismatches\":%d,"
                "\"nonnegative_div_mismatches\":%d,\"example_old\":%d,\"example_new\":%d}\n",
                old_errors, new_errors, old_values[24], new_values[24]);
    return new_errors != 0;
}
"""
    source = directory / "generated_scalar_host.cpp"
    source.write_text(code)
    binary = directory / "generated_scalar_host"
    subprocess.run(["c++", "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    result = subprocess.run([str(binary.resolve())], check=True, text=True, capture_output=True)
    evidence = json.loads(result.stdout)
    evidence["scope"] = "宿主执行生成Scalar算术；设备修饰符及缓存intrinsic置空，未运行NPU"
    return evidence


@pl.jit.incore
def owner_core(output: pl.InOut[pl.Tensor[[ROWS], pl.INT32]], USE_CEIL: pl.constexpr):
    for group in pl.range(8):
        group_first = group * 8
        for worker in pl.range(24):
            if USE_CEIL:
                first = worker + ((group_first + 23 - worker) // 24) * 24
            else:
                first = group_first + (worker + 24 - group_first % 24) % 24
            pl.write(output, [group * 24 + worker], pl.cast(first, pl.INT32))
    return output


@pl.jit
def owner_entry(output: pl.InOut[pl.Tensor[[ROWS], pl.INT32]], USE_CEIL: pl.constexpr):
    with pl.spmd(1):
        output = owner_core(output, USE_CEIL)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(device_execution=False, variants={}, signed_remainder_example={})
    for name, ceil in (("compensated_mod", False), ("nonnegative_div", True)):
        directory = args.output / name
        owner_entry._resolve_kernel_artifact(
            (),
            {"USE_CEIL": ceil, "config": CompileOptions(platform="a2a3", output_dir=str(directory), dump_passes=True)},
            allow_signature_mode=True,
        )
        ir = next((directory / "ptoas").glob("*.pto")).read_text()
        simplified = (directory / "passes_dump/05_after_Simplify.py").read_text()
        cpp = next((directory / "ptoas").glob("*.cpp")).read_text()
        report["variants"][name] = dict(
            simplified_assignments=[line.strip() for line in simplified.splitlines() if "first" in line],
            signed_remainder_sites=ir.count("arith.remsi"),
            cpp_remainders=[line.strip() for line in cpp.splitlines() if "%" in line],
            artifact=str(directory),
        )
    # C/C++截断余数，数值范围很小，整数截断除法可直接构造。
    group_first, worker = 8, 0
    numerator = worker - group_first % 24
    signed_rem = numerator - int(numerator / 24) * 24
    report["signed_remainder_example"] = dict(
        group_first=group_first,
        worker=worker,
        source_positive_mod=group_first + (worker + 24 - group_first % 24) % 24,
        simplified_with_signed_rem=group_first + signed_rem,
        nonnegative_div=worker + ((group_first + 23 - worker) // 24) * 24,
    )
    report["host_generated_scalar"] = verify_generated_cpp_on_host(args.output)
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
