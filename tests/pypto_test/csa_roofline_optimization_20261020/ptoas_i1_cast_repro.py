# SPDX-License-Identifier: Apache-2.0
"""纯PTO验证有符号i1转换；原生成算术在Host执行，并额外检查A3 CCEC。"""

import argparse
import ctypes
import json
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[4]
    ptoas_tree = root / ".worktrees/ptoas-csa-pair-compact-20261020"
    compiler = ptoas_tree / "build/bin/ptoas"
    reports = []
    for kind, dtype in (("index", "i32"), ("float", "f32")):
        conversions = (
            "%a = arith.index_cast %true : i1 to index\n"
            "    %b = arith.index_cast %condition : i1 to index\n"
            "    %constant_value = arith.index_cast %a : index to i32\n"
            "    %runtime_value = arith.index_cast %b : index to i32"
            if kind == "index" else
            "%constant_value = arith.sitofp %true : i1 to f32\n"
            "    %runtime_value = arith.sitofp %condition : i1 to f32"
        )
        source = f'''module attributes {{pto.target_arch = "a2a3"}} {{
  func.func @kernel_entry(%src: !pto.ptr<i32>, %out: !pto.ptr<{dtype}>) attributes {{pto.kernel_kind = #pto.kernel_kind<vector>}} {{
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %zero_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %input = pto.load %src[%zero] : !pto.ptr<i32> -> i32
    %condition = arith.cmpi sgt, %input, %zero_i32 : i32
    {conversions}
    pto.store %constant_value, %out[%zero] : !pto.ptr<{dtype}>, {dtype}
    pto.store %runtime_value, %out[%one] : !pto.ptr<{dtype}>, {dtype}
    return
  }}
}}
'''
        pto = args.output / f"{kind}.pto"
        cpp = args.output / f"{kind}.cpp"
        pto.write_text(source)
        command = [str(compiler), "--pto-arch=a3", "--pto-level=level2", str(pto), "-o", str(cpp)]
        result = subprocess.run(command, capture_output=True, text=True)
        (args.output / f"{kind}_ptoas.log").write_text(result.stdout + result.stderr)
        row = dict(kind=kind, ptoas_returncode=result.returncode, command=command,
                   expected_signed_i1_true=[-1, -1], device_execution=False)
        if result.returncode:
            row["error"] = result.stderr
            reports.append(row)
            continue
        text = cpp.read_text()
        match = re.search(r"AICORE void kernel_entry\(", text)
        assert match, text
        body = text[match.start():].replace("AICORE ", 'extern "C" ', 1)
        host = args.output / f"{kind}_host.cpp"
        host.write_text('''#include <cstdint>
#define __gm__
#define __DAV_VEC__ 1
#define PIPE_ALL 0
#define DSB_DDR 0
enum class cache_line_t {ENTIRE_DATA_CACHE, SINGLE_CACHE_LINE};
using mem_dsb_t = int;
inline void set_mask_norm() {}
inline void set_vector_mask(int64_t, int64_t) {}
inline void pipe_barrier(int) {}
inline void dcci(void*, cache_line_t) {}
inline void dsb(int) {}
''' + body)
        library = args.output / f"{kind}_host.so"
        subprocess.run([shutil.which("c++"), "-shared", "-fPIC", "-std=c++17", str(host), "-o", str(library)], check=True)
        loaded = ctypes.CDLL(str(library.resolve()))
        function = loaded.kernel_entry
        scalar = ctypes.c_int32 if kind == "index" else ctypes.c_float
        function.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(scalar)]
        function.restype = None
        row["cases"] = []
        for value in (1, 0):
            src, out = ctypes.c_int32(value), (scalar * 2)(777, 777)
            function(ctypes.byref(src), out)
            assert list(out) != [777, 777], "Host设备宏未启用，不能将未执行当作数值结果"
            row["cases"].append(dict(input=value, actual=list(out), expected=[-1, -1 if value > 0 else 0]))
        from pypto.runtime import ensure_pto_isa_root
        from simpler_setup.kernel_compiler import KernelCompiler
        try:
            binary = KernelCompiler("a2a3").compile_incore(str(cpp), core_type="aiv", pto_isa_root=ensure_pto_isa_root())
            (args.output / f"{kind}.bin").write_bytes(binary)
            row["ccec_passed"] = True
        except Exception as exc:
            message = str(exc)
            (args.output / f"{kind}_ccec_error.log").write_text(message)
            row.update(ccec_passed=False, ccec_errors=[line for line in message.splitlines() if "error:" in line])
        reports.append(row)
    report = dict(ptoas_commit=subprocess.check_output(["git", "-C", str(ptoas_tree), "rev-parse", "HEAD"], text=True).strip(),
                  pypto_used_to_generate_ir=False, host_arithmetic_rewritten=False, results=reports)
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
