# SPDX-License-Identifier: Apache-2.0
"""仅在 CPU 执行原样生成的标量 C++，复现 BOOL cast 的数值语义。"""

import argparse
import ctypes
import json
from pathlib import Path
import re
import shutil
import subprocess

from pypto import DataType, backend, codegen, ir
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.op import tensor as tensor_ops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    span = ir.Span.unknown()
    flag = ir.Var("flag", ir.ScalarType(DataType.BOOL), span)
    output_type = ir.TensorType([1], DataType.INT32)
    out = ir.Var("out", output_type, span)
    written = ir.Var("written", output_type, span)
    casted = ir.Cast(flag, DataType.INT32, span)
    body = ir.SeqStmts([
        ir.AssignStmt(written, tensor_ops.write(out, [0], casted), span),
        ir.ReturnStmt([written], span),
    ], span)
    function = ir.Function("bool_cast", [out, flag], [output_type], body,
                           span, ir.FunctionType.InCore)
    generated = codegen.PTOCodegen().generate(ir.Program([function], "repro", span))
    mlir = generated if isinstance(generated, str) else "".join(generated.values())
    pto, cpp = args.output / "bool_cast.pto", args.output / "bool_cast.cpp"
    pto.write_text(mlir)
    subprocess.run([str(find_ptoas_binary()), str(pto), "-o", str(cpp)], check=True)
    source = cpp.read_text()
    match = re.search(r"AICORE \w+ bool_cast\(", source)
    assert match, source
    body_cpp = source[match.start():].replace("AICORE ", 'extern "C" ', 1)
    host = args.output / "host.cpp"
    host.write_text("""#include <cstdint>
#define __gm__
#define PIPE_ALL 0
enum class cache_line_t {ENTIRE_DATA_CACHE, SINGLE_CACHE_LINE};
using mem_dsb_t = int;
inline void pipe_barrier(int) {}
inline void dcci(void*, cache_line_t) {}
inline void dsb(mem_dsb_t) {}
""" + body_cpp)
    library = args.output / "host.so"
    subprocess.run([shutil.which("c++"), "-std=c++17", "-shared", "-fPIC",
                    str(host), "-o", str(library)], check=True)
    loaded = ctypes.CDLL(str(library.resolve()))
    fn = loaded.bool_cast
    fn.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_bool]
    fn.restype = None
    actual = []
    for flag_value in (False, True):
        value = ctypes.c_int32()
        fn(ctypes.byref(value), flag_value)
        actual.append(value.value)
    report = {"actual_false_true": actual, "expected_false_true": [0, 1],
              "passed": actual == [0, 1], "device_execution": False,
              "cast_ops": [line for line in mlir.splitlines() if "arith.ext" in line]}
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
