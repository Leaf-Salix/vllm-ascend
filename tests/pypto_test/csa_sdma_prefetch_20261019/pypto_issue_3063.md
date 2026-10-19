### 问题

`pl.cast(flag, pl.INT32)` 的 Scalar BOOL → INT32 被降成 `arith.extsi i1 to i32`，使 true 得到 **-1** 而非 1。以 `cast_result == 1` 判断真实完成状态会错误跳过后续工作。

在配套 SDMA 测试中，C++ 直接读回 `event.Wait(session)` 为 true，但上述 cast 后的判定为 false。这个现象已与多 AIV SDMA 通道问题分离：使用 if/else 赋 INT32 的 0/1 后，单发起者 eager 和 ACLGraph 重放通过。本 issue 只讨论 Scalar 转换，不要求 SDMA 或 NPU 才能复现。

### 测试版本与范围

本地基于 CSA 复现提交 `2dfa7f120ba5e577566046d920845aa241bfa7bb` 迁入 #3001，得到 `56cca2c408dcca98adde5284558bfcc305fb846c`，另有 GM view / NZ 元数据本地修改。相关 `src/codegen/pto/pto_scalar_expr_codegen.cpp` 没有本轮修改。**未声称已在干净 main 上真机复现，也不认定由 #3001 新引入。**

用真实 PyPTO/PTOAS 生成标量 C++，仅移除设备前缀和无关 fence 实现，用宿主 C++ 编译执行；不改生成的算术或控制流。结果：`false -> 0, true -> -1`。这与 NPU 上的分支诊断一致。

### 生成结果

```mlir
%0 = arith.extsi %arg1 : i1 to i32
```

PTOAS 合法地按符号扩展生成：

```cpp
AICORE void bool_cast(__gm__ int32_t* out, bool flag) {
    int32_t value = 0 - (int32_t)flag;
    out[0] = value;
}
```

上面 C++ 仅缩短变量名；算术与实际产物相同。PTOAS 遵从输入 IR，并非本次误选符号扩展的位置。

### 完整 CPU 复现

保存为 `repro_bool_cast.py`，在已安装 PyPTO / PTOAS 的环境执行 `python repro_bool_cast.py --output /tmp/bool_cast_new_dir`。输出目录需不存在。

```python
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
```

实测报告：

```json
{"actual_false_true": [0, -1], "expected_false_true": [0, 1], "passed": false, "device_execution": false}
```

### 修复建议

`PTOCodegen::EmitCast` 目前通过 `src_dtype.IsUnsignedInt()` 决定 extui/extsi；BOOL 不属于 unsigned integer，因而落入 extsi。

建议在 Boolean → 整数扩宽时明确采用零扩展，或选择常量 0/1：

```cpp
const bool src_is_bool = src_dtype == DataType::BOOL;
// 仅示意整数扩宽分支，不改变普通有符号整数的语义。
mlir_op = (src_is_uint || src_is_bool) ? "arith.extui" : "arith.extsi";
```

建议覆盖 false/true → INT32/INT64/UINT32；另核查 BOOL → INDEX 和浮点分支是否需要同样的 0/1 语义。以上扩展方向尚未逐一验证，不把它们列为本次实测结果。需要保留普通 signed 扩宽和正常位转换的既有行为。

没有提交修改 PR，没有添加标签；本次给出问题、复现和建议。
