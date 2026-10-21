## 问题与范围

同一个 `i1 true` 经有符号转换，常量路径与运行时路径结果不一致：

| PTO IR | 常量 true | 从 GM 读取正整数后比较所得 true | 预期 |
|---|---:|---:|---:|
| `arith.index_cast i1 -> index -> i32` | -1 | +1 | 两者均 -1 |
| `arith.sitofp i1 -> f32` | -1.0 | +1.0（宿主执行生成算术） | 两者均 -1.0 |

第二项生成的 `(float)bool` 在 A3 CCEC 另被拒绝：`cast between floating and unsigned integer variable is not allowed in aicore function`。**浮点项没有在 NPU 执行；不把 Host 结果称为真机结果。** 整数/INDEX 项在独立 PyPTO 探针的 A3 真机上也出现 literal=-1、runtime=+1。

本复现直接给 PTOAS 喂手写 PTO IR，不依赖 PyPTO 生成 IR，也不需要 CSA 权重或大算子。按 [MLIR arith.index_cast 的定义](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithindex_cast-arithindexcastop)，扩宽需要符号扩展；`sitofp` 同样按有符号整数解释输入。因此 i1 的置位值按有符号语义为 -1，常量折叠与运行时 lowering 应一致。

## 测试版本

- PTOAS 本地独立 worktree：`c04ecdf9ca29a26b3b6d026faf18c12df476d201`，编译参数 `--pto-arch=a3 --pto-level=level2`。
- 这是保留 CSA 所需修复的本地分支，不声称在最新干净 main 上完成验证，也不声称由最近某 PR 引入。
- 本轮没有修改 PTOAS。涉及的 `lib/PTO/Transforms/PTOToEmitC/Arith/ArithCast.cpp` 最后本地历史记录为 `e78f20e0`，没有本轮未提交改动。
- PyPTO 关联真机探针使用本地 `8c360367e0239779a93b495ecf7910e26a9f1fc6`；下面的纯 PTO 复现无需此版本。

## 最小复现一：INDEX

保存为 `index.pto`：

```mlir
module attributes {pto.target_arch = "a2a3"} {
  func.func @kernel_entry(%src: !pto.ptr<i32>, %out: !pto.ptr<i32>) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %zero_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %input = pto.load %src[%zero] : !pto.ptr<i32> -> i32
    %condition = arith.cmpi sgt, %input, %zero_i32 : i32
    %a = arith.index_cast %true : i1 to index
    %b = arith.index_cast %condition : i1 to index
    %constant_value = arith.index_cast %a : index to i32
    %runtime_value = arith.index_cast %b : index to i32
    pto.store %constant_value, %out[%zero] : !pto.ptr<i32>, i32
    pto.store %runtime_value, %out[%one] : !pto.ptr<i32>, i32
    return
  }
}
```

运行：

```bash
ptoas --pto-arch=a3 --pto-level=level2 index.pto -o index.cpp
```

实际生成的相关片段（变量名保留）：

```cpp
const int32_t v3 = -1;
int32_t v7 = (v1)[v4];
int32_t v8 = (int32_t) ((int64_t) v7 > v6);
(v2)[v4] = v3;
(v2)[v5] = v8;
```

`v6=0`，输入 `src[0]=1` 时输出为 `[-1, +1]`；输入 0 时 `[-1, 0]`。

## 最小复现二：FP32

保存为 `float.pto`：

```mlir
module attributes {pto.target_arch = "a2a3"} {
  func.func @kernel_entry(%src: !pto.ptr<i32>, %out: !pto.ptr<f32>) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %zero_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %input = pto.load %src[%zero] : !pto.ptr<i32> -> i32
    %condition = arith.cmpi sgt, %input, %zero_i32 : i32
    %constant_value = arith.sitofp %true : i1 to f32
    %runtime_value = arith.sitofp %condition : i1 to f32
    pto.store %constant_value, %out[%zero] : !pto.ptr<f32>, f32
    pto.store %runtime_value, %out[%one] : !pto.ptr<f32>, f32
    return
  }
}
```

运行同样的 `ptoas --pto-arch=a3 --pto-level=level2 float.pto -o float.cpp`。
实际生成：

```cpp
const float v3 = -1.0f;
int32_t v7 = (v1)[v4];
float v8 = (float) (v7 > v6);
(v2)[v4] = v3;
(v2)[v5] = v8;
```

可以将原生成函数放入普通 C++ 测试，仅去掉 `AICORE`/`__gm__` 前缀、定义 `__DAV_VEC__` 并空实现 fence/mask 初始化；算术保持不动。输入 1 时为 `[-1.0,+1.0]`，输入 0 时 `[-1.0,0.0]`。本地脚本以 777 初始化输出，检查设备宏已启用，避免把未执行当成结果。

对原始 `float.cpp` 做 A3 CCEC 编译则在上面 `float v8` 处失败；原始 `index.cpp` 通过。使用已有 Simpler/PyPTO 编译环境可执行：

```python
from pypto.runtime import ensure_pto_isa_root
from simpler_setup.kernel_compiler import KernelCompiler
KernelCompiler("a2a3").compile_incore(
    "float.cpp", core_type="aiv", pto_isa_root=ensure_pto_isa_root())
```

## 定位与建议

`ArithCast.cpp` 的 `ArithCastOPToEmitC` 对 `IndexCastOp` 直接使用 `emitc::CastOp`；`SIToFPOp` 经通用 `ArithCastToEmitC` 同样直接 cast。转换为 C++ bool 后已经丢掉 i1 的有符号位宽语义。相比之下，`ArithExtSIToEmitC` 已通过 `emitWidenedInt(..., isSigned=true)` 特判这一情况。

建议在有符号 i1 扩宽/转浮点时复用同类规范化，示意生成：

```cpp
int32_t signed_i1 = 0 - static_cast<int32_t>(condition); // false=0, true=-1
int64_t as_index = static_cast<int64_t>(signed_i1);
float as_float = static_cast<float>(signed_i1);
```

对于 `index_castui`/`uitofp` 则保留 0/1 语义；A3 bool 转浮点应先经过合法的整型载体，不能直接产生被 CCEC 拒绝的 `(float)bool`。建议同时覆盖常量/动态 false/true、signed/unsigned、INDEX/INT32/INT64/FP32，并保留既有普通 signed 扩宽行为。这里只提供修改方向，没有提交修改 PR。

## 与 PyPTO #3063 的边界

https://github.com/hw-native-sys/pypto/issues/3063 是前端 BOOL 转数字误选有符号指令的问题。PyPTO 的逻辑 BOOL 预期 0/1，应选择零扩展；PTOAS 则必须忠实执行 IR 已经指定的 signed/unsigned 语义。两个问题应分别修复，不能靠当前 INDEX 动态路径错误返回 +1 来掩盖前端问题。

本次 CSA 原型只用 `flag != 0` 兼容布尔编码，没有更改工具链或共享安装。未添加标签，未提交 PR。
