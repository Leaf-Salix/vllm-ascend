## 问题

多层 inline 调用中，内层表达式被 hoist 后生成的合法自动名，在外层再次内联时
被直接追加 `_inlineN`。这破坏版本后缀，随后 ConvertToSSA 将整个名字当成
用户 base，报保留分隔符错误。用户代码没有使用双下划线。

这是 CPU 编译问题，不涉及 NPU/PTOAS/Simpler；也不是打印后重新导入。
与 #3022 的表现相关但触发链不同：本例直接 InlineFunctions → ConvertToSSA。

## 复现

在本地 PyPTO d01de8523 独立工作树复现；该树有其它 CSA 编译改动，但
inline_functions_pass.cpp 没有未提交修改。2026-10-19 只读核对上游 main
14736f65c04a7db06547edb656f515054950555c，FreshName / HoistTempName 仍为下述实现。
尚未构建该最新 main，以下错误是本地实际执行结果。

不需要任何权重、shape 实值或设备：

```python
import pypto.language as pl
from pypto import passes
from pypto.ir.printer import python_print

@pl.program
class Nested:
    @pl.function(type=pl.FunctionType.Inline)
    def base(self, requests: pl.Scalar[pl.INDEX], tokens: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        group_base = (requests + 16) // 16 * 16
        group_end = group_base + 4 * tokens
        return (group_end + 15) // 16 * 16

    @pl.function(type=pl.FunctionType.Inline)
    def layout(self, requests: pl.Scalar[pl.INDEX], tokens: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        return self.base(requests, tokens) + 48 + 4 * tokens

    @pl.function(type=pl.FunctionType.Inline)
    def wrapper(self, requests: pl.Scalar[pl.INDEX], tokens: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        result = self.layout(requests, tokens)
        return result + 2 * requests

    @pl.function
    def main(self, requests: pl.Scalar[pl.INDEX], tokens: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        result = self.wrapper(requests, tokens)
        return result


after = passes.inline_functions()(Nested)
print(python_print(after))
passes.convert_to_ssa()(after)
```

InlineFunctions 完成，生成名之一为：

```text
t__inline_arg_v0_inline5
```

ConvertToSSA 实际错误（数字后缀可随运行变化）：

```text
ValueError: IR auto-name base cannot contain reserved delimiter '__': t__inline_arg_v0_inline5
```

只有 main → layout → base 两层 inline 的简化版可以通过；再加 wrapper
即可复现。实际动态 CSA 中把 `return inline_base(...) + ...` 改为
`base = inline_base(...); return base + ...` 后完整编译通过，是目前算子端的临时规避。

## 根因定位

- `NestedInlineCallHoister::HoistTempName` 通过
  `auto_name::BuildName("t", "", "inline_arg", counter++)` 生成 `t__inline_arg_v0`。
- `FreshName` 无条件 `JoinNameSuffix(orig, "inlineN")`。该 helper 遇到已有 `__`
  直接拼后缀，得到 `t__inline_arg_v0_inline5`。
- `ConvertToSSA` 的命名解析只有 role 或 version 被识别才接受自动名。
  拼接后的尾部 `inline5` 已不是 `vN`，`inline_arg` 也不是已知 role，
  因此后续 BuildName/ValidateBaseName 拒绝整个字符串。

## 建议修复（供参考，未提 PR）

在 FreshName 中区分编译器合法自动名和普通名字。对前者通过 AddQualifier
插入新的内联身份，保留尾部 role/version；普通名字保持当前 JoinNameSuffix，
继续拒绝用户的非法 `a__b`，不要无条件删掉双下划线或放松全局校验。

示意代码，尚未编译验证：

```cpp
std::string FreshName(const std::string& orig) {
  static int counter = 0;
  const auto suffix = "inline" + std::to_string(counter++);
  const auto parsed = auto_name::Parse(orig);
  if (parsed.has_auto_suffix &&
      (parsed.role.has_value() || parsed.version.has_value())) {
    return auto_name::AddQualifier(orig, suffix);
  }
  return auto_name::JoinNameSuffix(orig, suffix);
}
```

期望得到如 `t__inline5_inline_arg_v0`，仍能解析版本。也可采用其它统一
自动命名方案，但需要保证 repeated inlining 的唯一性及 SSA 可识别性。

建议回归：本例三层 inline 后直接 SSA；同一 helper 多次调用；自动 tmp/role 名；
尾部下划线 `_`/`t_`；非法用户名 `a__b` 仍拒绝；tuple/循环中的 inline 临时变量。

## 期望

合法源码的嵌套 inline 表达式应完整编译，不能要求用户通过临时赋值规避
内部自动命名协议。当前 CSA 已采用等价显式赋值继续性能验证，未修改共享安装。
