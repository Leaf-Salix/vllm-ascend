## 需求

希望提供共享比较掩码的保序有界压缩 Tile 接口，一次返回值、索引和真实计数。
可以扩展现有 `gather_compare`，不要求固定采用下述原型名称。
这是性能能力请求，不是主线正确性 bug，也不声称已有完整性能收益。

当前精确Top-K候选在 `gather_compare` 后仍需索引补齐、短索引物化、
随机gather score及全局索引偏移。底层硬件可以用同一比较掩码压缩两路，
希望在DSL层显式表达它，便于编译器管理所有结果、生命周期和同步。

## 建议的接口与语义

```python
values, indices, count = pl.tile.compact_compare(
    src, threshold, scratch, offset,
    cmp_mode="ge", out_cols=1024,
)
```

- `src`和`values`为FP32；`indices`、`count`为INT32；`scratch`为UINT8。
- threshold是FP32 Scalar，offset是INT32 Scalar；二者可来自device数据。
- 按源顺序保留匹配项，score仅做位搬运，索引基址不能经浮点转换。
- count返回完整匹配数；每行只定义`min(count, out_cols)`个输出前缀，
  超容量时仍返回真实count且不能越界写。调用方据此选择安全回退。
- src有效行列数可变；物理列数64对齐，out_cols为8的倍数且不大于src宽度。
- count物理存储是`[1, align_up(rows, 8)]`，有效范围只含真实行数。
- 首版本地后端只支持A3与EQ/GT/GE；应显式拒绝不支持的模式/类型。

## 编译建议

三份输出用`TupleType`与`set_output_arity(3)`，复用现有tuple投影和分配机制。
scratch仅声明为workspace，不能暴露一个隐藏返回值。明确各参数的读写效果，
并禁止不安全的输入输出就地复用。offset作为真实Scalar operand传入PTOAS，
不能变成按数据值重编译的属性。

PTOAS应具有相应显式DPS合同，并区分Vector值/索引写和Scalar计数写。
多输出分配、有效形状、自动split和后继消费者同步都需要覆盖。
原型先提供Tile层接口，不假装已有完整Tensor层便捷路径。

## 当前证据与限制

独立本地PyPTO已构建；三输出类型、动态Scalar参数、容量/模式拒绝等，
以及旧gather相关回归共56项通过。底层原语辅助模板已通过A3 CCEC编译。
新PTOAS正在构建，尚无新原语的完整真机/整层性能结果，不能据此采用CSA候选。

直接设备验证会覆盖GE/GT/EQ、零长度、超容量、INT32大基址、FP32位保持、
同分和同地址变输入图重放。最终仍用完整CSA五窗交替配对衡量新增压缩成本。
本地只保留原型与验证，没有提交修改PR，也没有push依赖仓。

配套PTOAS能力请求：https://github.com/hw-native-sys/PTOAS/issues/1650
