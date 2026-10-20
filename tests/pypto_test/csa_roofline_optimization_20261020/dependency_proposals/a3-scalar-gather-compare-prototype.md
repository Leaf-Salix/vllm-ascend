# A3 Scalar 阈值 gather compare 的独立修复原型

对应上游问题：<https://github.com/hw-native-sys/PTOAS/issues/1647>。
本地原型尚未完成设备验证，不代表完整 API 支持已交付。

`pto.tgather` compare form 的阈值是与 src 相同 dtype 的 Scalar。旧 A3
EmitC 输出的模板实参与 ISA 的 Tile 阈值重载不匹配；不能通过把浮点阈值
转成整数或构造未计入资源的隐藏 Tile 来绕过。

本原型只在 A3 compare form 实际出现时生成辅助模板，保持 Scalar 类型。
EQ、GT、GE 使用真实比较模式；GE 是精确 Top-K 同分保留所需的扩展。
索引用 Vector 构造，比较掩码与 `vreducev2` 保序压缩之后返回真实计数。
输出容量不足时只写容量内的前缀，调用者必须检查计数是否超过容量。
index form、mask form 及 A5 不改用该辅助模板。INT32 仍只允许 EQ。

调用者提供的 tmp 必须容纳全部临时区，不隐式占用未声明 UB：

```text
indexBytes = ceil(physical_cols / 64) * 64 * 4
maskBytes = ceil(physical_cols / 256) * 32
requiredBytes = indexBytes + max(1024, maskBytes + indexBytes)
```

源行使用其实际 RowStride，每行只压缩 valid_cols。计数输出本身也必须
满足 UB 物理容量对齐；valid_shape 可以保持原逻辑行数。
自动生成 scratch 的 Tensor 层路径尚未适配，当前 CSA 明确提供足够 tmp。
浮点阈值、GE 边界、非整段 valid、计数、offset 和容量保护都需要设备测试，
单独 EmitC 或 CCEC 成功不能代替它们。
