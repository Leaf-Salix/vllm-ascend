补充一次**隔离本地原型**的验证结果及实现建议。基于本地配套 PyPTO
`2dfa7f120ba5e577566046d920845aa241bfa7bb`，不是干净最新 main 的测试，
没有向主仓提交修改 PR，也没有放宽入口对非连续根 Tensor 的保护。

### 原型实现范围

原连续 INT8 `[P,page_bytes]` 根仍原样传入，只在 incore 内建立别名：

```python
native_pages = pl.tensor.dim(native_cache, 0)
native_pitch = pl.tensor.dim(native_cache, 1)
keys = pl.tensor.view(
    native_cache, [native_pages, 32, 128],
    strides=[native_pitch, 128, 1],
)
pair = pl.load(keys, [first_page, 0, 0], [2, 32, 128], target_memory=pl.MemorySpace.Mat)
flat = pl.tile.reshape(pair, [64, 128])
```

上面是新增接口的示意，不是当前 main 已支持的 API。采用下列修改建议：

1. DSL/IR wrapper 增加可选 `strides`；显式路径以四个操作数表示
   `(source, shape, valid_shape, strides)`，无 stride 时保留原有调用和语义。
   IR printer 的第四个位置参数也需要可解析。
2. 显式视图类型推导保留原 MemRef。只开放本地、完全有效、packed ND 根和
   完全有效的非重叠 ND 目标，拒绝 FP4、分布式、非正 stride、rank 不符、
   静态跨度越界和 int64 溢出。符号尺寸仍有调用方的运行时边界合同，不能
   声称所有动态地址已在编译时证明。普通 `view` 的乘积检查不改。
3. `BackendHandler::SupportsMultiNdMatLoad()` 默认 false，A2/A3 显式 true。
   `FlattenTileNdTo2D` 对有页间隙的 rank-3 natural Mat load 保留原三维
   source window/stride，只折叠目的 Tile。普通 packed 源仍走旧路径。
4. PTO emitter 使用已有 TensorView stride 生成描述符。该原型在
   orchestration codegen 明确拒绝新四参数形式，只支持 **incore 内部别名**；
   修正我之前评论中“可先在编排中建立别名”的建议，不能把这次验证外推到它。
5. 页表快路仍须在 device 验证正序相邻、不触发重复尾页钳制、物理页有效、
   `4160 <= page_bytes <= 65535`；否则保留原逐页读取。上限来自当前
   `TLoadGm2L1Nd2nz` 的 srcNdMatrixStride 编码，不能删掉。

### 已完成验证

- 独立完整构建成功；没有拷贝扩展二进制或改共享安装。
- 定向 view/Mat load UT：54 通过，覆盖合法 padding、非法重叠/跨度/
  溢出、默认 view 语义，以及三维源、二维 Tile 的保留。
- CPU 生成物确有一次 Key `[2,32,128]` TLOAD，源 stride 保留页间隙，
  目的为 `[64,128]` Mat。另一次 TLOAD 是 query，不算 Key 搬运次数。
- A3 设备验证页宽 4160/4224/8192、同一图页号 0→1→7→0，共 12 项：
  INT8 点积精确相同，输入不变，输出两端保护通过。这个 probe 没测性能。
- 完整动态 CSA B28/S6/128K：58 符号轴、130 根参数、单 MIX/SPMD 编译
  通过；实际设备代码与审查过的 CPU 输出、同一等待变换相符。冻结精度、
  cache/state、保护区、metadata/padding 图重放和同步代际检查通过。
- 关闭新路径的正式版，在新编译器与原配套编译器的生成 C++ 去注释/
  空白后相同；不声称机器码逐字相同。

### 性能负例与不能外推的结论

完整 CSA 同进程、同地址，三方各五个轮换窗口，计完整 L1 最早 incore
开始到最晚结束（含内部等待），中位数：

| 实现 | 完整 L1 / μs |
| --- | ---: |
| 原多 task | 1289.56 |
| 当前正式单 SPMD | 1163.64 |
| 多 ND Key 候选 | 1310.50 |

候选比当前正式版慢 146.86 μs，五个配对窗口全部更慢，因此**没有采用，
没有扩到其他档位**。这里证明功能可实现，不证明能加速 CSA。

原父 Mat 双槽组装尝试产生了 Mat→Mat TMOV，被当前 PTOAS 明确拒绝；
最终改为 N64 Mat 直接送 Right，保留两个 Right 预取槽，但改变了 Mat
分配/复用方式。不能把上面的回退全部归因于 multi-ND 指令，也不能声称
原 Mat 双缓冲完全保留。当前没有足够的核内时序证据定量分离这些影响。

另有与该新接口无关的旧 API 问题：在 view shape/stride 中内联 tensor.dim
会留下未打印的描述符表达式。已用无新补丁的原配套树复现，并补充到
[#2951](https://github.com/hw-native-sys/pypto/issues/2951#issuecomment-6281892586)。
本地先绑定 device Scalar；没有把动态尺寸改成 Host 值或样本常量。

本轮按用户要求结束并暂停；保留本地补丁、完整复现和失败证据供后续参考。
