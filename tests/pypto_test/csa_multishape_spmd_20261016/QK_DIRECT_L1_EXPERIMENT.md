# QK/PV直接L1供数实验

本候选仅在实验生成的B16/S6单SPMD内启用，未替换生产入口或七档选择。
目标是去掉Vector上的64×512 BF16 KV临时块与GM交接；是否提速以完整CSA的
无插桩L1 incore首尾为准，不能从少了一次搬运直接推断。

## 改变与保持

旧路径：Native KV → 两AIV各64行UB → GM kv_transfer → Cube三槽L1。
候选：Native KV → Cube三槽L1。AIV继续执行原32-head softmax和PV在线合并。
QK/PV矩阵乘、五块合并顺序、归一化/RoPE/输出布局都保持。

原`csa_slots_build_valid_qk_plan`生成的私有稀疏索引临时区，仅该生产者和QK使用。
两AIV分别处理每query的256列，将逻辑索引批量转为物理KV行并原地写回。
外部TopK输出不改变；下一次调用仍由原valid-plan重建逻辑索引。
负索引保留负号，Cube按原无效行语义补零。页表、位置仍为原设备Tensor，未新增Host取值。

每组lane0在原kv_transfer的`core*384`起点写32行零值，供无效行和SWA尾部读取。
该空间现在只有零源用途。没有新GM分配或外部参数，原三槽L1大小不变。
SWA按最多五个32-token页段搬运；不完整页、负页、尾部逐段补零。

## 交接协议

|事件|生产者发布点|消费者|
|---|---|---|
|4|两AIV的MTE3完成物理索引/零源写入|Cube等待两lane，执行fence/cacheinvalid/fence后读取地址|
|1|Cube FIX完成score写回|两AIV计算原softmax|
|2|两AIV MTE3完成probability/mi/li写回|Cube计算原PV|
|3|Cube FIX完成PV写回|两AIV读取并在线合并|
|0|两AIV MTE2完成左右两份PV读取|Cube在复用三槽前等待两lane|

事件0原来表示KV_READY，现在表示旧PV已消费。不能只删除旧KV_READY等待：
它还间接限制流水超前程度。候选在Cube的tick≥3时等待tick-3对应的有效块；
循环结束后再等待最后一块，确保进入下一阶段/调用前没有残留事件。
有效块标记在三方保持一致，空块不发布也不等待。

地址表由DMA写回，不能假设Scalar缓存自动一致。Cube等待事件4后明确维护缓存；
原跨阶段fence、DCCI和GM代际通知也保留。这里使用已有FFTS组内交接，未增加自制Scalar信号。

## 证据与限制

- `qk_direct_l1_probe_cpu_v1`：真实编译子视图保留父Mat的Rows=384跨度，validRow可为1或运行时值。
- `task_20261017_042346_360920825849`：测试驱动漏初始化kernel runtime，未执行kernel，保留失败。
- `task_20261017_042437_362436518114` / `qk_direct_l1_probe_device_v2`：A3同图7项搬运哨兵PASS。
  三槽逐行、非对齐起点、动态行数、末行及回到原槽，全部数据/相邻行/保护区/只读输入通过。
- `qk_direct_l1_protocol_result.json`：3328个CPU随机交错调度、322443步，2/3代复用；
  包含两query全部1024种有效块组合及4query抽样，事件排空、slot生命周期和无死锁通过。
  SWA有1310种区间覆盖检查。模型不证明真实DMA/缓存可见性。
- `qk_direct_l1_cpu_v1`：完整CSA CPU代码生成PASS。
- `qk_direct_l1_prepare_v1/codegen_audit.json`：实际IR/C++的发布、等待、地址缓存维护、
  PV加载及末尾排空顺序人工核对。手工FFTS循环采用`pl.range`，原纯数学pipeline保留。
- `task_20261017_043021_371809025343` / `b16_cached_qk_direct_l1_v1`：完成exit0。
  原exact、TopK、六份cache/state、图、30代信号和保护区全部PASS，之后完成3窗L1。
  真实NPU构建C++与已审计CPU C++全文相同；仅比较文本，没有hash。

## 完整性能与结论

|路径|三窗完整CSA L1 incore首尾（μs）|中位（μs）|
|---|---|---|
|同轮原动态PTO|892.66 / 920.94 / 888.36|892.66|
|直接L1单SPMD|1096.08 / 1100.10 / 1100.20|1100.10|

候选比同轮原动态增加23.24%，明确不采用、不扩其他档。现选历史815.84μs仍保留；
本轮没有现选单SPMD同进程A/B，不把历史差值解释为精确回退幅度。
因为候选已被性能淘汰，不追加实际候选片段的负索引/短SWA或metadata重放测试；
这些边界不能由小搬运探针、CPU区间模型或固定fixture的PASS代替。

生成代码提供的有限归因：Vector整个QK阶段的`PIPE_ALL`静态位点从9降至5，
但整层变慢。原两AIV各64行的并行搬运变为一个Cube的128行循环，伴随逐行GM Scalar
地址读取与单行ND2NZ。真实128行循环内部没有额外pipe_barrier/set_flag/wait_flag，
不能说是“每行新增屏障”导致。搬运串行化及ND2NZ粒度成本是合理排查方向，但没有新增
内部探针量化各项耗时，不将全部回退定量归到某一项。证据：
`qk_direct_l1_codegen_comparison.json`和`qk_direct_l1_compressed_loop.cpp`。
首次只读统计试图按record注释定位阶段失败；改用C++花括号边界后得到上述计数，算子没改。

另只读确认当前A3的`TPUSH/TPop` Vec→Mat路径仍经GM FIFO，不能假定换接口就能绕过GM。
源码为独立Simpler构建所用`build/pto-isa/include/pto/npu/a2a3/TPush.hpp`的
`pushVec2GMFiFo` / `popMatTileFromGMFiFo`，没有为此改依赖或运行新测试。

所有变更位于本实验目录。冻结静态结果、PyPTO、Simpler、PTOAS及生产算子没有修改。
