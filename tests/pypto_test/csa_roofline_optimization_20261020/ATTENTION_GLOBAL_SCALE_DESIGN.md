# Attention 候选设计：五块统一尺度后在 L0C 跨块累加

状态：用户已明确回复“允许独立验证，再按精度和性能决定”。独立候选 v5
已通过完整动态编译，B16 真机验证已提交，尚无设备精度或性能结论；
该授权允许独立数值候选，不代表采用或放宽阈值。仍为一个完整动态 SPMD，真实长度、页表、位置及 mask
保持 device Tensor。以下容量是具体方案的静态资源账本，不能替代编译
分配结果和真实流水验证。初始设计及其修正均保留，当前实现见末尾 v5 记录。

## 目标关键路径与实质区别

当前每 query 最多五个 K128 块，64 heads、D512。每块 Cube 产生完整
64×512 FP32 PV，写 GM，两个 AIV 读回后按该块 softmax 尺度在线合并。
当前 B16/B28 的 Attention 阶段端点约 153/304 μs，内部含等待，不当作
可直接削减的独立延迟。希望减少完整 PV 的五次交接和 AIV 的大向量更新。

旧 wave 失败于额外 Left 恢复及局部工作量不均；旧 direct-L1 失败于把
逐行 gather 加到 AIC；旧共享 KV UB 不减少 PV 往返。本方案不重跑这些
机制：保留 AIV gather，明确支付两块 KV 的 AIC 重读，用一次最终 PV
交接换取跨块累加。它也会损失原 QK/softmax/PV 的块间重叠，收益未定。

## 一个 query 的执行安排

1. AIV 仍根据已验证物理索引 gather 五块 KV，写本核组独占的五个 GM 槽。
   GM 槽在本 query 完成前保留，不能被后续 query 覆盖。
2. AIC 的 L1 保留 KV 块 0、1；第三个 128 KiB 槽依次装块 2、3、4，
   完成五次 QK。QK 的 FP32 结果仍经 GM 交给两个 AIV。
3. 每个 AIV 保留自己 32 heads × 640 scores（80 KiB），按真实有效块
   mask 存放，将五块最大值与 attention sink 一起取最大，计算 FP32 exp/sum，再转 BF16 P。
   不对无效块执行非法 cache 访问；dummy 输出保持原规则。
4. QK 阶段结束后 Q 的 L1 生命周期终止。AIC 载入完整 64×640 BF16 P
   （80 KiB）。保持原 K128/N128 数学 tile，四个 N128 累加器一起占
   128 KiB L0C；按块 0、1、2、3、4 顺序累加。
5. 消费完块 0 后其 L1 槽可重读 GM 块 2；消费完块 1 后其槽可重读块 3。
   块 4 仍在第三槽，因此只额外读两块，且不重复 Native cache gather。
6. 五块累计后只写一次完整 FP32 PV；两个 AIV 读回，使用统一分母及 sink
   归一化，执行既有逆 RoPE、BF16 和 packed 发布。

每个核组仍有独立推进与点对点同步，不引入全核阶段 barrier。第一版按
query 阶段执行，不能假称保持原跨 query/跨 block 重叠。通知由当前每块
KV/QK/P/PV 各一次，变为五次 KV、五次 QK、一次完整 P、一次最终 PV；
尾块、空块、排空和最后消费者需形成配对状态机。

## 片上生命周期账本（每 AIC／每 AIV）

| 阶段 | L1 | L0A / L0B | L0C | 每 AIV UB 的主要活跃区 |
|---|---|---|---|---|
| 五块 QK | 三块 KV 384 KiB + Q 64 KiB = 448 KiB | 沿原 QK tile，分别不超过 64 KiB | QK 64×128 FP32 = 32 KiB | 累计 score 80 KiB + KV 半块 64 KiB + 块内临时约 16 KiB，尚需核实其它临时 |
| 全局 softmax | Q 已死亡，KV 三块仍活跃 | 不新增 AIC 运算 | QK accumulator 死亡 | score/exp 80 KiB + row_reduce scratch 最坏 80 KiB；BF16 P 40 KiB 须在 scratch 死亡后分配 |
| 五块 PV | 三块 KV 384 KiB + 全 P 80 KiB = 464 KiB | P 的 K128 slice 16 KiB；Right 的 N128 slice 32 KiB，可双槽 64 KiB | 四份 64×128 FP32 = 128 KiB | P 发布后不再保留整 score；最终 PV 半块 64 KiB，加分母/RoPE 临时 |

L1 上限 512 KiB，L0C 128 KiB；AIV 实际编译预算 184 KiB。上表要求
Q 与全 P 不同时常驻；否则 384+64+80=528 KiB 超限。也要求 reduce
scratch 与 BF16 P 不同时活跃，否则 80+80+40=200 KiB 超 UB。
最终必须检查生成 IR/C++ 的实际分配、double buffer 和依赖，不能只看
Python 变量离开作用域。

完整五块 KV 是 640 KiB，不能全放 L1。GM 保存五槽是支付额外空间以
免去第二次 Native gather；不能把这次重读漏出流量模型。

## 流量账本（十进制 MB；五块均有效）

基线每 query 的 PV GM 写+读：5×64×512×4×2 = 1,310,720 B。
候选一次写+读：262,144 B。毛减少 1,048,576 B/query。
新增 AIC 从原五槽 GM 重读两块 KV：2×128×512×2 = 262,144 B/query。
所以当前设计的**语义 GM 请求净减少上限为 786,432 B/query**。

| 档位 | PV 毛减少 | 额外 KV 重读 | 净减少上限 |
|---|---:|---:|---:|
| 128K/B16，T96 | 100.66 MB | 25.17 MB | 75.50 MB |
| 128K/B28，T168 | 176.16 MB | 44.04 MB | 132.12 MB |

QK score 原写读与 P 原写读量没有凭空消失；80 KiB score UB 常驻若无法
实现而回到 GM 暂存，还须另扣一次或多次 score 重读。若误改成第二次
Native→AIV→GM→AIC gather，将额外支付三个传输段，净收益会明显缩小。
上述是请求字节，非实测 HBM 字节，不能直接按 1.6 TB/s 宣称延迟收益。
五槽 KV 比原三槽多 6 MiB（24 核组），五槽 score 多 1.5 MiB，全 P 多
0.75 MiB；最终 PV 一个槽代替三个可回收 6 MiB，另核对 ABI/动态容量。

## 数值影响：必须单独确认

基线是每块独立 `m_b`，`BF16(exp(score-m_b)) @ V_b`，得到 FP32 块结果；
AIV 再以 alpha/beta 按块顺序缩放合并。候选改为统一 `m=max(sink, max_b(m_b))`，
`BF16(exp(score-m))`，Cube 在 FP32 累加器内跨五块累加。

实数数学等价，有限精度不逐 bit 等价：

- BF16 转换前的尺度改变，可能改变概率舍入和很小权重的下溢。
- FP32 分母归约顺序改变。
- 原来的“每块独立 Cube 结果，再 Vector 乘 beta 和相加”变为 Cube
  累加，改变 FP32 舍入点。

Indexer score/Top-K、cache/state 与保护区不应改变，Attention/最终输出
可能变化。已获独立验证授权，仍不默认放宽任何冻结阈值。实施后先
检查输出误差与最终 token；不能以数学等价代替真机精度证据。

## 采用与停止

先确认数值策略，再做完整编译资源检查和 B16/B28 同轮配对。至少一档
净省 20 μs、另一档无明确退步才扩档；完整精度/状态/图/代际及 S5/S1
要求保持。如果多一轮 KV 供数、丢失原流水重叠或新增阶段等待抵消收益，
停止该候选，不长期扫描槽宽。不将该候选结束标作 30% 总目标完成。

## 源码细化后的补充约束

- 原 running_m 由 sink 初始化；统一最大值也要包含 sink，防止 sink 项的
  exp 溢出。空 query 保持 m=sink、l=0、分子=0，从而得到零输出。
- 原入口 score/probability/PV/mi/li 共用 HEAD_RING_ROWS 动态轴。若保持
  原 ABI 并统一扩大到五槽，PV 虽只写一次，也不能据逻辑使用量直接缩小
  其物理 shape。上表回收 PV 空间只是后续拆分容量轴后的可能值；首版
  需如实记录实际 allocation，不能预记回收 6 MiB。
- P 可以按 K128 分块载入 16 KiB L1，再复用同一份做四个 N128 PV，
  不必同时常驻全 P 的 80 KiB。这样 PV 阶段 L1 主区为 384+16 KiB，
  若保留双 P 槽则为 416 KiB；实际编译需证明 L0C 四个片段持续累加。
- AIV 的 80 KiB score 可表示为五个独立 32×128 Tile，避免 A3 不支持的
  Vec 窄列 assemble。先分别求最大值再取统一 m，每块以此 m 做 exp/sum；
  各块临时区可复用，不必创建 32×640 的额外完整 reduce scratch。
- QK 采集仍可让下一块 gather 与上一块 score 消费错开一拍；但全局
  softmax 之后才启动 PV，这个阶段边界造成的重叠损失仍须完整测量。

## 实际编译迭代与 v5 状态

原设计中的 80 KiB score 常驻在 v2 编译中得到 Vec 197376 B，超过 A3
实际 188416 B。v3 把 KV gather 改为两次 32 行，分配结果未下降；该尝试
撤回，不能记为节省。v4 恢复原 64 行 gather，把每 AIV 的 32 个 head 分
两轮各 16 个处理，score 常驻降为 40 KiB。另一半 QK 先留在已经写完的
五个 GM 槽中，第二轮读取此前未读过的部分，QK 总读取字节没有增加。
v4 完整分配仍为 197376 B；IR 显示收尾还保留旧在线合并的左右半宽 PV，
并为拼接/归一化/RoPE 建立额外临时区，不能只从 score 缓冲大小判断峰值。

v5 直接按每次发布的 16 个 head 读取完整 `[16,512]` 最终 PV，删除左右
`[32,256]` 常驻及拼接。每 query 仍只读取一次完整 `[64,512]`，归一化、
逆 RoPE、BF16 和 packed 发布的表达式保持。未放宽硬件容量或修改编译器。

完整 B28 区间策略的符号编译结果 `attention_cpu_b28_v5/result.json`：

| 角色 | 内存空间 | MemoryReuse 后声明总量 |
|---|---|---:|
| AIV | Vec | 172032 B（168 KiB） |
| AIC | Mat/L1 | 458752 B（448 KiB） |
| AIC | Left/L0A | 65536 B |
| AIC | Right/L0B | 65536 B |
| AIC | Acc/L0C | 131072 B |

这些是完整函数的分配量，最终地址容量检查也通过。未传样例 shape，保留
58 个符号维度；唯一 MIX 提交、sync_start 和根动态维度审计通过。它们不等于
设备运行通过。四个 Acc 的真实跨块累加、事件配对、空块与图代际仍须真机核实。

B16 首轮任务：`task_20261020_190632_268921518547`。使用既有工具链和原
完整测试器；公共 helper、多 task 及正式参考另冻结，见
`SHARED_SOURCE_FREEZE.json`。测试器仍按逐 bit 记录相对旧 PTO 的差异，
不会为获得性能数据把数值 FAIL 改成 PASS。本轮未运行整模型 token 验证。

数值差异例子：某块局部最大值为 0，全局最大值为 1。该块最大项旧路径先
把 `exp(0)` 转 BF16 得 1，再于 Vector 乘 FP32 `exp(-1)`，约 0.367879；
新路径先得到 `exp(-1)` 再转 BF16，约 0.367188。这只解释舍入点差异，
不是本候选输出误差的实测结果，也不预判最终 token 是否改变。
