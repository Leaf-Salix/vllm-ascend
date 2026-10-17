# 593 balanced/G6 原生 Key 预取：协议与资源模型

结论：把双 AIV 的部分等待区间用于整理 Key，在协议上有可行的双槽方案；
**共用单份 scale 的原提议有真实覆写竞态，必须改为每 lane 私有 scale。**
但经 GM 暂存会为长 B16 增加约 132MiB Key 交接字节，可能抵消收益。按主
agent 最新要求，本轮只保存设计/CPU 模型，不实现 GM 版，不编译、不占卡。

设计只读基于 `D/parallel_20261018/baseline_source` 的 593 原路径，不基于
未采用的本地 score v6。D 是主仓动态整链测试目录。由于没有实现，本轮未新建
代码 worktree，也未改私有正式候选；后续若实现必须另从 593 分离。

## 已有证据与目标

当前 128K/B16 基线诊断的 Score AIV 原位 FFTS 区间中位约 107µs、AIC 约
16µs；B32 短历史也观测到 AIV READY 等待显著。这些是含探针的逐核区间，
并非可以全部拿来做任意工作的免费时间。要研究的工作是 AIC 原生 Key 的
页表解码、分支与每 1024 候选 32 个小页 DMA，保留 G6/N64 的 QK、WS、FIX
2^-10/ReLU、1024 score FIFO、balanced 叶划分和 TopK 输入/tie 顺序。

当前 AIC 处理每块 32 个 Native Key 页，每页 4096B Key 后跟 64B FP16 scale；
两 AIV 原本各遍历 32 页 gather 完整 scale。由两个 AIV 各整理连续 16 页
Key，同时保留各自完整 scale，可把 AIC 页解析迁出：总页表解析从每块
32+32+32 次变为 32+32 次，而不是把已有解析凭空再做一遍。

## 双槽协议

新增 KEY_READY 使用事件 6、mode2；现有 Score READY/ACK 是 0/1，Attention
事件基址2使用2..5。这里只提出接口合同，不修改或复用已有事件计数。

```text
AIV lane（两边即使 live query=0 也参与协议）:
  对同一有效 leaf:
    prepare Key(0) 的本 lane 半份 + 本 lane 私有完整 Scale(0)
    在 MTE3 完成后发布 KEY_READY(6)
    for s = 0 .. N-1:
      if s+1 < N:
        prepare Key(s+1) 半份 + 私有完整 Scale(s+1)
        在 MTE3 完成后发布 KEY_READY(6)
      等原 SCORE_READY(0)
      读私有 Scale(s)、读原 Score(s) 片段
      在原 MTE2 完成点发送 SCORE_CONSUMED(1)
      保持原 scale 乘法和 balanced 排序/根合并顺序

AIC:
  对同一有效 leaf:
    for s = 0 .. N-1:
      if s >= 2: 等原 SCORE_CONSUMED(1)
      等两 AIV 汇合的 KEY_READY(6)
      从连续暂存 Key(s) 一次装入 L1 [1024,128]
      原16个N64 QK/WS/FIX/score store；发布原 SCORE_READY(0)
    排空原 min(N,2) 个 SCORE_CONSUMED(1)
```

Key 与 Score 都用 `s%2` 槽。AIV 在处理 s 时写 s+1，覆盖的 Key 是 s−1；
进入这一轮之前该 lane 已收到并读取 Score(s−1)，而该 Score 的发布必然晚于
AIC 读完 Key(s−1)。Key 的唯一消费者是 AIC，所以这个证明不要求另一个 AIV
已经消费 Score(s−1)，**Key 本身不需要 KEY_ACK**。

原 Score 槽安全仍由双 lane MTE2 ACK 保证，不能把 ACK 提前到 score load
之前。每有效 leaf 两 AIV 各发 N 个 KEY_READY，AIC 等 N 次；最终 Score
发布已证明所有 Key 被消费。旧 Score ACK 仍在 leaf 末排空。下一 leaf 可以
提前积压至多两份 KEY_READY，按一致的 leaf/step 顺序消费，不重置计数。
跨代要求外层运行时已完成前一次调用；无效 leaf 三个角色都跳过，尾组空 lane
仍 gather Key、等待和 ACK。保留现有尾组物理 M 填充，不读取额外 GM query。

### 为什么单份共享 scale 不安全

若 lane0/1 各写共享 scale 的一半，但各自都要读整份 scale，那么快 lane
收到 Score0 后可立刻准备 Key2/Scale2 并覆盖 slot0；慢 lane 此时可能尚未
读 Scale0。快 lane 自己的 ACK 不证明慢 lane 已完成读取，AIC 的 Key 消费
证明也不涵盖 scale 的两个 AIV 消费者。两个仅一步的 leaf 跨边界也有同样竞态。

建议最小实现合同是每 lane 保留完整的 scale 副本：优先比较 UB 双槽与 GM
私有双槽两种存储，**不增加一个掩盖问题的全局 barrier**。UB 版省流量但增加
跨循环 Tile 生存期；GM 版每 lane 私有 scale 共占 192KiB，比原共享 scale 提议多 96KiB；相对于新增 Key 流量
仅增加每块 8KiB 读写。两种都保留原 Native scale 的 FP16 位模式和转换顺序。

`key_prefetch_protocol_model.py/json` 是纯 Python 有限交错模型：覆盖单 leaf
1..8步、1/2/3/5/6步混合、空 leaf 与边界共15组，遍历1793个可达状态。
私有 scale 方案未发现 Key/Score 覆写或死锁，每 lane 待消费 KEY_READY 最多2；
共享 scale 在 `[3]` 和 `[1,1]` 给出具体反例。模型假设 release/acquire 和
mode2 汇合语义成立，不能证明真实 DMA、别名、cache 或编译器顺序正确。

## 内存与接口

只在 `BALANCED && QUERY_GROUP==6 && SCORE_TILE==1024` 的路径研究。
score_arena 原非 balanced scratch 此时闲置，现有容量
`24*2*6*8192*4 = 9MiB`。G2 只有3MiB，不能盲目复用相同固定前缀。

| 项目 | 容量/约束 |
| --- | ---: |
| 24 AIC × 2 槽 Key，每槽1024×128 INT8 | 6MiB |
| 每槽每 lane 私有完整1024 FP16 scale | 192KiB |
| GM 私有 scale 方案合计 | 6.1875MiB；9MiB内余2.8125MiB |
| 每个 Key 半份 | 64KiB；两个 writer 区域不共64B cache行 |
| 每份私有 scale | 2048B；不共cache行 |

可在 JIT 编排层、进入 MIX 前，从现有 score_arena 建等字节的 INT8/FP16
Tensor view；按连续 Key 前缀和 scale 后缀定位。视图始终保留实际动态容量，
不要在 G2 入口无条件切出越界6MiB视图，也不要在 incore 内把整块 GM reshape
成 Tile。当前 PyPTO `pl.tensor.reinterpret_view(..., shape=...)` 有零拷贝
接口，现 entry 已在编排内部创建 idx_key0/key64。这里仍需后续独立 CPU
验证 writable alias 是否按期望 lowering，不得把接口存在当作已经可编译。
只传内部视图给 AIC/AIV，不新增外部根 Tensor、Host metadata 回读或 allocation。

整理时必须复用原安全页公式、物理页检查和 last-page clamp。尾块不能擅自
改变成全零或把 scale 字节拼进 Key；原有效列和 Native FP16 scale 按位保留。

### AIC 片上容量

只需要一个128KiB L1 Key块，GM双槽不意味着 L1也要双槽。每块装入后再按
0/64/.../960做原 N64 Right 提取；保持原 QK(current)/WS(previous) 和两份
Key Right 的提交顺序，不能照搬已经失败的循环化/新分片。

- L1：Query48KiB + coefficient12KiB + 两score片96KiB + Key128KiB =
  **284KiB**，比原16KiB Key池的172KiB多112KiB；局部逻辑容量小于512KiB。
- L0A：Query48KiB + coefficient12KiB =60KiB。
- L0B：WS Right48KiB + 两Key Right各8KiB =64KiB，已经顶满，不能再添第三份。
- L0C：QK96KiB + WS4KiB =100KiB，小于128KiB。

这只是同时存活对象的容量模型，不是完整 planner 地址证明。旧 whole-key-pool
CPU 产物支持128KiB L1池可表达，但实际新视图/TLOAD/自动事件仍未编译验证。

### AIV UB 生命周期

prepare 的局部上界模型是 roots12KiB + segments24KiB + Key半份64KiB +
一页临时4160B + 私有scale2048B，约106.06KiB。前提是 Key准备 helper 只返回
Tensor/Scalar，不把64KiB Tile跨步携带到 TopK；sorting峰值与此峰值应取最大。
本地 PyPTO safe Vec 上限是184KiB（物理192KiB顶部保留8KiB），不是256KiB。
若64KiB Key Tile错误地与约160KiB完整内核高水位叠加，会溢出；因此 lifetime
是必查门槛，不能简单在旧 TopK 旁再永久放一个缓冲。

## 带宽和可能抵消

原每1024块 Native输入为131072B Key + 两份2048B scale =135168B。
整理不会减少这些必需输入；它把其中Key搬到 AIV读，再写GM，再由AIC读。

| 方案 | 每1024块新增GM交接字节 | 128K/B16示例528块 |
| --- | ---: | ---: |
| scale留各AIV的UB双槽 | 262144B | 132MiB |
| scale留各AIV的GM私有双槽 | 270336B | 136.125MiB |

通用块数为各实际 query group 的 `ceil(max_visible/1024)` 之和；balanced
1024粒度分叶不增加该和。示例是16个G6组、每组33块，并非策略白名单。
若这部分字节以1TB/s有效路径吞吐完全暴露，串行等效约138/143µs；2TB/s
约69/71µs。它可能经L2/NoC并与计算重叠，不能把这些数称实测 HBM耗时或必然
性能损失；但足以说明不能把 AIV约108µs的等待直接全部转成收益。

还新增每块两次 AIV KEY_READY、一次 AIC wait；每 leaf 的 key0 prime 是新
启动依赖，N=1/2的短叶没有足够稳态摊销。AIV的 gather/Key store 会与原scale、
score load、排序争用 MTE/UB；A3 wait又阻塞后续发射，因此必须把下一块准备
放在原 Score wait之前才可能利用等待间隙，不能等READY后再准备然后声称重叠。

## 与历史失败方案的区别

主日志8896..9004附近的 `INDEXER_KEY_POOL_CPU_EXPERIMENT.md` 记录旧方案：
仅把 AIC Key行池16→128KiB，各页独立32行；原32个小页DMA、页解析、提交顺序
都不变，没有增加预取深度。完整32个matmul仍在，MTE1→MTE2 wait19→20、
MTE2→MTE1 wait18→18，没有减少目标等待，因此CPU后否决、未上卡。

本设计的区别是 AIV先把离散Native页变成连续GM块，AIC才可能一次大TLOAD；
不是仅扩池，也不是旧N64循环ring/两面板预取。旧失败不能直接否定这个迁移，
但同样不能为其额外GM交通背书。此前两面板预取完整PASS且无收益的记录继续有效。

暂不推进GM实现。下一步仅只读审计是否存在本地 A3 的 AIV→同组AIC L1直传
能力，核实 PyPTO、PTO-ISA、硬件方向和绑定约束，以判断是否能避免这次GM往返。
