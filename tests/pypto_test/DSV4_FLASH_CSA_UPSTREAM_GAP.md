# 当前性能版与 pypto-lib 泳道差距

更新：2026-09-26。先用现有原始泳道定位，不为本表新增 NPU 测试。

## 数据与口径

- 当前：保留的 `7eba45a3` 性能实现；B16/S6/H8192、mode=2、atomic=1，第二个 CSA 层复用 metadata。
  正式第 2 层权重及合成历史，PyPTO `88297437`、Simpler `a54c05095`、PTOAS 0.66。
  [原始泳道](results/csa_baseline_20260926/perf_qproj_upstream/swimlane/dfx/merged_swimlane.json)。
- 上游：原 `shangyou-merged_swimlane_20260924_005402.json`，从 Git `30795c69^` 读取并保留这一份
  [原始 Worker View](results/csa_baseline_20260926/upstream_gap/upstream_worker_trace.json)。
  文件只声明 `tensormap_and_ringbuffer`，没有源码提交、输入权重、NZ mode、工具链完整配置，
  也没有 Scheduler View；它是历史优化参照，不能当作同配置端到端验收。代码模式另核对本地
  pypto-lib `216456332c2a74d89cca23b7824dab264ce34bff`，不冒称这就是旧泳道的采样提交。
- 核内时间使用 `kernel-duration-us`：含核内搬运、同步与计算，不是纯 Cube/Vector 指令时间。
  setup 是 Worker receive→kernel start，当前 ringbuffer 的 early-dispatch 门控也可能落入此段。
  整组窗口是该类任务最早 receive→最晚 kernel end，含分批启动、竞争与等待。
  各任务窗口相互重叠，**不能相加**；核·μs 也不能当墙钟延迟。
- 两侧 Worker 首任务起点各归零。当前 Worker 区间 **806.14 μs**，上游 **727.98 μs**，
  差 **78.16 μs（10.74%）**。当前另有 dispatch→finish 810.00 μs，但上游缺同口径字段。
  不拿当前 851.07 μs 的整模型层中位数与这份上游单次 eager 泳道直接求差。

## 主要 incore task 与窗口差距

单位 μs，箭头均为上游→当前；核内列为每个 worker 实例的平均值。

| Task | 数量 | 平均核内 | 整组窗口 | 当前判断 |
| --- | --- | --- | --- | --- |
| Q 展开 `qproj_matmul` | 24→24 | 36.44→41.17 | 46.10→88.04 | 核内略慢，启动分散更显著；当前已采用上游 N256/M64/完整 K，不能继续归因于旧 K128 实现 |
| Q 反量化/RMS/RoPE | 48→48 | 20.59→19.70 | 48.84→125.66 | 核内已略快，主要异常在启动/资源可用性；仍是上游 8 tokens、head 循环流水模式 |
| Indexer Q 投影 | 24→24 | 18.89→15.77 | 34.40→45.54 | 核内更快，窗口仍变长 |
| Indexer Q 反量化/RoPE | 48→48 | 13.71→13.18 | 20.26→47.06 | 核内接近，任务铺开变慢 |
| Indexer score/Top-K AIC | 24→24 | 62.98→48.78 | 75.98→64.30 | 当前核本身更快，但前面新增 cache 重排 |
| Indexer score/Top-K AIV | 48→48 | 64.11→45.37 | 77.00→65.42 | 同上；不能忽略重排后只报告 leaf 的收益 |
| 稀疏 QK/PV AIC | 24→24 | 137.90→133.00 | 152.84→149.94 | 已接近且略快，不是当前相对上游的主要退化项 |
| 稀疏 QK/PV AIV | 48→48 | 140.23→135.02 | 155.18→152.18 | 128 列、24 核、预发 2 拍/3 槽与上游一致 |
| `merge_norm` | 48→48 | 16.57→21.65 | 23.32→36.62 | 核内及启动等待均需优化；当前在每个 worker 内生成交换索引，上游另有 rope_swap 任务 |
| O-A 投影 | 64→64 | 20.19→29.31 | 65.56→92.44 | 明确的核内差距；Native NZ 根方向、分块/转置消费方式与上游不同 |
| O-A 后量化 | 24→24 | 7.23→9.80 | 73.90→104.70 | 核内有差距，setup 也从 30.96→43.62；包含等待 O-A，不能全算量化算术 |
| O-B 投影 | 64→64 | 16.27→17.03 | 72.70→77.80 | 接近；当前 Native 二维 NZ、K512/N128，上游分组三维 NZ及完整 K 路径 |
| O-B 反量化汇总 | 24→24 | 14.38→14.48 | 39.22→33.50 | 核内基本相当 |

其余任务没有省略：[全部 54 类任务表](results/csa_baseline_20260926/upstream_gap/tasks.md)，
包括两个 Compressor、HC pre/post、QR/KV、量化、状态提交、Top-K 发布及适配任务。
[原始聚合数值](results/csa_baseline_20260926/upstream_gap/comparison.json) 另含最大核内耗时、setup、
物理核覆盖和当前 dispatch/finish 字段。旧 9 月 24 日模拟器结果没有作为当前 incore 数据套用。

## 调度与关键路径

| 可直接比较的指标 | 上游 | 当前 | 含义 |
| --- | ---: | ---: | --- |
| Worker 实例总数 | 983 | 1131 | 多 148，详见下一表 |
| Q 展开 receive 首末间隔 | 11.76 | 54.18 | 24 个任务从覆盖 24 核变为 20 核，部分核重复执行；有资源竞争/分批启动 |
| Q 反量化 receive 首末间隔 | 30.30 | 104.68 | 单核没有变慢，却更晚铺完；当前覆盖 46/48 个 AIV |
| Indexer Q 反量化 receive 首末间隔 | 4.04 | 37.06 | 同样有明显铺开差距 |
| AIC 核内区间占比 | 61.19% | 54.38% | 按 24 核×Worker 墙钟窗口归一；不代表 Cube 指令利用率 |
| AIV 核内区间占比 | 46.16% | 42.10% | 按 48 核×Worker 墙钟窗口归一；不代表 Vector 指令利用率 |
| AIC 核内合计，核·μs | 10690.92 | 10520.64 | 合计未增加，执行窗口却拉长 |
| AIV 核内合计，核·μs | 16130.84 | 16292.06 | 工作量大致相当，不能说所有核都算慢了 |

当前 48 个 Q 反量化实例中，39 个紧接同核的 `qr_hadamard_quant`，另 7 个紧接
`indexer_score_topk_leaf_aiv`；后 7 个到 380 μs 之后才接到任务。物理同核先后关系支持资源竞争，
不等同于已经证明一条额外数据依赖。上游缺 dispatch/finish，不能定量宣布“调度器自身慢了多少”。

当前可测的平均 dispatch→receive / kernel end→scheduler finish 分别为：
Q 展开 **9.54/7.77 μs**，Q 反量化 **4.99/8.67 μs**，Indexer Q 反量化 **0.51/5.59 μs**。
这是单实例传播/门控/完成回收区间，会互相重叠；上游缺测，也不能从层耗时直接扣掉。
`merge_norm` 的平均 dispatch→receive 127.71 μs 包含很早派发后等待前序任务的时间，
不能把它当成额外 127.71 μs 可优化延迟。

用两侧相同的四个时间边界把整条 Worker 窗口分成不重叠区间：

| 顺序区间 | 上游 | 当前 | 差值 |
| --- | ---: | ---: | ---: |
| 首 Worker→mix_x_rms_norm 完成 | 66.62 | 85.74 | +19.12 |
| norm 完成→QK/PV 首任务接收 | 317.14 | 346.48 | +29.34 |
| QK/PV 首任务接收→merge_norm 完成 | 184.90 | 183.64 | −1.26 |
| merge_norm 完成→HC_post 完成 | 159.32 | 190.28 | +30.96 |
| 合计 | 727.98 | 806.14 | +78.16 |

这是时间分段，不是各 kernel 的独立因果贡献。当前 Q 反量化在 441.84 μs 完成，Top-K 发布
在 455.38 μs 完成，QK plan 在 465.48 μs 完成；QK/PV 已从 460.78 μs 开始流水。
因此只缩短 Q 反量化，也不保证 QK/PV 提前同样多；必须同时看 Indexer 分支和资源竞争。

## 多做的事情，以及为什么模式不同

| 差异 | 实例净增 | 当前核内均值 / 窗口 μs | 原因、必要性与可优化处 |
| --- | ---: | --- | --- |
| `hc_widen` | +12 | 7.26 / 10.00 | Native 残差流是 BF16，上游 HC 入口是 FP32。转换语义必要，独立转换任务不一定必须；直接把 cast 下沉曾影响尾块和 Cube 调度，不能无证据删除 |
| `csa_row_offsets` | +1 | 2.24 / 3.36 | Native compact 输出按实际完成压缩组排列，需逐请求前缀偏移。可尝试随共享 metadata 复用/融合；不是 NZ 权重重排 |
| `indexer_boundary_init` | +16 | 2.22 / 4.18 | S=6 每请求可能完成 1 或 2 组，初始化预留但未填的行，防止 Hadamard 读取陈旧值。保护必要，单独发任务可优化 |
| `indexer_key_repack` | +48 | 20.92 / 43.20 | Native 页内是 4096B INT8 key + 64B FP16 scale，上游 key/scale 分开。当前每步按逻辑页序紧凑化，让 score 侧批量读；是主动实现选择，不是接入必然要求每步重排。应比较整条 repack→score→publish 链 |
| QR split-K | +48 | 任务 16→64 | 当前 split=8，上游代码 split=2；调优选择，非 Native 接口约束。之前改 2/4 未测到完整层额外收益，不能因此宣称调度已经最优 |
| KV 投影拆分 | +24 | 任务 8→32 | 当前 split=8/N128，上游 split=4/N256；同属调优选择，非格式要求 |
| 删除独立 `rope_swap` | −1 | 上游 1.52 / 2.06 | 当前 merge worker 内生成整数交换索引；少一个任务，但每个 worker 增加工作，需与 merge 总代价一起算 |
| **净增** | **+148** | **983→1131** | 所有任务数差额均已对应 |

其他模式差异不增加 task 数，但仍影响性能：

- `csa_rope_interleave` 16 个任务改成 `csa_rope_sign` 16 个任务。Native 已提供交错 cos/sin，
  可省上游的重复展开；同时处理 S=6 的非整块尾行。此项窗口 17.46→5.50 μs，属于减少工作。
- O-A 当前直接接受 Native 根 `[G,K,N]` NZ，默认 matmul；pypto-lib 是 `[G,N,K]` NZ，
  用 `b_trans=True`。O-B 当前复用 `[G*K,N]` 二维 NZ，核内计算组偏移；上游是 `[G,N,K]`。
  **NZ 物理打包规则相同，逻辑矩阵方向与分组方式不同**。接入处没有每次调用的 device 权重重排。
  当前保留 M128、O-B K512/N128；上游有 M32/96/128 自适应及完整 K/N256 路径。
  这些是需要继续缩小的代码/搬运差异，不能只用“Native NZ”解释 O-A 的全部 9.12 μs 核内差距。
- Native 的 state/cache 页寻址、FP16 scale 写回、有效请求/负 slot 保护仍在多个已有任务内部。
  例如 `idx_kv_scale_commit` 核内 3.82→7.36 μs；任务名相同不代表存储合同相同。
- 两个 Compressor、cache writeback、HC_pre、HC_post 上游也有，不能把整个这些阶段算成接入额外工作。
  本窗口已经复用 compact metadata，不能再把首层的两项 metadata 生产成本重复加进去。

## 按本次证据调整优先级

1. 优先处理 Q/Indexer 分支的派发与资源竞争，以及 repack→score→publish 整条链；不再把
   Q 反量化 125.66 μs 的窗口直接当成其算术耗时。连续 16 heads 的候选实测 861.06 μs，已撤回。
2. 优化 O-A 投影与量化流水，明确记录每个候选与上游权重方向、M/N/K 和任务边界的区别。
3. 再处理 HC 入口转换、merge 索引生成等附加工作；QK/PV 当前已接近上游，优先级相应后移。
4. 每个候选一次必要单卡计时；有明确收益再补受影响边界和整模型 token 看护。
   后续更新必须同时记录 incore、调度窗口、额外工作及其原因，缺测明确保留，不扩大测试矩阵。

CPU 复算：`python tests/pypto_test/results/csa_baseline_20260926/upstream_gap/compare.py`。
