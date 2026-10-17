# 长 B16 Attention 推进链：现成产物只读审计

日期：2026-10-18。本次只用 `current_clock_h131072_b16_v1` 已有融合与原分 task
产物，未编译、未上卡、未修改生产代码/依赖。独立于正在测试的 compact 优先编排；
没有将其整层收益算作 Attention 收益。

**结论：未发现融合额外展开 softmax/PV 数学或新增逐块 FFTS 往返。真实新增工作
主要是动态请求几何和地址保护；此前 query metadata/physical row plan 已针对这类
开销做过设备试验。本审计没有找到一个尚未尝试且证据充分的新改动方向。**
生成代码中确有局部同步差异，下面区分一次性初始化、每 tick 和每 query，不能只比
静态 wait/barrier 数量就宣称融合流水退化。

## 复现与证据范围

```bash
python tests/pypto_test/ultra_attention_readonly_20261018/extract_attention.py
```

脚本仅用 Python 标准库读取同次作业文件，输出 [evidence.json](evidence.json) 和
四份 `*_attention.{cpp,pto}.txt` 摘录。它不读取权重、不调用 PyPTO/JIT/设备库、
不计算 hash。C++/PTO 行号和字节数由脚本实际提取；二进制只记现有路径及实际大小。

输入作业根目录：
`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/current_clock_h131072_b16_v1`。

| 角色 | 同次作业的产物 |
| --- | --- |
| 原分 task | `build_output/_decode_csa_tp1_layer_0leidxnz/ptoas/qk_pv.pto`、`kernels/aiv/qk_pv_aiv.cpp` |
| 融合 | `build_output/decode_csa_spmd_jqlk5eey/ptoas/dynamic_csa_full.pto`、`kernels/aiv/vector_schedule.cpp` |
| 实际高层源码快照 | `sources/spmd_reference/attention_stream_stages.py`、`attention_plan_stages.py`、`request_geometry.py` |
| 实际旧诊断 | `result.json`、`phase_summary.json`，H131072/B16/S6；不是新性能测试 |

PTO 摘录从当前 AIV 的 `qk_core` 初始化到 Attention 最后循环携带状态回写。
C++ 摘录从 `m_idx` 初始化到同一末尾回写，不包括前置 GM phase 获取、后置完成发布。
同一 C++ 同时含 AIC/AIV 两个函数，局部 vN 重名；脚本先限制 AIV 函数再截取。

## 1. 实际推进顺序与事件没有新增往返

H131072/B16/S6 对应 T=96，落在已有 `T<144` 路径。24 组各 4 条 query，
每 query 最多 5 个 128 候选块；本作业每个 AIC/AIV 的 FFTS wait 计数均为 40，
与每组 20 个有效块、每块两类等待一致。AIC 22 拍、AIV 23 拍含最后排空。

| 角色/拍 | 原分 task 与融合共同顺序 |
| --- | --- |
| AIV tick | 采本块 KV→写 GM KV 环；等待上一块 SCORE→读 FP32 score/bias→局部 max/exp/sum→BF16 probability、mi/li 写 GM→发布 PROB；随后发布本块 KV；最后等滞后三拍 PV→更新本 query 的 m/l/两半 oi。 |
| AIC tick | 等本块 KV→读入同一 L1 三槽→QK→FIX 发布 SCORE；再等滞后两拍 PROB→读 probability→复用该 L1 KV 做四个 N128 PV→FIX 发布 PV。 |
| query 最后块 | AIV 用 live 累加器加入 sink、归一化、逆 RoPE、BF16，分两段 16 heads 写 packed；重置运行状态。槽号跨 query 延续，最后统一排空。 |

T≥144 的提前 KV 分支也在两份产物中保留：先消费上一块 SCORE 通知，再发布当前
KV，然后做前一块 softmax。此次长 B16 不执行这个分支，不能把两条分支静态相加。

PTO 均有 3 个 wait、3 个 set 文本位置，其中 SCORE wait 和 KV set 各有互斥的
大/小档位置；实际每块仍为 2 wait/2 set。原事件基址 0，融合基址 2：
KV/SCORE/PROB/PV 分别是 0/1/2/3 与 2/3/4/5。MTE3/FIX 发布、MTE2 获取、
mode=2 均保留，未增加每块 GM phase 通知或新的全核同步。

## 2. Tile 数学和逻辑搬运量

排除 `pto.texpands` 的分支初始化位置后，两份 Attention PTO 的 **106 个 Tile
操作名称序列完全同序**；这是生成代码事实，不单独替代操作数/地址/精度验证。

| 静态 Tile 操作位置 | 原分 task | 融合 |
| --- | ---: | ---: |
| TLOAD / TSTORE | 20 / 8 | 20 / 8 |
| row_max / row_sum | 1 / 1 | 1 / 1 |
| exp / cast | 5 / 5 | 5 / 5 |
| row_expand_mul / row_expand_div | 4 / 2 | 4 / 2 |
| concat / gather / mov | 4 / 2 / 8 | 4 / 2 / 8 |
| TEXPANDS | 6 | 5 |

TEXPANDS 少一处是原 task 在 SWA/compact 两个互斥分支各零填同一 KV Tile，融合
在分支前零填一次；不是原 task 每块多零填一次。五个 SWA 页 run 和两个 16-head
最终发布均保持原展开次数，没有因融合把 64 行 compact loop 或 5 块循环展开成
额外数学。mi/li 通过 GM 回环也是原 task 已有路径，不是融合新增搬运。

按本作业全部 5 块有效计算，两版共同的主要逻辑 GM 数据量如下。它们是指令访问
字节，不等于实际 HBM 流量，不能忽略缓存或据此换算耗时。

| 整卡本轮 AIV 工作 | 逻辑量 |
| --- | ---: |
| 两种 cache 采样 | 60 MiB；每 AIV 4 query，各 query 320 KiB |
| 写 KV 环 | 60 MiB |
| 读 score | 15 MiB |
| 写 probability | 7.5 MiB |
| 读 PV | 60 MiB |
| compact 单行 gather 内层 | 每 AIV 4×4×64=1024 次，全卡 49152 次 |

## 3. 融合独有的 Scalar 工作及现有反证

原 task 用 `qk_t//S` 找请求；融合需支持 device Q/L 中 mixed/odd/dummy，
`attention_query_geometry` 在每个有效块扫描所有请求，检查唯一所属请求与 causal
position。随后每次 gather 额外检查页表列和物理页容量。不能为性能直接删除这些
动态/安全合同。

最终 Attention PTO 的静态差值：融合多 1 个请求循环、35 个 cmp、18 个 if、
17 个 and、4 个 Scalar load 文本位置，以及相应 INDEX/地址操作。原 task 有
2 个循环（tick、64 行 compact）；融合有 3 个，额外那个是请求扫描。
原/fused 的 compact loop 在 PTO 原行 509/12903。

在全活动 B16 输入下，请求扫描每次读取 16×(L、Qbegin、Qend)，匹配后再读
position 和该请求 L，即约 50 次 Scalar 读取。每 AIV 20 个有效块重复扫描，
约 320 次 request visit、1000 次几何 Scalar 读取。计数说明有额外工作，
**不说明这 1000 次都是 HBM 事务或主要瓶颈**。

本会话已有 `query_metadata_v1` 用 plan 中的 owner/position 消除这类反查，长档
完整结果未改善；physical row plan 也已尝试消除逐候选页表推导，长 B16 完整
改善约 0.81%。这些是主会话已给定的历史结果，本脚本不重新测量或导入它们。
因此本审计不再把相同方向包装为新优化，更不重开已因整体 UB 规划超限失败的
mi/li 本地环。

## 4. 自动插入的局部同步确有差异，不能粗算成新停顿

按相同 C++ 摘录边界，原 task / 融合分别有 55/75 个 `wait_flag`、63/69 个
`set_flag` 文本位置。它们是本核流水事件，不能与 FFTS `wait_flag_dev` 混为一谈。
两者 `wait_flag_dev` 都为 3 个互斥分支文本位置。

`PIPE_ALL` 文本是 3/5，但动态位置不同：

- 融合 C++ **57374、57409、57522** 是 Attention 初始化、进入 tick 循环前的
  同步；原独立 task 从干净 kernel 上下文开始，没有对应的前三处。包含跨前一阶段
  UB 生命周期的局部事件。没有证据允许直接删除这些编译器依赖。
- 两版在进入 PV 消费前均有一个每 tick 的 PIPE_ALL：原 **2185**、融合 **58706**。
  这是已经存在的串行边，不能当融合新增问题。
- 两版最终第一段发布拼接前都有 PIPE_ALL：原 **2708**、融合 **59142**。
- 原 task 每 tick 最后 m 状态回写前 **3734** 还有 PIPE_ALL；融合对应位置
  **59884～59889** 改为 6 条 MTE2→V 等待，没有该 PIPE_ALL。因此“融合多两条
  PIPE_ALL 文本，所以每拍多两条全流水等待”是错误解读。

以上只确认实际生成的相对位置和种类；没有新增插桩定位其周期，也没有得到每个
本核事件的生产者机器指令范围。不能据此认定 20 个额外 wait 是冗余或直接删它们。

## 5. 代码体积，及不能推出的结论

| 量 | 原分 task | 融合 |
| --- | ---: | ---: |
| 所在 AIV C++ 整文件 | 206043 B | 2525043 B |
| Attention C++ 运行语句摘录 | 133782 B / 2538 行 | 99160 B / 2541 行 |
| Attention PTO 摘录 | 203990 B / 687 行 | 275627 B / 836 行 |
| PTO 去 source location 后 | 112855 B | 124423 B |
| 已有 AIV `.bin` | 10548 B，只有 qk_pv | 151012 B，完整 CSA + 诊断 |
| 已有 AIC `.bin` | 2272 B，只有 qk_pv | 52908 B，完整 CSA + 诊断 |

完整融合 AIV `.bin` 约 14.32 倍于独立 Attention，不能归因于 Attention 展开：
前者同时包含 HC/QKV/两路压缩/Indexer/Attention/输出及 phase/FFTS 插桩。
独立原 task 没有同样的诊断插桩。现成 raw `.bin` 没有 Attention 函数边界，
**无法从本次文件精确给出融合 Attention 独占的机器码字节数**。

Attention 自身 PTO 去 location 后只增加约 10.25%，主要是上节动态几何和地址
保护；C++ 两份声明风格、goto/结构化控制及注释不同，文本大小不等于可执行大小。
没有 PMU/icache miss 证据，不能把完整函数 151012 B 直接解释为指令缓存失速。

## 6. 本轮处置

同次旧插桩三轮 Attention AIC 的 FFTS Scalar 区间中位为
119.71/108.31/112.51 μs，AIV 为 18.35/14.51/14.99 μs；每核均 40 次。
这些值支持继续关注 AIV 推进链，但不是纯 AIV 数学瓶颈或可消除开销。
同次 `eligible_for_performance=false`；没有拿诊断阶段区间作为正式吞吐比较。

本轮只保存可复现的源码/体积/事件差异，**不提出新的代码改动或设备任务**。
已经确认的重复几何/地址工作有现成试验覆盖；数学/搬运和关键环内顺序未见
融合独有的扩张；剩余自动局部同步/指令缓存假说缺少实际耗时归因。按有界任务
要求在这里收口，不为得到一个新候选而做无依据参数扫描。
