# 最新 nalinaly 与我方 Native 精度 TND：差距定位（2026-10-23）

## 固定版本与口径

- 我方已验收组合版：`aad137911b8fa9deb06c60fa5e8e8ccec13f55f3`，显式`tnd_precision_combined`，原精度基准保留。
- nalinaly：`cafcadec96d48790608e5b55d456150fcf00e794`，17:27提交；21:14再次查询GitHub，无更新。本轮只更新远端引用和独立源码快照，不切换/合并工作分支。
- 公共同输入：真实第4层权重、合成hidden/history、8K/B32/S6/T192，NZ2/page32，HC-pre到HC-post，FULL graph。
- 确定性level1/HCCL开启；Native为SK0。这里不是截图的SK1数字复刻，也不是整模型token测试。
- nalinaly单SPMD在我方固定SDK编译：仅去掉SDK不支持的`backoff_ticks=150`关键字，数学不改；未改生成C++/SDK。不能冒充其原工具链性能。
- nalinaly多task TND原样运行，不需要该等待兼容；用于展开模块，不将其阶段耗时冒充单SPMD内部分段。

## 同组完整回放实测

单位μs，12个轮转窗口×100 replay，保留全部样本。

| 实现 | event均值 | event中位数 | Native精度 |
| --- | ---: | ---: | --- |
| Native | 1257.224 | 1256.848 | 对照 |
| 我方已验收组合版 | 1816.296 | 1817.781 | A/B八外部边界及ABA逐bit一致 |
| nalinaly最新单SPMD，固定SDK兼容 | 1212.730 | 1212.178 | 最终输出relative L2 0.333%，非Native逐bit一致 |

我方与nalinaly单SPMD的event均值差603.566μs，约慢49.77%。这是真实差距，但不能把全部差距归因于精度。

另一次独立的多task诊断同组event均值为Native1222.685、我方1813.529、nalinaly多task1144.421μs；与上表不是同一任务，不交叉计算优化收益。

## 最大差距在哪

L1计时与dep-gen分开采集；以同窗设备最早开始/最晚结束界定里程碑。各窗口分段之和严格等于该窗口总跨度；各列中位数不要求可相加。以下nalinaly为**多task诊断伴随版**，不是单SPMD阶段时钟。

| 不重叠区间 | 我方 | nalinaly多task | 差值 |
| --- | ---: | ---: | ---: |
| 完整入口开始→TopK就绪 | 708.06 | 565.50 | 142.56 |
| TopK就绪→heads就绪 | 664.47 | 280.29 | 384.18 |
| heads就绪→完整输出 | 346.42 | 218.97 | 127.45 |
| 完整L1跨度中位数 | 1716.11 | 1067.24 | 648.87 |

最大差距明确在attention区间。这个结论基于受控同输入整段及独立模块展开，不再仅凭源码猜测。

更细的kernel跨度可能相互重叠，不能直接相加：

| 阶段 | 我方 | nalinaly多task | 可比性限制 |
| --- | ---: | ---: | --- |
| QK/PV MIX | 约463.85 | 约254.76 | nalinaly还包含归一化/逆RoPE/packed发布；计算/归约语义不同 |
| 额外压缩索引检查 | 约91.38 | 无单独等价任务 | 我方保留Native物理成对排序，不可直接删除 |
| merge/归一化 | 约38.86 | 已在上述attention里 | 不是独立可相加收益 |
| heads输出布局整理 | 约18 | 已在上述attention里 | 我方是PTO复制，不是调用Native模型算子 |
| WO-A整个任务跨度 | 163.57 | 124.68 | 量化边界及后续消费者不同，不能照搬计算 |

当前单SPMD的L1只显示完整MIX包络，没有现成内部阶段时钟。不把AIC与AIV并行时长相加，也不把PMU占比当作纯轮询等待。

## 哪些实现可借，哪些数学必须保留

| 部分 | nalinaly | 我方Native精度约束 | 借鉴方式 |
| --- | --- | --- | --- |
| QK/PV搬运 | 三槽、跨块/跨query流水，KV一次入L1后复用 | 每输出元素的K顺序不变 | 独立验证L1驻留及提前搬运，不改点积 |
| softmax统计 | 同AIV保留UB环 | Native K512/actualK求和、sink初态、概率round | 只改mi/li/alpha存储位置，四组八头各自UB |
| 索引plan | 合并计划、缓存已验证geometry | Native负槽、最后索引fallback、物理pair地址顺序 | 合并重复page/有效性读取，保留全部谓词 |
| softmax与输出边界 | K128更新、sink末尾、概率rint，未舍入heads直接逆RoPE | K512更新、sink初态、actualK求和树、概率round、逆RoPE前BF16 | 不复制这些已知不一致的数学边界 |
| O-proj | 分组1024尺度 | 全8192列共同动态量化及Native BF16发布 | 借矩形调度和owner，不借分组量化 |

本轮nalinaly同输入外部八边界均与Native存在差异；TopK数组差异不等于已证明选点集合不同（尚未作排序集合核验）。其precision regression检查主要相对旧PTO；不能替代本轮对Native检查。当前尚没有证据把每个精度差异单独换算成多少μs。

## 已完成与进行中

| 单因素 | 结果 | 精度/处理 |
| --- | --- | --- |
| HC-pre删除H3冗余同步 | 8K两次下降1.52%/1.87%，各12/12窗胜；128K下降0.92%，12/12窗胜；L1更低 | 单层A/B八边界+ABA逐bit通过；真实8K和128K的两轮192 token/接受事件均Native一致 |
| SWA重复exp精简 | 8K慢0.457%，0/12窗胜，L1也更慢 | 不验精度、不合入 |
| SWA复用已驻留L1 Key | 完整64参数编译及L1物理空间检查通过 | 8K两次胜、128K回退；三组Native逐bit通过，不通用合入 |
| 索引检查合并重复读取 | 12100组原/新Scalar AST边界检查、完整64参数编译通过 | 8K两次胜、128K胜；三组Native逐bit通过，待组合 |
| 四组八头UB统计 | 前三种布局被CPU编译拒绝；独立状态版完整编译通过 | 8K硬件慢1.48%，不验精度、不合入 |

原基准源码/环境不变。仅对胜出的完整CSA候选验精度，再在8K/B40、128K/B24复用Native黄金验token和接受事件；不跑Native2。任何首差停止扩大，先定位；所有自有NPU任务780秒外层/750秒内部上限。

证据：`TIME_8K_B32_V1.json`、`STAGE_ANALYSIS.json`、`NAL_MPMD_STAGE_ANALYSIS.json`、`MILESTONE_ANALYSIS.json`、`TASKS.json`及原始归档。结果尚在增加，排队不等于设备执行通过。

## 21:21补充

预取独立大T候选8K两次下降3.84%/2.17%、128K下降2.23%，均12/12窗胜、L1更低及Native A/B八边界/ABA逐bit通过；混长T21逐bit通过但event慢1.52%，不在短T启用预取。两项新组合与SWA整列特化仍排队，尚未作为正式结果。H3正式8K与128K整模型forward均值103.792/88.640ms；没有新whole控制，不能计算成对加速。SSH中断时观察有空档，硬件上限未撤销，全部既有任务终态/释放已复核。
