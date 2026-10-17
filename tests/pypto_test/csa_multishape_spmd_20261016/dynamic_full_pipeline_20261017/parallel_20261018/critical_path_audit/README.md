# H128K/B16：已有关键路径证据的只读审计

本轮不追加猜测性的优化候选。可确认的主要服务区间是 Indexer Score 与
Attention 流水；尚不能把 Score 约 230 μs 的单角色区间归因于某一类运算。
最关键的缺失观测是当前 `593c5d25` 默认 G6/balanced Score 中，AIC 生产与
AIV 消费各自如何推进：旧阶段时钟不统计 helper 内 FFTS，`other_us` 不是纯计算。
以下分析作为主会话新诊断的待验证假设，不阻塞已经实现的 physical-plan 候选。

本目录只新增既有 JSON 统计、源码直接 AST 对照和中文说明；没有排程模型、
扫参、新编译或设备执行，没有修改 main/依赖/默认策略，也没有做 hash。
已排除的 two-wave 固定 owner、本地 mi/li 环、依赖折叠均不重开。

## 来源、版本和计时口径

`C=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1`。
`D=C/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017`。
源码分析直接读取 `git show 593c5d25:vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/...`。

1. 当前基线完整 L1：`D/parallel_20261018/sync_h131072_b16_v1/source_matrix/result.json`，
   同进程三方矩阵里的 `spmd_reference` 是默认 593，本文不把已否决的
   `spmd_candidate` 用作基线。原分 task 和 reference 同进程、同根地址、独占
   工作区及信号，五窗轮换；复用这份已有结果，不新测。
2. 原分 task 分段：上述目录 `l1_00_dynamic_pto` 至 `l1_04_dynamic_pto` 中的
   `chip_swimlane_records.json`、`deps.json`、`name_map.json`。
3. 最近可取得的长 B16 阶段时钟：
   `D/diag_h131072_b16_output_groups_v1/phase_clock/phase_clock.json`，1769 条端点。
   同目录 `source/` 是诊断时源码；其政策是 G6/N64/1024、balanced、N128 O-A、
   M96 O-B、输出组流水。它早于页有效列修正、默认 HC 行依赖及 S5 尾组修正，
   不是当前 593 的精确分解。更早的 `indexer_priority_v1`、`fast_backoff_v1`
   只用于检查历史方向，没有将它们的差值计作收益。
4. 插桩定义：`D/compile_diagnostics.py:39` 仅改写 `TWAIT/CSA_WAIT_LOCAL_GM`
   和 `TNOTIFY`；`:123` 用相邻发布点之差减累计 TWAIT 得到 `other_us`。

`analyze.py` 按既有 `D/l1_compare.py:70` 的定义，读取每条 raw tuple 的
`r[3]`（kernel 开始）和 `r[4]`（结束），全窗跨度为
`(max(r[4])-min(r[3])) / 50` μs，频率来自原文件的 50 MHz 字段。
任务名由 `task_id → deps.kernel_ids → name_map` 对应，AIC/AIV 分别匹配各自
kernel ID；不依赖 task 名排序，不根据结束顺序猜逻辑 block_idx。

所有相对时间均以该窗最早 kernel 开始为零。没有用含 local setup 的 Chrome
trace `ts/dur` 代替 kernel 首尾，也没有把每核时长相加当延迟。
每个阶段包络取该类 kernel 的最早开始到最晚结束，包含类内部启动偏斜及等待。
原分 task 的 Score 全部结束后才有 Attention 首次进入，因此这两个包络及
中间空隙可组成下面的连续分段；这不是逐依赖边求解的数学关键链。

## 同进程完整 L1 与选择窗

原分 task 五窗为 `880.56, 894.70, 882.04, 871.14, 885.36` μs，
默认 SPMD 为 `906.58, 912.02, 914.08, 900.94, 932.78` μs。
中位数分别为 **882.04 / 912.02 μs**，默认 SPMD 慢约 **3.40%**。
这是同进程完整首尾比较；不是旧时戳对新版本的估算。

目标仍以原分 task 为分母：该轮 30% 降低对应 617.428 μs，默认 SPMD 还需
减少 294.592 μs。此计算只是目标距离，不是某个阶段可以省掉的时间。

选择 `l1_02_dynamic_pto`，因为它的完整 882.04 μs 正好是原分 task 五窗中位数；
不是选择 Score 最慢、最快或最适合某个结论的一窗。该窗原始路径为：

```
/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/sync_h131072_b16_v1/source_matrix/l1_02_dynamic_pto/chip_swimlane_records.json
```

| 该窗连续分段 | 起点 μs | 终点 μs | 跨度 μs |
| --- | ---: | ---: | ---: |
| 完整开始 → 首个 Score kernel | 0 | 294.44 | 294.44 |
| Score AIC/AIV 全组包络 | 294.44 | 531.64 | 237.20 |
| 最后 Score → 首个 Attention kernel | 531.64 | 574.62 | 42.98 |
| Attention AIC/AIV 全组包络 | 574.62 | 735.36 | 160.74 |
| 最后 Attention → 完整结束 | 735.36 | 882.04 | 146.68 |

五段合计严格为 882.04 μs。Score+Attention 为 397.94 μs，约占该原分 task
全窗 45.12%；后段还包含 O-A、量化、O-B、HCpost 的执行与交接，不能当作纯空隙。
交接段包含 TopK merge（539.08～554.02）及 plan（557.16～565.82），
不是 42.98 μs 的无效轮询。Score 的 72 个实际 kernel 区间中位 229.33 μs，
最大 235.32 μs；Attention 分别为 154.04 / 160.38 μs。

为避免单窗选择偏差，完整五窗分段同时保留：

| 窗号 | 前段 | Score 包络 | 交接 | Attention 包络 | 后段 | 完整 L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 | 298.44 | 235.98 | 42.96 | 156.08 | 147.10 | 880.56 |
| 01 | 303.06 | 236.00 | 42.96 | 160.94 | 151.74 | 894.70 |
| 02 | 294.44 | 237.20 | 42.98 | 160.74 | 146.68 | 882.04 |
| 03 | 281.20 | 236.28 | 41.56 | 159.66 | 152.44 | 871.14 |
| 04 | 285.64 | 236.38 | 43.22 | 171.82 | 148.30 | 885.36 |

## 旧动态阶段时钟能支持什么

下表全部来自同一个 `output_groups_v1` 诊断，单位 μs。`other` 只是扣除已
包裹阶段 TWAIT 的剩余时间，仍含数学、Scalar metadata、DMA、FFTS、fence、
cache 维护和插桩，不可解释为某类纯硬件服务时间。

| 阶段 | 核数 | other 最小/中位/最大 | 阶段 TWAIT 中位 |
| --- | ---: | ---: | ---: |
| Score Cube | 24 | 223.38 / 229.48 / 233.90 | 59.25 |
| Score Vector | 48 | 222.72 / 230.96 / 239.50 | 10.43 |
| TopK merge | 48 | 6.16 / 8.73 / 11.26 | 12.03 |
| Attention plan 实际行 owner | 12 | 12.30 / 13.21 / 14.18 | 不混入空 owner |
| Attention Cube | 24 | 150.48 / 156.40 / 163.92 | 56.40 |
| Attention Vector | 48 | 151.98 / 158.71 / 168.58 | 20.86 |
| META | 1 | 7.04 / 7.04 / 7.04 | 0 |

Score 与 Attention 两侧都很长，是优先定位区域；两侧耗时相近并不证明计算
平衡，它们被 FIFO 互相节流。AIC 的 Score 阶段 TWAIT 包含等待 query/coeff/
scale 生产者，不是 Score helper 内的 59.25 μs 冗余等待；Attention AIC 的
56.40 μs 同样包含尚未完成的 TopK/plan 依赖。

前置 META 结束于 19.98 μs，而 HC partial 结束约 35 μs；源码首次消费公共
计划还在主投影之后（`cube_schedule.py:328`）。没有依据把这一份 7.04 μs
公共 metadata 当主要串行瓶颈。plan 只有 12 个有实际行的 AIV，剩余 36 个
会发布空阶段；使用全部 48 核 `other` 中位 2.28 μs 又会低估真实 plan 工作。

源码 AST 对照保存在 `result.json`：旧诊断的 Score AIV、组/叶计划 helper
与 593 完全相同；Score AIC 增加了 S5 尾组补齐修正，S6 完整组不执行 padding
gather。Attention Cube 相同，AIV 新增默认关闭的 query metadata 路径；
plan/HC 调度已有变化。因此可保留长阶段的定位假设，不能把旧阶段总时长
减去当前原分 task 的时长，或把完整 29.98 μs 差额硬分摊给某个 helper。

## 计算、Scalar、核分配与串行化的可确认边界

- **计算/搬运服务：**G6 Score 每块保持 16 个 N64 QK/head-reduction panel，
  AIV 每两块形成 2048 候选段再按原次序 sort/merge；代码在
  `indexer_score_stages.py:243`、`:435`、`:490`。旧时戳和原 task 都存在约
  230 μs 的单角色区间；没有证据表明它全是动态 SPMD 新增的算术或轮询。
- **Scalar metadata：**Score AIC 每 1024 候选块最多查 32 个页并提交 key
  gather（`:195`、`:260`）；两个 AIV 各查同一 32 页的 scale（`:448`）。
  当前完整有效 B16 共有 528 块，对应 16896 次 key 页 gather、33792 次 scale
  页 gather 的源码上界。Attention 还有最多 `96*512=49152` 次压缩候选行
  gather（`attention_stream_stages.py:309`）。这些计数说明有地址处理工作，
  但不能推出它阻塞了多少关键路径；不把计数下降当性能收益。
- **同步/轮询：**既有 `phase_sync.py:84`、`:121` 保留单写者发布、获取和
  cache 维护；helper 内 FFTS 则是另一套 FIFO 协议。已经完成的依赖折叠
  同轮收益方向相反，不再据累计槽读取次数提出重复候选。
- **核分配：**`indexer_long_leaf_plan` 与 `indexer_work_item` 给出 16 组、
  每组 6 叶，各叶 6/6/6/5/5/5 块；24 个 MIX 组均承担 22 个 Score 块。
  Attention 的 q%24 分工使各组都有 4 query、20 块。这沿用已完成的 owner
  审计，不新建模型；块数平衡不证明时长平衡，但没有简单“某核多一整波”依据。
- **流水串行化：**`cube_schedule.py:310` 至 `:350` 明确先主 QB/主投影、
  Indexer Hadamard 与输入获取，再进入 Score；AIV 也先完成所需 scale 写回。
  Score→merge→plan→Attention 是真实数据依赖。两阶段共享同一 24 个 MIX
  组，不能假定同核免费重叠。旧固定 owner 两 wave 反例及静态重排负结果
  已记录，不重做。原 task 前段 281～303 μs 是一个可观察的大区间，但没有
  当前 593 每角色服务分解，无法据此合法承诺能提前多少 Score。

## 唯一缺失观测与本轮决定

需要的是**当前 593 默认长历史 Score 最慢 MIX 组的生产/消费服务轨迹**：
同次调用中，按真实 `(group,leaf,step)` 对应两槽的 AIC `SCORE_CONSUMED`
和 AIV `SCORE_READY`，区分页地址/gather 准备、矩阵/排序区间与等待推进关系。
位置为 `indexer_score_stages.py:174`、`:377`、`:436`、`:469`、`:473`。
这能判别 230 μs 是 AIC 供给受限、AIV sort/merge 反压、Scalar 页循环主导，
还是交替形成的长链；当前只有阶段端点无法判别。

本审计未验证 `wait_flag_dev` 的阻塞语义，等待独立 A3 核对；前后 Scalar
时钟是否涵盖等待需以其证据为准。本审计撤回“只能测得 issue 耗时”的判断，
不根据 pipe 参数推断硬件阻塞范围，也不为测量擅自添加 fence 或改 FFTS 顺序。
主会话正准备的当前 593 baseline 诊断，以及 Indexer 同进程三次 phase 端点，
可先验证阶段位置/稳定性；Scalar 时钟测量的具体含义须结合适用 CANN 版本、
本地生成代码和 A3 阻塞语义的独立核对解释。

本轮没有据不完整归因选定区间策略或改写数学/同步。物理行 plan 已有具体
省工机制且在独立设备验证中，这份只读分析不会要求暂停它。只有当当前
Score 主导侧得到可解释证据后，才有依据选择一个能缩短最慢核组服务的下一
实现；本轮不追加新测，也不把获取这条观测泛化成重跑七档。

## 本地输出与复核

- `analyze.py`：只读原始 JSON、直接 AST 对照和统计；不执行任何算子代码。
- `result.json`：两边五窗样本、每个原 task 窗的命名 kernel 首尾/连续分段、
  旧阶段统计、源码路径/大小、AST 相同或不同的函数清单及明确限制。
- 运行命令：`python tests/pypto_test/ultra_critical_path_audit_20261018/analyze.py`，
  工作目录为本私有 worktree。脚本仅重写本目录 `result.json`，不触碰原始结果。

统计已执行：五窗连续分段非负且逐窗加回对应完整 L1。未新做数学测试、模型
搜索、编译或设备测量；所有原始证据仍在上述主仓只读路径。
