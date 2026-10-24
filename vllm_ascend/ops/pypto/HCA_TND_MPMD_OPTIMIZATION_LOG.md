# HCA TND MPMD 实现与优化过程

本文件按用户要求放在`ops/pypto`外层，集中记录HCA TND MPMD的设计、验证和正负优化结果。
正式算子：[deepseek_v4_flash_hca_tnd](deepseek_v4_flash_hca_tnd/)。
本轮汇总：[2026-10-23投影分档](../../../tests/pypto_test/hca_tnd_mpmd_20261022/PROJECTION_RESUME_SUMMARY.md)。

## 2026-10-22：目标、历史依据与首版

用户目标：参考 CSA-SPMD、CSA-MPMD-TND、CSA-SPMD-TND 三个会话，在独立
`deepseek_v4_flash_hca_tnd` 目录实现 HCA TND MPMD，性能超过 Native 30%。
性能比例采用用户此前指定的分母 PTO；不把该目标改为仅某个子算子或单档达标。

当前检出 `67134f28b`；CSA-MPMD-TND 目录正在另一会话重构，本轮只读参考，
不改其未提交内容。公共 SDK 清单为 `csa-shared-20261021-upstream-fixes`，状态 validated。

### 已查明的起点

旧 HCA 的外层 `[T,4,4096]` 虽然展平，但 raw cache 写入、压缩器与 attention
仍用 `token//6`、`request*6`、`token%6`；Native adapter 也拒绝非 S6 输入。
所以不能只修改参数名或去掉入口检查，就把旧实现叫作原生 TND。

旧 HCA 已有可复用的三槽 QK/PV 环形流水、跨 query 不排空、四组 Q_B 发布、
O 投影分组量化与融合 HC_post、按 Native 页搬运和直接 C128 state 读写。
首版保持这些算术与任务主体，先替换真实请求语义。

### 从三个会话吸收的原则

1. T/B/历史长度是不同的维度；Host 只处理形状和已有调度描述符，设备 Q/L 每步更新。
2. MPMD 跨 task 用运行时依赖，不移植单 SPMD 的 GM notify/wait、代际和退避循环。
3. 避免大范围全核同步与伪依赖，保护 Q、KV、compressor 可重叠的独立分支。
4. NZ 权重、双缓冲、真实尾行、宽向量工作分配等优化须按完整关键路径实测选择。
   不把某次 CSA 子段收益直接加到 HCA，也不优先追求 task 数量最少。
5. 不照搬 Indexer/Top-K：HCA 只读 C128 压缩历史与滑窗，没有 CSA Indexer 链。

### 首版改动（待验证）

- 独立 JIT 入口和 Native adapter，解除 T=B×6 的限制；支持 T≤384、请求容量≤64。
- 新 token geometry 任务读取设备 Q/L，生成 64B 对齐的请求行描述符；失活行长度为零。
  它是业务 metadata，消费者通过 TaskId/张量依赖等待，不是软同步信号。
- C128 使用请求 Q 起点和实际长度读取当前投影、遍历 state slot、计算关闭边界；
  pooled 缓冲按实际请求容量分配，不增加 state 搬运窗口。
- Attention 保留已验证的长档流水，按 token 描述符映射请求与滑窗偏移，inverse RoPE
  按真实 T 处理尾块。先统一长短历史功能基线，短档分工在有新证据后再决定。

首版尚未完成 CPU lowering 或设备检查。这里记录设计与已落地源码，不预记 PASS。

## 2026-10-22：首轮编译、数值归因与测试口径更新

### 编译与入口修正

CPU 尝试 `cpu_v1`～`cpu_v5` 保留各自报告。依次修正：

- 旧 HCA 调用公共 O_B 时未传新增的真实 token 数，导致 constexpr 绑定错位；
  新目录的 `output.py` 明确传 T，继承已验证的 O_B 多余尾行裁剪。
- C128 的 `begin/end` 在不同作用域有 INT32/INDEX 重用，统一地址量为 INDEX。
- inverse RoPE 的新显式 load 返回 Tile，索引构造改用 Tile API，消除 Tensor/Tile 混用。
- 当前后端不支持 INDEX 直接转 FP32；有效长度在 [0,128]，先转 INT32 再转 FP32。

`cpu_v5` 完整后端编译通过，生成45个 incore 函数，其中 attention 包含 AIC/AIV
两个硬件角色。这里是函数数，不是某档实际动态 task 数。随后删掉新目录里不属于
本目标的 HBG 工厂和 Host 标量入口；`cpu_eager_v1` 编译通过，根 ABI 为36个 device Tensor。
没有修改 PyPTO、Simpler、PTOAS，也没有改共享 CSA 或旧 HCA。

### 设备证据与失败边界

| 任务 | 场景 | 结论 |
|---|---|---|
| `task_20261022_163915_162997318004` | 首次 B16 提交 | pending 时取消，补测试的 `_pto_hca_layer` 绑定；未执行设备 |
| `task_20261022_164106_175565826883` | 128K/B16/S6 | 输出通过 atol/rtol=0.01/0.01；graph/eager 精确，保护区通过；Native raw KV/state 严格比较失败 |
| `task_20261022_165011_217196715640` | 旧 HCA 同进程对照 v1 | 旧 attention 的 INDEX→FP32 在公共新版工具链编译失败，尚未形成数值对照 |
| `task_20261022_165445_235041432597` | 旧 HCA 同进程对照 v2 | 新旧输出、raw/压缩 KV、state 全部逐 bit 相同；Native 差异也完全相同 |
| `task_20261022_165445_235063211300` | 混长、跨 C128、padding v1 | 入口错误要求 RoPE 描述符覆盖 T，Native 只暴露有效行，PTO 未执行 |
| `task_20261022_170351_277195714222` | 同场景直接调用 v2 | 输出通过容差；重复直接调用全状态精确，写保护/只读 metadata 通过；Native storage 严格比较仍有差异 |
| `task_20261022_170654_288635012019` | 固定 Native RMS 输入的 C128 | state 最大绝对差 5.364418e-7、在1e-5门槛内；压缩 KV 逐 bit 相同；写保护通过 |

旧 HCA 对照只在冻结快照中补 O_B 参数和有效长度的整数桥接，不更改旧目录。
对照前两侧完整 CPU 编译通过，见 `legacy_cpu_v2`。

B16 Native 差异：输出 max_abs=0.015625、RMSE=0.00134917，容差内；raw KV
15207个元素超过1e-5严格阈值、max_abs=0.015625；state 38606个元素超过阈值、
max_abs=0.000577271。这些值在旧 HCA 和本次 TND 中完全相同，不归因于新请求映射。
没有提高这个严格阈值、删除失败项，也没有声称 Native state 已无差异。

混长场景的 query=[1,5,6,2]、history=[127,123,128,254]，有效T=14、容量T=20、
请求容量5（末尾一个失活请求）。它同时覆盖三个 C128 关闭边界、一个不产出压缩行的
请求、S1/S5/S6/S2 和 padding。完整输出 max_abs=0.03125，仍在既有相对/绝对容差内；
压缩 KV 与 Native 有一个 BF16 元素不同。固定实际 Native RMS 输出再执行同一生产
C128 函数后，压缩 KV 精确一致，说明这次压缩输出差异来自压缩器上游输入，
不能把它当作 C128 请求边界或 compact 索引错误。

RoPE/slot 修正参考 CSA-SPMD-TND 的 `token_capacity_view`：先检查底层存储字节数，
再对 Native 已有容量建立同地址 as_strided 描述符。没有分配补齐 Tensor、没有复制
RoPE/slots，也没有 Host 回读 Q/L。业务访问依旧由设备有效长度和 slot 保护。

### 用户更新：后续不用 ACLGraph replay

用户本日明确要求后续不再使用 ACLGraph replay。之后提交的 `eager_tnd_v1`、
`compressor_probe_v1`、`eager_profile_v1` 均直接调用；重复一致性检查也由两次
恢复相同初态后的直接调用完成。后续同址 metadata 检查采用直接调用。

此前已结束的 Native 图基准 `task_20261022_165533_238089523155` 为573.25 μs，
实际 static/SuperKernel/NZ/ACLGraph 已核实。该结果只作历史，不能与新 eager 表计算收益。
`run_native.py` 保留历史源码并停止直接执行；`launch_native.sh` 已切到直接调用入口。

### 首个 eager 性能诊断（未表示完整精度验收或目标达标）

任务 `task_20261022_171223_31537281875`，128K/B16/S6，每侧5窗：

| 口径 | 中位数（μs） | 全部样本（μs） |
|---|---:|---|
| PTO L1 incore 首尾 | 515.08 | 515.08, 531.80, 511.72, 507.00, 515.30 |
| PTO 外部全部设备流（含 AICPU） | 637.25 | 664.75, 624.00, 630.25, 637.75, 637.25 |
| Native eager 外部全部设备流 | 3832.75 | 5563.00, 4315.50, 3603.50, 3832.75, 3490.00 |
| Native 设备任务活跃区间并集 | 677.00 | 671.50, 677.25, 673.25, 677.00, 681.75 |

Native eager 首尾包含数千微秒的任务间空隙；不能把这些空隙称为核内优化收益。
活跃区间并集只是辅助解释，既不是端到端延迟，也不能直接替代原来的图基准。
PTO外部活跃区间并集596.25 μs，包含AICPU，和纯L1的515.08也不是同一种指标。
详细证据见 `runs/h131072_b16_eager_profile_v1/EAGER_PERFORMANCE.json`。

测试参数 `--diagnostic-perf` 要求新旧 HCA 输出和所有 cache/state 精确相同、
Native 输出在容差内、重复直接调用与保护区通过，才允许采诊断性能。
它不修改整体 `passed=false` 或 Native storage 的失败项，不用于证明精度全面通过。

外部采集器用同步结束的 CPU 标记定位实际设备窗口，保留层内真实搬运及所有旁流，
不依赖 graph ID、不把同名复制一律删除。CPU 回归检查已覆盖复制边界和多流区间。
正式目录当前 Python AST 与 `eager_profile_v1` 已测快照一致，注释整理不触发重复真机测试。

### 下一阶段

1. 补齐 S1/S5、长历史混长、同址 metadata 改写及长短档功能覆盖，保留 Native 误差来源。
2. 完成 Native/PTO eager 的固定八档基线，明确外部跨度与设备活跃时间差别，不混用图数据。
3. 依据完整 L1 关键路径选择候选：B16 首轮 attention MIX 约177μs，O投影至post约164μs；
   HC前段、Q_B分组发布和历史gather之间也有可重叠区。实际优先级以多窗口和大档为准。
4. 逐个做保持数学结果的调度/供数改进，再评估HC_pre已验证的分档策略；任何数学变化
   明确单列精度证据，不把局部收益相加。补服务入口时继续保留Native回退与KV生命周期。
5. 逐档验证超过 Native 30% 的完整目标；当前只有一档诊断，没有八档达标结论。

本轮完成了实际源码、编译和设备归因，属于有进展；目标保持 active。

## 2026-10-22：Q_B负结果、TND边界及按Native核内重排优先级

### Q_B扩大到24核：两档均不采用

控制与候选在同一进程、同一卡上交错，各7个直接调用L1窗口；公共SDK为
`csa-shared-20261021-upstream-fixes`。保持四组INT8点积和后处理，只把每组4 Cube改为6。
两档输出、raw/压缩cache、state均与冻结控制逐bit相同，保护区及重复直接调用通过；
Native严格storage差异没有消失，不改总体精度状态。

| 场景 | 原版中位数 μs | 24核候选中位数 μs | 差值 | 任务 |
|---|---:|---:|---:|---|
| 128K/B16/S6 | 513.14 | 518.58 | +5.44 | `task_20261022_175355_101880012851` |
| 128K/B28/S6 | 771.62 | 778.56 | +6.94 | `task_20261022_175355_10189308318` |

不采用、不扩档；正式entry始终导入旧Q_B。候选完整模块移到
`tests/pypto_test/hca_tnd_mpmd_20261022/candidates/query_projection_rejected.py`，
冻结包与7窗原始泳道仍在 `experiments/qb_24_v2`、`runs/qb_24_b16_v1`、`runs/qb_24_b28_v1`。
不把更多核数当作性能优化结论。

### 同址metadata直接调用：四场景通过，最后一次Native失败待定位

`metadata_h131072_v1`先发现入口用`max_seqlen_q<=6`拒绝合法dummy容量；
Native这个字段可包含补位，不能据此推断真实请求长度。移除此检查，真实语义仍由设备Q/L决定。
收口时改用Native已有Host字段`max_seqlen_q_actual`检查1～6；字段缺失才退回旧max。
这样既接受长dummy，又不放行本实现尚未支持的真实长query。没有Host回读设备Tensor。
`metadata_h131072_v2`暴露测试把compact视图长度误当固定容量，改为初态提供足够现有容量，
测试更新只在原设备allocation内填写有效行；生产adapter不新增复制或展开缓冲。

`metadata_h131072_v3`的S1、混长及页属主反转、S5、无新压缩行四场景通过：
同址更新与新描述符调用全部输出/cache/state逐bit一致，Native输出容差通过，写保护通过。
最后A_s1_again在**Native调用后的同步**报MTE非法GM地址，尚未进入本轮PTO调用。
保留原错误与前四项，不写完整A→B→A通过，也不先归因于PTO同步或Native自身bug。
新性能工作先保持固定fixture，未围绕此异常反复扩测。

### 用户要求抓大放小：先核实真实Native路径和已有大项证据

完整来源与量化结果见
`tests/pypto_test/hca_tnd_mpmd_20261022/NATIVE_KERNEL_REVIEW.md`。
本次GitCode访问失败，使用只读recipes 6d060311及仓内/已安装A3源码，不声称读到了远端最新版本。

已核实C128走SparseAttnSharedKV的CFA分支；当前host、kernel和AICPU metadata都按
M=gSize=64，KV块512。不能用文档M256示意或旧ops-transformer版本断言当前实现合轴四条query。
已有三槽和跨query流水不重复列为新点；旧双query/512列/直读分页的负结果也一并核对。

新统计区分kernel-duration、task包络和设置/提前等待：B16 O_A包络84.10μs，
核占用/24为55.59μs；分组量化包络73.62μs，但核占用/48仅3.72μs，不能误当成
73μs计算可回收。B28 O_A包络129.16μs，O_B至归约/post仍是较大的尾链。
O_A生成码已包含真实双缓冲，不靠反复添加stage=2宣称新机制。

本轮结构性候选将O_B、八组反量化归约与HC_post放入一组MIX流水，保留O_A和分组量化。
每核两槽交接，某输出列块的g7完成后直接收尾；48MiB全量partials由M96/M128分支的
2.25/3MiB循环缓冲替代。A3交接仍有GM读写，不能把容量下降说成相同比例流量下降。
量化全组等待和大T权重重复加载是明确代价，将用B16/B28同进程配对判断。
候选在 `tests/pypto_test/hca_tnd_mpmd_20261022/candidates/output_fused.py`；正式output暂未修改。

公共SDK18:00切换为`csa-shared-20261022-logical-codegen`，18:14标为validated。
本轮CPU与后续设备配对统一新版本，控制也重新编译；不跨SDK与旧时间相减。
CPU v1～v4在NZ动态列偏移非负证明处失败，尚未上卡；正在检查pass中间IR，
不将CPU候选或理论结构写成已验证提速。

### 尾链融合候选完成：数值严格等价，B16明显变慢，不采用

CPU v5/v6增加pass dump后确认：编排侧的`workers=24`在outlining后仍作为标量参数，
此前Simplify已删除NZ列偏移的max(0,...)保护，后面的NZ地址检查无法证明range步长非负。
v7直接在核内range写编译常量24，候选与控制完整后端编译通过；没有修改共享SDK。
这些只是源码等价改写，不更改核数、数学和TND语义。

单卡任务`task_20261022_183801_30125409797`，128K/B16/S6，新公共SDK同进程交错各7窗：

| 版本 | L1中位数 μs | 全部样本 μs |
|---|---:|---|
| 原尾链 | 511.60 | 511.60, 504.54, 513.18, 516.84, 515.40, 503.78, 502.02 |
| O_B/归约/HC_post MIX | 623.54 | 613.98, 607.22, 623.54, 613.22, 657.12, 628.00, 634.82 |

输出及所有cache/state与控制逐bit相同，重复直接调用、只读metadata、写保护通过。
Native严格storage差异仍保留，`passed=false`与`diagnostic_performance_only=true`均如实记录。
性能增加111.94μs（21.88%）；不采用、不扩测B28、不修改正式output。

同批泳道的O_A核占用/24为52.99→54.38μs，基本未变；旧O_B核占用/24为30.39μs，
新MIX AIC为63.27μs；旧归约/post核占用/48为29.05μs，新MIX AIV为120.66μs。
新尾链开工约476μs，旧O_B约436μs已开始，失去原来分组量化后提前开工的重叠。
这证明当前组织的等待和核占用抵消了缓冲缩小；尚未用PMU区分各类指令/GM带宽占比，
不把全部额外时间武断归因于某条缓存或同步指令。

下一步保持原分组尾链，不围绕失败融合版继续扫小参数；仍按Native实际数据复用和
完整关键路径选择结构变化。全八档、服务接入、Native原有精度差异及30%目标均未收口。

NZ证明问题已用无权重、无设备的分档inline helper独立复现：v4中captured_step失败，
literal_step完整编译通过；早期不含inline分档的v2/v3两版都通过，不能删掉这个触发条件。
提交 [PyPTO #3105](https://github.com/hw-native-sys/pypto/issues/3105)，附完整可运行脚本、
pass证据、保留常量/范围事实的修改建议及负偏移回归要求；未提PR、未添加label、未改SDK。
复现脚本和issue正文分别为测试目录的`repro_nz_positive_stride.py`、`NZ_POSITIVE_STRIDE_ISSUE.md`。

## 2026-10-22：八档基线、真实服务接管及输出尾链精度修正

以下全部采用`csa-shared-20261022-logical-codegen`、正式第3层W8A8/NZ2、直接执行；
没有新建ACLGraph或使用replay。具体任务ID、报告和完整采样分别在测试目录
`MATRIX_TASKS.json`、`SERVICE_TASKS.json`、`PRECISION_TASKS.json`及各`NATIVE_OUTPUT_*_TASKS.json`。

### 先补功能证据

初始八档矩阵完成：七档输出通过原atol/rtol=0.01/0.01，8K/B32/S6有3处超差；
八档全部与冻结TND原数学一致，保护区和重复直接调用通过。Native严格storage仍失败。
`MATRIX_SUMMARY.md/json`保留当时数据，8K/B32在精度门禁处停止，没有伪造性能。
旧HCA短历史路径与新TND在8K/B32逐bit相同，排除TND迁移和长短策略本身导致这3处差异。

新增正式`service.py`，显式复用已有`dsv4_hca_forward`：
Host只检查已有shape/dtype/数量边界，真实请求Q/L、位置、slot与页表仍由设备核读取。
同一forward context中两次调用只生产一次compact metadata，新context再次生产；
等待KV→compact→记录计算起点→PTO→通知写完成→保存KV的次序经真实custom op验证。
S1/B4及混长[1,5,6,2]、token容量20/request容量5两组接管成功，
所有输出/cache/state与直接PTO调用精确一致。没有以Native回退冒充接入成功。
测试调用真实KV钩子，但不等于已完成实际PD传输或FFN/整网token验收。

### 输出差异来自历史量化算法，恢复Native边界

完整归因见测试目录`NATIVE_OUTPUT_PRECISION.md`。诊断从生产AST提取中间值，
采集版与原入口最终结果精确一致后才解释拼接结果，没有另写一套QKV或attention。
8K/B32：PTO attention接Native尾链时3处超差归零；Native attention及门控接旧PTO尾链
仍有1处超差。这锁定了O投影尾链自己的数学差异。

旧版O_A保留FP32、八组各自量化并进行FP32部分和归约；Native O_A先落BF16，
之后整个8192列token共享一个尺度。正式`output.py`已恢复该Native语义。
O_B点积在INT32内完成，按channel尺度、token尺度反量化后落BF16，再融合HC_post。
8K/B32的3处超差归零，RMSE从0.001952652896降至0.001737914970；
同Native输入的新PTO尾链RMSE=0.000031872203且0超差。容差没有改变。
新的八档输出均PASS，cache/state与旧TND控制精确相同，Native严格storage差异没有被隐藏。
修正后又通过S1及混长/padding的真实服务接管验证。

### 正负性能均记录，按区间采用O_B分块

完整L1同进程配对中位数如下，单位μs。数学修正版不是全档性能改善，采用的首要原因是精度修复。

| 场景 | 旧分组数学 | Native数学完整K | 样本数/侧 |
|---|---:|---:|---:|
| 128K/B1/S6 | 230.82 | 227.96 | 3 |
| 128K/B8/S6 | 355.90 | 366.40 | 3 |
| 128K/B16/S6 | 515.70 | 524.54 | 7 |
| 128K/B24/S6 | 696.56 | 706.56 | 3 |
| 128K/B28/S6 | 759.74 | 756.26 | 7 |
| 256K/B16/S6 | 649.72 | 659.14 | 3 |
| 8K/B32/S6 | 619.64 | 601.16 | 7 |
| 128K/B16混长T56 | 429.92 | 432.42 | 3 |

新泳道中post核占用减少，但整token量化要等齐O_A分组，推迟O_B开始。
后续候选扩大N128→N256，同时沿K8192分成两段4096，保持逻辑任务数；
理论上减少激活重复加载，代价是多一份INT32写回/读入和一次整数相加。
两段先在INT32域相加，保持相同整数点积；输出及全部cache/state逐bit等价。

| 同任务对照 | 完整K | K分两段 | 样本数/侧 |
|---|---:|---:|---:|
| 128K/B16/S6 | 518.92 | 515.56 | 7 |
| 128K/B28/S6 | 783.48 | 767.30 | 7 |
| 128K/B24/S6 | 698.22 | 699.32 | 5 |
| 8K/B32/S6 | 602.36 | 596.72 | 5 |

B16变化小，继续保留完整K；B24变化约0.16%，不把它称为改善。
综合B28及8K/B32收益，正式代码按T>96采用两段策略，其余用完整K。
不是B28单点特化。两种策略共用同一数学函数，M32/M96/M128按T区间选择，
每次从动态维度计算有效行，未将TND退化成固定B×6。
区间合并后B1/B16功能精确比较通过，B24/B32精度及配对性能通过。

### 编译与记录修正

CPU检查现已增加编排C++和全部incore C++的真实编译。此前只生成PTOAS代码的PASS
不能当作AscendC目标文件通过。Scalar FP32 max实际编译命中PTOAS#1667，
已在原issue补充证据；改为Vector两级max后通过，没有修改SDK。
诊断采集还暴露了scope内Tensor SSA别名在scope外使用的编排C++问题，
独立复现并提交PyPTO#3107，附修改建议；未提PR、未增加bug label。
具体失败、v1～v4源码和本地大产物路径均归档，不删除失败记录。

覆盖提交中五个v1命令遗漏`--batch`，在fixture构造前被拒绝；修命令重提v2，
两个已成功的B16场景没有重测。该错误属于测试提交，不归因于SDK。
Native cache/state严格门禁、最终服务性能与整模型验收仍未收口；不报告30%目标达成。

### 用户再次固定口径：只看完整L1，Host/JIT问题到issue为止

服务profiling的三个重点档确认实际接管成功，但PTO完整服务设备首尾达8.6～9.9ms，
这主要是compact producer与PTO启动之间的Host派发空隙，不能说成核内计算回退。
未插PyTorch profiler的B16复核：普通预绑定PTO提交7076.17μs、服务7636.17μs；
cProfile定位到重复的动态维度传播、constexpr计划展开、源码依赖检查。
使用SDK已预热registration、继续保留完整参数/别名/设备/存活/异步所有权检查的
测试对照中，提交分别降至1043.04/1572.14μs，输出及全部状态精确保持。
完整服务含同步仍为2220.07μs，同轮Native为2094.53μs，没有宣称服务性能达标。

无权重两Tensor/32层inline链的最小复现已经运行通过：15窗普通热提交4263.746μs，
prepared诊断178.572μs。动态T=16/32、新Tensor地址、错误参数拒绝均通过。
复现中v1多行提交命令被包装脚本截断；v2缺少SPMD block_idx被前端正确拒绝；
v3补齐后通过。保留提交日志，不把这两次用例错误归因于SDK。

已提交[PyPTO #3113](https://github.com/hw-native-sys/pypto/issues/3113)，附完整独立脚本、
热调用数据和显式prepared callable/不可变特化计划缓存的建议，没有PR或bug label。
**用户随后明确：性能只看L1 incore开始到结束，此类JIT问题提issue即可，当前不解决。**
因此不再扩测Host，不采用测试中的SDK内部接口，不修改共享工具链。
后续核内候选全部回到完整L1配对；上述数据仅作历史诊断，不能与L1耗时混算收益。

### 下一项大颗粒候选：整token量化之后重新评估Native O_A供数

旧`hca_native_five_20260930`的M96/N256/K64候选曾有O_A局部收益，但没有整层收益。
本次实质变化是下游已改为Native整token尺度，原分组量化/O_B提前开工不再存在。
因此新候选将八个O_A发布任务合成一个SPMD任务网格，同时采用A-L1 K512、
B-L1 K256、L0 K64双缓冲，保持完整FP32累加链与当前量化/post。
每次仍从设备描述符的动态T计算M96行块与尾块有效行；没有只写B16静态版本。

完整候选在`candidates/output_oa_native.py`，冻结包`experiments/oa_native_all_v1`。
CPU编排及全部incore实际编译已通过；B16/B28只采完整L1配对，结果另记，不预记通过或采用。

## 2026-10-23：O_A任务均衡与区间补验，未采用

两套O_A候选均与正式Native数学控制的输出、cache/state逐bit一致，
Native最终输出容差、重复直接调用和保护区通过。Native严格storage原差异仍保留。

| O_A候选 | 场景 | 控制L1 μs | 候选L1 μs | 采样/侧 |
|---|---|---:|---:|---:|
| M96/N256/K64，八组并为一个网格 | 128K/B16 | 515.14 | 537.50 | 7 |
| 同上 | 128K/B28 | 787.00 | 779.92 | 7 |
| N192×5＋N64×1，共48份/行块 | 128K/B16 | 519.70 | 507.48 | 7 |
| 同上 | 128K/B28 | 784.54 | 800.00 | 7 |

第一套在T96只有32份Cube任务，24核第二轮只剩8份；第二套每组6个列块，
T96共48份，正好两轮。最后64列使用独立编译期tile宽度，不读/写下一组。
前者B16回退，后者B16首轮七组均改善但B28回退，因此进一步验证33<=T<=96区间，
其余T沿用原八组投影。没有将B16单点条件硬编码进设备计算。

区间合并版`oa_ranges_v2`的完整L1配对如下：

| 场景 | 控制L1 μs | 候选L1 μs | 采样/侧 |
|---|---:|---:|---:|
| 128K/B8/S6，T48 | 341.18 | 352.26 | 5 |
| 128K/B16/S6，T96 | 513.64 | 512.56 | 5 |
| 256K/B16/S6，T96 | 663.90 | 676.14 | 5 |
| 128K/B16混长，T56 | 420.96 | 429.94 | 5 |
| 128K/B28/S6，T168，保留原O_A | 778.94 | 786.34 | 5 |

此版不采用，也不把首轮B16的12.22μs当成正式版本已获得的稳定收益。
首轮B16的O_A起点中位数365.46→354.54μs，已提前约10.92μs；
O_A核占用总和/24却从50.83→60.78μs增加，量化与post的交叠也改变。
区间版B16的起点仅359.04→355.98μs，O_A核占用48.87→58.89μs，
最终只剩约1μs变化。更均匀的任务数不等于更少的核内时间，不能只看波次作采用决定。
编排封装为兼容编译器还在原八组路径上增加了一次task_dummy汇聚；
不能把T168数值路径未变说成它的调度完全未变。

所有候选完整源码保留于测试目录`candidates/output_oa_native.py`、
`output_oa_balanced.py`、`output_oa_ranges.py`，正式`output.py`保持Native量化与O_B区间策略。
当前不会继续围绕这套M96列宽做参数扫描。下一阶段应由大段实际供数/核占用证据选择方向，
允许更多T、请求数和KV长度区间，参考CSA分档，但不能用分档掩盖缺少稳定收益。

### 本轮编译尝试和问题转交

- `oa_balanced_v1`使用动态valid N缩短NZ末块，被当前静态NZ维度约束拒绝；
  改成N192/N64两种编译期tile，不扩展权重或搬运新的格式。
- v2把cache policy放进普通inline helper，被`set_cache_policy`作用域规则拒绝；
  v3放回SPMD直接语句后完整编译，真机两档通过。两项均为算子写法调整。
- `oa_ranges_v1`在constexpr分支内嵌manual_scope/parallel，首次Simplify触发
  `control-flow body tail must be YieldStmt when return_vars is non-empty`。
  最小CPU复现已确认：ConvertToSSA把else的yield放在manual scope内部，外层分支末尾仍是ScopeStmt。
  已提交[PyPTO #3115](https://github.com/hw-native-sys/pypto/issues/3115)，附独立脚本、IR和建议。
  没有修改SDK；v2仅在算子中用两个独立producer及构造函数绑定绕过此表达方式。

用户再次允许增加有收益的档位，并明确可参考CSA的档位策略。后续延续此原则，
性能仍唯一按完整L1 incore开始到结束，JIT重复分析到#3113为止不继续处理。

## 2026-10-23：Q_B大T供数分档试验收口，按用户要求暂停

完整说明与原始采样见测试目录`Q_PROJECTION_RANGES.md`、`Q_PROJECTION_SUMMARY.json`。
本轮只在冻结TND候选中增加129≤T≤192的M192/N128策略，原M128/N256的两轮M
改成一轮M，减少权重重复加载。四个head组、K512/K128双缓冲及整数累加保持；
其他T沿用原函数。没有修改公共HCA模块、共享SDK或正式TND算子。
CPU编排及53份incore C++编译通过。两档同进程交错各7窗、直接执行，不用ACLGraph。

- 128K/B28：777.96→771.64μs，七对中四对改善；Q_B单核均值的中位数76.22→60.85μs。
- 128K/B16：522.36→532.20μs，七对中仅一对改善。该T仍用原数学核，但新入口的编排
  已包含额外运行时分支，Q_B/Attention起点后移；尚未隔离确定其具体原因。
- 两档输出与全部cache/state精确保持控制组，Native最终输出容差、保护区及重复调用通过。
  Native严格storage原差异仍保留，不能报告完整精度或30%目标通过。

局部Q_B收益未等量传到全层，且小档回退，所以本候选不采用、不继续扩档。
同时纠正前一条口头假设：当前PyPTO `pl.spmd`默认sync_start=False，
多数HCA任务已无默认整组起跑等待，没有为补写False启动无效测试。

用户明确本轮结束后停止，goal已暂停；两个设备任务均已退出0，没有下一轮排队任务。
本轮源码、失败/未采用结论、完整精度结果和泳道路径本地归档，保留详细中文commit，不push。

## 2026-10-23：恢复优化，按长历史与T区间采用Q_B/O_B大行块

用户恢复本轮工作，允许参考CSA分档。最终仍以完整HCA的L1 incore首尾为唯一候选性能指标，
使用正式第3层W8A8/NZ2和`csa-shared-20261022-logical-codegen`；全部直接调用，不用ACLGraph。
任务ID、完整命令、原始7窗/5窗、精度和阶段统计见测试目录`Q_RESUME_TASKS.json`、
`PROJECTION_RESUME_SUMMARY.md/json`，大产物路径/数量/大小见`PROJECTION_RESUME_ARTIFACTS.json`。

### 先区分核内供数收益与新增编排的影响

上一轮运行时if分档的Q_B在B28局部变快，但B16起点后移。新诊断直接绑定M192/N128，
保持四组head、每组4个Cube、K512/K128双缓冲以及完整INT32累加。
129～192行由两轮M改成一轮M，32MiB权重的GM→L1逻辑遍历次数减半。
这不等于已经用PMU证明HBM字节恰好减半，也不以跨任务的不同时间窗差值证明某条指令的因果。

| 独立大T Q_B候选 | 控制L1 μs | 候选L1 μs | 更快配对 |
|---|---:|---:|---:|
| 128K/B28 | 761.54 | 746.32 | 7/7 |
| 128K/B24 | 709.76 | 703.58 | 5/7 |
| 8K/B32 | 594.20 | 593.66 | 3/7 |

短历史未见稳定收益，不因此全局替换。将策略选择移到Native既有描述符绑定：
读取hidden.shape[0]及已有Host `max_seq_lens`，不新增item/D2H。
两个完整JIT入口通过工厂共享相同层主体、相同36项device Tensor ABI；
T/B、Q/L、position、slot与页表仍由设备描述符提供，计算不固定B×6。
默认入口继续使用原Q_B/O_B，选中的大T入口只绑定不同的投影依赖。
这是一项算子分档，不处理#3113的JIT重复分析问题，不使用内部prepared enqueue。

### 历史长度门槛的反例与收紧

第一版门槛64K试验：128K/B28为779.20→768.42μs，B16/8K正确走普通入口；
但64K/B28从657.84升至682.68μs，5/5配对回退。故正式门槛收紧至128K。
之后Q_B单项在128K/B24为702.80→693.14μs，256K/B28为997.00→988.40μs；
S1、S5及T=129混长边界的真实custom op、KV钩子和按步compact复用均通过。
没有用设备上限或页表容量冒充真实长度，也没有把128K某一点的结果推广为全长度收益。

### O_A负组合与O_B增量分开评价

O_A M192/N128独立在B28为770.38→753.48μs，但与Q_B结合后并未稳定叠加：
B28为771.16→763.66μs，B24为712.06→695.00μs，8K/B32为597.26→600.42μs。
该组合不采用，完整候选仍保留`candidates/output_oa_tall.py`。

O_B采用相同供数方向：M192/N128覆盖129～192行，K仍分为两段4096。
相同区间仍有64份逻辑Cube任务；两段先INT32相加再转FP32，量化、尺度和HC_post不变。
其增量控制**已经含Q_B优化**，不是本轮原版：

| O_B增量对照 | 含Q_B的控制 μs | 再加O_B μs | 更快配对 |
|---|---:|---:|---:|
| 128K/B28 | 767.30 | 760.92 | 5/7 |
| 128K/B24 | 720.94 | 703.62 | 6/7 |

测试工具已支持控制包的两个JIT入口，记录两侧policy，避免把控制包静默固定到普通入口。
最终将Q_B/O_B一起绑定长历史大T入口，O_A保持原策略；其他输入继续普通入口。

### 最终提交版本与验收边界

最终`projection_policy_v4`完整CPU编排及两个入口的incore C++编译通过。
同任务、同SDK、每侧7窗完整HCA结果如下；提升比例使用`control/candidate−1`。

| 最终版本对本轮原版 | 控制 μs | 最终 μs | 性能提升 | 更快配对 |
|---|---:|---:|---:|---:|
| 128K/B24/S6 | 717.94 | 701.02 | +2.41% | 5/7 |
| 128K/B28/S6 | 782.08 | 772.30 | +1.27% | 6/7 |

最终S5/B28也通过真实custom op三次调用、Native钩子顺序和compact按context缓存验证。
两重点档完整输出、raw/压缩KV及state与原控制逐bit一致，重复调用/保护区通过，
Native最终输出仍通过原0.01/0.01容差；Native严格storage既有差异继续保留。
不把独立Q_B、O_B、O_A测量的收益相加，不把普通入口采样抖动写成数学优化收益。
当前仍未完成全八档30%目标、连续状态轨迹和整模型token/PD验收。

### 编译尝试、记录和源码位置

首次工厂集成`q_tall_policy_cpu_v1`在重绑定Q返回值时被前端拒绝：
同一输入轴的T_DYN与tokens别名未通过类型重绑定检查。
改回公共QKV原有的写入manual_dep缓冲、q_ready发布方式后，v2完整编译并经真机验证。
没有修改共享SDK。三组纯CPU缩减探针均通过，尚未独立复现相同触发条件，
因此不把它未经核实写成新的组件bug；完整失败与缩减记录保留以供继续归因。

`q_projection.py`是新增Q_B核；`output.py`新增大T O_B入口；`entry.py`共享层主体；
`native_adapter.py`完成Host描述符策略选择。测试和采集中间值工具同步识别两条入口。
本日志按用户要求从算子子目录上移至`ops/pypto/HCA_TND_MPMD_OPTIMIZATION_LOG.md`，
更新当前文档引用；历史冻结包和结果路径保持原样，避免改写旧证据。
用户要求将本地积累详细中文commit；只提交本会话vllm-ascend文件，不push。

## 2026-10-23：按用户要求转向attention核内流水，参考recipes及Native A3实现

本轮先在`tests/pypto_test/hca_tnd_mpmd_20261022/`冻结包验证，随后按用户允许的区间接入正式attention。
保持公共SDK、正式第3层W8A8/NZ2、直接调用，不使用ACLGraph，唯一性能指标仍是完整HCA
L1 incore首尾。源码来源与算法差异见`ATTENTION_NATIVE_PIPELINE_REVIEW.md`；任务ID、
原始样本、精度、attention AIC/AIV核耗时见`ATTENTION_SUMMARY.md/json`。

### 实际参考范围

GitCode限流且本地部分clone缺少attention blob，随后按本地相同提交
`6d060311ae75f6105fe4216e20566875da5a8838`从GitHub镜像取得recipes CFA/SWA PyPTO、
TileLang sparse attention及AscendC接口说明。旧CFA示例固定四个压缩页、128页大小并用T//B，
不能直接用于本实现的32页、长历史与混长TND。实际供数/缓冲细节核对本仓arch32 SWA/CFA：
K512 softmax块，QK N128/D256/L0K128，PV分块，L0乒乓和三槽跨query调度。

### 两项逐bit候选未整体采用

- PV补齐Right与Acc双缓冲：B16 499.04→506.34μs，B28 766.96→758.38μs；
  B16回退，不能只报B28收益。第一版只有Right双缓冲却失去Acc双缓冲，未上卡。
- Q的K1/2/3驻留48KiB L0A，K0/PV共用16KiB：B16 514.90→520.02μs，
  B28 765.52→768.08μs；输出/cache/state精确但无稳定完整层收益，不采用。

### K512+D256大块候选的首轮证据

保留三CV槽、两步QK提前执行、跨query流水、原gather和device Q/L；
QK按D256供L1、K128供L0；PV用四份L0C完成整个K512后才写GM，减少PV中间写回和Vector更新。
代价是PV重载KV；不能宣称所有搬运都减少。扩大softmax块改变舍入，未当作逐bit优化。

独立B5/S6对密集FP64：128K候选max_abs=0.00110998、RMSE=0.000129745；
16507边界组合max_abs=0.00237789、RMSE=0.000206933，与原版量级接近。
负页、未初始化NaN、部分历史、零有效长度、两次eager精确和尾部写保护通过。
候选对原版attention最大差0.001953125，报告明确记录，未改容差。

完整层首轮同任务7窗：B16 **517.60→497.80μs**，B28 **769.46→748.48μs**，
均7/7配对更快；Native输出通过原0.01/0.01，cache/state相对控制逐bit一致。
attention AIC平均核耗时只分别缩短约7.4/9.6μs，其余完整层变化不能全部算作核内算术收益。
Native严格storage既有差异仍保留，不声称完整精度、整网token或30%目标已完成。
其余档位和动态边界验证继续记录到测试目录汇总。

### 工具链问题与有明确停止条件的尝试

大块v1/v2中，外层循环携带的Scalar偏移进入内层pipeline后被克隆为free variable。
四组纯CPU对照确认：条件yield/直接算术来源在pipeline下均失败，range下均通过；
`LowerPipelineLoops`首次丢失绑定。已提交[PyPTO #3130](https://github.com/hw-native-sys/pypto/issues/3130)，
含完整复现、pass证据及DeepClone区分内部定义/外部IterArg的建议，无SDK改动、无PR。
v3算子绕开该地址别名组合后完整CPU编译及真机验证通过。

每AIV Softmax由H16改H32需要202080B Vec，超过188416B，在CPU阶段停止，未占卡。
QK相邻N128用pipeline深度2的候选已完整编译，两个QK L0C地址0/32768；
另以已含K512版本为控制做B16/B28增量验证，收益不能与不同任务样本相加。
独立探针首次遗漏`--device`导致CLI退出2，修运行器后重提，保留失败记录。

### 多区间接入与最终结果

用户随后明确“不用全局替换，可以多一点档位选择”。正式`attention.py`合为工厂，
K128/K512共用请求聚合、mask、FFTS、PV输出归约和inverse RoPE；没有复制整份算子。
`entry.py`的三个根共享36项device Tensor ABI；`native_adapter.py`只用Native既有Host
max_seq_lens及输入shape选择，不新增设备回读或运行时TMR策略任务。

- 非NZ、max_seq_lens缺失或<131072、T≤32：K128并在L1复用KV。
- NZ、max_seq_lens≥131072、T>32：K512；T129～192叠加已有的大T Q_B/O_B投影。
  T>192仍用原通用投影，尚未补这一区间的新性能数据，不把已有测量推广为实测结论。
- 每个请求实际Q/L、position、slot和页表保持动态；S1/S5/S6不按B×6恢复。

共用模板v4完整三个JIT根及编排/incore C++编译通过。相对本轮原版，正式区间配对：

| 场景 | 原版μs | 正式分档μs | 性能提升 | 更快配对 |
|---|---:|---:|---:|---:|
| 128K/B8/S6 | 351.50 | 347.58 | +1.13% | 5/5 |
| 128K/B16/S6 | 537.84 | 518.42 | +3.75% | 7/7 |
| 128K/B24/S6 | 687.20 | 676.86 | +1.53% | 4/5 |
| 128K/B28/S6 | 771.66 | 765.88 | +0.75% | 6/7 |

短历史/小T保留K128：128K/B4为273.22→266.30μs，8K/B32为603.94→598.44μs，
输出/cache/state与原版精确。它们不是K512收益。首轮8K/B32强制K512的598.60→628.40μs
反例是保留短历史原策略的直接依据。

B28正式收益小于首轮，另做同任务K512原候选→共用模板对照为762.66→748.12μs，7/7更快，
输出/cache/state精确；没有证据支持共用模板导致回退。不同任务的控制不能拼接或相减报收益。
QK跨N128深度2候选在B16/B28均回退，不叠加。所有数据使用同一公共SDK、直接调用L1首尾。

真实custom op验证：S1/B16走default，S5/B28走tall_projection，混长Q=[3,6,6,6,6,6]、
T=33走long_attention；三次服务调用、KV钩子/compact复用、Native输出以及对应冻结数学的
精确比较通过。256K/B16的K512输出也与独立候选精确，Native输出通过原容差。

独立K512混长Q=[1,5,6,1,2]对FP64在128K为max_abs=0.00110998、RMSE=0.000129802，
16507边界组合为max_abs=0.00237789、RMSE=0.000217861；重复及写保护通过。
但**整层混长C128边界仍有遗留**：history=[127,16383,131073,0,16507]、T容量24、请求容量8，
旧K128与K512对Native最大差都为0.0810546875，旧版25702个超差、候选11332个超差。
候选cache/state与旧版精确，重复/保护通过，但Native输出门禁失败，未采性能，不记PASS。
这需要后续定位压缩更新边界的既有整层数值差异，不能用S1/S5或独立attention结果替代。

集成时发现H16归约写入H32宽环的列subview被A3 TMOV拒绝；最小CPU对照确认后，补在
[PyPTO #2717](https://github.com/hw-native-sys/pypto/issues/2717#issuecomment-6363110842)，未重复建issue。
正式统一用6×16环，K128一次写两行、K512逐行写，逻辑顺序和总字节数相同。
v1的constexpr误用、v2未选分支类型检查、v3列子视图失败均保存，v4才是验过的集成版本。
本轮两个组件报告都附复现和修改建议，没有修改/重建共享SDK，没有提交PR。

## 2026-10-23：继续按区间择优，排除累计max小收益并实现双query复用候选

正式HCA仍为`37bbd0f3c`的K128/K512与投影区间策略。本轮未修改正式算子，
SDK仍为`csa-shared-20261022-logical-codegen`，所有新设备执行为eager，无ACLGraph。
用户允许更多区间，不要求全局替换；仍需明确适用边界和完整层证据，不能以单次中值
波动新增档位。候选、失败、队列身份和原始证据位于`hca_tnd_mpmd_20261022/`。

### 累计max与N64/H8：容量改善不等于完整层提速

累计max把running max前移至概率BF16转换之前，PV更新不再缩放新结果；保持三槽、
跨query与原Cube供数。N64/H8版本把输出累积分成8份N64，缩小Vector临时块。
两者均完成三个根的编排/incore C++编译；独立混长FP64误差、失活零输出、重复与保护通过。

| 对照项 | 控制μs | 候选μs | 更快配对 | 结论 |
|---|---:|---:|---:|---|
| 累计max，128K/B16 | 531.32 | 524.38 | 4/7 | 波动明显，未采用 |
| 累计max，128K/B28 | 746.44 | 748.60 | 4/7 | 未采用 |
| N64/H8相对累计max，128K/B16 | 526.54 | 534.10 | 3/7 | 未采用，不扩档 |

Native输出通过原容差，cache/state相对控制精确；严格Native storage既有差异保留。
累计max的B16平均AIC/AIV核时间反而约增加2.8/3.4μs，不能把完整层中值减少归功于
该公式。静态Vec地址覆盖：正式178144B、累计max178272B、N64/H8为157280B；
N64省约21KB仍不足再放一个64KB完整query累积，不能据此强行扩成双query/AIV。
首个独立任务在SetDevice报507033，算子未执行；确认退出后才按同份冻结源码重提。

### 纠正上一节混长C128结论：旧Native参考含不一致的padding

原B5、Q=[1,5,6,1,2]、history=[127,16383,131073,0,16507]、T24/B8容量，
不能把上一节的25702/11332个差异当作有效PTO精度结论。分段同步定位到Native SAS，
此前mHC/RMS/QKV/C128及输入拷贝已完成；只截取输入、不再执行SAS后发现：
Native Q只有15行，而cu_q尾为24，并含三个dummy请求。参考输入不一致，可能越界。
这不是已经证明的Native算子bug，也不能通过修改PTO去复现不合法参考。

测试v5仅裁metadata导致RoPE维度失配；v6把Native metadata/hidden/position/output
统一取真实范围，两侧仍共享相同有效输入、物理cache和初态。PTO继续测试原T24/B8容量。
v6消除MTE和大面积误差，剩**1个**输出超差`[12,0,17]`，max_abs=0.015625、
RMSE=0.001655229；token12是history=0/Q1的冷请求，不能称为C128关闭行错误。

同一生产数学的诊断与真实PTO输出/cache/state逐bit一致。Native attention+Native gates
接PTO尾部完全精确；PTO attention接Native尾部仍复现该点。只替换Native Q或Native
cache都通过原输出容差，两者均为PTO时才复现，剩余来自前置Q/KV差异组合。
固定Native RMS的C128 state最大差5.364418e-7、compressed cache精确。源码也确认
公共mHC已有平衡加法和RMS前BF16舍入，不把这两个已实现点作为新修复。
旧失败完整保留，输入快照仅本地存放，路径/大小见清单，不提交Tensor快照或做hash。

### 大颗粒双query候选：12组×2Cube×4AIV

`build_attention_pair.py`生成完整实验`candidates/attention_pair512.py`，K128入口保留。
每组偶数Cube做两个query的M128压缩QK，奇数Cube做PV；四个AIV各处理一个query的
32heads，UB累积不翻倍。压缩KV被两条query复用，Raw起点不同则分别M64计算。
PV按两个D256段计算，每段两份M128/N128 Acc，共128KB；重复读取概率是明确代价。
信号与压缩KV搬运的净收益必须由完整L1确认，不预记加速。

device pair计划按真实请求前缀配对，保留S1/S5/S6和奇数尾，禁止跨请求配对；
length=0仍发出零输出工作，不访问其cache。12组×三槽×128heads交接区与旧容量相同。
跨pair连续，不能逐pair排空；每槽四个AIV完成旧PV消费后，QK才能覆盖该槽。
跨逻辑核组使用已有A3 signal_notify/wait，10种单写者信号各隔离512B，发布前
PIPE_ALL排空业务写回并维护缓存；等待后处理读方缓存。每次调用由计划任务清零，
MIX明确依赖计划及全部gather/QKV，sync_start准入并关闭early，避免占满核等未执行前驱。
v7还在首次读取plan/header之前显式失效Scalar缓存，以支持下一步同址metadata变化。

独立planner的v2/v3完整CPU通过。完整pair候选v1～v4依次修正本地DSL的INDEX写INT32、
M128/M64同名Tile类型、嵌套分支缺外层yield、H×1向量常量行对齐；均保留失败日志，
这些不是新确认的SDK bug。v5/v6三个JIT根及129份incore C++全部编译通过。
v5设备任务在执行前发现packed被误标InOut、与根Out ABI不一致；v6改为实际仅写Out。

v6独立S6和混长真机均完成：128K/16507、无效页、失活请求、重复和输出保护全部通过，
对正式K512 **逐bit一致**；S6包含跨pair的流水推进。静态地址覆盖Mat262144、Acc131072、
Left/Right各65536、Vec178144B。此统计不是PMU或性能结论。
v7的128K/B16与B28完整层七窗配对已排队，需待输出/cache/state精确及保护通过才采L1。
截至本条记录未取得v7完整层性能，未接入正式算子，不代表30%目标或全部精度已完成。

任务与结果：`ATTENTION_ONLINE_TASKS.json`、`ATTENTION_BOUNDARY_TASKS.json`、
`ATTENTION_PAIR_TASKS.json`、`ATTENTION_NEXT_RESULTS.md/json`。

### 双query完整层结果与同步归因，仍不采用

v7完整层输出、cache/state相对正式K512精确，Native输出通过原容差，重复及保护通过；
但B16为518.04→593.60μs，B28为749.40→896.36μs，均0/7更快，不采用、不扩档。
L1显示计划任务仅约4/6μs；attention的72核平均执行时间(含内部等待)约148/223→207/351μs。
这定位到配对阶段，但不能直接把差值全称为Scalar等待或HBM代价。

v8仅把100tick退避改为0，仍对同任务正式K512：B16为512.88→605.06μs，
B28为756.52→867.38μs，均未超过正式版，说明退避并非唯一原因。SDK未改。

v9收紧环的实际读写依赖：QK只覆盖score(k)，因此只需四个AIV已经发布P(k)，
无需等旧PV累加完毕。Vector在第k+2拍已消费PV(k)，才会在第k+3拍发布P(k+3)，
PV Cube又等全部P(k+3)才写V(k+3)，所以P/V缓冲也安全；删除四份冗余PV消费ACK。
完整S6仍与控制逐bit一致。B16为528.90→593.66μs，B28为737.94→820.36μs，
仍未采用；不同任务中值不能相减声称v9单项净收益。混长未扩测，不夸大v9覆盖。

另纠正此前对early的保守推断：公共Simpler TMR的`orchestrator.cpp:535/573`和
`scheduler.h:882`表明allow_early_resolve是生产者对消费者预派发的提示，
sync_start gated cohort仍在全核可运行且前驱释放后才启动。关闭该提示不是安全要求，
却会增加输入/输出的启动空隙。v10恢复plan和attention的early，保留sync_start及全部依赖；
B16/B28已提交，尚待结果，不能把此前人为关闭的代价全部归咎于配对数学。

### 等待设备时核对gather的串行mask构造

正式压缩KV聚合每页用32次Scalar UB写入生成mask，新增独立Vector mask候选：
用精确FP32列号和有效长度生成0/1，按[页数,32]整行拼接再reshape，保留负页、尾页和
未初始化KV的保护；KV地址与搬运完全不变，也不引入新的task或外部适配。
三个根的完整CPU后端通过，独立混长/无效页以及B16/B28完整层七窗已经提交。
候选位于`candidates/attention_gather_vector_mask.py`，任务见`ATTENTION_GATHER_TASKS.json`。
在真机输出与L1证据出来之前，正式源码仍不改变，也不会全局替换。

## 2026-10-23：本轮attention候选收口，用户随后要求继续

用户曾明确要求“本次优化结束之后停下来”，在设置paused前又指示“继续优化”。
因此撤销暂停安排，目标保持active。本节只归档已完成的在途候选和Vector mask分档收口，
没有修改SDK、没有push。正式HCA仍为`37bbd0f3c`已验证的区间策略；
此前过程已在`2f0dc7338`中文提交，本节记录之后取得的最终证据。

### 恢复early后双query仍不成立

v10恢复plan/attention的producer early提示，sync_start及全部数据依赖不变。
128K/B16为510.24→563.40μs，B28为756.10→836.06μs，均0/7更快；
输出、cache/state对正式K512精确，Native输出、重复及保护通过。关闭early不是全部原因，
配对核内的通信、角色等待与重复供数成本仍抵消收益。本轮不采用，不再扩档或调参。

### Vector mask：初版大工作量正向，核内分档收口未通过性能门槛

初版Vector逐页生成mask对独立密集FP64、无效页、部分尾页、失活、混长、重复和写保护
通过；与正式K512逐bit一致。完整层B28为772.92→762.22μs，7/7更快，配对节省中位数
9.36μs；B16为520.48→517.60μs，但仅3/7更快，配对节省中位数-1.92μs，证据不足。

因此补了同一方向的区间版本：`gather_work_tile>=5`时Vector，否则Scalar。门槛来自请求
容量与压缩页表容量在48个AIV上的工作划分，不固定B点、不引入device到Host取值。
两个路径共用逐页mask网格，选择位于核内。最终实测如下，都是同任务控制的完整L1首尾。

| 场景 | 正式控制μs | 分档候选μs | 结果 |
|---|---:|---:|---|
| 128K/B24/S6 | 692.80 | 692.04 | 收益很小 |
| 128K/B28/S6 | 769.52 | 768.50 | 未稳定保住初版收益 |
| 256K/B16/S6 | 659.04 | 655.28 | 收益很小 |
| 128K/B16/S6 | 517.40 | 521.16 | 回退 |
| 8K/B32/S6 | 597.52 | 612.46 | 明显回退 |

五档及独立FP64边界均保持原数学与全部控制storage精确，Native输出、重复、保护通过；
但性能不足以接入。核内if和共享的新网格布局不能视为“小档完全保留旧代码”，因此没有
为了保留初版B28正向数据而全局替换。恢复后可评估编译期分离原Scalar布局与Vector路径，
以区间完整L1再次决定是否采用；本轮到此为止，未实现下一版或承诺它会更快。

### H32 UB容量尝试止于CPU

`build_attention_reset_alpha.py`把首块旧输出累积的系数置零，意图避免两份32KiB归零Tile
及其if合流，同时只扩大softmax至H32，保留H16输出发布。完整CPU编译仍报Vec202080B
超过188416B，和此前H32相同；因此没有上卡，没有声称节省64KB，也没有据此提交SDK bug。
该等价思路依赖累积有限，数值及性能都未验，不进入正式算子。

### 收口时状态与复现位置

- 本节任务清单中的设备任务全部终态，没有留下排队/执行中的旧任务；新尝试使用新的冻结包和任务记录。
- 所有候选、失败与负结果保留在`tests/pypto_test/hca_tnd_mpmd_20261022/`，统一汇总为
  `ATTENTION_NEXT_RESULTS.md/json`，分支/协议解释在`ATTENTION_NEXT_STAGE.md`。
- 原始泳道、输入快照、冻结依赖和编译产物只本地保留，清单提供路径/大小；没有hash扫描。
- 本轮新增正式性能优化为零。完整八档超过Native30%、冷S1剩余一项输出差异、严格
  Native storage、连续状态轨迹与整模型token等目标仍未完成，不能把暂停写成完成。
- 用户续优已撤销暂停要求，未设置paused；保持原目标active，未push。下一步先让核外选择
  独立Scalar/Vector gather核，避免核内if及共享新布局使小档也改变，不重新提交旧设备任务。

## 2026-10-24：核外分档及直接KV收口，按用户再次要求停止

上一轮结果已中文提交`9b70c8483`，未push。用户恢复后仅推进核外gather分档和直接KV；
随后再次要求“本轮优化之后停下”。现本轮设备任务全部结束，归档提交后暂停目标。
正式HCA未接入任何新候选；完整30%目标及既有精度/整模型遗留均未完成。

核外分档v1因局部Tile越作用域引用在CPU失败，重复整模型IR约42MB，仅本地保留。
v2完整CPU通过，但Scalar核仍残留无消费者Vector初始化；v3源码体完全分离后，
Scalar C++不再包含Vector网格/列号，TCI=0、TEXPANDS=2。TMR只提交一种gather，
device Q/L、页表及原数学不变，没有增加设备task，L1依赖数与控制相同。
但完整层128K/B28为757.08→764.48μs、8K/B32为584.24→597.88μs、
128K/B24为673.54→681.60μs、256K/B16为664.48→663.82μs，仍无稳定收益，不采用。
输出/cache/state对控制精确，Native输出、重复、写保护与独立混长FP64边界通过。

直接KV候选把长档每128行的128KiB完整聚合窗口换为64B页计划及全层64KiB零模板。
计划不读cache内容，可提前执行；attention另外显式等待C128真实写回。完整连续四页
直接加载到L1，其他布局预清零后逐页覆盖，仅读取合法有效行，保留NaN尾区保护。
保持原K512/D256、两步预发、三槽、跨query及FFTS，不引入新的GM通知协议。
Native arch32的Raw有DataCopyPA，compressed实际用kvMergeGm；本方案是独立候选，
不能声称Native已直接读compressed。GM字节减少也不能当作HBM实测或性能收益。

v1修本地INDEX→INT32写入；v2的Mat if合流产生不支持的Mat→Mat TMOV；
v3改用同一份预分配L1由两分支原位填充，三个根完整CPU后端通过，未改共享SDK。
独立S6、混长、负页、部分尾页、失活、重复、保护在128K/16507通过，对正式K512精确。
另随机重排物理页并反向更新页表，逻辑KV与独立FP64参考不变，该非连续页检查也精确。

最后两个完整层任务均通过控制输出/cache/state精确、Native输出原容差、重复和保护后采七窗L1：

| 场景 | 正式控制μs | 直接KVμs | 增加μs |
|---|---:|---:|---:|
| 128K/B16/S6 | 504.08 | 641.22 | 137.14 |
| 128K/B28/S6 | 745.84 | 961.44 | 215.60 |

性能明显退步，不采用、不扩档、不继续调参。未完成Scalar/MTE/缓存成本的独立归因，
不将回退全部归因于其中某项，也不把本候选失败推广为所有直接读取策略都无效。

组件方面，CPU最小对照确认无循环/phi的未使用Tile初始化仍生成TCI/TCVT/TEXPANDS：
clean各0次，unused和真实使用的反例各1次。已提交[PyPTO #3136](https://github.com/hw-native-sys/pypto/issues/3136)，
含完整复现与建议，无PR、无标签；只证明PTO/生成C++，未声称最终机器码或微秒损失。
这与#3037的循环死状态相关但不假设根因相同；没有重建或修改公共SDK。

收口详情：`tests/pypto_test/hca_tnd_mpmd_20261022/ATTENTION_DIRECT_KV.md`；
任务清单、全部样本、CPU失败与本地大产物路径/大小已汇入`ATTENTION_NEXT_RESULTS.json`。
当前没有本轮在途设备任务。保存正式基线、所有候选和负结果后暂停；未push，目标未完成。

## 2026-10-24：按CSA性能规范完成HCA八档三列复测（不恢复优化）

用户要求按 `CSA_PERFORMANCE_TEST_SPEC.md` 测Native、HCA-MPMD-AICPU、HCA-MPMD-L1，
并明确：禁止ACLGraph仅针对PTO；Native须启用静态编译、SuperKernel、NZ、ACLGraph。
本轮生产HCA仍为37bbd0f3c的K128/K512及Q_B/O_B区间策略，未修改算子、数学或SDK。
公共工具链使用validated的csa-shared-20261024-group-dims。

八档为128K/B1、256K/B16、128K/B8/B16/B24/B28、8K/B32和128K/B16混长T56。
前七档S6，混长保持规范的16请求固定列表，不补成B×6。正式第4层（index=3）
W8A8/NZ2，seed1024，同一Native TND fixture、页映射与状态起点。
Native每档30窗，PTO两种边界各10窗，完整样本取中位数，不剔除慢窗。

Native实际检查静态编译返回成功、已安装静态包、SuperKernel优化成功、
四张主投影权重格式29、多stream和单一捕获图。PTO使用直接调用；先验证真实
服务接管、按步compact metadata与KV钩子，再采原始PyTorch trace和L1。
服务trace初始外框包含9～14ms的前导/派发间隙，经与CSA完整算子计时边界核对，
正式AICPU列按唯一PTO根op和外框结束前设备同步划分活动，保留全部PTO stream、
DMA、Event、AICPU/AICore；原服务跨度另外完整保留，不以删慢样本制造收益。
根op之前唯一Native compact producer已完成的条件在80个窗口中全部核实。
L1为PTO所有incore首尾，不累加task。两列取自同一实现/fixture/服务调用，
采集器分开运行；未移动PTO内部Q/L、slot或有效块推导。

| 场景 | Native μs | MPMD含AICPU μs | MPMD L1 μs | (Native−L1)/L1 |
|---|---:|---:|---:|---:|
| 128K/B1/S6 | 241.50 | 465.62 | 223.92 | 7.85% |
| 256K/B16/S6 | 747.25 | 947.50 | 649.43 | 15.06% |
| 128K/B8/S6 | 398.62 | 681.25 | 371.53 | 7.29% |
| 128K/B16/S6 | 590.00 | 812.50 | 543.49 | 8.56% |
| 128K/B24/S6 | 767.62 | 1017.88 | 671.76 | 14.27% |
| 128K/B28/S6 | 817.38 | 1009.38 | 737.82 | 10.78% |
| 8K/B32/S6 | 641.62 | 980.38 | 612.83 | 4.70% |
| 128K/B16/混长T56 | 456.50 | 734.00 | 451.01 | 1.22% |

八档Native输出沿用atol/rtol=0.01均通过，PTO输出/cache/state对冻结正式实现
逐bit相同；每个PTO性能窗口执行后在计时外复核状态与保护区。原有Native
cache/state严格1e-5差异继续单列，不能宣布全精度或整模型token验收通过。
该比例比较Native外部设备跨度与PTO L1，不是端到端吞吐提升，也未达到原30%目标。

共19个任务：1个CPU完整后端编译，8个PTO采集，8个成功Native采集，2个Native启动失败。
第一次历史脚本缺static_kernel字段，在新测试入口显式补True；第二次CANN把
长绝对目录编码进shape_info文件名，超过NAME_MAX，仅改用私有短编译工作目录后完成，
未关闭任何优化，也未改共享CANN载荷。失败报告与日志保留。

结果、完整样本、任务号、版本、状态差异及大文件路径/大小见
`tests/pypto_test/hca_performance_20261024/`。`validate.py`从原始Native CSV和
PTO L1重新计算八档跨度，核对两侧fixture、环境、样本数、中位数和百分比通过，
该核对没有再次执行设备。大trace、静态包、编译产物和权重不提交，不做hash扫描。
本轮只提交详细中文测试记录，不push；原优化目标保持暂停。

## 2026-10-24：HCA TND整模型候选完成16卡B16真实图重放，停在token精度讨论点

用户要求参考CSA-MPMD-TND进行完整vllm-ascend接入，有疑惑先停下来讨论；
并澄清“PTO不使用ACLGraph”仅限单算子性能测试，整模型应验证真实图重放。
本轮仅切换20层C128 HCA，21层CSA、SWA及FFN保持Native，原性能优化目标未恢复。

接入候选基于已提交15b64665a冻结框架，新增显式PyptoHCATND模型、HCA图服务、
C128 compact容量及地址/策略合同、捕获历史选择与逐层实际路由观测。复用原HCA
计算入口和公共TND图合同接口，数学、tile、共享SDK均未修改；没有带入另一个
CSA会话的未提交测试改动。代码在tests/pypto_test/hca_tnd_model_validation_20261024
的生成器、candidates和MODEL_ADAPTER.patch，正式模型入口尚未切换。

单卡过程：v01入口漏接队列注入--device，未执行算子；v02沿用CSA heap配置导致
ring2申请22044672B超过16MiB；v03增大ring2后B16和混长11步通过，但B24在ring0
申请20372480B超过16MiB。设备日志已给出明确容量不足证据，未归因为编译器bug。
v04采用heap=[64,256,64,64] MiB、task_window4096后，6类29步全部通过，
任务task_20261024_132445_31180867686 exit=0。覆盖B16/B24、混长T56、S1/S5/S6、
B1、C128关闭边界、历史131071/131073/131199、7请求和1请求尾批、padding、
请求重排及返回满批；图输出/cache/state对独立PTO eager逐bit一致、保护区通过，
replay期间Python提交数不增加；每类场景Native输出原0.01容差通过。

16卡任务task_20261024_132810_32584376495（冻结v04）exit=0，TP1、DP/EP16，
正式W8A8/NZ2权重及既有mixed128k_fixed离线KV。功能配置沿用CSA：
FULL_DECODE_ONLY/npugraph_ex、deterministic level1、HCCL1800、AIV展开、event1、
EPLB关闭、预算0.92、静态编译/SuperKernel关闭；不做模型性能比较。
256请求共49152输出token、零抢占；每rank33个真实满批S6 step、18个真实尾批step，
其中15步padding。全部51个真实长历史FULL step的20层HCA均真正replay，
无Python重新派发。另有1步正常S1 eager及34步短dummy图，单独记录，
不把dummy或捕获计数当作真实业务步骤。

对Native历史正确性参考做源码与配置审计：原数学路径未变，新增34行框架逻辑
仅在PTO TND模型/合同/回退标记下执行；复用同权重、bank、问题、预算和推理配置
的Native结果，仅比较正确性，不跨版本比较性能。报告NATIVE_REFERENCE_AUDIT.json。

精度结论：64/256请求完整token一致，其中简单问题32/32；其余192个复杂请求
有分歧，q08/q12/q14在第0个输出token即不同。HCA的16个DP rank输出完全相同。
平均接受草稿数HCA=3.753067484662577，Native=3.806501547987616，DSpark事件
不全相同。单卡图/eager精确一致与模型功能通过不能代替Native token一致，
当前未定位差异是数学误差传播还是接入问题，不擅自接受另一个会话的较宽口径。

按用户要求停止新设备任务，未提交B24，建议先定位首token分歧后再决定继续范围。
五次任务均已终态；完整命令、日志、29步单卡结果、16rank输出及接受事件、
源码/SDK清单、问答全文和首处分歧已归档。大权重、KV、二进制和编译快照本地保留，
不做hash扫描。阶段性详细中文commit，不push。

## 2026-10-24：定位真实模型QKV舍入差异，按用户要求保留FP32并以答案质量验收

用户要求先查精度；之后明确无需强制恢复Native BF16边界或逐token对齐，最终看
decode答案是否符合要求，并要求记录该差异。当前正式HCA计算代码没有修改，
恢复边界的代码只作为私有诊断候选保留。没有修改公共SDK，没有恢复性能优化。

首token控制：在相同模型壳、正式权重/离线KV、TP1/DP/EP16下分别运行Native和
原PTO HCA，两侧均复现各自历史256/256个首token。首轮实际为B16/S1/T16 eager，
尚未进入ACLGraph；输入token、位置和Q/L一致。rank0按同初态执行20层Native/PTO
对照，随后恢复实际PTO写入页和输出，逐bit确认恢复、输入不变、输出不变。
首个HCA层index3输入逐bit一致，输出max_abs0.00390625、RMSE0.000391288586，
57063/262144个元素不同但原0.01容差通过，后续误差逐层累积。
q08在Native选token9854（29.0对28.75），PTO两个候选均29.125并由argmax选较小ID3648；
这解释了首token分叉，不能据此判断哪种回答更好。

源码与真实输入实验证明五处边界不同：QA投影后、KV投影后、KV归一化后、
QB反量化后、Q每head归一化后。Native分别写成BF16，原融合版本继续FP32。
候选在原tile中加入FP32→BF16→FP32，复用已有生产QKV函数，不加Host取值或新task。
真实第4层Native归一化输入的三组实验（相同输入/Native参考已事后CPU核实）：

| 候选 | QR INT8不同元素 | KV不同元素 | Q不同元素 | Q RMSE |
|---|---:|---:|---:|---:|
| 原FP32融合 | 781 | 2309 | 381955 | 0.00537905655 |
| 只恢复QA BF16 | 0 | 2309 | 147473 | 0.00293782088 |
| 恢复五处BF16 | 0 | 0 | 77 | 0.000112857527 |

五处恢复后q08的Q/KV/QR逐bit一致；QR scale剩4行末位差，最大9.31322575e-10。
余下Q差异为row7/head24的75个、row10/head48和row14/head18各1个。
Native QB/RMS追加采样与原诊断输出一致，中间值留本地，当前不继续追求77→0。
候选S1/S6的10步完整单卡图/eager及保护区通过，Native输出原容差通过；
尚未进行候选16卡模型、答案质量或性能测试，未合入正式算子。

本轮探针第一次错误地把生产ND wkv声明成NZ，被编译器拒绝；只修诊断根签名，
不作为SDK bug提issue。其余单卡/模型诊断均完成，所有命令、失败日志和终态保留。

用户新口径已写入报告：保留FP32通常减少舍入；恢复BF16仅提高Native数值相似性，
不能推导出答案质量更高。Native也不是题意真值。仅用已有192token输出做CPU核对，
q02正确首行为[1,9,5,24]，Native/HCA分别给[1,20,92,63]/[1,20,100,71]，两侧均错；
q03两侧首条账本记录也不正确。两个简单重复问题仅前缀正确，192token没有完成
80/81次要求。大多数复杂题尚未生成完整答案，不计算整套准确率，也不把16个问题
重复16rank说成256个独立质量样本。当前不能宣称全面答案质量通过。

详细记录在tests/pypto_test/hca_tnd_model_validation_20261024/HCA_NUMERICAL_DIFFERENCES.md，
QKV_BOUNDARY_COMPARISON.json、FIRST_TOKEN_ANALYSIS.json、ANSWER_AUDIT.md/json及
precision_evidence/保留全部小证据；候选改动另存QKV_BF16_BOUNDARIES.patch。
大Tensor、冻结源码和编译产物本地保留并记录路径/大小，不hash、不提交权重或bin。
本轮到记录收口，不为Native逐项对齐追加整模型测试，未扩B24，详细中文本地commit，不push。
