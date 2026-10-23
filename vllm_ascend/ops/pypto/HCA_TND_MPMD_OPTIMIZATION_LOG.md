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
