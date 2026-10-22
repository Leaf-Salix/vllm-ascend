# HCA TND MPMD 实现与优化过程

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
