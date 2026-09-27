# CSA Indexer：Native 与 PTO 实现差距

更新：2026-09-27。按用户要求，后续性能优化先集中到 Indexer。
基线为性能版 v7（`9516acbe`），Native 为当前 release 的 A3 `arch32` QLI。
v7基线量测见[之前七档对照](results/csa_native_cube_matrix_20260927/README.md)，
当前统一实现的实测见[V10七档结果](results/csa_split_optimization_20260927/INDEXER_PROGRESS_V10.md)。
测量使用正式 layer 4 权重、合成历史，不能代替整模型 decode forward 或 token/DSpark 验收。

## v7基线已确认的差异

| 环节 | Native A3 QLI | PTO v7 性能版 | 性能含义 |
| --- | --- | --- | --- |
| key 寻址 | 直接读 Native 分页 cache；同请求最多4个 query 共用一次 key 装载 | 入口 Torch 拆成连续 key/scale，Score 内仍逐 query 读取 | 连续读消除了小页寻址，但没有消除 query 间重复加载；拆分/写回成本另列 |
| QK 分块 | L1 的 M256 包含4个 query；L0 每次 M128、N128、K128，两个 query 合并计算 | 每次 M64、N128、K128；N768 逻辑块内6个小面板 | 同样总乘加量，PTO 发起的 QK 次数更多，M 更小 |
| query/系数驻留 | 一个 query 组跨多个 S2 块复用，key 在组内复用 | 每个 query/8192候选 leaf 重新装入，query Left/系数 Left 在当前 N768 块复用 | 目前只表达块内复用，还未达到 Native 跨 query / 跨块的复用范围 |
| 长上下文 head 规约 | INT32 QK 经 FIXPIPE 做 ReLU、/1024、FP16 写 L1，再做 FP16×FP16→FP32 Cube WS | v7 已采用同类数据流，GM只传每候选一个FP32结果 | 已消除 v4 的64行FP16中间张量；不能再把32倍中间载荷差距当作 v7 的问题 |
| 短上下文 head 规约 | 同样用片上 FP16 QK 和 Cube WS | 仍将64行INT32分数交给Vector，转FP32、乘系数并 `col_sum` | 8K未受这轮长上下文优化影响，存在单独的搬运和Vector工作差距 |
| 系数准备 | QLI 内部 `ProcessVec0`，配合 query 组和跨核流水 | 单独48个SPMD任务，Score等待它们与cache写回 | 增加一次任务组与GM交接；不能把整个系数窗口直接从总时间扣除 |
| Cube/Vector 分工 | 两个AIV按query分工；外层2048候选块双缓冲，Cube算当前块，Vector处理前一块 | 两个AIV按候选区间切分，同一query的两个半leaf分别排序 | PTO每query有两份Top-K候选，需要额外合并；调度和局部性不同 |
| Top-K | 2048候选块分段排序，UB内持续维护Top512；跨核切分时再合并 | 分数写GM；长路径每4096半leaf加载/排序后写pair arena，再发48个query merge任务；短路径独立publish任务读分数并排序 | 分数/候选反复落GM、独立任务和全组依赖都是待处理项；Native也有必要的跨核归并，不能说它完全无workspace |
| 工作分配 | metadata 按query组、S2块和估计成本分核，QLI内部连续执行 | 24个逻辑SPMD块动态分配物理核，按query×leaf循环 | 已观察到部分窗口17个物理AIC接24块、形成两波；也有24核正常窗口。需分开看核内成本和任务排队 |

以 S6、压缩历史32768、INT8 head_dim128 为例，忽略尾块与缓存命中：
逐query读取 key 的逻辑载荷为每请求 **24 MiB**；Native 4+2 分组为 **8 MiB**。
两个query分组可降到 **12 MiB**。这是源码推导的装载量，不是实测DDR带宽或速度预测。
M64 改为 M128 后 QK 发起数减半，算术乘加量不变；WS 仍需逐query完成。

## 证据边界

当前表中的 Native QLI 是独立 PyTorch profile 的单个设备kernel；PTO Score→publish 是
独立 DFX 的首个Score Worker到最后Top-K Worker，含排队并与其他CSA分支交叠，尚不含前置系数任务。
只比较两侧覆盖了哪些工作与数量级，不把两份trace的时间相减当成确定可回收的整层收益。
PTO不同任务窗口有交叠，不能相加；Worker与Scheduler是同一批任务的不同视角，也不能相加。
主性能判据仍为无profiler完整CSA本体均值及p50/p95，入口/出口和完整路径单列。

## 与 pypto-lib 的关系

参考本地已记录的 `2164563`：短路径是 N384、INT32分数传Vector并做FP32 head规约；
长路径是 N768、两槽GM FP16分数，仍由Vector规约。两者均逐query处理并使用半leaf森林Top-K。
上游长分支条件为 B≥64 且压缩历史≥32768，当前矩阵不会命中其原条件。
v4移植并扩大启用范围；v7进一步采用 Native 的片上Score与Cube WS。
因此仅照搬 pypto-lib 不能自动得到 Native 的query复用、流式Top-K或固定核内工作分配。
v8～v10已借鉴 Native 的 M128 和query复用，继续沿用已接好的连续cache；精度版算术不变。

## v10仍存在的差距

V10是累计保留改动的一套实现，调用者不按档位选择V8/V9/V10版本。
`indexer_score_topk_forest`按本batch最大压缩历史是否达到2048行选择Cube/Vector；
本矩阵8K和128K都走Cube路径。`indexer_score_topk_native_cube`按query数是否小于48选择
leaf优先或query组优先的工作分配，Cube与Vector侧使用同一规则；S6/B4命中前者。
这些条件位于PTO算子内，报告中的历史label只标记数据采集批次。

- key已经按两个query共用，QK已为M128；S6的逻辑key装载量从24降到12 MiB/请求，
  相比Native 4+2的8 MiB仍有差距。不能继续把v7的单query重复读取作为现状。
- 压缩历史≥2048已采用片上FP16 QK与Cube head规约；8K不再搬运64行INT32给Vector。
  每个query仍由两个AIV分候选区间处理，各自写半leaf Top-K，再用独立任务归并。
  分数/候选GM交接、系数准备任务与调度等待仍在；Native的UB流式Top512尚未移植。
- B4完整leaf负载已均衡，但128K/B16仍有无profiler长尾；短路径也有物理核派发和独立merge等待。
  核内收益不等于整个Indexer窗口已赶上Native，需保留当前四窗口和全部主计时样本。

单leaf融合发布候选v11遇到输出ABI及scope别名的编译问题，未产生设备结果，已撤回到v10。
最小候选与恢复条件见[v11记录](results/csa_split_optimization_20260927/v11_native_publish/README.md)。

## 执行顺序与验收

1. 两个query共用key、合并M128 QK已保留，S6按2+2+2分组；独立Torch公式与128K三档已验证。
   后续以v10为起点，不重复同一子块和已覆盖档位测试。
2. 再评估4+2 query组共享key及跨S2块驻留，记录L1/L0占用和生成同步；不以“设置双缓冲”代替实际流水证据。
3. 8K片上head规约已保留；下一步减少分数落GM和独立Top-K归并的成本。
   如改变Top-K合并/tie规则，单独记录数值策略与索引结构检查，不归类为数值中性改动。
4. 核内变化与调度长尾分别量测。若核内已接近Native但长尾仍在，再针对具体依赖/物理分配修复，
   不做无依据的任务重排，也不把正常窗口当成稳定获益。
5. 保留候选后完成受影响边界与整模型输出看护；最终仍须满足token/DSpark和整模型性能合同。

源码依据：

- [Native实际调用与ABI](../../vllm_ascend/attention/dsa_v1.py)：`_indexer_qli`；`return_value=False`，PA_BSND，压缩率4。
- [Native Cube](../../csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch32/quant_lightning_indexer_service_cube.h)：`ProcessQk`、`ComputeMm1`、`FixpSToL1`、`ComputeWs`。
- [Native Vector](../../csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch32/quant_lightning_indexer_service_vector.h)：`ProcessVec0`、`ProcessVec1`。
- [Native流水和分块](../../csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch32/quant_lightning_indexer_kernel.h)：`M_BASE_SIZE=256`、`S2_BASE_SIZE=2048`、`ProcessBaseBlock`。
- [Native分核](../../csrc/attention/vllm_quant_lightning_indexer_metadata/op_kernel_aicpu/vllm_quant_lightning_indexer_metadata_aicpu.cpp)：`CalcCostTable`与S1G/S2任务划分。
- [PTO实现](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer.py)：`indexer_score_topk_native_cube`、`indexer_score_topk_forest`、`indexer_topk_half_leaf`。

## v8补测发现的工作分配问题

128K/B8两个query分组后，本体947.36→894.54 μs；B16为1458.16→1292.23 μs。
B4则749.03→766.65 μs，发生回退，不能用B16收益覆盖它。
S6验证包含新增压缩候选，使总范围有4个完整8192 leaf和第5个极短leaf。
B4共有12个query组，按 `query_group * 5 + leaf` 再以24步长轮转分派时，
24个逻辑worker承担的完整leaf数为：4个worker仅1个、16个worker各2个、4个worker各3个。
因此实例数接近均衡并不等于工作量均衡。
改为先遍历leaf、再遍历query组后，48个完整leaf可恰好分到24个worker，每个2个，短尾单独分配。
该工作量分布由源码/任务映射推导，不等于物理核重复分配的长尾已经解决。

v10单卡已验证该映射：B4本体降至716.345 μs，较v8的766.653 μs恢复并超过v7的749.032 μs。见验证日志§181。
