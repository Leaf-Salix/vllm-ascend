# DSV4 CSA 历史测试记录

更新日期：2026-09-24  
记录分支：`Leaf-Salix/vllm-ascend:dev/pypto-dsv4-csa-v0.25.1rc1-cann9.0.1`

被测运行时代码基线：`778bf158b53e3a7dbd86065d775115e3e6d01d3e`
本轮归档开始前 HEAD：`cf271b359be3ae3c72f883733721aaa296f69955`（仅测试历史文档提交，运行时代码与 `778bf158b` 相同）

本文汇总当前分支提交历史中可归属的主要功能、精度、回归和性能测试，也记录以当前分支源码为基线的隔离试验。没有证据的项目明确标记为未测；不同测试口径不合并比较。

## 当前结论

- 0.25.1rc1 / CANN 9.0.1 环境中，三层 `official-l3` 模型的原生、CSA eager、CSA graph 冒烟均运行成功；连续64-token请求也完成，CSA 确认命中。它是三层裁剪模型，不是完整 DeepSeek-V4。
- 早期直接 kernel S1/S6 对拍的状态/压缩缓存误差很小，但 output relative L2 为 1.21%–1.29%，超过既定 `torch.allclose(rtol=1e-2, atol=1e-2)` 门槛。
- 大幅重构提交 `fb734041c` 的 8K/32K 单层输出误差降到 0.0744%/0.0703%，但 replay 比旧实现慢约10.9%/31.8%。随后最后一次运行时代码提交 `778bf158b` 恢复原有 CSA kernel；不能把 `fb734041c` 的精度成绩记到 `778bf158b`。
- 运行时代码基线 `778bf158b` 的 S6、128K、BS4/8/16/24/32/40 单层 graph 数据已采集。output relative L2 均约1.81%–1.85%，精度未通过；CSA 在 BS8 及以上慢于原生，BS40 约为原生的1.98倍。CSA 的 BS8–40 诊断需把每个 ring heap 提高到2GiB，不能代表默认配置容量。
- 16卡完整 DeepSeek-V4 的短生成和实际 S6 CSA 命中已完成；采集到的纯 decode 步骤是每 rank B1，不是单卡 B4，也不能作为固定 GBS64 的持续 decode / 性能验收。观察到平均接受 draft token 数为3.0，目标3.8未达到；输入是重复合成文本。
- 因此当前分支达到“功能路径能运行、CSA能命中”的阶段；尚未通过整层数值精度，也没有证明固定 BS4/GBS64、目标接收步长或完整服务性能达标。
- 最新单因素边界对拍显示，Q-A BF16 舍入可把 heads relative-L2 从1.9196%降至0.7586%；O-B改为完整8192维共享量化 scale，可把固定同一 heads 的 O-proj 误差从1.5741%降至0.5943%并通过该局部阈值。但完整attention输出仍为2.0639%，heads误差仍为1.9196%，整层精度没有通过。详见[DSV4 CSA 数值边界单因素对拍（2026-09-24）](DSV4_CSA_PRECISION_BOUNDARIES_20260924.md)。

## 版本与提交对应

基线为 vLLM-Ascend `v0.25.1rc1`（`9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`）。以下为当前分支第一父提交线：

| 提交 | 变化 | 对应验证与解释 |
| --- | --- | --- |
| `19f784979c05b467e96745912330cd88c1c0bc33` | 将 PR #5 来源的 PyPTO CSA kernel 迁入 0.25.1rc1；早期 ABI 为40个 tensor 参数 | 迁移期 CPU 合约、参数绑定、原生/CSA服务分流测试及 NPU 冒烟在该代码阶段进行；详见下方“初始功能验证”。 |
| `873984b5c0f81dd935a3af2bd708920868bc8dc0` | 修正文档中的共享 state / KV 分配描述 | 本提交本身未增加 kernel 行为；早期 `official-l3` 功能测试记录在此阶段分支版本。 |
| `8990a7d8e320c5b9a8119bb4272441bd1c5c5491` | 让服务入口在均匀 S1–S6 下接入 DSpark 验证请求 | CPU 合约扩展通过；S6 NPU服务作业曾提交，但不能把早期直接 kernel S6 精度测试当成此提交的服务验收。 |
| `fb734041c8a45f4a35eafe87f3c403ad0191fd64` | 将 CSA 接入原生 DSA 实现，改为52-tensor kernel ABI、消费原生 slot metadata/cache layout | 做过旧/新单层精度及 graph 对照；精度改善、性能退化，见“native DSA 重构对照”。 |
| `778bf158b53e3a7dbd86065d775115e3e6d01d3e`（最后运行时代码基线） | 按要求恢复 `8990a7d8e` 的原 CSA kernel 数学/40-tensor ABI，同时保留必要的原生 DSA 接口校验与生命周期组织 | CPU合约、原生与 CSA 输出/cache 复现性、S6/128K 单层 profiling及16卡功能路径有记录。该代码基线的整层精度仍未通过。 |
| `cf271b359be3ae3c72f883733721aaa296f69955` | 新增历史测试文档、目录索引和后续记录要求；不改运行时代码 | 文档提交。之后在该运行时代码上完成了独立精度边界对拍，见日期报告。 |

每个提交没有独立、同口径硬件结果的，不能继承后续提交的测试成绩。尤其 `fb734041c` 和 `778bf158b` 是不同 kernel ABI/计算路径。

## 测试环境

### 初始 0.25.1rc1 / CANN 9.0.1 验证

| 组件 | 版本/备注 |
| --- | --- |
| vLLM-Ascend | 0.25.1rc1，Leaf 当前分支源码，A3构建 |
| vLLM | 0.25.1，`VLLM_TARGET_DEVICE=empty` 构建 |
| Python | 3.12.14 |
| Torch / Torch-NPU | 2.10.0+cpu / 2.10.0.post2 |
| CANN / NNAL | 9.0.1 / 9.0.1，独立安装 |
| PyPTO | `54957491ede07ad5d5015f5e69874f367113cf45`，带已归档的本地兼容补丁 |
| Simpler | `58480c1fc2fac5f333763e7e4ea8f451271c167b`，重编译的兼容修订 |
| PTOAS | 0.63 |
| 硬件 | Ascend A3 |

该阶段另对 PyPTO 生命周期做72项UT，并做 NPU eager/capture 退出测试；A3构建与导入检查通过。环境 `pip check` 有未满足的包元数据约束，离线生成通过不等于 HTTP 服务或所有可选工具依赖通过。

迁移阶段记录的针对性回归包括：PyPTO shutdown/context/capture 72项通过；环境和CSA eligibility/RoPE/import/异常传播测试23项通过；原生 dispatch/output/profiling 测试6项通过；KV page布局检查13项通过；B4/S1、B4/S6 kernel参数绑定检查通过；最小NPU eager、capture0、capture1各1项通过。对应测试文件与本分支实现见 [CSA接口说明](../../pto-csa-integration.md)、[`test_decode_contract_cpu.py`](../../../tests/pto_attn/test_decode_contract_cpu.py) 和 [`test_dsa_pto_dispatch.py`（迁移提交）](https://github.com/Leaf-Salix/vllm-ascend/blob/19f784979c05b467e96745912330cd88c1c0bc33/tests/ut/ops/test_dsa_pto_dispatch.py)。

### HEAD 图与整网 profiling 验证

后续复测采用 CANN 9.0.1、Torch 2.10.0+cpu / Torch-NPU 2.10.0.post2、PyPTO `54957491`（含 PR #2867 修复）、Simpler `166852bf`、PTOAS 0.63。报告中逐项记录了实际导入路径；这些依赖快照与初始安装阶段的 Simpler SHA 不同，结果不能在未核对环境时直接横向合并。

### 2026-09-24 单因素精度边界对拍

按相同单层真实权重、合成 hidden/history、TP1、B4/S6、position 8191，对 Q-A、Q-B、Q RMS、KV projection、KV RMS、O-A BF16 和 O-B full-row 8192 分别做了单因素对照。记录了 QR/Q/raw KV/heads/final output 和六类 cache，并保留原生及候选阶段张量。Q-A BF16 明显改善 attention heads；KV projection BF16 主要改善 raw KV；O-A BF16 的中间值精确对齐 native，但仍留下量化路径差异；O-B full-row scale 显著改善固定-heads O-proj 对拍，但最终输出仍未过 allclose。所有指标、源 hash、任务 ID 和失败尝试见[日期报告](DSV4_CSA_PRECISION_BOUNDARIES_20260924.md)。本轮是 eager 单层归因，不是 graph/performance 或完整模型验收。

## 初始功能验证：三层 `official-l3`

模型为固定三层 DeepSeek-V4-Flash `official-l3` 测试权重；TP1、BF16、block size 128、`kv_cache_dtype=auto`，相同596-token输入，seed 0、temperature 0。它用于服务路径冒烟，并不代表完整模型的语言质量或多卡性能。

| 测试 | 结果 | 结论 |
| --- | --- | --- |
| 原生 eager，生成4 token | 完成、正常退出并释放设备 | 基线服务路径可运行 |
| CSA eager，生成4 token | 完成；记录3次真实 CSA 调用 | CSA 路径命中 |
| 原生 eager，2请求各生成64 token | 两请求完成 | 功能通过 |
| CSA eager，2请求各生成64 token | 两请求完成；记录126次真实 CSA 调用 | 两请求分别与原生输出 token 一致 |
| CSA `FULL_DECODE_ONLY` graph，B1、生成4 token | capture 后完成并正常释放设备 | 仅是三层模型短 graph 冒烟；不等于完整模型 graph 或固定 batch graph 验收 |

短生成 native/CSA 输出 token 相同，但选中 token 的 logprob 最大绝对差为0.1121385，故不能称为 logits 等价。64-token结果只说明这组确定性输入下 token 相同。

### 早期直接 kernel 单层 S1/S6 精度

使用完整模型第2层真实 attention 权重、固定合成 hidden/cache；B4、S1或S6，跨过128-token KV页边界，`torch.allclose(rtol=1e-2, atol=1e-2)`阈值未放宽。

| 用例 | output relative L2 | cache / 边界观察 | 状态 |
| --- | ---: | --- | --- |
| B4/S1，position 127 | 1.21298% | 压缩KV、main state、inner state通过；raw KV与index key未通过 | output精度失败 |
| B4/S6，position 127–132 | 1.28745% | 跨KV页与ratio4压缩边界；state及guard通过，raw KV与index key未通过 | output精度失败 |

误差是输出差值 L2 / 原生输出 L2。单层对拍用于定位，hidden/cache为合成输入；不能据此声称完整服务数值通过。该 S6 是 kernel/ABI 测试，早期服务入口仍限制普通 S1 decode。

## `fb734041c` native DSA 重构对照

对比基线 `8990a7d8e` 与重构版 `fb734041c`，同一 CANN 9.0.1 设备、模型第2层真实权重、B4/S6、合成 hidden/history。旧版为40参数并重建部分地址语义；新版本为52参数、直接消费原生 slots，且cache物理页布局由128改32。代码路径不止“删一个 Python wrapper”，因此不能把结果因果归于适配层数量。

### 单层 capture/replay 对照

| 上下文 | 旧版 CSA replay | `fb734041c` CSA replay | 延迟变化 | 旧版 output relative L2 | 新版 output relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K | 0.7684 ms | 0.8523 ms | +10.93% | 1.7160%，失败 | 0.0744%，通过 |
| 32K | 0.7811 ms | 1.0293 ms | +31.78% | 1.7775%，失败 | 0.0703%，通过 |

同一物理卡、真实 layer-2 权重；输入逻辑值和cache快照经校验一致；每条实现单独capture，30次交替replay，输出与cache均与本路径eager一致。precision门槛为同一 `rtol=atol=1e-2`。这是两个版本整体实现对比，ABI、kernel、metadata、cache页布局均有改变，不能用于单独评估“去适配层”的性能贡献。

另一个独立 graph replay 组报告8K/32K native-to-CSA wall延迟增加20.8%/32.2%，CSA output relative L2为0.0569%/0.1254%，通过同一阈值。该组的测量边界和采样方法与上表不同，不与上表拼接。

## HEAD `778bf158b`：恢复 kernel 后的回归

### 对 `8990a7d8e` 的单层回归

恢复的旧 kernel 文件逐字节与 `8990a7d8e` 一致。新旧接入使用相同逻辑输入、attention输出和3个共享 cache pool；SHA256一致。两种实现各自的 capture/replay 输出、全部cache与本路径 eager 逐元素一致，guard通过。

| 上下文 | 旧/新 Graph wall (ms) | 旧/新 NPU event (ms) | 旧/新 eager wall (ms) | 对原生 output relative L2 |
| --- | --- | --- | --- | ---: |
| 8K | 0.7625 / 0.8029 | 0.6697 / 0.6884 | 58.75 / 65.04 | 1.716002%，未通过 |
| 32K | 0.7816 / 0.8764 | 0.6980 / 0.7196 | 60.43 / 68.49 | 1.777542%，未通过 |

CPU契约回归30项通过；增量pre-commit通过。`format.sh ci` 除本机缺少 shellcheck 外其余检查通过；没有据此宣称shellcheck通过。单层对照说明当前接入修正没有改变旧kernel数值，但这轮没有证明提速。

### 接口/graph耗时定位

同一进程、同一卡、同一已注册 kernel，以 `8990a7d8e` 与 `778bf158b` 对照；B4/S6、8K和32K。新旧graph任务序列完全相同，均为16个 binding 设备任务加2个runtime/kernel任务，输出/cache逐元素相同。

| 区域 | 对照结果 | 解释 |
| --- | --- | --- |
| eligibility/config/metadata host检查 | 新实现增加约17–20 μs | 是提交阶段CPU开销；graph replay不重复执行 |
| binding设备操作 | 两版均16项，约31 μs | 计算链仍在；外部接口重构没有消掉这些设备操作 |
| 8K完整CSA Graph | 两轮分别 +0.149%、−0.017% | 没有稳定差异 |
| 32K完整CSA Graph | 两轮分别 +0.308%、+2.048% | 一轮有2.05%差异，另一轮接近；不足以证明严格零退化或稳定回退 |
| eager kernel调用 | 约59.7–59.8 ms | PyPTO JIT依赖解析/AST路径占显著host时间；不能当作设备kernel时间 |

这轮结论是 graph 性能差距没有在同进程控制下稳定复现；不是“已证明完全无退化”。具体 CPU/NPU profile 的局限和测量方法应与数值一起看。

## 128K/S6 单层六档 batch profiling（HEAD源码）

完整 DeepSeek-V4-Flash-0731-w8a8 第2层真实权重，TP1、C4 attention全路径，128K合成历史KV、S6 query、block size 128，BS为4/8/16/24/32/40。native与CSA同进程同卡交替；各自独立graph；20次非profile NPU event取中位数。Graph trace另采5次 replay。不是整模型生成、DP/EP通信、MoE、EPLB或端到端吞吐。

| 单卡 BS | 若 DP16 均匀运行时的目标 GBS（仅乘算） | 原生 ms | CSA ms | CSA/原生 | output relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 64 | 0.843 | 0.846 | 1.00× | 1.809% |
| 8 | 128 | 1.023 | 1.180 | 1.15× | 1.836% |
| 16 | 256 | 1.229 | 1.895 | 1.54× | 1.831% |
| 24 | 384 | 1.441 | 2.570 | 1.78× | 1.833% |
| 32 | 512 | 1.873 | 3.252 | 1.74× | 1.845% |
| 40 | 640 | 1.990 | 3.944 | 1.98× | 1.846% |

六档 output 均未通过既定 allclose。BS4约与原生持平；BS8及以上更慢。CSA的默认256MiB/ring仅BS4能完成；其余档位用诊断配置 `ring_heap=[2GiB]*4` 完成。此处的BS×16只是目标规模换算，不等价于实测DP16 GBS。

trace定位到随batch增长的主要部分是 indexer score+local top-k、稀疏attention QK/PV、kernel内RoPE表整理；外部metadata设备操作累计约35–50 μs，不足以解释整层增长。各阶段trace跨度有重叠，不能把耗时相加；没有把完整 CSA kernel 的成本都归因于适配层。

### 单层实际cache边界上的indexer gather候选

基于HEAD源码做隔离候选：将Indexer packed-page gather由32行改为64行；生产分支未修改。CPU映射穷举1,098,240组通过，NPU五个场景与旧CSA输出逐位一致。

| 场景 | 旧 CSA ms | 候选 ms | 配对变化中位 | 变化范围 | 相对原生误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1/S6/4K | 0.514692 | 0.521548 | +1.572% | −1.05%…+3.22% | 1.731% |
| B1/S6/32K−4 | 0.560321 | 0.557504 | −0.924% | −2.52%…+2.72% | 1.830% |
| B1/S6/32K | 0.556181 | 0.553824 | −0.072% | −3.68%…+2.22% | 1.792% |
| B4/S6/128K | 0.881572 | 0.886230 | +0.641% | −1.85%…+3.02% | 1.809% |
| B40/S6/128K | 4.019301 | 4.018520 | −0.422% | −2.95%…+1.65% | 1.846% |

没有稳定性能收益，candidate与旧版逐位相同，也没有改变已有精度差距；不合入生产分支。

### O-proj精度诊断候选

另将 main 分支 O-proj 数值边界修复单独移入隔离候选，在CANN9.0.1环境测量；只修改 O-proj 路径，HEAD生产源码仍为 `778bf158b`。该结果用于说明候选收益，不是当前HEAD成绩。

| BS | 旧CSA output relative L2 | 候选 relative L2 | 原生 Graph ms | 旧 CSA ms | 候选 ms | 判断 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 1.80915% | 1.62588% | 0.75572 | 0.89893 | 0.92808 | 精度改善但仍失败；候选多约29.14 μs |
| 40 | 1.84624% | 1.66039% | 1.93529 | 4.01096 | 4.02773 | 精度改善但仍失败；性能差异在配对波动范围 |

候选让相同 CSA heads 的 O-proj T6/T24 对拍逐元素一致，但整层误差仍未通过。不得将局部一致写成整层数值验收通过。

## 完整模型 16卡验证与固定批量边界

模型：完整 DeepSeek-V4-Flash-0731-w8a8，16×A3，TP1、DP16、EP16、DSpark出5、EPLB开。两类测试都必须分清“提交请求总数”和“实际纯decode batch”。

| 测试 | 已知配置与观测 | 结论 |
| --- | --- | --- |
| 初始原生短生成 | 每rank提交4个512-token请求、各生成16 token；16 rank均完成。观察到每rank容量日志约18.7 GiB KV；该短句平均接受draft数6.0，不代表业务 | 功能冒烟通过；不是128K固定纯decode |
| 原生128K输入生成 | 每rank提交4个131072-token请求、各生成16 token；chunked prefill开启。请求完成，但日志中实际请求/调度快照不证明固定BS4；无外部KV导入 | 证明分块prefill后能完成生成；不是离线prefill后固定BS4 decode，也不是纯decode性能 |
| HEAD native 与 CSA S6 capture | 每rank4个合成文本请求、各生成64 token。CSA日志确认S6命中；16 rank捕获的纯decode步骤为B1/S6，输出token与原生对照一致 | 完整模型功能与真实CSA调用路径通过；不等同固定GBS64（单卡B4）或生产精度验收 |
| 接收长度 | 同一合成prompt下平均接受draft token数3.0；若把target token也计入步长则为4.0 | 目标接收步长3.8未验证达成；EPLB开不证明短任务内发生专家重排 |
| 长窗口 profiling | 原生采到B4/S6纯decode窗口后保存trace，再主动结束；CSA长窗口后续任务取消以优先进行单层诊断 | 原生profile不是完整生成通过；未完成16卡CSA固定B4/GBS64性能对照 |

更早“每rank有4个请求”或 `max_num_seqs=4` 不能推导每步 decode batch 为4。固定BS验收应先准备等长KV，确认16个rank同时处于decode，并记录每步实际 query token 数、活跃请求数、抢占、CSA命中与cache布局。当前没有离线prefill/cache导出再恢复的固定BS64整网验收结果。

## 失败、更正与不适用结果

- 单层 S1/S6 output 超阈值；kernel退出码表示精度失败，即使外层任务成功收集证据也不能改称通过。
- `fb734041c` graph变慢、精度变好是整体 kernel/ABI/metadata/cache 改动的合并影响；没有隔离“删除适配层”的单因素实验。
- 当前HEAD `778bf158b` 的旧kernel结果不能沿用 `fb734041c` 的较低误差。
- 128K native请求生成成功不证明固定BS4纯decode，因为启用了chunked prefill且没有离线KV导入；4个输入请求不等于同时4个decode请求。
- 当前16卡capture中CSA命中且token序列一致，不证明logits逐元素一致、单层allclose通过、固定GBS64吞吐通过或接收长度达到3.8。
- 单层固定metadata的Graph和16卡完整服务在batch、调度、通信、draft target、EPLB活动性方面不同，性能数字不能混用。
- `ring_heap=[2GiB]*4` 是隔离测试所需的诊断配置，不能用它宣称默认生产环境支持BS8–40。
- gather/O-proj候选均未进入生产HEAD；只有关联提交和代码真正合入后才可更新生产结论。

## 后续测试的固定记录要求

每次有意义的精度、性能、功能或回归测试，在对应代码提交同步更新本文件，或新增同目录日期报告并从这里链接。保留旧结果；更正时写清旧结论为什么失效，不静默覆盖。

每条记录至少包含：

1. 仓库分支和完整Git SHA；dirty候选要写基点、diff摘要及源码hash。
2. 模型/层、权重来源、真实或合成输入、seed、TP/DP/EP、batch/query长度、上下文和block布局。
3. CANN、Torch、Torch-NPU、vLLM、PyPTO、Simpler/PTOAS版本，补丁及实际导入路径。
4. 原生与CSA/fallback开关、Graph模式和实际capture档位；逐rank记录实际活跃请求、query长度、CSA命中及EPLB观测。
5. 命令、warmup/replay/重复次数、计时边界、原始样本及统计方式；性能对照必须同配置、同进程或说明偏差。
6. relative L2定义、max/mean误差、阈值、cache/权重不变性、guard及是否通过；把局部kernel结果与整层/整模型结果分开。
7. 失败、跳过、被取消、环境阻塞、硬件配置限制、不能外推的范围和下一步。

状态只用“通过 / 失败 / 仅诊断 / 未运行 / 阻塞 / 取消”等明确表述。任务 exit 0 只说明程序退出成功，不自动代表精度或性能通过。性能结论不跨不匹配的环境、batch、Graph档位或计时边界拼接。

### 新测试记录模板

```text
日期 / 测试名称：
分支 / 完整源码SHA / dirty diff与hash：
对照的历史版本与同口径说明：
模型、输入、环境、并行配置、batch/序列、Graph档位：
命令、warmup、测量边界、轮次与统计方法：
原生/CSA命中、权重/cache校验、精度指标与阈值：
延迟或吞吐原始样本及汇总：
结果：通过 / 失败 / 仅诊断 / 未运行 / 阻塞 / 取消
失败、更正、限制、证据链接、后续动作：
```

本文件保留结论性结果，不包括权重、设备trace全量文件或机器内部路径。测试运行器、原始JSON及profile需存于受控归档；提交到仓库前移除凭据、服务器内网信息和不可公开日志。
