# CSA核内优化：本地AscendC源码参考

更新：2026-09-28。按用户要求，将新下载的三个仓库纳入Native、pypto-lib之外的源码参考。
以下是本地已读取版本，不表示当前CANN二进制已包含这些实现，也不表示这些实现已测得比PTO快。

| 本地仓库 | 本次读取HEAD | CSA相关入口 |
| --- | --- | --- |
| [ops-transformer](../../../ops-transformer) | b5b33e14 | QLI/QLI V2、SparseFlashMla、mHC |
| [ops-nn](../../../ops-nn) | 7a71d54e | RMSNorm、动态量化及融合路径 |
| [ops-math](../../../ops-math) | 81802185 | 排序、Top-K及基础向量操作 |

三个HEAD提交日期均为2026-09-28。本次只读源码，没有安装这些仓库、升级CANN或改Native流程。
性能算子对照为e58ddc94，pypto-lib参考为73078d0；后续更新源码时记录实际使用版本即可，不做全仓hash扫描。
先按产品支持表、构建入口与指令确认A3适用性，不能单凭`arch22/arch32/arch35`目录名称类推。

## 1. 新的优先候选：跨分片Top-K四路归并

来源：QLI V2 A3路径
[ProcessLD](../../../ops-transformer/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h)。
其`ldProcessLen=4`，每轮将累计Top-512与三个新分片做MrgSort，保留前512对在UB，最后发布结果；
尾部分别使用二路或三路，`validBit`区分有效输入。

当前PTO的
[indexer_topk_query_merge_one / merge2_top512_pairs](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer.py)
逐份二路归并，每轮把累计根写回pair_arena，下轮再读取。
有H份半leaf时，当前H−1轮；四路累计归并可降至ceil((H−1)/3)轮。
例如H=8时7→3轮，H=2仍是一轮。实际H取可见候选数，不能按128K标签硬编码。

待做范围：

1. 仅在性能版跨leaf合并采用四路分组；Score算术、cache布局和任务数不动，短档保留原二路路径。
2. 先核对PyPTO四输入`mrgsort`的有效输入、截断及物理tile形状；原AscendC的耗尽暂停语义不能直接假定等价。
3. 明确相同score的输入优先次序。当前PTO约定新块在前，四路操作数需相应排列；
   若无法保持精确顺序，则作为有规则的Top-K策略差异独立记录，并满足token/DSpark验收。
4. 首轮先减少归并次数；UB循环携带根的形式必须检查生成的TMOV成本，不能把省GM字节数直接当作收益。
5. 单卡长档看merge核内与累计工作，短档作不退化检查；有明确收益才补受影响尾leaf及真实EP16。

这不是重复旧的[二路UB累计根候选](results/csa_topk_register_20260928/README.md)：
旧候选没有减少归并轮数，核内11.814→12.068μs，未保留；本项的新变量是四路归并。
QLI V2的metadata和`ProcessDecode()`还带全核同步，当前PTO已有任务依赖，不整体移植这套调度。
当前尚未实现或测量四路候选，轮数下降不是性能收益。

## 2. 已经采用的策略与仍需核对的差异

| 对象与源码 | 本次核实 | 后续重点 |
| --- | --- | --- |
| [QLI V2 Cube](../../../ops-transformer/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_cube_arch22.h) | 与本仓Native QLI Cube的主体流程相同，主要为名称、布局枚举及stride字段差异；FIXPIPE的1/1024缩放、ReLU、FP16第二次Cube规约并非新发现，性能版已采用 | 对照Q/Key/S的实际驻留周期与流水等待；不要再次把已采用的Cube head规约当作新优化 |
| [SparseFlashMla CSA Cube](../../../ops-transformer/attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_csa_block_cube.h) | Q/P四份L1区、KV三份L1区及DataCopyPA；当前Native Sharedkv已有同类配置 | 核对PTO QK/PV复用、跨query流水中实际等待的位置，再决定搬运/缓冲改动；旧16行UB双缓冲与成对DMA没有稳定收益，不原样重测 |
| [MhcPreSinkhorn Cube](../../../ops-transformer/mhc/mhc_pre_sinkhorn/op_kernel/mhc_pre_sinkhorn_cube_compute.h) | `ComputeDecode/MmadA2/MmadAB`复用输入计算平方和及投影，与Native HC参考同类 | 先完成已有[HC输入加宽/RMS融合候选](results/csa_hc_input_rms_20260928/README.md)，不另造一份重复实验；纯AIC投影与精度版归约边界继续保持 |
| [RmsNormDynamicQuant](../../../ops-nn/norm/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_normal_kernel.h) | A3支持，多行UB处理、权重驻留、归一化与量化融合；FP32→INT32 RINT→FP16→INT8 TRUNC链与当前PTO一致 | 检查QR的两遍输入/gamma读取能否减少，先算UB生命周期；当前性能版已把平方和与amax合在第一遍，不能把“融合”本身重复计为新改进 |

RMSNormDynamicQuant的新旧文件差异还包含单/双量化输出、smooth及beta接口，
不能把删去另一条输出分支带来的代码简化称为当前CSA的确定性能收益。
精度版仍以现有Native舍入及规约合同为准，不能直接套用性能版的代数化简。

## 3. ops-math的适用边界

- [TopKV2入口](../../../ops-math/math/top_k_v2/op_kernel/top_k_v2_apt.cpp)此次读到的实现引用arch35路径。
  README新增`sort_policy=1`的Bitonic Small TopK针对2≤k≤32，当前CSA为Top-512；本轮不直接套用。
- [experimental SortV2](../../../ops-math/experimental/math/sort_v2/op_kernel/sort_v2.h)提供A3相关排序参考，
  但读取到的是通用Concat/Sort/Extract流程，并未发现可以直接替换当前Top-512的确定收益。
- 后续优先比较A3可用的排序/归并原语及中间搬运；不因仓库更新日期新就切换算法或运行时。

## 4. 当前执行顺序

先处理[短档Score/Sparse长尾证据](results/csa_short_score_sync_20260928/README.md)，
随后恢复已有HC候选，再验证上述四路Top-K和有明确搬运差异的核内候选。
每项分别记录核内耗时、调度等待、完整CSA/P95与最终forward，解释与pypto-lib的任务和输入差异。
先单卡代表档，明确收益后再补必要的真实权重EP16；没有新证据不重跑旧失败方案或整矩阵。
