# HCA attention：累计max、UB容量与后续大颗粒方向

2026-10-23，接续正式提交`37bbd0f3c`。公共SDK仍为
`csa-shared-20261022-logical-codegen`，未改SDK，所有新设备任务直接执行，不使用ACLGraph。
本轮候选未写入正式算子；任务身份分别记录在`ATTENTION_ONLINE_TASKS.json`和
`ATTENTION_BOUNDARY_TASKS.json`。未结束任务不能被计为精度或性能通过。

## 当前两个候选的作用和已有证据

1. `attention_macro512_online.py`仅在K512把累计max前移到BF16概率转换之前，
   PV更新只缩放旧累积，去掉对整块新PV的beta缩放。生产端累计max每query重置，
   和延迟两拍的PV消费状态分开；保留三槽、跨query连续流水及原Cube供数。
   与9月的K128 online试验不同，当前基线已经是K512/D256，归约和供数粒度都改变。
2. `attention_macro512_online_n64.py`继续采用八份N64 FP32累积、H8 softmax/发布。
   参考9月双query容量解法，但当前仍是单query分工；并未把旧静态B/S代码搬入正式TND。
   每次PV临时块更小，代价是更多窄块加载和循环。必须测完整层，不能把省UB当作提速。

三根入口的CPU编排及incore C++均编译通过。对既有生成PTO的静态alloc_tile地址加逻辑
字节范围统计如下；这是地址覆盖范围，不是PMU测量，也不替代编译器容量证明。
数据与完整源路径见`evidence/ATTENTION_ONLINE_ALLOCATION.json`。

| K512实现 | Mat | Acc | Left/Right各 | Vec静态地址范围 |
|---|---:|---:|---:|---:|
| 当前正式块内max | 196608 | 131072 | 65536 | 178144 |
| 累计max | 196608 | 131072 | 65536 | 178272 |
| 累计max + N64/H8 | 196608 | 131072 | 65536 | 157280 |

累计max单独**没有降低整体Vec地址上界**，不能因删掉beta就声称已经腾出双query空间。
N64/H8确实减少约21KB，但再增加一份64KB完整累积仍会超过188416B上限。

独立混长Q=[1,5,6,1,2]的FP64诊断：累计max在128K的max_abs=0.001097228、
RMSE=0.000130727；16507边界组合为0.002190986、0.000220587。
两种候选相互逐bit一致，重复直接调用、非有限值、失活请求及输出尾保护通过。
该混长合成输入只有15个query，不能代替真实层多query跨轮流水检查。
完整层B16/B28对照以原Native输出容差、控制cache/state精确为门禁，结果见
`ATTENTION_NEXT_RESULTS.md/json`：累计max的B16为531.32→524.38μs、仅4/7配对更快；
B28为746.44→748.60μs。N64/H8相对累计max为526.54→534.10μs、仅3/7更快。
未把波动中的小收益新增为正式档位；两个候选均保留为实验。attention核耗时单独保留，
B16累计max的平均AIC/AIV时间实际上增加，整层中值降低不能直接归功于Vector公式。
第一项设备任务在SetDevice阶段报507033，算子未执行；确认退出后用相同冻结源码重提，
没有把环境启动失败记为候选数学失败。

## Native混长参考失配：先纠正结论，再继续验证

先前B5、Q=[1,5,6,1,2]、history=[127,16383,131073,0,16507]，
T容量24、请求容量8的Native输出超差，暂不能归因于PTO。
在两次采集中，尚未执行PTO便在Native报MTE非法地址。分段同步进一步定位到
`npu_sparse_attn_sharedkv`；mHC pre、RMS、前置QKV/压缩和输入拷贝均已完成。
给定Native RMS的C128独立检查通过：state最大差5.364418e−7、压缩cache完全一致。

随后只截取输入，在SAS执行前主动停止，CPU审计发现：

- Native query形状为`[15,64,512]`，Native前端已裁到实际token数。
- 传入的cu_seqlens_q却是`[0,1,6,12,13,15,24,24,24]`，末尾仍包含padding容量。
- seqused_kv为`[128,16388,131079,1,16509,0,0,0]`。

这是测试构造的参考输入不一致；旧版与候选对这份Native结果的误差不能作为有效精度结论。
证据在`evidence/NATIVE_MIXED_ATTENTION_INPUTS.json`；完整Tensor快照仅本地保留，未入Git。
不能据此声称Native算子本身有bug，也不能直接把PTO改成复现其越界结果。

`run_case.py`已为有padding的Native参考单独构造真实请求metadata，复用相同物理cache和初态，
实际调用对原hidden/positions/output只取真实token视图；PTO保留原T/B容量和dummy，所有数值容差不变。
这只修测试参考，不修改生产适配或共享CSA scenario脚本。v5仅裁metadata，暴露RoPE输入
维度仍不一致；v6统一所有Native有效视图后没有MTE，最大差0.015625、RMSE0.001655229，
输出还剩1个超差`[12,0,17]`，是history=0/Q1的冷请求。v6诊断输出与真实PTO路径逐bit一致。

给Native attention结果及Native gates接PTO尾部，最终输出逐bit一致；原PTO attention接Native
尾部仍复现同一超差。把Native Q或Native cache单独放回attention，都通过原输出容差；
两者都用PTO才复现，剩余是前置Q/KV误差的组合，不能再称为C128聚合独立故障。
源码复核还排除了两个猜测：公共mHC已经用平衡加法`(y0+y1)+(y2+y3)`，且RMS前已有BF16舍入。
所有旧失败按原状态保留；旧无效Native参考不能被重新解释为有效数值门禁。

## 后续大颗粒候选：两枚Cube共享两个query的压缩KV

`build_attention_pair.py`生成完整实验`candidates/attention_pair512*.py`，v5/v6完整CPU后端
编译通过，v6独立S6/混长与K512逐bit相同，v7～v9完整B16/B28也保持输出/cache/state精确。
已完成版本的性能均未超过正式版，不能因为KV读取减少就采用；正式算子保持不变。

旧双query让一个AIV承担完整64 heads，导致UB紧张；而且两槽、逐pair排空损失流水。
下一版考虑12个逻辑组，每组两枚Cube、四个AIV：一枚Cube负责M128压缩QK，另一枚负责PV，
四个AIV各保留一个query的32 heads。全卡仍24 Cube+48 Vector，Vector不持有两倍完整累积。
Raw窗口因两个query起点不同，继续各自M64，避免扩大raw窗口增加无效MAC。

- 压缩QK/PV各复用一次KV供两个query消费；QK M128/K128的L0A双缓冲合计64KB。
- PV不能同时保留四份M128/N128 Acc（需256KB）；应分两个D256组，每组两份64KB Acc。
  这会重复读取概率，须计入净GM字节与实测，不仅报告KV读取减半。
- 12组×3槽×128 heads的GM交接容量与现有24组×3槽×64 heads相同。
- B16/S6为48个pair、B28/S6为84个pair，分到12组分别4和7个，避免24组下B28的半波尾部。
  实际配对必须读device Q/L，只在同请求内配对，S1/S5和奇数尾部保留单query路径。
- 三槽和跨pair流水必须保留，不能退回旧逐pair排空方案。

跨逻辑组的QK→softmax→PV需要点对点通知。不能假设logical block映射相邻物理核，也不能
把现有同Cube组的FFTS直接当作跨组事件。优先复用已验证的A3通知/等待能力：每个信号
独占缓存行、单写者、发布前等业务写回完成，读侧处理Scalar缓存，明确环复用的确认顺序。
GM信号每次调用先初始化并建立依赖，防止旧代际误用；每槽用512B隔离区，沿用已有A3
signal_notify/wait接口，业务流水排空与缓存维护均显式执行。sync_start确保角色准入。
v7关闭early是保守候选设置，**不能把“必须关闭”当作API要求**：后续核对公共Simpler的
TMR `scheduler.h:882`及`orchestrator.cpp:535/573`，early是生产者允许消费者预派发的提示；
sync_start gated cohort仍等全部核可运行及前驱释放才启动。v10恢复plan/attention的early，
保留全部依赖和sync_start；单独测其对完整层启动空隙的影响，不把此前额外空隙全归给数学。
独立`attention_pair_plan.py`的v1因在incore内reshape得到Tile而错误用于GM store失败，
v2/v3完整CPU编译通过。完整pair实验需额外为length=0请求发出零输出工作，不能直接沿用
独立planner跳过失活行的逻辑；v1的INDEX写INT32错误在本地算子修正，不属于SDK问题。

停止条件：容量不成立不占卡；功能/重复/保护失败不计时；先只做B16/B28完整层配对，
若通信、启动等待或新增概率读取抵消收益，就保留负结果并停止扩档。
目标仍是完整HCA超过Native30%，不把局部Cube收益或CPU容量通过当作完成。

## 双query同步的递进对照

| 候选 | B16正式→候选μs | B28正式→候选μs | 决定 |
|---|---:|---:|---|
| v7：100tick，等待PV消费ACK | 518.04→593.60 | 749.40→896.36 | 不采用 |
| v8：0tick，其余同v7 | 512.88→605.06 | 756.52→867.38 | 不采用，不能说退避是唯一瓶颈 |
| v9：100tick，只等待score读取ACK | 528.90→593.66 | 737.94→820.36 | 不采用，不扩档 |

每一行都是同任务正式控制，不同任务不能用中值之差当作单项增量。
v9只覆盖score的真实读取者：四个AIV发布P(k)即表示score(k)已读完，允许QK写score(k+3)。
P(k+3)由各AIV在消费PV(k)之后才发布，Cube又等待四份P(k+3)，故P/V缓冲也不会提前覆盖。
由此删除四个独立PV消费通知。它仍保留每次调用初始化及Scalar缓存维护，不是删同步赌博。
v9完整S6已精确，混长需在采纳前补验；当前未采纳，所以未扩大功能测试范围。
v10恢复early的任务身份见`ATTENTION_PAIR_TASKS.json`；待结束后再作结论。

## 独立gather候选

`build_attention_gather_mask.py`从正式实现生成`attention_gather_vector_mask.py`。
原压缩KV聚合每页对最多32列逐个Scalar写UB，改为Vector生成精确0/1，并以[4,32]
整行拼接，最后reshape回[1,128]，避免已知Vec列subview限制。只处理合法页及有效行，
未初始化尾行仍不读取；地址、页表判断、KV搬运及attention数学不变。
三个根的完整CPU编排/incore C++已通过。独立混长/无效页检查与B16/B28完整层配对已提交，
见`ATTENTION_GATHER_TASKS.json`，未将CPU通过写成真机或性能通过。
