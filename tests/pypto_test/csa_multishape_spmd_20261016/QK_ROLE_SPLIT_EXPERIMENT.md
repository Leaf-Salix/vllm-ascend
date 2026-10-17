# QK 双 AIV 按职责分工（2026-10-17）

T96/128K两版均完整设备精度通过：原8-head分块L1中位885.08μs；
扩大softmax/PV分块并独立限制发布分块后为830.46μs。两版均未优于现选约807μs，
不采用、不扩档，不改变七档选择或生产实现。阶段时戳诊断单独记录，不参加性能验收。

## 改变的工作分配

原来每条AIV准备64行KV、处理32个head。候选中AIV0准备全部128行KV，AIV1处理全部
64个head的softmax、PV归并、归一化、RoPE和packed输出，尝试重叠搬运与数学。
这会减少并行搬运核数，也增加单条AIV的工作量；是否有净收益必须测完整CSA，不能由分工推断。

Cube QK/PV数学、五块归约次序、三槽L1与GM、pre-launch=2、正式权重和全部外部入参保持。
原四种FFTS事件保留两条AIV参与：无相应数学工作的AIV仍消费Score/PV通知，发送KV/Prob通知。
采用先消费上一块Score再通知本块KV的顺序，没有新增任务内Scalar GM信号或workspace。

两条AIV的事件序列与原提前KV策略一致；角色分工改变实际数据生产/消费方。
`qk_role_split_protocol.py`独立检查3328个调度、471867动作，包含两query全部1024种
有效块mask、256种四query随机mask以及2/3代复用。检查GM KV、L1 KV、score、prob、PV
五份槽的复用和每代事件排空；模型事件credit峰值3。它不证明硬件credit容量或DMA/缓存可见性。

## 数学状态与内存

AIV1的64×512 FP32状态共131072字节，拆为16个8×256循环值，逐块原位更新。
softmax与最终发布每块8个head。每个head仍按128列softmax、五块PV归并、FP32归一化后
rint转BF16；设计保留运算次序，设备精度不能仅由源码分析宣称通过。

首轮`qk_role_split_prepare_v1/`通过，`qk_role_split_cpu_v1/`在ConvertToSSA拒绝
分支外`kvbuf`引用，没有进入内存规划。失败源码快照`qk_role_split_source_v1.py.txt`保留。
v2把raw/cmp两条路径的store放回各自分支，以独立`kv_done`接收压缩行循环结果。
`qk_role_split_cpu_v2/`完整编译通过，未修改编译器。

实际IR地址检查：16个状态起点16384至139264，每份8192字节、互不重叠；初始化、回边和
有效块状态地址相同，16次PV加法逐个写回原槽。QK显式Vec Tile的shape地址范围到163328，
低于188416字节；该范围不是完整allocator报告。
C++有8个row-max、8个row-sum、8个最终row-divide；两分支合计4个wait、4个notify，
5个PIPE_ALL。实际提交顺序由`qk_role_split_codegen_audit.py`检查，不以静态计数当性能结论。

## 必须同时改变的输出依赖

旧O-A按head半区分别等AIV0/AIV1；现在packed的全部8组由AIV1写出，必须全部等待对应
QK组的AIV1。否则可能在AIV0完成KV搬运后提前读未写完的输出。

`row_dependencies.py`增加默认关闭的角色所有权参数。新分支检查真实写入代码确实在AIV1，
枚举96个query的768行packed结果，要求唯一写入、八组O-A消费范围全部覆盖。
旧策略的六项源合同和行覆盖证明仍通过。其他依赖、原物理核先后和信号发布维护保持。

`qk_role_split_prepare_v3/`使用新依赖证明生成完整变体，其角色分支AST与已编译v2相同。
不重复编译相同数学体；设备编译后再次检查地址/事件顺序，以及真实100tick等待实现。

## 设备任务

`task_20261017_115641_29670009014`，结果目录`b16_row_deps_qk_roles_v1/`。
仅128K/B16/S6，原动态PTO对照、正式权重、完整exact/图/30代/保护区门槛之后才测三窗L1。
任务exit0。eager/graph/final输出、TopK和六cache/state全部零差异；30代进度与输入、
写入范围、工作区保护区PASS。实际设备artifact通过状态地址、FFTS顺序和100tick等待门禁。
完整无插桩L1：887.72 / 885.08 / 870.46μs，中位885.08；同轮动态PTO中位901.30μs。
与历史现选807.16/807.50相比没有采用理由，未做本轮现选配对，不精确分摊跨轮差值。
摘要`qk_role_split_result_v1.json`。fixture逐次恢复，不是连续decode轨迹。

观察任务时有一次40秒等待工具超时，后续原任务正常exit0；没有重启、终止或设备超时错误。
只读核对本地PTO-ISA集群映射文档说明mode2的两AIV参与及4位计数；没有将事件模型当硬件证明。

为定位为何分工没有收益，提交一次阶段诊断`task_20261017_120238_306765112993`，
目录`b16_row_deps_qk_roles_stage_v1/`。仅一次L1窗口、warmup0/iters1；其结果含插桩且
warmup与历史不同，不能替代上方正式三窗结果或精确计算跨轮回退幅度。

诊断任务exit0，全部eager/graph/final exact、4代进度、保护区PASS。时戳见
`qk_role_split_stage_comparison_v1.json`；以下是阶段的“数学、内部等待与写侧维护”，不是pipe独占忙时：

| 阶段 | 历史现选中位（μs） | 分工候选中位（μs） |
|---|---:|---:|
| Indexer AIC | 211.03 | 212.99 |
| QK/PV AIC | 150.93 | 206.39 |
| QK/PV AIV0 | 157.90 | 207.90 |
| QK/PV AIV1 | 158.12 | 222.80 |

本次QK AIC最后结束701.30μs、AIV0为703.50μs、AIV1为718.38μs，O-A最早720.94μs。
候选的AIV1成为组内尾部，且Cube自身也没有因分工缩短。现有证据把主要退步定位于QK/PV，
不能归咎于外层等待策略或Indexer。尚未测出KV、softmax、PV、输出各自的独占成本，
不能精确地把多出的时间分给某个操作。

源码层面，单条AIV现在负责双倍head，且为控制UB改用8-head小块，softmax静态位点由
每AIV1组32-head变为AIV1上8组8-head，最终发布也变细。额外指令提交和工作集中是
下一步CPU分析的候选原因，不能仅凭调用数认定其时间占比。
不重复此版设备测试；若继续该方向，先验证更大head Tile的实际UB与别名，再决定是否值得上卡。


### 2026-10-17T12:22:59：角色分工的head Tile分离，CPU确认发布峰值超UB

在现有8-head分工版本上，将PV状态块扩为16×256、softmax扩为32×128，最初发布也为16head。
qk_role_tiles_h16_s32_prepare_v1准备PASS，cpu_v1在AllocateMemoryAddr拒绝Vec194048B>188416B；无设备执行。
从37_after_MemoryReuse逐项核对，AIV各allocation求和正好194048B。状态仍128KiB，PV的两份16KiB临时区和
最终32KiB FP32发布结果复用；16KiB BF16输出与两份4KiB RoPE临时量、4KiB swap索引同时存在，
因此单独缩softmax并不能消除发布峰值。诊断qk_role_tiles_ub_diagnosis_v1.json保留全部allocation。

将publication_tile从state head_tile解耦：仅发布回到8head，从对应16head状态slice连续行，
仍为32head softmax、16head PV；原运算次序、Cube、GM接口和事件保持。旧8/8及16/32同发布
版本AST回归一致，旧8/8/8编译产物扩展审计PASS。新增快照qk_role_split_source_h16_s32_p16.py.txt。
qk_role_tiles_h16_s32_p8_prepare_v1 PASS，cpu_v1编译中；当前尚无此变体设备结果，不修改七档选择。


### 2026-10-17T12:24:22：16/32/8候选完整CPU通过

qk_role_tiles_h16_s32_p8_cpu_v1 full_codegen_passed=true。MemoryReuse的全部AIV Vec allocation合计
167552B，低于188416B；与h16/s32/p16的194048B相比减少26496B，实际全局复用结果不同，
不能仅按局部Tile字节数估算。8个16×256 FP32状态起点0..114688，互不重叠且原位更新；
incoming不覆盖状态，8次旧值乘法/更新逐槽匹配。softmax2个32head rowmax/rowsum，发布8个8head rowdiv，
仍4wait/4notify/5PIPE_ALL，提交顺序PASS。全部记录qk_role_tiles_h16_s32_p8_codegen_audit_v1.json。

新增默认关闭--qk-role-wide-heads，限定与--qk-role-split组合，明确使用(状态16,softmax32,发布8)；
设备执行前检查实际编译Tile与请求一致。行依赖继续检查AIV1独占全部8组packed输出。
CPU集成准备中，不重复已经通过的数学编译；尚无设备性能，不修改现选。


2026-10-17T12:25:55：集成行依赖后全部数学体AST与CPU通过版本一致；提交task_20261017_122441_342567127888，b16_row_deps_qk_roles_wide_v1。仅B16/128K/S6，全门槛+三窗无插桩L1。任务执行中冻结driver/helper，不改现选。


### 2026-10-17T12:28:27：32head softmax/16head PV/8head发布全门槛PASS，未采用

task_20261017_122441_342567127888 exit0，b16_row_deps_qk_roles_wide_v1。
eager/graph/final的输出、TopK、六cache-state均零差异，30代进度及全部保护区PASS。
实际Tile(16,32,8)、状态/事件及100tick等待源码门禁PASS。无插桩完整L1
830.46/827.18/832.04，中位830.46；同轮动态899.68/911.08/912.48，中位911.08。
比旧8/8/8分工的历史885.08明显降低，但仍未优于现选807.16/807.50；不同轮次无现选配对，
不精确归因差额。未采用、不扩档；暂关闭仅靠分工/增大head Tile的优化方向。
见qk_role_tiles_result_v1.json。所有设备任务终态，700目标未达。

独立新增fixed_core_local_schedule.py，只在CPU从现选顺序进行固定物理分核的相邻交换，
与前轮三种从头贪心分配不同。数学边与216条FFTS组复用边固定，MIX按共同开始建模，
防止误用成员独立开始的时间；原模型771.53，共同开始模型773.41。120秒局部搜索中。
未生成二进制、未接入生产/测试driver，运行中的设备helper没有修改。
