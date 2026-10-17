# 2026-10-18：动态 CSA 三路并行优化

基线提交为 `593c5d25`。三个 agent 使用独立 worktree；主会话维护隔离源码
加载器与同进程三方对照。正式目录的动态形状、数学和默认策略未因候选而改动。
冻结基线源码在 `baseline_source/`，候选源码在各结果目录的 `source/`。

## 比较方法

正式 W8A8 权重，每请求 S6；先精确核对输出、Top-K、score、cache/state，
再检查 A→B→A、graph、未写区域与72核代际，最后轮换三方各五个 L1 窗口。
统计完整 incore 最早开始到最晚结束，包含内部等待和空隙；不测 Native 性能。
两侧 SPMD 都使用测试编译 hook 的100 tick失败轮询退避。

旧 SPMD 和候选共享同一批 Native 根 Tensor，各自独占 workspace、signal、
JIT 对象；不切换共享安装。原分 task PTO 始终作为最终30%目标的分母。
不同作业之间的差值不用于推导组合收益。已完成精度比较的参照是原 PTO，
不是 Native 或整模型 token。

## Attention 重复等待折叠：未采用

利用所有 Score AIV → Top-K → plan 的现有偏序，删除 Attention 入口重复
扫描 QUERY、原KV、主压缩输出和RoPE阶段；plan后的业务缓存获取保留。
发布、512B单写者信号槽、代际、FFTS及数学不变。144种依赖场景模型与
长/中历史完整符号编译通过；模型不代替设备验证。

| 场景 | 原分 task μs | 旧 SPMD μs | 候选 μs | 候选相对旧 SPMD 耗时降低 |
| --- | ---: | ---: | ---: | ---: |
| 128K / B16 | 882.04 | 912.02 | 915.70 | -0.40% |
| 8K / B32 | 986.30 | 1014.36 | 1011.18 | +0.31% |

两档完整数值、图、保护区和代际检查通过，收益幅度小且方向相反，不扩七档、
不合入正式策略。任务分别为 `task_20261018_003350_299816415546`、
`task_20261018_003400_30016459311`。原始五窗样本与复现见
`sync_evidence/README.md`、`sync_evidence/result_summary.json`，完整结果为
`sync_h131072_b16_v1/result.json`、`sync_h8192_b32_v1/result.json`。

## Attention mi/li 本地三槽：编译失败，未上卡

同 lane 的局部softmax在work+1拍生成mi/li，work+3拍归并消费，同槽下次
写入在work+4拍；拟用UB三槽替代两次GM重读，保留原GM写出、算术与事件。
源审计保留98项原算术赋值，所有GM store及FFTS调用不变；槽模型检查通过。

两个 `[3,32]` carry 环与一个 `[6,32]` 交错环均通过符号解析，但完整UB
规划分别为255232/255104 B，超出188416 B上限，因此停止该候选，没有设备
精度或性能数据。逻辑环只需768 B，物理环MemRef也未重复分配；失败表明
加入长期存活对象后整体内存规划超限，不能误写为环本身膨胀到255KB。
完整尝试、源快照、日志、补丁和本地产物路径见 `attention_local_ring_evidence/`。

只读后续建议将HC门控两处 `col_expand + add` 合为 `col_expand_add`；尚未
实现或测量。动态有效列的逐行vadd可能抵消收益，保留供后续批量广播优化
评估，不根据复制次数计算性能收益。

## Indexer 半叶 score 本地保存：两档通过，暂不合入

已有 G2/768 区间使用每AIV私有的两条半叶缓冲，保存最多4096个有效score，
按384列步长补齐到4224，随后直接做原六分支Top-K排序。保留每query的
visibility、根划分、排序树、并列输入顺序及握手，不修改 G6/balanced 策略。
物理缓冲排为22×384，避免A3窄列UB assemble限制。

| 场景 | 原分 task μs | 旧 SPMD μs | 候选 μs | 候选相对旧 SPMD 耗时降低 |
| --- | ---: | ---: | ---: | ---: |
| 8K / B16 | 675.50 | 712.38 | 691.98 | +2.86% |
| 8K / B32 | 991.12 | 1020.64 | 1025.04 | -0.43% |

任务 `task_20261018_003818_31105785398`、`task_20261018_004709_328630619605`
均完成三方完整精确对照、图、保护区和代际检查。不能把B16的20.40μs改善
推广到整个区间；B32没有改善。另一个helper任务
`task_20261018_004914_333028922857`完成25个可见长度边界×随机/全同分×
live2→1→2共150轮同地址graph：全部pair arena和Cube transfer逐字节一致，
cache只读及候选不写GM score scratch检查通过。该helper不替代整层mixedQ验收。

前五版CPU尝试依次处理Tile形参注解、重名动态valid_shape、跨分支SSA以及
A3 UB窄列assemble限制，第五版完整通过。未启用分支在constexpr裁剪前仍
被检查，G6/1024发现固定384列不匹配；第六版改用策略常量表达式，G2/768
两角色生成C++去除注释/空白后逐字相同，故未重复B16设备测试。

后续G6/768、G2/1024也编译通过，但无用缓存声明改变了实际活跃Tile的
UB地址；其中G6/768末端从163872升到166912 B，G2/1024末端仍为163872 B。
不能把这两类称为原版生成物等价；
本轮不把候选合入正式实现，也不根据两个点建立T96/B16专用策略。完整
失败与原始证据见 `indexer_evidence/`；仍需先隔离分支内存影响并定位负载差异。

## 后续独立工作

两波请求的Indexer→TopK→Attention重叠已完成离线依赖/核资源模型：301个
节点、882条边、72核资源无冲突，可构造跨组重叠。但B16固定owner的均衡
反例中每组都要做22个score块和20个Attention块，零交接成本下新旧计算链
同为22s+20a；没有大阶段计算收益依据，不上卡。未切helper query范围则有
`Attention0→Plan1→TopK1→Indexer1→Attention0`具体等待环。见 `wave_audit/`；
这不否定不同角色粒度或重新分核的其它方案，也不是实测性能下限。

`attention_local_ring_evidence/physical_row_plan_readonly.md`审计了地址准备：
直接把512次Scalar映射搬到更少的plan owner会集中负载，因此否决该写法；
只推进有限容量下的逐query Vector页映射，替代整页预扫描，保持原双AIV
搬运与三槽事件。在新独立worktree实现，首轮符号编译、132轮CPU oracle及
15轮真实A3地址probe已通过；整链结果见下一节，作为独立阶段归档。

## Attention 物理行 plan：整链通过，小幅改善，保持候选

任务 `task_20261018_010403_360472313010` 完成。128K/B16/S6 的候选相对
原动态分 task 输出、Top-K、score 及五份 Native cache/state allocation
逐 bit 一致，A→B→A、同址图、动态历史/页表交换、所有保护区和72核代际
检查通过。15轮真实A3地址probe还验证物理/逻辑回退、odd/mixed/dummy/stale
地址；该局部检查不等于整层mixedQ、S1或S5已经验收。

| 实现 | 完整 L1 五窗 μs | 中位数 μs |
| --- | --- | ---: |
| 原动态分 task | 887.82 / 880.32 / 858.44 / 894.06 / 874.94 | 880.32 |
| 593c5d25 旧 SPMD | 913.06 / 902.06 / 904.24 / 907.82 / 904.78 | 904.78 |
| 物理行 plan 候选 | 906.10 / 893.26 / 897.44 / 889.88 / 901.88 | 897.44 |

相对旧SPMD降低0.81%（7.34μs），相对原分task仍慢1.94%，未达到30%目标。
五个同轮差值均为正，但幅度不足以据此启用整个容量区间，亦不做显著性结论。
本轮保留默认关闭的隔离候选，不扩七档。源码、132轮CPU oracle、15轮设备
probe、CPU完整编译、原始结果、独立审阅范围与本地产物路径大小见
`physical_plan_evidence/`；完整三方泳道在 `physical_plan_h131072_b16_v2/`。

首个任务 `task_20261018_005742_348545224475` 在编译期间主动取消，以补齐
真实地址probe，未产出设备数学或性能结果；保留取消日志，不统计为算子失败。
正式动态算子源码没有因本候选改变；测试证据可提交，与是否采用候选分开。

## 当前阶段时钟与实际双缓冲审计：已完成定位，不计性能收益

两项诊断已完成，见 [CURRENT_PHASE_DIAGNOSTIC.md](CURRENT_PHASE_DIAGNOSTIC.md)。
当前593基线128K/B16的5次完整检查通过；8K/B32旧/半叶缓存候选共10次
完整检查通过。新增时钟包裹原FFTS wait，未插入fence；所有样本含插桩扰动，
不替代前面的未插桩完整L1。旧记录仅覆盖GM等待，不能将other当成纯计算。

长历史Score的AIV较多时间处于READY等待，Attention的AIC较多时间处于
KV/probability等待，支持分别追查Score供数链与Attention的AIV推进链。
短历史B32本地score缓存的局部省工主要转成READY等待，整段没有稳定大幅
缩短，不支持“已省很多Score时间，只被后段抵消”的推断。

[pipeline_depth_audit/README.md](pipeline_depth_audit/README.md)逐一核对
本次生成IR和C++：HC Right、Indexer压缩投影Right、主QB Left确实由
原task两地址变成一地址，并增加相应M→MTE1边；OA/OB/QA/KV仍有真实
两槽，Score也保留48KiB WS加两份8KiB Key，不能据性能提示认定全面降槽。
源码重排可能同时影响分配，后续候选需对照实际地址，不能只归因于调度顺序。

Compact优先候选仅移动两块完整生产阶段，动态维度、数学与通知/FFTS不变。
144组依赖模型和完整符号编译已通过，独立协议审阅未见新增等待环；
128K/B16三方设备任务`task_20261018_013723_19521519782`已完成，全检查通过。
五窗完整L1中位数：原task861.48、旧SPMD913.06、候选888.38μs，
相对旧版降低2.703%（24.68μs），相对原task仍慢3.123%，目标未达。
源补丁、独立审阅、完整样本及复现见`compact_priority_evidence/README.md`。
已提交另一负载代表8K/B32任务`task_20261018_014358_34138622015`，
尚不据长B16推广至整个区间，也没有扩七档或合入正式默认。

## Compact投影L0双槽：源码内修复已编译，设备结果待回收

另一独立worktree只改Indexer compact投影helper，保持M16/N32/K512，
每pair按KV0/gate0/KV1/gate1顺序累加。两份KV Right先同时存活，迫使
分配器保留0/32768两个地址，消除原gate0→KV1搬运覆盖等待；仍保留
gate0/gate1复用地址0的必要依赖，没有修改编译器或强行指定物理地址。

完整符号CPU首轮通过，156例实际helper AST控制流对照分别核对load、MMA、
store三类列表；各类内部操作数/shape/valid/init/顺序一致，不声称三类合并
后的时序相同。所有物理分配多重集仅有一份Right从0→32768，其余helper
的地址/形状/数量相同，整函数各片上空间地址上界未增加。

详细证据见`compact_l0_evidence/`。独立128K/B16任务
`task_20261018_014826_4729059881`已提交，尚无设备数值或性能结论，
没有将CPU两槽成立当作性能改善，也没有叠加compact阶段前移。

## 预取与Attention只读结论：避免重复无依据的试验

`key_prefetch_design/`记录将Score Key整理移给双AIV的双槽设计。15组、
1793状态有限模型中，私有scale无覆写/死锁；共享scale存在快lane提前覆盖
慢lane待读数据的明确反例。但GM中转会为长B16增加约132MiB Key交接字节，
不能将AIV的FFTS等待全当可免费利用的时间。这只是模型，没有实现或上卡。

`key_transport_audit/`复核当前依赖：A3的aic_gather/TPUSH/TPOP仍是
UB→GM FIFO→L1，A5直接写peer L1不适用；自动FIFO还涉及全函数L1保留、
新增工作区和手工FFTS事件避让。旧静态Scalar探针也未证明Key页提交占主因。
保留设计，不以换接口名称重复编译或执行GM版。

`attention_readonly_audit/`对照同次现成产物，106项Tile操作序列同序，
主要新增是已试验覆盖的请求几何与地址保护。同步局部差异已记录，但不能
按静态wait数量或整层二进制体积断言icache/热循环失速。本轮没有发现证据
充分的新候选；保留提取脚本、IR/C++摘录及明确限制，不为凑方案增加测试。

## 复现与归档

从上一级目录运行主工作区脚本，避免私有worktree改变依赖根的解析：

```bash
bash reproduce_cpu.sh /全新输出目录 --strategy medium --attention-query-metadata off --spmd-source /候选包目录
task-submit --device auto --max-time 1200 'bash /主工作区/dynamic_full_pipeline_20261017/run_device.sh /全新结果目录 --history 8192 --batch 16 --spmd-source /候选包目录 --source-matrix-reference /本目录/baseline_source --wait-backoff --l1-windows 5'
```

同步候选还需CPU的 `--controls-json` 或设备的 `--policy-json`，精确命令在
对应agent记录中。`summarize_parallel.py`只汇总已落盘结果，不初始化设备。
完整CPU编译产物和二进制本地保留；路径与大小列于各证据目录
`LOCAL_ARTIFACTS.json`，不做hash校验。
