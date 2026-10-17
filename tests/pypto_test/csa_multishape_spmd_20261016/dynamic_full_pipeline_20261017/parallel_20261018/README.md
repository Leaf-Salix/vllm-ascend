# 2026-10-18：动态 CSA 三路并行优化

基线提交为 `593c5d25`。三个 agent 使用独立 worktree；主会话维护隔离源码
加载器与同进程三方对照。首轮候选未改变正式默认；后续compact阶段前移已
在长/短两个策略代表通过并获得收益，现已将两份动态调度源迁回正式目录，
并补齐其余五个验收点。七档均通过本轮完整检查并改善1.73%～4.98%，但仍
全部慢于各自同进程原分task，30%目标没有任何一档达标。所有数学、动态shape
和device metadata合同保留。最新完整表与105个窗口见
[compact_priority_evidence/SEVEN_CASES.md](compact_priority_evidence/SEVEN_CASES.md)。
冻结基线源码在 `baseline_source/`，候选源码在各结果目录的 `source/`。

此后HC四阶段有效消费者通知已在128K/B4、128K/B16、8K/B32三个代表点完成
同进程验证，并迁入正式动态源：相对含compact前移的d56分别降低1.117%、
0.909%、0.354%，15个配对窗口均更快。当前仅有这三个代表点的新性能数据，
不能把下方d56的完整七档表改标为HC通知版。原数学/等待和72核代际未变；
真实设备信号快照还核对了未发布槽保留旧值。完整证据见
[hc_active_notifications_evidence/DEVICE_RESULTS.md](hc_active_notifications_evidence/DEVICE_RESULTS.md)。

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
另一负载代表8K/B32任务`task_20261018_014358_34138622015`已全PASS，
原task998.48/旧SPMD1024.52/候选1001.56μs，相对旧版降低2.241%。
两个策略代表均改善后，仅将两份调度源迁回正式动态包；33文件与已验证
冻结候选逐字一致。只补其余五个验收点，已测长B16和短B32不重测；尚未
更新全七档表。过程第一次阶段提交为`ce7c055d`，不含本段后到的结果。

随后五个补测任务全部exit0，未重测两个已完成代表点。最新完整L1中位数如下；
原task始终是30%性能目标分母，旧SPMD只是本项改动的同轮消融参照。

| 场景 | 原task μs | 旧SPMD μs | 正式compact前移 μs | 对旧SPMD降低 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 513.88 | 559.82 | 531.92 | 4.98% |
| 128K/B8 | 645.16 | 712.98 | 684.44 | 4.00% |
| 128K/B16 | 861.48 | 913.06 | 888.38 | 2.70% |
| 128K/B24 | 1155.24 | 1193.86 | 1163.08 | 2.58% |
| 128K/B28 | 1281.36 | 1357.70 | 1332.30 | 1.87% |
| 8K/B16 | 668.74 | 712.50 | 700.14 | 1.73% |
| 8K/B32 | 998.48 | 1024.52 | 1001.56 | 2.24% |

七档全部S6，正式第4层W8A8权重与合成历史；相对原PTO的输出/TopK/score/
cache-state、A→B→A、图、历史和页表重放、保护区、72核代际通过。本轮没有
验收Native attention、token或整模型，也不能代替S1/S5和整层mixed Q边界。
源级动态入口没有按七点固化；只有原来的两块调度顺序变化，未叠加其它候选。

另一个只读审计考察“仅有compact Had任务的AIC把Had移到QB前”。13例26种
偏序均无环，但这些owner也承担较重QB，旧端点存在13～20μs未就绪等待，
后续还有COEFF、全scale门和双槽背压。没有新收益依据，不生成或执行候选；
模型不是性能仿真，旧时钟不是该变体实测。完整记录见
[compact_owner_priority_analysis/README.md](compact_owner_priority_analysis/README.md)。

## Compact投影L0双槽：真实两槽恢复，整层回退，不采用

另一独立worktree只改Indexer compact投影helper，保持M16/N32/K512，
每pair按KV0/gate0/KV1/gate1顺序累加。两份KV Right先同时存活，迫使
分配器保留0/32768两个地址，消除原gate0→KV1搬运覆盖等待；仍保留
gate0/gate1复用地址0的必要依赖，没有修改编译器或强行指定物理地址。

完整符号CPU首轮通过，156例实际helper AST控制流对照分别核对load、MMA、
store三类列表；各类内部操作数/shape/valid/init/顺序一致，不声称三类合并
后的时序相同。所有物理分配多重集仅有一份Right从0→32768，其余helper
的地址/形状/数量相同，整函数各片上空间地址上界未增加。

详细证据见`compact_l0_evidence/`。独立128K/B16任务
`task_20261018_014826_4729059881`已完成，全部数值、图和保护检查通过，
但原task881.60/旧SPMD910.02/候选919.58μs，相对旧版慢1.051%。
不采用、不扩档；CPU两槽成立不代表整层改善，没有叠加compact阶段前移。

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

compact阶段前移与完整七档已经阶段提交为`d56c9eb6`；后续消融使用
`adopted_source_d56c9eb6/`中的33份动态源作同进程参照，避免仍与未采用
该项优化的593版本比较而混入已获得收益。当前正式源保持此提交。

`ob_hcpost_stream_audit/`只读审计了O_B逐组发布、提前八组反量化的可能性。
它与已有OA→quant→OB组流水不同：单行外层只隐藏每名AIV首行的部分归约，
并增加通知/获取；组外层需16～64KiB常驻累加器或额外GM往返。未形成足够
收益依据，本轮不制作设备候选。保留257例所有权检查、真实旧时钟、生成
尾段摘录与资源边界；既有约25～26μs的OB末端到OUT区间包含HCpost，不能
全部当成可隐藏时间。

另在隔离源准备HC四阶段有效消费者通知候选：只guard RAW/GATES/NORM/COMB
四处发布，真实生产者的发布、全部原数学/等待和72核begin均保留。`hc_active_notifications_evidence/`
保存后续独立证据；在实际结果回收前，不把减少发布次数记作性能收益。

上述HC候选随后三例均已通过，正式迁移只改变`vector_schedule.py`的四处
publish guard，33份源与验证副本逐字相同；详情及适用边界在该目录的
`formal_migration.json`与`DEVICE_RESULTS.md/json`。不新增shape白名单，
不据小幅同轮改善作统计显著性或全部区间达标结论。

`indexer_query_group_audit/`保存G4/G6和叶分配审查：六叶方案不推进，
五叶owner的41→40只是工作量示意、没有设备收益；当前G2/768本就采用N128，
不存在可重复实施的“N64改N128”优化。`qb_left_doublebuffer_evidence/`
保留主QB首次显式Left双槽导致Mat退化的负例及修正版：第二版完整CPU恢复
原Mat窗口，Left两槽、Right单槽，生成事件允许预取下一Left；设备性能待收。
`aic_orphan_notifications_evidence/`只删除SCORE_CUBE/ATTN_CUBE两处无人等待
的AIC发布；CPU全部物理allocation、FFTS与后继获取不变，设备性能待收。
两者都从d56独立出发，不叠加HC guard，不跨任务相减推算组合收益。

两项设备结果现已收齐：QB v2完整通过但870.98→876.70μs、慢0.657%，
五个配对均更慢，不采用、不扩档。AIC两通知独立B16为885.90→875.34μs，
降低1.192%，但与HC guard组合后效果不一致：长B4改善5.248%、长B16
近乎持平（0.009%）、短B32回退0.624%。三个组合任务均通过全部原检查，
仍不采用全区间默认、不按B4测试点建立白名单。正式源码保持仅HC四处guard。
原始窗口与完整限制见各目录`DEVICE_RESULTS.md/json`和
[notification_combo_evidence/README.md](notification_combo_evidence/README.md)。

从上一级目录运行主工作区脚本，避免私有worktree改变依赖根的解析：

```bash
bash reproduce_cpu.sh /全新输出目录 --strategy medium --attention-query-metadata off --spmd-source /候选包目录
task-submit --device auto --max-time 1200 'bash /主工作区/dynamic_full_pipeline_20261017/run_device.sh /全新结果目录 --history 8192 --batch 16 --spmd-source /候选包目录 --source-matrix-reference /本目录/baseline_source --wait-backoff --l1-windows 5'
```

同步候选还需CPU的 `--controls-json` 或设备的 `--policy-json`，精确命令在
对应agent记录中。`summarize_parallel.py`只汇总已落盘结果，不初始化设备。
完整CPU编译产物和二进制本地保留；路径与大小列于各证据目录
`LOCAL_ARTIFACTS.json`，不做hash校验。

## d253之后：预取与量化复用两项均不采用

两个独立源码候选各一次完整符号CPU、两档同进程三方五窗验证，四任务全部
数值/图/保护区/代际PASS。短G2预取在B16慢1.555%、B32快0.299%；O_A量化
复用在长B16慢0.052%、短B32慢0.794%。两项均不采用、不组合、不扩七档，
正式源码保持d253。减少真实TLOAD或UB分配没有自动变成完整L1收益。

详细原始结果与范围：[短G2预取](short_g2_prefetch_evidence/DEVICE_RESULTS.md)、
[O_A量化复用](oa_quant_reuse_evidence/DEVICE_RESULTS.md)。四任务共60窗，
用`report_prefetch_quant_round.py`只读复算；不能跨任务拼接组合收益。
当前Attention只读复核及dense排重分别保存在attention_aiv_d253_audit、
dense_projection_d253_audit。后续当前版本Attention逐tick插桩单独归档，
不计入正式性能证据。

## 当前Attention逐tick诊断完成

唯一128K/B16/S6任务完整PASS，A→B→A三轮共4896条记录。压缩块占gather
累计约94.94%，首压缩块每query约6.2μs，SWA约0.67μs，两lane无固定慢侧。
AIC KV等待与本组gather有较多时间重合；这是诊断范围，不是可直接节省量。
局部wait_flag的Scalar marker不能测作MTE真实完成，区间也不能相加当完整L1。

[三轮细分分析](attention_tick_clock_evidence/ANALYSIS.md)保留统计口径、最慢核、
计时边界、历史排重和独立审阅。下一候选仅删除plan已保证的consumer两层
地址guard，独立CPU通过、两档完整矩阵已提交；结果随后单独记录。

两档随后全部完成，五窗中位数长B16为874.12→874.28μs，短B32为
995.36→996.02μs，分别慢0.018%与0.066%。各有3/5配对窗口略快，
不足以支持优化收益；不采用、不扩七档、不组合其他失败候选。
[完整结果](attention_guard_elision_evidence/DEVICE_RESULTS.md)保留原分task
对照、五窗、策略、源码、独立地址契约审阅与生成代码变化。正式源仍为d253。

`scalar_resource_audit/`核对同轮融合2个和原分task73个kernel的现有raw
二进制及元数据，没有寄存器、stack/spill/local scratch统计；不能据融合
bin较大归因于Scalar溢出或指令缓存。缺失指标记为未知，不记零，也不重复
编译或尝试已有失败的反汇编路径。

## 后续有限复审：避免重复试验

`attention_gather_compare/`核对同轮原task与d253实际wrapper：压缩64行
循环无逐行等待或中间Tile拷贝；SWA与压缩复用同一UB/事件链，没有首压缩
块专属同步。首压缩发生在前一query末块发布之前，排除此前提出的一个
时序猜测。实际score与KV的UB布局有差异，但现有端点不能据此归因。
同时更正旧审计：原task压缩路径确有两次TEXPANDS，非互斥一次。

`hc_tail_clock_audit/`只读三轮已有phase clock：COMB发布结束先于最早QA
发布，平方和/逆RMS暂存已是UB Tile，未发现新的可删GM中转；输出尾部
约27μs包含工作和同步，不能全当可隐藏时间。`indexer_interval_reaudit/`
核对当前五份策略源码与旧模型一致，没有重跑已覆盖的分组/叶分配模型。
这三项均无新编译、无NPU作业，不制造新的性能通过结论。

## 独立PV N256候选：完整符号CPU通过，首档设备待收

`pv_n256_evidence/`仅改Attention PV输出列tile为256，两个块覆盖512列；
不改K128、64heads、FFTS、三槽、AIV或策略分界。唯一CPU无样例shape，
58动态轴/130根/single MIX通过。真实Acc为0/65536各64KiB，Right两次复用
同一64KiB；QK→PV与回边FIX→M依赖保留。全函数减少两次TEXTRACT/TMATMUL/
TSTORE、8对set/wait和两条PIPE_FIX，无新增同步及片上内存峰值。
这些只是编译证据。唯一128K/B16设备任务为
`task_20261018_035635_317824730121`，三方五窗结果待收，不预记收益。

PV首档随后完成：128K/B16原分task870.82μs，d253参考881.40μs，N256候选
873.04μs；相对d253降低0.948%，五个配对窗口全部更快。完整PTO精确对照、
图/状态/保护区及代际通过；相对原分task仍慢约0.255%，不是30%目标达成。
据此只补长B4与短B32两代表，任务分别为task_20261018_040259_32962431633、
task_20261018_040259_329628229360，源冻结不变，正式代码尚未迁移。

PV三档最终均完整PASS：128K/B16的d253→N256为881.40→873.04μs（降低0.948%，
5/5更快）；128K/B4为557.58→534.14μs（降低4.204%，5/5更快）；8K/B32为
998.02→1017.26μs（慢1.928%，0/5更快）。各自原分task分别870.82、505.30、
978.36μs，三个点仍均未达到相对原分task降低30%的目标。
不全局启用，不以B4/B16点名形成白名单；正式源码保持d253。
完整五窗/策略/协议/任务见pv_n256_evidence/DEVICE_RESULTS.md/json。
长/短历史、T、Score策略与输出tile同时变化，不能仅据这三点就把早KV或
任何单一差异定为原因；后续只读实际产物审计另行记录。

PV后续仅补两个可区分区间因素的短历史点，源保持同一冻结N256候选：
8K/B16/T96（七档之一，G2/OA128/OB96/晚KV），
8K/B20/T120（区间内探针，G2/OA256/OB128/晚KV）；既有8K/B32/T192为
G2/OA256/OB128/早KV。二者不用于制造测试点白名单；只有区间实测证据
一致才考虑正式策略。任务与完整命令见pv_n256_evidence/interval_registry.json。

PV实际资源复审未找到短档独有变化：两档Attention外allocation分别2178/2020条
全同，AIV全同，各空间地址上界全同；QK Acc原本就与PV首Acc同址0，跨tick
FIX→M两版均存在。两档都只是PV Acc 2×32→2×64KiB、Right单槽32→64KiB，
以及相同的4→2分块事件缩减。不能据此把T=144直接定成启用边界。
证据见pv_n256_allocation_audit/；最小constexpr接线仅作设计保存在
pv_n256_evidence/POLICY_WIRING_PLAN.md，未改正式实现。

WS移到AIV的额外只读复审见ws_aiv_readonly/README.md：有旧分task的
Vector→Cube WS历史，但没有当前动态G6 exact反向消融；直接迁移让单step
交接24→768KiB，半组384KiB超过单AIV UB，还要与TopK竞争并核对累加顺序。
当前不推进GM直迁，无新增编译/NPU。

两个短历史区间点均完整PASS但性能回退：B16/T96参考688.66→候选693.46μs
（慢0.697%，1/5更快）；B20/T120为808.90→821.04μs（慢1.501%，0/5更快）。
同T96的长/短表现相反，短T120走晚KV也回退，不能按T144或输出tile边界
启用N256。实际编译审计两侧均46个wait站点、100 tick未就绪退避，未发现
把reference漏打hook的证据。报告见pv_n256_evidence/INTERVAL_RESULTS.md/json。

当前只补剩余长历史B8/B24/B28，完整任务清单在long_remaining_registry.json。
另由独立agent在私有源准备PV constexpr接线，默认128，仅将“既有长G6
平衡单根域用256”作为待测策略候选；没有修改正式源码，也不宣称已找到
性能分化的根因。新增参数不得增加runtime根或静态化实际输入形状。


## 2026-10-18：PV N256七档收口，接线保留为历史候选

剩余三档全部完整PASS，按statistics_us角色键读取五窗中位数：
128K/B8 d253 677.70→N256 673.82μs（降低0.573%，3/5更快）；
128K/B24 1186.54→1183.62μs（降低0.246%，3/5）；
128K/B28 1352.18→1355.54μs（慢0.248%，1/5）。
初读grep显示顺序时把三方角色读反，已立即更正；最终汇总脚本只按明确角色键取数，
不按JSON键插入顺序推断候选。原始结果未改动。

完整七档另加短B20均通过原PTO精确对照、图/状态/保护区及72核协议；
长G6域也有回退，短域三点全部回退，故不统一启用、不造B4/B16白名单、
不做长历史边界扩测。没有一个N256目标点达到相对同轮原动态分task降低30%。
详细表和可只读重算脚本分别为pv_n256_evidence/SEVEN_AND_INTERVAL_RESULTS.md/json、
parallel_20261018/report_pv_n256.py。正式33份Python经直接文本逐份核对仍与d253一致。

四文件PV constexpr接线在独立副本完成，N128/N256各一次完整CPU：130根、
58动态轴、单MIX与sync_start保持；同控的完整分配序列和compiled_meta相同，
去掉pto SSA调试注释后整份生成C++逐字相同。它只证明接线无额外运行时代码，
不构成性能采用依据，源码/patch/审计/日志保留历史，主包不迁移。
大编译产物本地保留路径和大小，无hash。资源复审仍不能解释长短分化根因。

下一独立候选检查Query反量化的head分组：在原4/16/64基础上补8/32，
按实际T与48AIV均匀波次减少重复scale/RoPE准备；单独agent交叉核对生产者等待、
尾行数学、head owner与UB，不改矩阵乘、不叠加PV或此前失败优化。


## 2026-10-18：Query反量化按完整波次扩大head分组，三档待实测

独立候选仅改变query_projection_stages.py的分组控制：在完整八行时按
64/32/16/8降序选择能形成48AIV整波的最大head组，其他情况仍4；新加8/32。
等价选择条件为T>=6*h且T%(6*h)==0，不是batch测试点白名单。
T48/144/192分别由4/4/16变8/8/32，每核head数不增，重复scale/cos/sin/
swap准备任务数减半。T96保持16，有任何八行尾块时仍固定4，原尾块
recip(sqrt)与完整块rsqrt数学分别保持，不改QB矩阵乘、输入或工作区。

私有源owner模型枚举T1..512：每token/head唯一写、逐核head及token-head
负载均不增加，数学主体和其它函数AST一致。独立复审确认入口已等全部
24个QB owner、QUERY发布/消费覆盖全部48AIV，QR/ROPE owner也完整获取；
因此本候选不改schedule与notify/wait。T49等尾部不得误用8头分组。

一次完整符号CPU通过，130根/58动态轴/单MIX与sync_start不变；DQ完整
分配地址/形状与旧版相同，DQ UB末端131072B，全AIV UB163872B，AIC
Mat/Left/Right/Acc为475136/65536/65536/131072B；全部硬件调用和局部
同步计数不变。CPU不代表设备精度或完整L1收益。

三档冻结源同进程三方五窗任务：
128K/B8 task_20261018_043459_392309626395；
128K/B24 task_20261018_043459_392319716700；
8K/B32 task_20261018_043459_392330520909。
其余四个目标点规则未变，当前不重测。registry与独立review位于
parallel_20261018/query_head_group_evidence；此时未预记PASS或采用。


Q_B另一路只读复审已结束，暂不做M64/N512/K256的CPU候选：列块128→64
并未减少列分工最忙核的6×256=3×512个有效列；T96逻辑输入读取44→70MiB，
其中权重32→64MiB、QR12→6MiB。M/N联合分工在部分大T可降低峰值，但权重
读取仍增加，不能据列块轮数减半宣称收益。K256×N512的Right需128KiB，超过
64KiB必须继续切分；高层Mat可由AutoTile拆分，因此不称其必然编译失败。
报告与可复算模型见parallel_20261018/qb_tile_readonly，无新增CPU或NPU。

另一私有候选检查T<=48的Q_A统一M16块分工：原T24/48仅8个AIC参与QA，
扩核需改global block ordinal与QA消费者等待；只改DENSE_M或ceil无法把T24
第二个尾块分到另一组核。新方案最多24核，T>48保留原路径，正在一次完整CPU。
增加QA参与核可能延迟它们原本可提前执行的Indexer投影，不预估整层净收益。


## 2026-10-18：采用Query完整波次分组，三档完整结果保存

上一节三个任务均完整PASS。按明确角色键读取五窗中位数：
128K/B8 原分task636.22、d253参考677.02、候选674.40μs，降低0.387%，3/5更快；
128K/B24为1171.34、1185.34、1172.80μs，降低1.058%，5/5更快；
8K/B32为981.00、997.08、995.62μs，降低0.146%，3/5更快。
B24收益较一致，另两档幅度接近测量波动，不宣称大幅提速。三档相对原分task
仍未达到降低30%。其它四个目标点规则不变，没有重测或改标旧证据。

已把冻结候选的query_projection_stages.py一份文件迁回正式动态包，其他32份
保持d253；不叠加PV、预取或量化失败候选。源码直接按实际T做均匀波次分组，
没有新增Host值、Tensor/Scalar接口、工作区或同步。证据见query_head_group_evidence
和report_query_head_group.py。完整PTO精确PASS不等于Native/token验收。

最终TopK根的128前缀认证仅做只读评审，不推进：当前mrgsort没有输出限512
接口，exhausted表示任一输入耗尽；不同有效输入长度可用，阈值可从UB直接
取Scalar，但有本地流水等待、R+1读值及失败fallback。R4的128前缀要求各根
恰贡献128且边界无同分；当前R5第五根可仅1项。R6也只是两次原最终merge，
不能把32768候选全排序的成本当作可节约量。未隔离纯merge耗时，收益上界
未知，不为该方向另做CPU/NPU。见final_root_prefix_readonly/README.md。
