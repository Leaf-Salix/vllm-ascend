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


## 2026-10-18：小T QA M16分工通过CPU，独立两档已启动

私有33份源以d253为参照，仅qkv_stages.py/vector_schedule.py变化：
T<=48将ceil(T/16)个M16块按全局block ordinal分给最多24AIC，N128/K256
保持；尾行读取valid_shape，写入原pad16工作区。QR前的QA owner等待数
同步改为8*ceil(T/16)，不遗漏T24的core8..15；T>48完全保留原路径。
只改DENSE_M或floor→ceil仍会把T24尾块留在原8核，因此没有采用该错误简化。

一次完整CPU通过；源模型枚举T0..96及127/128/129/192/384；独立审阅另核对
T1/15/16/17/24/31/32/33/47/48/49，实际writer与waited集合相同、有效元素
唯一写，原M16尾数学仅局部变量更名，全NORM等待和全24QA发布保持。
已有2365条alloc的地址/shape/有效维度归一表达式不变，新增分支15条。
QA前同步序列相同；QA后4716条同步经局部ID归一相同，紧接投影的三对
M→FIX局部事件ID顺延，不把这描述为额外同步。详细audit另随本候选归档。

正式当前已提交Query head分组9d5327e5，本QA参照/候选刻意保持先前冻结
d253，以做独立消融，不重新编译或把DQ收益混进QA。两档任务：
128K/B4 task_20261018_044522_413498115985；
128K/B8 task_20261018_044522_413502612248。
命令及独立审阅在qa_small_m16_evidence，此时未预记完整验证/性能PASS。
新增QA owners原可早做Indexer投影，且T32权重逻辑读取8→16MiB、T33..48
16→24MiB；更充分用核不能替代完整CSA实测。若采用，再独立验证与DQ组合。


## 2026-10-18：QA M16 第一版两档完成，保留区间改进方向与三项只读结论

`qa_small_m16_evidence/DEVICE_RESULTS.md/json` 保存完整结果。五窗完整L1中位数：
128K/B4 原分task502.44、d253参考543.60、QA v1为523.90 μs，降低3.624%，4/5窗更快；
128K/B8为648.00、679.66、683.58 μs，回退0.577%，仅1/5更快。
两档均通过原PTO精确对照、A→B→A图重放、状态/保护区及72核代际，但都未达到
相对原分task降低30%。参照和候选均不含后来采用的DQ分组，不能与另轮DQ收益相加。

T<=48统一M16不采用。B4原路径的两块M16只改变owner，不增加总权重读取；
B8原M32+M16改成三个M16使权重逻辑读取16→24 MiB，这是结构性区别而非已证明
的唯一硬件归因。v1虽保留大T源码，完整编译仍使原M32/M64局部事件排列改变：
各有8/4组相邻同步序列内部顺序变化，数学语句与同步边界保持，不能声称机器行为不变。
因此另建v2，依据原纯尾块区间16<T<32启用QA M16并行，并使用constexpr编译分支。
区间外必须只生成旧路径，区间内仍以真实T计算块数和尾部，不写B白名单。
v2正在独立CPU核对，不用本节v1结果预记v2性能。

只读复审另归档三项：
- `wqa_cache_policy_readonly`：当前权重cache_policy声明可改变IR，但已有两轮候选的
  实际TLOAD C++不变；当前QA也没有显式L2 hint。因此不能把IR标签当作真实旁路缓存，
  不为单改声明重复编译或上卡；未修改依赖仓。
- `qa_dq_composition_readonly`：QA与DQ修改文件不重叠，但存在QA→QR→QB→DQ依赖及
  Indexer竞争，收益不能相加。v1组合没有生成或上卡；下一候选以正式9d作为冻结参考。
- `dq_rope_flat_index_readonly`：DQ预存八行展平索引在DQ侧等价，可去掉部分索引向量指令；
  但同一rope_swap还被KV完整32行路径按0..63列索引消费，直接改布局会越界。
  原两函数方案否决，不以T24测试会掩盖的问题作为通用实现。尚未做编译或上卡。

正式动态算子保持9d5327e5，未合入上述QA、缓存声明或RoPE候选。大编译产物只记录路径
和大小，源/审计/原始日志/精度与性能报告入Git；不做hash校验。


## 2026-10-18：Query连续分工与QA区间constexpr两路独立上卡

Query候选以正式9d为参考，把完整八行的(row_tile, head)线性工作量均分为
48个floor区间，跨64头行边界再切chunk；T8/T16和含尾行时保留原路径。
这样T144准备144→60次、T168准备336→66次，最大每核head计算量不增加。
只改变query_projection_stages.py，不改变每个head的数学、等待或工作区。
T1..512 owner/数学AST通过；唯一CPU中2370条Tile分配地址/shape均相同，
完整同步序列归一相同，AIC C++相同。长度1的pipeline实际C++可正确跳过
双头主循环、排空预置事件并执行单头尾部；仍需设备边界验证。
报告/source/CPU摘要见query_linear_evidence。任务：
128K/B24 task_20261018_050316_27283926583；
128K/B28 task_20261018_050316_27292720517。

QA v2只在16<T<32开启M16块并行，constexpr False/True各完整CPU一次。
False省略新参数仍兼容，2370条分配和四份实际生成代码去行注释后与9d一致；
True保留动态T的块数/尾行，130根/58动态轴/单MIX不变、QA外分配不变。
实际Host区间T18/T24/T30自动开启，T16/T32/T48关闭；不需要测试强制override。
为排除T18同时进入原G2策略，本轮先测同G6的B4/T24和B5/T30：
128K/B4 task_20261018_050316_2729667380；
128K/B5 task_20261018_050316_27300510000。

两路分别冻结正式9d参考和各自候选，未互相叠加。所有任务采用相同正式权重、
同进程三方五窗、完整L1首尾及100tick失败轮询退避。此处只记录提交，不预记PASS，
不将CPU工作量或编译通过当作性能结论。正式包尚未修改。


## 2026-10-18：连续head分工不合入，QA v2两档正向并补区间下端

两路四个任务全部完成，原PTO精确输出/TopK/分数、图状态/保护区及72核协议PASS。
只按同进程三方角色键汇总五窗完整L1中位数（单位μs）：

| 候选 / H / B | 原动态分task | 9d单SPMD | 候选 | 相对9d降低 | 更快窗 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 连续DQ / 131072 / 24 | 1178.12 | 1174.38 | 1181.24 | -0.584% | 1/5 |
| 连续DQ / 131072 / 28 | 1312.22 | 1337.28 | 1328.84 | 0.631% | 3/5 |
| QA v2 / 131072 / 4 | 515.94 | 548.26 | 540.98 | 1.328% | 4/5 |
| QA v2 / 131072 / 5 | 582.92 | 615.46 | 604.14 | 1.839% | 5/5 |

连续DQ虽减少准备次数，完整性能长B24回退、B28仅小幅正向，因此不合入正式包，
不根据两点人为制造B28白名单。真实T40的H8192/B8/S5长度1流水尾部已单独提交
`task_20261018_050845_41225224225`，仅做功能和因果前缀验证，不做Native性能，
结果保存用于动态候选的边界记录；它不改变本轮不采用的性能判断。

QA v2初始两点较一致，补原策略会选择G2的T18/H131072/B3：
`task_20261018_051020_44336828121`，仍以同轮9d为参考、原分task为目标分母。
此前T24/T30均为G6，不能代表跨Indexer策略后的整层净收益。关闭路径已有同控实际
C++逐字一致证据，不为关闭分支重复全量七档。正式源码仍未迁移QA。

结果可通过`report_query_qa_interval.py`只读重算；QA登记新增B3后须等该任务结束
才重新生成完整报告，此刻已保存的报告只含前两点。四个任务都未满足30%目标。

独立DQ本地索引方案见`dq_local_swap_readonly`：完整块内用arange(512) XOR 1，
不改变共享rope_swap和KV，不新增GM/root；A3 XOR会展开六个向量阶段，并非单指令。
现有mask gather不支持相邻交换，reshape+[1,0]也不会逐行广播；当前TINTERLEAVE
架构guard不含A3。仅据本地API说明，不断言硬件不存在该能力。下一候选基于正式9d，
独立核对源数学和一次CPU的真实资源/同步，不叠加已否决的连续DQ，不提前记收益。


## 2026-10-18：S5单head边界结果与大额省工方向复审

连续DQ的H8192/B8/S5（T40）已结束：真实单head尾流水、两次图重放、
地址/保护区/72核代际通过，functional_passed=True。原PTO S6前五行的
因果对照以额外第六行X和−X各一次，输出/TopK/分数以及所有可比cache/state
字节逐bit一致；只排除额外第六行的合法写入，覆盖全部实际S5写入。
Native逐bit不一致仍保留：输出max_abs=0.0234375、RMSE=0.0031557713，
TopK/cache/state也有差异，原顶层passed=False没有改写。没有性能测量，
也不改变连续DQ不采用的决定。见query_linear_evidence/S5_BOUNDARY_RESULT.md/json。

两项高价值方向只读收口：
- indexer_work_reduction_readonly：有效QK dot没有可直接删除的计算；将稀疏WS
  搬到AIV会把原聚合输出交接扩大32倍，逐bit归约顺序也未证明。既有短尾裁剪、
  WS搬运和TopK前缀方案均已排重。最终根只1–2次merge，exhausted不能保证限512项，
  未找到足以支持新一轮编译的省工候选；不把算术上限当设备收益。
- qb_dq_handoff_readonly：已有长B16三轮逐核诊断中，DQ可进入时全部QB已就绪
  12.72/13.26/14.64μs；QB再拆行/head通知不能提前当前消费者。DQ内24次QB检查
  累计约3.1–3.4μs是已就绪信号读取成本，不是QB生产未结束。QUERY由后继
  SCORE_VECTOR→plan获取链覆盖，之前合并等待已有完整负例，不重复上卡。
  逐列交接不减少GM字节，还新增通知；报告限于现有源码和已观察规格。

另一路局部索引首次CPU明确失败于A3 TXORS仅支持i8/i16，而非PyPTO接口表面
可接收的INT32，未生成有效C++。保留失败记录后只允许一个INT16修正版：
0..511索引范围不会溢出，XOR后转回gather要求的INT32；不修改依赖仓，
额外转换、真实UB与事件仍须以后端输出为准。此时没有设备或性能结果。


## 2026-10-18：最终采用G6两M16尾块区间，保留其它区间旧路径

QA下端G2/B3任务完整PASS，但五窗中位数原分task548.36、9d572.84、候选586.20μs，
回退2.332%，仅1/5窗更快。最终Host条件收窄为 `group==6 and 16<tokens<32`。
根据现有policy，这等价于长历史、query上界至少6、24<=T<32；沿用原G6分界，
不建B4/B5白名单。G6/T24和T30已分别改善1.328%/1.839%，但T/G/负载同时变化，
不宣称G6是唯一硬件原因或所有历史/混合Q均加速。后续范围验证仍需保留。

正式迁移policy/entry/cube_schedule/vector_schedule/qkv_stages五文件。
除Host条件与注释外，实际kernel AST与已测v2相同；关闭分支已有同控CPU实际
C++逐字相同证据，未追加编译或上卡。996个Host描述符组合核对除新开关外原策略
字段不变，开启的56种组合均符合新区间。全部33源与source_selected_g6逐文本一致。
原v1/v2和三档正负结果均保留，详细决策、审计和迁移文本见qa_small_m16_v2_evidence。
正式仍使用9d的DQ整波策略，连续DQ和局部XOR没有合入。

LD_DEV只读复审找到历史完整CSA的0/25tick负例；未找到当前动态+100tick的精确
同轮消融。ABI兼容并不等于值得继续扫描，保留报告，不为该方向新增设备试验。
不把异轮1327.28/1146.62与旧1067.36μs直接计算为同轮因果比例。

本轮采用项只改善一个目标点及一个区间代表点，七档没有任何一档已达到相对
原动态分task降低30%的目标。目标保持进行中，不把过程归档或阶段commit当完成。


## 2026-10-18：DQ本地XOR索引两次CPU收口，不采用

记录见parallel_20261018/dq_local_swap_cpu_evidence。v1使用INT32 TXORS，
虽符号解析保留58轴，A3 PTOAS明确拒绝该dtype，未生成有效C++；失败源与日志保留。
v2只做一个修正：TCI生成0..511的INT16、TXORS后显式转FP32再INT32，全部数值精确。
A3没有直接INT16→INT32 cast实现，不能把前端接受dtype当后端可执行。

v2完整CPU通过130根/58轴/单MIX；TXORS输入67328、临时65536、输出68352，
各1024B互不重叠，最终INT32索引仍为36864/2048B。全部空间地址上界与9d相同，
全AIV UB末端163872B，DQ以外Tile分配序列相同。但同步不是完全不变：局部event
重新分配，DQ发布后一个MTE3→V等待被PIPE_ALL替换，后续也有等待前移。

PTO层少1次TLOAD、3次TCVT、2次TMULS和1次TADD，增加TXORS；不能止于这层
就宣称更快。A3 TCI_b16使用1792B scratch，原b32为768B；固定512的模板展开
模型中Vector intrinsic调用由21到34，另多2个内部PIPE_V。TXORS有5个内部PIPE_V，
外层C++仅少4个PIPE_V。它以新增准备/屏障换少2KiB读取，收益依据不足，停止CPU，
不继续占设备。调用数是源码展开模型，不是机器指令计数或运行时性能。

三套33源、两次日志、CPU结果、必要生成片段与事件差分全部归档；大生成物保留
本地路径/大小，无hash。未修改正式源码/依赖，没有设备精度或性能结果。


## 2026-10-18：跨核搬运与权重复用三路并行复审

正式基线为66882634。当前三路分别审计Q_B跨M行块复用权重、输出投影A的显式
缓冲生命周期，以及A3 Cube/Vector交接；各自在独立worktree工作，未改共享依赖。

已完成的只读结论与证据位于parallel_20261018：
- acc_to_dq_transport_readonly：当前A3 Acc→AIV仍经GM，T96有效INT32交接单程
  12MiB、双向24MiB；小FIFO仅减少驻留容量，不自动省掉一写一读。当前AIV在DQ前
  等待QB之后的compact Hadamard，直接加入有限FIFO消费ACK会形成背压环。
  必须区分前端可表达、后端可执行与硬件直接传输，未提出无证据的直接替换。
- attention_kv_multiconsumer_readonly：同一query唯一归属一组，AIC计算全部64heads，
  双AIV各搬64候选，已有三槽KV复用；并不存在每head/Indexer分片重复搬同query KV。
  跨query物理row重合尚未测量，不能从共享prefix推定节省量，也不能省略READY/ACK。
- oa_pair_explicit_lifetime_readonly：以前OA pair的外层pipeline使两条Right流各申请
  双槽，导致128KiB超过64KiB，或被退为单槽。新设计借鉴HC_pre的手工交错生命周期，
  显式K256 Mat双缓冲共384KiB，任意时刻只保留两Right和两Acc。其loop-carried
  coalesce、事件和真实性能仍待验证，未把容量算术当成功实现。

当前只授权Q_B单项完整CPU lowering及OA独立最小CPU探针，尚无本轮设备或性能结果。
Q_B首选仅B常驻，保留M128与四段K256 INT32累加；QR+B同时常驻的M64备选虽然
进一步减少逻辑QR读取，却增加50% MMA/FIX和Right载荷，因此暂不叠加。该原理在
旧分task中已有先例，本轮核对的是当前动态SPMD双M区间，不能把它宣传为新发现。


## 2026-10-18：O_A显式双缓冲v1失败、v2通过CPU

完整最小探针归档于parallel_20261018/oa_pair_explicit_mat_cpu_evidence。
v1真实分配已达到Mat384KiB、Left/Right各64KiB、Acc128KiB，但MemoryReuse
在Mat循环回边插入三次64KiB Mat→Mat，当前A3 PTOAS拒绝，不存在可执行C++。
没有修改依赖，也没有忽略后端错误；源、命令、失败日志与IR均保留。

v2只改变这个明确的表达边界：外层普通K1024循环，内部四个固定K256数学块
手工交错供数，仅两条Acc跨回边；每条输出仍按原K128 FP32顺序累加。CPU完整
通过，实际Mat仍384KiB、Left/Right各64KiB、Acc128KiB，没有Mat→Mat。C++中
下一组三次TLOAD已位于当前组提取前，当前组仅等待自己的MTE2事件，不提前
等下一组，故生成的依赖保留供数窗口；这不是设备重叠或性能收益的证据。

该探针使用真实八组NZ根布局与动态valid_rows，但仅覆盖每组三对N128，余下
两个singleton没有包含，不能当成完整CSA或正式动态算子。下一步正在独立补齐
完整输出、唯一写者与组通知/等待，尚未修改正式包。

Q_B另一分支的一次完整CPU在Right128KiB>64KiB处失败；还没有本轮NPU任务。
已定位到pipeline区域CanonicalizeIOOrder提前Mat→Right extract，导致两个
K256 Right同时活跃。原路径真实K本来也为256，不是数学K突然变大。失败源与
修正候选仍在独立worktree，后续结果另记，不将当前过程提交误报为性能改善。


## 2026-10-18：两版Q_B通过CPU，独立提交受影响区间代表

B-only v1失败已定位：CanonicalizeIOOrder只在pipeline祖先内将Mat→Right
extract视为load前移，使K256/K512两份64KiB Right在首MMA前同时活跃；
不是first Right存活，也不是原K128改K256。原正式Q_B实际也采用K256。
修正v2去掉该pipeline，把固定四K256数学块写成手工QR双槽供数，显式
Right提取在上一MMA之后；不改依赖。完整CPU通过，B为256KiB、两QR槽
合64KiB，Right单64KiB，动态尾行与原四次INT32累加顺序保持。

独立QR+B版进一步在N列循环外加载QR容量256×1024，valid为实际device T。
QR位于Mat0，B位于262144，两者恰512KiB；没有固定64KiB Mat保留底座，
Left/Right各只用一个槽，Acc保持动态compact。CPU通过130根/58动态轴，
没有新增GM根、phase或依赖安装。相邻B复用的M→MTE1→MTE2链仍保留，
不能宣称整K载入没有启动成本或已与前一MMA重叠。

两版均仅由Host已有shape描述符选择128<T≤256的自然双M区间，实际T、
有效行和M循环在核内读取。关闭支经常量折叠后的源AST与正式相同；这不是
重新编译的关闭二进制逐字等价证明。参考均为正式66882634，不能把失败的
B-only v1当运行基线，也不能把两个候选的不同设备绝对时间直接判优。

已提交三方同进程五窗完整L1及完整精度/图状态验证，尚未收取结论：
- B-only：128K/B24 task_20261018_055057_11658413779；
  8K/B32 task_20261018_055057_116588019925。
- QR+B：128K/B24 task_20261018_055122_11762576461；
  8K/B32 task_20261018_055122_117629623586。

对应冻结源/CPU审阅/命令分别在qb_b_resident_evidence与
qb_qr_b_resident_evidence，报表由report_resident_round.py只读重算。
正式算子仍不变；只有设备验证完成后才讨论采用，不把逻辑字节节省当时间收益。


## 2026-10-18：O_A完整动态接线通过CPU并提交代表档位

独立候选保存在oa_pair_manual_evidence：沿原OA_COLS128/T≤96区间，每组
三对N128与两个singleton；pair归3g+p，single归下一组前两核(3g+3+s)%24。
本组五owner互异，全卡24对+16单块，每核最多三个N128输出，覆盖全部八组。
两端使用同一动态区间谓词，新增wait只等待该五owner并统一一次业务cache维护。
原信号槽/代际/工作区和GM根均不增加。所有本核O_A仍先于O_B的量化等待，
没有把未完成生产放在可能阻塞的消费之后。

完整CPU只使用longG6、OA128/OB96、QA=false，130根/58轴/单MIX保持。
pair内六份Mat总384KiB，在完整地址空间从64KiB起，末端448KiB；全AIC
Mat末端464KiB。Left/Right各64KiB、Acc128KiB，不能把局部384KiB误报为
全函数地址上界。T由device shape传给valid/Acc/store，pair中间无Mat copy。
当前块只等待自己的MTE2事件，下块TLOAD已发出；Right覆写事件继续保留。

代表设备任务各自重新JIT实际policy，不能用上述CPU替代其实际编译和正确性：
128K/B4 task_20261018_055534_12872405208（含已有QA小T策略）；
8K/B16 task_20261018_055534_128729815924（G2）。两任务尚待完成。
候选独立于两版Q_B，没有先合并多个变更再解释收益。


## 2026-10-18：两版Q_B四任务完整PASS，均无整层收益，不采用

正式权重、三方同进程五窗完整L1，单位μs；“参考”为正式668单SPMD。

| 候选 / 历史 / B | 原分task | 参考668 | 候选 | 相对参考降低 | 更快窗 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B-only / 131072 / 24 |1170.54|1164.80|1168.00|-0.275%|2/5|
| B-only / 8192 / 32 |979.06|997.16|1012.34|-1.522%|0/5|
| QR+B / 131072 / 24 |1170.90|1185.06|1185.38|-0.027%|2/5|
| QR+B / 8192 / 32 |977.96|993.88|997.86|-0.400%|3/5|

四任务均完成输出/TopK/分数、A→B→A图重放、cache/state/保护区以及72核
代际检查。逻辑B读取64→32MiB，进一步合并QR读取也没有证明整层净收益；
不能由理论带宽节省直接推算时延，也不能把两作业不同设备/时刻的绝对值
当候选间对照。两版都不迁移正式源、不扩测七档、不另建测试点白名单。

完整样本/策略/命令/原始trace与日志见各自evidence，ADOPTION.json明确
不采用，DEVICE_ARTIFACTS.json只为大编译产物与数值快照登记路径及大小。
report_resident_round.py、collect_resident_round.py可只读重算/收取，不启动
新设备任务。QR+B相对同控B-only v2，535项非QB AIC分配中262项Mat换址，
shape/dtype/容量不变；1812项AIV相同。这不是与正式668所有机器码相同的
证据，也不能据换址直接判定退步原因。完整性能已作为采用决策依据。

O_A双输出候选仍在独立设备验证，未把Q_B负结果扩大为所有优化方向均不可行。


## 2026-10-18：O_A双输出完整四档收口，区间整体暂不采用

四档均通过原分task PTO的精确输出/TopK/分数、A→B→A图、cache/state、
保护区与72核代际；性能为三方同进程五窗完整L1，单位μs。

| 历史 / B / S | 原分task | 正式668 | O_A pair | 相对668降低 | 更快窗 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K / 4 / 6 |505.98|555.80|527.14|5.157%|5/5|
| 128K / 8 / 6 |640.68|694.32|692.80|0.219%|3/5|
| 128K / 16 / 6 |870.96|884.84|889.28|-0.502%|1/5|
| 8K / 16 / 6 |676.50|710.28|695.36|2.101%|5/5|

前两代表的收益没有推广到整个T1..96区间：B8很小，长历史B16出现回退。
正式源继续保持66882634，不直接迁移四文件，不按测试点白名单绕开回退。
下一步只能基于真实工作量/分核机制建立更合理区间，或改变实现；不能用
四档平均值或异轮绝对值代替逐档结论。四档相对原分task都仍未达30%。

S1/H8K/B16/T16与S5/H8K/B7/T35均完成：功能、两次同输入graph与原PTO
S6因果前缀精确比较通过；Native零容差失败，顶层passed保留false。S1/S5
没有采性能，不代表连续decode、mixed Q或A→B→A metadata覆盖。Native
差异的明细及同规格历史对照由独立边界报告记录，不能自动归为既有差异。

正式迁移前的关闭分支审计另做一次OA256/OB128完整CPU：130根/58动态轴
保持，归一化完整PTO、553项AIC分配、AIC1700/AIV3088局部事件与同控参考
一致；不宣称wait辅助头和机器码全文一致。关闭组流水仅源AST等价。

结果、任务日志、原始trace、冻结33文件双侧源码与CPU审计在
parallel_20261018/oa_pair_manual_evidence；DEVICE_ARTIFACTS列文本及本地
大产物路径/大小，无hash。与两版Q_B负结果一起保留，避免重复试错。


独立S1/S5验收补充：详见oa_pair_manual_evidence/boundary_review/README.md。
S1输出对Native的max_abs=0.03125、RMSE=0.003030962，TopK16/16行集合不同，
50个替换项；S5 max_abs=0.0234375、RMSE=0.003181939，TopK35/35行集合不同，
124个替换项。检索现有890份result类JSON后，未找到同H/B/Q正式源旧结果，
因此相对旧正式源的Native差异仍未归因；不把它自动豁免为旧差异。
两档因果前缀精确对照均通过，故本轮支持的是与原PTO前缀数学一致的结论。
队列completed(exit=1)对应保留的Native精确失败，不是编译或设备崩溃。
内部70个allocator外侧红区与31项guards通过；外部输出未设独立红区，
协议检查为phase0代际+1及所有槽保留字不变，不扩张为全部业务phase owner审计。
