# CSA TND 前半段差距与精度保持优化实验（2026-10-27）

## 实现与范围

正式交付控制实现仍为 `11b380e3fc16efdc1cb85a0b2888720872c70655`，当前分支
`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。新工具链固定公开 PyPTO
`d44925c17d9ff7d7fa8e5067e41c4b4f80bf3063`、PTOAS
`914863ed3d0eb3e92a4433a4a706a86cabf4c91f`，Simpler 保持
`57e499238fd4ee3548947ad860be456aaeeb16b5`。旧验收环境没有覆盖。

本轮采用 8K/B32、S6、T192、正式第4层权重与合成 hidden/history；TP1、NZ2、
Native deterministic level1、HCCL deterministic、静态包成功选择、FULL graph。
Native SuperKernel 开关为0；不能冒充截图里 SK1 绝对耗时。测试范围覆盖 HC-pre 到
HC-post。它不是整模型 token 测试，不能由单层逐 bit 结果推出本轮新工具链整模型逐 token 已通过。

## 完整链路普通计时

单位 μs，20×100回放，五臂轮转并反转排列，不剔除离群值。

| 实现 | 均值 | 中位数 | 与 Native 比较 |
| --- | ---: | ---: | --- |
| Native | 1247.861 | 1247.944 | 基准 |
| 我方精度 Direct | 1585.767 | 1584.147 | A/B八边界逐bit一致；ABA与guard通过 |
| nalinaly c346 完整SPMD | 1088.158 | 1088.061 | 输出relative L2约0.333%；自身eager/raw/ABA稳定 |
| nalinaly止于TopK＋plan | 548.285 | 548.397 | 七个前段边界A/B与其完整SPMD逐bit；不计算最终输出 |
| nalinaly止于heads | 874.112 | 874.479 | 七个前段边界A/B与其完整SPMD逐bit；不计算最终输出 |

任务 `task_20261027_090855_15411730797`，完成exit0；原始结果见累计计时归档。
截断版最终输出未执行，其对Native的output差异不作为精度失败或精度候选结果。

## 分开的L1诊断

任务 `task_20261027_092122_69770722761`，完成exit0，4个PTO臂各8窗，
不同时开启dep-gen。72核、单次replay边界、无丢失窗口均检查通过。

| 实现/截断点 | L1完整跨度中位数 μs |
| --- | ---: |
| 我方完整Direct | 1502.14 |
| nalinaly c346 完整SPMD | 928.46 |
| nalinaly c346 到TopK＋plan | 429.35 |
| nalinaly c346 到heads | 706.18 |

截断版保持前段DSL数学、信号/FFTS配对，并经实际生成代码审查；只有阶段之后的业务删除。
它们独立编译，会影响寄存器、UB生命周期和指令布局，因此706.18−429.35与
928.46−706.18只是累计诊断差值，不能视为原完整SPMD内测得的纯attention/输出耗时。
L1同样包含搬运、Scalar及同步，不能直接拆成可相加的纯计算/搬运/等待。

单独dep任务 `task_20261027_093646_15063346873` 于10:11:34～10:14:20运行166秒，exit0。
新SDK CPU/JIT叶子二进制SHA、L1实际构建func_id与名字、两窗依赖图稳定性及全部
L1 task token覆盖均通过。依赖生成与L1计时分开。

| 对齐里程碑 | 我方实测中位μs | nalinaly c346累计诊断参考μs | 参考差μs |
| --- | ---: | ---: | ---: |
| 入口→TopK＋plan | 722.41 | 429.35 | 293.06 |
| plan→heads | 441.87 | 276.83 | 165.04 |
| heads→结束 | 337.85 | 222.28 | 115.57 |

我方每窗取相邻里程碑后再取中位数；各中位数相加未必等于全程中位数。
nalinaly两段来自独立累计cut中位数相减，仅作优先级诊断参考，不是同一原kernel内部时钟，
不能把差值当成精确可节省时间。此前预设attention是最大差距的方向需要修正，前半段优先。

我方内部实际inclusive跨度：HC/Norm前缀Vector184.85μs；Q_B矩阵乘86.86μs；
Indexer最后打分MIX80.89μs；Q_A单核预清零74.30μs；attention AIV429.62μs；
WO-A跨核跨度163.04μs而最长核65.34μs。这些跨度包含搬运、Scalar、等待和跨核调度，
互相重叠，不能相加当纯计算。Q_A预清零是由新依赖映射定位的具体目标：QR_OK1且非atomic，
后继矩阵乘覆盖整个ceil16输出；需先证明删除不会残留旧GM值，再做独立性能筛选。

## nalinaly最新公开更新

本轮计时结束后再次检查GitHub，分支最新为 `1685da6914ae6196537d7f1a66e84e09a9c48ae5`。
相对c346共4提交，主要为 WKV、Indexer Q_B、Indexer权重投影直读 Native NZ。
本报告上方计时仍是c346，不能把它改标为1685。

保留GitHub compare、变更文件与叠加旧公开相关源码得到的242文件快照；不是完整git历史clone。
其公开NZ验收只对各自旧ND路径，没有Native数值重测。代码层面没有改外K循环、量化与
舍入表达式；布局改变仍可能改变后端实际矩阵乘拆分，需要我方独立验收。

## 我方WKV NZ候选

独立候选，只修改 WKV root/prefix/cube/inline 的物理layout为NZ，并绑定原Native
权重storage；format29、指针、逻辑CPU权重值均有检查。矩阵方向我方原本就是
[512,4096]和b_trans=True，因此不照搬其旧转置ND改方向部分。

- 保留我方Native order_packet的16步K256、same_order 64×32及narrow fallback。
- 保留所有数学与BF16舍入；Sparse/QK/PV源文件逐字节相同。
- 两根CPU资格均64参数、Artifact.load通过、NPU未初始化。
- 实际生成代码WKV primitive仍为K256；NZ物理视图和子片地址检查通过。
- 最大L1/L0A/L0B/L0C/UB为49152/34816/18432/8192/141056 bytes，未增加。
- 18个MIX同步保持；仅Cube前缀leaf二进制改变，其余62个leaf SHA一致。
- 构造时仍生成未使用ND副本，当前仅测试运行期读取，不宣称已减少加载转换或显存。

第一版缺binding.json，在无设备预检失败，未初始化NPU；第二版补齐已验来源文件，
不修改数学或门槛。无设备失败、修复与原始证据全部保留。

## 预先约定的采用门槛

各独立候选均采用以下门槛（WKV NZ与HC-ready已按此否决，详见后文）：同卡交替12×100，
event中位数至少改善0.5%、≥10/12配对领先且L1更低才做Native A/B八边界和ABA。
通过后反向重复，再128K/B16和混合T126，最后整模型8K/B40→128K/B24的192 token。
发现首差先停在当前配置定位。每个设备任务外780秒、内750秒，不改SDK或生成C++。

当前没有合入新的运行期优化，原精度和整模型token验收基准保留。

## 第二个独立候选：HC-ready任务切分

只删除scope不足以提前启动QA，因为Norm仍归属于原完整MIX producer。
新候选在第7个handoff拆成HC/Norm就绪任务A与WKV/Indexer投影任务B；B只读Norm。
原HC私有scope保留，B不持有wide/packed_flat/mixed_tail。未组合NZ、未改数学helper。
两臂64参CPU/load/NPUfalse通过；实际生成QA没有新增B依赖，KV消费者正确等待B。
全部22组alloc的scope depth保持，原18个同步分成14+4；其余61个共同leaf二进制SHA相同。
T256逐depth全分支保守上界42.93/92.77/195.60MiB均低于256MiB；不是实际峰值或小ring资格。
B仍可能占满24组核心，因此DAG允许并行不等于实际重叠，性能没有实测收益不得合入。

## 本轮实际QA来源补充

单层比较实际采用binding指定的3399文件冻结snapshot（formal_math_commit为ad06a423…）加Direct根/前缀/后缀实验绑定，各文件SHA封存。冻结QA126只含126/168/generic分派；当前正式11b380树另有95 packet路由，因此不能将本轮S6/T192结果写成当前正式所有形状/路由已经重新验收。当前已测T192和后续T96不走95分支；已有八边界测量结论在明确形状内保持有效。

No-seed v1最初从当前本地formal派生，CPU生成审查发现引入95分支，设备前否决。v2改由实际冻结两文件派生，SHA分别ea30b2de…与773f04fd…匹配binding，除了两处seed所有路线AST相同；正式迁移时保留正式95处理，需其独立尾覆盖/数值资格及完整模型复测，不能直接用旧generic的95对拍代替。

## WKV NZ反向重复：未采用

| 8K/B32独立轮次 | 控制event中位μs | 候选event中位μs | 耗时降低 | 配对领先 | L1控制→候选μs | Native A/B8 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 首轮 | 1614.663 | 1584.650 | 1.8587% | 12/12 | 1524.16→1483.96 | 逐bit＋ABA通过 |
| 反向捕获重复 | 1586.924 | 1579.729 | 0.4534% | 12/12 | 1503.90→1481.66 | 性能门槛未过，未执行新精度扩测 |

按既定至少0.5%event门槛，不能称稳定收益；不合入、不扩测128K。首轮精度通过事实保留，但不掩盖反向重复性能失败。两个任务均exit0并释放设备。

### HC-ready设备结果：否决

首轮单卡8K/B32控制1601.104→候选1643.027μs，回退约2.62%；L11508.88→1552.22μs也更慢。
任务task_20261027_103626_334065410437完成exit0，性能门槛失败，未执行Native精度扩测。
因此逻辑DAG提前就绪没有转化为稳定收益，不合入，不因生成图更并行就继续扩大配置。

## Q_A硬删除v2实验：两轮小幅收益，不代表正式guard版本通过

本候选只删非atomic、QR_OK1路线的预清零，保留首次MMA初始化、K256顺序、dense/tail形状与舍入；未组合NZ或HC切分。冻结根生成的3个seed删除，其余60个共同leaf记录SHA一致。生成代码证明ceil16整输出均被非原子写覆盖，Normalizer仍依赖QA生产者。

| 8K/B32独立轮次 | 控制event中位μs | 候选event中位μs | 耗时降低 | 配对领先 | L1控制→候选μs | Native A/B8 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 首轮 | 1600.205 | 1579.849 | 1.2720% | 12/12 | 1503.28→1485.80 | 逐bit＋ABA/guard通过 |
| 反向捕获重复 | 1598.266 | 1588.261 | 0.6260% | 12/12 | 1507.96→1487.56 | 逐bit＋ABA/guard通过 |

两轮均通过预设性能门槛，但收益仅10～20μs，不能声称已填平整体差距。后续128K/B16单层、QA尾部poison与正式路由覆盖结果见下文。硬删除v2不能代替保留95与atomic语义后的guarded v3性能资格。

## 后续单层资格

- v2固定冻结来源128K/B16：1341.172→1318.705μs，耗时降低1.6751%，12/12配对领先；L11240.70→1213.50μs，Native A/B8＋ABA通过。
- 保留正式T95分派和atomic保护后的v3，T95完整根：1146.909→1131.481μs，降低1.3452%，Native A/B8＋ABA通过。
- v3混合1～6行重复6次、36请求T126：1342.618→1328.056μs，降低1.0846%，Native A/B8＋ABA通过；没有padding到S6。
- QA真实第4层权重独立Out单元覆盖T4/17/21/95/126/168/192/240/256；NaN、Inf、有限数预填充，27个A/B/A场景的整个ceil16输出、guard及逐phase只读检查通过。单元gold是原已验QA，不是Native整层或token。
- guarded v3生成代码保留65个相同leaf记录SHA；4个seed在if0中不launch，不能沿用硬删除v2‘删3个leaf/60共同leaf’描述正式迁移。

## Indexer Q_B Native NZ：未采用

学习1685最新Native NZ读取方案，单因素保持原K、INT32乘加、反量化、BF16/FP16及Hadamard边界，只有root→indexer→query的weight layout与Native同指针storage绑定改变。两臂63leaf仅idx_qr_proj_matmul改变，其余62记录SHA一致；Native INT8逻辑值逐bit与ND相同，format29、指针硬验证。

8K/B32首轮1600.485→1594.347μs，仅降低0.3835%，12/12配对领先，L11510.60→1495.92μs；未达到0.5%event门槛，accuracy_executed=false，不合入、不扩测。不是精度失败，也不能称Native精度通过。

## 最终正式guard候选的反向复测：未采用

硬删除v2没有保留atomic支持，不能直接合入正式实现。guarded v3保留原atomic分支和95路由后，须以其自身结果作采用决定。

| 8K/B32 guarded v3 | 控制event中位μs | 候选event中位μs | 耗时降低 | L1控制→候选μs | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 首轮 | 1588.585 | 1576.473 | 0.7624% | 1500.74→1481.52 | 性能及Native A/B8、ABA通过 |
| 反向捕获独立重复 | 1585.842 | 1583.267 | 0.1624% | 1496.24→1492.56 | 未达0.5%门槛，未扩测精度 |

最终guard版本收益不稳定，因此不合入。不能用硬删除v2的两轮通过替代guarded v3的失败。已取消尚未获卡的16卡8K/B40任务112507；它没有开始设备执行，没有本候选整模型token结论，也没有提交128K整模型。

## 整模型与下一步状态

当前正式运行实现及旧SDK整模型精度基准保留，本轮没有采用新的运行期优化。查询岛是独立后续原型：借鉴nalinaly查询阶段融合及4/20分核，保留我方combined-scale与BF16边界；目前只做CPU生成资格，不计入已取得性能优化或精度验收。具体设计、风险与否决条件见QUERY_ISLAND_NEXT_PLAN.md。

2026-10-27 12:18再次从GitHub确认最新仍为1685da69；本轮完整三方实测仍为c346，不改标。后续先验证生成代码的缓存策略作用域、GM视图、Tensor metadata、FFTS配对和私有缓冲生命周期，通过后才提交单卡性能筛选。

## 固定范围与复现入口

同工具链完整三方计时与两阶段cut固定nalinaly c346公开debug57 Tensor入口，而非其普通55 Tensor入口；最新1685已同步与审查，但上方三方数字不能改标为1685。GitHub参考：[c346](https://github.com/nalinaly/vllm-ascend/commit/c346573ab3313fcc71cd58756acd129f34755439)、[1685](https://github.com/nalinaly/vllm-ascend/commit/1685da6914ae6196537d7f1a66e84e09a9c48ae5)。

除明确whole项外，均真实单层权重＋合成hidden/history。新SDK本轮未完成整模型token资格；旧正式版本此前token一致不自动证明候选或新SDKtoken一致。所有本轮任务实际耗时见TASKS/原始日志，外780秒、内750秒；排队不等于占卡运行。Native2没有重跑。没有改SDK/runtime/codegen/生成C++，没有安装或覆盖Torch/vLLM/Simpler。

源码身份、受测helper、实际CPU生成物、原始事件及独立L1/dep、负面结果和复现步骤见[本轮证据目录](evidence_tnd_precision/stage_gap_qa_seed_20261027/README.md)。最终guarded QA候选反向性能未达门槛，未采用；排队whole已取消，本轮没有新的整模型token结果。
