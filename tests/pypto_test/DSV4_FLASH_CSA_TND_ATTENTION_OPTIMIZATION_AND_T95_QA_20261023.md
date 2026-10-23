# TND attention优化与T95 QA精度修正（2026-10-23）

## 范围与当前状态

本报告接续[模块差距与H3报告](DSV4_FLASH_CSA_TND_NALINALY_MODULE_GAP_AND_HC_OVERLAP_20261023.md)。原H3提交`a74eb99b9b30e27df845d855aff71f0b6f4146d6`保留，分支仍为`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。nalinaly公开分支再次查询仍为`cafcadec96d48790608e5b55d456150fcf00e794`。

继续保持纯DSL、完整HC-pre到HC-post单入口、真实TND，不补齐每请求S6，不改SDK/runtime或生成C++。下面区分性能候选和正式精度修正，单层对拍不代替整模型token验收。

## 已有共同基线与优化优先级

原同任务8K/B32/T192完整FULL graph的event均值：Native1257.224、我方原组合版1816.296、nalinaly单SPMD兼容版1212.730 μs。nalinaly在固定SDK中仅去掉不支持的等待退避参数，不能当作其原工具链速度。其最终输出relative L2为0.333%，不能把它视为Native精度基准。

独立同输入nalinaly多task伴随诊断显示，TopK就绪→heads就绪我方664.47 μs、nalinaly280.29 μs，是最大可见差距；入口→TopK为708.06/565.50、heads→输出346.42/218.97 μs。每kernel跨度可能重叠，不能把这些统计与单SPMD时间混加。

因此优先借鉴索引计划和KV提前搬运。保留Native物理pair排序、K512 softmax更新、sink初态、actualK求和树、概率round、逆RoPE前BF16和全8192列O-proj共同量化；不照搬已知会改变数值的边界。

## 三项组合的性能

三项为H3同步精简、Scalar索引共享读取、KV双缓冲预取。12个轮转event窗口×100 replay，另行采集L1，计时不打开dep-gen。以下均为同任务成对结果，单位μs，中位数。

| 配置/轮次 | 当前H3 | 三项组合 | 相对H3降低 | 精度资格 |
| --- | ---: | ---: | ---: | --- |
| 8K/B32，反向捕获 | 1783.977 | 1718.919 | 3.65% | Native A/B八边界及ABA逐bit一致 |
| 8K/B32，正式包候选 | 1798.933 | 1704.182 | 5.27% | Native A/B八边界及ABA逐bit一致 |
| 128K/B16 | 1462.985 | 1439.554 | 1.60% | 对二项组合增益仅0.185%，未过四臂增量门槛，未执行精度，不能记为通过 |

两个8K实验都12/12窗领先且L1更低。旧前缀+预取臂与新H3组合不是完整2×2设计，不据此分解交互收益。更晚的1704与早期nalinaly1213不在同一任务，不能拼成受控加速比。

短T21预取逐bit通过但慢1.52%，候选对T<96保持原路由。SWA128独立候选约快0.76%且逐bit通过，未叠加；SWA L1驻留版128K回退、UB统计版慢1.48%，均未通用合入。

## 新覆盖的T95首差及修正

T95不是每请求补成S6：27个请求长度为`[1,2,3,4,5,6]*4+[1,4,6]`，真实T95、history8192、真实第4层权重，FULL graph。原H3与三项候选在该档调用同一原根，输出相同，却都与Native存在首差。因此不能归因于未启用的预取。

| 证据阶段 | 结果 |
| --- | --- |
| 原完整H3 | x_out 642元素不同、relative L2 5.6043e-5；TopK46位置不同，排序后集合相同；六类cache/state精确 |
| 有效的五阶段观察 | 两边原八个外边界观察前后不变；row86/87的normed精确，QA row86/col484差一个BF16值，随后QR scale和Q出现差异 |
| 完整M95 QA单元 | Native独立F.linear以同NZ29真实权重重现FULL全部95行QA且重复一致；原helper实际14个、随机输入18/17个BF16差异 |
| K512 packet QA候选 | 实际、随机A、实际、随机B、实际的全95×1024输出逐bit一致；raw graph污染后回放、重复、守卫和权重只读通过 |
| 仅增加T95 packet路由的完整观察副本 | 全八外边界和五个观察点逐bit对齐Native |
| 无debug导出的正式T95路由回归 | H3修正版及三项候选的Native A/B八边界、恢复A全部逐bit一致，只读norm/HC/order包每轮保持不变 |
| 相邻T96混长路由回归 | 长度`[1,2,3,4,5,6]*4+[2,4,6]`，H3与预取候选Native A/B八边界及ABA全部逐bit一致 |

正式精度修正只在`qkv_precision126.py`增加T95走已有K512 packet helper，原126/168和其它T路由不变；本地正式文件计算AST与独立无debug受测副本相同。原H3可按父提交恢复。

该证据证明本次输入、随机QA样本和相邻路由回归，不是所有动态T/所有输入的数学证明，也不代替21层整模型token验收。观察副本有两个继承字段仍写`accuracy_executed=false`和旧failure描述；原始JSON未事后改写，实际门禁以`matches_original_native`八项和五个stage逐bit结果为准；最终无debug回归的`accuracy_executed=true`。

## Vector索引计划：首差与设备单元定位

Vector仅替换内部压缩索引检查，保留输入TopK及整数地址/物理pair排序。CPU616组、157696对数学等价通过，却不能保证平台ISA实现支持。

首个完整Vector v5任务先筛性能后做精度，只有Vector最终输出失败：relative L2 5.34014%、2273530元素不同；TopK与六类cache精确。同组H3和三项组合的A/B八边界及ABA都精确。该任务event中位数H3 1779.063、三项组合1695.259、Vector1407.477 μs，**1407不算有效性能结果**，没有合入。

只读核实受测PTO-ISA `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`的A2/A3 `TCmps.hpp`：INT32分支始终调用EQ，原Vector生成物却使用INT32 GE/LT/GT/LE。修正保留所有候选/地址/减法INT32，只将无溢出整数差转换FP32比较符号；2097152跨度阈值附近可精确表示，未将地址转成FP32。

六Tensor planner-only设备单元结果：

| 单元输入 | 原Vector v5不同项 | 修正后不同项 |
| --- | ---: | ---: |
| 失败任务真实TopK＋明确重建页表 | 98304 | 0 |
| INT32边界/负槽/回退/跨度临界合成输入 | 5597 | 0 |

修正版两组eager和三次带输出污染的raw graph回放全部精确，输入只读和外侧守卫通过。页表明确为重建，未声称保存了旧FULL实际metadata。生成物10个比较输入均FP32、UB峰值20544字节、同步配对审查通过。不改SDK或生成代码。

完整Vector修正版已完成两次8K/B32单层验收（中位数μs）：

| 轮次 | H3 | 三项组合 | Vector修正 | 比H3降低 | 比三项组合降低 | 精度 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 首轮 | 1805.675 | 1705.260 | 1658.162 | 8.17% | 2.76% | Native A/B八边界及ABA逐bit一致 |
| 反向捕获复测 | 1782.730 | 1705.502 | 1659.783 | 6.90% | 2.68% | Native A/B八边界及ABA逐bit一致 |
| 128K/B16 | 1470.121 | 1425.480 | 1398.735 | 4.86% | 1.88% | Native A/B八边界及ABA逐bit一致 |

三组对两个控制均12/12窗领先，独立L1也更低。整模型token资格另见下节，不能沿用H3历史结果。

## 复现与证据位置

工具链、模型和bank沿用[既有证据说明](evidence_tnd_precision/nal_module_gap_20261023/README.md)。PyPTO2dfa7f120、Simpler57e499238、PTOAS0.66、CANN9.2、Torch2.10/Torch-NPU2.10.0.post2、vLLM0.25.1，确定性level1/HCCL开启，Native SK0。

本轮助手在实验根`nal-module-gap-20261023-codex`下：`combo3-v2`、`tuned-formal`、`t95-hook-v2`、`t95-qa-unit-v2`、`t95-hook-fix-v3`、`tuned-formal-qa95-v2`、`vector-pair-v5`、`vector-plan-unit-v2`、最终封存`vector-pair-v7`。旧失败目录保留，v6的CPU等价JSON生成后manifest失配，因此未提交NPU；v7在最终结果生成后重新封存manifest并重新编译全部三根，不借旧收据。

复现先使用独立目录和新label，按manifest校验源码，再无设备编译；改变路径或源码后必须重新封存/编译，禁止拿旧收据替代。所有NPU工作通过task-submit，常规单卡外层780秒/内部750秒；planner-only外层300秒。首差立即停止扩大，不使用`--wait`。

```bash
gap=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/nal-module-gap-20261023-codex
# 精度首差：先运行原tuned-formal T95，保留first-difference.pt；观察脚本显式绑定该文件。
# QA单元绑定有效观察产物，必须先有Native FULL的native-stages.pt/pto-stages.pt。
bash "$gap/t95-qa-unit-v2/compile.sh"
task-submit --max-time 780 --device auto --device-num 1   "bash $gap/t95-qa-unit-v2/run.sh repro-qa95"
# 无debug正式路由回归。95通过之后才96；两者都检查Native A/B与恢复A。
task-submit --max-time 780 --device auto --device-num 1   "bash $gap/tuned-formal-qa95-v2/run.sh repro-full95 27 8192 1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,4,6 --qualify-boundary"
```

各`result.json`保留所有样本与实际root/schema；`TASKS.json`记录任务关系。后续完整Vector性能、真实8K/B40及128K/B24的192 token应继续追加独立结果，不能沿用H3历史整模型结果为新候选背书。

### 整模型助手类型门禁（10月24日补记）

首个8K/B40任务在开始生成token前退出，最早异常来自rank8：`mixed_perf.py`仍要求旧`combined.service`，实际预期的新`tuned.service`被严格门禁拒绝。不是已观测到的token精度差异；其他rank为连带关闭，128K未提交。

v3只把该类型检查更新为预期tuned，继续拒绝旧combined；实际guard AST的21层tuned CSA＋20层Native HCA正例和错误variant反例均通过。数值包与v2字节相同，whole/父manifest重新封存、新CPU根及Native金值预检通过后重试8K。此问题及失败记录保留，不将失败任务当精度通过。

## 正式独立候选与真实整模型（10月24日）

正式候选增加显式`tnd_precision_tuned`，不切换默认或旧精度alias。T<96复用原CombinedCapture，T>=96使用Vector检查、Scalar安全回退及双GM缓冲预取。源码保持纯DSL、64Tensor ABI、一个顶层入口；注册和workspace沿用原生命周期。编译物/运行源与已验收Vector v7计算AST一致；共享QA95修复与正式混长回归的计算AST一致。新增CPU路由/alias/ABI测试合计27项通过。

| 真实整模型配置 | 192 token与Native | 入场事件/抢占 | forward均值 ms | 周期均值 ms |
| --- | --- | --- | ---: | ---: |
| 8K/B40 | 16rank，warm＋measured共245760位置全部一致 | 全部一致/0 | 100.371 | 115.963 |
| 128K/B24 | 16rank，warm＋measured共147456位置全部一致 | 全部一致/0 | 86.665 | 99.122 |

8K每卡实际40条活跃/T240；128K提交24条、实际21活跃/真实T126/图桶144。先8K通过才提交128K。计时对齐20个forward及19个周期，每步取16rank最大再统计，不平均rank。周期包含草稿、采样、调度与CPU metadata观察；forward仍含未替换HCA/FFN。没有本轮新成对Native或冻结版whole计时，不将历史数据之差解释为受控整模型加速。

候选8K/128K任务模型窗口分别8分14秒/7分56秒，均正常退出0，所有16卡已释放。验收还检查21个CSA层小T6及大桶的实际root捕获、FULL replay、计时期间Native分发计数不变、测量期间没有JIT重建。不是只看最终token或环境变量。

最终证据见[本轮归档](evidence_tnd_precision/attention_qa95_20261023/README.md)。后续剩余重点仍是attention计算/搬运等待及O-proj尾部；应将当前v7的独立L1映射到实际kernel后重新排序，不能直接把优化前664 μs阶段沿用成优化后耗时。继续保持Native数值边界，不以nalinaly的误差换速度。

## 优化后DFX拆分与下一重点

最终v7的独立L1窗口与新采集的两次H3、两次Vector依赖图匹配，每臂9个计时窗口的全部task/core记录均有映射、重复依赖图一致。DFX记录前后Native八边界、只读包和外侧守卫通过。计时阶段关闭dep-gen，依赖阶段不计时。

CPU/JIT Core二进制SHA全部相同，编排二进制SHA不同：生成局部变量编号因编译上下文变化。首次诊断严格binary门禁失败并退出，未用于分析；第二次保留两侧SHA，要求编排全文在**仅局部声明的临时identifier一一改名**后相同，同时字符串/字符/raw-string/预处理内容、常量、运算、调用顺序、kernel配置及compiled metadata原样一致。门禁有正反例并经独立复核；这支持结构映射，不宣称编排二进制逐字节一致。

单位μs，中位数；单函数inclusive span包含Scalar、搬运与同步等待，各函数可能重叠，不应相加。

| 指标 | H3 | 新Vector精度候选 | 观察 |
| --- | ---: | ---: | --- |
| checked indices跨度 | 92.72 | 20.96 | Vector检查减少约71.76 μs |
| QK/PV AIV跨度 | 462.82 | 411.52 | 双缓冲及更早启动减少约51.30 μs |
| WO-A跨度 | 163.98 | 163.90 | 基本不变 |
| WO-B64跨度 | 88.96 | 87.36 | 基本不变 |
| 完整L1跨度 | 1710.02 | 1576.42 | 独立DFX，不混入event计时 |

非重叠里程碑逐窗计算后取中位数：

| 区间 | H3 | 新Vector | nalinaly多task历史参考 |
| --- | ---: | ---: | ---: |
| 入口→TopK就绪 | 691.48 | 691.22 | 565.50 |
| TopK就绪→heads就绪 | 667.22 | 539.28 | 280.29 |
| heads就绪→结束 | 352.18 | 349.16 | 218.97 |

每列是区间样本中位数，不能要求三个中位数之和等于完整中位数。nalinaly列来自同输入、同SDK的前序多task伴随诊断，不是本轮同任务重测，也不是单SPMD的内部拆分；不能据此精确分摊其1213 μs单SPMD event结果。

优先级仍然明确：attention段的剩余参考差距约259 μs，大于前缀约126 μs及尾部约130 μs。下一步先在现有K512更新和原生舍入边界内，拆QK/PV Vector更新、MTE2预取未覆盖等待及同步气泡；对照nalinaly的流水安排，单因素测性能再做精度验收。WO-A的搬运/流水为第二梯队，不能直接移植其改变全8192量化边界的实现。此次先冻结已取得稳定收益且通过两档整模型token验收的版本。
