# FP16系数对角块在UB构造后一次发布

基线92c747c6已保留按query组批量加载/转换/乘法。本轮仅减少系数对角块的GM写回，
保留独立系数任务，不叠加系数融合或dummy直接依赖候选。

## 修改及源码依据

参考ops-transformer28f40354 QLI V2 ProcessVec0先在UB布局再CopyOut的策略。
原实现先写完整零矩阵，再逐query补写有效对角行；候选在UB内补入这些行后一次写回。
FP32输入分别转FP16再乘、所有16个补齐矩阵行、worker数、stride48、依赖与early/sync均不变。
Native的Brcb布局与PTO对角块不同；pypto-lib2164563的Vector head规约也没有此矩阵，
只借鉴减少发布次数，不能直接套用两者的核时。

首次尝试直接向宽矩阵的局部列assemble，在四个特化上均未通过PTOAS：
`pto.tmov expects A2/A3 non-mat tmov to use matching src/dst shapes`。
当前解法在算子内把UB矩阵表示成[16×group,64]，每行宽度与源[1,64]一致，
将第lane行系数写到lane×(group+1)行，再reshape为[16,group×64]。
线性偏移仍为lane×(group+1)×64+c，等同原矩阵的[lane,lane×64+c]。
两种行宽均满足32B对齐，reshape只改UB视图，不做GM重排；完整补零仍保留。
未修改PyPTO、PTOAS或PTO-ISA，也没有使用逐元素scalar scatter绕行。

当前两根解析和完整CPU编译/load通过，四个特化覆盖S6和双query：

| 特化 | TLOAD/TCVT/TMUL | TEXTRACT | TMOV | TSTORE |
| --- | ---: | ---: | ---: | ---: |
| S6 | 2/2/1不变 | 6不变 | 0→6 | 7→1 |
| 双query | 2/2/1不变 | 2不变 | 0→2 | 3→1 |

新增UB搬运必须计入代价，不能仅按GM写回次数宣布性能收益。
初次编译失败保留在initial_assemble.patch和initial_assemble_errors.txt；
parse_candidate.log属于初次表达式，当前表达式的解析/编译证据以compile_candidate.json为准。

## 验证范围

私有冻结整包`pkg:dsv4_csa_coefficient_publish_92c747c6`及公共/测试源码，
auto单卡任务task_20260929_053528_168129524260。
128K/B16和8K/B24，同卡CANN9.2/mode2/atomic0/det0，正式layer4权重与独立合成历史，
EPLB关闭，ring=[256,128,256,32]MiB/task_window4096。
长档基线→候选、短档候选→基线，各5预热/20次无profiler实际编译计时，
另采四个level-4 DFX窗口；复用八类完整状态、图/eager、Top-K结构及保护区检查。
两代表档均为S6，双query仅编译通过，阶段出口仍须覆盖8K/B32等受影响形状。

先看相同worker/有效query下的系数核时，再按长短8:2判断CSA，单列P95/max。
DFX使用官方时钟域join和block核对；独立窗口不能与正式CSA相减推导开销。
不扩跑整模型，不以单卡状态检查宣称token/DSpark或EP16长尾已验证。
任务完成exit=0，两档八类完整状态跨版本零容差通过，图/eager、Top-K结构、
metadata与保护区通过；16个DFX窗口完成官方时钟域、行数和每任务block核对。

[候选](candidate.patch)、[生成调用点](lowering.json)、[冻结来源](source.txt)、
[编译入口](compile.py)、[设备入口](run.sh)、[收集入口](collect.py)。

## 结论：按核内收益保留，CSA与尾部代价单列

| 档位 | 系数核内μs | 核内变化 | CSAμs | CSA变化 | P95μs | maxμs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 3.415→2.534 | −25.798% | 1045.851→1040.177 | −0.543% | 1059.040→1060.940 | 1059.440→1061.740 |
| 8K/B24 | 3.394→2.156 | −36.454% | 953.475→967.333 | +1.453% | 973.380→987.180 | 980.380→995.540 |

系数四窗口均值长档2.270/3.711/3.896/3.781→2.757/2.735/1.786/2.856μs，
短档4.573/2.602/3.710/2.689→1.968/3.117/1.658/1.883μs。
同样16/24个有效worker、同样query与任务范围，核时包含DMA和等待。
每档四窗中有三窗候选更短，生成码同时证明GM发布次数减少；不声称每窗都快。
依据用户“incore有收益即保留”的规则采用性能版，不以CSA仅−0.143%的8:2结果作为主要依据。

短档CSA+1.453%，P95长档+0.179%、短档+1.418%均原样记录。
两档两侧各20次均无>P50×1.05样本；候选P95/P50长1.0194、短1.0160，
不能据此声称历史间歇拖尾或EP16稳定性问题解决。
Score首次start长324.020→321.345μs、短378.410→383.035μs；
Score AIC/AIV长253.889/259.732→261.743/267.404、短33.336/48.569→32.591/46.513μs。
Score源码未改，变化包含重叠和等待，不能归结为新的Score算术优化或单独认定其因果。
独立DFX的Worker跨度长1008.605→1016.870、短953.415→948.735μs，与正式CSA变化方向相反，
说明不能用独立profile替代正式计时或给其中未捕获的样本归因。

生产仅移入实测系数函数，两根依赖图解析通过；精度版和调度依赖未改。
双query仅编译验证，8K/B32及其他受影响形状在阶段出口补齐；最新完整七档仍为c93ec723，
不与此次两档拼接。下一阶段继续看AIV分工、独立merge与交接，当前朴素系数融合失败结论不变。
[结果](RESULTS.md)、[Worker分项](TASKS.md)、[精简样本与检查](summary.json)、[完整证据](evidence.json)。

固定第4次重放window_3，不挑最快窗口：

| 档位 | 基线泳道 | 一次发布候选泳道 |
| --- | --- | --- |
| 128K/B16 | [JSON](h131072_b16/swimlane/baseline/dfx/window_3/merged_swimlane.json) | [JSON](h131072_b16/swimlane/candidate/dfx/window_3/merged_swimlane.json) |
| 8K/B24 | [JSON](h8192_b24/swimlane/baseline/dfx/window_3/merged_swimlane.json) | [JSON](h8192_b24/swimlane/candidate/dfx/window_3/merged_swimlane.json) |
