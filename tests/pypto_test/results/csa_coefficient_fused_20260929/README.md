# 在Score MIX内部生成head系数

基线91276630已按有效query组删除系数空worker。本候选继续参考最新
`ops-transformer/attention/quant_lightning_indexer_v2/op_kernel/arch22/`
中`ProcessVec0`和`ProcessBaseBlock`：在QLI内部生成系数，通过MTE3→MTE2核间同步交给Cube。
参考源码不代表本机Native二进制已经重新构建成该版本。

## 修改与同步边界

删除独立`indexer_head_coefficients`任务，Score直接依赖QH量化、weights和cache写入。
每个Score worker拥有独立GM系数槽；AIV0按原顺序将qScale、weight分别转FP16再相乘，
保留原有对角块和补零。两个AIV都发送mode2事件2，AIC等待后加载系数。
原Score双缓冲事件0/1、cache布局、叶片分配、early和sync_start策略均保留。

单槽复用依赖既有Score协议：AIV0必须消费当前leaf最后一块Score READY后才能进入下一leaf，
该READY由Cube最后一次FIX输出产生，此时本leaf对系数的读取和使用均已完成。
双方按相同item顺序和valid条件进入leaf；不以删除必要依赖换取性能。
该策略可能为同一query在不同leaf重复计算系数，收益必须覆盖新增核时，不能仅按任务数判断。
Native采用自己的系数布局和双缓冲，本候选吸收融合位置，不宣称两者全部实现相同。

当前长B16的16个query组各分6个leaf，系数生成从16次变为96次，
每个Score worker负责4个leaf；短B24为24组单leaf，生成次数仍为24。
这是源码工作量，不能将其直接换算成设备耗时。
最新本地pypto-lib 2164563同样在Score内部生成head系数，但使用Vector加权规约，
不需要本候选交给第二次Cube的FP16对角矩阵，不能直接套用其核时。

## 验证范围

两侧整包及公共依赖冻结在私有目录，使用`pkg:dsv4_csa_coefficient_fused_91276630`。
生产/测试根依赖图已解析；新增候选完整CPU编译/load后才提交自动单卡任务。
128K/B16、8K/B24交替A/B顺序，各5次预热/20次无profiler真实编译图计时，独立四窗口DFX。
复用编译图/eager、八类完整状态、Top-K结构和保护区检查，不重复整模型或七档矩阵。
两侧固定CANN9.2/mode2/atomic0/det0、layer4正式权重和合成历史，EPLB关闭，
ring为[256,128,256,32]MiB、task_window4096；长短8:2判断，同时单列P95/max。

task_20260929_044158_1383300700完成exit=0，全部16个DFX窗口通过官方时钟域、行数和block核对。
独立系数worker长16→0、短24→0；两档八类跨版本完整状态零容差、编译图/eager和保护区通过。
[补丁](candidate.patch)、[冻结目录](source.txt)、[预编译](compile.py)、[单卡入口](run.sh)。

## 结论：不合入

128K/B16正式CSA均值1037.773→1066.543μs（+2.772%），
8K/B24为962.084→969.567μs（+0.778%），长短8:2为+2.373%；两档P95和max均升高。
长档Score首次start提前19.220μs，但AIC/AIV均值增加36.173/36.132μs，最终结束延后约17.840μs。
候选把系数计算和交接等待纳入Score，核时范围扩大，不将这36μs全算成纯算术成本。
长档Worker总跨度1001.040→1022.535μs，Sparse及后续任务也延后，与正式计时方向一致。

短档Score首次start提前29.375μs，独立DFX的Worker总跨度921.575→904.965μs，
与正式CSA的+0.778%方向不同。两者不是同次采样，不能用DFX给正式样本新增的7.483μs归因，
也不能用更短的泳道覆盖正式计时。长档已有明确回退，不为此失败候选扩大测试矩阵。

保留生产91276630和候选证据；精度版未改，未声称整模型token/DSpark或EP16长尾验收。
下一步先参考Native ProcessVec0的成块加载/乘法，减少逐query重复MTE2和Vector指令，
在独立系数任务上验证核内收益；之后再考虑融合及每worker补零缓冲复用。
[正式结果](RESULTS.md)、[Worker分项](TASKS.md)、[精简验证记录](summary.json)、[完整原始证据](evidence.json)。

固定第4次重放window_3供查看，不挑最快窗口：

| 档位 | 基线泳道 | 融合候选泳道 |
| --- | --- | --- |
| 128K/B16 | [JSON](h131072_b16/swimlane/baseline/dfx/window_3/merged_swimlane.json) | [JSON](h131072_b16/swimlane/candidate/dfx/window_3/merged_swimlane.json) |
| 8K/B24 | [JSON](h8192_b24/swimlane/baseline/dfx/window_3/merged_swimlane.json) | [JSON](h8192_b24/swimlane/candidate/dfx/window_3/merged_swimlane.json) |
