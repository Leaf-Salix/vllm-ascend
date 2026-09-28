# 参考Native按query组批量准备FP16系数

基线0153a8a9，生产算子与91276630相同。此前把系数融合进Score导致长短档CSA均回退，
本轮保留独立任务，只优化其加载和逐元素运算；不叠加融合或dummy候选。

## 修改及与上游的关系

参考ops-transformer28f40354的QLI V2 `ProcessVec0`，一次加载完整S1组的weight与qScale，
然后批量相乘。PTO继续将两个FP32输入分别转FP16再乘，保留对角块、补零、GM写回、
min(48,query组数)的worker数、stride48及任务依赖。新二维scale描述只是GM视图，没有device重排。

四个生成特化确认：S6的TLOAD/TCVT/TMUL调用点12/12/6→2/2/1，新增6个TEXTRACT；
双query为4/4/2→2/2/1，新增2个TEXTRACT；TSTORE分别维持7个和3个。
提取操作用于发布原对角行，不能只数减少的读取而忽略它。
Native已消费FP16入参并用Brcb布局交给Cube，当前PTO仍需FP32→FP16及对角块转换。
最新pypto-lib 2164563的Vector head规约没有这个对角块任务，不能直接比较同名核时。

## 验证范围

私有整包`pkg:dsv4_csa_coefficient_group_0153a8a9`冻结算子、公共依赖和测试入口。
两根_get_dep_graph、完整候选CPU编译/load通过；不重复编译未改变的生产基线。
自动单卡task_20260929_050655_15264614080完成exit=0，128K/B16与8K/B24交替A/B顺序，
各5预热/20次无profiler真实编译计时，另采四个level-4窗口。
固定CANN9.2/mode2/atomic0/det0、正式layer4权重与独立合成历史，EPLB关闭，
ring=[256,128,256,32]MiB、task_window4096。复用八类完整状态、图/eager、Top-K和保护区检查。

先看相同worker/有效query下的系数核时，再单列CSA、P95和max；长短8:2，不按局部核时推算CSA。
两代表档均走S6，双query特化本轮仅完成编译检查；若保留，在阶段出口覆盖8K/B32等受影响档位。
两档八类完整状态跨版本精确一致，编译图/eager、Top-K结构、metadata和保护区通过。
16个DFX窗口通过官方时钟域、行数和block核对，系数worker两侧均为16/24，没有空worker样本偏差。
[候选补丁](candidate.patch)、[生成调用点](lowering.json)、[冻结来源](source.txt)、
[编译入口](compile.py)、[单卡入口](run.sh)。

## 结论：保留性能版，短档CSA回退单列

| 档位 | 系数核时μs | 核时变化 | CSAμs | CSA变化 | P95μs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 4.579→3.318 | −27.544% | 1046.038→1031.526 | −1.387% | 1056.980→1042.780 |
| 8K/B24 | 4.559→3.165 | −30.589% | 968.199→978.979 | +1.113% | 986.920→1003.300 |

代表两档按8:2加权CSA−0.887%。系数任务范围、有效query、worker和工作分配相同，
因此其核时可以直接对照；核时仍包含DMA和等待，不等于纯算术吞吐。
按用户“保留真实核内收益”和长短8:2规则采用，不把短档CSA+1.113%、P95+1.660%隐去。
本轮短档候选P95/P50=1.0278、max/P50=1.0397，两侧均没有超过P50×1.05的样本；
这只描述20次采样，不关闭历史间歇拖尾或EP16稳定性问题。

长档Score首次start反而330.135→337.190μs，短档372.220→365.400μs。
Score AIC/AIV核时长254.627/260.250→250.000/255.733、短30.041/42.860→28.360/41.896μs，
但Score源码算术未变，不另记一项Score算法收益，也不把CSA降幅全部归因于系数提前。

生产仅移入已经实测的系数函数改动，生产/测试两根解析通过；精度版、Native cache和调度开关未改。
双query分支只完成编译，在阶段出口补8K/B32及其他受影响档位；完整七档表仍为c93ec723。
下一项检查在UB构造对角块后一次写回，以减少现有“先写零、再逐行补写”的MTE3操作，
仍须实测而不能用指令数量代替性能。
[正式结果](RESULTS.md)、[Worker分项](TASKS.md)、[精简验证记录](summary.json)、[完整证据](evidence.json)。

固定第4次重放window_3供查看，不挑最快窗口：

| 档位 | 基线泳道 | 批量候选泳道 |
| --- | --- | --- |
| 128K/B16 | [JSON](h131072_b16/swimlane/baseline/dfx/window_3/merged_swimlane.json) | [JSON](h131072_b16/swimlane/candidate/dfx/window_3/merged_swimlane.json) |
| 8K/B24 | [JSON](h8192_b24/swimlane/baseline/dfx/window_3/merged_swimlane.json) | [JSON](h8192_b24/swimlane/candidate/dfx/window_3/merged_swimlane.json) |
