# 长档 Score：KV scale 矩阵广播候选

基线c93ec723；两侧冻结整包私有pkg。仅长档balance_leaves路径改变scale乘法的组织：
先对S6×512的完整Score tile做col_expand_mul，再按各query可见长度发布；
短档沿用逐query slice+mul。量化/规约、排序、分页cache和调度开关保持原规则。

Native arch22 QLI V2 ProcessVec1先读取Cube WS结果和KeyScale，再在AIV完成FP32乘法；
当前PTO已对同组query复用同一份scale，本候选进一步利用PTO矩阵广播表达，减少逐行的依赖切换。
这是围绕Native数据流的PTO实现调整，不声称Native源码也使用同一矩阵广播。
最新本地pypto-lib 2164563的decode Indexer按单query分工，不能直接获得这项S6合并机会；
其prefill已有col_expand_mul API用法，但输入、规约和任务范围与这里不同。

完整编译/load通过。长S6 AIV生成代码由6个TMUL调用点变为1个TCOLEXPANDMUL调用点，
TSTORE仍60处、TEXTRACT仍48处，未消除score_arena GM中转。
最高静态Vec地址末端两侧均135168字节；动态gather子视图仍指向同一scale缓冲。
以上是静态生成代码证据，不等同设备指令条数、动态UB峰值或实测性能收益。
[生成代码摘要](codegen.json)、[候选差异](candidate.patch)。

初次CPU编译暴露两处DSL约束，均已在算子侧修正：跨constexpr条件使用的变量先显式定义；
tile slice的动态有效长度在slice内声明，不能对view再set_validshape。未修改PyPTO/PTOAS/ISA。

代表档128K/B16、8K/B24；5预热/20次无profiler图事件，各侧独立四个DFX窗口。
八类跨版本状态、A→B→A、保护区检查复用既有入口，按8:2分别报告核内/CSA及P95/max。
Native手工图列只作环境控制，不能与真实模板编译基线混用。

## 实测结论：不合入

task_20260929_024806_385090111805通过auto单卡完成，exit=0。
长B16 CSA1053.321→1040.351μs（−1.231%），短B24为961.783→948.199μs（−1.412%），
长短8:2为−1.268%，P95/max均下降，八类跨版本状态、A→B→A及保护区通过。

但长B16四窗口Score AIC250.570→268.474μs（+7.145%）、AIV256.184→274.200μs（+7.032%），
merge10.992→14.559μs；AIV四个候选窗口均慢于四个基线窗口。
短档Score AIC/AIV也上升，虽然算法分支保持原规则。加权Score AIC/AIV为+8.539%/+8.422%。
核时含内部等待，因此不能仅凭这些数字断言矩阵乘法指令自身变慢；
同样不能凭静态调用点减少或完整CSA略降就宣称本候选取得核内收益。

短档Native手工图控制1048.381→1014.591μs，本轮CSA降幅存在环境/调度变化的归因限制。
按当前优先降低长档核内耗时的目标，不合入生产；保留补丁和证据，不扩跑七档或整机。
此结果不覆盖实际编译半层的间歇长尾，也没有Native逐元素或模型token/DSpark验收结论。
[完整结果](RESULTS.md)、[原始计时、状态与泳道路径](evidence.json)。
