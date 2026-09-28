# 长档S6 Score：预取下一Key面板到L0B

独立基底2d2f9ca0，工作树`.cache/csa-score-key-prefetch-2d2f9ca0`，未改生产实现或组合验证源码。
参考最新ops-transformer b5b33e14 QLIV2Matmul::ProcessQk/LoadKeyToL0b：L0A/B使用四个16KiB槽轮转。
当前PTO S6/M384/N64的生成代码在稳态对INT8 Key Right使用L0B地址0，
下一panel的TMOV在PIPE_M→PIPE_MTE1等待后才能重用；FP16 WS Right占8192起的48KiB。
首块Key临时使用8192地址，此时WS尚未开始；这不构成稳态Key双缓冲。
已读2d2f9ca0生成代码确认：L1最大分配末端96KiB、L0A60KiB、L0B56KiB、L0C100KiB。
此前638976字节的L1失败是候选表达的分配结果，不能解释为基线已接近L1容量上限。

本候选在S6长档中让两个N64 Key Right同时存活，先装入下一panel再消费当前panel，
目标是2×8KiB Key加48KiB WS占满64KiB L0B，保持现有Query/系数驻留。
完整CPU编译和实际L0B分配通过，设备状态也通过，但长档核内和完整CSA均回退，本版不合入。
与旧页表UB预取不同，本项预取Key实体到L0B；分页查表、0/64B视图、tail clamp和读取总量保持。
QK/WS形状、FP16量化/规约、score分块、Top-K、24MIX任务及调度标志不改。

只在算子已有长历史S6路径启用：最大压缩长度>8192、query数>=96；
8K、长档双/三query保留原逐panel读取；这些分支生成二进制已定向比较一致。
最初抽取inline helper时，当前JIT无法推断中间cache视图的Tensor metadata；
已改为在原核内展开一轮Key预发，未修改PyPTO。constexpr开关在各调用点显式绑定。
已完成CPU完整编译并核查L0B，长短B16单卡任务`task_20260928_170324_293095832121`完成、退出0。
[CPU入口](compile.py)、[成功日志](compile.log)。

## CPU表达调整与当前状态

第一版统一循环加条件序段，Tile初始化在分支内，SSA检查报告153处作用域错误；
[完整诊断类别与位置](ssa_scope_failure.json)保留，省去153次重复打印的整程序IR。
将两个Right缓冲定义移到分支外后，SSA通过，但Mat分配638976字节超过524288上限；
[该版补丁](guarded_qk_failure.patch)、[CPU失败日志](guarded_qk_memory_failure.log)保留。
这是L1表达/生命周期限制，尚不能宣称所需L0B双缓冲已生成。

当前[candidate.patch](candidate.patch)将首个Key加载放到独立prologue，原QK/WS循环不增加计算条件，
只在既定长档S6路径提前读取下一块Key；尝试避免条件计算带来的Mat中间值存活扩大。
该表达将Mat分配恢复到96KiB，完整lowering/PTOAS/CCE与AICPU链接、load通过。
第一次解析发现bool不能与panel索引相加，已把constexpr改为0/1预发面板数；
[解析诊断](prologue_bool_parse_failure.log)保留，不修改工具链来隐式接受bool算术。

最初计划等组合EP16结束再编译。原任务持续排队后，改用[自动编译保护](compile_guarded.py)：
仅pending或已终止时执行CPU编译，模型进入running或状态不明时挂起本脚本创建的整个编译进程组。
本轮两次编译分别在16:57:04和16:58:35结束，当时EP16仍pending，未触发挂起，未与模型正式窗口交叠。
[UTC过程记录](compile_guard.jsonl)保留；没有取消或重建原EP16任务。

## 生成代码

[分配和指令顺序](codegen_evidence.json)确认：L1/L0A/L0B/L0C最大分配末端为96/60/64/100KiB。
首个Key临时位于16384，稳态两个Key槽在0/8192交替，WS Right固定16384起的48KiB。
每个N1024块仍是16次Key搬入、16次QK和16次WS；前15轮均在当前QK前发出下一Key的TMOV。
L1复用仍生成MTE1→FIX保护，不声称消除了全部同步等待。

仅对受inline修改影响但不启用预取的四组AIC/AIV，以及长S6的AIV做直接二进制比较，9份均一致；
未扫描其他算子或输入数据。[对应产物和大小](unchanged_kernels.json)。
C++中出现的额外Right描述符没有改变这些生成二进制；这不代替短档完整CSA/P95实测。
候选八类状态及图重放已经通过；设备结果不支持保留本版。

[单卡对照入口](run_layer.sh)完成。[CPU汇总](summarize.sh)复用现有比较，不新增设备执行。
基线为同一2d2f9ca0，长短B16交换执行顺序，每侧5次预热、20次无profiler图计时、4个独立DFX窗口；
八类输出/状态必须零容差，保留Native控制、metadata保护区、自重放及候选A→B→A。
重点判断128K Score核内及完整CSA，8K检查回退，二者按7:3记录；不改变算术容差或原生cache布局。

## 单卡结果与取舍

两档八类输出/状态逐元素零容差一致，metadata/保护区、自重放、计时图状态和A→B→A均通过。
下表为同轮2d2f9ca0基线/本候选；完整CSA来自20次无profiler图计时。

| 档位 | 完整CSA μs | 变化 | P95 μs | max μs | Score AIC核内 μs | AIC变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 1108.059/1152.131 | +3.98% | 1120.46/1172.54 | 1127.12/1179.86 | 289.107/332.088 | +14.87% |
| 8K/B16 | 776.113/779.537 | +0.44% | 791.94/793.34 | 795.00/798.42 | 43.700/43.486 | −0.49% |

7:3完整CSA **+2.917%**，Score AIC **+10.260%**；两个层级都没有保留依据，P95也未改善。
长档四个独立窗口AIC均值：基线289.739/286.789/288.947/290.951μs，
候选330.963/331.236/333.483/332.668μs，回退稳定；AIV均值301.839→344.959μs，含等待AIC。
长档Score AIC启动散布1.935→3.315μs，包络341.125→388.405μs；任务数未变。
所有窗口Score/Sparse均24个AIC各一份，不能据这次未复现宣布旧调度尾部已修复。

Native控制长档1329.045→1313.446μs、短档930.691→950.573μs；未改merge_norm长档28.118→22.584μs。
短档Score生成二进制不变，其小幅变化只能作为整层回退观察，不能归为核内收益。
保留全部样本，不归一化Native、不把独立DFX与正式计时逐个配对。

代码层面确认Key与Score复用96KiB L1池，预取路径每轮引入MTE1→FIX保护。
这提示预取的收益被缓冲复用/依赖抵消，但现有DFX没有逐指令stall，不能将全部43μs增量归给某条等待。
最新AscendC QLI V2在`InitBuffers`分别分配双槽Key L1与双槽Score L1，
PTO本版只完成L0B双槽，还没达到同样的L1隔离。下一候选先在算子侧表达独立Key池并检查生成代码；
编译和分配合理后再做必要单卡，不扩测失败版本的EP16。

[128K原始样本/状态/窗口/分派](h131072_b16/evidence.json)、[8K证据](h8192_b16/evidence.json)、
[各项七三变化率](weighted.json)。完整summary和泳道在对应本地目录，不重复提交大型trace。
