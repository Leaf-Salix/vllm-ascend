# 长档S6 Score：预取下一Key面板到L0B

独立基底2d2f9ca0，工作树`.cache/csa-score-key-prefetch-2d2f9ca0`，未改生产实现或正在排队的组合验证源码。
参考最新ops-transformer b5b33e14 QLIV2Matmul::ProcessQk/LoadKeyToL0b：L0A/B使用四个16KiB槽轮转。
当前PTO S6/M384/N64的生成代码在稳态对INT8 Key Right使用L0B地址0，
下一panel的TMOV在PIPE_M→PIPE_MTE1等待后才能重用；FP16 WS Right占8192起的48KiB。
首块Key临时使用8192地址，此时WS尚未开始；这不构成稳态Key双缓冲。
已读2d2f9ca0生成代码确认：L1最大分配末端96KiB、L0A60KiB、L0B56KiB、L0C100KiB。
此前638976字节的L1失败是候选表达的分配结果，不能解释为基线已接近L1容量上限。

本候选在S6长档中让两个N64 Key Right同时存活，先装入下一panel再消费当前panel，
目标是2×8KiB Key加48KiB WS占满64KiB L0B，保持现有Query/系数驻留。
完整CPU编译和实际L0B分配已通过，是否减少设备等待及收益仍需实测，不能用逻辑容量推导当作性能通过。
与旧页表UB预取不同，本项预取Key实体到L0B；分页查表、0/64B视图、tail clamp和读取总量保持。
QK/WS形状、FP16量化/规约、score分块、Top-K、24MIX任务及调度标志不改。

只在算子已有长历史S6路径启用：最大压缩长度>8192、query数>=96；
8K、长档双/三query保留原逐panel读取；这些分支生成二进制已定向比较一致。
最初抽取inline helper时，当前JIT无法推断中间cache视图的Tensor metadata；
已改为在原核内展开一轮Key预发，未修改PyPTO。constexpr开关在各调用点显式绑定。
已完成CPU完整编译并核查L0B，再提交长短B16单卡任务`task_20260928_170324_293095832121`，尚无设备结果。
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

## 生成代码与设备待验项

[分配和指令顺序](codegen_evidence.json)确认：L1/L0A/L0B/L0C最大分配末端为96/60/64/100KiB。
首个Key临时位于16384，稳态两个Key槽在0/8192交替，WS Right固定16384起的48KiB。
每个N1024块仍是16次Key搬入、16次QK和16次WS；前15轮均在当前QK前发出下一Key的TMOV。
L1复用仍生成MTE1→FIX保护，不声称消除了全部同步等待。

仅对受inline修改影响但不启用预取的四组AIC/AIV，以及长S6的AIV做直接二进制比较，9份均一致；
未扫描其他算子或输入数据。[对应产物和大小](unchanged_kernels.json)。
C++中出现的额外Right描述符没有改变这些生成二进制；这不代替短档完整CSA/P95实测。
当前候选尚未合入生产，没有设备正确性或收益结论。

[单卡对照入口](run_layer.sh)已提交，当前排队。CPU编译和实际L0B分配已通过。
基线为同一2d2f9ca0，长短B16交换执行顺序，每侧5次预热、20次无profiler图计时、4个独立DFX窗口；
八类输出/状态必须零容差，保留Native控制、metadata保护区、自重放及候选A→B→A。
重点判断128K Score核内及完整CSA，8K检查回退，二者按7:3记录；不改变算术容差或原生cache布局。
