# 当前动态CSA的Attention四类FIFO等待观察

原正式算子、原100tick阶段GM等待、原wait_flag_dev和数学均保持。仅在诊断
编译返回的C++上包围Attention事件2(KV)、3(score)、4(probability)、5(PV)，
累加get_sys_cnt差值和次数；Indexer事件0/1不计入。每个角色只在Attention
结束后的首次阶段发布，把四类64位ticks/32位count写入独占512B槽的第3条
64B行（INT32索引32..43）。第一条信号行、原阶段时钟16..19及其他padding
保持原约定。诊断写回增加一次DCCI/DSB，且计时/寄存器压力均会扰动结果。

四类观察是Scalar包围原wait_flag_dev的区间，不是独占MTE/Cube/Vector耗时。
未为了计时插入流水线完成屏障；不能未经验证把发射计时当成真正硬件等待。
累计值只用于判断后续应核对的供数方向，禁止写入完整L1性能验收表或减去
它们宣称可获得对应收益。先以短历史B32观察主要Attention区间，不扫描七档。

复用正式W8A8权重与原多task精确参考，完整output/Top-K/score/cache/state、
保护区和原阶段协议检查仍执行。导出器从本次真实attention_mask计算各core
有效块数，逐核核对AIC的KV/PROB、两AIV的SCORE/PV计数；非Attention phase
不得写32..43。只做一次诊断调用，不代表metadata/连续轨迹或性能验收。
source_reference是完整动态源码，不使用静态entry。当前尚无设备结论。

短B32任务92587510497 completed/exit0，完整精确/状态/保护与72角色事件次数
全部通过。各角色40个有效块：AIC等KV的Scalar累计中位178.90μs，等probability
1.12；AIV等score6.94、等PV0.94。表明本插桩例应优先定位AIV供数链；KV等待
包含前拍softmax/merge/最终发布影响，不能把178.90全部归因KV DMA，也不能
视为可直接消除的延迟。长B16追加一次相同诊断，确认这一方向的覆盖范围。

只读核对当前PTO-ISA MGatherRowImpl：ZERO是逐无效行置零，并非每次整块
清零；不能据错误的整块清零假设改实现。该复合调用前后有V→S、MTE3→S、
MTE2→V/MTE3等内部同步，限制与无关Vector工作的重叠。下一步先考虑用
显式逐行DMA与仅无效行清零表达真实数据依赖，不删除现有同步来冒险。
PTOAS已用SyncMacroModel表达Scalar读indices和MTE2写dst，组件代码未改。

长B16任务145644111014 completed/exit0，完整精确/保护与72角色次数核对通过。
每角色20块：AIC等KV中位101.61μs、等probability0.78；AIV等score4.80、
等PV0.96。两个代表档均显示KV供给方向值得优先分析，但包含上游AIV所有
前拍工作，仍不把数值当单独DMA耗时或可消除时间。原始每核计数全部保留。
尝试用llvm-objdump读prebuilt/kernel_*.bin被拒绝：它们是裸机器码而非ELF，
不据失败声明已经核对最终指令顺序。失败命令/状态在machine_code_audit。
