# Indexer Score大阶段的硬件流水归因

用户要求抓大放小。已有完整L1与阶段时钟显示，长B16的Score约230μs、Attention
约160μs；AIV有较长Score READY等待，但不能据此断定AIC在算矩阵还是等搬运。
不再追加G/leaf/panel/分核参数扫描，本次先补这一处关键证据。

## 使用完整CSA的真实中间输入

先以正式第4层W8A8跑完整CSA A→B→A及原多task精确对照，再从PreparedDynamicCSA
取query INT8、head系数、Native cache/页表、Q/L、position、当前group plan。
`score_kernel.py`直接调用正式`indexer_score_cube_stage`和
`indexer_score_vector_lane_stage`，两角色在一个24组MIX中并行。
保留原FP16/ReLU/加权顺序、动态有效范围、所有有效leaf、scale读取与sort/merge、
FFTS两槽READY/ACK和末尾排空；没有用理想化GEMM替代真实Score。

只另分配Score输出，在eager、graph和profiling后逐bit比较完整pair/score allocation。
本轮是长B16/G6/balanced/ROOTS1，完整CSA最终根发布只读这些leaf根；不推广为
会原地覆盖根的其它ROOTS策略的验证。保持被测完整CSA的输出和Native状态不变。

CPU `cpu_v1/result.json`符号编译通过，不传实值shape。生成代码只有一次MIX提交，
sync_start=true，13个根Tensor；内部Key0/64视图仍是零拷贝。没有修改生产数学。
初次shell从非Git工作目录source load-env返回128，尚未进入编译；补齐cd仓库后
首次有效编译通过。此环境脚本问题没有算作编译器缺陷。

## 采样及限制

- 完整CSA功能通过后，单独编译/执行同一Score函数，先核对上述输出。
- 5次预热、10次图外Event单独记录；它包含外围调度，不是完整CSA L1。
- 3个PyTorch profiler step，AiCMetrics.PipeUtilization；保留原始CSV、trace和各列。
- kernel模式外围等待仍可能进入PMU统计，独立启动也会影响cache；这些是归因
  指标，不将利用率简单换算为必然可省耗时，不用于宣称七档30%达标。

设备任务 `task_20261018_125149_165487929633`，命令在SUBMISSION_V1.json，
已完成，设备及PMU结果见下。不重测Native、静态或七档性能。

## 设备结果（已完成）

`task_20261018_125149_165487929633` exit0。完整CSA A→B→A检查通过，独立Score
在eager、graph和profiling后，完整pair/score allocation均与完整CSA逐bit一致。
三步原始CSV及列值保留，汇总见PMU_SUMMARY.json。

AIC三个样本的MTE1活跃时间中位154.814μs、矩阵103.951μs、FIX133.343μs、
MTE2 77.640μs；AIV Vector99.168μs。独立Score图外Event中位308.340μs。
这些pipeline时间互相重叠，且包含独立启动环境，不能相加、简单相减计算可省时间，
也不能代替原完整CSA中的约230μs阶段时钟。原始首步较慢样本完整保留。

源级核对WS FP16 NZ→L0B使用load3d，下一候选评估等价load2d转置搬运；
保持已存在的Key两槽、QK/WS交错及全部事件，未据PMU先行宣称提速。
