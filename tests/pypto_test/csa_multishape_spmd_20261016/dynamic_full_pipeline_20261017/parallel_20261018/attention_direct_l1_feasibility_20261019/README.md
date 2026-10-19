# Attention 压缩 KV 直接进入 AIC L1 的可行性

当前已采用路径：Native cache → AIV UB → GM kv_ring → AIC L1，压缩块每条
128×512 BF16 数据额外经过一写一读 GM。优先评估直接按本轮物理行计划
进入 AIC L1，保留原 QK/PV 数学、三槽与所有有效性检查；不是减少计算。

当前独立 PyPTO 的 tile.mgather 支持 target_memory=Mat、GM INT32 索引；
实际 Simpler build/pto-isa 的 A3 MGatherGm2L1RowImpl 逐行执行 ND2NZ。
这不是一次大 DMA，也不预设一定更快：AIC Scalar 索引读取会增加，且零填充、
槽切换和编译后的局部事件可能抵消少搬运的收益，必须用整层真机结算。

triple_slot_probe 仅确认三个 Mat Tile 的动态循环携带及索引子视图能力。
使用独立 gm-subview PyPTO 和同一 PTOAS v3，不改共享安装。
该小探针的编译或精度不能充当完整 CSA 性能或跨核协议证据。

整层接线约束：

- 只在生产者和消费者共同确认物理 plan 可用时进入；容量外走原逻辑路径。
- AIC 直接读主压缩 cache 前必须取得真正的 MAIN_OUTPUT 发布及数据获取语义。
  原来只有 AIV 获取该依赖，不能因 plan 已完成就假定 cache 已可读。
- 先保留原每个有效块的全部 FFTS READY/SCORE/PROB/PV 事件数量与顺序。
  压缩块的 KV 数据由 AIC 自行加载，AIV 的原 READY 继续约束环槽推进。
- SWA 保留原路径；无效行由 MGATHER ZERO 填充，不允许未初始化 Mat 值污染数学。
- 三个 L1 槽的数据必须活到对应 PV 完成，不能用第四份大 Tile 绕过容量或提前覆盖。
- Host 不读取新增 metadata；T/B/长度/页表/位置和尾块继续动态。

当前状态：仅准备 CPU 能力探针，尚未接入正式算子，尚无设备或性能结论。

CPU v1误用[128,1]索引，接口要求[1,R]，补无损GM reshape后重试；v2的
嵌套显式yield没有逐层返回，改为完整显式作用域；二者属于探针编写修正。
v3在三个不同物理Mat缓冲之间选择消费者时，PyPTO插入A3不支持的
Mat→Mat TMOV，和#3016的同存储误copy不是同一个边界。另做不含MGATHER/
gm_view/循环的最小复现以供上游判断。v4把消费者留在各槽分支，三槽
固定地址分别为16384/147456/278528字节，加query共409600字节；完整CPU
通过，无额外第四份大Mat槽。尚未证明全CSA容量、同步或性能。
真机最小用例task_20261019_105450_298137832722验证5/8步跨槽周转、非法
INT32索引、同地址A→B→A图和输入/输出保护；结果待实际任务完成。

三槽原版task298137832722失败；带失败数值的312303717021复现78个差异，
全部为部分应非零结果变零，eager与replay也不一致。仅在私有PTO-ISA头文件
GM→L1 ZERO填充后补PIPE_MTE2 barrier，task334363510447的6项同址图/保护
全部通过。首个私有头文件任务330837724509遗漏pkg_inc相对include，CCEC
编译失败；补齐本地头文件副本，不修改托管ISA或共享安装。

为排除GM子视图、三槽和循环，新增single_mat_device_repro.py，仅一次
MGATHER→QK，既有NEWP编译器和完全相同输入。原ISA327cd5869的12次结果
差异数[0,0,1,4,3,2,0,0,0,0,0,0]；加同一个barrier后12次全部精确一致。
任务分别341196029867(exit1)与341220127712(exit0)。这是原语级硬件数据
证据，仍不是完整CSA或性能通过。建议修复在gm_l1_zero_barrier.patch，
编译路径审计记录实际CCEC使用的私有头文件路径，未改共享依赖。

Mat条件选择已在未修改的上游72b7ffb9通过公开IR编译接口复现；首版使用
已移除的私有JIT方法失败不算复现。正式issue：PyPTO #3049，附完整最小
程序、两个Mat仍存活的约束和消费者下沉建议。未提交PR、未添加标签。

PTO-ISA正式问题：https://github.com/hw-native-sys/pto-isa/issues/352；
附单Mat完整复现与仅一处MTE2屏障的建议补丁，没有PR或标签。
完整动态候选另在attention_direct_l1_full_pipeline_20261019推进，不扩大本目录原语结论。
