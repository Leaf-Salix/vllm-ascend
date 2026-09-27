# 固定K QR投影按M分组

基底88d0744f，独立工作树`.cache/csa-qa-adaptive-88d0744f`；不包含尚在评估的KV sync_start。
Native七档投影证据显示QR任务存在核内差距：8K/B16 Native整核18.83μs、PTO单窗口block均值31.90μs，
B40对应19.81/61.06μs。输入和统计范围不同，只作定位，不能直接换算加速比。

性能版atomic0的小于128行用M32、大于等于128行用M64；按完整M块数启用最多3组。
N/K原分解外再加M组，每个输出依然由唯一块按K256顺序完整规约4096列，不引入atomic。
ND/NZ共用分组规则，NZ权重保持原Native存储；QR的RMS/量化、QB、KV、cache和精度版不改。
atomic1维持M64、单M组和原split-K/seed；编译后选择规则不按测试脚本换源码。

T96由8块扩展24块，三次权重遍历仍为三次；T240由8块扩展24块，每组M64+尾M16，总遍历仍六次。
期望减少QR串行M遍历及后方KV的忙核等待，但总核时间/资源竞争可能增大，必须分别报告。

先完整CPU编译；通过后补8K/B16、B40、B4的固定规约状态/图重放，长短代表档计时和核内。
本文件目前记录待测假设，不能作为已获得性能收益的证据。

CPU完整编译已通过，atomic0、四张NZ根权重，TaskId跨M分支返回和NZ偏移证明均通过编译。
task_20260928_070150_117104915830已开始；源码冻结。8K/B16先通过精确状态与图重放，完整CSA尚未改善，不提前采用。
B16/短、B40/短、B16/长保留20次计时，B4只看尾部padding和状态。
DFX每侧两窗口：长B16/短B40的基线复用刚结束sync候选的baseline（同88d0744f算子、同配置/输入），
短B16补基线DFX；候选三档分别采集。只读复用不新增baseline设备测试。

2026-09-28重新以depth=1读取官方hw-native-sys/pypto-lib upstream/main，仍为2164563；环境安装未变。
其QA仍为M64、split-K2、需要seed与atomic；KV采用已有split-M框架、split-K4。
本候选保留固定K并增加QR的M维并行，原因是此前真实EP16的固定规约策略共同获益，不能只为单层并行度恢复atomic。
Native历史CANN9 ND观测曾确认无跨core split-K（验证日志旧记录）；那不是本轮NZ的实际tiling，不能照抄其22核配置作当前证据。
当前Native投影定位依据见[七档真实模型耗时](../csa_atomic_matrix_20260928/NATIVE_PROJECTION.md)。
