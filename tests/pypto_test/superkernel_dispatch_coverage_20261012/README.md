# CSA 派发与 Scalar 前缀关联：pending 不等于仍有本核重叠窗口

2026-10-12，过程082。两次插桩执行中，221次pending未交接有220次在AICPU开始写派发寄存器之前，
前驱的完整尾部等待就已结束。285个在前驱函数返回前完成AICPU发布的后继全部交接成功。
本次没有发现这些实例被额外准入条件挡住；不能据此证明所有场景均已覆盖，也不能将插桩比例外推到正常运行。

手工局部`wait_pre_task_end()`保持，不做边界自动识别；chip swimlane、PMU、参数dump串行回退保持。
本轮只保留诊断工具和证据，没有修改正常派发、ACK、核选择、early-resolve或公共CSA默认路径。

## 采集方法与边界

私有vLLM驱动`7c7ecb41`在AICPU原有`write_reg(DATA_MAIN_BASE)`前后读取计数器，
后一个采样位于`wmb()`之后；记录GlobalContext地址、派发代次、逻辑核号、gate和前驱信息。
AICore沿用现有逐成员trace，增加GlobalContext地址，按地址和代次关联两侧，再按实际物理核号寻找前驱。
没有假设AICPU逻辑核号等于AICore硬件核号。两次各973条记录全部匹配，代次、前驱和标志也一致。

AICPU和AICore写不同trace缓冲，避免缓存行互相覆盖。运行时编译到本次输出目录，
构建结束立即恢复两处Simpler源码，普通运行时二进制未替换。
新增读钟、存储、屏障和缓存刷新会扰动调度；此诊断仅分析实际采集的时序，不是性能测试。
发布区间描述AICPU的写操作，**不等于已测得AICore寄存器可见时刻**，也不是原计算结束时刻。

当前平台接口将两侧系统计数统一为50MHz，即每tick为20ns；使用原始计数，不以Host时钟推导。
完整尾部等待仍包含通知DMA，不能据此把某个task判为Scalar-bound。
本次71个带尾部的生成成员都有编译通知回调，未命中资源耗尽后的阻塞回退。

## 两次实际执行

CANN9.2/TMR，单卡0、128K/B16、第2层HC_pre+norm+CSA+HC_post、NZ2、atomic0、deterministic0。
对照与候选使用相同Score算术和Native权重/cache接口；第二次改变hidden输入。
chip swimlane、PMU、参数dump均未开启；采用独立的手工诊断插桩。

| 计数 | 第一次 | 第二次 |
| --- | ---: | ---: |
| 成功关联的成员执行 | 973 | 973 |
| gated提前派发 | 558 | 530 |
| gated进入Scalar前缀 | 558 | 530 |
| 具有本核pending前驱的后继 | 264 | 269 |
| 实际本核交接 | 168 | 144 |
| pending未交接 | 96 | 125 |
| 其中：前驱完整等待结束后才开始AICPU发布 | 96 | 124 |
| 其中：落在返回后、尾部等待前的收尾间隙 | 0 | 1 |
| 前驱返回前完成AICPU发布 | 151 | 134 |
| 上一行中成功交接 | 151 | 134 |

每次9项输出/cache状态相对普通PTO逐bit一致、31项保护区通过。
这里的“进入前缀”“本核交接”不直接证明前驱原始Vector/Cube计算仍在运行。
gated task也可能派到空核或已完成的核上，其安全前缀照常先执行，Wait仍保护完整fan-in/cohort。

原通知计数两次805/823，回调后交接17/11。插桩本身会改变派发速度和这些比例，
不能用本次1628次通知、28次回调后交接替代未插桩环境的收益判断。

## 唯一的收尾间隙实例

第二次，物理核1（AICPU逻辑核4），派发14→15，前后均为`_proj_b_mm_nz_kernel__2`（O-B投影）。
以前驱函数返回为零点：

| 时刻 | 相对时间（μs） |
| --- | ---: |
| AICPU发布前采样 | 2.08 |
| AICPU发布后采样 | 2.14 |
| 前驱进入最后完整等待 | 2.62 |
| 前驱完整等待结束 | 2.64 |
| 后继进入函数 | 3.70 |

发布后采样比最后等待早0.48μs，但最后等待本身仅1tick（20ns）。
现有记录没有最后一次DMB探测时刻，也没有前驱原始Cube完成时刻；只能定位到收尾间隙，
不能定性为漏掉了0.48μs计算重叠，更不能认定前驱为Scalar-bound。
不为这一个未证实的计算窗口给所有task增加一次热路径探测，本轮没有新增性能候选。

## 原始数据、版本与复现

- [完整结果与逐成员原始记录](results.json)、[摘要](summary.json)、[离线汇总脚本](analyze.py)。
- [第一次物理核/代次/context关联](context_map_round0.json)、[第二次关联](context_map_round1.json)。
  每行依次为`[raw_core_id, reg_task_id, context_address]`；与结果中的AICPU原始8字段记录关联。
- [实际插桩补丁](scheduler_instrumentation.patch)、[O-B生成代码](example_ob.cpp)、[运行日志](run.log)。
- [完整版本](versions.json)、[诊断入口](repro/tmr_csa_dispatch_coverage.py)、[单卡排队脚本](repro/run.sh)。

任务`task_20261012_064423_24027093905`退出0并释放卡。PyPTO `f6edfc1fd`、Simpler `1d09ecd34`、
PTOAS `4df4627e`和PTO-ISA保持不变；只是私有驱动增加记录。

在相同私有环境、冻结算子包和同SDK普通对照产物下，将`repro/run.sh`放回私有环境的
`artifacts/run_csa_dispatch_coverage_20261012.sh`，三个Python驱动放回私有vLLM的`tests/pypto_test/`。
通过`task-submit --device auto --device-num 1 --run 'bash <私有环境>/artifacts/run_csa_dispatch_coverage_20261012.sh'`
执行；设备号取`TASK_DEVICE`。驱动依赖同目录既有CSA fixture和编译入口，归档不是独立发布包。
诊断二进制及生成CSA内嵌本进程trace地址，必须在运行进程内重编，不能跨进程复用。
完整`.pt`缓冲保留在私有输出目录；归档只存实际记录及关联字段，省去未使用的填充值。
运行`python3 analyze.py`可直接从归档结果重算摘要，无需NPU。

本轮没有新min/mean/max，最新未插桩性能继续引用过程081的保留路径；没有新增七档、Native或整模型结果。
后续优先依据未插桩对照检查交接/前缀本身的开销。一般只读GM前缀的依赖及实际别名依据、
更深控制流和通知资源耗尽的后到观察窗口仍需另行处理，不能以本次CSA覆盖代替功能范围验收。
