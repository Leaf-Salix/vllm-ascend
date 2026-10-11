# 128K/B16 CSA实际Scalar前缀覆盖

2026-10-11，实施过程055。单卡0任务 `task_20261011_185759_356175720609`，退出0，卡已释放。
本轮新增诊断证据，未修改SDK、调度策略、算术、tiling、原任务依赖或公共CSA默认。
插桩会增加时钟读取和GM写入；以下是诊断运行的覆盖与时序，不能当作未插桩性能收益。

## 观测到的执行次数

“执行”指一个物理核上的task实例，含SPMD block和MIX独立成员，不是编排节点数。
共编译73个成员，本档实际执行47种成员；两轮各973个实例，第二轮改变隐藏输入。

| 项目 | 第一轮 | 第二轮 |
| --- | ---: | ---: |
| task执行实例 | 973 | 973 |
| 带gated early-dispatch标志 | 551 | 586 |
| 上项进入可提前执行前缀路径 | 551 | 586 |
| 派到忙核后且具有交接资格 | 282 | 345 |
| 实际同核交接 | 210 | 237 |
| 其中gated后继交接 | 73 | 81 |
| 其中普通ready后继交接 | 137 | 156 |
| 交接后Wait处仍有尾等待>5tick | 148 | 168 |
| 具备pending资格但未交接 | 72 | 108 |
| 上项的前驱有异步尾部 | 65 | 96 |
| 上项的前驱为同步任务 | 7 | 12 |

全部带gated标志的实例都进入了前缀回调路径，但这不等于全部都获得有效计算重叠：
数据可能已经就绪，或者本核前驱已经结束，且诊断本身会改变任务到达时序。
全程保留完整fanin与本核尾等待；两组输入的9项输出/缓存/状态逐bit对齐普通PTO，
各31项保护区与metadata检查通过。没有重测七档或16卡整模型。

## 哪些任务形成了交接

下表为两轮合计；`尾等待>5tick`只是到局部Wait时仍有流水未完成的证据，
不是精确的净节省时间，不能把全核数字相加当CSA关键路径加速。

| 成员 | 执行数 | gated派发 | 本核交接 | 交接后仍有尾等待>5tick |
| --- | ---: | ---: | ---: | ---: |
| `proj_a_mm_0` | 128 | 48 | 128 | 122 |
| `_proj_b_mm_nz_kernel__2` | 128 | 97 | 79 | 68 |
| `qproj_matmul` | 40 | 0 | 31 | 29 |
| `idx_qr_proj_matmul` | 48 | 8 | 40 | 29 |
| `kv_score_proj` | 48 | 0 | 42 | 31 |
| `kv_score_proj_0` | 48 | 0 | 44 | 6 |
| `indexer_score_topk_native_pair_aic` | 48 | 48 | 0 | 0 |
| `indexer_score_topk_native_pair_aiv` | 96 | 96 | 0 | 0 |
| `qk_pv_aic` | 48 | 48 | 0 | 0 |
| `qk_pv_aiv` | 96 | 96 | 0 | 0 |

当前实际同核交接主要出现在投影链，O-A两轮均64/64交接，其中各61条仍有尾等待。
本次Score和QK/PV的MIX成员均进入了gated前缀路径，但没有观测到本核交接。
这与最小MIX例子能交接并不矛盾：机制允许MIX，实际CSA的到达次序和前驱状态决定命中。
因此不能把“增加12个MIX编译成员”直接换算成12组Scalar/Cube重叠收益。

## 尚不能归因的空隙

72/108条未交接不能全部归为一次探测丢失：其中7/12条的前驱根本没有异步尾部。
其余65/96条的后继入口晚于已记录的前驱完整完成，但当前没有AICPU派发时间戳，
无法区分“探测后、计算结束前到达”和“计算已结束、CPU尚未处理FIN时才派发”。
前者才是补探测能挽回的窗口；后者本来没有剩余本核计算可重叠。
[逐条列表](pending_without_handoff.json)保留了两者尚未分开的原始事实。

`indexer_score_topk_native_pair_aic`诊断中退出后的剩余尾等待很短，`qk_pv_aiv`也类似。
这不证明它们是Scalar bound：长循环内部的管线等待和MIX通信也计入函数返回前的时间。
目前没有足够证据把任一CSA成员定性为纯Scalar计算瓶颈。

## 非阻塞完成查询的核实

前一过程已证明A3拒绝try_wait。此轮CPU编译确认get_status和get_condition_flag可编译，
但这不证明它们具有流水完成语义。CANN本地TIK代码以get_status读取溢出告警；
没有找到可作为完整流水完成判据的契约，因此未将这些寄存器用于运行时正确性判断。
本地反汇编工具输出not available，不能凭它猜测寄存器位含义。

官方[PipeBarrier说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/API/ascendcopapi/atlasascendc_api_07_0174.html)
定义了流水阻塞同步；[HasIdleBuffer](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0156.html)
查询的是TQue空闲buffer，不是所有硬件流水结束状态。二者不能当成非阻塞完成查询。

另一候选是把各流水尾部有序接到MTE3，由DMA写完成标志，再由Scalar查询标志与后继。
A3 Cube侧MTE2→L1、M/MTE1/MTE2/FIX→MTE3同步、L1→GM通知已通过CPU编译；
首次缺少pad_t参数的错误与修正记录均保存。该探针还没有设备验证，也尚未接入runtime。
它需要独立scratch和完成标志，以及额外搬运/查询；下一步仅做小用例验证可中断等待与
无后继时的新增开销，不直接替换当前CSA路径，也不重新启用已否定的调度/ACK改动。

源码核对还发现旧文档把RuntimeBarrierAll写为6；当前Simpler枚举实值为5。
本次诊断依赖编译头文件的枚举，而不是文档数值。此说明更正文字，未改线上的协议值。

## 证据与复现

版本见[versions.json](versions.json)。[results.json](results.json)含函数映射、各任务计数、
数值和保护区结果；完整时序分别见[第一轮CSV](round0_records.csv)、[第二轮CSV](round1_records.csv)。
16列含原始设备tick、派发序号、前驱序号、能力位、实际回调/交接位与Wait前后时刻；
采用原始tick，不依赖未经核实的频率换算。未出现容量溢出或缺失前驱，逐实例等待不变量通过。

私有驱动和插桩器随目录保存；它们只在测试进程里生成新的kernel wrapper。
为避免改变编排入参和依赖，诊断地址绑定到当前进程的专属缓冲，编译在取得单卡之后完成；
这些产物不能跨进程重用，不能放入正式性能测试。使用私有env.sh，经task-submit执行
`run_csa_prefix_coverage_128k_b16.sh`。CPU编译检查先修复了调用方缺少include路径，
以及把公共enum声明误当实际tail调用导致的样本选择错误；这些都不是设备算子故障。

用户接受的局部Wait显式标记及诊断模式串行回退继续保留；本诊断使用专属trace缓冲，
没有开启chip swimlane、PMU或参数dump。公共算子和另一会话HC_PRE文件均未改动。
