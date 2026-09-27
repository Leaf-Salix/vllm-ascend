# B8同轮forward事件与trace边界诊断

71153bb3七档中，128K/B8正式10步Native55.390→PTO58.035ms（慢4.77%），
其另一次请求轮的rank0三步profile却为主图58.801→56.423ms（快4.05%）。
不能用后者的CSA/FFN分解闭合前者的2.64ms差额。

CPU核对完整主流图，从首个捕获任务到末个NOTIFY_RECORD，三次主流任务序列完全相同：
Native59.108ms、PTO56.691ms；HC边界以外分别只有306.913/267.913μs。
该核对只说明已有profile省略的图首尾较小，**未证明正式事件与图跨度之间没有空隙**。
[原始边界](../csa_indexer_adaptive_20260928/model/h131072/b8/boundary.json)、[分析器](analyze.py)。

## 新增观测方式

测试入口增加默认关闭的`--profile-forward-events`。
只在已有profile的三步内，给同一次`_model_forward`记录begin/end设备事件和已有CPU位置；
事件与trace按实际满档step编号配对。profile窗口外不加事件，结束时恢复原钩子。
结果放在`window[0].profile_forward`，与正式无profiler的`steady_window`分开，
不将受profiler影响的耗时当正式成绩，不新增设备tensor复制、hash或逐步同步。

## 单卡小图验证

task_20260928_043451_362333916880退出0。[程序](probe.py)、[命令](run_probe.sh)、[报告](probe/report.json)。
BF16小图先捕获、再重放；确认只采满档step2/3/4、事件时间戳更新、位置记录和窗口长度正确，
结束后execute/forward钩子均恢复，原无profiler forward观测仍可使用。
这不是CSA精度或EP16性能测试。

## EP16单档

task_20260928_043821_36503161301：[命令](run_model.sh)。冻结工作树`.cache/csa-forward-boundary-71153bb3`，
算子仍71153bb3，只更新两个测试文件；Native流程、算术、cache、确定性和调度均不改。
只测128K/B8，双方重新采控制，mode2/atomic1/det0/EPLB关，原正式10步仍保留，
另三步profile带同轮forward事件。Native/PTO运行时event模式按原样记录，不改配置制造胜负。

采完并CPU导出rank0后运行`python <本目录>/analyze.py`。
若事件显著长于同次完整图，再定位图外发射/队列等待；若二者相符，继续查不同请求轮的cache与EP等待，
不能提前将差额定性为纯CSA调度、atomic或某种算术策略。

## 已完成结果

task_20260928_043821_36503161301退出0；正式forward Native56.771→PTO57.541ms（慢1.36%），
P95 57.408→58.533、max57.639→58.850；每步最慢rank均值57.219→58.099ms。
32768输出token无差异，16rank DSpark一致；本次仍未取得B8整模型优势。
[正式结果](model/RESULTS.md)、[全部rank样本](model/forward.json)。

同一次rank0 profile：

| μs | Native | PTO |
| --- | ---: | ---: |
| forward事件均值 | 59941.788 | 58166.847 |
| 完整主流图均值 | 59749.587 | 57972.107 |
| 事件减完整图 | 192.201 | 194.741 |
| 完整图减HC首尾 | 305.413 | 273.907 |

三步CPU位置相同、主流任务序列各自一致、窗口8/9/10正确配对。
图外事件间隙在此profile中两侧相近，未见毫秒级PTO发射等待。
不能从带profiler数据证明正式窗口绝无图外等待，也不能将另轮profile快3%左右当成正式成绩。
历史验证日志§137～139已证明Native软件event与PTO硬件event对profile的扰动不同；
本次实际模式仍0/1。因此轮次状态和profile扰动都需保留，不再为其胜负反转重复跑相同诊断。

已有三步分解：21层CSA body节省2.669ms，FFN却增加1.407ms；
事件边界核对支持继续调查下游专家工作量，未支持更改计时边界来制造优势。
[边界证据](model/h131072/b8/boundary.json)、[CSA/FFN任务](model/model_gap_rank0.json)。
下一项仅对当前原cache源码执行atomic0干预，先单卡成本、后受影响的两档EP16，保留token/DSpark失败门禁。
