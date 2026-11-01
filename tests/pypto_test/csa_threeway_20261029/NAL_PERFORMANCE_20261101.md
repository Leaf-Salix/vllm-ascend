# nalinaly 1574真实模型效率补测

用户明确要求：nalinaly的精度差异已知，本轮测效率，不再以输出token首差终止性能采集。
原先首差停止的试验保留为精度证据，不拿空性能结果代替本轮测量。

## 固定口径

沿用[真实模型报告](REAL_MODEL_PERFORMANCE_20261031.md)的完整配置和共享工具链。
公共框架41ba392f，nalinaly CSA为1574ed59，Python单SPMD后端，NZ2，HCA全部Native。
真实DeepSeek-V4-Flash-0731-w8a8、TP1/DP=EP8、fixed8192前缀bank、每rank40题分5波×8、
192输出token、DSpark出5验6、FULL6/240、KV4000、window2048、确定性开启。
只用本轮批准的8–15（含14），每项内部720秒/调度780秒，不使用前八卡。

不重复Native及我方任务，直接复用：

| 实现 | 既有结果标签 | 最慢rank目标forward均值 |
|---|---|---:|
| Native 41ba392f | last8-native-8k-wave-perf-v1 | 73.494ms |
| 我方65bebf7f | last8-ours-8k-wave-perf-v1 | 77.079ms |

这些是所选整模型目标forward，不是单CSA耗时。每rank预热8步后计20个B8/FULL240事件，
八rank合计160样本；按rank先取均值，再取最大值。分波客户端墙钟含入场、draft、采样及CPU检查，
不能作为纯设备吞吐。新旧任务跨时运行，不将一次结果差值称为稳定收益。

## 独立性能模式

新增`--nal-performance-only`仅允许nalinaly臂，并要求同时开启原token观测与分波计时。
只将已有Checker的差异回调替换为首差记录，之后使用内存标志直接返回；
不替换token、不回灌Native结果，不改模型、kernel、cache、采样或调度返回。
原严格postflight仍会记录token/接受事件失败，不将本轮称为精度通过。
Native与我方原先首差终止行为不受影响。

完成后无论token是否相同均输出有效计时统计。
如果实际有效T/positions与Native不同，则报告各自生成轨迹下的描述性延迟及差值，
同时明确不满足严格同输入门禁；不会再因数值差异留空性能表。
仍需检查计时样本完整性、21CSA实际替换、HCA Native及抢占情况。

复现沿用原`run_eight_card.sh`，新结果标签，arm=nalinaly，cohort=
`eightcard-nal-pre-aclnn-1574-v1`，reference=`last8-native-8k-wave-perf-v1`，额外参数为：

```text
--inspect-token-sequence --measure-waves --nal-performance-only
```

本轮只补跑该实现，不跑Native2、我方或128K。

## 结果

任务`task_20261101_125104_390313812277`，结果标签
`last8-nal1574-performance-only-20261101-v1`，已完成生成与计时并释放8–15。
子进程exit0、416.60秒（含加载/编译/捕获），无中途终止；调度exit1来自保留的精度后检失败。
不使用该总时长作为效率。

| 实现 | 整模型目标forward最慢rank均值 | 相对Native耗时 | 客户端分波墙钟，仅辅助 |
|---|---:|---:|---:|
| Native（复用） | 73.494ms | 基线 | 40.990s |
| 我方精度TND（复用） | 77.079ms | +4.88% | 60.257s |
| nalinaly1574（本轮） | **69.707ms** | **−5.15%** | 55.111s |

nalinaly的目标forward耗时比我方低约9.56%。这组forward数据体现它的效率优势，
但尚不能将全部差值归因到某个CSA子模块，或说成稳定的同输入收益。
客户端观测墙钟上nalinaly仍慢于Native；该指标混有入场协调、草稿、接受轨迹和CPU检查，
不能把其差值解释成CSA本体性能或纯设备吞吐。

本轮nalinaly共160个有效NPU Event样本，均值69.645ms，中位69.420ms，P95为70.626ms，
各rank样本输入签名一致。样本全部B8/有效T48/物理T240，和Native的这些规模相同，
但positions序列不同，八rank逐样本同输入检查全部false，因此原严格
`performance_qualified=false`继续保留；**计时本身完整，此表为各自生成轨迹下的描述性比较**。
旧Native/我方及本轮nalinaly跨任务、跨时运行，未重新控制硬件负载漂移。

21CSA实际替换、HCA Native、FULL回放及运行时覆盖检查均通过，零抢占。
nalinaly平均接受长度3.552414605，Native/我方为3.695018226，不能当作同接受轨迹。
完整生成61,440个token；rank0有3,894个token与Native不同，原精度后检失败保留。
本轮不再因已知差异停止测性能，也不继续追加精度定位任务。

最终CPU门禁7项通过（11.62秒）；独立审阅建议的重复文件stat干扰已修正，
首差之后仅以内存标志返回，原精度模式及strict postflight不变。

原始40份JSON/日志、输出、轨迹及汇总在
`evidence/eightcard/nal-performance-20261101-evidence.tar.gz`（264,401 byte），
SHA256为`8c20830c4dcda0707a06ec3174f261c10867c8eec772b0a1695dc57f8b574cab`。
