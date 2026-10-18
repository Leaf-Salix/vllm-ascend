# 动态CSA：Hadamard分支与Indexer消费者优先

2026-10-18。代码在正式 `ops/pypto/deepseek_v4_flash_csa_spmd` 开发，
候选PROJECTION_BRANCH_GROUPS=4；完整实测无收益后已恢复正式代码。
本目录保存逐次快照、脚本和证据，CPU复现入口现指向候选快照。
继续完整动态单SPMD七档降低30%目标，不修改验收分母。

## 从上一轮证据改变什么

上一轮4辅助核候选完整L1为947.16μs，同轮多task880.06μs，失败未采用。
配对阶段时钟显示：主压缩投影提前约105μs，却没有使consumer提前；
4核KV投影晚124μs，AIV先处理KV再量化Indexer，把约75～80μs延迟传到Score。
Q_B就绪中位数基本不变，因此不先盲扫4/8核或猜测Q_B负载是主要原因。
证据见相邻 `projection_branch_evidence/PHASE_CLOCK_RESULT_V1.json`。

本候选保持所有原投影数学、Native布局、归约和核间发布协议，调整为：

1. AIC的head权重与KV仍按原24核条带计算和发布。
2. 随后前4核只做query/compact Hadamard，后20核做主Q_B，两核集合独立推进。
3. 各核完成自己的分支后做原owner的主压缩，不添加全核集合。
4. AIV把Indexer pool/commit/norm放在query RoPE前，保留共享state读完集合。
5. AIV完成query量化、head系数与key/scale发布之后，再处理原KV、主Query和主压缩。

这样减少不相关KV链对Score输入就绪的制约；同时保留KV与主Query输出在Attention
之前完成。这里只是待验证机制，不把旧插桩端点差直接当作本版可节省时间。

T/B/S、实际device Q/L、position、页表和compact计数继续动态。分组4/8是核资源
参数，不是B16/S6白名单；Hadamard M256按4个token×64head组成，尾部有效行
动态裁剪。候选默认不启用，其他档位的适用与收益由后续验证决定。

## 已验证及本轮任务

- `check_prefix_control.py`：直接执行正式源码的Scalar分支/等待表达式，450个
  T/S/compact/核分配组合通过唯一发布、等待来源、无环及Q_B128列块覆盖。
  不执行数学、缓存或FFTS，不作为性能证据。
- `cpu_v1/result.json`：完整无样例shape的符号编译/PTOAS通过，单MIX与sync_start；
  入口仍保留动态轴及原运行时Tensor接口。未修改依赖或共享安装。
- `DEVICE_SUBMISSION_V1.json`：提交128K/B16/S6，原分task、原正式动态SPMD、
  新候选三者同进程/同根地址，先A→B→A及图重放/保护区检查，再五窗完整L1。
  不重新测Native或历史静态。提交不代表功能或性能通过，终态另补。

本轮34份候选源码在 `source_candidate`，33份原版在 `source_reference`。
候选已停用；后续优化仍作用正式动态源码，测试快照仅用于还原本次实验。

## 终态：没有收益，不继续分核微调

任务 `task_20261018_123227_5932334047` exit0。A→B→A的输出、Top-K/score、
完整cache/state、全部保护区、代际及图重放通过。完整L1五窗：

| 实现 | 样本（μs） | 中位数（μs） |
| --- | --- | ---: |
| 原多task | 880.14 / 894.98 / 862.62 / 869.44 / 878.08 | 878.08 |
| 原正式动态SPMD | 890.16 / 878.64 / 870.08 / 873.84 / 886.46 | 878.64 |
| 本候选 | 892.24 / 881.98 / 877.04 / 882.20 / 888.78 | 882.20 |

比同轮正式动态版慢0.41%，比多task慢0.47%，没有性能收益。不用上一轮947.16
与本轮882.20的跨轮差值宣称优化，也不继续扫核数或扩跑七档。
核对源码与被测快照后，只恢复本轮正式修改，候选完整留档。

用户要求抓大放小，后续转向Score约230μs和Attention约160μs的大阶段，
先取得影响供数链的硬件利用率证据，不把已有小参数负结果换名字重测。
