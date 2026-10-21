# TND精度版：T168的QA首差定位与修正（2026-10-21）

## 结论与范围

**已通过单层完整CSA修复验证；新增真实整模型8K/B28的token结果仍待运行。**
最低交付标准为真实模型逐token一致；内部逐bit比较用于找到误差来源，不能用小relative L2推断token一致。

活动分支为`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
原数值基准`48c58ce7c467dfce63d5cb5c41afd225488d6012`及原验收环境封存，
本次在`1bc33a4b4c06861de208f5add7fa5db4b338a862`上仅新增T168的QA路由。
完整HC-pre到HC-post仍为纯DSL单一顶层入口、64个Tensor参数，保留真正TND；没有增加Native计算旁路。

| 阶段 | Native/PTO对比结果 | 可支持的结论 |
| --- | --- | --- |
| 原128K/B28、T168失败例 | 输出row86/103共1731元素不同；TopK row86有88位置重排，六类cache/state相同 | 存在内部数值差异，不能据此判断真实token失败 |
| 13阶段同输入观察 | HC-pre与输入Norm相同；最早观察到QR scale差异；Q、Indexer query与heads随后出现差异 | 首差区间在QA投影至QR归一化之间 |
| 新增QA BF16边界观察 | row86/484、row103/331各1元素不同；输入Norm逐位相同 | 差异已经进入QA BF16边界，不能只归因于TopK排序 |
| Native QA次序识别 | 独立Native QA完整168×1024逐位重现；有限候选族中的packet次序唯一匹配 | 固定环境、T168可复用既有T126的QA累加顺序 |
| QA单因素候选 | 实际输入及两组独立扰动下，旧路径分别24/37/37元素不同；packet路径全部逐位一致 | 仅改变QA累加包次序即可修复这三组QA输入 |
| 完整CSA修复候选 | 原输入A、扰动B及恢复A，输出、TopK及六cache/state共8边界全逐位一致 | 本例完整替换的原差异已消除 |
| 新真实整模型8K/B28、192 token | 待Native1重复与PTO对比 | 尚无新增token资格，不能从单层结果外推 |

完整CSA候选使用真实第4层C4权重、合成hidden/history，history=131072、B28、各6行、T168、page32、NZ2、seed1024。
FULL graph、compiled/raw一致、输出毒化恢复、实时输入A/B/A、每臂33项guard及只读检查全部通过。
Native A的8个外部边界仍与原未插桩失败快照逐byte相同，避免观察器改变基准。
本轮没有执行计时，不能据此更新性能表。

## 根因与修复

原路由仅在T126使用已识别的Native QA packet顺序，T168仍走普通K256顺序。
本例HC-pre与Norm输出一致，但QA在BF16舍入边界附近对累加顺序敏感：
row86差`2^-12`，row103差`2^-13`。后续QR归一化、动态量化及投影把该差异传播到heads和最终输出。
row103的TopK完全相同，因此排序不是全部输出误差的解释。

Native次序识别使用18组有限输入及重复运行，在44种候选次序族中识别每个输出分组。
本例10752个`(row, N16)`单元全部匹配既有packet规则：K512包内两段K256按升序处理；
前9组依组号轮转包起点，奇数组反向排列包；后续组保持自然K256次序。
另行展开既有QA函数的dense及tail两条AST，均与识别次序相同。
这是当前硬件、SDK、layout和T168的实测资格，不是对任意shape或所有可能归约算法的通用证明。

正式修改位于`spmd_bf16layout/qkv_precision126.py`：
新增常量`QA_NATIVE_PACKET_ROWS_B28=168`，让T168调用现有`q_proj_qa_126`；函数计算体不变。
T126及其他T保留原路由，未改变BF16物化、Norm、量化、TopK tie规则或cache寻址。
独立候选与正式补丁在常量展开、忽略文档字符串后AST相同。

当前固定SDK的orchestration codegen不支持`Or`表达式，因此使用现有DSL的`if/elif`。
第一版合并条件在无设备编译时失败，已保留失败证据，未修改SDK或生成C++。
通用lint会合并相同分支，故沿用邻近DSL代码的局部`SIM114`说明；未全局关闭lint。

## 运行记录

下表为调度任务身份，时长按实际设备窗口记录；无设备编译单独标出。

| 任务 | 操作 | 结果 | 设备运行时长 |
| --- | --- | --- | --- |
| `task_20261021_125130_324509630936` | 13阶段观察及原失败快照不变性 | 通过，取得首差区间 | 2分47秒 |
| `task_20261021_154002_401558724012` | QA观察器实际编译 | 通过 | 不占卡 |
| `task_20261021_154359_99322825833` | QA BF16观察 | 通过，找到两个首差坐标 | 2分51秒 |
| `task_20261021_160533_289197318338` | Native QA次序识别 | 通过 | 22秒 |
| `task_20261021_161901_404552416248` | QA单因素两臂实际编译 | 通过 | 不占卡 |
| `task_20261021_171911_31598208352` | QA base/packet对拍 | packet三组全逐位一致 | 59秒 |
| `task_20261021_172356_366977112788` | 完整候选合并条件编译 | 失败，SDK不支持Or | 不占卡 |
| `task_20261021_173347_55947617398` | 等价if/elif候选编译 | 通过 | 不占卡 |
| `task_20261021_173741_102482026407` | 完整CSA A/B/A | 8边界全逐位一致 | 3分11秒 |
| `task_20261021_175203_26387018485` | canonical完整根及preparer编译 | 通过；最终注释版另建v2，不覆盖本收据 | 不占卡 |
| `task_20261021_175514_302617310470` | 最终v2 canonical实际编译 | 根64参数、preparer2参数、GCC/CCEC均通过 | 不占卡 |
| `task_20261021_175759_337840512052` | 新bank及实际import/hook预检 | 通过，前后未初始化NPU | 不占卡 |
| `task_20261021_180516_970266700` | 新8K/B28 Native1 | 已提交，尚在排队，未取得token结果 | 尚未占卡 |

所有NPU任务仅经`task-submit`，调度上限780秒、实际入口内层750秒。
排队不占卡；表中后续任务只用status/log观察。前序曾误用`--timeout 45 --wait`取消排队任务，
已披露并改交125130任务，详见逐token补验准备报告；未将其记为精度失败。没有重启共享服务或操作他人任务。

## 新整模型验证方案

沿用[逐token补验准备](DSV4_FLASH_CSA_TND_TOKEN_ACCEPTANCE_PREPARATION_20261021.md)的真实模型方案：

- 完整DeepSeek-V4-Flash-0731-w8a8，75分片，21个CSA完整PTO替换、20个HCA保持Native。
- fixed8192 bank前28个不同请求，只读复用原target/MTP初态，不重新prefill。
- TP1、DP=EP16、batch/max_num_seqs28、NZ2、DSpark出5验6、greedy、ignore EOS、确定性level1/HCCL、EPLB关闭。
- FULL_DECODE_ONLY capture `[6,168]`、22000 blocks、token预算400；每臂warm192＋正式192，共172032个token。
- Native1在同次加载内比较两轮，86016对位置；Native稳定后PTO逐轮对各自Native比172032对位置，不追加Native2。
- 全16rank、两轮均须有真实28活跃/T168/FULL168及联合DP向量。仅有bucket168不算覆盖。
- token、接受事件及完整生成窗口输入/入场轨迹分列；出现首个token差异立即停止并定位，不跑后续大配置。

新canonical快照从原3399文件逐SHA独立复制，仅QA文件变化；77个运行文件也仅这一项变化。
完整根和preparer在实际新路径编译，随后验证真实模块import来自新快照。
原22份助手、旧snapshot、旧gold均保留；新Native1/PTO使用同一新source/helper身份。
3399清单中的旧历史commit/patch字段只解释父快照，活动候选另由新manifest和实际文件SHA绑定。

## 固定环境与复现

固定环境：Torch 2.10.0、Torch-NPU 2.10.0.post2、vLLM 0.25.1、原隔离PyPTO 0.1.0及对应Simpler，
CANN 9.2.0-beta.2、匹配ABI的ATB、既有PTOAS。具体import路径、版本和SHA以原始收据为准。
没有重装SDK、vLLM或复用另一套新工具链。

证据目录为`evidence_tnd_precision/b28_qa_alignment_20261021/`。
归档仅含诊断源码、JSON收据、review和复现工具，不含模型权重、cache payload、Tensor快照或二进制。
`SHA256.json`绑定原始JSON。完整原失败PT、Native QA权重标定文件仅保留在远端自有实验目录，不能上传Git。

复现需要既有227封存基准资产及原模型/bank；不是脱离这些输入的独立环境安装包。
使用归档内同名目录恢复工具，完整诊断按各README及compile/run入口顺序执行；
不能跳过原快照不变性、QA激活收据、actual compile或source身份门禁。
新整模型入口为独立`whole-b28-qa168-v2/run-arm.sh native`，Native READY通过后才执行`pto`。
每臂通过调度器提交16卡，显式传入封存环境的`PYTHONNOUSERSITE`、CANN、PTOAS、CCACHE_DIR和四路编译参数。

最终v2源码manifest SHA为`48ee0351ebd554c2d52c93d28ff7aa4be357cf6636a4325d247f64a1f2671539`；
31文件helper SHA为`5aec62d22523f9433381f1c9f362cfaba4f4d301ca3099db1f90133b7c50e23b`。
本报告会补记整模型实际结果；在结果落盘之前，新token与性能资格均为未知。
