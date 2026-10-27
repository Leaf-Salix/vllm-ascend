# 查询路径融合与实际owner等待单因素对拍（2026-10-27）

## 当前结论

正式运行和精度基准保持`11b380e3f`，本轮所有候选均未通过稳定完整回放门槛，没有合入运行源码。nalinaly最新公开分支仍为1685da69（本轮GitHub复核）；同SDK完整三方及累计cut实测参考c346，不能改标为最新1685。

8K/B32、T192、S6、真实第4层权重＋合成hidden/history，HC-pre至HC-post完整FULL图；Native确定性level1/HCCLtrue/NZ2/static13/SK0。不是整模型token测试。此前前半段累计参考差距约293us，因此先尝试查询路径；累计cut受独立编译影响，不是可直接节省时间。

## 三方差距与优化优先级

同SDK、8K/B32/S6/T192/L4的前一轮5臂20×100完整graph均值（本轮没有重新计时Native或nalinaly）：

| 实现 | 完整event均值us | 独立L1中位us | Native数值资格 |
| --- | ---: | ---: | --- |
| Native，SK0 | 1247.861 | 非PTO L1，不作相同分解 | 确定性参考 |
| nalinaly c346 TND | 1088.158 | 928.46 | 单层输出relative L2约0.333%，非逐位一致 |
| 我方精度Direct | 1585.767 | 1502.14 | 同输入Native A/B八边界与ABA逐位一致 |

我方完整event均值比nalinaly c346慢45.73%。这不是最新1685的性能结论，也不是整模型token延迟。前一报告的累计cut参考：入口到TopK plan差约293us、plan到heads差约165us、heads到末尾差约116us；cut独立编译，不能当可加的精确纯模块成本或保证可节省时间。当前先优化最大的前段，后续再处理attention和O-proj；既不能假定所有差距由精度造成，也不能直接舍弃Native的scale合并和BF16边界来换速度。

## 代码与独立资格

- v10把Indexer Q投影→RoPE→Hadamard及主Q_B/后处理放一个MIX。QB20核、Had4核，保留原所有数学与四次72角色握手。原QA seed、原权重和所有FP32边界未改。
- v12只把第三次全体握手替换为实际owner的DSL通知/等待；保留H1/H2和末尾全体同步。每调用H1前清零，H2后等待，末尾防止下次覆盖；独占顺序调用下不需要跨调用代际。512字节单写者槽，backoff150是设备tick，不是微秒。SDK原生DSL生成，不是手改C++。
- 这两项相比Direct分别是完整任务融合候选、在该融合上的等待单因素；不能将4/20与融合的效应分开归因。主Q和Indexer保留scale先合并再乘acc的我方Native边界；所有BF16/FP16舍入保持。
- CPU64/artifact.load/NPUfalse通过；63个leaf→59个，57个共同leaf记录SHA相同，6个查询leaf被2个MIX角色取代。v12与v10相比只有2个island leaf记录SHA变化。
- 生成物和源码经过独立审查：GM视图同base、子scope退出再Score/父scope退出再Sparse、cache/scale生产依赖、所有参与者握手、FIX/MTE3排空与业务cache获取、Notify Set0/1与GE等待均核实。
- T256/B40按每tensor1024对齐、所有分支保守每depth求和：43.558/92.146/134.854/60.75MiB；v12末层多12KiB。均适配8K默认每层256MiB，不能继承旧128K末层1MiB小ring资格。不是实测峰值。

## 实测性能

首轮v10：Direct1590.449→融合1663.575us，回退4.60%；L11501.26→1565.26us，无一窗领先。task122636完成exit0，原始完整ID见TASKS。

最终v14三臂同卡：12×100普通事件、12次独立L1，分别六种排列各两次；每臂每位置4次，每对先后6:6，未剔除离群值。Native图作为精度参考建立，不把其含DFX开关的观察当新Native性能结果。

| 8K/B32实现 | event中位us | L1中位us | 对Direct耗时变化 | 本轮Native精度 |
| --- | ---: | ---: | ---: | --- |
| 原Direct精度控制 | 1597.901 | 1509.37 | 基准 | 本次无性能winner，未新增Native对拍 |
| 全体同步Query融合 | 1667.925 | 1574.13 | +4.38% | 未执行 |
| 实际owner等待Query融合 | 1665.341 | 1580.45 | +4.22% | 未执行 |

owner对全体同步仅降低约0.155%，L1反而更高，不通过预设>=0.5%/10窗胜/L1低门槛。两项不合入、不做精度扩测、未扩展128K或整模型。流程顶层pass表示测量完成，不代表性能采用或Native精度通过。任务task124652完成exit0并释放卡。

## CPU阶段历史

v1局部变量覆盖、v2未用形参metadata、v3动态主QB scratch metadata、v4～v6缓存策略作用域、v7～v8 NZ非负地址证明、v9跨scope TaskId均在CPU拒绝，未占NPU；v10完成生成资格。v11信号InOut描述符重复使用被CPU拒绝；v12通过同GM只读别名修复。v13三根CPU通过，但审查发现三臂先后不均衡，设备前否决；v14只修改计时排列并重新取得三根CPU收据，各臂实际leaf SHA分别与v10控制/v10候选/v12候选完全一致。没有改SDK/安装环境或生成C++。

## 主Q_B整数流水单因素

借鉴nalinaly1685的NZ M128/N256/K256双缓冲，原INT32 dot1024项无溢出，保留原FP32 scale合并、Norm、RoPE。控制与候选均63个leaf，只有qproj_matmul记录SHA变化，其余62项相同；真实尾行TLOAD/TSTORE、NZ偏移、生命周期、L1/L0资源均经过独立生成审查。仅INT32 scratch上取整到128，不对请求补S6。

8K/B32 task130740正常exit0：普通event中位1595.498→1599.426us（回退0.246%，0/12窗领先）；L1中位1511.86→1501.14us（降低0.709%）。完整回放门槛失败，不因单独L1较好采用，没有执行Native精度扩测。原始event12×100及L19窗都保留，不剔除样本。

## 后续定位

独立dep-only任务task130009完成exit0；没有计时/新Native精度资格。三臂保留的计时构建leaf name/funcID/SHA与dep/CPU完全匹配；编排仅局部inline数字标识符不同，严格alpha等价；compiled_meta字节相同。Artifact.load重写kernel_config写法，因此对全部配置值按AST白名单归一核对，**不声称配置/编排二进制SHA相同**。原计时未逐轮保存Worker registration到保留构建的直接见证，故下表是**条件性离线关联**：假定保留构建就是实际编排。两次dep映射稳定并覆盖全部L1 token，但不能据此声称已证明每轮执行身份。三臂各12窗如下。

| 阶段中位数（us） | 原Direct | 全同步融合 | owner等待融合 |
| --- | ---: | ---: | ---: |
| 完整L1跨度 | 1509.37 | 1574.13 | 1580.45 |
| Query完整包含跨度 | 184.75 | 229.23 | 227.33 |
| Query结束→Score开始 | 11.77 | 30.40 | 27.08 |
| Score包含跨度 | 80.81 | 81.54 | 80.29 |
| 入口→TopK plan结束 | 733.83 | 796.25 | 806.30 |

在上述映射假设下，结果指向Query及其到Score的交接变长，Score本体没有同幅度回退；它是定位线索，尚非直接注册见证下的因果证明。它没有证明延长完全来自某条MTE/轮询指令。独立中位数不能精确相加，范围仍包含同步/搬运/排队。不能拿旧nalinaly cut数字减本表做纯模块成本。依赖边和每core包含跨度、36个L1窗的映射与原始SHA见QUERY_DEPENDENCY_ANALYSIS.json。

更具体的下一步：原Direct控制保留；不继续扩大Query融合/等待候选。先测HC布局复制运行时32次循环的静态展开，仅改变布局Scalar调度；若仍无稳定收益，则改测原独立Hadamard的M256行合并或在原任务边界内优化Q反量化的载入调度，保留K128累加/所有FP32和BF16边界。不得混入失败Query岛，不把CPU编译当性能结论。

Hadamard说明：我方本轮保持M64/K128；nalinaly通用helper也是M64，但其cooperative_prefix实际调用四tokenM256。不能把‘保持M64’写成已经对齐nalinaly实际协同分支。尚无本轮M256对拍资格。

## 精度基准与使用限制

本轮失败候选都没有合入正式运行源码；正式运行仍为11b380e3f，文档HEAD可比它更新。原验收环境与原整模型8K/B40、128K/B24的Native token一致性证据保持，**不代表本轮新SDK或新候选获得整模型资格**。本轮因性能未过门槛，没有新增整模型任务。

nalinaly c346的同SDK参考单层输出relative L2约0.333%，我方原Direct同口径Native A/B8和ABA逐位一致；单层数值差不能直接推断nalinaly整模型token必然不同，也不能将所有性能差距归因于它的误差。最新1685源码借鉴与旧c346测量分开记录。

## 复现

证据目录为`evidence_tnd_precision/query_front_pipeline_20261027/`。`executed_helpers.tar.gz`提供运行过的独立源码、manifest与SDK/source绑定；`generated_cpu_evidence.tar.gz`仅发布实际生成文本、编排与收据，不含预编译二进制。各原始CPU完整归档SHA见`RAW_CPU_INPUTS.json`。NPU原始日志、JSON和每窗DFX分别归档，不将dep-gen与计时同时开启。

按前一份`DSV4_FLASH_CSA_TND_STAGE_GAP_QA_SEED_20261027.md`的SDK和3399源文件快照准备环境后，将helper解包到`nal-sdk-alignment-20261026-codex/`下相应目录。每个helper独立执行`bash compile.sh baseline`与`bash compile.sh candidate`；三臂v14另编译`barrier`。必须检查manifest、64tensor ABI、artifact.load成功、NPUfalse和精确导入路径。若绝对路径变化，重新生成所有binding/manifest并重新CPU编译，不能伪称原身份。

提交参数均为`task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 --env PYTHONNOUSERSITE=1`，运行`bash <helper>/run.sh <unique-label> 32 8192 <32个逗号分隔6>`。inner timeout750秒。全同步/owner三臂使用`query-island-v14`，Q_B使用`main-qb-pipe-v1`。脚本先保存性能，仅满足预设门槛才做Native A/B8、ABA、保护区和只读检查；即使exit0，必须核对winners和accuracy_executed字段。

依赖诊断另用`query-dependency-v1`，没有计时资格，必须先与v14保留JIT构建及编排匹配才作条件性token关联，不能替代缺失的逐轮registration见证；不得仅靠同名函数连接两次运行。历史CPU失败与无性能winner均明确保留，不将其写成数值失败。

### 离线分析依赖说明

正式仓库检查要求使用`regex`，离线脚本因此采用`import regex as re`。Python环境需提供该包。本机默认`.venv311`没有它，本次使用已有pre-commit Python3.11环境（已含regex）复算，没有安装或修改NPU环境。格式、循环闭包显式绑定和regex调整后，所有阶段指标、逐窗映射和身份结论与原标准库版本完全相同；脚本本身SHA相应更新。复现时使用`python -c 'import regex'`先确认依赖。

## HC布局复制展开与反向复测

只将T>=64的BF16 pack32次独立复制改静态展开，短T<64原FP32路径不变。两臂63leaf仅prefix Vector变化，62项SHA相同；实际编排与Cube整段字节相同，source恢复单处unroll→range后AST相同。生成物64个const TMOV恰好无重叠覆盖目标，所有cast、transpose、尾行、9次handoff及静态UB最大141056字节不变。

| 8K/B32，12×100event/9窗L1 | Direct event | 展开event | 耗时变化 | 配对胜窗 | Direct L1 | 展开L1 | Native精度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 正向捕获 | 1597.822 | 1585.941 | -0.744% | 12/12 | 1503.16 | 1490.98 | 通过A/B8与ABA |
| 反向捕获 | 1584.202 | 1598.269 | +0.888% | 0/12 | 1498.06 | 1505.50 | 性能失败，未新增 |

不能将首轮Native精度通过误称两轮稳定性能通过。候选不合入、不扩128K或整模型。两次都出现后捕获臂较快的现象，需要完全相同实现A/A核验测量协议；还没有定位是地址、缓存、注册还是其他捕获状态，不能先断言物理原因。普通event每臂顺序6:6；9窗L1为5:4，用于复刻原协议，判定首先依赖12窗完整event和反向复测。

## Q_A保留清零、按行并行：两轮结果

原实际DFX中Q_A的单AIV seed跨度74.3us，仍是前段具体对象。新的独立候选保留全部FP32零值、16×32小tile、原atomic/sentinel与K/packet顺序，只把16行块分配给最多48个AIV（T<=256时最多16个）。每行块对应64KiB连续区域；这里64KiB是相对块步长，并非绝对地址对齐声明。基于ring基址对齐，其他worker不共64B行；枚举T1..256覆盖全部ceil16行，pad行仍清零，T<16仍单worker。这与此前删除seed或新增guard的候选不同。

只冻结原3399源126/168/generic；尚未声明formal95、atomic模式、128K或整模型资格。CPU/生成/数学AST检查通过后仍须性能优先、Native A/B8、反向重复，正式完整路由与短T/混长，最后真实token。没有结果之前不称其有效。

CPU及实际生成审查已通过：63个leaf只有3个seed变化，其余60个binary SHA相同；基准/候选64参数、artifact.load成功、未初始化NPU。三个seed是每owner按16行步进、每列32步进，2KiB UB的+0.0非atomic TSTORE，pad覆盖与MTE3→V复用等待正确。三个matmul binary SHA均不变，seed/matmul对同allocation的InOut、Phi→Norm及scope结束点保持，无新增GM或scope。首轮与反向结果见下表，不能只引用首轮。

## A/A观测仪器与最终状态

HC候选出现正反捕获收益反转，因此准备完全相同Direct的A/A。冻结parent两臂的全部63leaf、编排（逐字节）、compiled_meta及全部kernel_config值相同，均64参数/load/NPUfalse。设备计时前还必须核对实际执行产物的同一身份链和八个外部状态，不将A/A差异算优化。

- v2任务task133557在计时前退出1，尚无性能/数值结论。它错误假定持久缓存开启时entry._cache仍含一个产物。SDKd449的持久缓存分支调用resolve_persistent，不填这个字典，已从代码和env确认原因。
- v3仅关持久缓存以便观察，任务task134158未获卡；为了保留原cache1口径，在NPU执行前取消，由v4替代，未重复占卡。
- v4只读Worker实际prepared specialization：按精确JITFunction对象筛选、在RLock内检查注册的READY/PID/generation/owner，取真实artifact；不load、不注册、不改缓存/SDK。随后按原严格门禁核对所有leaf、编排、metadata/config。两前端op若被Worker按artifact identity去重为一个registration，需如实记录共享，不称两个独立Worker。
- task134803（helper capture-aa-v4）等待设备后完成exit0；实际身份门禁及两前端八个外部状态相同，通过后才计时。不是Native精度重测，也没有把A/A差异作为优化。

A/A完成后已依次运行行并行候选正反两轮。仍然只有稳定获益者进入正式路由/短T混长及整模型资格，不通过就保留原11b基准。全部自有任务outer780/inner750秒；排队时间不计作NPU运行时间。

### A/A最终结果

两前端op、独立fixture/graph实际执行的63leaf name/funcID/SHA、完整编排strictalpha、compiled_meta与全部config值均匹配CPU控制；A/A的八个外部状态逐字节相同。cache1原模式保留，只读实际Worker登记信息。

| 完全相同Direct，两前端 | 首臂 | 次臂 | 次臂耗时变化 |
| --- | ---: | ---: | ---: |
| Event中位us（12×100） | 1602.391 | 1600.872 | -0.0948% |
| L1中位us（9窗，5:4） | 1514.32 | 1507.20 | -0.4702% |

本次普通event差异只有约1.52us，未重现HC的正反0.7%～0.9%现象；不能据此宣称所有捕获/地址影响已排除，更不能将L1的差异当有效优化。HC仍因反向复测失败否决。保留双向event稳定性和独立L1门槛；并行seed候选在A/A完成后分别运行正反两轮，结果如下。

## Q_A并行清零正反对拍

| 8K/B32，12×100event/9窗L1 | Direct event us | 行并行event us | event耗时变化 | 配对胜窗 | Direct L1 us | 行并行L1 us | Native精度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 正向捕获 | 1593.579 | 1583.311 | -0.644% | 12/12 | 1505.36 | 1481.74 | 控制与候选A/B八边界及ABA全逐位一致 |
| 反向捕获 | 1579.250 | 1585.386 | +0.389% | 0/12 | 1499.48 | 1476.18 | 性能未通过，未新增精度验收 |

两轮分别少23.62/23.30us的L1跨度，但完整回放收益没有稳定复现，故不采用、不组合、不扩正式95路由/短T/混长/128K/整模型。两轮各44次执行保护区和只读检查均通过；不能据此把未执行的反向Native精度写为通过。

普通event包含图回放的完整边界，独立L1是设备任务时间跨度，两者在不同调用中采集。不得用它们中位数之差断言纯AICPU开销或认为省掉的23us一定全转化为整模型收益。

独立Q_A dependency-only任务只用于核对CPU/保留构建并尝试条件性关联原L1里程碑，不新增性能或Native精度资格。原计时未逐轮保存Worker registration到保留构建的见证；即使leaf/严格alpha编排/meta/config与CPU和dep相同，模块端点仍有假设，不能精确归因。该补测在获卡前取消，未产生新设备结果；status取消后not_found，无残余任务。后续新计时先保存只读registration见证，再分析模块。

## 下一步：改观测，再测独立行合并

1. 保留正式11b精度基准，不组合本轮失败候选。
2. 在新计时中沿用本轮A/A已通过的只读Worker观测：记录精确JIT对象、READY/PID/generation/owner及artifact identity、leaf/编排/meta/config，建立实际graph与生成物的直接见证。dep-gen继续独立采集。旧计时缺失字段不补造，不写精确模块因果结论。
3. 优先在现有任务边界内借鉴nalinaly的四token Hadamard M256行合并；每行仍完整K128，保留所有BF16、FP16及量化scale边界。短T、尾行与真正TND单独检查；仅做CPU准备不等于性能通过。
4. 若完整event与独立L1两向重复稳定更快，才做Native A/B八边界、ABA、保护区/只读、完整路由和短T混长；再复测真实整模型8K/B40，最后128K/B24的token一致性。单任务outer780/inner750，不跑大配置去验证性能未过门槛的候选。
