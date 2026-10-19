# TND精度版的单SPMD迁移：Norm/WKV与HC前缀进展

## 状态与代码位置

当前分支仍为`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，没有创建或切换分支。
新增同级包`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_precision_spmd/`。
原`7d5202300`精度版43个源码文件保持不变，原服务路由和已验收基准保留。

**当前完成Norm→WKV切片及HC-pre→Norm→WKV前缀的单SPMD实现。尚未完成HC-pre至HC-post完整单SPMD替换，
没有新SPMD版本的整模型token验收或完整CSA性能结论。**

## 设计

参考nalinaly单SPMD的MIX与角色分工形式，数学仍采用本分支的Native对齐实现。
24个AIC执行原WKV，48个AIV执行原Norm和BF16舍入；外层只有一次SPMD/Mixed提交。
全部参与者执行三次HARD MIX屏障，包含没有业务工作的核。

- Norm保留4096至64的fold、原平方和、INT64开方纠正、Scalar除法和最终两乘/BF16舍入。
- WKV保留M16/M64选择、逐列K256标定次序、FP32累加和最后BF16 rint。
- 只有连续BF16权重加载的等价形式变化：先load再Tile reshape，避免incore把GM reshape转成Tile后再次load。
- scratch由调用者持有；`accum_read`必须为`accum`同一个NPU Tensor对象，其他业务buffer与FFTS互不重叠。
- `normalized`为跨AIV写、AIC读的InOut工作区；通过kernel模式ABI检查，未绕过检查。

当前SDK没有nalinaly使用的signal_notify/signal_wait接口。首版使用现有DSL全核同步，
还不是nalinaly的细粒度通知和阶段重叠流水。

## 已完成验证

| 项目 | 结果 | 限定 |
| --- | --- | --- |
| CPU数学AST、T1至256所有权及原证据门禁 | 274项通过 | 不执行浮点或NPU |
| program及kernel模式无设备编译 | 通过 | 原PyPTO/Simpler/PTOAS环境，未重装 |
| 实际编排 | 一次MixedKernels={0,1,1}，24组，sync_start | 两路AIV共48worker |
| 同步生成物 | AIC/AIV各三次SYNCALL(Mix)，drain/cache/fence保留 | 无FREE_VAR，AIV在H2后读accum_read |
| T21，真实C4层权重 | Native Norm、WKV输出逐bit一致 | Norm输入为合成BF16，不是完整模型 |
| T21，A/B/A，各三次replay | 全通过，53项比较全逐bit | rstd对冻结control；accum仅检查finite |
| T21，250次计时replay后复核 | 全通过，guard正常，输入与权重不变 | 直接ACLGraph，不是vLLM FULL |

硬件任务`task_20261019_180515_142320222974`，运行1分1秒，exit0。
候选切片replay的五窗口为110.929、108.844、110.756、110.338、109.180微秒，中位数110.338微秒。
这组只有候选计时，不能据此判断相对Native或原PTO的加速。

前两项任务分别45秒、58秒在候选启动前退出：诊断脚本遗漏原service已有的NZ至ND布局整理；
以及kernel模式ABI发现normalized从Out变为InOut。均已修正，后者已加入无设备预检。
这些失败不能算作设备精度失败，也不能算作通过。

## T1与T240增补：精度和局部性能

三个T均为真实C4层权重和合成Norm输入；Native Norm/WKV BF16输出、冻结control的rstd均逐bit一致。
T1、T21各53项比较，T240含三路计时窗口后的额外对拍共89项，全部通过。

| T | Native中位数μs | 独立两入口PTO中位数μs | 单SPMD中位数μs | 相对独立两入口PTO |
| --- | --- | --- | --- | --- |
| 1 | 未计时 | 未计时 | 114.556 | 不下结论 |
| 21 | 未计时 | 未计时 | 110.338 | 不下结论 |
| 240 | 35.674 | 227.441 | 137.664 | 延迟降低39.47% |

T240同卡、同输入、串行三路计时：六个窗口，每路每窗口50次replay，轮换执行顺序；
没有剔除样本。每个窗口后三路都再次对拍，最后复核输入、权重、rstd和guard。
六组SPMD均低于同组独立两入口PTO，但仍约为Native延迟的3.86倍。

**这个对照分别调用独立注册的Norm和WKV，包含两次PTO入口，而SPMD只有一次。**
因此39.47%包含入口合并收益，不能解释为纯核内SPMD收益。当前完整精度版已具有单一PTO顶层入口，
当时尚缺“单入口、内部维持原Norm/WKV分任务”的控制组；下节补测后再评价SPMD对当前架构的收益。
表中的“原精度PTO”以此明确更名为“独立两入口PTO”，原始数据保持。

| 路径 | min μs | mean μs | max μs | 六个原始窗口μs |
| --- | --- | --- | --- | --- |
| Native | 35.530 | 37.547 | 46.884 | 46.884,35.544,35.560,35.974,35.788,35.530 |
| 独立两入口PTO | 223.524 | 230.727 | 251.584 | 251.584,225.411,229.472,224.380,229.988,223.524 |
| 单SPMD | 135.132 | 137.386 | 139.277 | 138.434,139.277,138.252,135.132,137.075,136.148 |

T1任务`task_20261019_180646_148893211419`；T240任务`task_20261019_182257_239318924508`，均exit0。
T1的Done时长15m26s包含此前约14分钟排队，执行日志区间约1分钟；T240为53秒。
两者调度max-time=780、内部timeout=750秒，不能把含排队时长当作占卡耗时。
T1包含大量零工作参与者，T240覆盖多波次和M64尾块；两者均不改变原数学。
这里仍只是Norm→WKV切片，不是完整CSA、vLLM FULL或真实整模型性能。
不能将39.47%推算到HC-pre至HC-post，也不能用这组数据比较nalinaly完整CSA速度。

## 匹配单入口后的T240性能复核

为消除入口数量的混杂，新增一个单PTO顶层入口，内部仍调用冻结版Norm和WKV分任务。
真实C4层权重、同一合成输入、同卡串行四路ACLGraph；八个窗口，每路每窗口50次replay，
循环移动执行顺序，每个路径处于每个位置各两次，保留全部样本。
计时前四图分别A/B/A换输入和输出poison，每窗口后对实际Native复核；176项比较全部逐bit通过。
PTO输出guard、输入、权重、rstd最后再次复核。

| 路径 | min μs | mean μs | max μs | median μs |
| --- | --- | --- | --- | --- |
| Native | 35.047 | 35.172 | 35.367 | 35.108 |
| 独立两入口PTO | 223.228 | 225.095 | 227.860 | 224.644 |
| 单入口、原分任务PTO | 165.219 | 168.879 | 172.186 | 168.947 |
| 单SPMD | 144.743 | 148.930 | 151.470 | 150.120 |

单SPMD比匹配单入口control的中位延迟降低11.14%，八个配对窗口均更低。
差值仍包含control内部临时accum分配与SPMD外部持久scratch的差别，
是完整切片架构收益，不能全部归因于MIX调度本身。
候选仍为Native的约4.28倍，不能据此称整个CSA已追平Native或nalinaly。
本任务为`task_20261019_190530_41913214606`，exit0。

## HC-pre→Norm→WKV前缀

同一分支新增`hc_prefix.py`与`hc_prefix_stages.py`，原43源及首切片四源不变。
单24组SPMD内，48个AIV负责widen、HF32 hi/lo位整理、split归约、门控/Sinkhorn、mix、Norm和BF16舍入，
24个AIC负责原22 split平方和、HC线性及WKV。两角色各执行十次无条件HARD MIX同步。
依然只用现有DSL，不修改SDK，不使用手写C++helper。

保留原HF32位转换、hi/lo的K16交织、22路split及各自K次序、M16/M64选择、
split0至21归约顺序、SQRT/DIV、门控和四lane相加，以及已验证Norm/WKV舍入。
三组跨角色工作区采用同一Tensor对象读别名，其他buffer独立；32B对齐guard检查越界。
HC前缀原十阶段各函数与冻结实现进行了数学AST对照，T1至256所有权/尾块容量及29参数ABI检查266项通过。

| 配置 | 实际源码 | Native对拍 | 其他检查 | 候选median μs |
| --- | --- | --- | --- | --- |
| T21 | 报告内草稿；与迁入正式包的14函数AST一致 | HC mixed/post/comb、Norm、WKV逐bit一致 | A/B/A各三次replay，103项比较全通过 | 257.540 |
| T1 | 当前分支正式包 | 同上，全部逐bit一致 | 103项比较全通过，包含大量零工作核 | 246.881 |
| T240 | 当前分支正式包；program/kernel重新编译 | 同上，全部逐bit一致 | 103项比较全通过，多波次及M64尾块通过 | 386.500 |

T1为`task_20261019_191025_63335019606`；T21为`task_20261019_184715_349642614097`；T240为`task_20261019_190852_55859521334`，均exit0。
这里的HC post/comb为Native公开FP32输出；rstd只对冻结control。
没有声称每个FP32内部partial已与Native逐bit对拍；计时只含候选，不能从此表判断前缀相对加速。
使用真实C4权重和合成HC输入，不是完整CSA、vLLM FULL或真实整模型token验收。
真实T保持不变，M16/M64工作区尾块容量不等于把每个请求padding到S6。

## 环境与复现

227/A3，CANN9.2beta2、Torch2.10.0、Torch-NPU2.10.0.post2、vLLM0.25.1、vLLM-Ascend0.25.1rc1。
沿用已验收PyPTO/Simpler环境和原真实模型权重，不下载或重装框架。

源码、编译身份和完整53项结果在`evidence_tnd_precision/spmd_slice_20261019/`：
`manifest.json`绑定四个压缩JSON及两个工具归档。
工具包保留真实运行脚本、无设备编译助手、43源基准SHA和离线审计器。
运行必须使用对应227实验环境；脚本内记录当时使用的具体路径，不是通用安装包。

1. 解开tools，核对基准43源SHA、候选源码SHA及实际PyPTO/Simpler导入路径。
2. 上传独立实验目录，运行compile-no-device.sh；program与kernel两种签名检查都须通过。
3. 通过task-submit单卡运行run-slice.sh，tokens=21。调度max-time=780，内部timeout=750秒。
4. 运行audit_slice.py，传result.json和当前候选源码目录。首差保存张量并立即结束。
5. T1和T240分别使用新任务，首个差异未修正前不扩大配置。

### 本轮前缀与匹配性能的复现

补充证据在`evidence_tnd_precision/spmd_prefix_20261019/`，manifest绑定八个归档：
原草稿及正式包编译receipt、T1/T21/T240前缀结果、匹配四臂结果、control编译receipt和tools。
T21结果绑定原草稿两源SHA；工具归档保留草稿，CPU测试验证其14函数完整AST与正式包一致。
T1/T240结果直接绑定当前正式包六源SHA。两类结果按真实provenance保留，没有改写源码身份。

1. 核对manifest、冻结43源、六个SPMD包文件及SDK导入路径；沿用上述227环境。
2. 运行tools内`compile-prefix-packaged.sh`，program/kernel ABI及CCEC必须成功。
3. 分别单卡提交`run-prefix-packaged.sh <独立输出名> 1`和`... 240`，每任务max780、内部750秒。
4. 原草稿T21用tools保留的草稿及原四源快照复现；若用当前六源，重新编译正式包后提交tokens21。
5. 四臂性能：先`compile-control.sh`，Norm/WKV切片使用首切片源码与编译receipt，
   再单卡`run-slice-matched-perf.sh <独立输出名> 240`。两工具均在tools归档。
   新增前缀文件后，应重新运行切片无设备编译刷新包身份，不能绕过源SHA门禁。
6. 本轮首差即保存张量并退出；A/B/A各三次replay、poison、guard及计时后复核不可省略。

所有本轮NPU任务均已完成；未留排队/运行任务，单个设备执行均在15分钟以内。

## 后续验收范围

继续在同一分支增加文件：已迁移HC-pre至WKV，接下来扩展QKV、压缩、Indexer、Attention、O-proj和HC-post。
每个阶段按Native同输入对拍。完整候选仍须先8K/B40，再128K/B24，复用冻结Native金值，
检查warm192加三轮192 token及调度事件一致性；每任务不超过15分钟。
后续性能需同环境、同数学边界、同输入对拍，不能将本切片耗时与nalinaly完整CSA数字直接比较。
