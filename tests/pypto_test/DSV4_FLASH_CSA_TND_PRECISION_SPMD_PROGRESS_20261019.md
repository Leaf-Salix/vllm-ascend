# TND精度版的单SPMD迁移：首切片进展

## 状态与代码位置

当前分支仍为`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，没有创建或切换分支。
新增同级包`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_precision_spmd/`。
原`7d5202300`精度版43个源码文件保持不变，原服务路由和已验收基准保留。

**当前仅完成Norm→WKV单SPMD切片。尚未完成HC-pre至HC-post完整单SPMD替换，
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
还需要补充“单入口、内部维持原Norm/WKV分任务”的控制组，才能评价SPMD对当前架构的实际收益。
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

## 后续验收范围

继续在同一分支增加文件：先迁移HC-pre至WKV，再扩展QKV、压缩、Indexer、Attention、O-proj和HC-post。
每个阶段按Native同输入对拍。完整候选仍须先8K/B40，再128K/B24，复用冻结Native金值，
检查warm192加三轮192 token及调度事件一致性；每任务不超过15分钟。
后续性能需同环境、同数学边界、同输入对拍，不能将本切片耗时与nalinaly完整CSA数字直接比较。
