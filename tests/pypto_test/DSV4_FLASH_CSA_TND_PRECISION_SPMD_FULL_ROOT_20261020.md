# TND精度＋SPMD：完整CSA入口与首差定位

## 状态

沿用`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，不创建或切换分支。
原精度基准`7d5202300a1b3828ec9ecd01278f45654a51b7c5`的43个运行源码文件保持不变。
新实现独立放在`deepseek_v4_flash_csa_tnd_precision_spmd/`，原精度版仍可使用。

本次把已验证的HC-pre、输入Norm、WKV和Indexer权重投影SPMD前缀接入完整CSA顶层入口。
后续Q路径、compressor、Indexer score/TopK、attention、O-proj和HC-post仍为原精度PTO实现。
**这是一个完整CSA入口内的SPMD前缀，不是HC-pre至HC-post全链路单SPMD。**
当前已完成8K/B40与128K/B24真实整模型token验收，详见末节链接；性能尚未追平Native或nalinaly。

## ABI、常量与工作区

- 当前Torch dispatcher限制64个根Tensor参数，MIX callable限制32个描述符。
  最初72参数草稿在实际Torch注册时失败，没有执行候选设备计算。
- 去掉两个不参与数学的旧权重参数；五段原BF16 Norm权重按字节拼为`norm_packet[6272]`；
  三段原INT32归约顺序表拼为`order_packet[2560]`。没有反量化、再量化或dtype转换。
- 根参数降为64，保留十个显式工作区参数。四组读别名必须是同一Tensor对象，未绕过ABI或别名检查。
- 每模型共享工作区1,869,824字节；Norm常量每层12,544字节；33份顺序组合每模型337,920字节。
  T6保留原静态资格选择；实际选择的三元组和字节SHA单独记录。
- 较大的HC临时tensor放在前缀子scope，输出及跨阶段tensor在父scope持有。
- capture限于同一捕获stream；replay必须按设备执行顺序串行使用工作区。
  未捕获暖身可以在计算stream借用工作区，借用前等待设备、调用返回前等待当前stream。
  已验证同权重、不同输入及cache的两个完整根共享工作区；
  后续整模型验证覆盖不同层权重的串行复用；并发stream仍不支持。

无设备门禁包含program生成、kernel ABI生成、CCEC和**实际Torch注册64参数**。
program与kernel生成物具有各自哈希，不混用两类生成物身份。
prefix实际MIX并集32，24组参与者，AIC/AIV各十次HARD MIX同步，保持原数学与舍入。

## 单层FULL graph结果

使用真实模型C4层权重，合成hidden和历史cache；不是整模型token测试。
Native开启确定性；npugraph_ex实际FULL graph、`force_eager=False`、static kernel，
本轮诊断为`super_kernel_optimize=False`。不把此计时与nalinaly的SK1报告直接比较。

| 配置 | Native与新实现 | 其他验证 | 诊断计时Native／新实现，μs |
| --- | --- | --- | --- |
| L4，history8191，四请求各1行，T4 | A/B各8边界全部逐bit一致 | A/B/A、poison、cache及metadata guard、计时后复核 | mean 442.232／719.536 |
| L4，history8191，请求1～6行，T21 | A/B各8边界全部逐bit一致 | 同上；两完整根共享工作区，三轮共48项逐bit一致 | mean 581.599／866.779 |
| L26，history8189，单请求6行，T6 | 输出及6组cache/state逐bit；TopK排列155处不同 | 首差停止，未进行两根复用及计时 | 未计时 |

T4和T21物理graph行数分别为4和21；没有把每个请求padding到S6。
这两项对应service静态选择修正前的14源；修正后的当前14源在本节只有下文T6的新增实测，
不把修正前结果标成当前源的重新验收。
T4/T21计时各8个窗口，每窗口10次raw replay，保留全部样本。
未配套测量冻结完整精度PTO控制组，不能据此判断SPMD相对当前精度版的性能收益。

对应设备任务：

- `task_20261020_001536_278216460`：T4通过，exit0。
- `task_20261020_002106_302427231885`：T21及双根共享通过，exit0。
- `task_20261020_002703_32524037904`：T6首差，exit1，按预期停止扩展。

## T6首差的三臂复核

使用失败样本的同一hidden、positions和全部cache初态，逐字节核验后比较
Native、冻结完整精度版、新SPMD版。没有修改两版数学，也没有计时。
任务`task_20261020_003901_368926520195`完成、exit0；**诊断完成不等于内部精度门禁通过**。

| 边界 | 冻结精度版对Native FULL | SPMD对Native FULL | SPMD对冻结精度版FULL |
| --- | --- | --- | --- |
| 最终输出 | 逐bit一致 | 逐bit一致 | 逐bit一致 |
| 六组cache/state | 全部逐bit一致 | 全部逐bit一致 | 全部逐bit一致 |
| 有序TopK | 155个位置不同 | 同样155个位置不同 | 全部逐bit一致 |

差异只在第2号token、position8191；两侧512个索引各自唯一，完整索引集合相同，最大rank移动3位。
本次Native FULL和SPMD FULL均完整复现上次保存的8个边界。
实际SPMD顺序包选择`(1,0,0)`，SHA256为
`746192bf899c4cd4d2d70ea8f5eaa51f3a3d8b99663addc56a7ea9a5a8a3c10b`。

这证明该样本差异不是本次SPMD融合新引入的回退；不证明所有形状或所有模型层均无回退。
初次Native eager取样在静态reselect之前，与后两臂eager的环境不同，不能直接混作公平eager三臂结论。
该初次eager的TopK与两版PTO相同，只有Native FULL发生这155处排列变化。

PTO侧512个已选分数各不相同；把Native索引按PTO分数映射，155个变位位置的分数均不同，
并出现72处相邻逆序。因此没有证据将问题归为同分排序。
Native调用`return_value=False`，第二返回为空，不能把PTO或CPU重建分数冒充Native实测分数。
因此继续导出Native实际QLI kwargs及图后eager观测，按以下过程定位。

## T6静态Indexer归约选择：定位及修复

后续两项仍仅测试相同保存输入：

1. `task_20261020_004303_384258332073`：Native静态选择后的eager与FULL所有8个边界一致，
   9类QLI Tensor输入逐字节相同；冻结版与SPMD的FULL分数也逐字节相同。
   CPU重建与最小二乘仅作为定位线索，将差异缩小到head48系数，不能当作Native实测score。
2. `task_20261020_004804_402844631122`：Native独立前缀重现实际FULL的QLI权重输入；
   PTO HC mixed/post/comb及Norm均逐bit一致。再以完全相同Native Norm输出测试原投影算子：

| Indexer归约顺序 | raw BF16 | 缩放后BF16 | QLI输入FP16 |
| --- | --- | --- | --- |
| 原T6 mode0 | 仅token2/head48不同 | 同一个元素不同 | 同一个元素不同 |
| 已有T16 mode1 | 384个元素全逐bit一致 | 全逐bit一致 | 全逐bit一致 |

原PTO raw为0.578125，Native为0.57421875；缩放后分别为0.006378173828125与0.00634765625。
最早差异已经直接定位在投影BF16发布边界，不是依据拟合就认定weight或query scale。

修复只作用于新SPMD service：**完成原静态包验证及实际reselect后，T6选择已有mode1**；
其他T1～256沿用原表，未通过静态资格仍禁止执行。投影数学、舍入、score与TopK排序代码均未修改。
普通order输入与完整常量包统一使用该选择，实际顺序包变为`(1,1,0)`。
该包原先已由T16准备；CPU新增全部256种实际组合的存在性门禁。
原43源和修复前14源快照均保留，不能把旧首差结果改写为通过。

`task_20261020_005315_311308132`修复后完整FULL复测exit0：
同一L26/T6/history8189，A/B各8个边界全部逐bit一致，TopK的155处首差消失；
A/B/A、poison、cache/metadata guard、计时后复核以及双完整根共享的48项比较全部通过。
这仍是单层、真实权重、合成hidden/history验证，不是整模型token验收。

## 复现与验收边界

运行环境保持CANN9.2 beta2、Torch2.10.0、Torch-NPU2.10.0.post2、vLLM0.25.1及原PyPTO/Simpler。
无需重装依赖。编译使用checkout-local ccache和四路并行；每项调度780秒、内部timeout750秒。
所有单层任务设置`PYTHONNOUSERSITE=1`，核验实际import、PTOAS和ASCEND_HOME_PATH。

原始helper、source binding、编译receipt与各任务JSON记录在本次证据中；
helper的`run-full-csa.sh <新输出名> <逗号分隔长度> <层号> <history>`复用相同配置。
输出目录须全新，fixture固定seed1024，Native与PTO使用独立cache、相同初态。
单层例子为`1,1,1,1 4 8191`、`1,2,3,4,5,6 4 8191`及`6 26 8189`。

上述内部TopK首差已在该样本上修复。真实8K/B40首次启动遇到下节的工作区限制；
修复后重新做warm192＋三轮192 token，复用冻结Native金值和入场轨迹。
8K通过后再测128K/B24；不能沿用原精度版token结果冒充SPMD结果。

## 多bucket暖身与工作区生命周期

真实8K/B40首次任务`task_20261020_010239_45278030583`在初始化阶段退出：
16个rank均遇到第二bucket未捕获暖身的stream限制。没有生成token对拍或性能结果，
因此不能将此故障描述为精度失败。vLLM逐bucket在计算stream暖身，再在共享side stream捕获；
原SPMD工作区错误地把后续未捕获暖身也限定在捕获stream。

修复只改新实现的`workspace.py`：已经绑定捕获stream后，其他stream的未捕获调用先等待设备；
保留service原有的未捕获调用返回前同步。真正capture仍检查同一stream，捕获路径不新增同步或分配。
两个graph必须按设备执行顺序串行replay；单纯不同stream上按Python调用先后入队不足以保证安全。
不支持并发模型调用、并发graph replay或跨stream同时使用同一个工作区。

独立验证复现了T21捕获→另一计算stream暖身T6→同一捕获stream捕获T6→交替replay：

| 阶段 | 实测结果 |
| --- | --- |
| T21，请求1～6行，完整Native/PTO FULL graph | A/B各8边界全部逐bit一致，实际21行、每请求padding为0 |
| 另一计算stream暖身T6 | 8边界全部逐bit一致 |
| 第二个T6 FULL graph | 实际捕获成功，绑定stream不变 |
| 三轮T21→T6→T21，共9次replay | 每次8边界，共72项全部逐bit一致；工作区及输出poison后仍通过 |

`task_20261020_012615_16133544094`为通过结果。
前一诊断任务`task_20261020_012013_134550819657`曾在交替replay首差停止，原失败记录保留：
Native RoPE使用共享runtime buffer，创建6行fixture会覆盖原21行fixture的前缀。
实际地址与storage相同，cos改变314个、sin改变320个元素。
修正验证脚本，在每次replay前把**当前fixture原先保存的只读输入**恢复到原地址，并检查输入guard；
没有重新生成gold或修改正式数学。恢复后交替replay通过。

这一轮编译receipt绑定当前14个源码；与前节T6通过时相比，仅`workspace.py`发生变化。
旧T4/T21/T6结果继续绑定旧workspace，证据中保留其原文件，不将旧硬件结果重标为新源。
新增证据在`evidence_tnd_precision/spmd_workspace_lifecycle_20261020/`，
包括首次初始化失败摘要、两次生命周期结果、原始helpers、旧workspace与当前编译receipt。

该门禁仍为单层真实权重、合成hidden/history，不代表整模型token或性能验收。

## 后续整模型验收

当前14源随后完成8K/B40与128K/B24真实整模型验收，全部保存token及接受事件与Native金值一致。
详细配置、性能限制及可离线复算的原始证据见
[整模型报告](DSV4_FLASH_CSA_TND_PRECISION_SPMD_REAL_MODEL_20261020.md)。
前文各单层结果和首次失败仍保留原验证范围，不用整模型通过覆盖或改写失败记录。
