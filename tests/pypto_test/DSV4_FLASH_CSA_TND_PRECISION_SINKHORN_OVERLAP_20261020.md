# TND 精度版：Sinkhorn 与投影重叠实验（2026-10-20）

## 结论与版本

在当前分支 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex` 上新增独立候选
`pkg:deepseek_v4_flash_csa_tnd_precision_spmd_overlap`。旧精度基准 `7371bf0e2b2b26def28e5e98965fa96b9ef1bdbf` 的71份运行源码逐SHA保留。

本轮只调整 Sinkhorn 的执行时机，计算和舍入顺序保持不变。最终源版本的单层 FULL graph 逐位验收通过；
真实整模型8K/B40、128K/B24的全部输出token和接受事件均与各自Native基准一致。

收益幅度有限：前缀单独测量下降约3%～6%，完整CSA的长窗口配对均值仅下降约0.2%。
整模型表中是跨任务实测变化，不能据此把全部差值归因于该调度修改，也不能认定速度已经对齐nalinaly的完整SPMD。
本候选单独保存，旧基准及默认路由未切换。

## 参考了什么，保留了什么

参考nalinaly的固定提交 [beaefa6f6788a9448d49293238f6818c246f24f5](https://github.com/nalinaly/vllm-ascend/commit/beaefa6f6788a9448d49293238f6818c246f24f5)，
特别是 [vector_schedule.py](https://github.com/nalinaly/vllm-ascend/blob/beaefa6f6788a9448d49293238f6818c246f24f5/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/vector_schedule.py)
先发布Norm、让AIC启动投影，再计算仅供HC-post使用的Sinkhorn的依赖安排。

| 项目 | 旧精度版 | 本轮候选 |
| --- | --- | --- |
| 前缀次序 | pre/post → Sinkhorn → mix → Norm → WKV/Indexer权重投影 | pre/post → mix → Norm → Sinkhorn与两种投影并行 |
| 同步 | 每角色10次handoff | 同步数和H8/H9发布/等待语义均保留 |
| Native数值边界 | FP32 HC/Norm、Native舍入、QA126顺序、WKV/Indexer/O-proj精度修正 | 全部复用，不重新改写 |
| 替换范围 | HC-pre到HC-post，纯DSL，一个64Tensor顶层入口 | 相同；本轮SPMD优化仅在前缀，后缀仍按原任务执行 |
| 请求布局 | 真TND，真实请求1～6行 | 相同，不要求每请求补齐S6 |
| 加载与生命周期 | 不可变权重、加载期PTO预整理、共享cache和跨流合同 | 相同 |
| 源码组织 | 原71源 | 原71源加6个独立入口/接口文件 |

代码入口为 [prefix.py](../../vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_precision_spmd_overlap/prefix.py)。
AST审计确认：Vector函数内只移动了一次完全相同的comb调用，其余语句顺序不变；Cube和所有数学helper继续使用冻结实现。
Norm写vec_ws主区48～55列，Sinkhorn读8～23及40～47列；尾部暂存与投影输出也不重叠。
H8发布normalized，H9同时等Sinkhorn写回与Cube投影完成，保留原缓存/流水屏障。

## 单因素实测：前缀

同卡、相同真实C4层权重、合成HC输入，独立同尺寸缓冲和原始ACLGraph。
每档12个交替顺序窗口，每窗口25次回放；单位µs，表为窗口中位数，不剔除样本。
这一步不包含请求metadata或完整CSA。

| T | 原精度版 | 新候选 | 延迟下降 | 更快窗口 | 精度 |
| --- | --- | --- | --- | --- | --- |
| 21 | 227.070 | 219.786 | 3.208% | 12/12 | 536 项逐位通过 |
| 126 | 296.770 | 279.110 | 5.951% | 12/12 | 536 项逐位通过 |
| 240 | 351.552 | 337.614 | 3.965% | 12/12 | 536 项逐位通过 |

检查包括对Native的mixed/post/comb、Norm输出、WKV和Indexer三输出；rstd与冻结PTO控制实现对拍。另验A/B/A、scratch毒化、只读权重与外侧guard。

## 完整CSA FULL graph：最终源码与长窗口

同一任务内依次捕获Native、旧精度版serial、新候选candidate，正反顺序交替计时。
最终协议为12窗口×100次回放；单位µs，所有样本均保留。
单层Native明确使用SK0，此表用于精度与本次调度的配对比较，生产性能应结合后面的真实整模型协议。

| T | Native SK0 | 原精度版 | 新候选 | 相对原精度版下降 | 更快窗口 | 精度 |
| --- | --- | --- | --- | --- | --- | --- |
| 126 | 1090.497 | 1584.199 | 1581.411 | 0.1760% | 11/12 | 八边界及跨桶通过 |
| 240 | 1378.817 | 2211.700 | 2206.889 | 0.2175% | 12/12 | 八边界及跨桶通过 |

- T126：36个请求，`[1,2,3,4,5,6]×6`；真实及物理图行数均126，每请求补S6将变216行，本轮没有该补齐。
- T240：40个S6请求；实际及物理图行数均240。
- 八边界：最终输出、TopK、SWA、压缩KV、主state、Indexer两组cache及Indexer state。
- 两种输入A/B分别对Native逐byte，恢复A后再次验证；禁止Native回退，实际FULL捕获、compiled/raw一致、guard通过。
- 新候选另外通过T6、不同history、跨流warmup与共享workspace跨桶回放，共80项边界检查。此80项只授予candidate。
- 早期8窗口×10次探索曾测得约1.10%/0.45%收益；最终长窗口值较小，应使用本表。旧记录保留在原始包中。

## 真实整模型：精度与性能

沿用[nalinaly低接受率测试](https://github.com/nalinaly/vllm-ascend/blob/dsv4-flash-pto-v0.25.1rc1/tests/pypto_test/low_acceptance_20261001/RESULTS.md)
对应的现有本项目复测协议：`DeepSeek-V4-Flash-0731-w8a8`完整75分片权重、TP1、DP=EP16、DSpark出5验6，
FULL_DECODE_ONLY、Native静态包，确定性level1和HCCL确定性开启，EPLB关闭。
每轮每请求192token，先warmup192，再计时3轮192；计数包含warmup和3轮，forward/cycle来自计时轮。
每步先取16rank的最大耗时，再求所有轮次均值；forward共60点、cycle共57点，单位ms，未删除样本。
使用本项目固定bank和已验收Native gold，没有声称与nalinaly原bank的cache字节相同，也未重新运行Native2。

| 配置 | 对Native的token / 接受事件 | Native历史forward ms | 原精度版历史forward ms | 新候选forward ms | 相对原精度版实测下降 | 新候选cycle ms | 比Native历史forward慢 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8K/B40 | 491,520 / 105,152 全一致 | 89.4822 | 107.0902 | 106.4669 | +0.582% | 122.0472 | +18.981% |
| 128K/B24 | 294,912 / 61,568 全一致 | 79.7299 | 92.2142 | 90.7129 | +1.628% | 103.0798 | +13.775% |

两档均16个rank完整通过、零抢占，21个CSA层真实捕获PTO根，20个HCA层保持Native，64Tensor ABI。
8K每卡40条并发；128K每卡提交24条，测量窗口实际21条并发，允许等待入场。
128K实际126行与全局144行Graph档位共存；全局bucket填充不等于按请求补S6。

8K的全部测量相位CPU请求key、边界、position、input ID、DP token数对齐Native。
128K沿用已验收的Native相位映射，严格检查入场及工作量；CPU复用缓冲中的已知0/-1哨兵差异单独列在audit中，不能将其表述为所有GPU输入逐位证明。
整模型验收证明的是本协议下token和接受事件一致；未hook所有层的所有中间Tensor，也未覆盖任意模型、任意batch或EPLB开启配置。

## 软件与源码身份

- Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、CANN9.2.0-beta.2；沿用原SDK与二进制。
- PyPTO `3e87a843`、Simpler `a54c0509`；实际import、toolchain、CANN/ATB环境由每次preflight核对。
- 新快照3399文件，运行数学/绑定源码77份；旧71源与已验收父版本逐SHA保持。
- 快照manifest SHA256：`b229b29c69dc4aec637f99d586c1b68bafc9cc28cec9ebae2f54483baf1cdd4c`。
- 六份新源码同时匹配最终FULL收据、整模型snapshot和项目文件；原始包包含全部77份运行源。
- 11项CPU观察器/资格测试通过；独立复核直接重算原始token/事件、几何和性能，不仅依赖audit.pass。

## 复测方法

原始证据及完整脚本位于 [evidence_tnd_precision/spmd_sinkhorn_overlap_20261020](evidence_tnd_precision/spmd_sinkhorn_overlap_20261020/)。
微测试、两档整模型、Native金值/历史计时、源码和SHA清单分别打包；不包含模型权重或大型cache bank。

在227既有隔离环境中，实验根为：

```bash
exp=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/precision-overlap-20261020
# 新输出名称必须未存在；原始文件不覆盖。
bash "$exp/compile-full.sh"
# 通过task-submit提交 full-tools/run-full-csa.sh <new-case> <lengths> 4 8191
# 完成T126/T240后，用两个新case建立新的整模型快照：
python "$exp/prepare_whole_overlap.py" --job-name whole-overlap-retest1 \
  --t126-case <new-t126-case> --t240-case <new-t240-case>
bash "$exp/whole-overlap-retest1/preflight.sh" 8k
```

硬件任务统一由`task-submit --device auto --max-time 780 --ptoas 0.63`提交；整模型额外`--device-num 16`。
显式传入PYTHONNOUSERSITE、ASCEND_HOME_PATH、PTOAS_ROOT、CCACHE_DIR、CMAKE_BUILD_PARALLEL_LEVEL=4；具体值见归档的run脚本和环境收据。
整模型命令为`bash <job>/run-case.sh 8k pto`，结束后`bash "$exp/audit-case.sh" 8k <job-name>`；
严格审计通过后才进行128K的preflight、`run-case.sh 128k pto`和相同CPU审计。
内层750秒、外层780秒，每项任务均小于15分钟。独立复核和全部检查发现首差即停止扩展配置。

原始bank和gold路径、plan SHA、静态包SHA、完整环境及helper SHA在configuration/preflight及工具归档中；
换环境时应先恢复相同依赖、固定输入和Native参考，不能只复制一个variant开关就当作同协议复测。

## 任务与保留范围

- 前缀：`task_20261020_120905_146726214387`、`task_20261020_121100_155222716128`、`task_20261020_121244_168651713593`。
- 最终FULL：`task_20261020_123227_29130661160`、`task_20261020_123630_312365529388`。
- 整模型8K：`task_20261020_124207_344163327865`；128K：`task_20261020_125608_4191861840`。
- 首次FULL在argparse阶段退出：调度器追加的device参数被旧可选参数误读。修为显式layer/history并重建manifest，未进入数值计算。
- 最终FULL前完成提交格式整理，两档均按最终六源重新运行，旧源码探索数据没有重标为最终通过。

下一步应先测量Norm发布到Q_A启动的间隔、前缀Scalar成本与后缀计算占比，再决定更大的流水候选。
当前前缀MIX已有32描述符上限；把Q_A直接加进去需要先处理描述符与产物依赖，继续保留Native K顺序及舍入。
本轮没有把nalinaly的全链路SPMD或其数值实现整体移植，也没有重测其最新速度。
