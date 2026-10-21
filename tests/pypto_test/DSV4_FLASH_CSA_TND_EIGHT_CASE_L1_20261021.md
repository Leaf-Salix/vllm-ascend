# TND精度版：参照Nal公开方法的八档完整L1实测（2026-10-21）

本文保留修复前的历史结果。B28首差已修复，最新八档重测见
[QA168修复版完整CSA L1报告](DSV4_FLASH_CSA_TND_QA168_FIXED_EIGHT_CASE_L1_20261021.md)。

## 结论

八项任务均已结束：**七档逐位通过并测得L1；128K/B28首次对Native逐位比较失败，按门禁退出，未采集L1。**
本轮冻结完整TND精度候选，先与确定性Native逐位对拍，再按公开Nal方法采集完整CSA的L1。
已获得L1结果的七档精度门禁全部通过，性能仍明显慢于用户截图中的Nal TND参考，不能宣称已对齐其速度。

这次测的是完整HC-pre→HC-post单层链路，单位μs，不再混用真实整模型forward的ms数据。
没有修改正式算子、SDK、生成C++等待函数、默认路由或已验收精度基准。

## GitHub出处：找到方法，未找到截图对应版本

直接查询GitHub提交API、代码搜索和公开分支，并反查`a241fa61`、`363.28`、`357.18`等标识。
`nalinaly/vllm-ascend`提交API对`a241fa61`返回422；相关公开检索未找到该八档摘要。
对已知短SHA `91c4400e`的相同接口查询成功，排除了接口不接受短SHA的问题。
这仅说明在本轮可访问的公开内容中未定位到出处，不能推断对方一定没有上传。

已找到并固定公开参考提交`91c4400e19ec10bd96e8de6314376f29229f6e0e`：

- [完整动态CSA测试说明](https://github.com/nalinaly/vllm-ascend/blob/91c4400e19ec10bd96e8de6314376f29229f6e0e/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/README.md)。
- [run_device.py：真实权重、fixture和门禁](https://github.com/nalinaly/vllm-ascend/blob/91c4400e19ec10bd96e8de6314376f29229f6e0e/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/run_device.py)。
- [l1_compare.py：完整incore起止与窗口中位数](https://github.com/nalinaly/vllm-ascend/blob/91c4400e19ec10bd96e8de6314376f29229f6e0e/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/l1_compare.py)。

因此本报告是**参照公开方法实测我方精度版**，不是重新运行截图的`a241fa61`和未知TND候选。
下表Nal两列均为用户提供的截图参考，没有将旧91c复测或其它实现冒名填入。

## 八档结果

全部S6，完整PTO链路L1，5窗中位数；保留所有窗口，不剔除离群值。
“Native逐位”指同一次任务内A/B输入的最终输出、TopK索引及六组cache/state边界，不代表整模型token测试。

| 档位 | Nal BSH参考 μs | Nal TND参考 μs | 我方完整TND精度版 μs | 我方对Native逐位 |
| --- | ---: | ---: | ---: | --- |
| 128K/B1 | 363.28 | 357.18 | 668.08 | 通过 |
| 128K/B4 | 506.12 | 493.52 | 850.94 | 通过 |
| 128K/B8 | 617.16 | 620.40 | 1038.86 | 通过 |
| 128K/B16 | 789.94 | 777.44 | 1419.38 | 通过 |
| 128K/B24 | 1017.90 | 1013.46 | 2033.90 | 通过 |
| 128K/B28 | 1153.56 | 1156.34 | 未计时 | 首差失败，待定位 |
| 8K/B16 | 635.20 | 634.16 | 1213.20 | 通过 |
| 8K/B32 | 915.52 | 918.32 | 1793.44 | 通过 |

两组结果没有建立同卡同进程、同工具链和同等待策略的受控关系，故不把相对截图的差值或几何平均作为优化收益证明。
仅按表内数值观察，当前精度实现尚未接近参考速度；这也不能单独归因为TND布局。
各窗原始值、跨度重算和源码身份见证据，结果不使用之前history8191的单点替代。

### B28首差：TopK排列及两行最终输出

`task_20261021_112831_224465013915`最终状态`completed (exit=1)`。
最后成功读取的日志显示Native自身检查通过，候选在首次A输入对Native的byte比较处抛出
`candidate: first bit difference; stop matrix and timing`，尚未进入B输入及L1采样。
六组cache/state均逐位一致；不同边界为`idx_topk`和`x_out`。
TopK只有第86行的88个索引位置不同。对自身保存的CPU首差文件只读分析确认：该行排序后的512个索引完全相同，**选点集合相同，排列不同**；
但不只是每个固定相邻二元组内互换，不能简单套用pair-swap来修复。

最终输出有1731个元素不同，整体relative L2为`7.1388531e-5`，max abs为0.015625；不同输出行是86和103，分别1343/388个元素。
**第103行TopK索引没有改变但输出仍不同**，因此不能把全部误差归因于第86行TopK重排，也不能认定仅修排序就能逐位一致。
当前最早可见的外部差异在TopK；尚未hook Q、系数、score、attention heads与O-proj中间量，未确定内部首个误差模块。

取证一度因三次SSH连接关闭中断，退避后正常连接恢复。失败JSON、日志、preflight与CPU差异分析均已取回；大型首差tensor保留远端，未下载。
前七档七份成功包及35个L1窗口已独立重算；B28失败包不含L1，不纳入性能中位数或“通过”计数。
没有继续提交性能或大配置任务，最终CLI核对八项任务均已结束并释放设备。

## 固定实现和软件环境

- 当前分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，本轮起始HEAD `f849bb1f155744d3ece74266d2a6e365a8e88aae`。
- 候选：`pkg:deepseek_v4_flash_csa_tnd_precision_spmd_overlap`；正式数值源提交`48c58ce7c467dfce63d5cb5c41afd225488d6012`。
- 完整HC-pre→HC-post、一个64Tensor顶层PTO入口、纯DSL、真实TND；前缀SPMD、后缀多task，保持原实现。
- 每档重查43份冻结基础源、6份候选源及父包收据；复用既有二进制/缓存，不安装或重编PyPTO、Simpler、vLLM。
- 使用已验收原环境；Python3.12、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、CANN9.2beta2。
- PyPTO/Simpler使用旧验收环境，未切换到10月21日四方对拍的新SDK；六个实际import路径、PTOAS_ROOT及环境见逐任务preflight。
- Native为SK0、13份静态包实际安装/reselect、DSA多流；本轮没有把SK0写成SK1。

## 输入和精度门禁

正式`DeepSeek-V4-Flash-0731-w8a8`第4层权重，合成hidden/history，TP1、page32、NZ2，seed1024。
8K历史为8192，128K为131072，query positions从history起；每请求6个真实token，图行数就是`6×B`。
页表按公开fixture保留每请求独占页和逆序循环映射，不切换升序或碎片页策略。
Indexer历史scale按公开run_device覆盖为`0.00390625 + ((row * 37) % 251) / 16384`，同时刷新初始cache快照。

Native与我方各自独立cache，实际核对hidden、positions、初始cache、readonly metadata和compact cos/sin/slot逐byte相同。
每档先做以下检查，首差立即退出，不继续后续矩阵：

1. 同输入Native和PTO的最终输出、TopK索引、SWA、压缩KV、主state、Indexer key/scale及Indexer state逐byte相同。
2. 输入A→B→A，同址图读取新hidden；B同样对Native逐位，恢复A后再检查输出/state及guard。
3. 实际FULL capture、compiled/raw一致；输出和TopK毒化后能重放恢复；eligible必须通过，禁止Native静默回退。
4. readonly、合法cache写区、只读Norm/order数据包及权重资格保持，实际调用64Tensor overlap根；warmup及各测量窗后复验。

Native开启Torch确定性、NPU level1和HCCL确定性。
本轮只覆盖上述固定单层样例；没有新增192-token整模型或混合短query验收。
S6是这张性能表的输入条件，正式TND代码没有改成按请求补齐6行。
既有整模型token验收继续以[Sinkhorn重叠报告](DSV4_FLASH_CSA_TND_PRECISION_SINKHORN_OVERLAP_20261020.md)为准，不能用本轮L1代替它。

## L1计时方法和必要差异

每档在捕获和精度门禁完成后warmup3次，再采5个独立窗口，每窗一次完整FULL raw graph replay。
恢复cache/readonly及同步均在窗口外；`begin_dfx/end_dfx`只包围实际回放。
从六列原始记录的第3/4下标提取incore开始/结束，取全部核最早开始到最晚结束，按原始clock_freq转换μs，再取5窗中位数。
每窗检查24AIC+48AIV、全部72核、无丢失run边界、一次完整根提交；多task根不要求只有72条记录。
该跨度包含内部等待和阶段空隙，排除PTO incore之外的头尾开销，不是纯计算时间、图外event或服务延迟。

与公开91c脚本有明确差异：

- 公开runner使用level0；我们保留用户要求的Native level1精度基准。
- 公开runner在L1时同时启用dep_gen；我们按工作区要求分开采集，本轮dep_gen关闭。
- 公开性能说明启用了100tick生成C++等待hook；我们按严格纯DSL要求不启用，不替换生成代码。
- 使用我方已验收原SDK和数值计算顺序；截图具体工具链、TND commit、等待策略及统计配置仍未核实。
- 我方graph仍保留Native metadata生产，L1只统计PTO incore；公开参考的metadata在图外。本轮不能据L1推断这部分开销为零。

本轮没有优化算法。**下一步先定位B28首差，保留基准，对照Native修正后复测该档；逐位通过后才补L1。**
先分别hook第86/103行的Q、Indexer系数/score/TopK和attention/O-proj边界，不能只替换TopK或忽略第103行。
精度门禁解决后，再逐项验证heads发布合并、Native K512搬运流水和前缀放行粒度。
只有完整CSA稳定更快且逐位门禁通过，才进入8K/B40→128K/B24真实整模型192-token回归。

## 复现与证据

见[证据目录说明](evidence_tnd_precision/eight_l1_20261021/README.md)：固定工具、参考源码、SHA、八个小任务状态、前七档逐窗原始记录及两套CPU核验。
任务依次单独提交，内层750秒、队列780秒，任一首差停止后续；没有合为一个长占卡任务，也没有使用定时自动化。
