# 当前PTO精度版128K/B24首差回归（2026-10-17）

## 结论

**未通过逐bit精度验证。** 候选在第75轮（forward index 74）首次执行实际FULL6时，
rank0的target hidden、三组aux hidden及MTP residual位摘要均与冻结Native1不同，按门禁立即停止。
本轮没有完成192 token全量验收，不能报告完整输出token或接受事件一致。

截至停止，全部16rank的75轮实际设备输入和入场轨迹与Native1完全相同；
rank0前74轮outer浮点摘要一致。差异已经缩小到首次使用FULL6的实际执行，
**本任务未定位内部模块；后续同轨迹hook已定位到L20的attention heads边界，见下文关联记录。**

原8K/B40通过结果保留，但不能外推到本长档或真实活跃的FULL6。
本轮未修改模型数学、旧混合精度基准或性能版，未提交后续大配置或Native2。

## 测试对象及参数

分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
数学提交：`c1ef141ae`；已有8K结果文档提交：`dec33820d`。
使用step09冻结overlay，3361文件清单SHA256：
`d997d41080882209007134564ca488ce19cfe7f136397b6fd0e389c929001638`。
该目录是在旧测试源码上明确接入正式数学文件的受控overlay，不是整个Git tree的复制。

| 项目 | 配置 |
| --- | --- |
| 模型 | DeepSeek-V4-Flash-0731-w8a8，完整75分片真实权重 |
| 输入 | 原fixed131072 bank，q00～q23，history为131104～131346 |
| 并行/请求 | TP1、DP=EP16，每rank提交B24，允许等待入场 |
| 计划输出 | 每请求192 token；greedy、ignore EOS、DSpark出5验6 |
| 图/容量 | FULL_DECODE_ONLY `[6,144]`；max_num_batched_tokens=400；固定24000 KV blocks；gpu util=0.97 |
| 确定性 | level1、Torch确定性、HCCL_DETERMINISTIC=true、atomic0、NZ2、event1、EPLB关闭 |
| 环境 | CANN9.2 beta2、ATB9.2 ABI1、Torch2.10.0、Torch-NPU2.10.0.post2、vLLM0.25.1；复用既有私有PyPTO/Simpler工具链 |
| 替换范围 | 配置为21层CSA使用tnd_full_precision，非CSA层保留Native；O-proj仍复用Native |
| 当前恢复的PTO部分 | HC-pre、输入Norm、WKV、Indexer权重投影、HC-post，以及已有CSA主体；保留真实TND请求边界 |
| Native对照 | 冻结任务`task_20261016_091954_247756916995`的Native1，未重新提交 |
| 新CSA任务 | `task_20261017_130310_40506967427`，16卡，max-time780秒 |
| 实际任务结果 | 7分59秒，exit1，首差停止后已释放设备 |

这不是新增的不等长query矩阵测试，也不产出性能结论。
沿用此前bank；未核验其完整prefix/cache与nalinaly原始bank逐字节相同。

## 验证步骤与首差

1. 先核验源码清单、导入路径、PTOAS/CANN/ATB、官方13份静态包和私有runtime身份。
2. 核验fixed131072 plan SHA，以及原Native1证据与派生生成窗口全部文件SHA。
   bank包含24条有序key；首次无设备预检因错误沿用8K的40条假设退出，修正为q00～q23完整顺序检查后通过。
   bank大cache payload未重新全量hash；没有修改bank或将预检失败计入数值结果。
3. 独立目录运行同一192 token配置；admission helper与已验证版本逐字节一致。
   相比旧128K的outer helper，只启用已在8K验证过的control模式实时outer比较。
4. 每轮先核对全部实际输入，包括请求顺序、query bounds、input IDs、positions、seq lens、padding及联合DP token向量。
   rank0同步记录target hidden、aux hidden、MTP residual摘要；首差抛错退出，不继续后续配置。
5. 任务停止后，CPU独立重算全部16rank轨迹前缀与rank0 outer前缀；保留失败原件，不运行完整通过资格器或生成成功JSON。

| 检查 | 结果 |
| --- | --- |
| 全部16rank实际设备输入/入场 | 已观测75轮全部与Native前缀相同 |
| 抢占/峰值请求 | 各rank零抢占，峰值21 |
| rank0 outer | index0～73逐bit摘要相同；index74首次不同 |
| 首差图档位 | FULL6，是本次各rank第一个实际FULL6轮次 |
| 首差请求 | `h131072_q21`，query bounds `[0,6]` |
| 首差位置 | `[131524,131525,131526,131527,131528,131529]`，seq_len=131530 |
| 首差input IDs | `[1457,27,438,223,2012,223]` |
| 联合DP向量 | 全部16rank均为6 |
| Native生成窗口 | 78轮；候选第75轮停止 |
| 完整输出/接受事件 | 未完成全量比较，不作一致结论 |
| 内部首差模块/ULP | 本任务没有内部stage hook或首差原值快照，尚未确定 |

“摘要不同”证明对应浮点tensor并非逐bit一致，但不能据此计算不同元素数、ULP或relative L2。
FULL6切换与首差同时发生是定位线索，不是已证明的根因；此前cache/state的潜在累积差异也尚未排除。
后续应在相同轨迹、同一首差输入上加入内部stage观测，找到最早变化的模块，
再区分算术边界与跨图缓存/状态生命周期；不能直接改Indexer数学或放宽验收。

## 复现与证据

远端独立helper目录：
`/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/full-pto-model-128k-b24-20261017`。
输出目录：`/data/pyptouser/yejia/lc128/full-pto128k-indexer-production1-20261017`。
Native原件目录：`/data/pyptouser/yejia/lc128/retest192-native1-b24000`。
Native生成参考：实验根下`retest192-20261016/native1-generation-reference`，
manifest SHA256为`347b452c4eed32897b54bd05305a317708fbea987ffeab1ad1ac1461f9153dac`。
本轮helper manifest SHA256为`8405e7e3b1f852ca38e2254f22b81bf40430bec7bacaf2027a6d871d4fee92fa`。

证据包中保留run-case、env、source_preflight、check_inputs、audit_alignment、qualify_current、
实时观测helper、helper manifest、Native参考、两臂原始小型日志/输入/摘要/STOP及六份正式数学源码。
复现依赖既有模型、bank、SDK、冻结源码及私有运行包，证据包不是新机器自包含环境。

在同一资产环境下，先运行`bash <helper目录>/preflight.sh`，再以新标签提交，禁止覆盖历史输出：

```bash
task-submit --device auto --device-num 16 --max-time 780 --ptoas 0.63 \
  --env PYTHONNOUSERSITE=1 \
  --env ASCEND_HOME_PATH=<预检确认的CANN路径> \
  --env PTOAS_ROOT=<预检确认的PTOAS路径> \
  --env CCACHE_DIR=<既有checkout的.ccache路径> \
  --env CMAKE_BUILD_PARALLEL_LEVEL=4 \
  "bash <helper目录>/run-case.sh precision-witness <新标签>"
```

确切已执行路径及参数均保存在证据的脚本、配置和预检日志中。现有qualify_current面向完整运行，
本轮因首差退出不具备其前提，不应修改门禁以得到通过。

- [首差结构化结果](evidence_tnd_precision/full_pto_20261016/model-128k-indexer-production1-failure.json)。
- [266文件原始证据包](evidence_tnd_precision/full_pto_20261016/model-128k-indexer-production1-evidence.tar.gz)，
  SHA256=`2937d2f9c2ac0243820c708e18745638b726ca9c011f5d22f49874f0cced7048`，2,208,515 bytes。

已在本地逐文件SHA核验266项证据，包内六份正式数学源码与当前分支逐字节一致。
不包含模型权重、大cache或首差内部原值tensor。

相关基准：[8K/B40 Indexer恢复结果](DSV4_FLASH_CSA_TND_INDEXER_PROJECTION_20261017.md)、
[旧混合精度版128K通过结果](DSV4_FLASH_CSA_TND_192_ADMISSION_RETEST_20261016.md)。

后续定位：[首次FULL6同输入内部hook记录](DSV4_FLASH_CSA_TND_FULL6_LOCALIZATION_20261017.md)。
