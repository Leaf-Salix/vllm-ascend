# TND精度版：逐token验收范围与B28补验准备（2026-10-21）

## 当前结论

用户将最低交付要求调整为真实整模型与Native逐token一致，不再要求所有内部Tensor逐bit相同。
**本轮没有新的NPU精度结果，也没有修复或修改正式数学实现。**
内部差异继续作为定位证据；不能因为relative L2很小，就推断生成token不会变化。

| 场景 | 已记录的精度证据 | 当前资格 |
| --- | --- | --- |
| 真实整模型8K/B40，实际40活跃/T240 | 491,520个token、105,152个接受事件全部等于各自Native | 历史固定协议通过 |
| 真实整模型128K/B24，实际21活跃/T126 | 294,912个token、61,568个接受事件全部等于各自Native | 历史固定协议通过 |
| 128K/B28单层合成hidden/cache，T168 | TopK仅row86有88位置重排；输出row86/103共1731元素不同，六cache/state逐位相同 | 内部逐位失败；未证明真实token失败 |
| 新真实整模型8K/B28，要求28活跃/T168 | 12项本机控制、真实bank头及无设备环境预检通过 | Native1/PTO尚未运行，token结果未知 |

历史两档以[Sinkhorn重叠报告](DSV4_FLASH_CSA_TND_PRECISION_SINKHORN_OVERLAP_20261020.md)为准，
均为warm192＋三轮192，TP1、DP=EP16、确定性level1/HCCL、EPLB关闭。
本轮独立复核重新读取原始16rank请求token和事件，而非仅引用audit.pass；77份运行源仍与历史证据逐SHA相同。
这些结论不能外推任意请求、所有T、EPLB开启或其它SDK。

## 首差线索与修复原则

B28单层完整证据见[八档L1报告](DSV4_FLASH_CSA_TND_EIGHT_CASE_L1_20261021.md)。
本轮只用CPU读取自身保存的输入，确认T168的前126行hidden与旧T126诊断逐byte相同，包括row86/103。
旧T126曾定位到Q_A投影归约次序；当前T168没有对应QA逐位资格，因此优先核查该链路。
**这不是T168根因证明，不能直接扩大T126特判。** row103的TopK完全相同，也不能把全部误差归到排序。

已准备13阶段同输入诊断：mixed/post/comb/Norm、QR/scale/Q、Indexer系数和量化query、heads、WO-A/WO-B输出。
独立root保持64Tensor和原计算；剥离观察器后root/Indexer AST可精确还原，已通过原环境无设备program/kernel ABI编译。
解释中间值前必须验证Native和PTO全部8个外部边界均重现未插桩失败快照，并通过毒化、guard及只读检查。
截至本报告写作时，该单卡任务尚在排队，未取得阶段结果。

修复保留Native的BF16物化、RMSNorm和动态量化边界；不删除舍入、不增加量化epsilon、不用稳定重排掩盖分数差异。
若找到寻址、同步或错误dtype等逻辑问题，先修正；若仅余合法浮点归约差异，则记录差异并由真实token验收决定资格。
没有把Opus55保留Native QA的旁路作为完整PTO交付：本分支仍要求HC-pre到HC-post完整纯DSL和真正TND。

## 新8K/B28的固定方案

- 模型与原token验收相同：完整DeepSeek-V4-Flash-0731-w8a8、75分片，21个CSA使用完整PTO根，20个HCA及其余模型计算保留Native。
- 分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`；冻结数值源`48c58ce7c467dfce63d5cb5c41afd225488d6012`。
- 入口`pkg:deepseek_v4_flash_csa_tnd_precision_spmd_overlap`，纯DSL、64Tensor、真实1～6行请求边界；不改为每请求padding到S6。
- 原fixed8192 bank前28个唯一key，顺序不变，独立新plan。每请求保存的target/MTP cache按logical page加载，无B40批维，不重新prefill或复制payload。
- TP1、DP=EP16、batch/max_num_seqs28、FULL_DECODE_ONLY capture `[6,168]`、22000 blocks、token预算400、greedy、DSpark出5验6、ignore EOS、NZ2、level1/HCCL确定性、EPLB关闭。
- 新Native1与PTO各一个独立任务，均warm192＋一轮192。每臂记录172,032个token；Native轮间实际比较86,016对位置，PTO对两轮Native比较172,032对位置。
- 不从B40旧gold裁28条冒充同shape Native；新Native1内部两轮token不稳定时停止，不创建可用gold。不重跑旧场景Native2，不运行128K。
- 全16rank、两轮均要求真实28活跃、168实际query行、FULL168及联合DP向量；只有bucket168或名义B28不算覆盖。

逐请求token ID严格比较，同时另报接受事件和完整生成窗口输入/入场轨迹。
事件或轨迹不同而token相同标记`TOKENS_EQUAL_PROTOCOL_DIFFERENCE`，不能误报token失败，也不能宣称同轨迹验收通过。
新8K/B28通过只授予该协议资格，不等于全128K/B28或任意长上下文通过。

## 已完成的无设备验证

本机12项正负控制和独立复核通过：包括token首差、Native真实轮间差异、事件/输入差异分列、gold篡改、重复key、假T168覆盖、bank dtype/MTP/path及真实wrapper激活检查。

227上实际验证了前28条token/manifest身份、全部target/MTP条目名称/shape/dtype/压缩及owner、文件大小。
新plan SHA为`3340c0efaffc9baf976f10d7a42b7a51d72e6568642777a146025d45877b4d75`。
这不是约2GB cache payload的重新全量hash；两臂将只读复用同一冻结初态，也未证明与nalinaly原bank逐byte相同。

环境7模块file/version及CANN/ATB/PTOAS与旧token验收逐项相同；3399文件manifest、77数学源、实际扩展connector、75模型分片和hook激活通过。
CPU准备前后均检查`torch.npu.is_initialized()==False`，任务不占卡，未安装软件、切换SDK或改原bank。

两类CPU准备问题及失败日志均保留：

1. 关闭B28观察hook时，editable导入优先级使CPU准备选到另一PyPTO；仅改PYTHONPATH不足，复用原验收SDK选择桥后正确通过。
2. CPU wrapper未设置ADMISSION_TRACE_DIR，worker/observer finder按设计跳过；补齐与真实入口相同的启动环境后通过，没有删除门禁。

bank阶段在一项整体exit1的任务中先通过，后续hook预检失败；最终独立preflight任务exit0。
因此证据明确分阶段计分，没有把失败任务整体改记成功。本机重算为`PASS_CPU_PREPARATION_ONLY`，不是token通过。

## 复现与证据

证据目录：[b28_token_preparation_20261021](evidence_tnd_precision/b28_token_preparation_20261021/)。
其中`whole-b28-tools.tar.gz`包含封存22助手、CPU控制、独立审查、环境/准备wrapper和重算器；
`stage-diagnostic-prepared-tools.tar.gz`为已编译、尚未运行的13阶段观察器；CPU原始日志/收据单独打包。
没有模型权重、cache payload或NPU阶段Tensor。工具README保留本地封存时说明，后续实际状态以本报告及CPU收据为准。

在已验收227环境中，恢复独立目录后：

```bash
# base为既有227实验根，不覆盖原输出。
exp="$base/b28-precision-20261021-codex"
job="$exp/whole-b28-next"
# 新bank不存在时，CPU-only准备；已经准备完不要重复创建。
bash "$exp/cpu_prepare_whole.sh"
# 检查通过后，先收回13阶段诊断并决定最小修复，不提前提交大任务。
# 再分别经task-submit --device auto --device-num 16 --max-time 780提交：
# bash "$job/run-arm.sh" native
# READY_GOLD.json真实通过后：bash "$job/run-arm.sh" pto
```

每项内层750秒、调度上限780秒；不能在登录shell裸跑NPU入口。
实际task还须显式传原PYTHONNOUSERSITE、ASCEND_HOME_PATH、PTOAS_ROOT、CCACHE_DIR和四路构建参数，详见工具环境收据。
等待只用`--status/--log`和本地间隔：本轮误用`--timeout 45 --wait`自动取消了未分卡的原任务，已向用户说明并同命令重提一次，未产生设备结果。
原任务和重提关系保留在工作记录，不能把被取消排队记为精度失败。

源码/收据SHA见目录SHA256.json。CPU bank、hook、身份核验已完成；新的单层阶段及真实模型token结果仍待设备。
