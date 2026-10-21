# 完整CSA四方差距定位与寻址单因素证据

## 结果入口

- 主报告：[`DSV4_FLASH_CSA_TND_NALINALY_FULL_CSA_GAP_20261021.md`](../../DSV4_FLASH_CSA_TND_NALINALY_FULL_CSA_GAP_20261021.md)。
- `nal-full-common-metadata-8k-b16-v1.json`：最终共同producer边界的四方性能与逐位门禁。
- `nal-full-four-arm-8k-b16-v5.json`：早期Nal未含producer的结果，保留追溯，不能替代最终表。
- `nal-full-physical-a-8k-b16-v1.json`、`nal-full-physical-b-8k-b16-v1.json`：Native/冻结/A及Native/冻结/A/B同轮结果。
- `STAGE_ANALYSIS.json`：独立L1记录与独立dep映射的实际执行函数，中位数与每窗原始数据。
- `PMU_ANALYSIS.json`：完整Simpler外壳计数，不能解释成内部模块的纯等待。
- `RUNS_FINAL.json`：全部13项任务最终状态；没有pending/running。
- `NALINALY_USER_BASELINE.json`：用户新截图八档数值的转录与几何平均重算，非本轮硬件结果。
- `CURRENT_WHOLE_MODEL_RECHECK.json`：本轮CPU直接重读10月20日两档归档，逐项重比保存token/接受事件/零抢占并重算16rank最慢步的均值；`recheck_current_whole.py`在工具包中，以checkout绝对路径为唯一参数运行，不提交NPU任务。
- 当前最优已验收完整候选的整模型结果沿用[10月20日Sinkhorn重叠证据](../spmd_sinkhorn_overlap_20261020/README.md)：8K/B40 forward106.4669ms，128K/B24 forward90.7129ms，保存的token及接受事件全与各自Native一致；本目录未新增整模型测试。详细口径见主报告。
- 三个DFX/dep/PMU压缩包保留原始记录，`final-result-logs.tar.gz`保留原结果、任务日志、CPU门禁与失败记录。
- `cpu-generated.tar.gz`只保留A/B相关QK/PV、checked、编排与ABI文件；不是SDK安装包。
- `reproduction-tools.tar.gz`包含本轮工具、固定Nal参考源、源码证明、独立审查和各候选。
- `SHA256.json`校验本目录交付文件；不含其自身。

## 版本与前置条件

被测分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，本轮起始HEAD `ac80d0e3b44dcacf0e6ea8d62329e4bf48f79a5c`。
正式数值源仍为`48c58ce7c467dfce63d5cb5c41afd225488d6012`，无正式数学改动。
Nal固定`91c4400e19ec10bd96e8de6314376f29229f6e0e`，参考文件逐SHA验证。

复现依赖227上已保留的私有SDK与冻结源副本；本材料不把旧env覆盖为新SDK。

- SDK：`/data/pyptouser/yejia/csa-spmd-sdk-20261020`。
- 模型：`/data/model/DeepSeek-V4-Flash-0731-w8a8`，读取正式第4层。
- 冻结控制源/环境由各`run*.sh`明确绑定，`source_identity.py`拒绝源码或导入混用。
- PyPTO `2dfa7f120ba5e577566046d920845aa241bfa7bb`。
- Simpler `57e499238fd4ee3548947ad860be456aaeeb16b5`。
- PTO-ISA `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`。
- Python3.12.14、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、CANN9.2beta2。
- 实际PTOAS为隔离0.66 wheel，SHA256 `d121b5b5db9a23a98eb3c0acc74d0abaef8aaa90e4df0385e79feb67636ef997`。
- `--ptoas 0.63`只是队列选择器，运行脚本显式覆盖PTOAS_ROOT，不能据该参数认定实际编译器版本。
- 没有启用C++等待退避、源码替换、phase clock或atomic；DFX/计时使用相同Worker。

实际PyPTO/Simpler导入和DSO记录在device预检日志、SDK_SOURCES及源码身份收据中；各raw JSON的字段按运行模式不同，并非每份都含全部环境字段。
控制源/helper SHA、二进制manifest及加载资格另见对应收据。若前置文件缺失或SHA不符，应恢复相应快照，不要绕过校验。
固定参考的dynamic SPMD kernel已支持TND；本次其旧NativeCSACall绑定器要求全S6，实际仅测试共同的B16/T96。

## 准备工具

在Mac将`reproduction-tools.tar.gz`解包到一个新的本地临时目录，再用rsync上传到上述SDK的`nal-full-compare-20261021/`。
保留冻结源/SDK不变，排除`._*`、`.DS_Store`、`__pycache__`。所有NPU执行必须经过task-submit，不裸跑。

```bash
mkdir -p /private/tmp/csa-gap-repro
COPYFILE_DISABLE=1 tar -xzf reproduction-tools.tar.gz -C /private/tmp/csa-gap-repro
/opt/homebrew/bin/rsync -av --exclude='._*' --exclude=.DS_Store --exclude=__pycache__ \
  /private/tmp/csa-gap-repro/ \
  pto227:/data/pyptouser/yejia/csa-spmd-sdk-20261020/nal-full-compare-20261021/
```

`check.sh`、`check-reuse.sh`、`physical-plan/check_a.sh`、`physical-plan/check_b.sh`为无设备门禁，先做完整specialize/ABI/PTOAS/CCEC及二进制恢复资格检查。
首次CPU检查中的输出目录固定且拒绝覆盖；若已有同名结果，仅将脚本输出目录和日志改为新的唯一后缀，不改变DSL源码。
`check-reuse.sh`使用只读`load_prebuilt`恢复，不能换成会重编译的`KernelArtifact.load()`。
所有CPU检查验证NPU未初始化，不需占卡。

## 硬件复现顺序

先检查同输入逐位，再计时；每项结果目录必须不存在。下面`repro-*`也要使用新的唯一名字。
任务内层750秒、队列780秒，每次完成后检查status和result，再提交下一项。不要同时跑计时、DFX和dep。

### 1. 完整图原始四方与二进制准备

`run.sh`的timing模式首次会编译Nal原始DSL。复用器固定验证`nal-full-four-arm-8k-b16-v5`的source/policy/B16/T96/8191收据与全部binary SHA。
在已保留的227目录可直接复用该收据；新搭建SDK时需先以这一结果名生成同版本基准，不能手工伪造READY或pass字段。

```bash
ssh pto227 'task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash /data/pyptouser/yejia/csa-spmd-sdk-20261020/nal-full-compare-20261021/run.sh repro-four-arm 16 8191 timing 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6"'
```

### 2. 最终共同producer口径

```bash
ssh pto227 'task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash /data/pyptouser/yejia/csa-spmd-sdk-20261020/nal-full-compare-20261021/run-common-metadata.sh repro-common 16 8191 timing 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6"'
```

两个Native producer在Nal图捕获内执行，输出视图零拷贝传入本次DSL参数。捕获计数必须各2，eager输出必须与fixture逐位相同；不凭flag跳过门禁。
本次全固定metadata，未证明跨请求动态更新。所有臂有独立cache，输入扰动A/B/A、readonly/guard、自重复；我方首个bit差即失败退出，不继续矩阵。

### 3. 阶段与硬件计数

分别将`run.sh`的模式改为`dfx`、`dep`、`pmu`并使用不同新结果名。
DFX只level1/dep=false，dep只level0/dep=true。PMU单独采3次raw replay，不用于报告主延迟。
`analyze_stages.py`使用本证据中的两个归档名；重跑后替换成相应新归档，必须核对所有task token与函数映射。
L1只见单SPMD外壳，PMU只见完整Simpler kernel；不能据此杜撰内部阶段或纯轮询时间。

### 4. 两个寻址单因素

```bash
ssh pto227 'task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash /data/pyptouser/yejia/csa-spmd-sdk-20261020/nal-full-compare-20261021/physical-plan/run_a.sh repro-physical-a 16 8191 timing 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6"'
ssh pto227 'task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash /data/pyptouser/yejia/csa-spmd-sdk-20261020/nal-full-compare-20261021/physical-plan/run_b.sh repro-physical-b 16 8191 timing 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6"'
```

先确认A逐位通过，再运行B。A保留逐行gather_row；B同轮包含Native/Frozen/A/B，唯一新增是64项索引DMA+mgather。
页数须小于2^24，物理行保持INT32，Native pair顺序、K512、归约、BF16边界及FFTS保持；保护区或首差失败立即退出。
原始12窗全部保留，臂顺序轮转并反转；3臂每位置4次，4臂每位置3次。

## 验收解释

`pass=true`表示该任务的软件/精度/保护区门禁通过；Nal臂已知对Native差异只记录，不作为gold。
A/B虽然数值通过，但**性能门禁未通过，都没有合入正式实现**。本轮未运行新的8K/B40、128K/B24整模型192-token测试。
不能把这些单层结果说成新的整模型token验收，也不能把候选源码当作已交付优化。
