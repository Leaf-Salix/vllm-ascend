# 本轮测量与独立SDK复现说明

## 固定身份与边界

控制实现为Leaf-Salix分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`的
`48c58ce7c467dfce63d5cb5c41afd225488d6012`，运行源保持原验收版本。
参考SPMD为nalinaly `beaefa6f6788a9448d49293238f6818c246f24f5`。
SDK全部固定提交和实际包/DSO路径见`SDK_SOURCES.json`、`sdk-results/toolchain-wheel.json`。

本包不含模型权重、完整SDK源码/二进制、40MB首差tensor、历史大cache或整模型bank。
SDK源码按记录的Git提交恢复；实际PTOAS wheel须另行取得并验证SHA，不能以同版本字符串代替同一artifact。
本轮独立wheel没有已证实的Git提交，不将参考源构建结果冒充此wheel。

路径约定：实验根为227自有`vllm-nalinaly-tnd-20261008-codex/spmd-baseline-20261020`；
独立SDK为自有`csa-spmd-sdk-20261020`。脚本中的绝对路径需要按同身份映射。
权重是`DeepSeek-V4-Flash-0731-w8a8`，75个分片；单层用正式第4层权重，hidden/history为合成输入。

## 旧SDK单层计时、DFX及PMU

`reproduction-tools.tar.gz`保留实际运行脚本、fixture SHA绑定、只读状态及保护区检查。
依赖原`full-csa-live-staging-v2/env.sh`、冻结snapshot以及已验收77份运行源和编译收据；
这些并非仅解开新脚本即可替代的前提。冻结运行源及旧环境来源也见前序Sinkhorn overlap报告。

- `run-sk0.sh <新名称> 16 8191`：8K/B16、S6、实际FULL/raw、12个交替窗口×100回放，不开DFX/dep/PMU。
- `run-dfx.sh`：单独L1泳道；`run-dep.sh`单独依赖图。按task token关联内核，不把registration ID当func ID。
- `run-pmu.sh`：单独CANN PipeUtilization，不采正式计时。只能解释外层executor聚合计数。
- `analyze_dfx.py`、`analyze_cann.py`及原始bundle用于重新计算；不将窗口独立中位start/end拼成实际时间线。

原生SK1的包、const修正及缓存输入以`sk1-offline`收据固定；runner验证完整集合和SHA，
调用官方安装/reselect，并在缓存缺失时拒绝设备任务内重编。
本轮SK1与SK0存在首差，故没有SK1正式计时结果。SK0导出任务复现了历史八边界及输入/初始cache哈希。

## 新SDK先做无设备门禁

`build-sdk.sh`及旁路脚本均写在独立SDK目录，不能向原环境安装。
先按固定源恢复离线依赖、Git元数据和PyPTO runtime关联，再使用checkout自身ccache和显式并行度。
PyPTO与Simpler的原生构建、7项CPU API/ABI测试、LLVM源码构建已有记录。
PTOAS源码构建未完成；当前实际路线是`prepare-ptoas-wheel.sh`离线安装已验SHA的0.66 wheel。

按顺序执行：

```bash
bash test-sdk-cpu.sh
bash test-sdk-wide-signal-lowering.sh
bash test-sdk-frozen-lowering.sh
bash test-sdk-suffix-lowering.sh
bash test-sdk-full-compile.sh signal
bash test-sdk-full-compile.sh frozen
bash test-sdk-full-compile.sh suffix
bash test-sdk-full-resident.sh
```

脚本输出目录拒绝覆盖旧结果；重新运行应换新的目录名，不删除原始失败证据。
`seal-sdk-wheel.py`应在独立ptoas-wheel-env下运行，绑定wheel、DSO、动态库和完整编译收据。
实际PyPTO、Simpler、core、adapter、native binding必须全部指向新独立目录。
继承的editable finder可能抢占，不能仅凭venv名称判断；`sdk-preflight.py`逐项硬核对。

宽入口使用原生JIT prepare/enqueue。现有Torch仅允许64参数schema，不要注册130或132参数Torch算子。
完整编译包括PTOAS、CCEC和native kernel ABI编排二进制，但不初始化NPU，也不授予设备数值资格。

## 设备门禁与停止条件

使用task-submit，单次外780秒/内750秒；任务内源码固定，首次错误停止。
以下是提交形式，是否实际执行及结果以`RUNS.json`为准：

```bash
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  'bash /data/pyptouser/yejia/csa-spmd-sdk-20261020/run-sdk-wide-signal-device.sh'
```

启动脚本显式把PTOAS_ROOT切到独立`toolchain-wheel`，实际0.66来源以收据为准，不能根据外层`--ptoas 0.63`误写运行身份。
信号门禁验证24 AIC/48 AIV、132个实参、不同owner、FIX写回后的可见性、代际、31次graph replay、A/B/A和尾部guard。
即使通过，也不等于完整CSA、后缀或整模型通过。

已执行`run-sdk-frozen-device.sh frozen-8k-t18-v1 4 8191 6,5,4,3`验证真实TND共18行、FULL/raw和八边界逐bit一致；
信号门禁及该冻结CSA回归均已通过；新名字可用于复现。该新SDK回归只禁用计时/trace，保留只读、poison、guard、A/B/A和失败tensor导出。
常驻后缀尚未接入正式服务；host需核验flat视图同storage/pointer/offset、容量、无别名、NZ2及信号独占。
有新的稳定完整CSA收益后，才继续8K/B40、128K/B24整模型192token及接受事件与历史Native逐项对齐。

机器JSON记录原样存于`raw-results.tar.gz`，先展开到证据目录即可恢复上述相对路径；逐文件身份见`RAW_SHA256.json`。
