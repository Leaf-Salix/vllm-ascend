# 复现：新SDK完整CSA与常驻后缀

## 固定版本与依赖

沿用[独立SDK准备记录](../spmd_sdk_baseline_20261020/REPRODUCTION.md)及项目中同名工具链报告。
本轮没有重新安装vLLM、Torch-NPU、CANN或模型，也没有改默认CSA路由。
先按上一轮收据验证实际Python、PyPTO、Simpler、PTOAS路径和二进制SHA；身份不符应停止。
本轮结果内保存实际SDK导入路径、DSO SHA、PTOAS_ROOT和候选/基准源SHA。

冻结运行源为`48c58ce7c467dfce63d5cb5c41afd225488d6012`，
77份运行源保留；完整层runner另用binding与source_identity校验实际源。
Native采用SK0，不能改成SK1后继续沿用本轮金值。
模型目录为`/data/model/DeepSeek-V4-Flash-0731-w8a8`。

## 文件与身份

`reproduction-tools.tar.gz`保留本轮原样脚本、full-csa-tools、候选与proof、sdk-preflight和本轮review。
`raw-results.tar.gz`保留原始JSON及失败记录，`RAW_SHA256.json`逐项校验；不修正原始JSON字段。
源码/脚本若因移机改路径，需重新记录SHA并重新授予设备资格，不能继续冒认已测SHA。

常驻候选SHA256：`a31a78f89e1ed71c007de2b692434cc0a0f1bc35626b22188418b3e715edd022`。
实际测试harness SHA256：`66a59034eeade9582614fd0519fdbb4e1669c30470f89bb001b7c49786a67aa5`。
候选、proof和sdk-preflight放在SDK根；其余脚本与full-csa-tools放在SDK的`precision-perf-20261020`。
脚本使用自己的构建/cache/OPP目录；换一个从未使用的case名称，不覆盖原始产物。

## 无设备检查

```bash
sdk=/data/pyptouser/yejia/csa-spmd-sdk-20261020
bash "$sdk/precision-perf-20261020/check-references.sh"
```

此脚本执行`skip_ptoas=False`的PTOAS、CCEC完整编译，检查冻结O-proj的9参数与HC-post的5参数schema，
并加载KernelArtifact，要求NPU未初始化。早期`skip_ptoas=True`失败只是缺失kernel_config，不能算精度失败。
已有缓存应复用；不要为本次对拍重编vLLM或原环境。

## 设备任务

以下每个命令都单独提交、完成后读result，精度全通过才进入下一项。不能在登录shell执行NPU。

```bash
sdk=/data/pyptouser/yejia/csa-spmd-sdk-20261020
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash $sdk/precision-perf-20261020/run-full-csa.sh full-csa-repro-b16 16 8191 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6"
```

此项验证完整HC-pre至HC-post，Native与CSA八边界精确对拍、A/B/A与guards通过后，交替12窗×100回放。
显式传16项lengths，避免早期默认lengths启动失败；首次启动失败原因未证明。

```bash
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 \
  "bash $sdk/precision-perf-20261020/run-suffix.sh resident-repro-t18 18"
```

T18通过后，分别以新case名称、tokens126和240提交两个独立任务。脚本内进程上限750秒；
调度上限780秒。提交后读取自身task状态与`$sdk/results/<case>-run.log`，
结果在`$sdk/results/<case>/result.json`。每5分钟以内查看一次，不遗留pending/running任务。
脚本明确覆盖PTOAS_ROOT为隔离0.66 wheel，不能把提交命令中的0.63当实际编译版本。

后缀使用seed=`4190+T`；heads正态std0.03、residual正态std0.1，post在0～1、comb在0～0.25均匀分布。
权重来自Native loader的实际第4层，读取NZ29矩阵及Native加载后的通道scale，无Native后缀计算金值。
五边界先逐位比较，FULL回放A/B/A并毒化输出，guard/readonly/generation均通过才计时。
每窗100回放、两臂顺序交替，总12窗；无离群值剔除，无DFX、依赖图、PMU。

## 结果解释

完整CSA与Native SK0对拍，后缀与冻结PTO数学对拍，两种资格不可混用。
后缀没有请求metadata，本轮不单独验证混合TND入场，也不验证整模型token、DP16吞吐或EPLB。
即便后缀逐位一致，也需要接入完整CSA后重新对拍，再复核8K/B40和128K/B24整模型token。
后缀性能若回退，保持独立实验文件，不切换默认路由。
