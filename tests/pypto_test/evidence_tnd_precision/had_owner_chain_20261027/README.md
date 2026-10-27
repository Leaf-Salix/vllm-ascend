# Hadamard直接移植与HC owner链证据

主结论见上两级`DSV4_FLASH_CSA_TND_HAD_OWNER_CHAIN_20261027.md`。正式运行基准仍为11b380e3f；归档是独立候选，不代表已采用。

## 无设备复算Had双向结果

在本目录、Python3.11执行：

```bash
python verify_had_measurements.py --evidence-dir . --out /tmp/had-recheck.json
cmp /tmp/had-recheck.json HAD_MEASUREMENT_RECHECK.json
```

脚本不需要Torch/vLLM/NPU、不解包到磁盘，从两份原始tar直接复算全部event中位与配对胜窗、18窗L1的72core原始时间戳、实际63个已注册子项的CPU见证，以及反向Native A/B八边界和ABA。未执行精度用null表示，不能当失败或通过。

## 硬件复现

需要前一报告`DSV4_FLASH_CSA_TND_STAGE_GAP_QA_SEED_20261027.md`指定的3399源快照、工具链和真实模型权重，以及本报告所列环境。解包executed_helpers.tar.gz到原实验根；had-four-token-v1和hc-owner-chain-v2各自用compile.sh baseline/candidate做无设备完整CPU资格，随后用相应observer的cpu_check.py验证序列化正例/单字节反例，NPU必须仍未初始化。修改绝对路径必须重新生成binding/manifest和CPU资格，不能冒用旧身份。

每个新唯一label使用`task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 780 --env PYTHONNOUSERSITE=1 'bash <observer>/run.sh <label> 32 8192 <32个逗号分隔6> [--reverse-capture] [--qualify-boundary]'`。inner750s。原env、cache1、static13、NZ2、SK0和Native确定性不变。原始criterion有沿用query-island的文字；实际被测修改以candidate_variant/SOURCE/manifest为准。

observer从immutable parent校验CPU资格和运行脚本；Had observer-v2的derived_probe只增加两个观测身份字段，HC普通observer沿用该方式，计时/精度门槛与父probe一致；shared派生另增加spy、共享图和实际capture观察，见下节。计时前检查READY/PID/generation/owner以及真实loaded callable的全部序列化子项，再核对严格alpha编排、meta字节和config值。完整event和独立L1分开采集，不启dep-gen。

## 失败和范围

首个Had任务因旧observer假定持久缓存generated目录含binary_manifest而在计时前失败；不计数学失败。HC v1缺少PARTIAL_ROWS导入，只在CPU失败，v2修复导入后重新完整资格，未改正式环境。executed_helpers保留这些历史，不等于每个helper都上过NPU。

单层使用真实第4层权重、合成hidden/history，不是整模型token测试。性能未稳定通过则不扩短T/混长/128K和整模型；单层精度也不能外推全shape/真实token。原基准验收保持，不能替候选背书。

## HC正反与共享实参的离线复算

```bash
python verify_owner_measurements.py --evidence-dir . --out /tmp/owner-recheck.json
cmp /tmp/owner-recheck.json OWNER_MEASUREMENT_RECHECK.json
python verify_shared_measurements.py --evidence-dir . --out /tmp/shared-recheck.json
cmp /tmp/shared-recheck.json SHARED_MEASUREMENT_RECHECK.json
```

共享v2还验证same-arm eager/FULL八边界、真实capture CallFrame64个Tensor见证、共享图重新READY注册资格、Native A/B八边界和ABA。只用host构造tuple的v1未提交设备；v2增加实际enqueue观察，通过独立审查后执行。hook只在捕获时只读观察、原样转发，计时之前已恢复，SDK文件保持原样。共享直接root FULL和原服务图结果分表，不能混当同一性能口径。

共享诊断启动方式为`bash <hc-owner-shared-v2>/run.sh <unique-label> 32 8192 <32个6> --qualify-boundary`，同样通过task-submit单卡780秒，inner750秒；需要原parent hc-owner-chain-v2 CPU资格目录。cpu导入首次命令缺少parent sys.path导致candidate_overlay找不到；修正启动命令后无设备导入成功，没有改算子或SDK。

本轮所有设备任务最终exit0（最早旧observer失败任务除外），原始task状态和未执行的范围分别见HAD_FOUR_TOKEN_TASKS、HC_OWNER_CHAIN_TASKS和SHARED_ARGUMENT_TASKS。没有稳定性能winner，全部未采用，正式11b基准不变。

HC_V1_CPU_FAILURE_RECEIPT_EXCERPT.json是通过既有SSH读取的CPU失败收据字段摘要。服务器随后关闭两次新的SSH连接，未取得v1完整失败日志归档；不要将该摘要标为完整原档。本轮所有有效性能、精度及DFX原档均已先下载并核验，未受影响。
