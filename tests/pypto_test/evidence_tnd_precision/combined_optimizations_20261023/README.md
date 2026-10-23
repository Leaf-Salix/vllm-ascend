# TND精度版组合优化证据

对应正式入口为`PTO_CSA_VARIANT=tnd_precision_combined`，原包保留。两项优化是Native顺序WO-B反量化与HC-post融合、独立input packing提前以重叠平方和。不是nalinaly实现的同环境重测。

## 原始证据

- `single_layer/`：四臂原始FULL graph结果，包含所有event/L1窗口、捕获顺序、Native A/B八边界及ABA、实际root schema、执行检查和padding收据；含无组合收益配置。
- `SINGLE_SUMMARY.json`：从上述原始结果重算中位数、直接对照、成对领先和资格，不是再次设备运行。
- `SOURCE.json`：基准及候选计算体准备证明。其hardware_precision_validated=false是不可变准备状态，硬件结果见result和whole审计。
- `FORMAL_SOURCE_PROOF.json`：正式checkout与实际import overlay的逐字节等价SHA。
- `cpu-receipt-final.json`：最终29文件manifest下，64Tensor真实编译、artifact加载、schema/NZ layout/T1..256路由通过且无NPU初始化。
- `LAUNCH_ARGUMENT_CHECK.json`：调度附加参数的纯shell header控制；不执行框架或设备代码。
- `TASKS.json`、`TASK_STATUS.json`、CPU任务日志：实际任务和终态。早期启动失败单独保留，不算数值失败。
- `executed_helpers.tar.gz`、`EXECUTED_FILES.json`：实际单层和整模型助手、formal package/alias/env overlay、manifest与复算脚本的原样归档和SHA；不含模型、bank cache payload、SDK二进制或编译缓存。
- 整模型验收进行中；`whole8k_summary.json`、`whole128k_summary.json`和raw归档尚未生成。完成后才归档各16rank的warm192和测192输出/事件、FULL轨迹、计时、配置及容量。

计划在整模型通过后，公开whole摘要仅将可重算的输出/事件SHA字段warm_tokens_events_sha256、measured_tokens_events_sha256重命名为warm_payload_sha256、measured_payload_sha256，避免密钥扫描误判。原样SUMMARY保留在raw归档；字段映射、反向还原等价和SHA见PUBLIC_SUMMARY_DERIVATION.json。

## 整模型加载修复证据

原v1八组单层证据仍绑定原29文件manifest。首次8K任务在加载时缺少package.qkv_proj_rope而exit1，尚未做token对拍；`failed_whole8k_postload_v1.tar.gz`及FILES索引保留19个原始配置/日志文件。

新增qkv_proj_rope.py只重导出QR_OK/KV_OK，和旧overlap模块逐字节相同。`POSTLOAD_FIX_PROOF.json`证明七个既有计算/路由文件未变；`postload_v2_cpu_receipt.json`真实导入并比较metadata；`POSTLOAD_IMPORT_RECEIPT.json`补充导入之后NPU仍false的独立无设备收据。v2原样助手及新30文件manifest封存在`postload_v2_helpers.tar.gz`和FILES索引，未覆盖v1。

## 精确复现

依赖版本、模型、Native黄金、bank、ring和范围限制见上级主报告。需要同版3399文件控制快照、兼容SDK、13包官方静态缓存、完整W8A8权重及已验固定bank/黄金；不能删除身份断言来适配不明环境。

解包到新的自有实验目录。原样env.sh带原路径，新目录需显式更改实验输出路径并重新生成manifest；保留原manifest作来源证据，不覆盖原验收目录或SDK。先无设备资格，再硬件任务，不裸跑。

```bash
tar -xzf executed_helpers.tar.gz -C <新的实验目录>
cd <新的实验目录>
# 适配明确的自有输出目录后，重封主目录/whole manifest并保留来源映射。
cat > compile-local.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
source ./env.sh
export PREFLIGHT_RECEIPT="$PWD/cpu-preflight.json"
timeout --signal=TERM --kill-after=15s 750s python ./compile.py combined
SH
task-submit --no-device --max-time 780 "bash ./compile-local.sh"
# 仅在cpu-combined/receipt.json所有门禁通过后执行：
task-submit --device auto --device-num 1 --max-time 780 \
  "bash ./run.sh <新label> 32 8191 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6"
```

同一入口可加`--reverse-capture`；混长使用`6 8191 1,2,3,4,5,6 --qualify-boundary`，短T4用`4 8191 1,1,1,1 --qualify-boundary`；T16/17分别为`5 8191 3,3,3,3,4`和`5 8191 3,3,3,4,4`并加qualification。128K/B16用history131071、16个6和对应已验小ring环境；详见原始result的环境身份与主报告。不能跨ring配置算加速。

正式整模型使用补齐加载契约后的v2；先解包`postload_v2_helpers.tar.gz`到同一新的实验根，适配自有路径后重封manifest。进入`postload-v2`，按上面的compile-local.sh方法生成该目录自己的cpu-combined/receipt.json（必须包含postload_reduction_metadata.equal_original），不能复用v1旧收据。导入后的NPU false检查可在实验根经no-device运行postload_import_check.py。只在这些CPU门禁及单层性能/精度通过后提交：

```bash
cd postload-v2/whole
task-submit --device auto --device-num 16 --max-time 780 "bash ./run.sh 8k"
# 8K通过全部token/事件/FULL资格后：
task-submit --device auto --device-num 16 --max-time 780 "bash ./run.sh 128k"
```

每任务内部750秒、调度780秒；每5分钟内观察。原脚本固定combined-8k-v1/combined-128k-v1输出，新复现必须用新目录，不能复写证据。不跑Native2。两个whole摘要没有本轮新Native或冻结版计时臂，不能据此声称受控整模型加速。
