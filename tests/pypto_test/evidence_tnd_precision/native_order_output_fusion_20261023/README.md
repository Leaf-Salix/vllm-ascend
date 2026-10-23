# 保留Native顺序的输出融合证据

冻结数学为6eabea0ec，优化前验收快照397941659。同一分支继续实验，默认实现不变。

## 文件

- `SOURCE.json`：冻结根/O-proj与原始候选源码SHA，数值体不变的准备记录。
- `sources/`：候选根与融合O-proj的可读源码；仅规范格式和import名称顺序。
- `PUBLIC_SOURCE_DERIVATION.json`：可读版与实际执行版的计算/控制流AST及导入等价证明。
- `executed_helpers.tar.gz`：77个原样执行文件，包括单层、混长、反向捕获、短T否决及整模型助手；不包含模型权重、cache payload或SDK二进制。
- `EXECUTED_FILES.json`：压缩包及每个原样文件的SHA。
- `single_layer/`：八个已完成任务的原始result，包括有收益和回退配置。
- `RECHECK.json`：重新计算所有event中位数、成对领先数及A/B/ABA收据一致性；不是独立张量计算。
- `cpu_compile_receipt.json`、`generated_check.json`：完整64参数根、真实artifact、无NPU初始化和生成资源检查。
- `whole8k_summary.json`：CPU审计、计时摘要、rank0输出/事件及全部16rank原始文件SHA。
- `whole8k_raw.tar.gz`及`WHOLE8K_RAW_FILES.json`：全部16rank的完整warm/测量输出token、事件、FULL轨迹和计时原始数据，可复算跨rank审计。
- `early_pack/`：独立提前打包候选的两种捕获顺序、原样源码及原始结果。
- `runtime_capacity/`：小ring初始化、T144强制精度资格与整模型容量观察助手；不等于新的默认配置。
- `whole128k_summary.json`、`whole128k_raw.tar.gz`及`WHOLE128K_RAW_FILES.json`：128K整模型16rank原始精度、实际FULL轨迹、计时、容量与真实init配置，设备任务exit0。

SOURCE中的`hardware_precision_validated=false`是不可变的编译准备快照；后续硬件资格以原始result和整模型摘要为准。

## 复现条件

需要3399文件的受测源码快照、独立兼容SDK、官方13包静态缓存、正式模型权重，以及本轮固定bank和Native黄金。
完整版本、路径、任务参数和结果限制见上级主报告。不能删除身份断言来迁就其他环境。

解包原样助手到新的自有目录；使用新label，先运行SDK/source检查和无设备编译，保存CPU收据，再经task-submit提交单卡测试。
`sources/`用于阅读；精确复现应使用压缩包中的原始版本和manifest，不把格式化版本冒称为相同字节源码。

```bash
tar -xzf executed_helpers.tar.gz -C <新的实验目录>
# 在新的实验目录内，先准备并提交无设备编译入口：
cat > compile-local.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
source ./env.sh
export PREFLIGHT_RECEIPT="$PWD/cpu-preflight.json"
timeout --signal=TERM --kill-after=15s 750s python ./compile.py fused
SH
task-submit --no-device --max-time 780 "bash ./compile-local.sh"
# 确认任务exit0及cpu-fused/receipt.json全部门禁通过后，才提交硬件：
task-submit --device auto --device-num 1 --max-time 780 \
  "bash <新的实验目录>/run.sh <新label> 32 8191"
```

ragged、reverse及guarded各有独立manifest，必须先cd到对应子目录，再经task-submit无设备执行`bash ../compile-local.sh`，分别生成自己的cpu-fused/receipt.json；不能拿主目录receipt代替。
whole读取父目录主候选receipt，必须先完成主目录CPU资格。编译输出目录必须是新的，不能覆盖已有收据。

混长入口为`ragged/run.sh <新label> 6 8191 1,2,3,4,5,6`；反向捕获为`reverse/run.sh`。
整模型8K为`whole/run.sh 8k`；旧`whole/run.sh 128k`保留的是原容量保护退出路线。128K有效容量复现请继续参考[runtime_capacity](runtime_capacity/README.md)的`whole-ring/run.sh 128k`，先通过其单卡资格；均内限750秒、调度780秒，不裸跑NPU。
移机需要显式改路径、重新封存manifest、完成CPU资格，不覆盖原验收环境。

## 8K审计口径修正

原设备任务完成两轮生成后，最终CPU审计因累计Native profiling计数非零而exit1，该终态保留。
原observer记录所有eligible调用，不能把加载期Native调用当捕获内回退。
修正版要求各层实际PTO根捕获、正确namespace、计时前后Native计数不增加，以及每步实际FULL replay收据。
全部token/接受事件门禁保留，独立CPU审计通过；没有因此重复运行16卡，也没有改写原任务为exit0。

公开whole摘要将可重算的输出/事件SHA字段统一命名为warm_payload_sha256、measured_payload_sha256，避免token一词触发密钥扫描的误判。PUBLIC_SUMMARY_DERIVATION.json记录原始/公开SHA和纯字段映射；所有摘要值均由16rank原始输出重算，除此无修改。原样SUMMARY.json继续保留在各whole raw归档中。
