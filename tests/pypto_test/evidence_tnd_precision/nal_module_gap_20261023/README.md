# 模块差距定位与H3优化证据

对应[报告](../../DSV4_FLASH_CSA_TND_NALINALY_MODULE_GAP_AND_HC_OVERLAP_20261023.md)。控制为aad137911，nalinaly为cafcadec96。`FILES.json`封存本目录收据及归档SHA；它不包含自身和本文，避免自引用。

## 文件

- `TIME_8K_B32_V1.json`：Native/我方原组合/nalinaly单SPMD同组完整event样本与精度。
- `MILESTONE_ANALYSIS.json`：同窗不重叠阶段；另两份STAGE分析包含函数跨度，不可简单相加。
- 四份`*-evidence.tar.gz`：原始L1、独立dep及函数表；计时与dep未同时开启。
- `single_layer/h3`：H3原/反向8K和128K完整结果。
- `single_layer/checked`、`resident`、`stats-v4`、`prefetch-v2`：独立诊断，**不表示这些候选已正式合入**；`accuracy_executed=false`不能解释为精度通过。
- `h3_whole8k_raw.tar.gz`、`h3_whole128k_raw.tar.gz`：16rank原始token、接受事件、FULL资格和计时，含audit及configuration。
- 两份`h3_whole*_summary.json`：由原始rank数据按对齐步取最大值重算，可用归档内`summarize_whole.py`复核。
- `PUBLIC_SUMMARY_DERIVATION.json`：公开summary把两项轨迹SHA字段命名为`warm_trace_digest`和`measured_trace_digest`，值不变；保留原始summary的SHA及字段映射。它们是JSON内容摘要，不是凭据；原字段保留在raw归档中。
- `executed_sources.tar.gz`：实际运行manifest内的脚本/DSL副本、whole助手及重算脚本；不含模型、bank大文件、SDK二进制或编译输出。
- `ACTUAL_CONTROL.json`：实际原组合版身份；助手`SOURCE.json`中部分baseline/checkpoint字段是继承的历史谱系，不是本轮实际控制。已运行文件未事后改写。
- `FORMAL_H3_PROOF.json`：单因素受测H3与正式H3的函数AST关系及其他包文件不变证明。

## 227复现资产

共同根为`/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex`，本轮独立目录`nal-module-gap-20261023-codex`。

| 资产 | 路径/身份 |
| --- | --- |
| 原控制源码 | `b28-precision-20261021-codex/whole-b28-qa168-v2/snapshot/source`，3399文件，manifest `48ee0351ebd554c2d52c93d28ff7aa4be357cf6636a4325d247f64a1f2671539` |
| SDK | `/data/pyptouser/yejia/csa-spmd-sdk-20261020`，不覆盖原SDK |
| 模型 | `/data/model/DeepSeek-V4-Flash-0731-w8a8` |
| 官方静态包 | `low-acceptance-20261010/official-static-cache`，13包，metadata SHA `863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602` |
| 8K bank/Native | `low-acceptance-20261010/banks/fixed8192`；`/data/pyptouser/yejia/lc128/full-pto8k-native-20261016` |
| 128K bank/Native | `low-acceptance-128k-b24-20261014/banks/fixed131072`；`/data/pyptouser/yejia/lc128/retest192-native1-b24000` |

8K使用22000 KV块和默认ring；128K使用24000块及`[56,96,176,1]MiB/window2048`，仍保留2GiB剩余门禁。whole preflight核对bank plan、16rank Native gold、5个Native数值源文件和实际SDK导入；不声称重新哈希全部大cache payload。

## 命令

在227上使用现有不可变目录、新label，不覆盖旧结果。不占设备的`compile.sh`已完成且生成收据；若从归档重建，要在独立目录重新生成manifest并编译，禁止拿旧CPU收据替代变化后的源码。

```bash
gap=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/nal-module-gap-20261023-codex
task-submit --max-time 780 --env PYTHONNOUSERSITE=1 --device auto --device-num 1 \
  "bash $gap/h3/run.sh repro-h3-8k-b32 32 8192 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6"
```

反向增加`--reverse-capture`。128K/B16使用history131072、16个6并加`--qualify-boundary`。只通过task-submit占卡，禁止裸跑。观察用`--status`/`--log`，不用会超时取消排队项的`--wait`。

whole实际命令如下；同一输出目录存在时会拒绝执行，复现必须用独立助手目录/结果名并更新manifest。先8K通过后才128K。

```bash
task-submit --max-time 780 --env PYTHONNOUSERSITE=1 --device auto --device-num 16 \
  "bash $gap/h3-formal/whole/run.sh 8k"
task-submit --max-time 780 --env PYTHONNOUSERSITE=1 --device auto --device-num 16 \
  "bash $gap/h3-formal/whole/run.sh 128k"
```

配置为192 warm token＋192 measured token、TP1/DP=EP16、NZ2、FULL decode graph、确定性开启、EPLB关闭；不是在线P/D或EPLB验收。小型阶段为真实层权重＋合成hidden/history，whole为真实完整模型，两类证据分别记录。
