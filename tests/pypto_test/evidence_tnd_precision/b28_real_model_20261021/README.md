# 8K/B28真实整模型：T168 QA修复后的逐token验收

源修复提交：`ad06a423bc889ca1730a726303845625df29d91d`，分支
`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
详细参数、根因、运行身份及范围见[主报告](../../DSV4_FLASH_CSA_TND_T168_QA_NATIVE_ALIGNMENT_20261021.md)。

## 保存证据的CPU复核

在仓库根使用Python 3.10及以上（本轮复核为3.11）执行，不需要Torch、vLLM、模型或NPU：

```bash
python tests/pypto_test/evidence_tnd_precision/b28_real_model_20261021/verify.py
```

脚本校验归档和31文件助手SHA，安全解包到临时目录，用实际运行时未修改的`audit.py`
分别重算Native READY与PTO audit，并逐对象比较原收据。成功输出
`PASS_SAVED_REAL_MODEL_8K_B28`，172032对token位置；这不是重新执行NPU推理。

| 文件 | 内容 |
| --- | --- |
| native1-whole-v2-text-evidence.tar.gz | Native原始132个文本文件：16rank报告、token/事件/trace、gold及日志 |
| pto1-whole-v2-text-evidence.tar.gz | PTO原始132个文本文件：相同结构及audit |
| sealed-whole-tools.tar.gz | 原31文件助手、manifest、CPU收据及四组独立复核 |
| RESULTS.json | 独立复核摘要和覆盖限制 |
| VERIFICATION.json | 最终封包SHA刷新后，最新版CPU重算成功收据 |
| SHA256.json | 以上归档与摘要的身份 |

封存助手自身SHA为`5aec62d22523f9433381f1c9f362cfaba4f4d301ca3099db1f90133b7c50e23b`；
source manifest为`48ee0351ebd554c2d52c93d28ff7aa4be357cf6636a4325d247f64a1f2671539`。
归档不包含模型、cache payload、完整3399文件源码快照、SDK或编译二进制。
图模式覆盖以最终`review-delivery-whole-v2/ACTUAL_GRAPH_SCOPE.json`及本说明为准；
此前复核中“所有活跃步骤均FULL”的文字判断已由原始trace全量检查纠正，原审计和token结果不变。

## 重新执行NPU测试所需资产

复现依赖227现有自有封存资产，不是可在任意机器独立安装运行的发行包：

- 完整`/data/model/DeepSeek-V4-Flash-0731-w8a8`，75分片。
- 基目录`/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex`，下文记为`base`。
- 原源码快照`base/precision-overlap-20261020/whole-overlap-v1/snapshot`，3399文件。
- `base/root-fix-20261016/env.sh`、`base/low-acceptance-20261010`下的SDK桥接、官方静态缓存及target/MTP bank初态。
- 具体七模块import、CANN/ATB/PTOAS及版本见助手`EXPECTED_ENV.json`，逐文件运行身份见`runtime77.json`。
- 输入来自助手`bank-input-8k-qualified.json`及`prepare_bank.py`；取fixed8192 bank前28个不同key，不能替换成随机prompt。

任何缺失、SHA或import不符都应报告为环境阻塞，不能跳过门禁后继续。

## 使用新目录复现的顺序

原实际任务目录为`base/b28-precision-20261021-codex/whole-b28-qa168-v2`。
`run-arm.sh`通过`mkdir`拒绝覆盖旧输出；不要删除已有results来重跑。

1. 校验并解包助手到同一实验父目录下的新job名称，复制原31文件集合中除`compile-receipt.json`外的30文件及helper manifest。
   不复制旧`results`、`bank28`、`snapshot`、编译缓存或CPU运行收据。
2. 新job中的`env.sh`仅把`SOURCE_ROOT`改为新job的`snapshot/source`，
   `EXPECTED_ENV.json`仅把`modules.vllm_ascend.file`改为该目录的实际import路径。
   其余SDK、模型和原资产路径保持封存值。
3. 执行`prepare_snapshot.py`，校验原3399文件后独立复制，仅替换canonical QA文件。
   source manifest仍须等于上述48ee值，canonical QA SHA须为
   `ea30b2dec9bba1dc7fc0ae5d6a947a88c5eb4ef471b3291307e4524448a06e80`。
4. 确认新job没有`compile-receipt.json`，用`compile.sh`在实际新路径执行无设备编译，生成新收据。
   要求根64参数、preparer2参数、GCC/CCEC及orchestration通过，NPU未初始化。
   不得复制旧实际编译收据代替这一步。
5. 对helper manifest原来的31个文件重新计算SHA，保持文件集合和其他字段不变。
   helper SHA会因新路径和新编译收据变化，应作为新试验身份记录，不能冒用本轮5aec值。
6. 执行`cpu_prepare.sh`，先准备只读bank，再做实际import/hook预检；两项收据均须通过且NPU未初始化。
   可独立运行`test_local.py`的正负控。此阶段不装包、不改SDK。
7. 经`task-submit --device auto --device-num 16 --ptoas 0.63 --max-time 780`提交
   `bash <新job>/run-arm.sh native`。显式`--env`传入原环境的`PYTHONNOUSERSITE=1`、
   `ASCEND_HOME_PATH`、`PTOAS_ROOT`、`CCACHE_DIR`及`CMAKE_BUILD_PARALLEL_LEVEL=4`。
   `run-arm.sh`内部750秒、退出清理另有20秒宽限；不在普通SSH shell直接运行设备入口。
8. 只用status/log监控，每5分钟检查一次；不使用会超时取消排队的`--wait/--run`。
   Native退出0且`results/native1/READY_GOLD.json`通过后，才以相同方式提交`run-arm.sh pto`。
   不运行Native2。任意首差立即停止后续配置并保存完整证据。
9. PTO退出0后，重跑`audit.py --native <新job>/results/native1 --candidate <新job>/results/pto1 --output <新审计文件>`。
   要求172032对token、接受事件、完整生成窗口采集输入/入场轨迹一致，零抢占，
   每rank每轮至少一次真实28活跃/T168/FULL168，21层单根完整PTO资格全部通过。

重新复现必须生成自己的编译收据、helper身份、CPU收据、Native gold和PTO审计，
不能编辑本归档的原始数据使之匹配新路径。提交至完成的墙钟时间含排队，不作为占卡或性能数据。

## 验收边界

本轮完整模型、固定bank、TP1/DP=EP16、192 token×两轮、确定性开启、EPLB关闭。
每rank每轮首次全1行请求为NONE、真实及物理T28，另52步为FULL。
本轮未证明短query的FULL graph，也未覆盖同一步任意混合1～6行。
token一致不等同于所有隐藏张量逐bit一致；本轮未测新的128K整模型及性能。
