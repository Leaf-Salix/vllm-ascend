# 正式静态 WKV 接入与 PTO O-proj 恢复

## 基准和范围

当前工作分支为 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
原基准 `b1fcd9653` 已保存到 Leaf-Salix 备份分支：
`backup/csa-full-precision-before-static-wkv-20261017`。
旧 `tnd_precision`、性能版和冻结 step09 源码均不修改。

本次首先把已通过完整 128K/B24/192-token 复核的 WKV 选择规则移入
`tnd_full_precision`。之前完整验证见
[静态 WKV 128K 复核](DSV4_FLASH_CSA_TND_STATIC_WKV_128K_192_RETEST_20261017.md)。
那份整模型结果属于旧冻结代码加独立修正助手，不能冒充本次正式代码的新整模型测试。

## 正式 WKV 规则

- 原 `WKV_TOKEN_MODES`、次序表和 WKV 算术保持不变。
- 每个 `CSAServiceRuntime` 持有自己的 `WKVProfile`，不修改模块全局次序表。
- 启动器完成原 13 个静态包、SDK、SOC、确定性校验，并确认真实
  `_aclnn_reselect_static_kernel()` 成功返回后，调用每层
  `qualify_static_wkv(metadata_sha256=..., reselected=True)`。
- SHA 固定为已验证配置的
  `863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602`。
  SHA 是资格身份，调用者仍须执行完整环境和包内容校验。
- 有资格时仅 T=6 使用既有 mode1（自然 K256 次序）；其他 token 档位保持原表。
- 未安装阶段的 eager 保持旧标定；仅开启 `enable_static_kernel` 不产生资格。
- 静态图捕获缺资格直接报错，防止悄悄使用旧 T6 次序；首次捕获后禁止迟到资格。
- 已捕获图之后不得重新 reselection。独立加载桥在执行实际 reselection 前检查所有
  runtime，发现已有捕获即拒绝；需要更换静态包必须重建 runtime 和图。

资格通知是显式启动契约，需要配合验证加载桥；不能仅设静态配置便宣称精度已验证。
没有引入新的环境变量、设备到主机同步或热路径包哈希检查。

## 本次第一项 NPU 对拍

任务 `task_20261017_203711_209134330793`，单卡，`--max-time 780`，exit=0。
使用真实模型 L20、保存的第一次 FULL6 heads/Norm 输入，非新整模型 Native2。

| 检查 | 不同元素 | 结论 |
| --- | ---: | --- |
| 正式 WKVProfile 选择 + PTO WKV eager 对 Native | 0 | 逐 bit 一致 |
| 正式 WKVProfile 选择 + PTO WKV Graph replay 对 Native | 0 | 逐 bit 一致 |
| 相同 heads 经 Native O-proj 对保存的整模型 O-proj | 0 | 参考保真 |
| PTO WO-A BF16 eager 对实际 Native BMM 返回 | 0 | 逐 bit 一致 |
| PTO WO-A BF16 Graph replay 对实际 Native BMM 返回 | 0 | 逐 bit 一致 |

保存输入 `step074.pt` SHA256：
`5b873509016a7419abf5639a808da58950eb2d3d3bd332c00ed8625e00174481`。
源快照清单 SHA256：
`d06950a541ebe545372e29366be5935044af65c411ad1daa97cf5c5226af5d44`。

这项仅使用一组真实 T6 输入，尚未证明完整 PTO O-proj 已对齐，更不能推断所有层、
所有档位。后续按 WO-A BF16、整行 INT8/scale、WO-B BF16 的顺序检查，首差立即停止。

## 环境及复现

沿用前述 128K 复核的 Python 3.12、Torch 2.10.0+cpu、Torch-NPU 2.10.0.post2、
vLLM 0.25.1、CANN 9.2.0-beta.2、ATB ABI1 和冻结 PyPTO/Simpler。
模型为 `DeepSeek-V4-Flash-0731-w8a8`，TP1，确定性 level1，NZ2，atomic=0。
任务私有 OPP 安装已校验的 13 个静态包，不修改共享 CANN。

每项提交都使用：

```bash
task-submit --device auto --max-time 780 --ptoas 0.63 \
  --env PYTHONNOUSERSITE=1 \
  --env ASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  --env PTOAS_ROOT="$PTOAS_ROOT" \
  --env CCACHE_DIR="$CCACHE_DIR" --env CMAKE_BUILD_PARALLEL_LEVEL=4 \
  "bash $BASE/wkv-production-oproj-20261017/oproj-v1/run.sh formal-wkv-oa1"
```

路径和精确依赖沿用被引用报告；诊断助手、清单和原始 JSON/日志在本次证据包内。
先运行相同环境下的 `source_preflight.py`，检查 imports 和完整源码清单后再提交。
所有 NPU 操作由 task-submit 执行。

CPU 回归命令：

```bash
pytest -q tests/pypto_test/test_full_precision_wkv_cpu.py \
  tests/pypto_test/test_full_precision_wkv_profile_cpu.py \
  tests/pypto_test/test_full_precision_eligibility_cpu.py
```

结果 16 passed。独立审查通过；修改文件 pre-commit 全通过。
`bash format.sh ci` 仍因既有证据/格式/拼写问题失败，日志已保存；其修改的 346 个
无关文件已恢复，未重写历史原始证据。

WKV 证据归档：`evidence_tnd_precision/full_pto_20261016/formal-wkv-evidence.tar.gz`，
21 个文件，SHA256：
`e4d95bef5c68224d143545ae694e1ca70ff4b85a0327869896b6a6e60c3e4bea`。
其中整模型 `wkv-service-v1` 是新显式资格 API 的复现加载助手，本次没有重新执行其完整
128K 整模型任务；上表数值来自已完成的单卡 `oproj-v1`。

## PTO O-proj 逐阶段进展：先定位首差

本段实验仍在独立候选，不修改正式 O-proj 调用。正式精度版的 WKV 修正为
`57cb732f6`；O-proj 暂时继续复用 Native，直至跨形状误差修正完成。

所有任务单卡、上限 780 秒；使用同一静态包资格、真实 L20 权重和保存的真实 heads。
Native 的同输入 O-proj 必须先逐 bit 复现保存的整模型输出，之后才比较 PTO。
未新增整模型 Native2，也未运行新的 128K 大配置。

| 候选 | 任务 | 首差/结果 |
| --- | --- | --- |
| v2 | `task_20261017_204137_220302120329` | 编译前变量类型检查失败，未比较数值 |
| v3 | `task_20261017_205119_23835115161` | T6：WO-A、INT8、scale 一致；WO-B 输出 1 元素差 1 BF16 ULP |
| v4 | `task_20261017_205800_250699231052` | 只交换两个 scale 的顺序，原首差消失，但另一元素差 1 ULP |
| v4b | `task_20261017_210209_258344519210` | 先合并 FP32 scale，再乘整数点积；eager 和 Graph A/B/A 全部逐 bit 一致 |
| v5 | `task_20261017_210446_26335847493` | 再去掉非 Native 量化下限；41 项检查全逐 bit 一致 |
| v6 | `task_20261017_210752_2691503387` | task-submit 附带 device 参数未被 CLI 接受，启动退出，未比较数值 |
| v6a | `task_20261017_211006_272968021910` | 真实 126 行补到原 FULL144 档位，首差在 WO-A：434 元素不同；立即停止 |

### 已定位并验证的两个边界

1. WO-B 反量化：Native 先计算 `FP32(channel_scale * token_scale)`，再乘
   `FP32(INT32_dot)`，最后 RINT 到 BF16。不能逐次用 dot 乘两个 scale。
   8 个实际 Native 有限值样本区分了这三种计算顺序，均支持合并系数路径。
2. Native 动态量化没有旧候选的 `1e-12` 下限。直接观测零行得到 scale=0、INT8=0；
   非零微值按实际 amax 计算 scale。候选仅在量化分母副本为零时改成 1，原 scale 保持零。

v3 首差为 `[0,198]`：PTO `-0.1015625`、Native `-0.10205078125`。
该点整数点积为 -5003；通道 scale 为 0.004058837890625，token scale 为
0.005013533402234316。先乘通道 scale 恰好落在 BF16 舍入中点。
v4 的新首差为 `[3,2158]`，说明交换顺序也不能复现 Native 的合并系数边界。

v5 覆盖 eager A、Graph A/B/A、零行、真实 heads 缩放 `2^-40` 与 `2^-20`；
逐阶段比较 WO-A BF16、INT8、FP32 scale、最终 BF16，并检查输出保护区、输入和权重不变。
独立审查 11 份生成 C++：amax、dequant 和分母副本分配独立；设备标量读写的 V/S
同步完整；合并系数及最终 RINT 顺序正确；阶段 RAW 依赖和尾行 valid shape 完整。

### 当前未解决：FULL144 的 WO-A

v6a 使用保存的 `step010.pt`：126 条 token 行，补到原图的全局 144 行容量，
不改变请求边界，也不是把每个请求补齐到 S6。输入文件 SHA256：
`8ded12000a7f58a5567d28b7f57a1baf334a4de8b6d969174ac8aca456f256b6`。
Native 同输入结果先复现了保存的整模型 O-proj，PTO 首差才在 WO-A 被发现。

首差坐标 `[0,3720]`，PTO 为 1.0967254638671875e-5，Native 为
1.0907649993896484e-5；WO-A 总共 434 元素不同，max_abs=0.001953125。
未继续验收后面的量化、WO-B 或整模型。T6 的自然 K256 次序不能外推 FULL144。
后续仅定位 Native 按输出组/列块变化的累加次序，再回到同输入复测。

完整阶段诊断复现命令与第一项相同，替换助手目录和 label：

```bash
bash "$BASE/wkv-production-oproj-20261017/oproj-v5/run.sh" oproj-zero5
bash "$BASE/wkv-production-oproj-20261017/oproj-v6a/run.sh" \
  oproj-real126-v6a --layer 20 --step 10 --pad 144
```

这两条均需通过同样的 `task-submit --device auto --max-time 780` 提交。
按历史 Native 输入 SHA 与原静态包清单校验后执行，首差保存并退出。

本阶段归档 `evidence_tnd_precision/full_pto_20261016/oproj-stages-through-full144.tar.gz`，43 个文件，SHA256：
`64f61bcf561beb3a9d8c8bf823fe40754f9c226fa3d8518c42120d155b2fbc19`。
包括成功、失败的原始 JSON、日志、版本化助手及生成 C++ 证据包。
