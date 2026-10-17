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

本段记录早期独立候选阶段，当时没有修改正式 O-proj 调用。WKV 修正为
`57cb732f6`；后文跨形状及实际服务验收通过后，才将 O-proj 改为 PTO。

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

### FULL144 的 WO-A 首差（历史记录，后续已解决）

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

## 正式 PTO O-proj：K16 次序与跨 T 验证

后续没有放宽误差门槛。Native 同输入先复现保存的整模型 O-proj 后，
继续对照 Native 的计算和舍入边界；任何首差均立即退出。

### WO-A 首差根因

Native 在当前 A3/CANN9.2 静态环境中，按输出组/列块使用不同的 K 循环起点。
FULL144 的起点按 K16 编号为
`0, 28, 56, 84, 112, 140, 168, 196, 224, 252`，对应 K 元素步长 448。
K256 或 K128 的粗粒度查询会丢掉不在块边界上的起点，不能据此复现 Native 累加。
例如此前首差所在组/列块的起点是 K=4032，粗粒度结果却被识别成从 K=0 开始。

| 候选 | 任务 | FULL144 WO-A 不同元素 | 后续阶段 |
| --- | --- | ---: | --- |
| 自然 K256 | `task_20261017_211006_272968021910` | 434 | 首差即停 |
| 按 Native 粗 K256 次序 | `task_20261017_212417_299644327832` | 302 | 首差即停 |
| 按 Native 粗 K128 次序 | `task_20261017_215528_364403419661` | 189 | 首差即停 |
| 按 Native K16 循环起点 | `task_20261017_220332_392168232583` | 0 | 41 项全部逐 bit 一致 |

K16 查询任务 `task_20261017_220013_378455021959`：256 个循环候选、86 个三元组、
258 个单点 case。N16 内分列打包成 17 批；每批检查 Native 重复、各行结果、一次列旋转
及反旋转。候选最终唯一，N128 内的起点相同。
这证明的是有限查询在该候选族中的可区分性，不能单独证明任意输入的数值一致。
随后真实 heads/权重对拍才用于验收实际候选。

T6 的同一 K16 候选也通过，任务 `task_20261017_220559_397548512227`。
两项均覆盖 WO-A BF16、整行量化 INT8/FP32 scale、WO-B BF16，包含 eager、
Graph A/B/A、zero、真实 heads 缩放 `2^-40` 和 `2^-20`，各 41 项逐 bit 一致；
输入、权重和输出保护区不变。

### T1..256 的 Native 次序标定

每段均为独立单卡任务，上限 780 秒。保留每个 M16/group/N128 的起点，不预先假设
所有 M16 相同；各 M16 内部行一致和 N128 内部一致都有硬门禁。

| 实际 T 范围 | 任务 | 结果 |
| --- | --- | --- |
| 1..64 | `task_20261017_220825_406288530724` | 全通过 |
| 65..128 | `task_20261017_221022_41239213646` | 全通过 |
| 129..192 | `task_20261017_221305_418887118832` | 全通过 |
| 193..256 | `task_20261017_221545_6271720029` | 全通过 |

合并后的原始映射 SHA256：
`2c6ed88f69024a894543228a818cc5e7a4081ffacbceebb9fec5452a2b187686`。
仓库保存为 `evidence_tnd_precision/full_pto_20261016/oproj-native-order-map-v1.json.gz`；
SHA 指解压后的 JSON。正式表去重为 16 个模式，保留固定 `[16,8,8]` 起点描述符。
仅加载时创建 INT32 tensor；运行时按实际 T 选择，PTO 内按
`K=((start+step)%256)*16` 累加。无效尾 M16 不执行。

这是该 SDK、SOC、确定性及已验证静态包下的限定标定，不外推其他环境。

### 最终接入形式的真实权重测试

沿用同一真实 L20 权重；保存的真实 heads 按各 T 重复/裁剪形成输入。
先以原 FULL144 输入校验 Native 组件能够复现保存的整模型输出，再逐 T 生成同输入
Native gold。每 T 独立 capture，eager 和 Graph A/B/A、zero/tiny/small 均逐阶段首差即停。
这属于真实权重模块测试，不能冒充这些 T 的整模型 token 测试。

| T | 任务 | 各阶段及 Graph |
| --- | --- | --- |
| 1,6,7,16,17,31,32,33 | `task_20261017_221856_1995156669` | 全部逐 bit 一致 |
| 63,64,65,96,127,128,129,144 | `task_20261017_222113_276895632` | 全部逐 bit 一致 |
| 145,192,193,240,255,256 | `task_20261017_222345_35891917582` | 全部逐 bit 一致 |

22 个实际 T 覆盖尾行及 M16 边界，包括非 6 倍数；O-proj 不要求每个请求补到 S6。
请求边界仍使用原 TND metadata，未改变其定义。

### 实际服务链：禁止 Native O-proj 回退

任务 `task_20261017_222550_41779224907`，exit=0，单卡上限 780 秒。
直接调用候选 `CSAServiceRuntime`，使用保存的真实 L20 FULL6 输入和 cache/state，
授予已验证的静态资格后，将 `impl._forward_o_proj` 替换为抛异常的函数。

| 检查阶段 | eager | 两次 raw Graph replay |
| --- | --- | --- |
| Norm、WKV、Indexer 权重 | 逐 bit 一致 | 逐 bit 一致 |
| heads、O-proj、TopK | 逐 bit 一致 | 逐 bit 一致 |
| HC-pre 到 HC-post 输出 | 逐 bit 一致 | 逐 bit 一致 |
| Native O-proj 调用 | 0 | 0 |

cache 保留原共享 allocation/stride/page 关系，调用前恢复初始值，并检查外侧保护区。
本项只有同 A 的 eager/两次 replay，不包含 B/ABA 或全部写后 cache 的逐 bit gold 比较。
不能据此宣称新的整模型 192-token 测试通过。

step11 是带白名单的独立源 overlay，3364 文件的清单 SHA256：
`7b65a580f12ac382b65f2345e923b7c1c8c525e166e4936a5e83354aff35bcc0`。
正式 O-proj kernel/表与已测副本 AST 一致；service 仅删除不再使用的 `impl` 赋值。
旧 step09/step10、性能版和原精度基准保留。

### 正式接入的资格与范围

- 完成已验证的静态包安装及实际 reselection 后，显式调用现有
  `qualify_static_wkv`；O-proj 使用同一冻结环境资格。
- 未授予资格时，适用的非 capture 预热回退完整 Native 半层；若此时尝试捕获，明确报错，
  防止悄悄录入 Native 图。资格后先完成各 bucket 的 PTO 预热，再 capture。
- 正式服务的执行范围现为 HC-pre、输入 Norm、WKV、Q/KV 后处理、Indexer、
  sparse attention、O-proj、HC-post；这些计算均由 PTO 路径执行。
  原生 metadata 生产、cache 管理及 connector 生命周期保留。
- 限定 A3、CANN9.2、TP1、NZ2、已验证 W8A8 对称无 bias 的 WO-B、T1..256。
  原 metadata 不满足替换条件时仍走 Native；未扩展 HCA。
- 本轮没有新的 128K 或整模型 Native2，没有给出新整模型吞吐结论。

### 复现小任务

下列命令均通过前文同环境的 `task-submit --device auto --max-time 780` 提交，
为每次复测更换 label，保留历史输出。

```bash
H=$BASE/wkv-production-oproj-20261017
# Native 逐 T 标定；其余分段为65..128、129..192、193..256。
bash "$H/oproj-order16-range-v1/run.sh" <新label> \
  --layer 20 --step 10 --pad 144 --begin 1 --end 64
# 最终起点核的真实权重对拍，每项最多8个T。
bash "$H/oproj-sweep-v1/run.sh" <新label> \
  --layer 20 --step 10 --pad 144 --sizes 1,6,7,16,17,31,32,33 \
  --orders-sha 2c6ed88f69024a894543228a818cc5e7a4081ffacbceebb9fec5452a2b187686
# 实际服务链；输入/cache SHA在脚本中固定校验。
bash "$H/full-service-v1/run.sh" <新label> \
  782f75c8ccbce27810d56ba16c9969990f0ab7ccb90bb41953fb255f0a77ea42
```

补测任务 `task_20261017_222932_49787926388` 已 exit=0：L2、L42 各自的 T6/T144
同输入 Native 对拍及 Graph A/B/A、zero/tiny/small 也全逐 bit 一致，各 82 项。
与 L20 合计覆盖前、中、后三个 CSA 层；未把三个代表层写成全 21 层中间值验收。

原始证据归档 `evidence_tnd_precision/full_pto_20261016/oproj-final-evidence-v1.tar.gz`，
108 个文件，454093 字节，SHA256：
`87f4e5d3559d413673f0553489d58fb5facf6882238b3ec0977ef623a0e02407`。
包括成功/失败原始 JSON 与日志、实际远端助手、四段 Native 源表、合并映射、
step11 清单/已测源码和 v11 生成 C++。真实权重三组为 322、322、242 项比较；
实际服务链为 22 项；全部通过的条目不同元素为 0。

CPU 命令 `pytest -q tests/pypto_test/test_full_precision_*_cpu.py` 当前 22 passed；
修改文件增量 pre-commit 和独立审查通过。

本轮另在临时源码副本执行完整 `bash format.sh ci`，仍因既有 results/冻结证据中的
格式、拼写、shell 和禁止导入问题失败；没有把批量自动改动写回正式 checkout。
本次改动文件的增量检查、手工 Markdown 检查均通过。
完整 CI 日志保存在本地本轮 reports 目录；这项全仓库 gate 未通过，不能记成全 CI 通过。

## 完整替换后的整模型8K回归

正式`425ae1f11`完整精度包已完成
[真实整模型8K/B40、192-token复测](DSV4_FLASH_CSA_TND_COMPLETE_PTO_8K_B40_20261018.md)：
640条请求、122880token全部与原Native1一致，接受事件、16rank入场轨迹及rank0外层
位摘要也一致；全21CSA层确认完整PTO捕获。复用旧Native，不跑Native2或128K。
加载助手的第二档静态选择复用修正、实际缓存计数及验证边界见该报告。
