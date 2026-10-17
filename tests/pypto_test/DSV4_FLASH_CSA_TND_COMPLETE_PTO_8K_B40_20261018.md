# 完整PTO CSA：真实整模型8K/B40/192-token复测

## 版本和替换范围

工作分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
候选数学代码为`425ae1f11`的`tnd_full_precision`完整包，17份Python/C++源码逐字节绑定。
运行快照是在既有冻结overlay上更新这17份文件，不宣称整个运行树等于该Git提交。
Native1、原精度基准、性能实现和此前输出均保留。

21个CSA层（L2、L4、…、L42）从HC-pre到HC-post使用完整替换链：
HC-pre、输入Norm、Q/KV投影及后处理、Indexer权重投影与其余主体、sparse attention、
O-proj和HC-post；HCA的20层保持Native。继续使用Native请求metadata、cache所有权和connector，
没有把请求统一补成S6。本组请求实际行数由入场轨迹记录，不作为新增非等长请求矩阵测试。

恢复O-proj的模块/服务逐bit对拍见
[正式WKV与O-proj记录](DSV4_FLASH_CSA_TND_FORMAL_WKV_OPROJ_20261017.md)。
本次补足该正式完整替换链的新整模型验收。

## 测试配置

| 项目 | 本次配置 |
| --- | --- |
| 模型 | DeepSeek-V4-Flash-0731-w8a8，真实75分片权重 |
| 输入 | 既有固定8192-token前缀bank，加40条公开问题的原chat后缀 |
| 并行 | TP1，DP=EP16，单rank提交B40，共640条请求 |
| 输出 | 每请求192 token，合计122880 token |
| 推测解码 | DSpark出5验6；完整接受事件逐项对比 |
| 图 | FULL decode，捕获档位[6,240]；max_num_batched_tokens400 |
| 确定性 | Native与PTO均level1、Torch deterministic、HCCL_DETERMINISTIC=true |
| 其他 | NZ2，event_work_mode1，EPLB关闭，固定KV块22000，gpu_memory_utilization0.95 |
| 任务限制 | 每项16卡，max-time780秒；不运行Native2或128K |

EPLB关闭及固定KV容量均沿用已保存Native1，不能将这组结果写成EPLB开启的验证。
公开问题内容、顺序和实际chat后缀40条均与参考材料一致；bank由本机组先前Native生成。
没有证明其完整前缀/cache张量逐字节等于nalinaly原bank；本次只复用自己的冻结Native1。
cache头部、manifest和文件stat已核验，没有将文件stat冒充张量内容全量SHA。

环境沿用Python3.12、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、
vLLM-Ascend0.25.1rc1 overlay、CANN9.2.0-beta.2、ATB ABI1和既有私有PyPTO/Simpler。
确切imports、工具链、环境路径与75分片检查保存在预检日志；没有重装或重编译上述框架。
预检阶段仅对正式O-proj做无设备lower/compile，复用既有编译缓存；
模型运行时是否发生其他operator编译，以本轮各rank实际cache计数为准。

## 判定方法：首差即停

1. 原Native1是`task_20261016_192621_7066592194`，代码`259cf9a5f`。
   本次不重新生成Native输出；核验原始结果与派生generation-reference的SHA。
2. 保持原`admission_trace.py`、`generation_window.py`、`full_probe.py`、
   `capture_boundaries.py`、`audit_alignment.py`和`qualify192.py`逐字节不变。
3. 全16rank在线比较设备实际input IDs、positions、请求键/边界、DP metadata、
   padding和入场记录。首差写STOP并抛错，不继续生成。
4. rank0逐有效轮比较target hidden、aux和MTP residual的真实行位摘要；首差立即退出。
5. 离线再比较全部640请求完整192-token序列和接受事件；检查零抢占、实际并发、
   FULL240实际重放以及原判据限定的非活跃空尾。不能随意删轮对齐。
6. 追加21层每层T6/T240的PTO O-proj非捕获暖身及真实捕获计数，确认没有回退到Native。
   计数不声称Python在graph replay中重新执行；replay执行捕获的设备图。
7. 原13个静态包、SDK、SOC和确定性配置每次加载仍完整验证。
   新助手首次真实reselect一次并通知全部runtime；后续档位校验后复用相同选择，
   禁止捕获后真正重选。验收要求actual_reselections=1且validated_reuses>0。

该模式不安装内部逐层stage hook，不能据外层位摘要一致断言所有中间tensor对任意输入都逐bit一致。
Native2和性能测量均不在本次范围内。

## 启动失败与修正

第一项`task_20261018_010753_36803907165`在第二档位准备时触发
`Do not reselect after capture`，exit1，没有生成对拍结果。
框架会为每个档位调用compile_static_kernel入口，原助手把重复入口一律当成危险重选。

独立helper-v2修正启动生命周期，helper-v1和失败输出冻结保留；CSA数学源码不变。
后续仍调用原bridge验证13包/SDK/SOC/确定性并执行原包安装流程，但底层无参reselect
返回首次成功结果，不再次改变真实选择，不重新授权已经捕获的runtime。
这与旧Native每档真实重选的启动调用序列不同；本次以旧Native数值轨迹作为oracle，
不宣称启动过程完全复刻。原bridge在复用时仍打印RESELECT success=1，
实际调用次数只认新增actual_reselections/validated_reuses，不按旧日志行数计算。

助手CPU6项通过：不带资格调用拒绝、无暖身捕获拒绝、全部21层两档位计数、
第二档位校验后不重选、修改metadata拒绝、任一runtime身份不符拒绝；独立审查通过。

## 本轮结果

任务`task_20261018_012134_402497631259`已exit0；提交至完成9分29秒（含等待），
设置的实际执行上限仍为780秒。严格CPU验收`numeric_pass=true`，不是仅按退出码或平均接受率判定。

| 检查 | 结果 |
| --- | --- |
| 完整输出token | 640条请求各192token，122880token全部与原Native1一致 |
| 全部接受事件 | 逐项一致；接受0…5计数[1696,2432,2816,2592,2240,14512] |
| 平均接受草稿数 | 3.7035909920876446，与原Native1一致 |
| 实际输入与入场 | 全16rank有效生成窗口51轮完全一致 |
| rank0 target hidden/aux/MTP | 51轮真实行位摘要全部一致 |
| 实际并发/抢占 | 每rank峰值40，零抢占 |
| 真实执行图 | 首轮T40 NONE/eager，之后各rank50次FULL240重放 |
| 非活跃空尾 | raw记录52/54轮；仅按原门禁限定的空尾1/3轮排除 |
| 完整PTO调用 | 全16rank×21层T40非捕获调用各1次，T6/T240各暖身及捕获1次 |
| 静态资格 | 各rank首次真实重选1次，后续校验后复用1次 |
| 算子缓存 | rank5生成/二进制各5次，rank9各2次；其余均0；全rank所有缓存错误项0 |
| READY资格 | strict_ready_reuse=false，未冒充严格READY性能验收 |

**结论：在本次固定8K/B40、真实完整模型、192-token场景中，HC-pre到HC-post完整PTO
CSA替换后的全部token输出与保存的Native1完全一致，接受事件、入场轨迹和rank0外层位摘要也一致。**
本轮没有出现需要继续定位的首个数值差异。原精度/性能基准保留；没有跑Native2或128K。
内部stage仅引用此前模块与服务对拍，不把本轮结果扩展成任意输入/硬件/SDK的普遍逐bit证明。
本轮有数值观察同步和启动修正，不用于报告服务吞吐或延迟改善。

## 冻结身份与复现

- 源快照清单SHA：56d688b218d0d634024aed98a1568adac982f6005e5cd87373dd7553df8e6496。
- 正式O-proj次序表SHA：2c6ed88f69024a894543228a818cc5e7a4081ffacbceebb9fec5452a2b187686。
- Native1派生参考清单SHA：be6ba496d189c91f9c40fcea5d8d73c68577d646fb34ea203add5612297c2834。
- helper-v2清单SHA：6407401173ad8762dd8f1e90a0f25ea9f90cdfd81c2038d9518b519c884ca480。
- 静态包metadata SHA：863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602。

在227保留的实验根目录：
`/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/full-pto-8k-b40-20261018`。
先运行helper-v2/preflight.sh，核对环境、bank、原Native证据、全部源码和助手SHA。
再使用新的唯一label，不能覆盖历史输出：

```bash
BASE=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex
JOB="$BASE/full-pto-8k-b40-20261018"
task-submit --device auto --device-num 16 --max-time 780 --ptoas 0.63 \
  --env PYTHONNOUSERSITE=1 \
  --env ASCEND_HOME_PATH=/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2 \
  --env PTOAS_ROOT=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/experiments/nalinaly-71153bb3/reference-env/bin \
  --env CCACHE_DIR=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/src/pypto/.ccache \
  --env CMAKE_BUILD_PARALLEL_LEVEL=4 \
  "bash $JOB/helper-v2/run-case.sh precision-witness <新的label>"
# 获得确切任务ID后，用--status/--log持续监控；首差即停。
# 完成后在helper-v2/env.sh对应CPU环境执行：
python "$JOB/helper-v2/qualify_production.py" \
  /data/pyptouser/yejia/lc128/full-pto8k-native-20261016 \
  /data/pyptouser/yejia/lc128/<新的label> --output <新的验收JSON>
```

证据包不包含模型权重、venv、完整bank张量或完整运行源码树，复现仍依赖上述冻结资产。
迁移路径/模型/SDK后须重新显式绑定身份，不能直接沿用本次资格结论。

## 原始证据

[完整CPU验收JSON](evidence_tnd_precision/full_pto_20261016/full-pto-8k-b40-qualified.json)
与[两臂原始证据包](evidence_tnd_precision/full_pto_20261016/full-pto-8k-b40-evidence-v1.tar.gz)
已保存。归档355份文件，2551666字节，SHA256：
`f70231d46ef851a0a745b25c397677919c155f55c702d728a13f14d2317ef628`。
全部文件的长度和SHA已在本地按归档manifest逐项核对。

包括Native1及本轮全部16rank结果、完整事件、输入轨迹与审计、rank0outer摘要、
Native派生参考及SHA绑定、正式包和overlay增改源码、两版助手、CPU测试、预检/验收/
归档脚本、任务日志，以及首次启动失败的原始记录。未重写旧基准或将失败记录记作精度结果。

## 提交前检查

助手CPU6项、独立原始数据审查、修改文件增量pre-commit和Markdown检查均通过。
另外在独立临时源码副本运行`bash format.sh ci`，仍因既有results/冻结证据中的
Ruff、拼写、shell和禁止导入问题exit1；全仓库格式gate没有通过。
批量自动修复只发生在临时副本，没有写回正式checkout或重写历史原始证据。
完整日志保存在本轮本地reports/dsv4-full-pto-8k-b40-20261018/full-format.log。

## 同一完整PTO代码的128K对照

后续[128K/B24、192-token对照](DSV4_FLASH_CSA_TND_COMPLETE_PTO_128K_B24_20261018.md)
也已通过：384条请求、73728token全部与原Native一致，78轮入场/外层位摘要及接受事件一致。
每rank提交24条、实际峰值21条，零抢占；不重跑Native2，数学源码与本组8K相同。
