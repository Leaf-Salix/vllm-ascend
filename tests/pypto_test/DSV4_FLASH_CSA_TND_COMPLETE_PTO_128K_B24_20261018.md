# 完整PTO CSA：真实模型128K/B24/192-token对照

## 测试对象

分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，数学代码`425ae1f11`。
本轮使用刚通过[8K/B40整模型验证](DSV4_FLASH_CSA_TND_COMPLETE_PTO_8K_B40_20261018.md)
的同一17份完整精度包源码，直接复用冻结step12/source，没有修改数学或默认实现。
运行树是保留旧Native及测试工具的受控overlay，不宣称全部文件等于该Git提交。
原精度/性能基准、8K通过证据及之前128K记录均保留。

21个CSA层L2、L4、…、L42从HC-pre到HC-post完整替换，包含输入Norm、Q/KV、
Indexer权重投影和主体、sparse attention、O-proj、HC-post；其余20个HCA层继续Native。
请求边界仍来自Native metadata，保留真正TND。

之前[128K静态WKV候选](DSV4_FLASH_CSA_TND_STATIC_WKV_128K_192_RETEST_20261017.md)
虽然完整192 token通过，但当时O-proj仍Native；不能将该旧结果冒充本轮完整PTO验证。

## 对照方案

| 项目 | 配置 |
| --- | --- |
| 真实权重 | DeepSeek-V4-Flash-0731-w8a8，完整75分片 |
| bank | 原fixed131072，q00…q23；history131104…131346 |
| 并行/提交请求 | TP1、DP=EP16，每rank提交B24，共384请求 |
| 输出 | 每请求192 token，共73728 token，greedy/ignore EOS |
| 推测解码 | DSpark出5验6，逐项比较接受事件 |
| 图/容量 | FULL_DECODE_ONLY[6,144]，max_num_batched_tokens400，KV24000，gpu util0.97 |
| 确定性 | Torch/NPU level1，Torch deterministic，HCCL true，atomic0，NZ2，event1 |
| EPLB/入场 | EPLB关闭；沿用原Native口径，允许等待入场，记录实际并发并要求零抢占 |
| 任务边界 | 16卡，max-time780秒；复用保存Native1，不重新跑Native2 |

环境沿用CANN9.2 beta2、ATB9.2 ABI1、Python3.12、Torch2.10.0+cpu、
Torch-NPU2.10.0.post2、vLLM0.25.1及既有私有PyPTO/Simpler/PTOAS。
没有重装或重编译这些框架，实际算子cache冷构建另行统计，不作为性能测量。

Native1为`retest192-native1-b24000`，原代码`cbc810c8afab47c2425068049440424f014dcde3`。
继续使用原bank及已保存的Native输出/事件/完整入场参考，原始和派生证据逐项SHA绑定。
bank plan和24条token文件存在性沿用原检查，没有重算大型cache张量全部payload SHA，
也不宣称与nalinaly原始cache逐字节相同。

## 首差门禁及资格

- `admission_trace.py`、`generation_window.py`、`audit_alignment.py`、`saved_reference.py`
  与之前128K完整测试逐字节相同，保留全部16rank实际设备输入/入场/DP向量/图档位比较。
- `full_probe.py`与8K已通过助手逐字节相同，使用control模式：rank0每轮真实行
  target hidden、aux及MTP residual位摘要与原Native比较，首差写STOP并抛错。
- 完整输出token、所有接受事件及计数、有效窗口78轮必须精确一致；
  仅允许原判据验证的生成完成后empty tail，不能删活动轮次对齐。
- 首次实际FULL6仍要求forward74；全rank必须有FULL144及FULL6执行，零抢占。
- 每rank精确21个CSA runtime，T6/T144各有PTO O-proj非捕获暖身与真实捕获计数，
  且原capture_time_selection确认两档PTO；HCA保持Native。
- 沿用8K已验证静态助手：首次真正reselect一次并资格通知全部21层，后续每档仍经过
  原13包/SDK/SOC/确定性校验，但复用相同选择，不在捕获后真正重选。
  要求actual_reselections=1、validated_reuses>0；原bridge打印的RESELECT行不作为实际次数。

本次先做完整token/outer对照，不安装43层内部stage witness。
因此不重新声称本轮所有内部中间tensor逐bit一致；若出现outer首差，停止生成再定位内部。
只复用原Native1作为数值oracle，没有运行新的Native2，也不据此声明Native重复性新结论。

## 本轮结果

任务`task_20261018_014155_28865031713`已exit0，提交至完成9分16秒（含等待）；
实际执行仍受780秒硬上限约束，未超过15分钟。CPU严格资格器`numeric_pass=true`。

| 检查 | 结果 |
| --- | --- |
| 完整输出token | 384请求各192token，共73728个，全部与保存Native1一致 |
| 接受事件 | 全部逐项一致；num_drafts15392，draft_tokens76960，accepted_tokens58896 |
| 接受0…5直方图 | [1200,1392,864,1216,1472,9248]，与Native一致 |
| 平均接受长度 | 3.8264033264033266，与Native一致 |
| 实际输入/入场 | 全16rank完整78轮生成窗口精确相同 |
| rank0外层浮点 | 78轮target hidden/aux/MTP真实行位摘要与Native一致 |
| 实际并发/抢占 | 每rank提交24条，峰值21条同时活跃；零抢占 |
| FULL144 | 各rank实际73次FULL144；首个FULL6在forward74，后续通过至77 |
| 原始尾轮 | 各rank raw79轮，最后1轮仅为原门禁验证的生成完成后empty tail |
| 完整PTO资格 | 全16rank×21层，T21非捕获PTO O-proj各1次，T6/T144各暖身及捕获1次 |
| 静态选择 | 全rank真正reselect1次，后续验证并复用1次 |
| 缓存冷构建 | rank3生成/二进制各3次，rank11各4次，其余均0；缓存错误项全0 |
| 严格READY | strict_ready_reuse=false，未当作性能验收通过 |

**结论：本组真实128K/B24整模型、192-token场景下，HC-pre到HC-post完整PTO CSA
替换后的全部token输出与保存Native1一致；接受事件、实际入场和rank0外层位摘要也一致。**
未出现需要继续定位的首个数值差异；没有修改正式数学，没有新的Native1或Native2运行。
本轮结果与8K通过记录共同保留，但不能推广为任意配置/硬件/SDK下所有中间值的逐bit证明，
也不将带数值观察同步和冷构建的总耗时当作性能结果。

## 冻结身份与复现

- 完整精度包所属数学提交：425ae1f11（17份文件逐字节匹配）。
- 源清单SHA：56d688b218d0d634024aed98a1568adac982f6005e5cd87373dd7553df8e6496。
- 本轮helper清单SHA：9f84e645c8d282581c6e1b0a790add5daa60cbdd9fb73f988a4b7d536f6e3170。
- Native派生参考SHA：347b452c4eed32897b54bd05305a317708fbea987ffeab1ad1ac1461f9153dac。
- 原Native源清单SHA：e4de4927d2e5758df65784bddc412d1d90227b56ed978ad2218df50c2bd540ca。
- 128K bank plan SHA：b4ca67ee33bdf5f1ddadcd9578e437865e15bacfcb6fdf45094f431abb704a1e。
- 静态13包metadata SHA：863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602。

所有助手、env和门禁在本轮独立helper-v1中；先运行preflight.sh验证准确imports、
CANN/ATB/PTOAS、75权重分片、完整源清单、原Native/派生证据及助手SHA。
在227既有资产环境，用新的唯一label运行，不能覆盖原输出：

```bash
BASE=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex
JOB="$BASE/full-pto-128k-b24-20261018"
bash "$JOB/helper-v1/preflight.sh"
task-submit --device auto --device-num 16 --max-time 780 --ptoas 0.63 \
  --env PYTHONNOUSERSITE=1 \
  --env ASCEND_HOME_PATH=/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2 \
  --env PTOAS_ROOT=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/experiments/nalinaly-71153bb3/reference-env/bin \
  --env CCACHE_DIR=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/src/pypto/.ccache \
  --env CMAKE_BUILD_PARALLEL_LEVEL=4 \
  "bash $JOB/helper-v1/run-case.sh precision-witness <新label>"
# 获得确切ID后用--status/--log观察；首差即停。
# 完成后在helper-v1/env.sh对应的无设备CPU环境执行严格验收：
python "$JOB/helper-v1/qualify_control.py" \
  /data/pyptouser/yejia/lc128/<新label> --output <新的资格JSON>
```

证据包不含模型权重、venv、bank张量和全部运行源码树。复现仍需要上述冻结资产；
改变路径、SDK或内容必须显式更新绑定，不能继承本次资格结论。

## 原始证据

[严格CPU验收结果](evidence_tnd_precision/full_pto_20261016/complete-pto-128k-b24-qualified.json)
与[完整两臂证据包](evidence_tnd_precision/full_pto_20261016/complete-pto-128k-b24-evidence-v1.tar.gz)
已保存。291份文件、1979573字节，SHA256：
`73f82259ada6d3ff53785bcd79a3775acdd1dc7d8a6a59a2737bbd6dc7c90380`。
全部归档文件已按manifest在本地逐项核对长度和SHA。

包括两臂全部16rank原始token/接受事件/入场轨迹/审计、rank0outer、Native派生参考、
完整运行/资格/预检/归档助手、CPU测试、正式完整精度包及overlay增改源码、源清单、
静态包metadata和真实选择桥。没有将旧内部stage证据冒充本轮新观测。

## 提交前检查

CPU6项门禁、独立原始数据复算、修改文件增量pre-commit和Markdown检查均通过。
另在独立临时源码副本执行完整`bash format.sh ci`，仍因已有results/冻结证据的
Ruff、拼写、shell和禁止导入问题exit1，全仓库格式gate没有通过。
批量自动格式改动没有写回正式checkout或历史证据。日志保留在本轮本地
reports/dsv4-full-pto-128k-b24-20261018/full-format.log。
