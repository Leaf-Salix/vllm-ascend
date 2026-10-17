# 128K/B24静态WKV修正候选完整192 token复核（2026-10-17）

## 状态

**完整192 token数值复核通过。** 任务`task_20261017_183144_34519299596`正常exit0；
GPU锁0～15已全部释放，CPU无可见NPU独立资格器也exit0。
本轮验证的是已装载静态核资格下的独立WKV修正候选，正式默认基准继续保留。

## 实测结果

| 项目 | 结果 |
| --- | --- |
| 完整请求/token | 16rank×24条=384请求；每条192 token，共73728 token，逐项与Native1相同 |
| 接受事件 | 全部逐步事件及计数与Native1相同，num_drafts15392、num_draft_tokens76960、num_accepted_tokens58896 |
| 平均接受长度 | 3.8264033264；0～5接受长度均覆盖，直方图`[1200,1392,864,1216,1472,9248]` |
| 实际输入/入场 | 全部16rank完整78轮生成窗口相同；每rank峰值21条请求，零抢占 |
| 首FULL6之后 | checkpoint仍在forward74；已继续通过forward75～77，不再停在首FULL6 |
| 内部浮点 | rank0完整78轮、43层全部已观测边界及outer hidden/aux/MTP与已存Native逐bit一致 |
| 与前次候选 | 前75轮stage完全相同，完整stage JSON SHA也与旧Native逐字节相同 |
| 实际图/替换 | 全部rank有FULL144与FULL6，精确21个CSA层均static-ready后T6 mode1，6/144实际PTO选择通过 |
| 原始尾轮次 | 每rank raw79轮；完整生成窗口78轮，额外1轮仅为经原门禁验证的生成完成后empty tail |
| NPU时间 | 19:03:52启动、19:12:33摘要完成，约8m41s；任务max780；此前约32分钟等待未占卡 |
| 新Native运行 | 0；没有重跑Native2 |

raw尾轮只允许既有严格empty-tail契约；生成窗口内部idle、活动轮次及全部位置/草稿输入均未删除。
上一轮只证明首75轮，本轮增加了完整192输出token和接受事件证据；内部逐bit观测范围仍仅rank0，
不能把all16 token一致表述为已直接观测all16内部浮点。

### 缓存资格与性能范围

数值通过，**严格READY热复用门禁未通过**，不据本轮报告性能收益。
例如rank0记录ready_hits=0、object_hits=372、generated_hits=1、generation_builds=0、binary_builds=0；
另有rank5的generation/binary各4次、rank13各2次，其余rank这两项为0；
全部rank的bypass/invalid/storage_error/forced_rebuild均为0。
这与数值门禁分开保留，未放宽或伪报strict_ready_reuse。

## 对象与前提

- 分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`；记录基于`fa2d1541b`，正式数学`c1ef141ae`不改。
- 沿用先前[首次FULL6定位与修正记录](DSV4_FLASH_CSA_TND_FULL6_LOCALIZATION_20261017.md)中的独立候选。
- WKV修正helper与已通过候选逐字节相同：只有原13包/SDK/确定性/摘要验证及实际reselect成功后，T6才选原有自然K256 mode1；其他形状仍旧表。
- 保留真正TND请求边界；当前PTO包括HC-pre、输入Norm、WKV投影、Indexer权重投影、CSA主体和HC-post；O-proj仍Native，非CSA层Native。
- 本次不跑新的Native1/Native2，只使用原Native1完整输出/入场参考及已保存的Native内部78轮gold。
- 本次不测性能、不扩配置、不改变正式默认；原失败和首FULL6成功证据均保留。

## 相同测试配置

| 项目 | 值 |
| --- | --- |
| 真实权重 | DeepSeek-V4-Flash-0731-w8a8，完整75分片 |
| 输入 | 原fixed131072 bank，q00～q23，history131104～131346 |
| 并行/请求 | TP1、DP=EP16，每rank提交B24，允许等待入场；记录实际活跃数 |
| 生成 | 每请求192 token、greedy、ignore EOS、DSpark出5验6 |
| Graph/容量 | FULL_DECODE_ONLY `[6,144]`、max_num_batched_tokens400、KV24000 blocks、gpu util0.97 |
| 确定性 | Torch/NPU level1、HCCL true、atomic0、NZ2、event1；EPLB关闭，保持旧精度参考 |
| 环境 | 同旧CANN9.2 beta2、ATB9.2 ABI1、Torch2.10.0/Torch-NPU2.10.0.post2、vLLM0.25.1，既有私有PyPTO/Simpler |
| 替换/观测 | 21个CSA层2,4,…,42使用精度候选；rank0实际43层边界观测，其他rank比输入/输出token及事件 |

## 相比已通过首FULL6的唯一运行变更

独立新helper目录`retest128k-wkv-20261017/witness-v1`。
`qualified_wkv.py`、admission trace、copy-boundary和L20 snapshot完全不变；
只去掉forward74通过后的主动抛错，保留该checkpoint，然后继续同一78轮参考窗口。
每轮输入、全部stage、outer首差抛错与原值保存门禁不变。

最终`complete.json`必须78轮；5种CPU门禁测试证明checkpoint-only、75轮伪complete、78轮未完成和79轮多余均拒绝，只有78轮完整通过。
独立review已核验运行helper及专用`qualify_full.py`。后者复用原collect/extract和身份检查，不降低容差。

## 通过标准

1. all16实际输入、请求顺序/入场、query bounds、position、padding、联合DP向量与Native一致；无抢占。
2. 完整384请求、73728输出token，以及所有接受事件/统计与Native1逐项相同。
3. rank0全部78轮、43层的已观测字段摘要与保存Native gold逐bit一致，outer hidden/aux/MTP一致。
4. 每rank21个精确CSA层均在真实static-ready后捕获T6 mode1，6/144实际使用PTO，非CSA保持Native。
5. 冻结源码、helper、原Native/派生参考逐项SHA核验；SDK/runtime身份门禁通过。
6. READY复用与冷构建单独报告，不把缓存构建情况当作数值通过或性能结论。

## 复现资产

- 冻结源码清单：`d997d41080882209007134564ca488ce19cfe7f136397b6fd0e389c929001638`。
- helper清单：`c3841880a1aff68a7c8a56a9c5cb93940ca66d84601462e7456bcfcbbb9134a0`。
- Native内部stage：`87fa6b787a138a58cafc9b18ddb10554c8ce08b9064f831ecd0a6a105d8a5945`。
- 原Native派生参考：`347b452c4eed32897b54bd05305a317708fbea987ffeab1ad1ac1461f9153dac`。
- 静态13包metadata：`863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602`。

在同一227资产环境，先运行新helper的`preflight.sh`，然后用新输出标签提交：

```bash
task-submit --device auto --device-num 16 --max-time 780 --ptoas 0.63 \
  --env PYTHONNOUSERSITE=1 \
  --env ASCEND_HOME_PATH=<预检确认的CANN路径> \
  --env PTOAS_ROOT=<预检确认的PTOAS路径> \
  --env CCACHE_DIR=<既有checkout的.ccache> \
  --env CMAKE_BUILD_PARALLEL_LEVEL=4 \
  "bash <新helper目录>/run-case.sh precision-witness <新标签>"
```

运行成功后，在同一环境CPU调用`qualify_full.py <输出目录> --output <新结果JSON>`。
不能绕过task-submit裸跑NPU。大模型/cache/原值tensor不入Git，小证据和helper将归档并逐SHA绑定。

## 结果身份

| 证据 | SHA256 |
| --- | --- |
| qualified.json | `eba30239ced18a001f0651d4f03fcde64988fcffdeceeb1cef7835d650747353` |
| full-probe complete.json | `73ac10cadfff2fbf321c67bfbc4ca64b10b5723e04a845b0d475c436183a711c` |
| 完整stage JSON（与Native同SHA） | `87fa6b787a138a58cafc9b18ddb10554c8ce08b9064f831ecd0a6a105d8a5945` |
| 完整outer JSON | `3f582c0d08046decb72c9dec33dbc479c0316ecc1d76d31f345541084323ec47` |

正式源码、性能版及原失败/成功基准均未覆盖；本轮只新增独立helper及证据。
准备阶段目录未创建导致一次rsync/preflight文件缺失，未占卡、不计数值失败；修复后无设备预检通过。
本地5种完整轮次门禁与控制流/原代码身份检查通过，独立review无阻断。
仓库全量format.sh ci仍因既有generated/log/import问题失败，日志归档；其346个无关formatter改动已全部恢复。
新增文档与证据的增量pre-commit及manual markdownlint均已通过。

本地无Torch的独立`verify_local.py`也已复算两臂原始token/逐步事件：384请求、73728 token，差异均0；
177个下载小文件逐项SHA核验通过，完整78轮stage与outer一致。

CPU资格命令已保存为`run_cpu_qualification.sh`，在227以原结果标签及新的JSON路径调用：
`bash run_cpu_qualification.sh retest128k-wkv192-1-20261017 <新资格JSON路径>`。
解包后在本机运行`python3 static-wkv-128k192-retest/verify_local.py`即可不依赖Torch独立复算原始token/事件和输入/阶段。

## 归档

[完整复核证据包](evidence_tnd_precision/full_pto_20261016/static-wkv-128k192-retest-evidence.tar.gz)，
2,315,659 bytes，203个逐文件SHA条目；包内manifest同时绑定5份正式数学文件，证明默认实现未改。
SHA256：`a1da506c0921cc4f6e315b9270632e832a76ffb9b8d94bafcbf73d72396bc8e3`。

包含独立运行helper、完整资格器、CPU门禁/本地复算脚本、两边token及request_stats原始事件、
全部输入轨迹、Native阶段参考、当前阶段/outer、运行日志、资格JSON与静态检查日志。
大权重、KV bank内容及PT原值快照不入Git；重跑仍依赖原227资产环境，纯JSON结果复算可在本地完成。

远端helper：`/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/retest128k-wkv-20261017/witness-v1`。
输出：`/data/pyptouser/yejia/lc128/retest128k-wkv192-1-20261017`。
完整SDK/PTOAS/ATB/Python/source路径固化在归档env/preflight/run-case中，不以名称猜版本或导入来源。
