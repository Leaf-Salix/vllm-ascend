# 直接发布heads：精度与性能证据（2026-10-26）

本目录对应显式`tnd_precision_tuned`，父提交79840c776。冻结基准、默认alias、原combined及短T路径保留。全链HC-pre→HC-post一个入口、纯DSL、真实TND、64Tensor。

## 文件与身份

- `results/`：四个Direct单层性能/Native逐位资格、单独dep、负面候选及三方原始result.json。字段winners与accuracy_executed共同判断，顶层pass不等于性能或精度全部通过。
- `executed_sources.tar.gz`：实际运行源码、逐文件manifest、formal whole启动/预检/审计脚本、固定SDK编译收据；封存前逐文件SHA校验。
- `l1_dependency_raw.tar.gz`：独立L1计时窗口和另外任务的dep原始文件，18窗全部映射，不能将函数包络相加。
- `directpub_generated_raw.tar.gz`：CPU实际QK/PV与编排源码、编译收据，只读证据，没有修改生成C++。
- `panel_events_sources.tar.gz`：未合入独立READY事件候选及生成物。两个panel协议checker和JSON模型记录保留，用于复现否决方案；不混入正式runtime。
- `whole8k_raw.tar.gz`和`whole8k_summary.json`：真实整模型warm192＋测量192，16rank共245760位置，token及事件完全匹配Native，无抢占。
- `whole128k_raw.tar.gz`和`whole128k_summary.json`：128K/B24、24提交21活跃/T126桶144，warm192＋测量192的147456位置token及事件与Native完全一致。模型13:45:01～13:53:07，486秒；16卡均释放。
- `PUBLIC_SUMMARY_DERIVATION.json`：公共摘要仅将两类trace摘要字段改为中性名称；数值完全相同，原字段和摘要值保存在raw压缩包SUMMARY.json中。
- `FILES.json`：最终发布时生成的逐文件SHA索引（不包含自身）。

## 复现顺序

1. 在新的自有实验目录解开sources，逐项核对manifest；固定SDK和canonical SOURCE_ROOT必须满足source_identity/sdk_bind。新路径或源码变化必须重新编译并生成新收据，不能复用不同SHA的receipt。
2. 先CPU `bash compile.sh baseline`、`bash compile.sh candidate`；formal为`bash compile.sh tuned`。检查root64、NPU未初始化和artifact load。编译使用固定checkout ccache与四并行，不重装框架。
3. 单层顺序提交一项一个case：`task-submit --device auto --device-num 1 --max-time 780 "bash <case>/run.sh <newlabel> <batch> <history> <comma-separated-1to6-lengths>"`。8K/B32为32个6；128K/B16为16个6；mixed126是1～6重复6次、共36请求。第二次8K增加`--reverse-capture`。
4. Direct控制为lookahead，候选为所属AIV直接写Native heads。门槛event中位数至少降低0.5%、至少10/12配对领先、L1降低，随后Native A/B八边界、ABA、guard/只读检查。没有性能收益则跳过精度，不用顶层pass混淆。三方harness另行记录双方精度，nalinaly不是gold。
5. 计时不要开启dep-gen。需要模块归因时单独dep任务，再用analyze脚本核对完整task/core映射、CPU/JIT同一core源码和编排等价。这里的span含等待、搬运及计算，不是纯算力时间。
6. 两档whole仍各一项16卡任务，8K通过才128K。使用`whole/run.sh 8k`后`whole/run.sh 128k`，warm192＋测量192，固定Native bank和黄金，不跑Native2。actual FULL replay与Native分发/JIT计数差分全部检查；每任务外780秒、模型阶段内750秒，排队与实际占卡运行分开记录。
7. 用`summarize_whole.py <extracted-whole-dir>`重算16rank审计、每步最大rank耗时、20forward/19cycle统计；不能拿历史whole均值直接相除当成本轮受控加速。

完整参数、工具链差异、性能与精度解释见[报告](../../DSV4_FLASH_CSA_TND_DIRECT_HEADS_20261026.md)。三方中nalinaly固定于cafcadec96，只有backoff_ticks兼容删除，不能冒充最新c346或其新SDK公开数据。Nal单层输出relative L2非零不等于已证明整模型token错误。

封存的`SOURCE.json`保留创建时继承的历史字段，`hardware_precision_validated=false`是设备验收前状态，不事后修改manifest。实际本次控制为lookahead、候选为Direct；以受测root/sparse文件SHA、result中的arm、编译收据和本报告的验收结论确定身份，不把继承的旧baseline提交字段当作本次控制实现。
