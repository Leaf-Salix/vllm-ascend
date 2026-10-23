# 公共PyPTO逻辑表达式修复，2026-10-22

当前状态以[manifest.json](manifest.json)和公共工具链清单为准。

原公共PyPTO `e891351e`的前端支持逻辑表达式，但AICPU编排生成器缺少And/Or/Xor/Not。
原C05 128K/B24/S6启用scale_page_coalescing后在GenerateExprString报And；仅改嵌套if
不能消除优化Pass产生的逻辑节点。已有[issue #2984](https://github.com/hw-native-sys/pypto/issues/2984)，
官方[PR #3000](https://github.com/hw-native-sys/pypto/pull/3000)已于10月15日合入main，未重复提issue。

新PyPTO `26e94e05`建立在原公共版本上，只回移官方17行生产代码及123行回归测试。
And/Or用归一化bool的位运算，保持两侧求值；Xor返回bool不等，Not返回逻辑非。
PTOAS和Simpler保持原公共版本。所有新构建在新工作树完成，不复制旧.so，原发布目录未修改。

已完成：

- pypto_core及torch_npu适配器构建通过；CANN为公共9.2，runtime头文件来自原Simpler固定版本。
- 三个编排测试文件共113项全部通过，四条现有弃用告警。
- 原失败C05使用相同v6算子、相同输入和True策略，完整CPU编译通过。
- 公共bootstrap加载来源检查通过，父子进程使用同一新PyPTO及原PTOAS/Simpler。

初次pytest被历史editable finder导向旧安装，七个新增用例失败；这不是新构建验证。
保留wrong_import日志，固定来源并断言实际DSO路径后重跑，113项通过，没有更改断言。
首次构建漏初始化runtime子模块，补齐同一gitlink后构建通过，没有升级Simpler。

128K/B24/S6及旧混长B16的三侧eager、图重放、post-L1、cache/state和保护区全部通过，
数值逐bit一致。新SDK已标validated；具体任务和检查项见manifest。
这不代表算子性能达标：C05的AICPU串行页表校验带来约3.57ms的AICore空档，需另行优化。
新的性能比较必须两侧同用本版本，原版样本不改标成新工具链结果。
