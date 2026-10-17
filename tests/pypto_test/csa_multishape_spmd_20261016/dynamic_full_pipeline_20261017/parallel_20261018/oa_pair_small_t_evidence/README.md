# O_A小T连续资源域候选

尚待设备结果，未迁移正式源。上一版T1..96的四档正负结果和完整源保留在
[oa_pair_manual_evidence](../oa_pair_manual_evidence/README.md)。本版只在既有
OB32/单32行量化任务域启用pair；两调度端谓词以外的执行AST保持旧候选。

- [源级审阅与复现](SOURCE_PREPARATION.md)、[审阅结论](SOURCE_REVIEW.json)。
- [冻结候选](source_candidate)、[同轮668参考](source_reference)。
- [实际提交命令与任务](registry.json)：128K/B4/T24、128K/B5/T30，均S6。
- [中文总览整理与逐docstring说明](documentation_review/README.md)。

不重复已有数学体CPU；两个任务会分别JIT、完整精确比较、检查图/状态保护，
再测三方同进程五窗完整L1。没有结果前不称新策略获益，更不称30%目标完成。

## 实测结论

两档完整验证PASS，但无足够稳定收益，**不采用**。见[全部五窗结果](DEVICE_RESULTS.md)及[采用决策](ADOPTION.json)。不再追加本候选设备测试。
