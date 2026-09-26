# DSV4 Flash CSA 接续入口

当前任务、验收合同与待办见
[任务清单](tests/pypto_test/DSV4_FLASH_CSA_TASK_CHECKLIST.md)。
环境和可运行入口见 [验证 README](tests/pypto_test/README.md)。

基线固定为 vLLM 0.25.1 / vLLM-Ascend 0.25.1rc1、A3 / CANN 9.0，
使用 `source ../env-dsv4-0251rc1.sh`。
正式权重固定为 `/data/model/DeepSeek-V4-Flash-0731-w8a8`。
PyPTO/Simpler 使用指定调试分支，PTOAS/ISA 的实际固定版本见清单 A4。

先单卡构造用例、定位和验证，再进行真实权重 16 卡验收。
当前精度版 mode=0 的单卡整层诊断已跑通，仍有待分析的跨实现数值差异；
阶段状态以清单为准，不将历史 PASS 外推为当前验收完成。

旧 main fixture、executor 测试、旧图重放/连续轨迹/profile 链路及专用辅助脚本已删除。
有效 helper 已提取到当前用例；过时测试记录、重复快照和旧失败产物一并删除；仅保留当前输入和有效证据。
