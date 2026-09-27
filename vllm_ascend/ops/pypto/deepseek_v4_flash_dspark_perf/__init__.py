# SPDX-License-Identifier: Apache-2.0
"""CSA 性能版，与 deepseek_v4_flash_dspark 精度版保留独立算术入口。

共享适配、metadata和数值中性实现；允许采用pypto-lib或Native的高效精度策略。
性能优化先在单卡验证，再以真实权重整模型的decode forward和CSA完整区间验收。
允许明确规则下的量化/Top-K差异，但仍须逐元素误差受控、逐token及DSpark统计一致。
metadata、保护区、缓存写入和非有限值属于功能约束，不因性能优先而放宽。

当前约束及有效证据见tests/pypto_test/DSV4_FLASH_CSA_TASK_CHECKLIST.md；
历史版本的计时和数值结果不能自动作为当前实现的验收依据。
"""
