# 过程118/119同卡顺序复核与诊断

任务task_20261012_203350_21870272865已提交，目前排队。只用原5μs负载。
正常05220c71e前→关闭分段计时候选→同频换算候选→正常后，同一卡顺序执行。
各候选独立runtime，复用正常SDK匹配任务；每组30样本，8次调用/图，普通/标记交替。

完成后单独使用诊断runtime检查开启level4的profiling记录：
诊断版本包含关闭分段计时候选，额外在一次性simpler_aicpu_init里记录实际CNTFRQ与目标频率。
该日志只属于诊断runtime，不加入正式性能组，也不保留到正常SDK。
检查一条17级×48 AIV、work5链的输出/保护区、816条AICore记录和非零scheduler阶段时间。
该诊断打开了profiling，因此只验采集功能，不用于评估early_start重叠或性能。

[运行脚本](repro/run.sh)、[诊断源码](diagnostic.patch)、[诊断脚本](repro/profile_check.py)。
