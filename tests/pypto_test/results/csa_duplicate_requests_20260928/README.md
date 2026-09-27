# 相同请求副本：规约不确定性单卡筛查

基底2a740c1f原Native cache布局，正式layer4权重、B16/S6/H8192，mode2，Native确定性level0。
所有请求的6个输入及逻辑cache/state完全相同，物理页独立；保持原页表、slot和保护区。
[执行器](duplicate_case.py)、[命令](run.sh)；task_20260928_005959_25474208374退出0。

atomic1/0两轮均通过metadata/保护区/有限值/Top-K结构检查。
各自两次eager和最后一次graph输出中，相同请求副本的x_out及Top-K集合均零差异。
两次eager自比也零差异。**该合成case未复现模型专家分化，不证明真实模型atomic没有影响。**

| atomic | Native CSA均值 μs | PTO CSA均值 μs | PTO P95 μs | PTO对Native输出RMSE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 929.234 | 809.528 | 830.700 | 0.002957749 |
| 0 | 903.710 | 769.910 | 782.100 | 0.002957749 |

每轮3次预热、10次无profiler图计时。两轮Native自身有2.75%波动，不能把PTO的4.89%差额全部归因于关闭atomic。
输出误差计数相同不等于两轮PTO输出已作逐元素对照，本试验没有保存大份state作额外全量比较。
完整报告：[atomic1](atomic1/report.json)、[atomic0](atomic0/report.json)。

单卡结果支持关闭atomic路径可执行、没有明显本体性能惩罚；真实EP16的路由和forward影响另行验证。
