# Native后端生成图的设备重放计时

superkernel A/B两代表档已有明确收益，八类状态开关两侧零容差相同；本轮固定开启，
不再比较或调参。目的是校准后续Native/PTO的CSA本体计时边界。

发现：8K/B24直接编译调用的图外事件均值1209.602μs，旧外图重放为1018.715μs；
各自独立profile的首末kernel跨度却为1036.250/1030.500μs。两种采样不能相减作精确归因，
但说明不能将Python编译包装调用的事件区间直接当作纯CSA设备跨度。
包装的参数处理/调度在短档暴露、长档被较长cache恢复掩盖只是推测，未证明为完整因果。

仍显式torch.compile(..., backend="npugraph_ex")，force_eager=False，
图由npugraph_ex捕获并应用static kernel/superkernel；保持Native多流/event依赖。
只在固定地址/shape的单卡fixture中，从实际backend实例取得唯一完整图，直接调用其replay计时。
保留图owner、module、fixture与Tensor存储；若存在多图或需要主机更新的节点则报错，不绕过必要处理。
八类状态与同初态、同一张图通过compiled callable执行的结果零容差比较，失败不接纳计时。
这不是修改输入地址的整模型执行接口，不能推广为生产跳过backend参数处理。

仅128K/B16和8K/B24，5预热20次设备事件及独立profile，不重新测superkernel关闭组。
私有整包冻结，语法检查后等待当前PTO AIV实验结束再正常入队，避免两项本地编译/计时相互干扰。

## 校准完成

2026-09-29 08:16正常入队task_20260929_081604_23093924514，auto设备1，退出0。
长B16均值1122.652μs/P95 1128.400μs；短B24均值940.762μs/P95 945.280μs。
每档八类同图状态零容差、metadata/slot保护区均通过，实际两条计算stream和SuperKernel均确认。
没有新增关闭组，未改变生产模型执行。后续单卡Native对照采用本口径；新版七档留到阶段出口统一完成。
旧PTO同源码短B24约960–980μs，已不能以旧Native 1018.715μs继续声称优化后Native仍慢于PTO；
这只是新基线揭示的风险，不把异轮数据计算成正式新加速比。
[结果](RESULTS.md)、[精简证据](summary.json)、[CPU收集器](collect.py)。
