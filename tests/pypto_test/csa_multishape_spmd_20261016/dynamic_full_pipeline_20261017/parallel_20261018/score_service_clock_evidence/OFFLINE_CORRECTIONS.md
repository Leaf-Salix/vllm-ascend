# 离线定位与检查更正

- formal668 的 `ptoas/dynamic_csa_full.cpp` 首轮 instrument_cpp 已 PASS。
- 第一次同时核对 wrapper 时 `_function` 仅识别 `AICORE void cube_schedule(`，而实际 wrapper 为 `static __aicore__ void cube_schedule(`，因此抛出 `ValueError: 函数不唯一：cube_schedule`。没有调用 CCEC/PyPTO；辅助定位函数兼容两种文本后，raw/wrapper 均 PASS。
- 在源码冻结前审查发现：ACK累计包括 leaf 尾部排空，位于 step 外，不能要求 `ack_ticks <= step_ticks`。已改为 `ack_ticks <= body_ticks`，并加入合法 ACK 大于 step 的合成反例。数学计数 ACK=step 仍由每叶 `(n-2)+min(n,2)=n` 合同约束。
- 最终 `offline_checks.json` 为标准库离线检查终态；没有新编译或设备结论。

主会话提交包装初次读取独立审阅JSON时使用了未约定的ready_for_single_device_diagnostic键，实际审阅字段为ready，导致KeyError。错误发生在调用task-submit之前、没有登记任务或占卡；已将包装改为读取实际ready字段。hook/runner/冻结源未改，未重新运行离线模型。
