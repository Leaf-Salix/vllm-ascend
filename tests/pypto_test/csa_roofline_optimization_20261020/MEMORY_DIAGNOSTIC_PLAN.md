# 下一步：先取得真实内存证据，再决定缓存策略

用户已要求放弃 Attention 方向，本轮不再修改或调测 Attention。
精确 Top-K 首版两重点档严格精度通过，但净省 -0.46/9.10 μs，低于门槛。
暂不扩档；独立 GE 输出计数与有序索引 probe 用于排除完整排序回退掩盖
筛选未生效，不能把它当成新的性能验收。

支撑方向先只采一组 128K/B28（当前正式单 SPMD 与冻结多 task）：
沿用原六步轮换 PMU 采集流程，每侧三次，改用 MemoryAccess 并请求 L2
计数。复用现有权重/fixture/严格精度、状态、图和代际检查。仅补缺少的
内存证据，不重复 PipeUtilization，不以外层 Duration 替代 L1。
若硬件/工具不返回所需字段，记录不支持，不把空列写成零流量。

当前已核对的源代码事实：

- `pl.func_attr({"cache_policy": [(...,1),...]})` 的 1 对应 BYPASS。
- 当前 PTOAS 的 `TLoad.cpp` 仅在 BYPASS 且显式携带 offset 时调整源描述符；
  没有 offset 时继续生成普通 TLOAD。
- 先前 `weight_cache_policy_codegen_audit_v1.json` 已记录属性改变而 C++
  全文相同。本轮完整 C++ 也没有据此新增 cache hint，不能把 IR 属性视为
  权重已经绕过或保留在 L2。
- PTO-ISA 另有 `TLoadL2Hint` 模板入口。尚未确认它与地址偏移 BYPASS
  的语义对应，不直接等同，不擅自猜测地址偏移。

候选实施前仍需说明具体读路径、实际指令和真实流量变化；不能对全部
可写 Native KV/state 根统一改 BYPASS。只有具备数据证据及正确的发布获取
合同后才做区间策略候选，并沿用两重点档净省20μs门槛。

上游已有相关问题 https://github.com/hw-native-sys/PTOAS/issues/1537（仍open）：旧NotAlloc模板所依赖的runtime offset未正确填充曾产生错数/设备异常。当前先保留普通读，不重新启用该旧路径、不硬编码别名偏移；没有重复提issue。
