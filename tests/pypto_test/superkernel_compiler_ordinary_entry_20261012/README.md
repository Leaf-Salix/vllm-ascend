# 普通入口证明接入正常编译链

2026-10-12。私有PyPTO提交 `180d9d307`，公共安装不变。
此前290.84μs等结果使用了测试钩子声明普通入口；本提交去掉这一使用限制，
由正常编译流程从函数体推导是否完全不使用交接状态，生成现有能力字段的逐成员高位。

只适用于A2/A3 TMR的编译器自有AIC/AIV函数。外部源码、模板、旧异步返回、Wait/Set、
延迟完成与未解析辅助调用不赋予证明；HBG/A5不生成标记。后端重新推导，不信任用户同名属性。
MIX成员分别表达，不能用AIC的证明覆盖AIV。不自动寻找或插入边界，不改变early-resolve。
旧私有运行时忽略高位并执行原路径；配套候选才跳过普通入口的交接冷状态。

15项分类、保守回退及平台隔离检查通过，3项原有MIX/early-resolve编排检查通过。
新增平台检查后，测试夹具遗漏backend配置，13项报“Backend type not configured”；
补齐910B测试配置后15项通过，没有放松证明条件。失败及修正日志均保留。
编译器在本工作树重建并安装到私有环境，普通和标记链均实际编译完成：
[普通编排](plain/orchestration.cpp)生成mask=56，[标记编排](early/orchestration.cpp)保留原能力。

[完整补丁](compiler.patch)、[分类检查](proof_tests_complete.log)、
[编排检查](compiler_orch_tests.log)、[lint](compiler_lint_complete.log)。
这里没有新增性能成绩；通用入口的设备测量随下一隔离候选排队，不能把测试钩子的成绩直接当成其验收。
