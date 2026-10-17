# 连续DQ：真实S5长度1流水边界

任务 `task_20261018_050845_41225224225`，H8192/B8/S5、T40。
owner28跨行区间产生一个head的chunk，真实设备运行、两次图重放、地址固定、
保护区与72核代际通过；功能 `functional_passed=True`。

原分task真实S6的前五行作为因果参照，第六行分别使用X和−X：
输出、TopK、分数和所有可比cache/state字节均逐bit相同；忽略区域仅为额外第六行
合法写入，覆盖全部实际S5写入。`causal_extension.passed=True`。

Native逐bit比较未通过：输出max_abs=0.0234375、RMSE=0.0031557713，
TopK/cache/state也有差异。原始顶层 `passed=False` 保留，不把功能通过
冒充Native/token验收；没有测性能。它验证候选新增的单head路径，不改变
B24回退、B28小幅改善而不采用连续分工的决定。详细证据见同名JSON和原始目录。
