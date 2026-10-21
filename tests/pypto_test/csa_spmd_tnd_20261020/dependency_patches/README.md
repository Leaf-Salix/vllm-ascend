# TND 验证使用的隔离工具链补丁

本目录保存原生 `signal_wait(..., backoff_ticks=150)` 及固定宽度动态 slice
推导所需的 PyPTO/PTOAS 增量。生产算子直接调用原生 API；测试中的旧 BSH
仍保留其原等待适配。两者的实际等待策略由设备构建产物审计核对。

- PyPTO 基线：`65477973435f7d717ffaa7f602f4783ad5310417`。
- PTOAS 基线：`b48c84ebfea88c58439181e76f5e00fd12950418`，以
  [manifest.json](manifest.json) 中的完整提交号为准。
- 增量：[pypto.patch](pypto.patch)、[ptoas.patch](ptoas.patch)。
- 已在各自冻结基线上通过 `git apply --check --whitespace=error-all`。
- 隔离源码、构建工具和二进制指纹见上层 `native_toolchain_manifest.json`；
  构建产物不提交，不能复制其他 worktree 的共享库代替从源码构建。

已完成的相关验证包括原生等待 274 项、动态 slice 推导 810 项、PTOAS
相关 lit 15 项。PyPTO 其余回归 13890 项通过，18 项已有环境/基线失败
单独归属，未修改断言掩盖失败。这些历史结果的日志和路径保留在上层
实现记录与恢复状态中，不把它们写成当前重新运行了全套测试。

`bootstrap.py --workspace-root` 支持从独立 vLLM worktree 显式绑定工作区，
避免把 `.worktrees` 误当成依赖根目录；工具链选择仍由调用者显式传入。
