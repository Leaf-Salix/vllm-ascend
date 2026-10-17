# 完整 B4 单 SPMD：二进制编译与组装结果

2026-10-16。等待用户决定 Torch 入口方案期间，继续验证与该选择无关的
现有 132 参数核；保留现行接口，没有实现 56+6 workspace 候选。

## 结果

`static_b4_binary_v1` 通过。复用 `static_b4_jit_cpu_v21_carry_scoped` 的
默认 PYPTO 代码生成产物，在新的结果目录编译，不修改原始证据：

- AIC：CCE 编译、设备链接、提取可装载代码成功，58548 B。
- AIV：CCE 编译、设备链接、提取可装载代码成功，146956 B。
- AICPU 调度：交叉编译共享库成功，861416 B。
- 两个 132 参数 CORE Callable 与 132 参数 ChipCallable 组装成功。
- 总编译和组装耗时 115.418 秒；这是 Host 编译时间，不是算子延迟。

使用独立 PyPTO/Simpler worktree 及既有 CANN 9.2.0-beta.2 工具链。
原始命令包含 CCE 的 Cube/Vector 架构、优化和头文件参数，保存在
`session_logs/csa-static-b4-binary-v1.log`。没有修改 PTOAS/PTO-ISA 或共享安装。

## 验证范围

这是从 JIT 源码产生的 **program 模式产物** 的完整二进制编译和内存中
callable 组装。不创建 Worker，不加载设备，不调用 Torch 注册接口。
因此它补齐“代码生成之后能否由 CCE 编译”的证据，但不代表完整 CSA 的
单次设备执行、跨核进度协议、状态、数值或性能已经通过。

当前 Torch 64 入参限制仍在，见
[注册限制与候选方案](TORCH_SCHEMA_CAPACITY_20261016.md)。
目标保持 active；按原目标的关键点讨论要求，接口方案仍待用户答复。

## 复现及产物

- [编译驱动](compile_existing_binary.py)。
- [结构化结果](evidence/static_b4_binary_v1/result.json)。
- 可复现命令：先按 [独立组件环境](PYPTO_STATIC_FIXES_20261016.md#独立环境复现)
  设置环境，在本目录执行：

```bash
python bootstrap_isolated.py compile_existing_binary.py \
  --artifact ../results/csa_static_spmd_feasibility_20261015/static_b4_jit_cpu_v21_carry_scoped/kernel \
  --output /tmp/csa_static_b4_binary_repro
```

`--artifact` 应指向原开发 worktree 的现存完整产物；主仓归档不复制大产物，
其绝对路径见结果 JSON 的 `source_artifact`。输出目录必须不存在，避免覆盖
旧尝试。二进制仅留原结果目录，路径和大小进入 `artifact_inventory.json`，
不提交二进制、不额外计算内容 hash。归档不会重跑这个编译。
