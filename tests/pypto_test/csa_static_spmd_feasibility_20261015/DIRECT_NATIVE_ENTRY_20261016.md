# CSA 大参数入口：用户选择直调与 TorchAir 参考核对

2026-10-16。用户明确要求直接使用 pybind 路径，并提醒参考 TorchAir 的多种接入方式。
此前“56 Native + 6 workspace”的 62 项入口方案取消，不为绕过 Torch schema 改写 CSA 参数。

## 两层接口不能混为一谈

当前 Torch build 的 dispatcher 在 DispatchKeyExtractor 中使用 64 位 bitset 标记 schema
参数位置，因此 132 项具名 schema 无法定义。这是 torch.library 注册限制，
不代表 Python→C++ binding、Simpler CORE 或 A3 限制为 64 项。

本地 PyPTO 已经有直接调用路径，无需先增加新的 C++ 导出：

1. `JITFunction.__call__`（`python/pypto/jit/decorator.py:3374`）解析实参并获取 kernel artifact。
2. `python/pypto/torch/launch.py:88` 的 `invoke` 进入现有 Worker / capture 生命周期。
3. 同文件 `native.prepare` 接收 Tensor 列表、dtype、scalar、当前 stream 和 device。
4. `python/bindings/torch/torch_npu_adapter.cpp` 使用 nanobind 的 `NB_MODULE` /
   `m.def("prepare", ...)` 导出；Tensor 列表按 CHIP 容量检查，不定义 132 项 Torch schema。
5. 同 adapter 的 `OpCommand::RunOpApiV2("PyPTOKernel", ...)` 把启动放入 torch_npu task queue，
   lambda 内使用 `stream(false)`。已有 Tensor/storage 持有及 capture 处理应直接复用。

这里用户所说的“pybind 直调”对应已有的 native binding；该实现实际使用 nanobind。
`pypto.torch.register` 是另外一层可选封装，不应把其 schema 限制误写成整个 PyPTO
kernel 调用的限制。此处仅完成源码核对，尚未把完整 132 参数 CSA 的设备执行记为通过。

## TorchAir 示例实际展示了什么

用户提供的 GitCode 页面本次读取失败，因此核对的是 Ascend 官方 GitHub 仓库对应目录，
记录的 master commit 为 `e1118e8b0fc1b95190ab53b732c0872989550e20`。
未声称两个站点在本次读取时的分支头完全一致。

- [pybind 示例](https://github.com/Ascend/torchair/tree/e1118e8b0fc1b95190ab53b732c0872989550e20/examples/_kernel_extension_aclgraph/pybind)：
  C++ 函数通过 PYBIND11_MODULE 导出；其示例测试还在 Python 层包了一层 torch.library / Meta。
  因此不能仅因目录名为 pybind 就说示例的 torch.compile 路径完全不经过 dispatcher。
- [torch_library 示例](https://github.com/Ascend/torchair/tree/e1118e8b0fc1b95190ab53b732c0872989550e20/examples/_kernel_extension_aclgraph/torch_library)：
  在 C++ 中定义 schema、PrivateUse1 实现及 Meta 实现。
- 两组示例均演示 NPUGraph、make_graphed_callables、npugraph_ex 三种调用环境。
- [启动方式说明](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/examples/_kernel_extension_aclgraph/pybind/csrc/add_custom.asc)
  列出八种内核启动组合，包括正确方式与错误对照。注册方式和 task queue/stream 启动方式是两层问题。
  当前 PyPTO adapter 已采用其中对应的 stream(false) + OpCommand 入队模式；
  不需要改成裸 stream(false) 启动。

## 接续边界

保留 132/133 个 CORE 描述符，优先调用已有 JIT native 路径验证完整 kernel。
先做单次功能、状态与保护区，再检查原始 NPUGraph 捕获/重放。
整网 torch.compile / npugraph_ex 的外层 opaque op、Meta 及 mutation 语义单独核对；
直调接口存在不等于这些组合已经全部验证通过。

用户随后要求优先提交必要 bug issue，并让扩容等改动继续留在本地；
本轮没有为了入口直调修改依赖源码，没有新设备性能数据。
