# 完整 CSA 单 SPMD：Torch 注册 64 入参限制

2026-10-16。本文记录完整 B4 已完成默认 PYPTO 代码生成后，在真实注册入口
发现的新限制。它与 Simpler CORE 的 Tensor 描述符容量是不同层的问题。

## 实际结果与归属

环境为本会话既有 `torch 2.10.0+cpu`、`torch_npu 2.10.0.post2`；没有替换
PyTorch 或共享组件安装。完整 `entry_jit.py` 的 132 个具名 Tensor 在
`pypto.torch.register` → `RegistrationSignature.define` → `torch.library.define`
时报错：

```text
The function schema has 132 arguments but this PyTorch build only supports 64
```

错误发生在 dispatcher schema 定义阶段，还没有构造 Meta 实参、编译 CCE
二进制或启动 NPU。因此这不是 A3 计算错误，也不能归因于 Simpler 扩容失败。
不推断其他 Torch 版本的上限；本轮没有为此重建或修改 PyTorch。

| 检查 | 实际结果 | 不能据此推出什么 |
| --- | --- | --- |
| 64 个具名 Tensor 的 schema | 定义成功 | 未测试完整 CSA |
| 65、132 个具名 Tensor 的 schema | 定义失败，明确上限 64 | 不是 CORE 参数装载失败 |
| 56 个只读 Tensor 列表 + 76 个可变 Tensor 列表 | CPU、Meta 成功；共 132 Tensor、2 个 schema 参数，原对象身份保持 | 只是 Torch 小例子，没有接入 PyPTO |
| 上述 Tensor[] 小例子的 functionalization | 失败，报告可变 alias schema 不受支持 | 不能把 Tensor[] 当成现成的整网解决方案 |
| 扩容后的 CORE 133/256 Tensor + 16 scalar | 之前四类 runtime 设备验证已通过 | 不改变 Torch dispatcher schema 的限制 |

## 复现与原始证据

- [独立 CPU 复现](repro_torch_schema_capacity.py)：只依赖 Torch。
- [完整 CSA Meta 注册驱动](test_static_jit_meta.py)：复用真实生成入口和准备对象。
- [独立复现结果](evidence/torch_schema_capacity_cpu_v1/result.json)。
- [完整注册结果和异常栈](evidence/static_b4_jit_meta_v1/result.json)。
- `session_logs/csa-torch-schema-capacity-cpu-v1.log`。
- `session_logs/csa-static-b4-jit-meta-v1.log`。

独立 CPU 复现命令（使用本会话 Python 环境，不占 NPU）：

```bash
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python repro_torch_schema_capacity.py --output /tmp/csa_torch_schema_repro
```

完整入口复现先按 [独立组件环境](PYPTO_STATIC_FIXES_20261016.md#独立环境复现)
设置两个 worktree，在本目录执行：

```bash
python bootstrap_isolated.py test_static_jit_meta.py \
  --case static/h131072_b4 --output /tmp/csa_static_jit_meta_repro
```

以上是供后续接手复现的命令；本轮归档没有再次执行这些测试。

## 历史候选入口：已取消，不是已完成实现

2026-10-16 后续用户明确选择直接 native binding 调用。下述 62 项方案仅保留
调查历史，不再等待确认、不再推进。已核对现有 PyPTO JIT 直调入口及 TorchAir
示例，见 [直调入口说明](DIRECT_NATIVE_ENTRY_20261016.md)。

建议保留原 56 个 Native 具名 Tensor，把临时存储按六种 dtype 分成六个
workspace Tensor，根入口合计 62 项。内部继续使用展开的独立描述符，
不要求 CORE 从 132 项降到 62 项。预期由固定偏移 view 建立内部绑定，
不引入逐步 metadata 搬运或新的设备适配算子。

当前 B4 临时视图的 dtype 统计为 FP32 42、BF16 10、INT32 8、INT8 4、
FP16 1、INT64 2；另外三个协议区也是 INT64。Native 的 56 个根参数在
核体内对应 60 个具体描述符，因此当前总数是 60 + 69 + 3 = 132。
根入口参数数目与内部视图描述符数目不能混用。

这个候选与之前失败的单 INT8 arena 不同：每个 workspace 已具有自己的
dtype，避免依赖 orchestration 的跨 dtype reinterpret。仍需实际验证
kernel 模式的 view 绑定、mutation/返回别名、注册和运行时行为，不能因
62 小于 64 就预记成功。并发调用仍需各自 workspace。

此前按原目标“有过不去的关键点停下讨论”询问用户。用户现已选择直调；
没有实现这套 62 项新入口，也没有修改共享 PyTorch。
本问题尚未提交上游 issue/PR，后续真实链接统一登记到
[ISSUES_AND_PRS.md](ISSUES_AND_PRS.md)。
