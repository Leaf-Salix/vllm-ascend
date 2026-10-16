# CSA 单 SPMD：核参数容量约束与 CPU 复现

这是独立于 PyPTO [#3009](https://github.com/hw-native-sys/pypto/issues/3009)
的接口限制，不是 A3 不支持这些数学运算，也不是 while 修复后自动消失的问题。
本次没有修改或重新安装 Simpler/PyPTO/PTOAS/PTO-ISA。

## 已确认的容量

本地 Simpler `src/common/task_interface/arg_direction.h` 定义：

```cpp
CORE_MAX_TENSOR_ARGS = 32;
CHIP_MAX_TENSOR_ARGS = 256;
CORE_MAX_SCALAR_ARGS = 16;
```

`TaskArgsTpl::add_tensor` 在达到容量时抛出 `TaskArgs: tensor capacity exceeded`。
`CoreCallable` 签名也使用 CORE 容量；Host/CHIP 或 Graph 的较大容量不能替代它。
手写 Group 的 AIC/AIV 成员共享同一份 Tensor-first payload，不能简单地为两半
各声明一套不一致的 32 参数布局。依据是 PyPTO
`docs/en/dev/passes/24-expand_mixed_kernel.md` 的 shared argument ABI 说明。

CPU 复现文件：[repro_core_arg_capacity.cpp](repro_core_arg_capacity.cpp)。
在本 worktree 根目录执行：

```bash
c++ -std=c++17 \
  -I /data/pyptouser/qinchuanyu/pto-eager/simpler/src/common/task_interface \
  tests/pypto_test/csa_static_spmd_feasibility_20261015/repro_core_arg_capacity.cpp \
  -o /tmp/csa_core_capacity
/tmp/csa_core_capacity
```

本次结果保存在 `static/abi_capacity.log`：

```text
core_limit=32, next_tensor_rejected=TaskArgs: tensor capacity exceeded
chip_limit=256, root_entry_tensor_count=57
```

这里调用的是实际头文件中的容器，没有 NPU 初始化、设备任务或手写容量模拟。

## CSA 需要区分的几个数字

| 对象 | Tensor 数 | 含义 |
| --- | ---: | --- |
| 原 CSA 根入口 | 56 | Native 权重、metadata、cache/state 和输出 |
| 早期 arena 原型根入口 | 57 | 原 56 项加一个 GM 工作区，orchestration dtype reinterpret 未被支持 |
| 最终测试原型根入口 | 132 / 133 | 具体 GM 视图在初始化时建立，与核共享 ABI 一致 |
| 原型 MIX 核共享参数 | 132 / 133 | 长/短 history 模板；包含原计算调用用到的 GM 视图及三份协议视图 |

132/133 来自逐项实参去重，见各档 `static/*/entry.bindings.json` 的
`_entry_contract`。同一 allocation 的不同 shape/offset 视图仍是不同描述符。
工作区里的 67 个临时 allocation 已合并到一个物理缓冲，但在 Host 初始化时
恢复 typed view 并传入核，仍会消耗核参数槽。**早期根入口 57 不是核参数数。**

## 为什么没有强行把权重打包

共有 26 份只读权重。假设它们都能零拷贝恢复原视图，根入口可压成 32 项；
这只是算术上的候选方案，尚不是当前 API 可执行的接口。

其中 `wq_a/wq_b/wo_a/wo_b` 使用 NZ。当前 `tensor.view` 不能产生 NZ，
`tensor.reinterpret_view` 不接受 NZ，依据为 PyPTO
`src/ir/op/tensor_ops/transform.cpp`。只打包 ND 权重仍有 36 个根描述符。
本次既没有把 NZ 标成 ND，也没有修改权重 dtype、量化方式或搬运 Native state。

另一个不能混淆的点：在 `InCore` 里写 `reinterpret_view` 会参与 Tensor-to-Tile
转换，不能未经验证就当作核内任意 GM 指针重建接口。orchestration 又不支持
改变 dtype 的视图。因此最终测试原型在 Host 初始化绑定工作区别名，原 Native
Tensor 保持原存储，如实保留 132/133 的核 ABI。七档 meta 绑定检查通过，未启动 NPU。

## 可供组件维护者评估的方向

1. 为一个完整 MIX 核支持更大的或可变长的 Tensor 描述符集合。本原型需要至少
   133 项；不能只改一个常量，需共同检查 CoreCallable、payload、签名校验、
   wrapper 参数解包、HBG/动态 runtime 与图重放的 ABI 一致性。
2. 支持一个描述符集合参数，以及明确的核内 GM 子视图构造。编译器保留 dtype、
   NZ、stride、offset 和别名/读写信息，runtime 继续识别外部依赖；避免新增每步
   数据打包调用。此方向需 PyPTO 与 Simpler 共同定义，不能用未声明的地址转换替代。

上述为待评估接口方向，不是已实现的修复，也没有要求组件采用某个具体 ABI。
扩展前本地脚本应拒绝 NPU 提交；完整核尚未做功能、数值或性能验证。
