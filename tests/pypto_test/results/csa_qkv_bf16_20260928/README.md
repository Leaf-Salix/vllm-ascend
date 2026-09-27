# QKV的Native BF16数值边界：独立筛查

基于生产2a740c1f，独立工作树`.cache/csa-qkv-bf16-2a740c1f`，仅修改性能版QKV。
没有叠加整token WO-B量化或source-split cache，Native和精度版不变。

Native路径在`attention/dsa_v1.py`的decode prolog：QA投影后进入fused RMS/quant，
QB反量化后进入Q RMS，再进入RoPE；KV投影后进入RMS，再进入RoPE。
这些独立算子的输入/输出使用BF16。性能版沿上游融合策略，中间直接使用FP32值，省略了部分舍入。
这种差别不是功能错误，也不意味着更多FP32一定更差；当前实验只检查是否会扩大实际模型的专家工作量。

候选在片内补齐以下BF16→FP32 round-trip：

- QA输出：平方和/amax扫描和量化扫描都转换，保证使用同一数值。
- QB反量化输出，以及Q RMS之后进入RoPE的分量。
- KV投影输出的所有扫描，以及KV RMS之后进入RoPE的分量。

完整行和尾块均覆盖，不增加GM张量、任务或跨任务依赖。仍保留性能版NZ分块、split-K/atomic、
Q动态head分块、平方和/rsqrt/amax及量化规则，不能宣称已完全复刻Native算术。
[最小补丁](candidate.patch)。

CPU完整QKV lowering/PTOAS/CCE/链接通过，[入口](compile.py)、[日志](compile.log)。
这不是设备正确性或性能证据。编译命令：

```bash
source ../env-dsv4-0251rc1.sh
python tests/pypto_test/results/csa_qkv_bf16_20260928/compile.py \
  --source ../.cache/csa-qkv-bf16-2a740c1f \
  --output tests/pypto_test/results/csa_qkv_bf16_20260928/compiled
```

单卡task_20260928_023535_1798714557：[命令](run_layer.sh)。
同轮原版/候选B16/H8192、正式layer4权重、mode2/atomic1/det0，5次预热20次图计时；
另测B3/atomic0/det1固定形状A→B→A和尾块。主要观察完整CSA成本和完整HC输出误差，
不能凭某个中间量更接近Native就认定整网受益。当前未合入，未提交16卡扩测。
