# HCA 原生 TND 多任务实现

目标：在正式 W8A8 权重和相同 Native TND 输入上，实现完整 HCA attention 半层，
性能提升超过 30%。提升比例沿用用户约定 `(Native / PTO - 1) × 100%`。
八档单步输出已通过原容差，S1及混长/padding的真实custom op接管已验证。
Native cache/state的严格比较仍有差异；连续状态轨迹、整模型token和八档30%目标尚未验收。

## 输入与执行边界

- 输入是 BF16 `[T,4,4096]`，实际请求由设备 `query_start_loc/seq_lens` 描述。
  支持真实请求 S1～S6 混合、失活 dummy 与 token 容量尾部；T 不等于固定 B×6。
- Native positions、slots、页表、C128 state 和 compact RoPE 原地消费；norm 权重保留 BF16。
- 一个 JIT 入口包含 HC_pre、input RMS、QKV/RoPE、C128、attention、O projection、HC_post。
  多个 incore task 由 AICPU 的依赖图调度；不在算法内实现 GM 完成信号或跨核轮询。
- `entry.py` 是 JIT 入口；`native_adapter.py` 绑定 Native 描述符；`request_geometry.py`
  从设备 Q/L 建立本次 token 归属；`compressor.py` 与 `attention.py` 实现 TND 特有逻辑。
  与请求边界无关的数学函数继续复用本仓库既有 HCA/CSA 实现。
- `output.py`恢复Native的O_A BF16边界及完整8192列token量化。T<=96使用完整K点积，
  T>96使用N256、K分两段；两份INT32部分和先相加才转FP32，随后融合HC_post。
- `service.py:bind_layer`显式接管已有`dsv4_hca_forward`边界，保留Native回退、KV等待/通知/保存，
  按每步forward context复用compact metadata；未自动修改全模型的默认选择。

## 测试与性能口径

使用公共 `csa_shared_toolchain/manifest.json` 的 PyPTO、Simpler、PTOAS。
先单卡功能/精度，再性能；所有 NPU 执行通过 task-submit。
正式权重第 3 层、NZ2。用户于 2026-10-22 明确要求后续不用 ACLGraph replay，
因此两侧均直接调用；Native 的静态图/SuperKernel 历史基准只作历史参考，不混算收益。
按用户最新要求，性能候选只以完整 HCA 的 L1 incore 最早开始到最晚结束判定，
不把Host/JIT时间算入核内收益。外部服务及Host诊断只作历史证据和SDK问题复现。

性能矩阵沿用三个 CSA 会话最新固定八档的输入定义：128K B1/B8/B16/B24/B28，
256K B16，8K B32，以及 128K B16 的 S1～S6 混长 T56。
另做 S1、S5、C128 跨界及同址更新 metadata 后的直接调用检查。
历史 HCA 七档可作参考，不能与新工具链、新输入混算性能提升。

过程见 [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)，独立测试和原始结果放在
`tests/pypto_test/hca_tnd_mpmd_20261022/`。
