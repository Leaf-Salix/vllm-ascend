# Sparse Attention固定Native输入的PMU诊断

2026-09-27，算子为 `21d99f8a` 保留实现；任务
`task_20260927_154452_175670032077` 退出0。
先从正式layer4、8K/B40、S6/TP1/mode2、确定性0、无EPLB的Native单卡case保存实际
Q、cache、Top-K、positions、sink及逆RoPE前输出，再运行同一份性能版Sparse Attention。
输入是正式权重下的合成历史case，不是全零PMU样例。

## 口径

- 本次是独立Sparse Attention program模式的一次PMU采集，包含plan、qk_pv和merge_norm。
  不包含完整CSA的前后算子，也不代表warmup后的kernel-mode图重放性能。
- 逆RoPE用cos=1/sin=0隔离。PMU事件组2（PIPE_UTILIZATION），不加入无profiler性能矩阵。
- 按同名task跨block累计busy cycles / 累计total cycles；各流水可以交叠，百分比不能相加。
  保留原始cycle，不假定时钟频率换算为μs。
- qk_pv记录完整：24个AIC block、48个AIV block，所有PMU total cycles均为正。
- 当前PyPTO `torch.init` 与Simpler `KernelStaticConfig` 没有kernel-mode PMU通路。
  首次kernel-mode采集在初始化时报TypeError，任务 `task_20260927_154057_172940414996` 退出1；
  已撤回该入口改动。现在使用已有program PMU能力，没有修改PyPTO/Simpler/PTOAS/PTO-ISA。

## 实测

| 任务 | block数 | 平均total cycles | Cube busy | Vector busy | Scalar busy | MTE1 busy | MTE2 busy | MTE3 busy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qk_pv AIC | 24 | 545378.21 | 21.23% | 0 | 55.35% | 24.87% | 22.56% | 0 |
| qk_pv AIV | 48 | 549963.79 | 0 | 33.97% | 47.83% | 0 | 35.41% | 14.71% |
| merge_norm AIV | 48 | 72176.85 | 0 | 43.50% | 19.80% | 0 | 31.91% | 17.92% |

qk_pv的指令cache miss/request比率：AIC 0.443%，AIV 0.172%。
这些是独立case的计数器结果，不能仅据低miss比率断定完整CSA没有指令cache影响。

判断：当前qk_pv不是持续占满Cube的纯矩阵乘瓶颈，Vector和MTE也没有全程busy。
这与只改PV N128、联合softmax、16行搬运未获得明确本体收益的现象相容，
但不能反过来用PMU证明每个失败候选的成因。
Scalar busy较高，尚不能区分地址/循环控制、同步等待和其他停顿，更不能全部算成Python或跨任务调度。
下一步沿已有KV-ready、Score-ready、Prob-ready、PV-ready边界定位核内串行段，
再决定是否移植Native成对DMA、改变预取或压缩循环控制；不直接继续扩大分块参数搜索。

## 输出看护与证据

固定Native输入后，稀疏注意力输出max_abs=0.0009765625、RMSE=0.00006348421，非有限值0。
零容差仍FAIL，3328649/7864320元素存在差异；这不是整模型token/DSpark验收。

- [采集命令](run.sh)
- [离线汇总脚本](summarize.py)
- [完整计数器汇总与误差](summary.json)
- [原始PMU CSV](standalone_pipe/build/dfx_outputs/pmu.csv)
- [func_id映射](kernel_names.json)
- 本地Native输入：`native/native_sparse.pt`（大张量不入Git）。
- 本地生成代码：`standalone_pipe/build/kernels/`。

复现先通过 `task-submit --device 0 --max-time 600 'bash <本目录>/run.sh'` 执行，
结束后在CPU执行 `python <本目录>/summarize.py`。
