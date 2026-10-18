# 独立静态CSA参考测试

算子全部来自 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_static_reference/`：
数学入口、signals、cached100编译处理、参数与调度表都在该包内。
本目录只保留测试驱动、通用依赖环境加载和L1导出；不调用历史静态生成器，
也不读取旧的specialization、layout或结果目录。

在仓库根执行：

```bash
task-submit --device auto --max-time 1800 \
  'bash tests/pypto_test/csa_static_reference/run.sh h131072_b16_example --batch 16 --history 131072 --l1-windows 3'
```

支持归档的128K/B4、B8、B16、B24和8K/B16、B24、B32，均S6。
128K/B28尚无已验收静态参考，直接拒绝；它的历史导入失败按用户要求不修复。

使用正式权重 `/data/model/DeepSeek-V4-Flash-0731-w8a8` 第4层和固定seed合成历史；
原多task PTO作为精度与性能参考，不调用Native attention。
先检查A→B→A的完整输出、Top-K及score、cache/state、arena/Native保护区和进度，
再检查图重放。L1统计24 AIC+48 AIV最早incore开始至最晚结束，保留内部等待；
不计外围AICPU/orchestrator首尾。`--l1-windows`可选0、3、5。

独立搬迁验证任务 `task_20261018_095114_228736420852` 已完成，exit=0；
输出 [result.json](h131072_b16_v1/result.json)。128K/B16/S6 的 A→B→A 输出、
Top-K/score、完整 cache/state、Native/arena 保护区、72 核进度与只读控制表、
图重放均通过，实际编译来源为 `ops/pypto` 中本次归档入口。

| 完整 L1 incore 跨度 | 三次样本（μs） | 中位数（μs） |
| --- | --- | ---: |
| 原多 task PTO | 864.90、854.08、880.98 | 864.90 |
| 静态单 SPMD | 782.66、778.06、787.36 | 782.66 |

本轮实际采样顺序均为多 task 后静态，原始结果如实保留。后续测试驱动已修正
两方采样的 AB/BA 轮换；这是测试编排修正，没有修改算子或追改已测结果。
其余六档本次未重测，只保留原历史精度与性能证据。
历史静态结果不计入动态单SPMD降低至少30%的目标。
