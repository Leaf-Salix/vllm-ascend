# d253：dense projection 有界只读机会审查

结论：建议只保留一个最小新候选——**O_A→O_B 量化中复用第一次 FP32
输入加载，删除相同 tile 的第二次 GM load**。它不改变 GEMM 分块、K 累加、
量化算术、输出所有权或同步。尚未实现、编译或上卡；不能预记性能收益。

本次只读正式 `d253d627` 的五份投影源码，逐字核对它们仍与已执行 d56
版本相同。因此可用 d56 的真实生成物检查这些 helper 的实际行为；这不表示
d253 的整个调度程序与 d56 相同。证据为 `evidence.json`，提取脚本 `extract.py`
只读文件并做整数统计，不导入设备库、不编译。

## 已审查范围及排除项

| 阶段 | 当前源码数学 tile / 原 task 实际 L0 | 本轮判断 |
| --- | --- | --- |
| Q_A | M32/M64 完整块、M16 尾块，N128/K256；原 task L0 拆 K128 | 每个输出保持单条 K 链。完整块与尾块之间可能重复权重读取，但消除需要双 Acc 与循环重排；没有发现可直接删掉的重复 load。 |
| KV | M32/M64、M16 尾块，N128/K512；L0 K128 | 同上；还需保留 512 行分段与 3 个行组 owner。不能用当前七个 T 列表代替动态尾块合同。 |
| 主 compressor | M64/N64/K512，两条 FP32 链；L0 K256 | 输入 Mat 已供 KV/gate 共同使用。拆成 K256 后的 Left 搬运复用涉及两链交错和 L0 生命周期；不在已失败 L0 候选后再作无新依据变形。 |
| Indexer compressor | M16/N32/K512；L0 K512 | 真实 PTO 中同一 x Left 已同时供 KV/gate MMA 使用，不存在“每条链分别重复 load x”的问题。 |
| O_A | M128/N128 或 N256，GM K256；对应 L0 K128 或 K64 | 双输出 tile 输入复用已有专门历史探针；K512、N256 与共享八组循环已有完整测试，均不重复提出。 |
| O_B | RB32/96/128、N256；实际 L0 K128 | RB≤96 已一次载入整行 activation 并复用两个 N256；RB128 已权重在外、行块在内驻留。交换循环树会改变当前真实驻留策略，不是漏做一次简单 hoist。 |

源码位置位于正式 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/`：
`qkv_stages.py:97`、`kv_stages.py:86`、`compressor_projection_stages.py:67/108`、
`output_projection_a_stages.py:114/184`、`output_projection_b_stages.py:127`。

已查总账 `DSV4_CSA_SPMD_OPTIMIZATION_LOG.md` 与历史 `ATTEMPTS.md`：

- `seven_stage_opportunities_v1/README.md` 已讨论 O_A 双输出独立 Acc 复用。
  T192 单组 CPU v2 虽通过，但 Left 退为单份、跨八组 NZ 合同未闭合；
  v3 Right 超容量。不能把同一个方案重新命名为新优化。
- O_A K512/N256/八阶段共享循环及 HCstream 已有完整结果。
- compact Right、主 Q_B Left 已有本轮设备结果，不再扫槽数或单点参数。
- 未在总账、ATTEMPTS 及过程 Markdown 找到 O_A 量化第一次 load 复用的
  独立尝试。原源码本来就有两次读取，不能称为动态融合新引入的 bug。

## 唯一候选：O_A 量化单次加载

正式 `output_projection_a_stages.py:212`：

```python
oc_amax = pl.load(o_r_pad, [qt, col_g], [8, 1024], valid_shape=[rows, 1024])
```

第 230 行对完全相同的 tensor、offset、物理形状和有效形状再次 `pl.load`，
得到 `oc_q`，随后只用于 `row_expand_mul(oc_q, g_sq_col)`。
拟议最小改动是删除这次 load，直接用第一次加载的 **原始带符号值**做乘法。
不能复用 `abs` 的结果；也不能手工强制原值与 abs 输出共用地址。

其余算术保持：abs → row_max → amax floor → `127/amax`；dequant scale
仍为 `amax*(1/127)`；缩放后依次 INT32 rint → FP16 round → INT8 trunc。
8 行有效尾块、32 行任务、组区间、INT8 pad 零写和 scale 未写区均不变。
没有新增跨循环长寿命 Tile，只延长同一 qt 内原值的使用期；原两级流水保留。

### 它是实际 load，不只是 Python 重复切片

`fused_quant.pto.txt` 对应现成融合编译物：

`parallel_20261018/compact_priority_h131072_b16_v1/build_output/decode_csa_spmd_ad5etrbf/ptoas/dynamic_csa_full.pto`

第 13448/13453 行分别有当前块的 `oc_amax` 和 `oc_q` TLOAD；下一块
还有同样两次 TLOAD。现成 stage2 的四份输入槽地址是
`65536/98304/0/32768`，各 32 KiB。
原分 task 的 `quant.pto` 第 39/44 行也存在这两次读取，保存在
`original_quant.pto.txt`。这证明当前编译器没有自动消掉冗余，但不是融合独有问题。

### 资源依据与待证限制

现在 abs 覆盖 `oc_amax` 的槽，`oc_q` 保留原值；另有 32 KiB row_max 临时区。
两级流水共四份输入 128 KiB，加临时区为 160 KiB 大块占用。
单次加载不能继续原地覆盖原值，必须为 abs 另设临时生命周期。
一种可行活跃安排为两份输入 64 KiB + abs 32 KiB + row_max 临时 32 KiB，
即 128 KiB；后续乘法/转换可复用已释放槽。小尺度 tile 另计。

这是有现成地址支持的生命周期可行性分析，**尚未证明 PyPTO 会实际给出该
分配或保住 stage2**。若实现，完整 CPU 审计应拒绝隐式额外 copy、流水退化
或整体 UB 超限；绝不为得到更少 load 添加全局 reserve/固定地址。

七档每次完整调用可删除的逻辑 GM 读取量：

| H / B / S | T | 少读 MiB |
| --- | ---: | ---: |
| 128K / 4 / 6 | 24 | 0.75 |
| 128K / 8 / 6 | 48 | 1.50 |
| 128K / 16 / 6 | 96 | 3.00 |
| 128K / 24 / 6 | 144 | 4.50 |
| 128K / 28 / 6 | 168 | 5.25 |
| 8K / 16 / 6 | 96 | 3.00 |
| 8K / 32 / 6 | 192 | 6.00 |

公式为 `T×8×1024×4` 字节；实际逻辑调用少 `8×ceil(T/8)` 次 tile load。
该规则自然覆盖任意合法 T/尾行，不按测试点白名单分支。
逻辑 GM 请求不等于 HBM 流量，原两次可能命中 L2；量化又与 O_A/O_B 按组
重叠，因此不能把字节数折算成整层收益，也没有依据宣称会接近 30% 目标。

本轮止于以上审查。正式源码、依赖、历史结果均未改；没有新增 CPU 编译或
设备任务。报告目录：

`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/ultra_dense_projection_audit_20261018`
