# TND精度CSA：真实整模型性能（2026-10-16）

## 结论

测量当前 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex` 的TND精度实现，
生产数学commit为 `cbc810c8afab47c2425068049440424f014dcde3`。
8K/B40三轮中，替换CSA后target整模型forward均比Native慢。
128K/B24在排队期间按用户要求取消，没有开始占卡，不提供本轮128K结果。
没有更改生产数学源码，没有切换到另存的性能变体。

| 配置 | Native forward均值ms | TND精度forward均值ms | TND延迟变化 | Native周期均值ms | TND周期均值ms | TND周期变化 |
| --- | --- | --- | --- | --- | --- | --- |
| 8K/B40 | 89.116 | 114.459 | +28.44% | 104.627 | 133.132 | +27.24% |

每个均值来自3轮：每轮20个连续target forward、19个连续begin→begin周期。
先按同一步取16rank中最大的设备时长，再跨样本统计。
周期包含logits、sampling、DSpark draft、调度间隙和CPU metadata观测开销。
它不是全局时钟同步的服务吞吐；forward也不是单层CSA算子耗时。

## 配置与测量口径

- 正式完整模型 `DeepSeek-V4-Flash-0731-w8a8`，75分片，约313.36GB；真实问题和Native bank恢复的target/draft cache/state。
- TP1、DP=EP16；每卡40个不同问题。DP副本不计为不同问题。
- 每请求192 token，greedy、ignore EOS、DSpark提出5、实际接受；每臂先每请求96 token预热，再独立恢复bank测3轮。
- 每轮跳过8个满档decode step，采后续20个连续target forward。本轮满档240 token/40请求。
- 仅替换21层CSA，20层HCA保持Native；精度版保留Native HC/norm、WKV、Indexer权重投影和O-proj计算边界。
- FULL_DECODE_ONLY，capture [6,240]；NZ2、atomic0、确定性level1/HCCL true、event1、EPLB关闭。
- 官方static13包，super_kernel=false；不能与旧SK1/确定性关闭的单层表直接混比。
- KV blocks22000/util.95；异步调度、共享专家多流、CPU binding、OMP10。
- 不含prefill，forward不含模型加载、bank恢复、输入排队、logits/sampling/draft；cycle包含forward间全部实际间隙。
- 无profiler、DFX、模块tensor hook、逐步GPU→CPU input拷贝；计时事件预先创建，结束后统一同步。
- 三轮同进程重复；没有独立进程置信区间，也没有不同运行顺序的反转复测。

实际环境沿用[精度复现手册](DSV4_FLASH_CSA_TND_PRECISION_REPRO_20261016.md)：
Python3.12、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、CANN9.2.0-beta.2、
ATB9.2.0-beta.2、PTOAS0.66，私有PyPTO3e87a843基线运行包/Simpler a54c0509。
冻结源3337文件清单SHA `e4de4927d2e5758df65784bddc412d1d90227b56ed978ad2218df50c2bd540ca`；
运行包、bank计划、worker配置和本次helpers的SHA随原始证据保存。

## 8K/B40原始轮次

| 轮次 | Native forward ms | TND forward ms | Native周期ms | TND周期ms |
| --- | --- | --- | --- | --- |
| 1 | 89.000 | 114.102 | 104.621 | 132.488 |
| 2 | 89.091 | 113.244 | 104.544 | 128.817 |
| 3 | 89.258 | 116.031 | 104.716 | 138.092 |

| 8K/B40样本分布，ms | min | mean | max | P95 |
| --- | --- | --- | --- | --- |
| Native forward（60样本） | 87.332 | 89.116 | 90.237 | 89.932 |
| TND forward（60样本） | 111.617 | 114.459 | 155.312 | 117.904 |
| Native周期（57样本） | 103.613 | 104.627 | 105.632 | 105.425 |
| TND周期（57样本） | 127.810 | 133.132 | 224.050 | 144.090 |

TND分布更散，但三个完整轮次都明显慢于Native。
8K两臂各3轮×16rank×40请求×192 token，合计737,280个实际输出token，
全部与已验收的Native黄金token/完整接受事件一致，零抢占。
正式计时轮中，PyPTO generation/binary build、bypass/invalid/storage/forced-rebuild增量为0。
全部实际timing窗口为FULL，PTO capture选中21层精度实现，HCA均Native。

CPU几何观测包含请求身份/顺序、positions、query bounds、实际与padding token、DP向量，
跨臂/轮次/rank全部一致。保留的 `input_ids_cpu` 不是最终GPU输入：异步sample/draft会仅在GPU
scatter新token，placeholder也仅在GPU清理。最初审计将此字段纳入等值门禁失败；
只读核对实际vLLM源码后，排除了这一无效观测并保留原值。
Native三轮差异23,936处，其中16,224为-1↔0，7,712含陈旧正token，不能仅归因为placeholder。
本轮不宣称重新对拍了每步GPU input IDs或内部浮点逐bit；前一轮精度见证另有记录。

## 验证与证据

每臂每轮都检查：完整192 token、完整接受事件、零抢占、满档连续样本、
实际采样输出数量等于相邻positions推进、有限正计时、配置/源码/运行包一致、无窗口内新增编译。
预热96 token检查Native黄金前缀。首个已观察差异停止后续轮次/配置，不把失败数据当性能验收。
完整输出/事件核对在每个batch完成后执行，不宣称逐token即时中断。
独立审核8K配对157份文件及原始统计通过。

8K归档 `pair-8k-evidence.tar.gz` SHA256：
`bbc8043e9881a026955c99b32d23782bdc664254ac7b9943b570d1cc107e406e`。
包含实际脚本、全部16rank原始三轮token/events/时序/CPU metadata、配置、日志与身份清单；
不含大模型权重、bank tensor、设备编译缓存或venv。

本轮仅完成8K/B40；128K任务在pending状态已取消，后续不继续提交。
[结构化结果](evidence_tnd_precision/performance_20261016/comparison-8k.json)、
[身份和取消记录](evidence_tnd_precision/performance_20261016/index.json)、
[原始157文件证据归档](evidence_tnd_precision/performance_20261016/pair-8k-evidence.tar.gz)。

## 复现操作

以下在227已有同身份模型、bank、runtime、冻结源和精度黄金输出的前提下执行。
完整资产身份先按精度复现手册无设备校验；不要缺失bank后重新随机生成。
归档中的脚本保留实际测试字节，不把格式化后的副本当历史测量入口。

1. 本机校验证据tar的SHA，解包到不存在的新目录，逐项核对 `evidence-manifest.json`。
2. 参考 `helpers/run-case.sh`、`env.sh`、`sitecustomize.py`、`mixed_perf.py`、`perf_probe.py`。
   还原精度复现手册中的冻结源及bootstrap，核对3337文件源清单和运行包/static身份。
3. 新建独立实验目录和两个输出目录名字。只修改路径/输出标签，不改变测量配置或数学代码。
   同一轮所有臂使用完全相同的helpers；任务运行/排队期间禁止改脚本。
4. 无设备执行preflight、bank checker及4项CPU门禁，再提交8K Native。
5. Native成功后运行CPU审计；通过后提交同档PTO精度臂，再做配对CPU审计。
   本轮仅复测8K；任何输出差异停止后续任务。
6. 每个任务780秒上限，不把所有臂封装为长任务，不使用 `--wait` 或scheduler。
   每约5分钟读status，出现已确认异常只终止自己的任务；完整退出后归档。

未来新实验标签示例（在227，已解包档案 `E` 指向包含helpers的目录）：

```bash
export B=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex
export E=/tmp/real-model-perf-record
export RUN=r20261016a
python3 - <<'PY'
import os
from pathlib import Path
run = os.environ['RUN']
assert run and all(c.isascii() and (c.isalnum() or c == '_') for c in run)
job = Path(os.environ['B']) / ('real-model-perf-' + run)
job.mkdir()
for old in (Path(os.environ['E']) / 'helpers').iterdir():
    if old.is_file() and old.suffix in ('.py', '.sh'):
        text = old.read_text().replace('real-model-perf-20261016', job.name)
        text = text.replace('perf-real-', run + '-perf-real-')
        (job / old.name).write_text(text)
(job / 'analysis').mkdir()
for old in (Path(os.environ['E']) / 'helpers' / 'analysis').glob('*.py'):
    (job / 'analysis' / old.name).write_bytes(old.read_bytes())
PY
export J="$B/real-model-perf-$RUN"
bash -n "$J/run-case.sh" "$J/env.sh"
python3 "$J/test_perf_probe.py"
```

`run-case.sh`在分配卡后仍先检查实际Python/SDK/toolchain和bank；资源检查不满足立即退出。
上述4项CPU测试无需vLLM/NPU，不等价于设备预检。
新输出根为 `/data/pyptouser/yejia/lc128/$RUN-perf-real-...`。
源码和运行环境路径以实际脚本及前页精度资产为准，不搬错旧环境。

每臂单独提交（本轮 `CASE=8k`，`ARM=native` 或 `pto`）：

```bash
export CASE=8k
export ARM=native
task-submit --ptoas 0.63 --device auto --device-num 16 --max-time 780 \
  --env PYTHONNOUSERSITE=1 "bash $J/run-case.sh $CASE $ARM"
task-submit --status <返回的task-id>
```

必须先确认本臂completed exit0、16rank有3个round、没有STOP，再继续下一臂。
先Native单臂审计，再配对，输出必须为尚不存在的新文件：

```bash
export R=/data/pyptouser/yejia/lc128
python3 "$J/analysis/analyze.py" "$R/$RUN-perf-real-$CASE-native-20261016" \
  --output "$J/analysis/native-$CASE.json"
python3 "$J/analysis/analyze.py" "$R/$RUN-perf-real-$CASE-native-20261016" \
  --pto "$R/$RUN-perf-real-$CASE-pto-20261016" \
  --output "$J/analysis/comparison-$CASE.json"
```

审计拒绝不等工作量、图档/运行时漂移、缺样本、token/events差异及窗口内编译等异常。
黄金token比较发生在实际测试脚本中；归档后也可独立对照前一轮precision证据。
CPU staging IDs的已知非权威性质必须保留，不能为了“所有字段相同”误称GPU输入已复核。

## 本轮任务记录

| 臂 | task | 占卡执行时长 | 状态 |
| --- | --- | --- | --- |
| 8K Native | `task_20261016_141148_11396376298` | 6m48s | exit0，CPU审计通过 |
| 8K TND精度CSA | `task_20261016_142344_15207858973` | 10m50s | exit0，配对审计通过 |
| 128K Native | `task_20261016_143655_21623543737` | 未开始 | 用户要求取消pending，返回已取消；后查not_found |

没有本轮128K CSA任务，没有遗留本轮NPU任务，也没有创建自动调度。
