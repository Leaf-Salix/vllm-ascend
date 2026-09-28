# CSA 单层 graph 精度回归

此目录提供显式运行的 NPU 测试工具，不会被 pytest 自动收集。目标环境为
CANN 9.0.1、vLLM-Ascend 0.25.1rc1 及与当前 CSA 分支匹配的 PyPTO/simpler feat 环境。

## 测试范围

- 从本地 `DeepSeek-V4-Flash-0731-w8a8` 权重及索引读取一个真实 C4 attention 层，默认第 2 层。
- hidden 和历史 cache 是固定种子的合成数据；这是单层 attention 测试，不是完整模型生成或真实 prefill 的产物。
- 原生分支直接调用 `AscendDSAImpl.forward`；CSA 分支调用同一实例的生产 `forward`，使用原生接口、metadata 和 cache。
- 两边使用相同权重、hidden、初始历史和 metadata；CSA 若回退原生、返回非调用方 output，或者导入了错误源码，测试立即失败。
- 对原生与 CSA 分别先保存 A/B eager 结果，再 capture 一张图，依次 replay A/B/A。A/B 只修改固定地址上的 hidden 内容；metadata、position、seq length 和初始历史保持固定。
- 每轮恢复初始 cache。replay 输出及全部 cache allocation 必须与该实现对应的 eager 结果完全相同；CSA Python 调用计数不应增长，output 尾部保护区必须完整。
- 精度报告对比原生与 CSA 的 output，以及本轮写入的 6 类 cache slot；仅对写入位置做精度统计，避免大量未改历史稀释误差。

这里的 A/B/A 不验证 metadata 更新、多步 cache 累积、跨请求调度、整网 DSpark 接收率或多卡执行。
`--seq 6` 表示每个请求一次处理 6 个 query token，并不模拟 draft 生成及接受决策。

## 参数

从仓库根目录使用模块方式执行：

```bash
python -m tests.pto_attn.precision.graph_replay_ab --help
```

| 参数 | 含义 |
| --- | --- |
| `--model` | 本地模型目录，须有 `quant_model_weights.safetensors.index.json` 及对应分片 |
| `--extension-dir` | 已编译且版本匹配的 `vllm_ascend` 包目录，包含原生扩展和 custom vendor |
| `--variant-dir` | 待测源码根目录，其下须有 `vllm_ascend/attention/pto_attn.py` 及生产 CSA kernel |
| `--out-dir` | 本轮结果目录，已有 `result.json` 时拒绝覆盖 |
| `--layer` | checkpoint 层号，默认 2；必须是当前 CSA 支持的 C4 层 |
| `--batch` / `--seq` | 默认 B=4、S=6；标准回归使用 S=6 |
| `--start-pos` | 默认 8191，首个 query 的绝对位置；本轮最大位置为 `start-pos + seq - 1` |
| `--iterations` | 每边计时样本数，默认 20；另有 5 轮计时预热 |
| `--device` | 任务内部可见设备编号，默认 0 |

## 执行命令

先激活已验证的独立运行环境。`ENV_SETUP` 应配置相容的 Python、PyPTO、simpler、CANN、
`PTOAS_ROOT` 和动态库路径；测试不会安装包或编译 vLLM-Ascend 原生扩展。
以下是单卡任务脚本内容；将五个绝对路径换为自己的目录，保存为 `run_graph.sh`。

```bash
#!/usr/bin/env bash
set -euo pipefail
ENV_SETUP=/absolute/path/to/environment/activate.sh
TEST_REPO=/absolute/path/to/this/checkout
MODEL_DIR=/absolute/path/to/DeepSeek-V4-Flash-0731-w8a8
EXTENSION_DIR=/absolute/path/to/compiled/vllm_ascend
OUT_DIR=/absolute/path/to/new/result-directory

source "$ENV_SETUP"
export PYTHONNOUSERSITE=1
export ASCEND_RT_VISIBLE_DEVICES="${TASK_DEVICE:?submit through task-submit}"
export VLLM_ASCEND_PYPTO_DSV4_CSA=1 VLLM_ASCEND_ENABLE_NZ=0
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HCCL_DETERMINISTIC=true PYPTO_CACHE=0
vendor="$EXTENSION_DIR/_cann_ops_custom/vendors/custom_transformer"
export ASCEND_CUSTOM_OPP_PATH="$vendor"
export LD_LIBRARY_PATH="$vendor/op_api/lib:$EXTENSION_DIR:${LD_LIBRARY_PATH:-}"
export PYPTO_PROG_BUILD_DIR="$OUT_DIR/cache/build"
export ASCEND_CACHE_PATH="$OUT_DIR/cache/ascend"
export ASCEND_WORK_PATH="$OUT_DIR/cache/work"
export ASCEND_PROCESS_LOG_PATH="$OUT_DIR/cache/log"
mkdir -p "$OUT_DIR" "$PYPTO_PROG_BUILD_DIR" "$ASCEND_CACHE_PATH" \
  "$ASCEND_WORK_PATH" "$ASCEND_PROCESS_LOG_PATH"
cd "$TEST_REPO"
python -B -m tests.pto_attn.precision.graph_replay_ab \
  --model "$MODEL_DIR" --extension-dir "$EXTENSION_DIR" \
  --variant-dir "$TEST_REPO" --out-dir "$OUT_DIR" \
  --layer 2 --device 0 --batch 4 --seq 6 --start-pos 8191 --iterations 20 \
  2>&1 | tee "$OUT_DIR/run.log"
```

使用服务器的设备队列提交，按当地队列配置选择超时：

```bash
task-submit --device auto --max-time 1800 'bash /absolute/path/to/run_graph.sh'
```

调度器需提供 `TASK_DEVICE`。只在队列分配的设备上执行；其他队列可用自己的等价设备绑定。
修改 B、S 或 context 时使用新的结果目录，首次编译时间不包含在 replay 计时中。

## 结果解释

`result.json` 保存运行环境、生产 Python 源码 SHA256、权重清单、输入摘要、replay 校验、
output/cache 误差及所有 NPU event 计时样本。异常会记录 traceback 并返回非零退出码。

- `complete=true`：执行及 A/B/A replay 合约通过。原生/CSA 精度是否可接受仍应检查指标；脚本未设置或放宽精度阈值。
- `output_metrics.A/B`：relative L2、绝对误差、逐元素相等比例、BF16 ULP。
- `cache_metrics.A/B`：本轮写入的 compressed、raw、main state、inner state、index key、index scale。
- ULP 同时报告所有有限值和参考绝对值 ≥0.01 的子集，另报非有限数；不能只摘选有利子集。
- `event_ms`：用 NPU event 包围单次 graph replay；每次 reset、CPU/Python 调用和同步均在计时区间之外。原生/CSA 执行顺序交替，去掉前 5 轮，再报告中位数和全部样本。

比较历史版本时固定模型、层号、输入摘要、B/S/position、运行环境及计时方法，并把结果和源码身份写入
`docs/source/developer_guide/DSV4_CSA_TEST_HISTORY.md` 或其链接的报告。
