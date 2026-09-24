# dev/pypto-dsv4-csa-tnd-opus55-20260924 测试清单

**日期**：2026-09-24  
**分支**：`dev/pypto-dsv4-csa-tnd-opus55-20260924`  
**起点提交**：`8b9d430aa`（tnd 分支 HEAD）  
**本分支当前提交**：见 `git log -1`  
**Worktree**：`/Users/jiayetcs/Desktop/Project/PyPTO/vllm-ascend-pypto-dsv4-csa-tnd-opus55-20260924`  
**Fork**：`Leaf-Salix/vllm-ascend`  
**设计文档**：`reports/dsv4-tnd-opus55-20260924/design.md`

---

## 本分支相对 tnd 分支的改动

| 文件 | 类型 | 内容 |
|------|------|------|
| `vllm_ascend/attention/pto_kernels/dspark/service_config.py` | 新建 | `QUERY_TOKENS=6`、`can_replay_csa_graph()` |
| `vllm_ascend/platform.py` | 新增逻辑 | ACLGraph 捕获档位对齐到 6 的倍数 |
| `vllm_ascend/worker/model_runner_v1.py` | 新增逻辑 | CSA 图重放闸门 + `process_weights_after_loading` hook |

`vllm_ascend/attention/pto_attn.py` 与 `decode_indexer_compressor.py` 经过两轮精度尝试和
两轮回退，当前内容与 tnd 起点 `8b9d430aa` 一致，不计入本分支净改动。

**已被推翻的结论（保留作为排查记录）**：初版曾把 `cmp_norm_w` / `inner_norm_w` 的 dtype
由 FP32 改成 BF16，并称其为「核心精度修复」。227 实测证明该方向错误：kernel ABI 要求
FP32，`process_weights_after_loading` 把这两个权重从 BF16 扩展到 FP32 是正确行为，tnd 原始
的 FP32 验证也是对的。改成 BF16 会直接报
`Parameter 'cmp_norm_w' expects dtype torch.float32, got torch.bfloat16`（批次 1、2，
exit=1）。该改动已由 `0b268ff8a` 回退。

---

## 源码身份（每次测试前核对）

### 本机

| 组件 | repo | branch | commit |
|------|------|--------|--------|
| vllm-ascend | Leaf-Salix/vllm-ascend | dev/pypto-dsv4-csa-tnd-opus55-20260924 | 待填 |
| PyPTO | — | feat/kernel-mode-integration-test | 54957491ede07ad5d5015f5e69874f367113cf45 |
| Simpler | — | feat | 32dff953d07f6bd2aacab8532860f28aca6df931 |

### 227 服务器（测试前建立）

| 项目 | 值 |
|------|-----|
| SSH | `pto227`，用户 `yejia` |
| 实验根 | `/data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924`（待建立） |
| Python | `<ROOT>/env/bin/python`（复用既有 env 符号链接） |
| CANN | `<ROOT>/Ascend/cann-9.2.0-beta.2` |
| ATB | `<ROOT>/Ascend/nnal/atb/9.2.0-beta.2/atb --cxx_abi=1` |
| Torch | 2.10.0 |
| Torch-NPU | 2.10.0.post4 |
| vLLM/vLLM-Ascend | 0.29 系列 |
| DSV4 权重 | `/data/model/DeepSeek-V4-Flash-0731-w8a8` |

---

## 阶段一：本机 CPU 合约测试

在本机 Python 环境下运行，不需要 NPU，验证 service_config 逻辑。

```bash
source /Users/jiayetcs/Desktop/Project/PyPTO/.venv311/bin/activate
# 本机 .venv311 没有装 vllm，导入整个 vllm_ascend 包会失败，按文件直接加载模块。
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location(
    'service_config', 'vllm_ascend/attention/pto_kernels/dspark/service_config.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
QUERY_TOKENS, can_replay_csa_graph = m.QUERY_TOKENS, m.can_replay_csa_graph

assert QUERY_TOKENS == 6

# 在 CSA 范围内，uniform decode，通过
assert can_replay_csa_graph(padded_tokens=48, num_tokens=48, num_reqs=8, uniform_decode=True, max_batch_size=64)

# 在 CSA 范围内，非 uniform，拒绝
assert not can_replay_csa_graph(padded_tokens=48, num_tokens=42, num_reqs=7, uniform_decode=False, max_batch_size=64)

# 在 CSA 范围内，uniform 但 token 数不是 reqs*6，拒绝
assert not can_replay_csa_graph(padded_tokens=48, num_tokens=42, num_reqs=8, uniform_decode=True, max_batch_size=64)

# 超出 CSA 范围，放行
assert can_replay_csa_graph(padded_tokens=512, num_tokens=100, num_reqs=10, uniform_decode=False, max_batch_size=64)

# padded_tokens 不是 6 的倍数，放行（不是 CSA 档位）
assert can_replay_csa_graph(padded_tokens=50, num_tokens=48, num_reqs=8, uniform_decode=True, max_batch_size=64)

print('service_config CPU 合约：PASS')
"
```

**预期结果**：`service_config CPU 合约：PASS`  
**实际结果**：PASS（2026-09-24）

---

## 阶段二：导入和 dtype 静态验证（本机，无 NPU）

验证 `cmp_norm_w` / `inner_norm_w` 仍按 kernel ABI 保持 FP32 验证（不得改回 BF16）。

```bash
source /Users/jiayetcs/Desktop/Project/PyPTO/.venv311/bin/activate
python -c "
import sys
with open('vllm_ascend/attention/pto_attn.py') as f:
    src = f.read()
lines = [l for l in src.splitlines() if ('cmp_norm_w' in l or 'inner_norm_w' in l) and 'float32' in l]
if not lines:
    print('FAIL: norm weight 的 float32 验证丢失，kernel ABI 要求 FP32')
    sys.exit(1)
print('dtype 静态检查：PASS')
"
```

**预期结果**：`dtype 静态检查：PASS`  
**实际结果**：PASS（2026-09-24）

---

## 阶段三：227 环境预检（无设备）

SSH 到 227，建立实验根，rsync 最新源码，然后做无 NPU 预检。

### 3.1 建立实验根（新目录，符号链接复用已有 env 和 Ascend）

```bash
ssh pto227 "
ROOT=/data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924
mkdir -p \$ROOT/src \$ROOT/logs \$ROOT/bin
ln -sfn /data/pyptouser/yejia/vllm-cann92-dsv4-tnd-main-20260924/env \$ROOT/env
ln -sfn /data/pyptouser/yejia/vllm-cann92-dsv4-tnd-main-20260924/Ascend \$ROOT/Ascend
echo 'ROOT built'
"
```

### 3.2 rsync 源码

```bash
COPYFILE_DISABLE=1 /opt/homebrew/bin/rsync -avP \
  --exclude='._*' --exclude='.DS_Store' --exclude='__pycache__' \
  --exclude='*.pyc' --exclude='.git' \
  -e "ssh" \
  /Users/jiayetcs/Desktop/Project/PyPTO/vllm-ascend-pypto-dsv4-csa-tnd-opus55-20260924/ \
  pto227:/data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924/src/vllm-ascend/
```

### 3.3 版本和路径核对

```bash
ssh pto227 "
ROOT=/data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924
set +u
source \$ROOT/Ascend/cann-9.2.0-beta.2/set_env.sh
source \$ROOT/Ascend/nnal/atb/9.2.0-beta.2/atb/set_env.sh --cxx_abi=1
set -u
export PYTHONNOUSERSITE=1
export PATH=\"\$ROOT/env/bin:\$PATH\"
export PYTHONPATH=\"\$ROOT/src/vllm-ascend\${PYTHONPATH:+:\$PYTHONPATH}\"

\$ROOT/env/bin/python - <<'PY'
import os, acl, torch, torch_npu, vllm, vllm_ascend
for m in (acl, torch, torch_npu, vllm, vllm_ascend):
    print(m.__name__, getattr(m, '__version__', None), m.__file__)
for k in ('ASCEND_HOME_PATH','ATB_HOME_PATH','PYTHONPATH'):
    print(k, os.environ.get(k))
PY
"
```

**预期**：vllm_ascend 路径指向 `vllm-cann92-dsv4-tnd-opus55-20260924/src/vllm-ascend`，无 import 错误。  
**实际结果**：待填

### 3.4 ATB 插件动态库检查

```bash
ssh pto227 "
ROOT=/data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924
set +u
source \$ROOT/Ascend/cann-9.2.0-beta.2/set_env.sh
source \$ROOT/Ascend/nnal/atb/9.2.0-beta.2/atb/set_env.sh --cxx_abi=1
set -u
export PYTHONNOUSERSITE=1
export PATH=\"\$ROOT/env/bin:\$PATH\"
plugin=\$(\$ROOT/env/bin/python - <<'PY'
from pathlib import Path
import torch_npu
root = Path(torch_npu.__file__).resolve().parent
matches = list(root.rglob('libop_plugin_atb.so'))
if len(matches) != 1:
    raise SystemExit(f'expected one libop_plugin_atb.so, got {matches}')
print(matches[0])
PY
)
ldd \$plugin | grep -E 'libatb|not found'
\$ROOT/env/bin/python -c 'import ctypes, sys; ctypes.CDLL(sys.argv[1]); print(\"ATB plugin: ok\")' \$plugin
"
```

**预期**：`libatb.so` 正常解析，`ATB plugin: ok`。  
**实际结果**：待填

---

## 阶段四：227 NPU 单层精度对拍

对拍脚本基于 tnd 分支已有的 `ab.py` 模式，逐中间量对比 CSA 与原生路径。
使用第 2 层真实 C4 权重，B4/S6，合成 hidden。

### 4.1 对拍脚本路径

```bash
ssh pto227 "ls /data/pyptouser/yejia/vllm-cann92-dsv4-tnd-main-20260924/bin/tnd-ab/"
```

复用旧脚本，修改 `vllm_ascend` PYTHONPATH 指向新 opus55 checkout：

```bash
ssh pto227 "
ROOT=/data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924
TND_ROOT=/data/pyptouser/yejia/vllm-cann92-dsv4-tnd-main-20260924
mkdir -p \$ROOT/bin
cp -r \$TND_ROOT/bin/tnd-ab \$ROOT/bin/opus55-ab
# 修改 PYTHONPATH 指向 opus55 checkout
sed -i 's|vllm-cann92-dsv4-tnd-main-20260924/src/vllm-ascend|vllm-cann92-dsv4-tnd-opus55-20260924/src/vllm-ascend|g' \$ROOT/bin/opus55-ab/ab.py
"
```

### 4.2 等长 B4×S6，T=24（uniform）

```bash
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 1800 \
  --env VLLM_ASCEND_PYPTO_DSV4_CSA=1 \
  --env ROOT=/data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924 \
  "bash /data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924/bin/opus55-ab/ab.py uniform-b4"
```

**预期**：relative L2 显著低于 tnd 分支的 1.5%，目标接近 nalinaly 的"几个 bit"级别。  
**实际结果**：待填（task ID、退出码、L2 误差、CSA 执行次数）

### 4.3 非等长 B4，T=18

```bash
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 1800 \
  --env VLLM_ASCEND_PYPTO_DSV4_CSA=1 \
  --env ROOT=/data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924 \
  "bash /data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924/bin/opus55-ab/ab.py varlen-b4"
```

**预期**：relative L2 低于 tnd 分支。  
**实际结果**：待填

### 4.4 关键指标记录格式

每次任务完成后记录到本目录 `result-YYYYMMDD-HHMMSS.json`，格式参考：

```json
{
  "date": "2026-09-24",
  "branch": "dev/pypto-dsv4-csa-tnd-opus55-20260924",
  "vllm_ascend_commit": "",
  "pypto_commit": "54957491",
  "simpler_commit": "32dff953",
  "task_id": "",
  "device_num": 1,
  "case": "uniform-b4",
  "B": 4, "S": 6, "T": 24,
  "native_graph_ms": null,
  "csa_graph_ms": null,
  "csa_vs_native_pct": null,
  "relative_l2_eager": null,
  "relative_l2_graph": null,
  "csa_runs": null,
  "exit_code": null,
  "notes": ""
}
```

---

## 阶段五：图捕获和 ACLGraph 对齐验证

启动 smoke 服务，验证档位对齐逻辑实际触发。

### 5.1 检查档位是否为 6 的倍数

服务启动日志中搜索 `cudagraph_capture_sizes`：

```bash
grep -i "cudagraph_capture_sizes\|align_decode" \
  /data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924/logs/service.log | head -20
```

**预期**：所有档位是 6 的倍数。  
**实际结果**：待填

### 5.2 CSA 替换路径确认

日志中确认 `[pto-attn-ran]` 条目出现，不全是 `[pto-attn] declined`：

```bash
grep "pto-attn" \
  /data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924/logs/service.log | head -30
```

**预期**：有 `[pto-attn-ran]` 条目，`capturing=True` 出现在至少一个档位。  
**实际结果**：待填

---

## 精度目标

| 测试 | tnd 分支基线 | 本分支目标 | nalinaly 参考 |
|------|-------------|-----------|--------------|
| B4 等长，graph L2 | ~1.5% | < 0.1%（~几个 bit） | "几个 bit" |
| B4 非等长，graph L2 | ~1.7% | < 0.1% | — |

---

## 验收标准

本次测试通过的最低要求：

1. 阶段一 CPU 合约：PASS
2. 阶段二 dtype 检查：PASS
3. 阶段三 环境预检：无 import 错误，vllm_ascend 路径正确，ATB 插件加载 ok
4. 阶段四 精度对拍：relative L2 显著低于 tnd 分支基线（1.5%），方向正确
5. 阶段五 档位验证：所有捕获档位是 6 的倍数，CSA 替换路径实际执行

---

## 历史测试结果

### 2026-09-24 第一批精度测试（commit `0b268ff8a`）

**环境**
- vllm-ascend branch: `dev/pypto-dsv4-csa-tnd-opus55-20260924` commit `0b268ff8a`
- 227 实验根: `/data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924`
- CANN 9.2.0-beta.2, ATB 9.2.0-beta.2, Torch 2.10.0, Torch-NPU 2.10.0.post4, vLLM 0.29.0
- PyPTO: `feat/kernel-mode-integration-test` commit `54957491`
- Simpler: commit `32dff953`

**调试过程**
- 初版 BF16 验证（错误方向）：kernel ABI 期望 FP32，`cmp_norm_w must retain native torch.bfloat16` 报错，任务 exit=1
- 原因：`process_weights_after_loading` 把 norm weight 从 BF16 扩展到 FP32；kernel 签名是 FP32；tnd 原始 FP32 验证是正确的
- 修复：恢复 FP32 验证，task exit=0

**精度结果（均使用 8186 历史缓存，200 iter，确定性模式）**

| 中间量 | uniform-b4 (`6,6,6,6`) L2 | varlen-b4 (`3,4,5,6`) L2 | allclose 1e-2 |
|--------|---------------------------|--------------------------|---------------|
| compressed KV | 0.0001% | ~0% | ✅ |
| main_state | ~0% | ~0% | ✅ |
| inner_state | ~0% | ~0% | ✅ |
| raw KV | 0.286% | 0.286% | ✅ |
| index_scale | 0.261% | 0.403% | ✅ |
| **index_key** | **0.633%** | **0.753%** | ❌ |
| **output** | **1.435%** | **1.494%** | ❌ |

任务 ID：
- uniform-b4：`task_20260924_181554_191992030190`（exit=0）
- varlen-b4：`task_20260924_181554_19198061867`（exit=0）

**性能（graph median）**
- uniform-b4：native 0.588 ms，CSA 0.638 ms（CSA 慢 8.5%，可能含首轮编译 warmup 抖动）
- varlen-b4：native 0.571 ms，CSA 0.545 ms（CSA **快 4.5%**）

### 2026-09-24 精度根因诊断（commit `d309e9e27`）

**结论：误差不在 vllm-ascend 绑定层，在 pypto-lib kernel 本身**

opus55 第一批精度数字与 tnd scope3 **完全一致**（8 位小数精确到小数点后 8 位），确认问题在 kernel 层面，不是 vllm-ascend 绑定差异。

**已排除的方向：**
- `cmp_norm_w` / `inner_norm_w` dtype：kernel ABI 是 FP32，`process_weights_after_loading` 把它们从 BF16 扩到 FP32，tnd 原始 FP32 验证是正确的。
- HADAMARD_SCALE（`indexer_compressor_write` 旧路径）：该函数不被 ab.py 走到，加了也无效（精度数字完全未变）。
- HADAMARD_SCALE（`indexer_key_write_vllm`）：tnd 的 `prepare_weights` 里 `hadamard_idx = H^T / sqrt(IDX_HEAD_DIM)` 已做归一化，kernel 里不能再乘，否则 index_scale L2 爆炸到 91%。

**关键架构区别：**
- nalinaly 用 `indexer_compressor_write`（非 vLLM 路径），`hadamard_idx` 没有归一化，需要 kernel 里乘 HADAMARD_SCALE
- tnd/opus55 用 `indexer_compressor_pool_projected_vllm`，`hadamard_idx` 在 `prepare_weights:179` 已做 `.T / sqrt(IDX_HEAD_DIM)` 归一化，kernel 里不能再乘

**下一步**：用 tnd-main 现有的 `decode_csa_stage_probe.py` / `precision_probe.py` 框架，hook `normed_kv` 中间量（Hadamard 之前），对比 CSA 和 native 在 RMS norm 之后、Hadamard 之前的数值，定位 index_key 0.63% 误差的具体引入位置。

*后续每次测试追加新节，带日期和提交 hash。*

### 2026-09-24 第六批：Hadamard 缩放位置对齐原生（待跑）

**改动内容**

| 文件 | 改动 |
|------|------|
| `vllm_ascend/attention/pto_attn.py` | `hadamard_idx` 传原生未归一化的 `H.T`，绑定层不再除 `sqrt(IDX_HEAD_DIM)` |
| `decode_indexer_compressor.py` | 新增 `HADAMARD_SCALE`；`indexer_compressor_write_vllm` 与旧路径 `indexer_compressor_write` 均改为「矩阵乘 → 舍入 BF16 → 乘 scale → 再舍入 BF16」 |
| `decode_indexer.py` | 新增 `HADAMARD_SCALE`；q 路径在矩阵乘之后乘 scale（FP32，本轮不加新舍入） |
| 两个文件的 golden 模型 | `init_hadamard` 改为未归一化，参考实现同步乘 scale |

**依据**：原生 `models/deepseek_v4/indexer.py` 的 `rotate_activation` 是
`F.linear(x, H)` → BF16 舍入 → `* dim**-0.5` → BF16 舍入。原先 opus55 把
`1/sqrt(128)` 折进 BF16 权重，该值在 BF16 下不可精确表示，且少一次舍入，计算顺序与
原生不同。

**与第五批的区别**：第五批只在 kernel 里补乘 scale，绑定层的除法仍在，等于除了两次，
`index_scale` 爆到 91%。本轮是成对改动——绑定层不再除，kernel 在原生位置乘。q 与 kv
两条路共用同一份 `hadamard_idx`，必须同时改。

**本机静态验证**
- 阶段一 service_config CPU 合约：PASS
- 阶段二 dtype 静态检查：PASS
- 三个改动文件 `py_compile`：PASS

**227 AB 任务**

改了 kernel，提交前必须清空 build 目录强制重编：

```bash
ssh pto227 "rm -rf /data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924/logs/opus55-ab/build-*"
```

```bash
task-submit --ptoas 0.63 --device auto --device-num 1 --max-time 1800 \
  --env AB_RUN=uniform-b4 --env AB_LENGTHS=6,6,6,6 --env AB_ITERATIONS=200 \
  'bash /data/pyptouser/yejia/vllm-cann92-dsv4-tnd-opus55-20260924/bin/opus55-ab/run.sh'
```

**关注指标**：`index_key` 应从 0.633% / 0.753% 下降；`index_scale` 必须仍在 0.3%
量级，若再次出现 90% 量级即说明缩放被重复施加。

- uniform-b4 task ID：待填
- varlen-b4 task ID：待填
- 结果：待填
