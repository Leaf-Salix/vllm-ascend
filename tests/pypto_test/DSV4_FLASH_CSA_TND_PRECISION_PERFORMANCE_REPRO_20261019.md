# TND性能调优V4：在227以新标签复现

仓库工具源码见 [prepare_tnd_precision_performance_repro.py](prepare_tnd_precision_performance_repro.py)。下面命令使用227已冻结、已验证的同逻辑原助手文件名。

本工具只准备目录和核对文件，不提交设备任务，也不修改原冻结助手。运行环境仍依赖227已有的模型、Native黄金结果、固定bank、静态13包、SDK和历史校验器；它不是可离线搬到任意服务器的独立发行包。

## 身份与复制边界

- 参考助手manifest：`b52d07a160c144704e2cf9def48a03ee5706efd9f155a1765515548d3eb1e24a`。
- 源码manifest：`6107e6ed747925793dcfe0e5188dee8486c3a8b91d0c1b7e073b62d38215bc28`，3365个文件；CLI严格锁定以上两个值。
- 仅复制helper-manifest列出的文件（含11个patch、全部JSON、`reference/mixed8k_cases.json`），以及snapshot manifest列出的源码。源码及其manifest原字节不变。
- 新job仅重定位`env.sh`、`preflight.sh`、`run-case.sh`的job位置，以及run-case的输出前缀；其余原助手逐字不变。重建新helper-manifest，并保留原manifest及变更记录。
- 不复制job运行时build/cache、模型、bank、venv。原3365清单本身包含6个历史`.pytest_cache`文件、归档证据和2个`.so`，为保持manifest身份仍照单复制；这些不是本次运行产生的缓存。
- `prepare_snapshot.py`只作为受校验的原助手归档。**不要执行它**：新job已具备精确snapshot，重复创建会被拒绝。
- destination必须不存在，且其父目录已经存在；拒绝路径重叠、符号链接、shell特殊字符、SHA变化和已有输出目录。复制途中若异常，保留半成品供核查，换一个全新destination；工具不覆盖或清理旧目录。

## 准备与无设备预检

在227使用全新标签，例如下列`repro-v4-20261019-a`（如已用过须换标签/目录）：

```bash
BASE=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex
TOOLS="$BASE/precision-overnight-20261019/fresh-label-repro-v4"
REF="$BASE/precision-overnight-20261019/whole-formal-v4"
NEW="$BASE/precision-overnight-20261019/whole-formal-v4-repro-a"
python3 "$TOOLS/prepare_fresh_repro.py" \
  --reference-job "$REF" --destination "$NEW" --label repro-v4-20261019-a
bash "$NEW/preflight.sh" 8k
bash "$NEW/preflight.sh" 128k
```

预检明确设置`ASCEND_RT_VISIBLE_DEVICES=""`、`TORCH_DEVICE_BACKEND_AUTOLOAD=0`，检查真实imports、3365源码SHA、62参数schema、原bank、Native黄金结果、工具链和助手身份；应均输出`PERFORMANCE_BOOTSTRAP_NO_DEVICE_OK`。不会调用真实模型推理。预检可重复；设备运行输出由排他mkdir保护，不能覆盖。

## 设备复现顺序（由操作者手动提交，本工具不会执行）

先8K/B40，192 token warmup +192 token measured；不开Native2。以下命令需要已通过上述预检的`NEW`变量：

```bash
task-submit --device auto --device-num 16 --max-time 780 --ptoas 0.63 \
  --env PYTHONNOUSERSITE=1 \
  --env ASCEND_HOME_PATH=/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2 \
  --env PTOAS_ROOT=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/experiments/nalinaly-71153bb3/reference-env/bin \
  --env CCACHE_DIR=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/src/pypto/.ccache \
  --env CMAKE_BUILD_PARALLEL_LEVEL=4 \
  "bash $NEW/run-case.sh 8k pto"
```

输出目录为`/data/pyptouser/yejia/lc128/perf-repro-v4-20261019-a-8k-pto`。任务完成exit0后，必须按原报告的CPU审计要求逐rank复核warm/round的token、事件、零抢占、21层CSA资格/62schema、actual43源码SHA和计时；改变job路径产生的是新的helper SHA，不能把旧helper SHA或旧PASS当新结果。审计助手应显式指向新的输出/root和新的helper-manifest，但source manifest仍必须为上面的6107值。

已有独立CPU审计器接受显式新root，可直接用于这次fresh-label输出（不改旧脚本）：

```bash
AUDIT="$BASE/precision-overnight-20261019/whole-formal-v4-audit"
python3 "$AUDIT/independent8k.py" \
  --root /data/pyptouser/yejia/lc128/perf-repro-v4-20261019-a-8k-pto \
  --identity FormalV4-repro-v4-20261019-a \
  --expected-source-manifest 6107e6ed747925793dcfe0e5188dee8486c3a8b91d0c1b7e073b62d38215bc28 \
  --candidate-math "$NEW/candidate-math.json" > "$NEW/independent8k.json"
```

审计退出0且JSON `pass: true` 后才继续；不能用输出文件存在代替完整通过。128K完成后使用同样参数运行`independent128k.py`，将root改为`...-128k-pto`、重定向到新的`independent128k.json`。审计会重新哈希actual3365源码及所有新helper，并从16rank原始样本重算token/事件及max-rank性能。

**仅8K全部通过后**，用同样的task-submit选项提交`"bash $NEW/run-case.sh 128k pto"`；仍max780秒、独立任务，输出`...-128k-pto`。128K是每rank提交24请求、计时窗21并发的既有口径，不能写成24同时活跃。发现首差立即定位，不跑后续大配置。

历史Native性能是旧基线，不是本次新Native2。128K后窗CPU input_ids观察存在既有0/-1 sentinel差异，必须单列，不宣称GPU输入逐字节一致。完整token/事件仍逐rank对原Native黄金验收。

## 本次工具验证

6项合成边界检查通过；227实际创建新job并分别完成8K和128K无设备预检，两者均输出`PERFORMANCE_BOOTSTRAP_NO_DEVICE_OK`，旧助手SHA未变。实际检查job使用独立`whole-formal-v4-fresh-cpu-check`，没有提交任何NPU任务。
