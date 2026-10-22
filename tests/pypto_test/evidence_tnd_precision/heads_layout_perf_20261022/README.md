# Heads布局性能筛选证据

## 结论范围

8K/B16两轮先性能筛选，再做Native A/B八边界逐位比较，Direct和Packed均通过。
128K/B24完整图仅改善约0.6%，低于预设1%门槛，**跳过精度验收**。
三个任务均completed exit0并释放；不新增整模型token、短query或混合1～6行资格。

## CPU重算

Python3.11.8以上，在仓库根执行：

```bash
python tests/pypto_test/evidence_tnd_precision/heads_layout_perf_20261022/verify.py
```

仅使用标准库，不导入Torch、不占NPU。校验封包SHA、实际17助手身份与编译收据，
逐窗重算108个event样本的中位数/筛选和45个原始L1跨度，核对162个执行检查点。
8K核对已执行的A/B八边界及ABA；128K核对未进入精度步骤。它不是重新运行硬件验收。

## 文件

- 三份`heads-*-evidence.tar.gz`：实际result、preflight、运行日志、15个L1原始trace及当时助手/收据。
- `tools.tar.gz`：独立候选代码、逐处diff、准备/提交/观察/收集/编译脚本、重算器、编排C++文本、提交与终态记录。
- `RECHECK.json`：三任务CPU重算汇总。
- `STATIC_REVIEW.md`：源码、地址和实际编排检查说明。
- `SHA256SUMS`：封包及验证器的SHA256绑定。

归档不含权重、cache payload、失败Tensor或构建二进制。
生成编排C++仅用于证明真实缓冲与依赖，没有修改其代码。

## 227重新执行

复用已验收隔离环境和资产，具体身份在`binding.json`、preflight和compile receipts中：

1. 正式模型`/data/model/DeepSeek-V4-Flash-0731-w8a8`，实际只加载第4层。
2. `vector-mte2-20261019-codex/full-csa-live-staging-v2/env.sh`及其原SDK、静态13包和缓存。
3. `b28-precision-20261021-codex/whole-b28-qa168-v2/snapshot/source`、3399文件manifest、77运行源清单与原编译收据。

工具内`env.sh`引用现有资产；`task.py`/`collect.py`的REMOTE引用独立目录
`/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/heads-perf-20261022-codex`。
将工具还原到独立实验目录，若更换目录，须一致更新引用并记录新助手身份，不能冒用旧manifest。
不要覆盖原环境、清空旧缓存或重新构建整套SDK。

首次准备使用`compile.sh direct`和`compile.sh packed`，均为无设备编译，CCACHE_DIR明确指定、并行4。
新目录编译前不应已有`compiled-direct/packed`；保留已完成目录，不在旧目录反复编译。
完成后生成`manifest.json`，实际预检须匹配原环境，且NPU未初始化。

以下在本机工具目录执行，每次使用新case标签：

```bash
python task.py submit heads-8k-b16-v3 16 8192
python task.py observe heads-8k-b16-v3
python collect.py heads-8k-b16-v3
```

运行前用`env.sh`核对环境；只有task-submit任务设置TASK_DEVICE后才能执行`run.sh`。
每项单卡、外限780秒、入口750秒；提交后每5分钟内观察，不创建后台watch或定时任务。
未知提交结果先查队列，不自动重试。128K参数为batch24、history131072。

## 计时与筛选

完整HC-pre→HC-post、64Tensor单入口、纯DSL，前缀SPMD+后缀多task，禁止Native回退。
TP1、page32、NZ2、全S6、seed1024、Native确定性level1/HCCL、SK0、13静态包和DSA多流。

warm3，12窗×100次raw FULL图event，六种顺序各两次；另采5窗×1次完整incore L1，dep-gen关闭。
restore和检查在计时外；每窗口后、下一次restore前检查finite、33项guard/metadata及readonly。
先落盘性能汇总，event中位数至少快1%、至少10/12窗领先且L1中位数更快者才进入精度。
精度检查为相同A、hidden+0.125的B、恢复A；首差停。性能未达门槛就跳过，不登记为精度通过。
