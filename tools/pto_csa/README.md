# PTO CSA 接入的脚手架

`vllm_ascend/attention/pto_csa.py` 是**产品代码**——跑在 vLLM 进程里，把 DeepSeek-V4
ratio-4 层 decode 路径的 CSA 换成 pypto-lib 的 PTO 算子。这个目录是**让它可被验证**的外围：
建环境、占卡起服务、采 trace、以及不依赖模型权重的离线自查。

完整复现步骤、版本坐标、已知限制见 [REPRODUCE.md](REPRODUCE.md)。

## 先跑这个

```bash
tools/pto_csa/kernel/run_adapter_check.sh
```

**不要模型权重、不起服务、约 1 分钟。** 它用 pypto-lib 自己的 fixture 反造出 vLLM 侧的
数据约定，过一遍适配器，再对 pypto-lib 的 golden。适配器里那套索引换算（压缩页重映射、
窗口物理槽反推、wo_a 转置、wo_b 再量化、decode 批次摊到编译期固定的 `(B, S)`）全靠它守着。
改完 `pto_csa.py` 先跑它，不要直接去起整网。

## 三组脚本

| | 用途 |
| --- | --- |
| `serving/` | 建 venv、拉权重、占卡起 vLLM、压测、采 trace |
| `kernel/` | 只验 CSA 本身：离线对拍、共存门禁、kernel 模式捕获验证、失败归因 |
| `stack/` | 从零建一套**不依赖他人目录**的栈：自带 CPython、从上游自建 PyPTO/simpler |

`stack/build_own.sh` 是 `serving/build_env.sh` 的替代路线。（目录不叫 `env/` 是因为仓库的 `.gitignore` 用 `env/` 挡 virtualenv。）后者的基础解释器和 ATB 都借自
别人的目录，前者自带 CPython（python-build-standalone，含 ssl）、把 ATB 拷进自己目录、
从上游 clone 并自建 PyPTO 与它的 simpler 子模块。kernel 模式必须走这条 —— 它要求
`-DPYPTO_BUILD_TORCH_NPU=ON`，而那个扩展必须与实际运行的 torch 版本同套重编。

`serving/` 里的入口是 `run_dsv4_mtp_vllm.sh`（挑端口 → `task-submit` 拿设备锁 → 把整轮交给
`dsv4_case_inner.sh`）。卡是共用的，**不要绕过 `task-submit` 直接起服务**。

`kernel/run_kernel_mode.sh` 验 kernel 模式：真实 CSA kernel 能否被 `NPUGraph` 捕获、
重放是否正确、改输入后输出是否跟着变。第三条最容易漏 —— 捕获若把值烘死在图里，重放会
"成功"但结果不更新，接到服务里表现为输出静止不动，很难定位。

## 出了问题从哪查

`kernel/analyze_dump.py` 把 live 那一步的入参重放一遍，切成三段分别归因：

```
golden(翻译后入参) vs PTO 输出   -> kernel 有没有照着算
golden stage-1 vs vendor stage-1 -> 窗口/压缩槽的翻译对不对（不含投影）
golden 后半段 vs vendor 输出     -> 逆 RoPE + o_proj 的建模对不对
```

服务端只会给你一句"输出不对"。这次接入踩到的两处翻译错误（压缩缓存页大小、RoPE 表排布）
就是靠这个切开的——第一段只差 0.0156 说明 kernel 没问题，第二段差 0.39 说明翻译走偏了。

## 开关

| 环境变量 | 作用 |
| --- | --- |
| `PTO_CSA=1` | 打开层级替换。不设时 `pto_csa.py` 对 vLLM 没有任何影响 |
| `PTO_CSA_VERIFY=1` | 每次替换的第一步做一遍逐入参一致性断言 |
| `PTO_CSA_PROBE=<dir>` | 只读探针，把两处调用点的张量长相落盘 |
| `PTO_CSA_DUMP=<dir>` | 把第一次替换用到的全部张量落盘，供 `analyze_dump.py` 重放 |
| `PTO_CSA_REPORT=<file>` | 逐步 PTO vs vendor 对拍与回退统计 |
| `VLLM_ASCEND_PROFILER_LEVEL` | `Level0` / `Level1`（默认）/ `Level2` |
