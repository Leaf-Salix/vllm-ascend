# Native CANN 9.2.0-beta.2 兼容性与性能验证

用户指定 `/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2/set_env.sh`。
先以相同 vLLM Ascend 源码、Python、ATB 和自定义算子比较 CANN 9.0/9.2，
两档设备验证已通过，已将公共环境切到 9.2，用于后续 CSA/HCA 新进程及 Native 性能验证；
不混入旧 CANN 9.0 历史结果。[结果与四份 profiling JSON](RESULTS.md)。

## 固定范围

- 冻结源码 `9d237d33`，独立 worktree `.cache/csa-native-cann92-9d237d33`；
  包含 WO_A ND 和 arena 配置修改，但本轮只运行 Native。
- A3 单卡、正式权重第二个 CSA 层（layer 4）、合成历史、query 6、mode 2、det 0、EPLB 关闭。
- Python 3.10、torch 2.10.0、torch_npu 2.10.0.post2，ATB 9.0 与 release 自定义算子包保持一致。
- 无 profiler 的图重放预热 5 次、采样 20 次；另采一次 PyTorch profiling JSON。
  区间为 HC_pre→norm→CSA→HC_post，初态恢复在计时之外；不含 MoE/通信，不是整网 forward。
- 优先 128K/B16 和 8K/B16。有限值、Top-K 索引和保护区检查复用现有 Native 图回放入口。
  det 0 不要求浮点逐 bit 自一致；这不是新的模型 token/DSpark 验收。

## 版本与算子来源

[CPU 导入证据](cpu_import.json)确认现有 Torch/NPU 与 Native C++ 扩展可以加载。
`libascendcl`、`libruntime` 和 `libopapi*` 的进程映射均来自指定的 9.2 目录。
安装信息记录 runtime/ops-transformer 为 `9.2.0-beta.2`，构建时间 `20260826_163105218`。
仅目录名或导入成功不作为设备兼容性通过的结论。

Native `GetOpApiFuncAddr` 优先查 `ASCEND_CUSTOM_OPP_PATH`。
当前 `aclnnVllmQuantLightningIndexer`、`aclnnSparseAttnSharedkv` 是 release 自定义接口，
9.2 内置库不提供这两个同名入口。RMSNormDynamicQuant/Compressor 的同名符号也优先使用配套 custom ABI。
因此本轮是“Native release 自定义算子 + CANN 9.2 运行库及内置算子”，
**不是已经将 Indexer/Sparse 换成最新 ops-transformer 实现**。
不可直接去掉 custom 包或按同名符号替换：历史验证日志 §14 记录过参数个数不同造成的 ABI 错配。

最新 ops-transformer 源码仍是核内优化的首要参考；源码参考版本与本次安装包版本分别记录。
官方版本资料：[CANN 9.2 beta.2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/index/index.html)、
[TorchNPU 配套关系](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)。
当前组合是否可用以本机实测为准，不将其声明为官方推荐配套。

## 复现与当前状态

```bash
task-submit --device auto --max-time 1800 \
  'bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_native_cann92_20260928/run.sh'
```

[隔离环境](env.sh)、[设备入口](run.sh)、[Native 用例与动态库取证](native_case.py)。
脚本拒绝覆盖已有 report；下一轮用新输出目录，不混入本轮采样。

首轮 `task_20260928_214924_80499418895` 因测试脚本未创建 `ASCEND_CACHE_PATH` 目录退出 1，
尚未执行完整 CSA。原始失败在 `failures/cache_directory_missing/run.log`，不是 CANN 算子不兼容结论。
修正后 `task_20260928_215010_8189894329` 已完成退出 0，四次进程在同一张 card 1 上完成。
128K/B16 均值 1326.612→1323.392 μs（−0.243%），8K/B16 939.637→938.628 μs（−0.107%）；
均无明显加速，不将微小变化作为新算法收益。两档 P95 下降；长档 max 1332.96→1333.86 μs略升，
短档 max 946.00→943.74 μs。有限值、Top-K、metadata 和保护区通过。

两份 9.2 profile 最初因另一用户安装目录的属主检查而导出失败，设备原始记录已保留。
把同版本 `tools/profiler`（约39MB）复制至本用户 `.cache/dsv4-toolchain/cann92-profiler/tools/profiler`，
本地副本去掉 group/other 写权限后，离线重导出通过，每份包含43个真实设备 kernel。
见 [重导出日志](reexport.log)。没有修改对方目录，没有重跑 NPU，没有使用旧9.0 profiler解码9.2数据。

## 公共环境与 HCA

用户随后要求直接修改公共入口，现已修改工作区根目录 `env-dsv4-0251rc1.sh`，`env.sh`继续转调它。
HCA 的 `run_hca_single_layer.sh`、`run_hca_compressor.sh`、`run_hca_decode.sh`直接source这个公共入口，
性能/candidate脚本经单层脚本继承，因此不需修改HCA算子或覆盖该会话的未提交工作。

公共脚本清除上一套CANN的PATH、动态库、Python和CMake路径，保留固定release custom包及ATB 9.0。
先退出已激活venv的旧PATH再切CANN，避免后续activate把旧9.0编译器路径恢复回来。
profiler使用前述同版本本用户副本；bisheng、CANN运行库、OPP来自用户指定9.2目录。
PTOAS仍为0.66，PyPTO/Simpler调试分支和Python环境保持原值。

[公共环境检查](shared_env_check.json)覆盖：已激活9.0环境→env.sh→重复source→HCA源码激活、
PyPTO导入与Native扩展导入，旧CANN路径为零。该项只做CPU加载检查，未声称HCA整层或PTO新工具链设备验收完成。
已启动进程仍使用启动时环境，后续新任务自动切换；旧9.0结果仅作历史证据。
根目录公共脚本不在本仓Git范围，原样归档[修改前](shared_env_before.sh)和[修改后](shared_env_after.sh)供审查；
这些快照依赖工作区根目录位置，不能从results目录直接source。
