# O-A 核内对照（2026-09-26）

本目录仅保留数值摘要和上游方向诊断源码；原始模拟器采集、清理后的 trace 与
编译产物留在本地 `build_output/oa_incore_experiment/`，不作为真机整层性能证据。

| 实现 | 指令窗口 μs | MTE2 区间并集 μs | CUBE 区间并集 μs | Mat/L1 KiB |
| --- | ---: | ---: | ---: | ---: |
| 当前 Native `[G,K,N]` NZ，stage=2 | 13.459 | 11.763 | 7.008 | 256 |
| 同上，stage=4 | 16.425 | 12.964 | 7.008 | 512 |
| 上游 `[G,N,K]` NZ、b_trans，stage=2 | 13.417 | 11.767 | 7.008 | 256 |

三者均执行 32 条 MMAD，M 有效行 96、M/N/K tile=128/128/256、总 K=4096。
两种方向的指令窗口接近，不能据此把历史 Worker 均值 29.31 对 20.19 μs 的差距
归因于权重方向；也不能反推两种方向在真实多核、缓存状态下必然等效。

上游代码参考为 pypto-lib `216456332c2a74d89cca23b7824dab264ce34bff`。
`upstream_layout_probe.py` 只复现 O-A 的形状、算术顺序与读取方向，非上游全链重放，
更不等于配置缺失的历史泳道。没有修改生产权重方向，也没有增加 device 权重重排。

stage=4 的单卡完整区间 p50/p95=849.78/866.94 μs，未确认优于已保留版本
817.22/843.12 μs，已撤回。该轮没有交错重测 baseline，不把两者差值全部归因于候选。
补丁、20 次原始样本、配置及功能保护见
[候选记录](../../perf_qproj_upstream/rejected/oa_pipeline4/measurement.json)。

## 复现所需输入与范围

1. 从仓库根目录、既定环境运行 `python <本目录>/upstream_layout_probe.py` 可生成诊断核，
   输出位于 `build_output/oa_incore_experiment/upstream_layout_build/`；只 CPU 编译。
   当前方向使用已编译的 `build_output/_jit__decode_csa_tp1_layer_k_0ugv8n/ptoas/proj_a_mm.{cpp,pto}`；
   stage=4 的局部源文件已保存到 `build_output/oa_incore_experiment/pipeline4_source/ptoas/`。
   当前方向对应基线提交 `25227f9c`，stage=4 可由候选补丁重建。
2. 使用 pypto-lib `.claude/skills/incore-profiling/incore_profile.py`，先列函数，再仅选择
   `proj_a_mm`，`target=a2a3`、`dynamic-dim=384`，关闭自动 provisioning。
   CANN `/usr/local/Ascend/cann-9.0.0/set_env.sh`、PTOAS 0.66、PTO-ISA `327cd586`。
   自动选择的 camodel 是 **Ascend910B1**，架构 dav-c220；不是 A3 真机计时。
3. 独立 case 的 tensor 使用生成器合成输入；静态 `.pto` 分配仍是完整形状。
   按生成的 orchestration 参数顺序，把 `main.cpp` 的标量改为
   `v4=1, v5=96, v6=0, v7=0, v8=0, v9=0, v10=8`，再编译/采集。
   含义依次为行块数、有效 token 数、输入组行偏移、组号、输出组列偏移、block index、block count。
   这三个 case 的签名相同；更换生成源码后须重新核对参数顺序。
4. 生成器的旧兼容声明与当前 PTO-ISA 的 `MrgSortExecutedNumList` 重复，
   仅删除独立 case C++ 中的旧声明；没有修改 PyPTO、PTOAS、PTO-ISA 或生产 kernel。
5. 三份 `manifest_export.csv` 均为 `exported`，各有 1 个 Cube、2 个空 Vector core。
   将 manifest 的 **visualize_data_bin** 传给 `python -m pypto.tools.clean_sim_trace`，
   不传 CANN 9 的外层 `export_dir`。清理器每份跳过了 22 个无法重定位的同步 flag，
   保留原始 trace 和 API_INSTR 数据。

`summary.json` 保存精确路径、标量、实际 `alloc_tile` 地址高水位、指令数及分 pipe 数值。
窗口取清理后 Cube core 首条指令开始至末条指令结束；每个 pipe 跨通道求时间区间并集。
各 pipe 可重叠，不能相加。API_INSTR cycles 即使排除 SET/WAIT/BAR，仍可能有排队/重叠，
仅作指令执行诊断，不能求和作为墙钟或忙碌时间。本次没有做数值验收。
