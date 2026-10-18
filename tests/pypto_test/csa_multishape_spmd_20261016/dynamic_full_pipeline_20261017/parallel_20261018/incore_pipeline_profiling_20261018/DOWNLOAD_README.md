# CSA动态单SPMD：128K/B16/S6，全部72核incore流水

本目录是CANN模拟器的指令流水，来自有界轮询退避的诊断副本。
**不是当前正式算子的真机性能结果。** `get_sys_cnt()`在本次模型中恒为0，
因此仅在独立模拟副本给退避内循环增加128次读取上限；正式源码不变。
GM依赖条件、缓存发布、业务数据维护和FFTS协议保持。退避时间不同，
不能据此定量解释原版的等待时间或完整耗时。

这一限制也影响等待以外的流水：退避频率改变可能影响DCCI/GM访问、共享
访存竞争和阶段起点，继而改变计算/搬运重叠。128次上限没有按100tick
校准；数值一致不能证明机器指令或时序一致。图用于辅助结构分析与提出
候选，不能直接据此给正式版的性能瓶颈或优化收益定量下结论。

## 建议阅读顺序

优先打开`readable/index.html`（10.40MB）：离线总览，可筛选核组/角色和
缩放时间。完整阅读下载包为`CSA_128K_B16_S6_72cores_readable.zip`（19.54MB）。
`readable/groups/`中每份核组图6.82～9.43MB；`readable/cores/`中每份单核图
2.18～3.46MB，适合在Perfetto中查看具体指令。72核全部保留，轻量化仅移除
重复的长源码栈和寄存器参数，保留PC，原始大图继续保留。下面的大文件用于回查。

1. `CSA_128K_B16_S6_72cores_model_bounded.pipe_overview.json`：约32.5 MB的
   全核计算/搬运概览，可导入Perfetto；每核每pipe保留实际指令区间的并集。
2. `CSA_128K_B16_S6_72cores_model_bounded.clean.json`：详细指令与原版PyPTO
   清理工具重建的SET/WAIT_FLAG箭头。默认隐藏Scalar，完整信息仍在原始图。
3. `raw_simulator/trace.json`：约19.34 GB的原始全核图，包含Scalar和同步；
   `raw_simulator/core*`提供72条逐核图和指令CSV，适合分核查看。
4. `pipeline_findings.md`与`pipeline_role_summary.json`：首轮定位结果，
   以及全核指令数核对。不能将pipe时长相加当完整耗时。

共同时间轴保持原样，没有把每个核分别归零。普通GM notify/wait没有被
自动重建为跨核箭头；请结合原始Scalar事件、阶段协议和源码分析。

## 输入、工具链与验证

- 128K/B16/S6，T=96；正式W8A8第4层权重，历史数据沿用完整CSA测试fixture。
- 正式源：`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd`。
  详细工作区、132个指针与56个动态shape标量见`workload_bindings.json`。
- 24-block MIX，24 AIC+48 AIV，同一完整launch；没有把核独立执行后拼接。
- CANN 9.2.0-beta.2及同安装配套worker/injection；a2a3、Ascend910_9392、
  dav-c220、camodel dav_2201。无跨版本工具回退。
- `manifest_export.csv`为exported；`incore_coverage.json`检查72核覆盖，
  每个AIC都有MMAD、每个AIV都有非同步Vector指令。
- `replay_validation.json`：模拟副本77份完整可写storage、756895848字节
  与原真机after逐字节一致；包括输出、Top-K/score、cache/state和保护区。
- `device_replay_validation.json`：原独立二进制在真机的同项检查通过。
- `instr_metrics.json`与`pipeline_summary.csv`保留指令计数及周期数据；
  指令周期可重叠，不能直接求和解释为墙钟耗时。

源码、采集脚本、完整尝试及问题记录位于vllm-ascend仓库的
`tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/incore_pipeline_profiling_20261018/`。
本目录只包含轨迹与说明，不打包物理权重或输入快照；原始大文件只本地保留。

清理已完成：72核，1392602个事件；原工具有2460条SET/WAIT_FLAG连线未能
重建，原始数据完整保留。不将这些缺失的展示连线解释为算子缺失同步。
