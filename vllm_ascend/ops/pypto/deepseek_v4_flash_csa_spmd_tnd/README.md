# CSA 单 SPMD 的原生 TND 服务候选

本包基于动态单 SPMD，补充 Native TND 绑定、服务选择、容量处理及图合同。
已同步另一会话 `a241fa61` 采用的B1动态半叶/UB数学，使用原生150tick等待
接口，并采用已验证的B1及长G6/T64～96的O_A/O_B互补分工。
未采用出现语义回归的G6/24叶方案。
性能对照使用独立冻结的最新BSH源码，原包 `deepseek_v4_flash_csa_spmd` 保留。

当前实现覆盖 `HC_pre → input RMSNorm → CSA → HC_post`。
整链输入为 Native `hidden[T_capacity,4,4096]`；其中4是mHC路数。
活动请求的query长度可以在1～6之间逐请求变化，由device上的
`query_start_loc[B_capacity+1]` 和 `seq_lens[B_capacity]` 决定。
token容量、有效token数和请求槽数分别处理。

## 使用入口

在现有PTO CSA服务配置中选择：

```bash
export PTO_CSA_VARIANT=spmd_tnd
```

模型架构继续使用 `PyptoCSADeepseekV4ForCausalLM`。当前要求单卡TP1/PP1、
DeepSeek V4 Flash、BF16激活、Native W8A8、NZ2权重、32-token ND分页KV，
以及 `tensormap_and_ringbuffer` 运行时。DSpark上限保持5个投机token，
本轮实际query无需全部达到6。

`prepare_csa_model` 为每个C4层准备新 `CSAServiceRuntime`。层前向经
`torch.ops.vllm.dsv4_csa_forward` 进入服务，复用Native的cache connector、
页表、slots及compact metadata生产者，再直接提交单SPMD。

## 支持合同

- `native_adapter.py` 只建立同存储别名。Native按输入容量初始化的RoPE/slots
  可能返回较短decode视图；`token_capacity_view` 检查实际存储容量后恢复容量
  描述符，不复制、不回读Q/L。
- `binding.py` 按Tensor描述符及策略准备工作区。捕获前可以重新绑定相同描述符
  的输入地址；工作区与信号代际不重新清零。一个准备对象不能供并发流同时使用。
- 服务策略的Host长度上界当前为 **8192～1048576，包含本轮query**。
  这是batch的最大未压缩长度；同批较短请求仍由各自device Q/L限定。
  更短的整个batch及范围外请求回退Native。短历史Vector Score没有在本轮新增。
- 空token或全dummy调用不启动SPMD、不更新cache/state。dummy输出不作为有效
  token使用；有效写入范围和整个cache/state分配都参与验证。
- prefill、混合prefill/decode、draft侧、HCA组合模型、新并行方式、
  ACLNN/SuperKernel不在本轮范围。

实际设备已验证范围、尚待完成的边界和性能见独立
[实现记录](../DSV4_CSA_SPMD_TND_IMPLEMENTATION_LOG.md)，不能仅凭上述策略上限
认为所有长度与形状都已逐项测过。

## 图与回退

配置支持eager及框架的 `FULL_DECODE_ONLY`。框架保留其原有图档位选择；
本实现不把混长输入伪标为uniform。没有合适图时，支持范围内仍可eager提交
单SPMD。

`ACLGraphWrapper` 在重放前检查捕获合同，包括根Tensor地址/描述符、资源策略，
以及Native compact metadata生产者的输入和Host标量。合同失配时重新进行服务
分派；捕获为Native/空路径的图也不会直接用于新的SPMD工作量。
范围外真实请求在runner的图padding之前回退eager，避免将dummy Q端点交给
只处理有效query行的Native eager路径。

单层custom-op和独立SPMD图的设备证据，与完整模型图/服务验收分别记录。

## 验证口径

用户于2026-10-21明确要求：**原有Native数值差异单列，本轮聚焦TND正确性与
无新增回归**。既有阈值和Native差异保留；新旧单SPMD使用相同真实输入比较。
Native TND、独立C4整数几何、页写入范围、保护区和连续状态共同检查TND语义。
固定S6多task仅用于其支持的S6性能对照，不作为混长golden。

可复现脚本及任务清单位于：

```text
tests/pypto_test/csa_spmd_tnd_20261020/
  GOAL.txt / GOAL_UPDATES.md
  baseline_manifest.json / runtime_manifest.json
  submit_probe.py / run_probe.sh
  run_probe.py / service_probe.py / metadata_matrix.py
  jobs.json
  runs/                 # 原始证据，git忽略
```

提交器固定launcher；每个设备任务保存实际算子和fixture源码快照及SHA256。
八档性能使用同卡、同进程、同输入的旧SPMD/新TND配对，计完整L1首尾跨度：
128K/B1、B4、B8、B16、B24、B28；8K/B16、B32。B1是用户新增的场景，
允许根据同轮实测单独分档，若采用则另验B1的S1～S6。
旧工具输出中的30%目标字段不作为本轮门槛。

用户随后新增性能目标：以另一CSA会话最新已采用的单SPMD BSH性能为基线，
八档`T_TND/T_BSH`的等权几何平均不超过0.90，并逐档报告。基线源码和工具链
当前冻结为`a241fa61`，记录在
`tests/pypto_test/csa_spmd_tnd_20261020/bsh_baseline_a241fa61_short_typefix_manifest.json`。
另保留未经修复的a241fa61快照。两侧仅将B1专用临时张量明确为4×1024，
修复8K未启用分支仍被解析时的形状错误；六个128K档位的生成代码完全不变。
该基线包含已采用的B1动态半叶/UB优化及150tick等待；此前基线和raw TWAIT
测量保留版本标记，不能用于宣称达到新目标。原生150tick等待的依赖补丁和
隔离构建指纹位于同目录`dependency_patches/`及`native_toolchain_manifest.json`。
上述B1数学和原生等待已进入本包。输出互补分工仅在既有B1半叶域或长G6
且T容量64～96时开启：同24个物理AIC，O_A较重的前16核各承担2项O_B，
后8核各承担4项；不改变数学、量化写者或最终发布。B1 S1～S6及长G6
T65/B容量16的混长、请求变化、连续状态和同址图均已通过。
当前完整八档的等权几何平均为0.991928，耗时降低约0.81%，10%目标尚未达到；
128K/B1降低5.29%，128K/B16降低1.42%，8K/B16回退3.01%（新分工关闭）。
逐档结果、全部窗口和B1/B16生成代码及Host绑定等价复用证据见实现记录及
`cohort_output_balance_v4_result.json`。其他细分档候选保留在独立`runs/`目录，
不拼接不同试验的最快值。

原生等待依赖本轮隔离构建的PyPTO/PTOAS扩展，普通旧安装没有
`signal_wait(..., backoff_ticks=150)`能力。复现时使用同目录记录的源码基线、
依赖补丁和构建产物指纹；每个worktree从自身源码构建，不能直接复制其他
worktree的共享库。生产服务不安装测试用的C++等待改写回调。
