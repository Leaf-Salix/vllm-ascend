# Native attention 半层：实际模板编译对照（进行中）

目标是补齐“手工 NPUGraph 不等于模板 npugraph_ex/static kernel”这一基线缺口。
只使用 Native HC_pre + norm + CSA + HC_post，正式第 4 层权重、合成独立历史。
没有 MoE/EP16，不据此宣称完整部署模板的整模型性能。

`CompiledAttentionHalf` 通过 vLLM 的 `support_torch_compile` 路径编译，
保留 Native `dsa_forward` 边界；同进程先测手工图，再测编译半层。
必须确认 wrapper compiled、static_compile 实际返回 True、有安装包，
并结合生成图及 kernel/profile 核实覆盖范围。不能仅检查 additional_config。

先用 128K/B16 验证真实编译路径，再按受影响项推进 8K/B24。
每侧 5 次预热/20 次计时，profile 单独采集；mode2、det0、CANN9.2。
Native det0 不要求跨次浮点逐 bit 一致，仍检查状态、索引结构及保护区。
当前任务见 [task.txt](task.txt)，入口 [run.sh](run.sh)、[native_layer.py](native_layer.py)。
补充 vendor 与私有可写 OPP 来自 [Native 模板依赖实验](../csa_native_template_20260929/README.md)。
128K/B16 的 task_20260929_011152_128409915531 已完成 exit=0，auto/card0。
wrapper compiled、static_compile=True、一个安装包均确认；39 个静态编译算子描述覆盖
HC、QLI、Compressor、Sparse Attention、Q/O 投影等，独立 profile 完整执行这些节点。
进一步检查 38 份编译成功日志及已安装 binary manifest，确认关键计算算子有静态二进制；
CompressorMetadata 有描述但未列入静态二进制，不能宣称所有节点都被静态化。
20 次均值 1294.493→1238.764 μs（−4.305%），P95 1299.440→1245.240 μs。
compiled profile 42 个 kernel，manual 43 个，少了开头 residual clone 的 TensorMove；
不能把全部 55.729 μs 改善归因于这一个拷贝或声称全部核都获益。
数值、原始采样、实际编译产物列表及 profile 路径见 [RESULTS.md](RESULTS.md) 和 [evidence.json](evidence.json)。

8K/B24 经 [run_short.sh](run_short.sh) 完成，task_20260929_012814_198666224698 auto/card4，exit=0。
均值 1113.380→1041.954 μs，P95 1116.700→1046.240 μs；实际编译及安装检查通过。
编译后 Top-K 有 2 行仅顺序变化、2 行集合变化，共替换 4 个索引；无非法行。
输出 max_abs=0.015625，RMSE=0.000324394；这不是已完成数值或整网 token 验收。

两档共用私有可写 OPP，长档安装包在短档开始前仍存在。进一步读取实际安装 manifest，
两档各 38 个 CANN simplifiedKeyWithPlatform 选择键，交集为 0；没有这两档静态包的同键复用。
[选择键证据](static_selection_overlap.json)排除了此前提出的跨档匹配形状疑点，无需重跑设备测试。
这里比较的是手工调用与模板编译半层，包含图优化及静态 kernel 的综合变化，不单独归因于某个开关。
两份测试入口未经过整模型 Worker，CPU 绑核和共享专家重叠属于请求配置，
本 attention 半层不具备验证 MoE 多流或实际 Worker 绑核的条件，仍待整机阶段核实。
