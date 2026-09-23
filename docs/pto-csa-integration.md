# v0.25.1rc1：原 CSA kernel 的原生接口接入

## 基线

官方 vLLM-Ascend `v0.25.1rc1`（`9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`）定义接口契约。
CSA kernel 完整保留 Leaf `8990a7d8e`，来源是 sunkaixuan 分支与 PR #5；本次不使用 nalinaly kernel。
环境为 vLLM 0.25.1、CANN 9.0.1、Torch 2.10.0、Torch-NPU 2.10.0.post2。

## 接入及必要的内部绑定

`VLLM_ASCEND_PYPTO_DSV4_CSA=1` 时 backend 选择继承原生实现的 `PyptoDSAImpl`；默认关闭，CP backend 优先。
`forward` 签名、默认参数和返回类型均与 `AscendDSAImpl.forward` 一致，原地写入并返回原 output。
外部 `dsa_forward` 保持官方代码；没有额外调用参数或独立 substitute 接入层。

| 对照项 | 本次处理 | 保留的限制 |
| --- | --- | --- |
| metadata | 接收官方原对象及原顺序：compressed、main state、inner state、index、SWA | 仅均匀请求；不重建 metadata 对象 |
| 请求边界 | 核对 CPU query offsets、token/request 数、positions 和 seq_lens 长度 | ragged 与不一致的 dummy descriptor 回原生 |
| block table | 直接使用原 tensor 的切片 | 要求连续布局，不复制重排 |
| RoPE | 引用官方持久完整 FP32 表 | 原 kernel 用绝对位置索引，未改成消费 compact RoPE |
| token_valid | 保留8990的 position、block table、seq_lens 计算 | 未宣称直接消费官方 slot_mapping |
| KV/cache | 原6项tuple及共享allocation；零拷贝物理页view | 128-token KV页、8-row逻辑state页；其他规格回原生 |
| 权重 | 在原生 post-load 后进行8990原有准备 | 保留已有反量化/量化处理，不宣称与原生数学完全一致 |
| kernel | 保留40 tensor参数及计算实现 | 无52参数、32页kernel迁移 |
| 生命周期 | wait、提交kernel、通知cache write、save；启动错误直接抛出 | fused kernel没有原生prolog/attention之间的独立通知边界 |

本次不修改 allocator、metadata builder、调度器、HC、RMSNorm 或 MoE。
内部 `_kernel_args` 只是原 kernel 必需的参数绑定，仍有 token mask 计算和view构造；不能把本次描述成消除了全部数据适配或已证明提速。

## 支持与回退

支持范围为 A3、TP1、C4、均匀 S1；DSpark 出5验6配置下允许均匀 S1～S6，B不超过64。
prefill、profiling、gather、CP、LoRA、KV transfer、特殊 output projection TP、index cache、非因果draft窗口及不支持的cache布局交给原生实现。
保留原 kernel 的 capture/replay 路径；首次权重准备和算子注册必须在 capture 前完成。静态padding描述不能满足检查时回原生。

## 验证口径

CPU契约测试位于 `tests/pto_attn/test_decode_contract_cpu.py`。
硬件对照需使用相同128页、相同原kernel、相同输入与权重，并区分 eager、独立capture/replay和整模型graph。
实际命中记录为 `[pto-native-csa] ... seq=...`；原生回退不能算CSA通过。

8990的单层精度此前未通过。保留其计算路径不等于修复精度，不放宽阈值，也不以生成成功替代精度验收。
上一版52参数kernel的精度和整模型成绩不适用于此实现。
