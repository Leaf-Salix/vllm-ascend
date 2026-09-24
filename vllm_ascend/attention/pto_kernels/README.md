# 实验性 CSA 原生 metadata 接入

此实现集成在 vLLM-Ascend main 分支上，kernel 沿用 PR5 路线并经过本地适配。
入口位于 `AscendDSAImpl.forward`，通过已有的 `VLLM_ASCEND_PYPTO_DSV4_CSA`
开关启用，默认关闭。保留原生 connector 生命周期及不支持配置的回退。

当前支持 TP/CP=1、C4、固定 S6、B≤64、128槽 cache 页以及指定原生权重 dtype。
50个 tensor 参数直接携带 slot mapping、compact RoPE 和 start_pos。
Python 侧不再构造 token_valid，但 kernel 内仍有有效性掩码和必要的格式转换。

## 验证与限制

25项CPU合约测试通过。CANN9.2 beta2、Torch-NPU2.10.0.post4、vLLM0.29，
PyPTO5495749（包含PR2867）、Simpler32dff95环境完成单卡真实C4层权重测试。
测试使用合成hidden/history、B4/S6，固定步graph replay，非整模型生成。
现有实验环境使用退出兼容处理，不能据此认定任意同版本安装均可直接运行。

| 上下文 | 原生graph | 当前CSA graph | 输出relative L2 |
| --- | --- | --- | --- |
| 8K | 0.5842 ms | 0.5731 ms | 约1.83% |
| 128K | 0.7763 ms | 0.8159 ms | 约1.84% |

精度未通过 `rtol=atol=1e-2`，尚不能作为生产等价替换。
精简前后三阶段保存的CSA输出在这两组输入中逐元素相同。
最近一次消费端精简使128K相对中间版降低约4.3%，8K收益不明确；
不能将该单层结果解释成整模型加速。kernel源码保留原有许可证。
