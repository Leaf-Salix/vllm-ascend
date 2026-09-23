# v0.25.1rc1：PyPTO DSV4 CSA 接入

## 来源与目标

- 基线：[v0.25.1rc1](https://github.com/vllm-project/vllm-ascend/tree/v0.25.1rc1)，提交 `9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`。
- kernel 与原生页适配来自 [CSA 分支](https://github.com/sunkaixuan2018/vllm-ascend/tree/feat/csa-attn-cut-20260920) 和 [PR #5](https://github.com/sunkaixuan2018/vllm-ascend/pull/5) 的累计提交 `721209207edcc4ad8de2cb174859c5411067073b`，包含前序 PR #3/#4。
- 目标组合：A3、CANN 9.0.1、vLLM 0.25.1、Python 3.12、Torch 2.10.0、TorchNPU 2.10.0.post2。此处是验证目标，不能据此认为完整组合已经验收。

## 启用与原生接口

设置 `VLLM_ASCEND_PYPTO_DSV4_CSA=1` 启用；默认 `0` 使用原生 attention。

入口位于 `dsa_forward` 构建原生 KV tuple 之后、调用 `impl.forward` 之前。适配层接收原生 hidden states、六项 KV cache、五组 metadata 和调用方 output；成功时直接写入该 output，保持 `dsa_forward` 返回 `None`。不改变 vLLM 模型接口、KV cache 分配或原生 attention 函数签名。

本次只替换 attention：HC-pre、输入 RMSNorm、HC-post、MoE 仍由原模型执行。profiling 的 metadata 为 `None` 时保留原生调用，以维持原有分布式通信语义。

当前 serving 接入范围：A3、TP1、ratio4，支持普通单 token decode 和 DSpark 出5验6的 target verification。DSpark 读取原生 CPU query offsets，只有每请求均匀 S1～S6 的调用进入 CSA，不依赖 `PTO_ATTN_SEQ` 强制解释行数。混合 query 长度、非因果 draft 窗口、其他 speculative 方法、KV transfer、CP、特殊 output projection TP 或 index cache 在运行 kernel 前回退原生。DP/EP 由原生框架处理。

S6 直接绑定原生绝对位置和分页缓存；被拒绝 token 的未来槽位由后续位置覆盖，逐 query 因果长度限制可见范围，适配层不自行提交或回滚 speculative 状态。该语义需通过多步接受/拒绝测试验收。`[pto-attn-ran]` 包含 `seq`，真实 DSpark 验收必须观察 `seq=6`，不能仅凭生成成功判断替换命中。

kernel 开始执行后发生错误会向上传播，不在可能已修改 KV cache 后再执行原生 attention。

## 数据绑定

- 沿用 PR #5 的 40 参数 attention-only ABI，直接绑定真实 token 和调用方输出。
- 主 compressor state 与 compressed KV 共享原生物理分配；raw KV 独立。
- index state、INT8 index key 和 FP16 scale 使用同一原生页的不同 view。
- RoPE 读取此标签的 `_ROPE_STATE.full_rope_cache`，保持常驻 FP32 全表 view；旧分支的 `static_cache` 字段不适用于此标签。
- 不引入私有分页 KV cache，也不复制 main/vLLM 0.29 的 KV 分配修补。
- 新 kernel/ABI 必须使用新编译缓存，不能复用旧的 46 参数版本。

## 验证与限制

回归文件：`tests/pto_attn/test_decode_contract_cpu.py`、独立脚本 `check_pto_attn_cpu.py`、`check_build_args_cpu.py` 和 `tests/ut/ops/test_dsa_pto_dispatch.py`。分别检查接入条件、共享页布局、参数绑定、调用方输出与原生回退契约。

完整验证需要在同一环境完成原生与 CSA 的真实模型短生成，并确认 `[pto-attn-ran]`、正常进程退出及设备释放。连续 decode、ACLGraph capture/replay、单层数值对拍分别验收，不能互相替代。

PR #5 历史单层测试存在 native/PTO 数值差异，不能把历史 kernel golden 通过或本次生成成功称为精度通过。W8A8 权重的适配和原生动态激活量化也需单独对拍。遗留 `compare_once` 不接入 serving；精度测试应从相同但独立的 cache 初态分别执行两条路径。
