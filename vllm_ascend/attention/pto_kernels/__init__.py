"""PR #5 的 DSV4 attention-only CSA kernel 及其直接依赖。

来源：sunkaixuan2018/vllm-ascend，提交
721209207edcc4ad8de2cb174859c5411067073b，包含前序 native-page 适配。
HC、外层 RMSNorm 和 MoE 由 vLLM-Ascend 原生路径执行。
本目录需要匹配的 PyPTO、Simpler 和 PTOAS，不能复用不同 ABI 的编译缓存。
"""
