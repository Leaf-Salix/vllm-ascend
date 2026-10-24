# SPDX-License-Identifier: Apache-2.0
"""HCA 单独及 CSA/HCA 联合的显式 PTO 模型入口。"""

from types import MethodType

import torch
from vllm.logger import logger

from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM
from vllm_ascend.models.pypto_deepseek_v4 import PyptoCSADeepseekV4ForCausalLM, init_pto_runtime
from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.service_config import validate_configuration


def hca_layer_forward(layer, positions, hidden_states, residual, llama_4_scaling=None):
    output = torch.empty_like(hidden_states)
    torch.ops.vllm.dsv4_hca_forward(hidden_states, positions, output, layer.self_attn.dsa_attn.prefix)
    # attention 半层完成后，按 Native 的顺序执行 FFN 半层。
    residual = output.clone()
    hidden_states, post, comb = layer.hc_pre(output, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base)
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = layer.hc_post(hidden_states, residual, post, comb)
    return hidden_states, residual


def install_hca_forward(layer):
    import vllm_ascend.ops.dsv4_hca  # noqa: F401

    if layer.self_attn.compress_ratio == 128:
        wrapper = layer.self_attn.dsa_attn
        wrapper._pto_hca_runtime = None
        # tuple 避免将父 layer 再登记成 wrapper 的子模块而造成循环。
        wrapper._pto_hca_layer = (layer,)
        layer.forward = MethodType(hca_layer_forward, layer)


class PyptoHCADeepseekV4ForCausalLM(AscendDeepseekV4ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        validate_configuration(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.model.layers:
            install_hca_forward(layer)

    def process_weights_after_loading(self):
        init_pto_runtime()
        prepare_hca_model(self)


def prepare_hca_model(model):
    """运行时初始化后，为全部 C128 层绑定 HCA 算子。"""
    from vllm.config import get_current_vllm_config

    from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.native_adapter import HCAOperators
    from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.service import HCAServiceRuntime

    operators = HCAOperators.register()
    capacity = get_current_vllm_config().scheduler_config.max_num_seqs
    count = 0
    for layer in model.model.layers:
        if layer.self_attn.compress_ratio == 128:
            layer.self_attn.dsa_attn._pto_hca_runtime = HCAServiceRuntime(layer.self_attn, operators, capacity, layer)
            count += 1
    if not count:
        raise ValueError("模型中未找到 C128 HCA 层")
    logger.info("PTO HCA 已绑定 %d 个 C128 层，TP1/S6，公共计算使用 CSA 性能版", count)


class PyptoCSAHCADeepseekV4ForCausalLM(PyptoCSADeepseekV4ForCausalLM):
    """C4 和 C128 同时使用 PTO，SWA 与 FFN 保持 Native。"""

    def __init__(self, *, vllm_config, prefix=""):
        validate_configuration(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.model.layers:
            install_hca_forward(layer)

    def process_weights_after_loading(self):
        # CSA 初始化一次公共运行时；HCA 随后只注册算子并绑定各层。
        super().process_weights_after_loading()
        prepare_hca_model(self)


class PyptoHCATNDDeepseekV4ForCausalLM(PyptoHCADeepseekV4ForCausalLM):
    """显式HCA TND模型；仅C128使用新MPMD，C4/SWA/FFN保持Native。"""
    def process_weights_after_loading(self):
        from pypto import passes
        from pypto.runtime import CompileOptions
        from vllm.config import get_current_vllm_config
        from vllm_ascend.ops.pypto.deepseek_v4_flash_hca_tnd.native_adapter import HCAOperators
        from vllm_ascend.ops.pypto.deepseek_v4_flash_hca_tnd.graph_service import HCATNDGraphServiceRuntime

        init_pto_runtime()
        config = get_current_vllm_config()
        operators = HCAOperators.register(config=CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO))
        count = 0
        for layer in self.model.layers:
            if layer.self_attn.compress_ratio == 128:
                layer.self_attn.dsa_attn._pto_hca_runtime = HCATNDGraphServiceRuntime(
                    layer.self_attn, operators, config.scheduler_config.max_num_seqs, layer,
                    config.scheduler_config.max_num_batched_tokens,
                )
                count += 1
        if not count:
            raise ValueError("模型中未找到C128 HCA层")
        # 捕获时使用服务的历史容量，避免在短dummy上冻结K128后真实长历史只能回退。
        self._pto_hca_tnd_capture_sequence_length = config.model_config.max_model_len
        logger.info("PTO_HCA_MULTITASK_TND layers=%d package=deepseek_v4_flash_hca_tnd", count)
