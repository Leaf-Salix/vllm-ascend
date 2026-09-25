# SPDX-License-Identifier: Apache-2.0
"""Opt-in Native DeepSeek V4 model with a complete PTO CSA decode call."""

from types import MethodType

import torch
from vllm.logger import logger

from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service_config import validate_configuration


def csa_layer_forward(layer, positions, hidden_states, residual, llama_4_scaling=None):
    """接管 decoder layer 的 attention 半边，FFN 半边原样留给 Native。

    PTO kernel 现在是整层入口（mHC pre + input_layernorm + attention + mHC post，
    见 `decode_csa._decode_csa_tp1_layer`），所以替换点从 `self_attn.forward` 上移到
    这里：`hidden_states` 进出都是层间的 mHC 残差流 [T, HC_MULT, D]。`self_attn`
    本身保持 Native 不动，custom op 里回退时照常调用它。

    The Native DSA wrapper owns the prefix and all six cache tensors. Keep
    positions explicit in the custom-op ABI, including during torch.compile.
    """
    attn_out = torch.empty_like(hidden_states)
    torch.ops.vllm.dsv4_csa_forward(
        hidden_states, positions, attn_out, layer.self_attn.dsa_attn.prefix
    )
    hidden_states = attn_out

    # FFN 半边：与 release 的 DeepseekV4DecoderLayer.forward 后半段逐行一致。
    residual = hidden_states.clone()
    hidden_states, post, comb = layer.hc_pre(
        hidden_states, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base
    )
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = layer.hc_post(hidden_states, residual, post, comb)
    return hidden_states, residual


def install_csa_forward(layer):
    import vllm_ascend.ops.dsv4_csa  # noqa: F401

    attention = layer.self_attn
    if attention.compress_ratio == 4:
        attention.dsa_attn._pto_csa_runtime = None
        # custom op 回退时要拿 mHC 的门控权重与 input_layernorm，它们挂在 layer 上。
        # 必须用 tuple 包一层：直接赋一个 nn.Module 会被 nn.Module.__setattr__ 登记成
        # dsa_attn 的子模块，于是 layer -> self_attn -> dsa_attn -> layer 成环，
        # model.eval() 里 module.train() 的递归遍历会直接栈溢出。
        attention.dsa_attn._pto_csa_layer = (layer,)
        layer.forward = MethodType(csa_layer_forward, layer)


def prepare_csa_model(model):
    # The opt-in runner hook runs after Native per-layer quant finalization.
    # No PyPTO initialization or NPU allocation occurs during model inspection.
    import importlib

    import pypto.torch
    from vllm.config import get_current_vllm_config
    from vllm_ascend.ops.pypto.variant import selected_variant, variant_package

    # 两套 CSA 算子并存，由 PTO_CSA_VARIANT 选择，默认性能版。只有算子与其适配层
    # 按版本取；service_config 的档位与图重放闸门两套共用一份（性能版里是重导出），
    # 因为 model_runner_v1.py 直接从精度版导入那些闸门，各留一份就会在判据上分叉。
    package = variant_package()
    CSAOperators = importlib.import_module(f"{package}.native_adapter").CSAOperators
    CSAServiceRuntime = importlib.import_module(f"{package}.service").CSAServiceRuntime

    pypto.torch.init(device=torch.npu.current_device(), platform="a2a3", runtime="tensormap_and_ringbuffer")
    operators = CSAOperators.register()
    max_num_seqs = get_current_vllm_config().scheduler_config.max_num_seqs
    count = 0
    for layer in model.model.layers:
        attention = layer.self_attn
        if attention.compress_ratio == 4:
            attention.dsa_attn._pto_csa_runtime = CSAServiceRuntime(
                attention, operators, max_num_seqs, layer
            )
            count += 1
    if not count:
        raise ValueError("No target C4 attention layers found for PTO CSA")
    logger.info("Prepared PTO CSA (%s variant) for %d target C4 layers; "
                "prefill and unsupported decode batches use Native", selected_variant(), count)


class PyptoCSADeepseekV4ForCausalLM(AscendDeepseekV4ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        validate_configuration(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.model.layers:
            install_csa_forward(layer)

    def process_weights_after_loading(self):
        prepare_csa_model(self)
