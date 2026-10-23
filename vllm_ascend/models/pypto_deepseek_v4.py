# SPDX-License-Identifier: Apache-2.0
"""Opt-in Native DeepSeek V4 model with a complete PTO CSA decode call."""

from types import MethodType

import torch
from vllm.logger import logger

from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.service_config import validate_configuration


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
    torch.ops.vllm.dsv4_csa_forward(hidden_states, positions, attn_out, layer.self_attn.dsa_attn.prefix)
    hidden_states = attn_out

    # FFN 半边：与 release 的 DeepseekV4DecoderLayer.forward 后半段逐行一致。
    residual = hidden_states.clone()
    hidden_states, post, comb = layer.hc_pre(hidden_states, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base)
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


def init_pto_runtime():
    """CSA 与 HCA 共用同一个设备运行时和 arena 配置。"""
    import inspect

    import pypto.torch
    from vllm.config import get_current_vllm_config

    from vllm_ascend.ops.pypto.variant import csa_runtime, ring_sizing_kwargs

    # 运行时默认仍是 tensormap_and_ringbuffer。显存排查发现它在 init 期固定占用
    # 1.343 GiB 设备显存（与 kernel / batch / 层数无关），而 host_build_graph 只占
    # 0.598 GiB（仅默认 arena 初始化，不能当作完整 CSA 的运行占用）。
    # HBG kernel 的 GM heap 使用 ring_heap[0]；B4 CSA 已超过默认 256 MiB，
    # 实测需显式配置 PTO_CSA_RING_HEAP_MB=320。其他形状需按实际请求容量配置。
    # 见 tests/pypto_test/results/mem_128k_b24_20260928/ANALYSIS.md。
    # PyPTO 运行时 arena 的尺寸只能在 init 时给（之后 prepare_callable / launch
    # 都不再携带 CallConfig），默认那 4x256 MiB heap + 16384 深 task window 合计
    # 占 1.343 GiB 设备显存。取值来自 additional_config["pto_csa_ring_config"]，
    # 也可用 PTO_CSA_RING_* 环境变量临时覆盖；不设则沿用 Simpler 的编译期默认。
    ring_kwargs = ring_sizing_kwargs(get_current_vllm_config().additional_config)
    if ring_kwargs and "ring_heap" not in inspect.signature(pypto.torch.init).parameters:
        # 这条接口是 hw-native-sys/pypto#2940 才加的。没有它时静默忽略会让人以为
        # 显存已经省下来了，所以宁可在这里直接报错。
        raise ValueError(
            "当前 PyPTO 的 pypto.torch.init 不接受 ring 尺寸参数；"
            "请升级到含 kernel 模式 ring sizing 的版本，或去掉 pto_csa_ring_config "
            "与 PTO_CSA_RING_* 设置"
        )
    pypto.torch.init(
        device=torch.npu.current_device(),
        platform="a2a3",
        runtime=csa_runtime(),
        **ring_kwargs,
    )


def prepare_csa_model(model):
    # The opt-in runner hook runs after Native per-layer quant finalization.
    # No PyPTO initialization or NPU allocation occurs during model inspection.
    import importlib

    from vllm.config import get_current_vllm_config

    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.nz_mode import (
        root_weight_layouts,
        validate_weight_nz_mode,
    )
    from vllm_ascend.ops.pypto.variant import csa_runtime, selected_variant, variant_package

    # 默认保持多task；TND单SPMD使用独立Native直调入口和工作区管理。
    # service_config与model_runner的图合同必须与所选实现一致。
    package = variant_package()
    hbg = csa_runtime() == "host_build_graph"
    effective_mode = get_ascend_config().weight_nz_mode
    validate_weight_nz_mode(effective_mode)
    CSAServiceRuntime = importlib.import_module(f"{package}.service").CSAServiceRuntime
    if selected_variant() in ("spmd_tnd", "mpmd_tnd"):
        if hbg:
            raise ValueError("TND CSA使用tensormap_and_ringbuffer运行时")
        CSAOperators = None
        entry = importlib.import_module(f"{package}.entry")
        root_function = entry.decode_csa_spmd if selected_variant() == "spmd_tnd" else entry.decode_csa_tnd
    else:
        adapter = importlib.import_module(f"{package}.native_adapter")
        CSAOperators = adapter.HBGCSAOperators if hbg else adapter.CSAOperators
        root_function = importlib.import_module(f"{package}.decode_csa")._decode_csa_tp1_layer
    layouts = root_weight_layouts(root_function)
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.reduction import ATOMIC_ADD

    if selected_variant() == "spmd_tnd":
        logger.info("PTO_CSA_SINGLE_SPMD_TND package=%s atomic_add=%d", package, ATOMIC_ADD)
    elif selected_variant() == "mpmd_tnd":
        logger.info("PTO_CSA_MULTITASK_TND package=%s atomic_add=%d", package, ATOMIC_ADD)
    else:
        reduction = importlib.import_module(f"{package}.qkv_proj_rope")
        logger.info(
            "PTO_CSA_REDUCTION atomic_add=%d qr_split_k=%d kv_split_k=%d",
            ATOMIC_ADD,
            reduction.QR_OK,
            reduction.KV_OK,
        )

    init_pto_runtime()
    operators = CSAOperators.register() if CSAOperators is not None else None
    config = get_current_vllm_config()
    max_num_seqs = config.scheduler_config.max_num_seqs
    count = 0
    for layer in model.model.layers:
        attention = layer.self_attn
        if attention.compress_ratio == 4:
            attention.dsa_attn._pto_csa_runtime = CSAServiceRuntime(attention, operators, max_num_seqs, layer)
            count += 1
    if not count:
        raise ValueError("No target C4 attention layers found for PTO CSA")
    logger.info(
        "PTO_CSA_LAYOUT variant=%s effective_mode=%d root_layouts=%s nz_packed=%s",
        selected_variant(),
        effective_mode,
        layouts,
        [name for name, layout in layouts.items() if layout == "NZ"],
    )
    logger.info(
        "Prepared PTO CSA (%s variant) for %d target C4 layers; prefill and unsupported decode batches use Native",
        selected_variant(),
        count,
    )


class PyptoCSADeepseekV4ForCausalLM(AscendDeepseekV4ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        validate_configuration(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.model.layers:
            install_csa_forward(layer)

    def process_weights_after_loading(self):
        prepare_csa_model(self)
