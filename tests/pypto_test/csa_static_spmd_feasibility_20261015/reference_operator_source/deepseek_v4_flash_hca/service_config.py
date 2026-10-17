# SPDX-License-Identifier: Apache-2.0
"""HCA 的模型选择；与 CSA 共用 TP1/S6 配置和图重放约束。"""

from ..deepseek_v4_flash_csa.service_config import validate_configuration as _validate_configuration

MODEL_ARCHITECTURE = "PyptoHCADeepseekV4ForCausalLM"


def is_hca_model(config):
    return any(name in (MODEL_ARCHITECTURE, "PyptoCSAHCADeepseekV4ForCausalLM")
               for name in getattr(config.model_config.hf_config, "architectures", ()))


def validate_configuration(config):
    from vllm_ascend import envs

    _validate_configuration(config)
    if envs.VLLM_ASCEND_PTO_CSA_ATOMIC_ADD != 0:
        raise ValueError("HCA 首版使用固定归约，请设置 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0")
