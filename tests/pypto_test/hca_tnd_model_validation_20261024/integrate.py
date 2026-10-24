# SPDX-License-Identifier: Apache-2.0
"""仅修改本轮冻结框架；不覆盖CSA会话正在编辑的共享文件。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def integrate(root):
    def replace(path, old, new):
        file = root / path
        text = file.read_text()
        assert text.count(old) == 1, (path, old)
        file.write_text(text.replace(old, new))

    name = "PyptoHCATNDDeepseekV4ForCausalLM"
    replace("vllm_ascend/models/__init__.py", "def register_model():\n",
            'def register_model():\n    ModelRegistry.register_model(\n'
            f'        "{name}", "vllm_ascend.models.pypto_deepseek_v4_hca:{name}",\n    )\n')
    target = root / "vllm_ascend/models/pypto_deepseek_v4_hca.py"
    with target.open("a") as stream:
        stream.write('''

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
''')
    replace("vllm_ascend/ops/pypto/deepseek_v4_flash_hca/service_config.py",
            '(MODEL_ARCHITECTURE, "PyptoCSAHCADeepseekV4ForCausalLM")',
            f'(MODEL_ARCHITECTURE, "PyptoCSAHCADeepseekV4ForCausalLM", "{name}")')
    config_path = root / "vllm_ascend/ops/pypto/deepseek_v4_flash_hca/service_config.py"
    with config_path.open("a") as stream:
        stream.write(f'\n\ndef is_tnd_hca_model(config):\n    return "{name}" in getattr(config.model_config.hf_config, "architectures", ())\n')
    replace("vllm_ascend/ops/dsv4_hca.py", '    runtime = getattr(wrapper, "_pto_hca_runtime", None)\n',
            '    runtime = getattr(wrapper, "_pto_hca_runtime", None)\n'
            '    record_route = getattr(type(runtime), "record_graph_route", None)\n'
            '    if record_route is not None:\n        record_route(runtime, context, hidden, positions)\n'
            '    empty_call = getattr(type(runtime), "handle_empty_call", None)\n'
            '    if empty_call is not None and empty_call(runtime, context, hidden, positions, output):\n        return\n')
    replace("vllm_ascend/worker/model_runner_v1.py",
            'from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.service_config import is_hca_model',
            'from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.service_config import is_hca_model, is_tnd_hca_model')
    replace("vllm_ascend/worker/model_runner_v1.py", 'single_spmd_tnd=is_tnd_csa_model(self.vllm_config),',
            'single_spmd_tnd=(is_tnd_csa_model(self.vllm_config) or is_tnd_hca_model(self.vllm_config)),')
    replace("vllm_ascend/worker/model_runner_v1.py",
            '                    tnd_capture_length = getattr(self.get_model(), "_pto_csa_tnd_capture_sequence_length", None)',
            '                    tnd_capture_length = getattr(self.get_model(), "_pto_csa_tnd_capture_sequence_length", None)\n'
            '                    if tnd_capture_length is None:\n'
            '                        tnd_capture_length = getattr(self.get_model(), "_pto_hca_tnd_capture_sequence_length", None)')
    shutil.copy2(HERE / "candidates/graph_service.py",
                 root / "vllm_ascend/ops/pypto/deepseek_v4_flash_hca_tnd/graph_service.py")
