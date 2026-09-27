"""CPU-only: actual runner layout selection and offset-preserving shared views."""

import json
import os
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

SOURCE = Path('/data/pyptouser/qinchuanyu/pto-eager/.cache/csa-source-split-2a740c1f')
sys.path.insert(0, str(SOURCE / 'tests/pypto_test'))
from dsv4_csa_env import activate  # noqa: E402

activate()
import torch  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402

current_platform.pre_register_and_update()
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.split_storage import indexer_storage  # noqa: E402

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec  # noqa: E402
from vllm_ascend.worker import model_runner_v1 as module  # noqa: E402


def main():
    pages, guard = 5, 128
    spec = AscendMLAAttentionSpec(block_size=32, num_kv_heads=1, head_size=128,
                                 dtype=torch.int8, scale_dim=1, scale_dtype=torch.float16,
                                 compress_ratio=4, model_version='deepseek_v4')
    backend = SimpleNamespace(get_kv_cache_shape=lambda p, b, h, d: (p, b, h, d))
    group = SimpleNamespace(backend=backend, kv_cache_spec=spec, layer_names=['indexer'])
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[])))
    runner = SimpleNamespace(use_compress=True, runner_only_attn_layers=set(),
                             attn_backend=backend, vllm_config=config,
                             _get_layer_kv_cache_specs=lambda _: {'indexer': spec},
                             _kv_cache_spec_attn_group_iterator=lambda: [group])
    runner._adjust_kv_layout = MethodType(module.NPUModelRunner._adjust_kv_layout, runner)
    old_type = module.get_ascend_device_type
    old_variant = os.environ.get('PTO_CSA_VARIANT')
    records = []
    try:
        module.get_ascend_device_type = lambda: module.AscendDeviceType.A3
        for architecture, variant, expected in (
                ('AscendDeepseekV4ForCausalLM', 'performance', (4160, 2080)),
                ('PyptoCSADeepseekV4ForCausalLM', 'precision', (4160, 2080)),
                ('PyptoCSADeepseekV4ForCausalLM', 'performance', (4096, 32))):
            config.model_config.hf_config.architectures = [architecture]
            os.environ['PTO_CSA_VARIANT'] = variant
            allocation = torch.full((pages * spec.page_size_bytes + guard * 2,), 37, dtype=torch.int8)
            raw = allocation[guard:-guard]
            views = module.NPUModelRunner._reshape_kv_cache_tensors(
                runner, SimpleNamespace(num_blocks=pages), {'indexer': raw})['indexer']
            assert tuple(t.stride(0) for t in views) == expected
            assert all(t.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr() for t in views)
            for value, tensor in enumerate(views):
                tensor.fill_(value + 1)
            assert bool((allocation[:guard] == 37).all() & (allocation[-guard:] == 37).all())
            if expected[0] == 4096:
                descriptor = indexer_storage(*views)
                assert descriptor.data_ptr() == raw.data_ptr() and descriptor.numel() == raw.numel()
                assert bool((views[0] == 1).all() & (views[1] == 2).all())
            else:
                try:
                    indexer_storage(*views)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Experimental ABI accepted the old layout')
            records.append({'architecture': architecture, 'variant': variant,
                            'page_strides': list(expected), 'status': 'PASS'})
        assert not torch.npu.is_initialized(), 'CPU layout check initialized NPU'
    finally:
        module.get_ascend_device_type = old_type
        if old_variant is None:
            os.environ.pop('PTO_CSA_VARIANT', None)
        else:
            os.environ['PTO_CSA_VARIANT'] = old_variant
    result = {'status': 'PASS', 'scope': 'CPU runner initializer and alias/guard checks', 'checks': records}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
