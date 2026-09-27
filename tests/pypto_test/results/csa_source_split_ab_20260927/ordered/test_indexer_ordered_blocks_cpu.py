# SPDX-License-Identifier: Apache-2.0
"""PTO-only page ordering: real allocator, no device initialization."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.platforms import current_platform

current_platform.pre_register_and_update()
from vllm.v1.core.block_pool import BlockPool  # noqa: E402

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec  # noqa: E402
from vllm_ascend.core.single_type_kv_cache_manager import CompressAttentionManager  # noqa: E402
from vllm_ascend.worker import model_runner_v1 as runner_module  # noqa: E402


def indexer_spec(ordered=False):
    return AscendMLAAttentionSpec(block_size=32, num_kv_heads=1, head_size=128,
                                dtype=torch.int8, scale_dim=1, scale_dtype=torch.float16,
                                compress_ratio=4, model_version="deepseek_v4",
                                prefer_contiguous_blocks=ordered)


def test_model_selection_and_merge(monkeypatch):
    spec = indexer_spec()
    layer = SimpleNamespace(get_kv_cache_spec=lambda _: spec)
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[])))
    runner = SimpleNamespace(use_compress=True, vllm_config=config)
    monkeypatch.setattr(runner_module, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(runner_module, "get_layers_from_vllm_config", lambda *_: {"indexer": layer})
    monkeypatch.setattr(runner_module, "get_ascend_device_type", lambda: runner_module.AscendDeviceType.A3)
    for architecture, variant, ordered in (
            ("AscendDeepseekV4ForCausalLM", "performance", False),
            ("PyptoCSADeepseekV4ForCausalLM", "precision", False),
            ("PyptoCSADeepseekV4ForCausalLM", "performance", True)):
        config.model_config.hf_config.architectures = [architecture]
        monkeypatch.setenv("PTO_CSA_VARIANT", variant)
        actual = runner_module.NPUModelRunner.get_kv_cache_spec(runner)["indexer"]
        assert actual == replace(spec, prefer_contiguous_blocks=ordered)
        assert AscendMLAAttentionSpec.merge([actual, actual]).prefer_contiguous_blocks == ordered
    with pytest.raises(AssertionError):
        AscendMLAAttentionSpec.merge([spec, replace(spec, prefer_contiguous_blocks=True)])
    assert not torch.npu.is_initialized()


@pytest.mark.parametrize("ordered", [False, True])
def test_reuse_and_incremental_append(ordered):
    pool = BlockPool(65, False, 128)
    manager = CompressAttentionManager(indexer_spec(ordered), pool, enable_caching=False,
                                      kv_cache_group_id=0, scheduler_block_size=128)
    first_ids = []
    for phase in range(3):
        request = str(phase)
        manager.add_local_computed_blocks(request, [], 0, 2048)
        manager.allocate_external_computed_blocks(request, 0, 2048)
        before = list(manager.req_to_blocks[request])
        ids = [b.block_id for b in before]
        first_ids.append(ids[0])
        if ordered:
            assert ids == sorted(ids)
        assert len(ids) == len(set(ids)) == 16
        if phase == 0:
            extra = manager.allocate_new_blocks(request, 2560, 2560)
            assert manager.req_to_blocks[request][:16] == before
            assert manager.req_to_blocks[request][16:] == extra
            assert len(extra) == 4
        manager.free(request)
        assert pool.get_num_free_blocks() == 64
    assert first_ids[0] == 1
    assert first_ids[1] == (5 if ordered else 20)
    assert not torch.npu.is_initialized()


def test_shared_prefix_stays_in_logical_order():
    pool = BlockPool(65, True, 128)
    manager = CompressAttentionManager(indexer_spec(True), pool, enable_caching=True,
                                      kv_cache_group_id=0, scheduler_block_size=128)
    owned = pool.get_new_blocks(4)
    prefix = [owned[2], owned[0]]
    manager.add_local_computed_blocks("shared", prefix, 256, 1024)
    manager.allocate_external_computed_blocks("shared", 256, 1024)
    assert manager.req_to_blocks["shared"][:2] == prefix
    assert [b.ref_cnt for b in prefix] == [2, 2]
    assert len(manager.req_to_blocks["shared"]) == 10
    manager.free("shared")
    assert [b.ref_cnt for b in prefix] == [1, 1]
    pool.free_blocks(owned)
    assert pool.get_num_free_blocks() == 64
    assert not torch.npu.is_initialized()
