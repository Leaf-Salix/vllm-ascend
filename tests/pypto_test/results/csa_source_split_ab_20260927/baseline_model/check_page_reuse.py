"""CPU-only allocator probe; not a reconstruction of the real model page tables."""

import json
from pathlib import Path

import torch
from vllm.platforms import current_platform

current_platform.pre_register_and_update()
from vllm.v1.core.block_pool import BlockPool  # noqa: E402

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec  # noqa: E402
from vllm_ascend.core.single_type_kv_cache_manager import CompressAttentionManager  # noqa: E402


def main():
    pool = BlockPool(2049, False, 128)
    spec = AscendMLAAttentionSpec(block_size=32, num_kv_heads=1, head_size=128,
                                dtype=torch.int8, scale_dim=1, scale_dtype=torch.float16,
                                compress_ratio=4, model_version="deepseek_v4")
    manager = CompressAttentionManager(spec, pool, enable_caching=False,
                                      kv_cache_group_id=0, scheduler_block_size=128)
    result = {"scope": "CPU真实allocator，单组单请求1024页；不能冒充模型真实页分布", "rounds": []}
    for phase in ("warmup", "steady", "profile"):
        manager.add_local_computed_blocks(phase, [], 0, 131072)
        manager.allocate_external_computed_blocks(phase, 0, 131072)
        ids = [b.block_id for b in manager.req_to_blocks[phase]]
        panels = [ids[i:i + 4] for i in range(0, len(ids), 4)]
        result["rounds"].append({"phase": phase, "first_pages": ids[:8], "panels": len(panels),
                                 "ascending": sum(all(b == a + 1 for a, b in zip(p, p[1:])) for p in panels),
                                 "descending": sum(all(b == a - 1 for a, b in zip(p, p[1:])) for p in panels)})
        manager.free(phase)
        assert pool.get_num_free_blocks() == 2048
    assert not torch.npu.is_initialized()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
