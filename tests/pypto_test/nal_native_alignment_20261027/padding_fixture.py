# SPDX-License-Identifier: Apache-2.0
"""只加载当前容量fixture；Native包和算子继续由公共bootstrap冻结。"""

import importlib.util
import sys
from pathlib import Path


def load_fixture_module():
    path = Path(__file__).resolve().parents[1] / "dsv4_csa_single_layer.py"
    spec = importlib.util.spec_from_file_location("padding_capacity_fixture", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def refresh_fixture(fixture, attention, queries, tables, fixture_module):
    """同一Native builder/输入存储重建派生metadata，不改decode token数。"""
    import torch

    active = len(queries)
    actual = sum(queries)
    capacity, requests = fixture["tokens"], fixture["request_capacity"]
    assert active <= requests and all(1 <= value <= 6 for value in queries)
    padded = list(queries) + [0] * (requests - active)
    remaining = capacity - actual
    assert remaining >= 0
    for index in range(active, requests):
        padded[index] = min(6, remaining)
        remaining -= padded[index]
    assert remaining == 0
    bounds = torch.tensor([0, *torch.tensor(padded).cumsum(0).tolist()], dtype=torch.int32)
    histories = fixture["history_lengths"][:active]
    lengths = torch.tensor([h + q for h, q in zip(histories, queries)] + [0] * (requests - active), dtype=torch.int32)
    positions = torch.cat([h + torch.arange(q) for h, q in zip(histories, queries)])
    # Tail positions/RoPE stay finite and stale on purpose: only Q/L/slots define validity.
    fixture["positions"][:actual].copy_(positions)
    metadata, common_cache, prefill_cache, decode_cache = {}, {}, {}, {}
    for name, group in fixture["groups"].items():
        common, table, spec = group["common"], group["table"], group["spec"]
        common.query_start_loc_cpu.copy_(bounds)
        common.query_start_loc.copy_(bounds)
        common._seq_lens_cpu.copy_(lengths)
        if common.seq_lens_cpu is not None:
            common.seq_lens_cpu.copy_(lengths)
        common.seq_lens.copy_(lengths)
        common.num_actual_tokens = actual
        common.max_query_len = max(padded)
        common.max_seq_len = int(lengths.max())
        common.block_table_tensor.copy_(tables[name])
        common.block_table_tensor[active:].zero_()
        ratio = getattr(spec, "compress_ratio", 1)
        table.compute_slot_mapping(requests, common.query_start_loc, fixture["positions"] // ratio)
        common.slot_mapping[actual:].fill_(-1)
        metadata[group["prefix"]] = group["builder"].build(
            0,
            common,
            num_reqs_actual=active,
            block_size=spec.block_size,
            common_ratio_to_sas_metadata=common_cache,
            prefill_ratio_to_sas_metadata=prefill_cache,
            decode_ratio_to_sas_metadata=decode_cache,
        )
    compact = {
        name: attention.dsa_attn.dsa_attn.impl._compute_compressor_metadata(
            metadata[fixture["groups"][name]["prefix"]].decode
        )
        for name in ("compressed", "indexer")
    }
    for name, group in fixture["groups"].items():
        slots = compact[name][2] if name in compact else metadata[group["prefix"]].decode.slot_mapping
        group["allowed"] = fixture_module.writable_bytes(group["allocation"], group["views"], slots)
    fixture.update(metadata=metadata, compact=compact, actual_tokens=actual, query_lengths=list(queries))
