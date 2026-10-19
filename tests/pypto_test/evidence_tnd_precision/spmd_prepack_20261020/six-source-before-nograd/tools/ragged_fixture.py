"""复用Native owner/builder生成真实TND元数据；不对每请求填充S6。"""
def make_ragged_fixture(config, attention, batch, history, seed, device, request_lengths, table_history=None, reverse_pages=False):
    import torch
    from dsv4_csa_single_layer import writable_bytes
    from dsv4_csa_native_case import allocate_native_cache, make_cache_groups

    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.worker.block_table import BlockTable

    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    assert len(request_lengths) == batch and all(1 <= q <= 6 for q in request_lengths)
    tokens = sum(request_lengths)
    bounds_cpu = torch.tensor([0, *torch.tensor(request_lengths).cumsum(0).tolist()], dtype=torch.int32)
    lengths_cpu = torch.tensor([history + q for q in request_lengths], dtype=torch.int32)
    bounds, lengths = bounds_cpu.to(device), lengths_cpu.to(device)
    positions = torch.tensor([history + j for q in request_lengths for j in range(q)], device=device, dtype=torch.int64)
    hidden = torch.randn((tokens, 4, 4096), device=device, dtype=torch.bfloat16)
    groups = make_cache_groups(config, device, attention)
    metadata, common_cache, prefill_cache, decode_cache = {}, {}, {}, {}
    for name, group in groups.items():
        spec = group["spec"]
        ratio = getattr(spec, "compress_ratio", 1)
        # table_history 固定页表宽度与分配页数，使不同历史长度的两份 fixture 形状相同，
        # 供同地址 metadata A→B→A 图重放；默认按本次历史计算，行为不变。
        width_history = history if table_history is None else table_history
        if width_history < history:
            raise ValueError("table_history 不能小于 history")
        columns = (width_history + 6 + spec.block_size * ratio - 1) // (spec.block_size * ratio) + 1
        # 非压缩历史仅需保留滑窗/近期 state；页表保留完整逻辑列并循环映射独占物理页。
        per_request = columns if name in ("compressed", "indexer") else 9
        if name == "state" and attention.compress_ratio == 128:
            per_request = 18
        pages = batch * per_request + 1
        group["layout"] = allocate_native_cache(group, pages, device)
        for i, view in enumerate(group["views"]):
            if view.dtype == torch.int8:
                view.random_(-64, 64)
            elif name == "indexer" and i == 1:
                view.fill_(0.01)
            else:
                view.normal_(0, 0.5)
        group["owner"].kv_cache = [group["views"]] if name == "indexer" else group["views"]
        table = BlockTable(spec.block_size, 40, columns, 256, True, device, num_speculative_tokens=5)
        for row in range(batch):
            owner = batch - 1 - row if reverse_pages else row
            table.add_row(
                [1 + owner * per_request + (per_request - 1 - col) % per_request for col in range(columns)], row
            )
        table.commit_block_table(batch)
        table.compute_slot_mapping(batch, bounds, positions // ratio)
        common = AscendCommonAttentionMetadata(
            query_start_loc=bounds,
            query_start_loc_cpu=bounds_cpu,
            seq_lens=lengths,
            _seq_lens_cpu=lengths_cpu,
            seq_lens_cpu=lengths_cpu,
            num_reqs=batch,
            num_actual_tokens=tokens,
            num_input_tokens=tokens,
            max_query_len=max(request_lengths),
            max_seq_len=history + max(request_lengths),
            block_table_tensor=table.block_table.gpu[:batch],
            slot_mapping=table.slot_mapping.gpu[:tokens],
            positions=positions,
            attn_state=AscendAttentionState.SpecDecoding,
            causal=True,
        )
        metadata[group["prefix"]] = group["builder"].build(
            0,
            common,
            num_reqs_actual=batch,
            block_size=spec.block_size,
            common_ratio_to_sas_metadata=common_cache,
            prefill_ratio_to_sas_metadata=prefill_cache,
            decode_ratio_to_sas_metadata=decode_cache,
        )
        group.update(table=table, common=common)
    compact = {
        name: attention.dsa_attn.dsa_attn.impl._compute_compressor_metadata(metadata[groups[name]["prefix"]].decode)
        for name in ("compressed", "indexer")
        if name in groups
    }
    for name, group in groups.items():
        slots = compact[name][2] if name in compact else metadata[group["prefix"]].decode.slot_mapping
        group["allowed"] = writable_bytes(group["allocation"], group["views"], slots)
        group["initial"] = group["allocation"].cpu()
    layer_name=groups['compressed']['prefix']
    main=metadata[groups['compressed']['prefix']].decode
    readonly = {"positions": positions, 'rope.cos':main.cos[layer_name], 'rope.sin':main.sin[layer_name]}
    for name, group in groups.items():
        req = metadata[group["prefix"]].decode
        for field in ("query_start_loc", "seq_lens", "block_table", "slot_mapping", "start_pos"):
            value = getattr(req, field)
            if value is not None:
                readonly[f"{name}.{field}"] = value
    readonly.update({f"{name}.compact_slots": value[2] for name, value in compact.items()})
    readonly = {name: (value, value.cpu()) for name, value in readonly.items()}
    return {
        "groups": groups,
        "metadata": metadata,
        "compact": compact,
        "positions": positions,
        "readonly": readonly,
        "hidden": hidden,
        "tokens": tokens,
    }

