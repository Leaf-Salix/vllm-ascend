# SPDX-License-Identifier: Apache-2.0
"""固定 TND 测试场景：Native 的单个失活尾部请求规则。

来自 84751e34 的单层 Native 构造工具，只调整尾部 padding 表达：
model_runner_v1._pad_query_start_loc_for_fia 的 mixed 分支保留真实 Q，
在末尾增加一个 dummy 请求，并将其 Q 终点置为 token 容量。dummy 的
query 区间允许超过六行，但 L=0、slots=-1；真实请求仍限定 S1～S6。
Native 的 max_query_len Host 描述符来自真实调度量，不包含这个 dummy。
所有生产元数据仍由 Native builder/compact producer 构造。
"""

from dsv4_csa_single_layer import writable_bytes


def make_fixture(
    config,
    attention,
    batch,
    history,
    seed,
    device,
    table_history=None,
    reverse_pages=False,
    *,
    query_tokens=6,
    indexer_page_pattern="original",
    query_lengths=None,
    history_lengths=None,
    token_capacity=None,
    request_capacity=None,
    page_owners=None,
    physical_request_capacity=None,
    allow_unassigned_tail=False,
    cache_template=None,
):
    """按实际本轮长度构造Native输入；Dspark配置上限与本轮有效长度分开。

    默认S6保持既有用例。S5用于置信度导致本轮长度缩短的验证，不把模型的
    num_speculative_tokens从5改为4。query_lengths/history_lengths允许按请求构造
    真正的TND前缀与位置，仍交给Native builder生成metadata，不补成伪S6。
    indexer_page_pattern只改变完整历史前缀的页顺序；须在Native metadata、
    compact slots及只读快照生成之前完成，避免测试输入的不同视图不一致。
    original保持所有旧调用的原输入；升序/碎片样例不并入原七档性能统计。
    """
    import torch
    from dsv4_csa_native_case import allocate_native_cache, make_cache_groups

    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.worker.block_table import BlockTable

    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    if type(query_tokens) is not int or not 1 <= query_tokens <= 6:
        raise ValueError("本scenario支持Dspark=5下的1..6个实际query token")
    queries = [query_tokens] * batch if query_lengths is None else list(query_lengths)
    histories = [history] * batch if history_lengths is None else list(history_lengths)
    if len(queries) != batch or any(type(value) is not int or not 1 <= value <= 6 for value in queries):
        raise ValueError("query_lengths须为每个请求提供1..6个实际query token")
    if len(histories) != batch or any(type(value) is not int or value < 0 for value in histories):
        raise ValueError("history_lengths须为每个请求提供非负历史长度")
    if batch <= 0:
        raise ValueError("本scenario构造非空请求；空调用走独立服务回退验证")
    query_tokens = max(queries)
    max_history = max(histories)
    if indexer_page_pattern not in ("original", "ascending_history", "fragmented_history"):
        raise ValueError("未知Indexer历史页排列")
    actual_tokens = sum(queries)
    tokens = actual_tokens if token_capacity is None else token_capacity
    requests = batch if request_capacity is None else request_capacity
    physical_requests = requests if physical_request_capacity is None else physical_request_capacity
    if type(tokens) is not int or tokens < actual_tokens or type(requests) is not int or requests < batch:
        raise ValueError("TND scenario容量不能小于有效token/请求数")
    if type(physical_requests) is not int or physical_requests < requests:
        raise ValueError("物理页请求容量须覆盖metadata请求容量")
    if page_owners is not None and (
        len(page_owners) != batch
        or len(set(page_owners)) != batch
        or any(type(owner) is not int or not 0 <= owner < physical_requests for owner in page_owners)
    ):
        raise ValueError("page_owners须为有效且互异的物理请求槽")
    padded_queries = queries + [0] * (requests - batch)
    remaining = tokens - actual_tokens
    for request in range(batch, requests):
        padded_queries[request] = remaining  # Native mixed padding: one inactive tail request
        remaining -= padded_queries[request]
    if remaining and not allow_unassigned_tail:
        raise ValueError("Native 尾部 padding 需要一个 dummy 请求槽")
    bounds_cpu = torch.zeros(requests + 1, dtype=torch.int32)
    bounds_cpu[1:] = torch.tensor(padded_queries, dtype=torch.int32).cumsum(0)
    lengths_cpu = torch.tensor(
        [h + q for h, q in zip(histories, queries)] + [0] * (requests - batch),
        dtype=torch.int32,
    )
    bounds, lengths = bounds_cpu.to(device), lengths_cpu.to(device)
    active_positions = torch.cat([h + torch.arange(q) for h, q in zip(histories, queries)])
    positions = torch.zeros(tokens, device=device, dtype=torch.int64)
    positions[:actual_tokens].copy_(active_positions)
    hidden = torch.randn((tokens, 4, 4096), device=device, dtype=torch.bfloat16)
    groups = make_cache_groups(config, device, attention)
    metadata, common_cache, prefill_cache, decode_cache = {}, {}, {}, {}
    for name, group in groups.items():
        spec = group["spec"]
        ratio = getattr(spec, "compress_ratio", 1)
        # table_history 固定页表宽度与分配页数，使不同历史长度的两份 scenario 形状相同，
        # 供同地址 metadata A→B→A 图重放；默认按本次历史计算，行为不变。
        width_history = max_history if table_history is None else table_history
        if width_history < max_history:
            raise ValueError("table_history 不能小于实际最大 history")
        width_query = 6 if table_history is not None else query_tokens
        columns = (width_history + width_query + spec.block_size * ratio - 1) // (spec.block_size * ratio) + 1
        # 非压缩历史仅需保留滑窗/近期 state；页表保留完整逻辑列并循环映射独占物理页。
        per_request = columns if name in ("compressed", "indexer") else 9
        if name == "state" and attention.compress_ratio == 128:
            per_request = 18
        pages = physical_requests * per_request + 1
        if cache_template is None:
            group["layout"] = allocate_native_cache(group, pages, device)
            for i, view in enumerate(group["views"]):
                if view.dtype == torch.int8:
                    view.random_(-64, 64)
                elif name == "indexer" and i == 1:
                    view.fill_(0.01)
                else:
                    view.normal_(0, 0.5)
        else:
            # 仅供同址功能验证构造新metadata；不重分配、清零或随机化持久cache。
            template = cache_template[name]
            if template["pages"] != pages or template["raw"].numel() != pages * spec.page_size_bytes:
                raise ValueError(f"{name}: 同址metadata场景改变了物理页容量")
            if template["allocation"].device != device:
                raise ValueError("同址cache须与场景位于同一设备")
            for field in ("layout", "allocation", "raw", "views", "page_size", "pages"):
                group[field] = template[field]
        group["owner"].kv_cache = [group["views"]] if name == "indexer" else group["views"]
        # 保留原七档的 40 请求 / 256 token 容量；较大的单层边界用例按实际
        # 请求数和 token 数扩容，避免 scenario 在进入算子前发生页表越界。
        table = BlockTable(
            spec.block_size,
            max(40, requests),
            columns,
            max(256, tokens),
            True,
            device,
            num_speculative_tokens=5,
        )
        for row in range(requests):
            if row >= batch:
                table.add_row([0] * columns, row)
                continue
            owner = page_owners[row] if page_owners is not None else batch - 1 - row if reverse_pages else row
            page_ids = [1 + owner * per_request + (per_request - 1 - col) % per_request for col in range(columns)]
            if name == "indexer" and indexer_page_pattern != "original":
                full_history_pages = histories[row] // (spec.block_size * ratio)
                if full_history_pages < 4:
                    raise ValueError("页序功能样例至少需要四个完整历史页")
                current_suffix = page_ids[full_history_pages:]
                if indexer_page_pattern == "ascending_history":
                    page_ids[:full_history_pages] = reversed(page_ids[:full_history_pages])
                else:
                    page_ids[1], page_ids[2] = page_ids[2], page_ids[1]
                assert page_ids[full_history_pages:] == current_suffix
            table.add_row(page_ids, row)
        table.commit_block_table(requests)
        table.compute_slot_mapping(requests, bounds, positions // ratio)
        if tokens > actual_tokens:
            table.slot_mapping.gpu[actual_tokens:tokens].fill_(-1)
        common = AscendCommonAttentionMetadata(
            query_start_loc=bounds,
            query_start_loc_cpu=bounds_cpu,
            seq_lens=lengths,
            _seq_lens_cpu=lengths_cpu,
            seq_lens_cpu=lengths_cpu,
            num_reqs=requests,
            num_actual_tokens=actual_tokens,
            num_input_tokens=tokens,
            max_query_len=max(queries),  # Native Host descriptor describes scheduled real requests
            max_seq_len=int(lengths_cpu.max()),
            block_table_tensor=table.block_table.gpu[:requests],
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
        group["initial"] = group["allocation"].cpu() if cache_template is None else cache_template[name]["initial"]
    readonly = {"positions": positions}
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
        "actual_tokens": actual_tokens,
        "request_capacity": requests,
        "query_tokens": query_tokens,
        "query_lengths": queries,
        "history_lengths": histories,
        "table_history": table_history,
    }
