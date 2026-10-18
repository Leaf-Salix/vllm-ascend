# SPDX-License-Identifier: Apache-2.0
"""保存原native_query_probe的纯Native入参绑定，不依赖其CLI目录或执行Native算子。"""

def bind_native_inputs(adapter, layer, fixture, output, torch):
    """机械绑定已有Native存储；不修改旧S6适配入口或构造伪S6请求。

    PreparedDynamicCSA沿用旧参数命名，但不需要旧NativeCSACall的S6闸门。
    此处只收集现有Tensor与view；逐请求几何继续由正式算子从Q/L读取。
    """
    groups = fixture["groups"]
    req = {name: fixture["metadata"][g["prefix"]].decode for name, g in groups.items()}
    indexer_metadata = fixture["metadata"][groups["indexer"]["prefix"]]
    root = adapter.prepare_weights(layer.self_attn, indexer_metadata.hadamard, layer)
    root.update(
        x_hc=fixture["hidden"], x_out=output, position_ids=fixture["positions"],
        kv_cache=groups["swa"]["views"][0], cmp_kv=groups["compressed"]["views"][0],
        idx_native_kv_cache=adapter.indexer_storage(*groups["indexer"]["views"]),
        kv_seq_lens=req["indexer"].seq_lens,
        cmp_query_start_loc=req["compressed"].query_start_loc,
        cmp_seq_lens=req["compressed"].seq_lens,
        idx_query_start_loc=req["indexer"].query_start_loc,
        compress_state=adapter.physical_pages(groups["state"]["views"][0]),
        inner_compress_state=adapter.physical_pages(groups["indexer_state"]["views"][0]),
        idx_topk=torch.empty((fixture["tokens"], 512), dtype=torch.int32, device=output.device),
        idx_topk_scores=torch.empty((fixture["tokens"], 512), dtype=torch.float32, device=output.device),
    )
    for name, key in (("swa", "ori"), ("compressed", "cmp"), ("indexer", "idx"),
                      ("state", "state"), ("indexer_state", "inner_state")):
        root[key + "_block_table"] = adapter.table_storage(req[name].block_table)
    for name, key in (("swa", "ori"), ("state", "state"), ("indexer_state", "inner_state")):
        root[key + "_slot_mapping"] = req[name].slot_mapping
    for name, prefix in (("compressed", "cmp"), ("indexer", "inner")):
        cos, sin, slots = fixture["compact"][name]
        root[prefix + "_freqs_cos"] = cos.view(-1, 64)
        root[prefix + "_freqs_sin"] = sin.view(-1, 64)
        root[("cmp" if name == "compressed" else "idx") + "_slot_mapping"] = slots
    layer_name = layer.self_attn.dsa_attn.dsa_attn.layer_name
    for kind in ("cos", "sin"):
        root["freqs_" + kind] = getattr(req["compressed"], kind)[layer_name][:fixture["tokens"]].view(-1, 64)
    return root
