"""在既有三方门禁之后做只读阶段观测；不改变算子或SDK。"""

import hashlib
import json


def analyze(ns):
    import torch

    def tensor(value):
        if isinstance(value, tuple):
            assert len(value) == 2 and value[1] is None
            value = value[0]
        assert isinstance(value, torch.Tensor)
        return value

    def cpu(value):
        return value.detach().cpu().clone()

    metric, restore = ns["metric"], ns["restore"]
    layer, impl, tokens = ns["layer"], ns["impl"], ns["tokens"]
    out = ns["args"].out
    payload, comparisons = {}, {}
    graph, f, output, topk = ns["rows"]["nal_spmd"]
    restore(f)
    graph.replay()
    torch.npu.synchronize()
    state = ns["collect_state"](f, output, topk)
    assert all(ns["exact"](value, ns["states"]["nal_spmd"][name]) for name, value in state.items())
    buffers = ns["prepared"].buffers
    for name in (
        "mixed",
        "normed",
        "post",
        "comb",
        "qa",
        "qr",
        "qr_scale",
        "kv_fp32",
        "kv",
        "head_weights",
        "o_r",
        "o_scales",
    ):
        value = buffers[name]
        payload["nal_" + name] = cpu(value[:, :tokens] if name == "o_scales" else value[:tokens])
    packed = buffers["packed"]
    heads = packed.view(8, packed.shape[0] // 8, 4096)[:, :tokens].permute(1, 0, 2).contiguous()
    heads = heads.view(tokens, 64, 512)
    payload["nal_heads"] = cpu(heads)
    payload["nal_output"] = cpu(output)

    # 观测原生eager；必须先证明它与本任务Native FULL A金值相同。
    native_fixture = ns["fixtures"]["native"]
    original_hc, original_o = layer.hc_pre, impl._forward_o_proj
    original_sparse = torch.ops._C_ascend.npu_sparse_attn_sharedkv
    original_qli = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer
    sparse_calls, qli_calls = [], []
    # make_ragged_fixture逐臂更新共享owner，当前仍绑定最后Nal臂。
    # FULL图地址已录入；额外Native eager必须显式切回自己的cache。
    owner_bindings = []
    owner_proof = {}
    for name, group in native_fixture["groups"].items():
        owner = group["owner"]
        owner_bindings.append((owner, owner.kv_cache))
        owner.kv_cache = [group["views"]] if name == "indexer" else group["views"]
        actual_views = owner.kv_cache[0] if name == "indexer" else owner.kv_cache
        assert len(actual_views) == len(group["views"])
        owner_proof[name] = [a.data_ptr() == b.data_ptr() for a, b in zip(actual_views, group["views"], strict=True)]
        assert all(owner_proof[name])
    handles = []

    def hc(*args, **kwargs):
        result = original_hc(*args, **kwargs)
        for name, value in zip(("mixed", "post", "comb"), result, strict=True):
            payload["native_" + name] = cpu(value)
        return result

    def oproj(value, destination):
        payload["native_heads"] = cpu(value)
        result = original_o(value, destination)
        payload["native_attended"] = cpu(destination)
        return result

    def sparse(query, *args, **kwargs):
        assert not args
        result = original_sparse(query, **kwargs)
        sparse_calls.append((query.clone(), dict(kwargs), result[0].clone()))
        return result

    def qli(*args, **kwargs):
        assert not args
        result = original_qli(**kwargs)
        qli_calls.append(dict(kwargs))
        return result

    layer.hc_pre, impl._forward_o_proj = hc, oproj
    torch.ops._C_ascend.npu_sparse_attn_sharedkv = sparse
    torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = qli

    def norm_hook(module, inputs, value):
        payload["native_normed"] = cpu(tensor(value))

    handles.append(layer.input_layernorm.register_forward_hook(norm_hook))
    ns["layer"].self_attn.dsa_attn._pto_csa_runtime = None
    native_output = torch.empty_like(native_fixture["hidden"])
    try:
        restore(native_fixture)
        with ns["set_ascend_forward_context"](
            native_fixture["metadata"],
            ns["config"],
            num_tokens=tokens,
            num_actual_tokens=tokens,
            aclgraph_runtime_mode=ns["CUDAGraphMode"].FULL,
        ):
            ns["NativeHalf"](layer)(native_fixture["hidden"], native_fixture["positions"], native_output)
        torch.npu.synchronize()
        native_state = ns["collect_state"](native_fixture, native_output, ns["observed"]["topk"])
        control = {name: metric(value, ns["states"]["native"][name]) for name, value in native_state.items()}
        (out / "NATIVE_OBSERVATION_GATE.json").write_text(
            json.dumps({"owners_bound_to_native_views": owner_proof, "comparisons": control}, indent=2) + "\n"
        )
        torch.save(payload, out / "observation.pt")
        assert all(row["bit_equal"] for row in control.values()), "Native observation changed FULL gold"
    finally:
        layer.hc_pre, impl._forward_o_proj = original_hc, original_o
        torch.ops._C_ascend.npu_sparse_attn_sharedkv = original_sparse
        torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_qli
        for owner, previous in owner_bindings:
            owner.kv_cache = previous
        for handle in handles:
            handle.remove()
    payload["native_output"] = cpu(native_output)
    for name in ("mixed", "post", "comb", "normed", "heads", "output"):
        actual, expected = payload["nal_" + name], payload["native_" + name]
        assert actual.numel() == expected.numel() and actual.dtype == expected.dtype
        comparisons["end_to_end_" + name] = metric(actual.reshape(expected.shape), expected)

    # 各模块使用同一份Nal中间输入。上游差异不再混入该局部比较。
    local = {}
    conditional_norm = tensor(layer.input_layernorm(buffers["mixed"][:tokens]))
    local["norm_on_nal_mixed"] = metric(buffers["normed"], conditional_norm)
    payload["conditional_norm"] = cpu(conditional_norm)
    qa_native = tensor(impl.wq_a(buffers["normed"]))
    kv_native = tensor(impl.wkv(buffers["normed"]))
    assert qa_native.dtype == kv_native.dtype == torch.bfloat16
    local["qa_bf16_on_nal_normed"] = metric(buffers["qa"][:tokens].to(torch.bfloat16), qa_native)
    local["wkv_bf16_on_nal_normed"] = metric(buffers["kv_fp32"][:tokens].to(torch.bfloat16), kv_native)
    qr, scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
        buffers["qa"][:tokens].to(torch.bfloat16), impl.q_norm.weight, epsilon=impl.eps
    )
    local["qr_on_nal_qa_bf16"] = metric(buffers["qr"], qr)
    local["qr_scale_on_nal_qa_bf16"] = metric(buffers["qr_scale"], scale.reshape(tokens, 1))
    for name, value in (
        ("conditional_qa", qa_native),
        ("conditional_wkv", kv_native),
        ("conditional_qr", qr),
        ("conditional_qr_scale", scale),
    ):
        payload[name] = cpu(value)

    # 同heads对比输出后半段；重建Nal反量化后用Native HCpost验证HCpost契约。
    attended = torch.empty((tokens, 4096), dtype=torch.bfloat16, device=heads.device)
    with ns["set_ascend_forward_context"](
        native_fixture["metadata"],
        ns["config"],
        num_tokens=tokens,
        num_actual_tokens=tokens,
        aclgraph_runtime_mode=ns["CUDAGraphMode"].FULL,
    ):
        original_o(heads, attended)
    conditional_output = layer.hc_post(attended, f["hidden"], buffers["post"], buffers["comb"].view(tokens, 4, 4))
    local["oproj_hcpost_on_nal_heads"] = metric(output, conditional_output)
    partials = buffers["o_partials"][:tokens].view(tokens, 8, 4096)
    reconstructed = torch.zeros((tokens, 4096), dtype=torch.float32, device=heads.device)
    for group in range(8):
        reconstructed = reconstructed + partials[:, group].float() * buffers["o_scales"][group, :tokens, None]
    reconstructed = (reconstructed * buffers["wo_b_scale"].reshape(1, 4096)).to(torch.bfloat16)
    reconstructed_output = layer.hc_post(
        reconstructed, f["hidden"], buffers["post"], buffers["comb"].view(tokens, 4, 4)
    )
    local["native_hcpost_on_reconstructed_nal_attended"] = metric(reconstructed_output, output)
    local["native_oproj_vs_reconstructed_nal_attended"] = metric(reconstructed, attended)
    payload.update(
        conditional_attended=cpu(attended),
        reconstructed_nal_attended=cpu(reconstructed),
        reconstructed_nal_output=cpu(reconstructed_output),
        conditional_output=cpu(conditional_output),
    )
    (out / "STAGE_BASE_CHECKPOINT.json").write_text(
        json.dumps({"native_control": control, "end_to_end": comparisons, "same_input_modules": local}, indent=2) + "\n"
    )
    torch.save(payload, out / "stages-base.pt")

    # 条件重放：Sparse attention消费完全相同Nal Q/cache/TopK，排除上游数值差异。
    assert len(sparse_calls) == len(qli_calls) == 1
    native_q, sparse_kwargs, expected_pre_inverse = sparse_calls[0]
    repeated_pre_inverse = original_sparse(native_q, **sparse_kwargs)[0]
    assert ns["exact"](cpu(repeated_pre_inverse), cpu(expected_pre_inverse)), "Native sparse primitive repeat changed"
    layer_name = layer.self_attn.dsa_attn.dsa_attn.layer_name
    metadata = f["metadata"][f["groups"]["compressed"]["prefix"]]

    def inverse(value):
        result = value.clone()
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            result.unsqueeze(1),
            metadata.cos[layer_name],
            -metadata.sin[layer_name],
            rotary_mode="interleave",
            partial_slice=[impl.nope_head_dim, impl.head_dim],
        )
        return result

    inverse_control = inverse(repeated_pre_inverse)
    assert ns["exact"](cpu(inverse_control), payload["native_heads"]), "Native inverse RoPE observation changed"
    conditional_sparse_kwargs = dict(sparse_kwargs)
    conditional_sparse_kwargs.update(
        ori_kv=f["groups"]["swa"]["views"][0],
        cmp_kv=f["groups"]["compressed"]["views"][0],
        cmp_sparse_indices=topk.unsqueeze(1),
    )
    conditional_pre_inverse = original_sparse(buffers["q_heads"].view(tokens, 64, 512), **conditional_sparse_kwargs)[0]
    conditional_heads = inverse(conditional_pre_inverse)
    local["attention_inverse_on_nal_q_cache_topk"] = metric(heads, conditional_heads)
    payload["conditional_attention_heads"] = cpu(conditional_heads)
    payload["native_q"] = cpu(native_q)

    # Indexer同输入：独立检查权重系数、Hadamard/量化链和score/TopK。
    import vllm_ascend.attention.dsa_v1 as dsa
    from vllm_ascend.device.device_op import DeviceOperator

    weights = tensor(impl.weights_proj(buffers["normed"])) * (impl.indexer_softmax_scale * impl.indexer_heads**-0.5)
    weights = DeviceOperator.prepare_dsa_indexer_weights(weights)
    nal_weights = DeviceOperator.prepare_dsa_indexer_weights(buffers["head_weights"][:tokens])
    local["indexer_coeff_on_nal_normed"] = metric(nal_weights, weights)
    rope_query = buffers["idx_query_rope"][:tokens].view(tokens, 64, 128)
    native_hadamard = f["metadata"][f["groups"]["indexer"]["prefix"]].hadamard
    assert ns["exact"](cpu(buffers["hadamard"]), cpu(native_hadamard.T.contiguous()))
    rotated_query = dsa.rotate_activation(rope_query, native_hadamard)
    quant_query, quant_scale = DeviceOperator.indexer_quantize_query(rotated_query)
    quant_scale = DeviceOperator.prepare_dsa_indexer_query_scale(quant_scale)
    nal_query = buffers["idx_query_i8"].view(tokens, 64, 128)
    # Nal GM scale为FP32，score消费者先舍入FP16；Native QLI也要求FP16。
    nal_scale = buffers["idx_query_scale"].view(tokens, 64).to(torch.float16)
    local["indexer_hadamard_quant_on_nal_rope_query"] = metric(nal_query, quant_query)
    local["indexer_hadamard_scale_on_nal_rope_query"] = metric(nal_scale.reshape(quant_scale.shape), quant_scale)
    repeated_native_topk = original_qli(**qli_calls[0])[0].reshape(topk.shape)
    assert ns["exact"](cpu(repeated_native_topk), ns["states"]["native"]["idx_topk"].reshape(topk.shape)), (
        "Native QLI repeat changed"
    )
    conditional_qli = dict(qli_calls[0])
    conditional_qli.update(
        query=nal_query,
        key=f["groups"]["indexer"]["views"][0],
        weights=nal_weights,
        query_dequant_scale=nal_scale.reshape(conditional_qli["query_dequant_scale"].shape),
        key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(f["groups"]["indexer"]["views"][1]),
    )
    conditional_topk = original_qli(**conditional_qli)[0].reshape(topk.shape)
    local["indexer_topk_on_nal_query_key_coeff_scales"] = metric(topk, conditional_topk)
    payload.update(
        conditional_indexer_coeff=cpu(weights),
        nal_indexer_coeff=cpu(nal_weights),
        conditional_indexer_query=cpu(quant_query),
        nal_indexer_query=cpu(nal_query),
        conditional_indexer_scale=cpu(quant_scale),
        nal_indexer_scale=cpu(nal_scale),
        conditional_topk=cpu(conditional_topk),
        nal_topk=cpu(topk),
    )
    torch.npu.synchronize()
    assert all(row["status"] == "PASS" for row in ns["guard_checks"](f).values())
    ns["check_guards"]("nal_spmd", f)
    record = dict(
        complete=True,
        scope="同一8K/B32/L4单层；非性能、非整模型；只比较有效T",
        native_eager_vs_full_control=control,
        end_to_end=comparisons,
        same_input_modules=local,
        nal_hcpost_reconstruction_exact=local["native_hcpost_on_reconstructed_nal_attended"]["bit_equal"],
    )
    torch.save(payload, out / "stages.pt")
    record["stages_sha256"] = hashlib.sha256((out / "stages.pt").read_bytes()).hexdigest()
    (out / "stages.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print("STAGE_DIAGNOSTIC_COMPLETE", json.dumps(record, ensure_ascii=False), flush=True)
