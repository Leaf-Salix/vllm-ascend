# SPDX-License-Identifier: Apache-2.0
"""S5的原PTO因果前缀参照：真实S6计算的前五行应与S5计算相同。

只用于正确性诊断，不进入正式适配和性能计时。旧分task真实执行S6，输入前
五行、位置、页表与初始KV逐项对应S5；第六行分别取X和-X，验证前五行及共同
缓存范围不受未来query影响后，才与单SPMD的真实S5比较。忽略区仅限S6额外
合法写槽，S5保护区仍由原驱动完整检查；不忽略任何S5合法输出或缓存写入。
这不是Native精度或整模型token验收，也不修改DECODE_SEQ或构造假S6输入。
"""

import json


def validate(adapter, layer, config, fixture, eager, scores, options, output_path):
    """复用真实Native fixture及原分task数学；返回前缀/公共缓存的精确比较。"""
    import torch
    from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, restore
    from dsv4_csa_validation import compare_tensor

    tokens = fixture["tokens"]
    if fixture["query_tokens"] != 5 or tokens % 5:
        raise ValueError("本探针只处理等长真实S5")
    batch = tokens // 5
    history = int(fixture["positions"][0].cpu())
    device = fixture["hidden"].device
    six = make_fixture(config, layer.self_attn, batch, history, 1024, device, query_tokens=6)
    six["hidden"].view(batch, 6, 4, 4096)[:, :5].copy_(fixture["hidden"].view(batch, 5, 4, 4096))
    assert torch.equal(six["positions"].cpu().view(batch, 6)[:, :5].reshape(tokens), fixture["positions"].cpu())
    masks = {}
    geometry = {}
    for name, target in six["groups"].items():
        original = fixture["groups"][name]
        if target["allocation"].shape != original["allocation"].shape:
            raise ValueError(f"{name}分配容量不同，不能直接做物理页前缀参照")
        target["allocation"].copy_(original["initial"])
        target["initial"] = original["initial"].clone()
        actual_table = six["metadata"][target["prefix"]].decode.block_table.cpu()
        expected_table = fixture["metadata"][original["prefix"]].decode.block_table.cpu()
        assert torch.equal(actual_table, expected_table), f"{name}物理页映射必须完全一致"
        extra = target["allowed"] & ~original["allowed"]
        # 比较所有共同写入、S5独有写入和两边都不应写的字节；只排除S6独有槽。
        masks[name] = ~extra
        geometry[name] = dict(extra_s6_write_bytes=int(extra.sum()),
                              compared_bytes=int((~extra).sum()),
                              s5_writable_bytes=int(original["allowed"].sum()),
                              covers_all_s5_writes=bool((masks[name] | ~original["allowed"]).all()))
        assert geometry[name]["covers_all_s5_writes"]
    groups = {name: (six["metadata"][g["prefix"]], g["views"]) for name, g in six["groups"].items()}
    weights = adapter.prepare_weights(layer.self_attn, groups["indexer"][0].hadamard, layer)
    output = torch.empty_like(six["hidden"])
    call = adapter.NativeCSACall(
        adapter.CSAOperators.register(), weights, six["hidden"], six["positions"], groups,
        layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name, compact_metadata=six["compact"],
        buffers=dict(x_out=output,
                     idx_topk=torch.empty((batch * 6, 512), dtype=torch.int32, device=device),
                     idx_topk_scores=torch.empty((batch * 6, 512), dtype=torch.float32, device=device)),
    )
    extra_query = six["hidden"].view(batch, 6, 4, 4096)[:, 5].clone()
    baseline = None
    s5 = dict(x_out=eager["x_out"], idx_topk=eager["idx_topk"], idx_topk_scores=scores)
    for name, group in fixture["groups"].items():
        s5["bytes." + name] = group["allocation"].cpu()[masks[name]]
    report = dict(passed=False, scope="真实S6前五行对真实S5，非Native/性能验收", geometry=geometry,
                  rounds=[], initial_kv_identical=True, position_prefix_identical=True)
    try:
        for sign in (1, -1):
            restore(six)
            six["hidden"].view(batch, 6, 4, 4096)[:, 5].copy_(sign * extra_query)
            adapter.decode_csa_tp1_layer_test(*call.core_args, config=options)
            torch.npu.synchronize()
            all_values = collect_state(six, output, call.args["idx_topk"])
            selected = {key: all_values[key].reshape(batch, 6, *all_values[key].shape[1:])[:, :5].reshape_as(s5[key])
                        for key in ("x_out", "idx_topk")}
            selected["idx_topk_scores"] = call.args["idx_topk_scores"].cpu().view(batch, 6, 512)[:, :5].reshape(tokens, 512)
            for name, group in six["groups"].items():
                selected["bytes." + name] = group["allocation"].cpu()[masks[name]]
            guards = guard_checks(six)
            checks = {key: compare_tensor(value, s5[key], 0, 0) for key, value in selected.items()}
            causal = {} if baseline is None else {key: compare_tensor(value, baseline[key], 0, 0)
                                                  for key, value in selected.items()}
            report["rounds"].append(dict(extra_query_sign=sign, comparisons=checks, guards=guards,
                                         causal_prefix_comparisons=causal))
            if baseline is None:
                baseline = selected
            if any(c["status"] != "PASS" for c in (*checks.values(), *causal.values(), *guards.values())):
                torch.save(dict(reference=selected, s5=s5), output_path.parent / f"causal_failure_{sign}.pt")
        report["passed"] = all(c["status"] == "PASS" for row in report["rounds"]
                               for group in (row["comparisons"], row["guards"], row["causal_prefix_comparisons"])
                               for c in group.values())
    finally:
        # 构造第二份fixture会临时重绑Native模块，结束后恢复原S5绑定。
        for name, group in fixture["groups"].items():
            group["owner"].kv_cache = [group["views"]] if name == "indexer" else group["views"]
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
