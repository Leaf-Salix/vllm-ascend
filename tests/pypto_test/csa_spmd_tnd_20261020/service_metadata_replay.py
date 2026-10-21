# SPDX-License-Identifier: Apache-2.0
"""Exercise changed Native metadata through an actual ACLGraphWrapper replay."""


def validate(*, runtime, layer, config, fixture, output, baseline, options, call, contracts):
    import torch
    from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, restore
    from dsv4_csa_validation import compare_tensor
    from vllm.forward_context import get_forward_context

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.native_adapter import (
        bind_native_inputs,
        token_capacity_view,
    )

    queries = list(reversed(fixture["query_lengths"]))
    histories = list(reversed(fixture["history_lengths"]))
    if queries == fixture["query_lengths"]:
        raise ValueError("变metadata框架图用例需要非均匀query长度")
    case = make_fixture(
        config,
        layer.self_attn,
        len(queries),
        max(histories),
        1024,
        output.device,
        query_lengths=queries,
        history_lengths=histories,
        token_capacity=fixture["tokens"],
        request_capacity=fixture["request_capacity"],
        table_history=fixture.get("table_history"),
        reverse_pages=True,
    )
    case["hidden"].copy_(fixture["hidden"])
    for name, group in case["groups"].items():
        initial = fixture["groups"][name]["initial"]
        if group["allocation"].shape != initial.shape:
            raise ValueError(f"变metadata用例物理cache描述符不一致：{name}")
        group["initial"] = initial
    restore(case)
    prepared = runtime.last_binding
    source = bind_native_inputs(
        runtime.weights,
        case["hidden"],
        case["positions"],
        {name: (case["metadata"][group["prefix"]], group["views"]) for name, group in case["groups"].items()},
        layer_name=runtime.layer_name,
        compact_metadata=case["compact"],
        output=output,
        topk=prepared.native["idx_topk"],
        topk_scores=prepared.native["idx_topk_scores"],
    )
    changed = {}

    def copy_in_place(target, value):
        if target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError("变metadata重放不能更换捕获Tensor描述符")
        if target.data_ptr() not in changed:
            changed[target.data_ptr()] = (target, target.clone())
        target.copy_(value)

    try:
        expected_entry = type(baseline)(
            source,
            max_query_tokens=6,
            max_seq_len=max(h + q for h, q in zip(histories, queries)),
            policy=baseline.policy,
        )
        expected_entry(config=options)
        torch.npu.synchronize()
        expected = collect_state(case, output, source["idx_topk"])
        expected_scores = source["idx_topk_scores"][: case["actual_tokens"]].cpu()
        for name, group in fixture["groups"].items():
            group["owner"].kv_cache = [group["views"]] if name == "indexer" else group["views"]
            target = fixture["metadata"][group["prefix"]].decode
            origin = case["metadata"][group["prefix"]].decode
            for field in ("num_reqs_actual", "num_compressed_tokens", "block_size", "max_seqlen_q", "max_seq_lens"):
                if getattr(target, field) != getattr(origin, field):
                    raise ValueError(f"本例须保持捕获Host参数：{name}.{field}")
            for field in ("input_positions", "query_start_loc", "seq_lens", "block_table", "start_pos"):
                left, right = getattr(target, field), getattr(origin, field)
                if left is not None:
                    copy_in_place(left, right)
            if target.slot_mapping is not None:
                copy_in_place(
                    token_capacity_view(target.slot_mapping, fixture["tokens"], 2),
                    token_capacity_view(origin.slot_mapping, fixture["tokens"], 2),
                )
            if name == "compressed":
                for field in ("cos", "sin"):
                    copy_in_place(
                        token_capacity_view(getattr(target, field)[runtime.layer_name], fixture["tokens"], 64),
                        token_capacity_view(getattr(origin, field)[runtime.layer_name], fixture["tokens"], 64),
                    )
        copy_in_place(fixture["positions"], case["positions"])
        restore(fixture)
        addresses = {name: tensor.data_ptr() for name, tensor in prepared.buffers.items()}
        before = runtime.submissions
        with set_ascend_forward_context(
            fixture["metadata"],
            config,
            num_tokens=fixture["tokens"],
            num_actual_tokens=fixture["actual_tokens"],
        ):
            allowed = bool(contracts) and all(
                contract.allows_replay(get_forward_context()) for contract in contracts.values()
            )
        if not allowed:
            raise AssertionError("同描述符、同Host合同的Q/L变化不应导致框架图失配")
        call(fixture, model_graph=True)
        torch.npu.synchronize()
        actual = collect_state(fixture, output, prepared.native["idx_topk"])
        checks = {name: compare_tensor(value, expected[name], 0, 0) for name, value in actual.items()}
        checks["scores"] = compare_tensor(
            prepared.native["idx_topk_scores"][: case["actual_tokens"]],
            expected_scores,
            0,
            0,
        )
        guards = guard_checks(
            {
                "groups": {
                    name: dict(
                        allocation=group["allocation"],
                        initial=group["initial"],
                        allowed=case["groups"][name]["allowed"],
                    )
                    for name, group in fixture["groups"].items()
                }
            }
        )
        no_submit = before == runtime.submissions
        same = addresses == {name: tensor.data_ptr() for name, tensor in prepared.buffers.items()}
        return dict(
            passed=no_submit
            and same
            and all(value["status"] == "PASS" for value in (*checks.values(), *guards.values())),
            query_lengths=queries,
            history_lengths=histories,
            reversed_physical_pages=True,
            python_not_resubmitted=no_submit,
            same_addresses=same,
            checks=checks,
            guards=guards,
        )
    finally:
        for target, value in changed.values():
            target.copy_(value)
        for name, group in fixture["groups"].items():
            group["owner"].kv_cache = [group["views"]] if name == "indexer" else group["views"]
