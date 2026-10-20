# SPDX-License-Identifier: Apache-2.0
"""复用Native fixture/builder，验证多task同图metadata变化和补位请求。

只检查受影响的S6合同。原多task逐状态作为参照，候选始终回放同一张图；
内部TMR工作区不向测试暴露，不冒称检查了单SPMD信号或所有内部scratch。
"""

import json

import torch
from dsv4_csa_single_layer import guard_checks, make_fixture, restore, writable_bytes
from dsv4_csa_validation import compare_tensor
from metadata_replay import METADATA_ARGUMENTS
from padding_replay import COMPACT_ARGUMENTS, OUTPUT_ARGUMENTS


def validate(
    config,
    layer,
    fixture,
    call,
    adapter,
    original,
    candidate,
    output_path,
    history,
    batch,
    context_switch_history=None,
):
    groups = fixture["groups"]
    device = fixture["hidden"].device
    original_inputs = {name: call.args[name].cpu().clone() for name in METADATA_ARGUMENTS}
    original_hidden = fixture["hidden"].clone()
    addresses = {name: value.data_ptr() for name, value in call.args.items() if isinstance(value, torch.Tensor)}
    common_original = {
        name: dict(
            slots=g["common"].slot_mapping.cpu().clone(),
            table=g["common"].block_table_tensor.cpu().clone(),
            lengths=g["common"]._seq_lens_cpu.clone(),
            actual_tokens=g["common"].num_actual_tokens,
        )
        for name, g in groups.items()
    }
    states = {"A": dict(inputs=original_inputs, allowed={n: g["allowed"] for n, g in groups.items()}, active=batch)}
    report = dict(
        passed=False, scope="S6同址metadata及padding；不计性能", runtime="多task TMR，不使用单SPMD代际协议", steps=[]
    )
    if context_switch_history is not None:
        if not 0 < context_switch_history < history:
            raise ValueError("同图长短切换要求较短的正history")
        report["context_switch_history"] = context_switch_history

    def save():
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def update_common(active):
        common_cache, prefill_cache, decode_cache, metadata = {}, {}, {}, {}
        for name, group in groups.items():
            common, initial = group["common"], common_original[name]
            common._seq_lens_cpu.copy_(initial["lengths"])
            common._seq_lens_cpu[active:].zero_()
            common.seq_lens.copy_(common._seq_lens_cpu)
            common.num_actual_tokens = active * 6
            common.slot_mapping.copy_(initial["slots"])
            common.slot_mapping[active * 6 :].fill_(-1)
            common.block_table_tensor.copy_(initial["table"])
            common.block_table_tensor[active:].zero_()
            current = group["builder"].build(
                0,
                common,
                num_reqs_actual=active,
                block_size=group["spec"].block_size,
                common_ratio_to_sas_metadata=common_cache,
                prefill_ratio_to_sas_metadata=prefill_cache,
                decode_ratio_to_sas_metadata=decode_cache,
            )
            old = fixture["metadata"][group["prefix"]].decode
            for field in ("query_start_loc", "seq_lens", "block_table", "slot_mapping", "start_pos"):
                before, after = getattr(old, field), getattr(current.decode, field)
                if before is not None and (after is None or before.data_ptr() != after.data_ptr()):
                    raise AssertionError(f"Native builder改变已捕获地址：{name}.{field}")
            metadata[name] = current
        return metadata

    def reset(label):
        for name, value in states[label]["inputs"].items():
            call.args[name].copy_(value)
        fixture["hidden"].copy_(original_hidden)
        restore(fixture)
        call.args["x_out"].fill_(float("nan"))
        call.args["idx_topk"].fill_(-777)
        call.args["idx_topk_scores"].fill_(float("nan"))

    def collect(active):
        return {
            **{name: call.args[name][: active * 6].cpu() for name in OUTPUT_ARGUMENTS},
            **{name: group["allocation"].cpu() for name, group in groups.items()},
        }

    def guards(label):
        state = states[label]
        return guard_checks(
            dict(
                groups={
                    n: dict(allocation=g["allocation"], initial=g["initial"], allowed=state["allowed"][n])
                    for n, g in groups.items()
                },
                readonly={n: (call.args[n], v) for n, v in state["inputs"].items()},
            )
        )

    try:
        # Native构造本轮不同长度、位置及页顺序；先快照A，避免常驻RoPE被覆盖。
        variants = [("B", history - 1, True)]
        if context_switch_history is not None:
            variants.append(("C", context_switch_history, False))
        for label, alternate_history, reverse_pages in variants:
            variant = make_fixture(
                config,
                layer.self_attn,
                batch,
                alternate_history,
                1024,
                device,
                table_history=history,
                reverse_pages=reverse_pages,
            )
            other_groups = {n: (variant["metadata"][g["prefix"]], g["views"]) for n, g in variant["groups"].items()}
            other = adapter.NativeCSACall(
                call.ops,
                adapter.prepare_weights(layer.self_attn, other_groups["indexer"][0].hadamard, layer),
                variant["hidden"],
                variant["positions"],
                other_groups,
                layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                compact_metadata=variant["compact"],
            )
            for name in METADATA_ARGUMENTS:
                if call.args[name].shape != other.args[name].shape or call.args[name].dtype != other.args[name].dtype:
                    raise AssertionError(f"metadata边界改变描述符：{name}")
            states[label] = dict(
                inputs={n: other.args[n].cpu().clone() for n in METADATA_ARGUMENTS},
                allowed={n: g["allowed"] for n, g in variant["groups"].items()},
                active=batch,
            )
            del other, other_groups, variant
        for name, value in original_inputs.items():
            call.args[name].copy_(value)
        impl = layer.self_attn.dsa_attn.dsa_attn.impl
        for active in dict.fromkeys((batch - 1, 1)):
            metadata = update_common(active)
            compact = {}
            for name, arguments in COMPACT_ARGUMENTS:
                captured = impl._compute_compressor_metadata(fixture["metadata"][groups[name]["prefix"]].decode)
                compact[name] = impl._compute_compressor_metadata(metadata[name].decode)
                valid = (compact[name][2].cpu() >= 0).all(dim=1).nonzero().flatten()
                for argument, value, oracle in zip(arguments, captured, compact[name]):
                    target = call.args[argument]
                    if value.numel() != target.numel() or value.dtype != target.dtype:
                        raise AssertionError(f"补位compact描述符改变：{argument}")
                    target.copy_(value.view_as(target))
                    check = compare_tensor(target.cpu()[valid], oracle.cpu().view(-1, target.shape[-1])[valid], 0, 0)
                    if check["status"] != "PASS":
                        raise AssertionError(f"Native满档compact与活动请求oracle不一致：{argument}")
            allowed = {}
            for name, group in groups.items():
                slots = compact[name][2] if name in compact else metadata[name].decode.slot_mapping
                allowed[name] = writable_bytes(group["allocation"], group["views"], slots)
            states[f"P{active}"] = dict(
                inputs={n: call.args[n].cpu().clone() for n in METADATA_ARGUMENTS}, allowed=allowed, active=active
            )
        update_common(batch)
        reset("A")
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            candidate()
        torch.npu.synchronize()
        reference_a = None
        labels = ["A", "B", "A"] + (["C", "A"] if "C" in states else [])
        labels += [n for n in states if n.startswith("P")] + ["A"]
        for label in labels:
            active = states[label]["active"]
            reset(label)
            original()
            torch.npu.synchronize()
            expected = collect(active)
            reference_guards = guards(label)
            if label == "A" and reference_a is None:
                reference_a = {n: v.clone() for n, v in expected.items() if n in OUTPUT_ARGUMENTS}
            if label in ("B", "C") and torch.equal(expected["x_out"], reference_a["x_out"]):
                raise AssertionError("metadata变化没有形成有效输出变化")
            prefix = {}
            if label.startswith("P"):
                prefix = {n: compare_tensor(expected[n], reference_a[n][: active * 6], 0, 0) for n in OUTPUT_ARGUMENTS}
            if any(c["status"] != "PASS" for c in [*reference_guards.values(), *prefix.values()]):
                raise AssertionError(f"原多task的边界参照失败：{label}")
            reset(label)
            graph.replay()
            torch.npu.synchronize()
            actual = collect(active)
            checked = {n: compare_tensor(v, expected[n], 0, 0) for n, v in actual.items()}
            candidate_guards = guards(label)
            same_addresses = addresses == {n: v.data_ptr() for n, v in call.args.items() if isinstance(v, torch.Tensor)}
            passed = same_addresses and all(
                c["status"] == "PASS" for c in [*checked.values(), *candidate_guards.values()]
            )
            report["steps"].append(
                dict(
                    state=label,
                    active=active,
                    passed=passed,
                    comparisons=checked,
                    guards=candidate_guards,
                    reference_prefix=prefix,
                    same_addresses=same_addresses,
                )
            )
            save()
            if not passed:
                raise AssertionError(f"多task同图metadata/padding失败：{label}")
        report.update(
            passed=True,
            same_graph_replays=len(report["steps"]),
            internal_scratch_guards_checked=False,
            mixed_query_validated=False,
        )
        save()
        return report
    finally:
        update_common(batch)
        for name, initial in common_original.items():
            groups[name]["common"].num_actual_tokens = initial["actual_tokens"]
        for name, value in original_inputs.items():
            call.args[name].copy_(value)
        fixture["hidden"].copy_(original_hidden)
        restore(fixture)
