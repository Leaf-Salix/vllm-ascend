# SPDX-License-Identifier: Apache-2.0
"""完整动态CSA同址图回放：本轮历史长度、位置、页表与紧凑关闭数变化。

复用Native fixture与原分task PTO作为每个输入状态的独立参照。候选只捕获
一张图，改变现有Tensor内容；不重建Prepared、不更新Host tiling、不换地址。
输出/TopK/score和五份完整cache/state allocation均零差异比较，另外检查
本轮允许写区、输入metadata与工作区保护区。此测试不测性能、不证明混合Q。
"""

import json

import torch
from dsv4_csa_single_layer import guard_checks, make_fixture, restore
from dsv4_csa_validation import compare_tensor

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa import native_adapter as adapter
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.runtime_abi import PHASE_COUNT

METADATA_ARGUMENTS = (
    "position_ids",
    "ori_slot_mapping",
    "state_slot_mapping",
    "inner_state_slot_mapping",
    "ori_block_table",
    "cmp_block_table",
    "idx_block_table",
    "state_block_table",
    "inner_state_block_table",
    "kv_seq_lens",
    "cmp_seq_lens",
    "cmp_query_start_loc",
    "idx_query_start_loc",
    "freqs_cos",
    "freqs_sin",
    "cmp_freqs_cos",
    "cmp_freqs_sin",
    "inner_freqs_cos",
    "inner_freqs_sin",
    "cmp_slot_mapping",
    "idx_slot_mapping",
)


def validate(config, layer, fixture, call, candidate, allocations, compile_options, output_path, history, batch):
    device = fixture["hidden"].device
    hidden = fixture["hidden"].clone()
    addresses = {name: value.data_ptr() for name, value in candidate.buffers.items()}
    # Native RoPE可能共用常驻缓冲；必须在构造下一fixture前快照A，不能事后读A指针。
    states = {"A": {"fixture": fixture, "args": {name: call.args[name].cpu().clone() for name in METADATA_ARGUMENTS}}}
    for label, shift, reverse in (("B", -1, True), ("C", -2, False)):
        variant = make_fixture(
            config,
            layer.self_attn,
            batch,
            history + shift,
            1024,
            device,
            table_history=history,
            reverse_pages=reverse,
        )
        variant["hidden"].copy_(hidden)
        # 相同历史cache数据与hidden，只改变metadata；不让payload变化掩盖陈旧计划。
        for name, group in variant["groups"].items():
            initial = fixture["groups"][name]["initial"]
            if group["allocation"].shape != initial.shape:
                raise ValueError(f"不同状态cache容量不一致：{name}")
            group["initial"] = initial
            group["allocation"].copy_(initial)
        groups = {
            name: (variant["metadata"][group["prefix"]], group["views"]) for name, group in variant["groups"].items()
        }
        other = adapter.NativeCSACall(
            call.ops,
            adapter.prepare_weights(layer.self_attn, groups["indexer"][0].hadamard, layer),
            variant["hidden"],
            variant["positions"],
            groups,
            layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
            compact_metadata=variant["compact"],
        )
        metadata = {}
        for name in METADATA_ARGUMENTS:
            a, b = call.args[name], other.args[name]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"同址测试不能改变Tensor描述符：{name} {a.shape} != {b.shape}")
            metadata[name] = b.cpu().clone()
        states[label] = {"fixture": variant, "args": metadata}

    # 所有基线与候选都通过同一组A地址执行；其他fixture仅提供本轮元数据与写保护合同。
    def load(label):
        source = states[label]
        fixture["hidden"].copy_(hidden)
        for name, value in source["args"].items():
            call.args[name].copy_(value)
        restore(fixture)
        call.args["x_out"].fill_(float("nan"))
        call.args["idx_topk"].fill_(-777)
        call.args["idx_topk_scores"].fill_(float("nan"))

    def collect():
        return {
            "output": call.args["x_out"].cpu(),
            "topk": call.args["idx_topk"].cpu(),
            "scores": call.args["idx_topk_scores"].cpu(),
            **{name: group["allocation"].cpu() for name, group in fixture["groups"].items()},
        }

    def guards(label):
        source = states[label]["fixture"]
        view = {
            "groups": {
                name: {
                    "allocation": group["allocation"],
                    "initial": group["initial"],
                    "allowed": source["groups"][name]["allowed"],
                }
                for name, group in fixture["groups"].items()
            }
        }
        checks = guard_checks(view)
        checks.update(
            {
                "input." + name: compare_tensor(call.args[name], value, 0, 0)
                for name, value in states[label]["args"].items()
            }
        )
        return checks

    result = {"passed": False, "scope": "同一完整CSA图的历史/位置/页表/紧凑metadata变化，固定S6", "steps": []}

    def save():
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    references = {}
    for label in states:
        load(label)
        adapter.decode_csa_tp1_layer_test(*call.core_args, config=compile_options)
        torch.npu.synchronize()
        references[label] = collect()
        checked = guards(label)
        assert all(c["status"] == "PASS" for c in checked.values()), (label, checked)
    assert not torch.equal(references["A"]["output"], references["B"]["output"]), "A/B输出相同，未形成有效变化"
    result["metadata_changed"] = {
        label: [name for name in METADATA_ARGUMENTS if not torch.equal(states["A"]["args"][name], state["args"][name])]
        for label, state in states.items()
        if label != "A"
    }
    load("A")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        candidate(config=compile_options)
    torch.npu.synchronize()
    for label in ("A", "B", "C", "A", "B", "A"):
        load(label)
        # 故意覆盖旧计划与临时结果；signal代际与FFTS缓冲保留。
        for name, value in candidate.allocations.items():
            if name not in ("signal", "ffts"):
                value.fill_(37)
        before = candidate.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)[0, :, 0]
        graph.replay()
        torch.npu.synchronize()
        actual = collect()
        compared = {name: compare_tensor(value, references[label][name], 0, 0) for name, value in actual.items()}
        checked = guards(label)
        scratch = {
            name: bool((base[:g].cpu() == 37).all() and (base[-g:].cpu() == 37).all())
            for name, (base, g) in allocations.items()
        }
        signals = candidate.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)
        same_addresses = addresses == {name: value.data_ptr() for name, value in candidate.buffers.items()}
        protocol = bool((signals[0, :, 0] == before + 1).all() and (signals[:, :, 1:] == 0).all())
        passed = (
            same_addresses
            and protocol
            and all(scratch.values())
            and all(c["status"] == "PASS" for c in (*compared.values(), *checked.values()))
        )
        result["steps"].append(
            {
                "state": label,
                "passed": passed,
                "same_addresses": same_addresses,
                "protocol": protocol,
                "comparisons": compared,
                "guards": checked,
                "scratch_guards": scratch,
            }
        )
        save()
        if not passed:
            torch.save({"candidate": actual, "reference": references[label]}, output_path.with_suffix(".failure.pt"))
            raise AssertionError(f"完整CSA同址metadata图重放失败：{label}")
    load("A")
    result["passed"] = True
    result["same_graph_replays"] = len(result["steps"])
    result["performance_measured"] = False
    result["mixed_query_validated"] = False
    save()
    return result
