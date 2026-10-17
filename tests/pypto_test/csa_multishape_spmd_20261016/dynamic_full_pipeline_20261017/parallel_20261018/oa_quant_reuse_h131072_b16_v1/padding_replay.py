# SPDX-License-Identifier: Apache-2.0
"""完整动态 CSA 的同址 padding 图验证，不采性能。

沿用 Native builder 的满档捕获合同：Tensor 容量和 Q 前缀保持整档，补位
请求的 L=0、原始/state slot=-1、页表=0，positions/RoPE 尾部保留旧值。
Native compact producer 用捕获时的满档描述符计算，再更新已绑定的输入
内容；另用实际请求数的 Native producer 确定有效 compact 行与合法写区。
这些准备在测试图外执行，图内只有完整单 SPMD；不宣称验证了 Native
producer 的捕获，也不把图外准备当成性能收益或正式适配代码。

同一张图按 B→B-1→1→B 重放，既对照原分 task PTO 的本轮有效输出，
又检查有效前缀与满档结果一致。五份 cache/state allocation 全量比较，
dummy 所属页及其他非 slot 写区必须保持初态；不规定 dummy 输出数值。
工作区填毒、所有输入地址、只读 metadata 和 72 核代际均逐轮核对。
此用例仍是每活动请求 S6，未覆盖空 rank、异构 Q 或连续 cache 演进。
"""

import json

import torch
from dsv4_csa_single_layer import guard_checks, restore, writable_bytes
from dsv4_csa_validation import compare_tensor
from metadata_replay import METADATA_ARGUMENTS

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa import native_adapter as adapter
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.runtime_abi import PHASE_COUNT

OUTPUT_ARGUMENTS = ("x_out", "idx_topk", "idx_topk_scores")
COMPACT_ARGUMENTS = (
    ("compressed", ("cmp_freqs_cos", "cmp_freqs_sin", "cmp_slot_mapping")),
    ("indexer", ("inner_freqs_cos", "inner_freqs_sin", "idx_slot_mapping")),
)


def validate(layer, fixture, call, candidate, allocations, compile_options, output_path, batch):
    """使用已有满档绑定和同一编译入口；不重建候选、工作区或 Host 策略。"""
    if batch < 2:
        raise ValueError("padding 验证至少需要两请求容量，空 rank 另行验证")
    groups = fixture["groups"]
    impl = layer.self_attn.dsa_attn.dsa_attn.impl
    original_inputs = {name: call.args[name].cpu().clone() for name in METADATA_ARGUMENTS}
    originals = {
        name: {
            "slots": group["common"].slot_mapping.cpu().clone(),
            "table": group["common"].block_table_tensor.cpu().clone(),
            "lengths": group["common"]._seq_lens_cpu.clone(),
            "actual_tokens": group["common"].num_actual_tokens,
        }
        for name, group in groups.items()
    }
    addresses = {name: value.data_ptr() for name, value in candidate.buffers.items()}
    result = {
        "passed": False,
        "bucket_batch": batch,
        "query_tokens": 6,
        "scope": "Native builder 更新同址输入；原分task PTO与完整单SPMD对照；只捕获SPMD",
        "metadata_preparation": "图外Native compact producer，保持满档捕获时的描述符与Host实际请求数",
        "padding_contract": "Q保留整档，L=0、原始/state slot=-1、页表=0，positions/RoPE保留旧尾部",
        "dummy_output_contract": "不规定dummy输出值；其cache/state及全部非允许写区必须保持初态",
        "performance_measured": False,
        "empty_rank_validated": False,
        "mixed_query_validated": False,
        "steps": [],
    }

    def save():
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    def update_metadata(active):
        common_cache, prefill_cache, decode_cache, metadata = {}, {}, {}, {}
        for name, group in groups.items():
            common, original = group["common"], originals[name]
            common._seq_lens_cpu.copy_(original["lengths"])
            common._seq_lens_cpu[active:].zero_()
            common.seq_lens.copy_(common._seq_lens_cpu)
            common.num_actual_tokens = active * 6
            common.slot_mapping.copy_(original["slots"])
            common.slot_mapping[active * 6 :].fill_(-1)
            common.block_table_tensor.copy_(original["table"])
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
            # 不把改变后的 Python metadata 对象替换到捕获绑定上；只能更新同址内容。
            old = fixture["metadata"][group["prefix"]].decode
            for field in ("query_start_loc", "seq_lens", "block_table", "slot_mapping", "start_pos"):
                before, after = getattr(old, field), getattr(current.decode, field)
                if before is not None and (after is None or before.data_ptr() != after.data_ptr()):
                    raise ValueError(f"Native builder 更换了捕获输入地址：{name}.{field}")
            metadata[name] = current
        return metadata

    def prepare(active):
        metadata = update_metadata(active)
        oracle, compact_checks = {}, {}
        for name, arguments in COMPACT_ARGUMENTS:
            # 捕获时固定 num_reqs_actual=B，dummy 是否有效由 device Q/L 等输入决定。
            captured = impl._compute_compressor_metadata(fixture["metadata"][groups[name]["prefix"]].decode)
            oracle[name] = impl._compute_compressor_metadata(metadata[name].decode)
            valid = (oracle[name][2].cpu() >= 0).all(dim=1).nonzero().flatten()
            for argument, actual, expected in zip(arguments, captured, oracle[name]):
                destination = call.args[argument]
                if actual.numel() != destination.numel() or actual.dtype != destination.dtype:
                    raise ValueError(f"满档 compact producer 改变了容量/dtype：{argument}")
                destination.copy_(actual.view_as(destination))
                width = destination.shape[-1]
                compact_checks[argument] = compare_tensor(
                    destination.cpu()[valid], expected.cpu().view(-1, width)[valid], 0, 0
                )
        allowed = {}
        for name, group in groups.items():
            slots = oracle[name][2] if name in oracle else metadata[name].decode.slot_mapping
            allowed[name] = writable_bytes(group["allocation"], group["views"], slots)
        inputs = {name: call.args[name].cpu().clone() for name in METADATA_ARGUMENTS}
        if not bool((inputs["kv_seq_lens"][active:] == 0).all()):
            raise AssertionError("补位请求的 Native L 未清零")
        for name in ("ori_slot_mapping", "state_slot_mapping", "inner_state_slot_mapping"):
            if not bool((inputs[name][active * 6 :, 0] < 0).all()):
                raise AssertionError(f"补位 slot 没有失效：{name}")
        return inputs, allowed, compact_checks

    def reset():
        restore(fixture)
        call.args["x_out"].fill_(float("nan"))
        call.args["idx_topk"].fill_(-777)
        call.args["idx_topk_scores"].fill_(float("nan"))

    def collect(active):
        return {
            **{name: call.args[name][: active * 6].cpu() for name in OUTPUT_ARGUMENTS},
            **{name: group["allocation"].cpu() for name, group in groups.items()},
        }

    def guards(inputs, allowed):
        return guard_checks(
            {
                "groups": {
                    name: {"allocation": group["allocation"], "initial": group["initial"], "allowed": allowed[name]}
                    for name, group in groups.items()
                },
                "readonly": {name: (call.args[name], value) for name, value in inputs.items()},
            }
        )

    save()
    try:
        prepare(batch)
        reset()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            candidate(config=compile_options)
        torch.npu.synchronize()
        full_outputs = None
        counts = [batch, *dict.fromkeys((batch - 1, 1)), batch]
        for active in counts:
            inputs, allowed, compact_checks = prepare(active)
            if any(check["status"] != "PASS" for check in compact_checks.values()):
                raise AssertionError(f"Native compact producer 的有效行与实际请求 oracle 不一致：{active}/{batch}")
            reset()
            adapter.decode_csa_tp1_layer_test(*call.core_args, config=compile_options)
            torch.npu.synchronize()
            reference = collect(active)
            reference_guards = guards(inputs, allowed)
            if full_outputs is None:
                full_outputs = {name: reference[name] for name in OUTPUT_ARGUMENTS}
            prefix_checks = {
                name: compare_tensor(reference[name], full_outputs[name][: active * 6], 0, 0)
                for name in OUTPUT_ARGUMENTS
            }
            if any(check["status"] != "PASS" for check in (*reference_guards.values(), *prefix_checks.values())):
                result["reference_failure"] = {"active_batch": active, "guards": reference_guards, "prefix": prefix_checks}
                raise AssertionError(f"原分task基线的padding写区/有效前缀检查失败：{active}/{batch}")
            reset()
            for name, value in candidate.allocations.items():
                if name not in ("signal", "ffts"):
                    value.fill_(37)
            before = candidate.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)[0, :, 0]
            graph.replay()
            torch.npu.synchronize()
            actual = collect(active)
            compared = {name: compare_tensor(value, reference[name], 0, 0) for name, value in actual.items()}
            checked = guards(inputs, allowed)
            scratch = {
                name: bool((base[:g].cpu() == 37).all() and (base[-g:].cpu() == 37).all())
                for name, (base, g) in allocations.items()
            }
            signal = candidate.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)
            protocol = bool((signal[0, :, 0] == before + 1).all() and (signal[:, :, 1:] == 0).all())
            same_addresses = addresses == {name: value.data_ptr() for name, value in candidate.buffers.items()}
            passed = (
                protocol and same_addresses and all(scratch.values())
                and all(check["status"] == "PASS" for check in (*compared.values(), *checked.values()))
            )
            result["steps"].append(
                {
                    "active_batch": active,
                    "passed": passed,
                    "same_addresses": same_addresses,
                    "protocol": protocol,
                    "compact_metadata": compact_checks,
                    "reference_prefix": prefix_checks,
                    "reference_guards": reference_guards,
                    "comparisons": compared,
                    "guards": checked,
                    "scratch_guards": scratch,
                }
            )
            save()
            if not passed:
                torch.save({"candidate": actual, "reference": reference}, output_path.with_suffix(".failure.pt"))
                torch.save(
                    {name: value.cpu() for name, value in candidate.allocations.items()},
                    output_path.with_suffix(".workspace.pt"),
                )
                raise AssertionError(f"完整CSA padding 同址重放失败：{active}/{batch}")
        result["passed"] = True
        result["same_graph_replays"] = len(result["steps"])
        save()
        return result
    except BaseException:
        save()
        raise
    finally:
        update_metadata(batch)
        for name, original in originals.items():
            groups[name]["common"].num_actual_tokens = original["actual_tokens"]
        for name, value in original_inputs.items():
            call.args[name].copy_(value)
        restore(fixture)
