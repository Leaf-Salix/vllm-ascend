# SPDX-License-Identifier: Apache-2.0
"""固定容量/地址/二进制，更新设备 metadata 后重放完整 CSA。

仅在计时外构造测试输入和读取比较结果。这里不增加生产入口的 Host tiling，
也不把还原历史的独立样例称为连续 decode。两侧均使用 PTO，不执行 Native forward。
"""

import copy
import json
from pathlib import Path

from shape_specialization import descriptors_from_tensors


OUTPUTS = frozenset(("x_out", "idx_topk", "idx_topk_scores"))
CACHES = frozenset(("kv_cache", "cmp_kv", "idx_native_kv_cache", "compress_state",
                    "inner_compress_state"))


def metadata_cases(descriptors, history, extra_histories=None):
    """仅用测试已知长度核对原编排分支，不要求跨数学策略共用捕获图。

    这里执行CPU编排模拟器，输入是用例常量，不从设备读取长度。实际核内仍读
    原device metadata。生产分档需要以已有Host长度选择对应模板，不能忽略边界。
    """
    from shape_specialization import bind, call_signature, catalog

    logical = Path(catalog()["logical_source_ir"])
    cache = {}

    def path_for(value):
        if value not in cache:
            inputs = copy.deepcopy(descriptors)
            inputs["kv_seq_lens"]["metadata_values"] = [value + 6] * inputs["kv_seq_lens"]["shape"][0]
            bindings = bind(logical, inputs)
            cache[value] = (call_signature(bindings), bindings["selected_control_path"])
        return cache[value]

    baseline_signature, baseline_path = path_for(history)
    if path_for(history - 1) != (baseline_signature, baseline_path):
        raise ValueError("B变体已跨原编排策略，须为对应长度选择另一个模板")
    rejected = []
    if extra_histories:
        selected = []
        for index, value in enumerate(extra_histories):
            if not 0 <= value < history - 1:
                raise ValueError("显式metadata历史必须小于原容量和B变体")
            signature, path = path_for(value)
            if (signature, path) != (baseline_signature, baseline_path):
                raise ValueError(f"显式history={value}跨原编排策略，需要另一个模板")
            selected.append((f"C{index + 1}", value, True))
        return [("A", history, False), ("B", history - 1, True), *selected,
                ("A_return", history, False)], {
                    "original_selected_control_path": baseline_path,
                    "rejected_cross_strategy_lengths": [],
                    "explicit_extra_histories": list(extra_histories),
                    "all_executed_variants_match_original_strategy": True}
    chosen = None
    for delta in (1024, 512, 256, 128, 64, 32, 16, 8, 6, 4, 2):
        candidate = history - delta
        if candidate < 0:
            continue
        signature, path = path_for(candidate)
        if signature == baseline_signature and path == baseline_path:
            chosen = candidate
            break
        rejected.append({"history": candidate, "reason": "原编排策略改变，不能用当前捕获图",
                         "selected_control_path": path})
    if chosen is None:
        raise ValueError("当前档内没有第三个不同长度样例")
    return [("A", history, False), ("B", history - 1, True),
            ("C", chosen, True), ("A_return", history, False)], {
                "original_selected_control_path": baseline_path,
                "rejected_cross_strategy_lengths": rejected,
                "all_executed_variants_match_original_strategy": True}


def address_signature(call, prepared):
    return {
        "roots": {name: int(value.data_ptr()) for name, value in call.args.items()},
        "entry": [int(value.data_ptr()) for value in prepared.core_args],
        "workspace": int(prepared.workspace_allocation.data_ptr()),
    }


def validate(*, config, layer, adapter, weights, fixture, call, prepared, entries,
             graphs, stream, batch, history, device, directory, extra_histories=None):
    import torch
    from test_static_jit_device import (
        checks_passed, collect_state, exact_checks, guard_checks, make_fixture,
    )

    if history < 1024:
        raise ValueError("本动态 metadata 用例要求 history>=1024，以改变实际分片工作量")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "eligible_for_700us_target": False,
              "scope": "固定batch/S6/页容量的A→B→C→A；长度、位置、页表和RoPE原址更新",
              "consecutive_decode_trajectory": False, "native_forward_executed": False,
              "batch": batch, "capacity_history": history, "steps": []}
    start_signature = address_signature(call, prepared)
    descriptors = descriptors_from_tensors(call.args)
    cases, report["strategy_boundary"] = metadata_cases(descriptors, history, extra_histories)
    mutable_names = sorted(set(call.args) - set(weights) - OUTPUTS - CACHES)
    old_guards = {name: (group["initial"], group["allowed"])
                  for name, group in fixture["groups"].items()}
    old_readonly = fixture["readonly"]

    def snapshot(value, binding, label, actual_history, reverse_pages):
        actual_descriptors = descriptors_from_tensors(binding.args)
        if actual_descriptors != descriptors:
            changed = {name: [descriptors[name], actual_descriptors.get(name)]
                       for name in descriptors
                       if descriptors[name] != actual_descriptors.get(name)}
            raise ValueError("动态metadata改变了根参数描述符：" + str(changed))
        if set(value["readonly"]) != set(old_readonly):
            raise ValueError("只读metadata字段集合改变")
        return {"label": label, "history": actual_history, "reverse_pages": reverse_pages,
                "roots": {name: binding.args[name].detach().cpu().clone() for name in mutable_names},
                "readonly": {name: tensor.detach().cpu().clone()
                             for name, (tensor, _) in value["readonly"].items()},
                "initial": {name: group["initial"] for name, group in value["groups"].items()},
                "allowed": {name: group["allowed"] for name, group in value["groups"].items()}}

    # 必须在 Native metadata builder 更新常驻 RoPE 缓冲之前保存 A 的内容。
    saved_a = snapshot(fixture, call, "A", history, False)

    def apply(snapshot_value):
        for name, (target, _) in old_readonly.items():
            target.copy_(snapshot_value["readonly"][name])
        for name in mutable_names:
            call.args[name].copy_(snapshot_value["roots"][name])
        fixture["readonly"] = {name: (target, snapshot_value["readonly"][name])
                               for name, (target, _) in old_readonly.items()}
        for name, group in fixture["groups"].items():
            group["initial"] = snapshot_value["initial"][name]
            group["allowed"] = snapshot_value["allowed"][name]
        stream.synchronize()

    def reset():
        for group in fixture["groups"].values():
            group["allocation"].copy_(group["initial"])
        call.args["x_out"].fill_(float("nan"))
        call.args["idx_topk"].fill_(-777)

    def check(reference):
        checks = exact_checks(collect_state(fixture, call.args["x_out"], call.args["idx_topk"]), reference)
        guards = guard_checks(fixture)
        if not checks_passed(checks) or not checks_passed(guards):
            report["failed_checks"] = {"exact": checks, "guards": guards}
            raise AssertionError("同地址metadata重放检查失败：" + str(report.get("active_step")))
        if address_signature(call, prepared) != start_signature:
            raise AssertionError("图根参数或工作区地址改变")
        return {"exact": checks, "guards": guards, "addresses_unchanged": True}

    single_replays = 0
    saved_a_output = None
    try:
        for label, next_history, reverse_pages in cases:
            report["active_step"] = {"variant": label, "history": next_history, "side": "prepare"}
            if label.startswith("A"):
                current = saved_a
            else:
                source_fixture = make_fixture(config, layer.self_attn, batch, next_history, 1024, device,
                                              table_history=history, reverse_pages=reverse_pages)
                scale = source_fixture["groups"]["indexer"]["views"][1]
                rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
                scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
                source_fixture["groups"]["indexer"]["initial"] = source_fixture["groups"]["indexer"]["allocation"].cpu()
                groups = {name: (source_fixture["metadata"][group["prefix"]], group["views"])
                          for name, group in source_fixture["groups"].items()}
                source_call = adapter.NativeCSACall(
                    call.ops, weights, source_fixture["hidden"], source_fixture["positions"], groups,
                    layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                    compact_metadata=source_fixture["compact"],
                    buffers={name: call.args[name] for name in OUTPUTS})
                current = snapshot(source_fixture, source_call, label, next_history, reverse_pages)
                # builder 临时挂载的新 cache 不能留在原层对象上。实际执行始终使用原根参数。
                for name, group in fixture["groups"].items():
                    group["owner"].kv_cache = [group["views"]] if name == "indexer" else group["views"]
                del source_call, source_fixture, groups, scale, rows
            apply(current)
            changed = [name for name in mutable_names
                       if not torch.equal(current["roots"][name], saved_a["roots"][name])]
            if not label.startswith("A"):
                required = {"position_ids", "kv_seq_lens", "idx_block_table", "cmp_block_table"}
                if not required.issubset(changed):
                    raise AssertionError("用例没有真正改变长度/位置/页表：" + str(changed))
            reset()
            report["active_step"]["side"] = "dynamic_eager"
            entries["dynamic_pto"]()
            stream.synchronize()
            reference = collect_state(fixture, call.args["x_out"], call.args["idx_topk"])
            eager_checks = check(reference)
            if label == "A":
                saved_a_output = reference["x_out"].clone()
            output_changed = not torch.equal(reference["x_out"], saved_a_output)
            if (label.startswith("A") and output_changed) or (not label.startswith("A") and not output_changed):
                raise AssertionError("A/B/C参考输出没有按输入内容变化或A恢复不一致")
            item = {"variant": label, "history": next_history, "reverse_pages": reverse_pages,
                    "changed_root_inputs": changed, "output_differs_from_a": output_changed,
                    "dynamic_eager": eager_checks}
            for side, graph in graphs.items():
                report["active_step"]["side"] = side
                (directory / "replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                reset()
                graph.replay()
                stream.synchronize()
                if side == "single_spmd":
                    single_replays += 1
                item[side] = check(reference)
            report["steps"].append(item)
            (directory / "replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            del reference
        report.update(passed=True, single_spmd_replays=single_replays, fixed_addresses=True,
                      graph_captured_once_per_side=True, signal_reset_between_variants=False)
        report.pop("active_step", None)
        return report, single_replays
    finally:
        # 即使比较抛异常，也还原 A，便于保留其既有保护区和末次进度检查。
        apply(saved_a)
        for name, group in fixture["groups"].items():
            group["initial"], group["allowed"] = old_guards[name]
            group["owner"].kv_cache = [group["views"]] if name == "indexer" else group["views"]
        fixture["readonly"] = old_readonly
        (directory / "replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
