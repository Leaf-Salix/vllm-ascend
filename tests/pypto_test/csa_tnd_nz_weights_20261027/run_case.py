# SPDX-License-Identifier: Apache-2.0
"""两种TND调度各自与修改前版本配对，验证Native NZ零拷贝及完整计算链。"""

import argparse
import importlib
import json
import os
import statistics
import traceback
from pathlib import Path

from csa_mpmd_tnd_20261021.run_case import check_case
from csa_mpmd_tnd_20261021.scenarios import make_fixture
from csa_spmd_tnd_eight_20261022.measurement_provenance import capture_toolchain, package_hashes
from l1 import export as export_l1
from load_snapshot import load_snapshot

NEW_WEIGHTS = ("wkv", "idx_wq_b", "weights_proj")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("spmd", "mpmd"), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int)
    parser.add_argument("--prepare-order", choices=("reference-first", "candidate-first"), default="reference-first")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    manifest = json.loads((here.parent / "csa_performance_20261022/cases.json").read_text())
    case = next(c for c in manifest["cases"] if c["name"] == args.case)
    check_case(case)
    order = ("reference", "candidate") if args.prepare_order == "reference-first" else ("candidate", "reference")
    report = {
        "passed": False,
        "device_execution": False,
        "case": case,
        "variant": args.variant,
        "prepare_order": args.prepare_order,
        "checks": {},
        "windows": [],
        "sampling": {"warmup": 3, "valid_windows_per_side": 10},
        "scope": "同一种TND调度的旧ND权重路径与Native NZ权重路径配对；完整L1包含原有初始化",
    }

    def save(stage):
        report["stage"] = stage
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"stage": stage, "passed": report["passed"], "variant": args.variant}), flush=True)

    try:
        import pypto
        import pypto.torch
        import torch
        import torch_npu
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import make_layer, restore
        from dsv4_csa_validation import compare_tensor
        from pypto import passes
        from pypto.runtime import CompileOptions
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        from vllm_ascend.utils import enable_custom_op

        pypto.configure_cache(pypto.CacheConfig(enabled=False))
        report["shared_toolchain"] = capture_toolchain(json.loads((here / "expected_sdk.json").read_text()))
        device_id = int(os.environ["TASK_DEVICE"])
        assert args.device is None or args.device == device_id
        report["device_id"] = device_id
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        current_platform.pre_register_and_update()
        assert enable_custom_op()
        variant_package = "deepseek_v4_flash_csa_spmd_tnd" if args.variant == "spmd" else "deepseek_v4_flash_csa_tnd"
        roots = {"reference": here / "snapshots/reference", "candidate": args.candidate}
        namespaces, packages = {}, {}
        report["source_hashes"] = {}
        for side in order:
            frozen = json.loads((here / (roots[side].name + "_source.json")).read_text())
            hashes = {name: package_hashes(roots[side] / name) for name in frozen["sources"]}
            assert hashes == frozen["sources"]
            report["source_hashes"][side] = hashes
            namespaces[side] = load_snapshot(roots[side], side)
            packages[side] = namespaces[side] + "." + variant_package
        config = EngineArgs(
            model=manifest["model"],
            tokenizer_mode="deepseek_v4",
            trust_remote_code=True,
            dtype="bfloat16",
            quantization="ascend",
            tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128},
            max_model_len=max(16384, case["history_label"] + 128),
            max_num_seqs=40,
            max_num_batched_tokens=256,
            block_size=32,
            enable_prefix_caching=False,
            enforce_eager=True,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        device = torch.device(f"npu:{device_id}")
        options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        with native_session(config, device_id), torch.inference_mode():
            layer, report["formal_weights"] = make_layer(config, Path(manifest["model"]), device, manifest["layer"])
            fixture = make_fixture(
                config,
                layer.self_attn,
                case["active_requests"],
                case["history_label"],
                case["seed"],
                device,
                query_lengths=case["query_lengths"],
                history_lengths=case["history_lengths"],
                token_capacity=case["token_capacity"],
                request_capacity=case["request_capacity"],
                table_history=case["table_history"],
                indexer_page_pattern=case["indexer_page_pattern"],
                reverse_pages=case["reverse_pages"],
            )
            scales = fixture["groups"]["indexer"]["views"][1]
            rows = torch.arange(scales.numel(), dtype=torch.int64, device=device)
            scales.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scales.dtype).reshape(scales.shape))
            fixture["groups"]["indexer"]["initial"] = fixture["groups"]["indexer"]["allocation"].cpu()
            groups = {
                name: (fixture["metadata"][item["prefix"]], item["views"]) for name, item in fixture["groups"].items()
            }
            weights = {}
            for side in order:
                service = importlib.import_module(packages[side] + ".service")
                weights[side] = service.CSAServiceRuntime._prepare_weights(layer.self_attn, layer)
                weights[side]["hadamard_idx"] = groups["indexer"][0].hadamard.detach().T.to(torch.bfloat16).contiguous()
            native_originals = {
                name: getattr(layer.self_attn, name).weight for name in ("wq_a", "wq_b", "wo_a", "wo_b", "wkv")
            }
            native_originals.update(
                idx_wq_b=layer.self_attn.indexer.wq_b.weight, weights_proj=layer.self_attn.indexer.weights_proj.weight
            )
            report["weight_binding"] = {}
            immutable = {name: value.cpu() for name, value in native_originals.items() if name in NEW_WEIGHTS}
            for name, original in native_originals.items():
                value = weights["candidate"][name]
                row = {
                    "native_shape": list(original.shape),
                    "prepared_shape": list(value.shape),
                    "native_format": int(torch_npu.get_npu_format(original)),
                    "prepared_format": int(torch_npu.get_npu_format(value)),
                    "same_data_ptr": value.data_ptr() == original.data_ptr(),
                    "same_storage": value.untyped_storage().data_ptr() == original.untyped_storage().data_ptr(),
                    "same_offset": value.storage_offset() == original.storage_offset(),
                    "logical_bytes": original.numel() * original.element_size(),
                }
                report["weight_binding"][name] = row
                assert (
                    row["native_format"] == row["prepared_format"] == 29
                    and row["native_shape"] == row["prepared_shape"]
                )
                assert row["same_data_ptr"] and row["same_storage"] and row["same_offset"]
                if name in NEW_WEIGHTS:
                    ref_value = weights["reference"][name]
                    expected = immutable[name].T if name != "idx_wq_b" else immutable[name]
                    assert torch_npu.get_npu_format(ref_value) in (0, 2)
                    assert torch.equal(ref_value.cpu(), expected)
            output = torch.empty_like(fixture["hidden"])
            topk = torch.empty((fixture["tokens"], 512), dtype=torch.int32, device=device)
            topk_scores = torch.empty((fixture["tokens"], 512), dtype=torch.float32, device=device)
            native_bind = importlib.import_module(packages["candidate"] + ".native_adapter").bind_native_inputs
            native_by_side = {
                side: native_bind(
                    weight,
                    fixture["hidden"],
                    fixture["positions"],
                    groups,
                    layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                    compact_metadata=fixture["compact"],
                    output=output,
                    topk=topk,
                    topk_scores=topk_scores,
                )
                for side, weight in weights.items()
            }
            native = native_by_side["reference"]
            report["fixture_checks"] = {
                "q": native["idx_query_start_loc"].cpu().tolist() == case["query_start_loc"],
                "l": native["kv_seq_lens"].cpu().tolist() == case["seq_lens"],
                "active_tokens": fixture["actual_tokens"] == case["valid_tokens"],
            }
            assert all(report["fixture_checks"].values())
            entries = {}
            for side in order:
                binding = importlib.import_module(packages[side] + ".binding").PreparedDynamicCSA
                entries[side] = binding(
                    native_by_side[side], max_query_tokens=max(case["query_lengths"]), max_seq_len=max(case["seq_lens"])
                )
            report["policies"] = {side: entry.policy.constexprs() for side, entry in entries.items()}
            pypto.torch.init(
                device=device_id,
                platform="a2a3",
                runtime="tensormap_and_ringbuffer",
                ring_heap=tuple(m * 1024 * 1024 for m in (16, 256, 16, 16)),
                ring_task_window=4096,
                enable_chip_swimlane=1,
                enable_dep_gen=False,
                output_dir=str(args.output / "dfx"),
            )
            slot_indices = {}
            for name in fixture["groups"]:
                slots = (
                    fixture["compact"][name][2] if name in fixture["compact"] else groups[name][0].decode.slot_mapping
                )
                pairs = sorted({(p, r) for p, r in slots.cpu().tolist() if p >= 0 and r >= 0})
                slot_indices[name] = ([p for p, _ in pairs], [r for _, r in pairs])

            def snapshot():
                state = {"output": output.cpu(), "topk": topk.cpu(), "scores": topk_scores.cpu()}
                checks = {}
                for name, group in fixture["groups"].items():
                    raw = group["allocation"].cpu()
                    checks[name] = bool(((raw == group["initial"]) | group["allowed"]).all())
                    for index, view in enumerate(group["views"]):
                        cpu_view = raw.view(view.dtype).as_strided(view.shape, view.stride(), view.storage_offset())
                        state[f"{name}.{index}"] = cpu_view[slot_indices[name]].clone()
                for name, (value, initial) in fixture["readonly"].items():
                    checks["readonly/" + name] = torch.equal(value.cpu(), initial)
                return state, checks

            def compare(actual, expected):
                checked = {}
                for name, value in actual.items():
                    rule = manifest["correctness"][
                        "floating_output"
                        if name == "output"
                        else "topk_scores"
                        if name == "scores"
                        else "floating_cache_state"
                    ]
                    atol, rtol = (rule["atol"], rule["rtol"]) if value.is_floating_point() else (0, 0)
                    item = (
                        compare_tensor(value, expected[name], atol, rtol)
                        if value.numel()
                        else {"status": "PASS", "elements": 0}
                    )
                    item["bit_equal"] = torch.equal(value.view(torch.uint8), expected[name].view(torch.uint8))
                    if not item["bit_equal"]:
                        item["status"] = "FAIL"
                    checked[name] = item
                return checked

            def verify(name, expected, *, normal=False):
                actual, guards = snapshot()
                if normal:
                    actual.pop("topk")
                    actual.pop("scores")
                numerical = compare(actual, expected)
                report["checks"][name] = {"guards": guards, "numerical": numerical}
                save(name)
                assert all(guards.values()) and all(v["status"] == "PASS" for v in numerical.values()), name

            snapshots, tables = {}, {}
            build = Path(os.environ["PYPTO_PROG_BUILD_DIR"])
            for side, entry in entries.items():
                save("compile_and_eager_" + side)
                restore(fixture)
                before = set(build.glob("*/kernel_config.py"))
                entry(config=options)
                torch.npu.synchronize()
                report["device_execution"] = True
                created = set(build.glob("*/kernel_config.py")) - before
                assert len(created) == 1, created
                tables[side] = created.pop()
                snapshots[side] = snapshot()
            expected = snapshots["reference"][0]
            for side in order:
                actual, guards = snapshots[side]
                numerical = compare(actual, expected)
                report["checks"][side + "_eager"] = {"guards": guards, "numerical": numerical}
                save(side + "_eager")
                assert all(guards.values()) and all(v["status"] == "PASS" for v in numerical.values())
            graphs = {}
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for side, entry in entries.items():
                    restore(fixture)
                    stream.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        entry(config=options)
                    graphs[side] = graph
                    restore(fixture)
                    graph.replay()
                    stream.synchronize()
                    verify(side + "_graph", expected)
                for sample in range(3):
                    for side in order if sample % 2 == 0 else tuple(reversed(order)):
                        restore(fixture)
                        graphs[side].replay()
                        stream.synchronize()
                save("paired_complete_L1")
                for sample in range(10):
                    for side in order if sample % 2 == 0 else tuple(reversed(order)):
                        restore(fixture)
                        stream.synchronize()
                        pypto.torch.begin_dfx()
                        try:
                            graphs[side].replay()
                        finally:
                            pypto.torch.end_dfx()
                        source = args.output / "dfx"
                        if report["windows"]:
                            source /= f"window_{len(report['windows'])}"
                        report["windows"].append(
                            export_l1(
                                source,
                                args.output / f"l1_{sample:02d}_{side}",
                                side=side,
                                variant=args.variant,
                                kernel_config=tables[side],
                            )
                        )
                for side, graph in graphs.items():
                    restore(fixture)
                    graph.replay()
                    stream.synchronize()
                    verify(side + "_post_L1", expected)
                if case["id"] == "C04":
                    normal_native = {
                        k: v for k, v in native_by_side["candidate"].items() if k not in ("idx_topk", "idx_topk_scores")
                    }
                    binding = importlib.import_module(packages["candidate"] + ".binding").PreparedDynamicCSA
                    normal_entry = binding(
                        normal_native, max_query_tokens=max(case["query_lengths"]), max_seq_len=max(case["seq_lens"])
                    )
                    save("compile_normal_entry")
                    restore(fixture)
                    normal_entry(config=options)
                    stream.synchronize()
                    verify("normal_eager", expected, normal=True)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        normal_entry(config=options)
                    restore(fixture)
                    graph.replay()
                    stream.synchronize()
                    verify("normal_graph", expected, normal=True)
            report["weights_unchanged"] = {
                name: torch.equal(native_originals[name].cpu(), original) for name, original in immutable.items()
            }
            assert all(report["weights_unchanged"].values())
            report["statistics_us"] = {}
            for side in order:
                samples = [w["incore_span_us"] for w in report["windows"] if w["side"] == side]
                assert len(samples) == 10
                report["statistics_us"][side] = {
                    "median": statistics.median(samples),
                    "min": min(samples),
                    "max": max(samples),
                    "samples": samples,
                }
            report["passed"] = True
            save("complete")
    except BaseException:
        report["error"] = traceback.format_exc()
        save("failed")
        raise


if __name__ == "__main__":
    main()
