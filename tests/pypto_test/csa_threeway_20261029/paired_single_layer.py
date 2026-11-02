# SPDX-License-Identifier: Apache-2.0
"""冻结两份完整CSA，同一Native fixture，图恢复在计时外；不测整模型。"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import sys
from pathlib import Path


def mount_package(source, name):
    """与既有单层runner相同的隔离导入，避免依赖未同步的历史脚本。"""
    assert name not in sys.modules
    spec = importlib.util.spec_from_file_location(
        name, source / "__init__.py", submodule_search_locations=[str(source)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return name


def main():
    import pypto.torch
    import torch
    import torch_npu
    from dsv4_csa_native_case import native_session
    from dsv4_csa_single_layer import guard_checks, make_fixture, make_layer, restore
    from nal_native_alignment_20261027.ragged_fixture import make_ragged_fixture
    from pypto.runtime.kernel.abi import KernelConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu", choices=("ours", "nalinaly", "candidate"))
    parser.add_argument("--clock", action="store_true")
    parser.add_argument("--hc-detail", action="store_true")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--native-check", action="store_true")
    parser.add_argument("--batch", type=int, choices=range(1, 41), default=8)
    parser.add_argument("--query-lengths", help="逗号分隔的真实1..6行请求，不按S6补齐")
    parser.add_argument("--qualification-reference", type=Path, help="已通过性能和冻结字节门禁的结果JSON")
    parser.add_argument("--token-capacity", type=int)
    parser.add_argument("--request-capacity", type=int)
    parser.add_argument("--clock-cpu-label", default="20261102-v2")
    args = parser.parse_args()
    assert not args.hc_detail or args.clock
    queries = [6] * args.batch if args.query_lengths is None else [int(q) for q in args.query_lengths.split(",")]
    assert len(queries) == args.batch and all(1 <= q <= 6 for q in queries)
    actual_tokens = sum(queries)
    tokens = actual_tokens if args.token_capacity is None else args.token_capacity
    requests = args.batch if args.request_capacity is None else args.request_capacity
    assert actual_tokens <= tokens <= 240 and args.batch <= requests <= 40
    assert tokens - actual_tokens <= (requests - args.batch) * 6
    if args.native_check:
        assert tokens == actual_tokens and requests == args.batch, (
            "padded Native oracle requires full-capacity capture then in-place metadata update; use probe_padding.py"
        )
    args.output.mkdir(exist_ok=False)
    package = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    sources = {
        "ours": args.root / "cohorts/eightcard-source-v5/ours/source",
        "nalinaly": args.root / "cohorts/eightcard-nal-pre-aclnn-1574-v1/nalinaly/source",
    }
    if args.baseline:
        assert args.candidate and not args.clock
        sources["ours"] = args.baseline
    if args.candidate:
        assert not args.clock
        sources = {"ours": sources["ours"], "candidate": args.candidate}
    modules = {}
    source_contracts = {}
    report = {"scope": f"single real layer4, synthetic common hidden/history, B{args.batch}/8K", "sources": {}}
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["geometry"] = dict(
        actual_requests=args.batch,
        actual_tokens=actual_tokens,
        tokens=tokens,
        requests=requests,
        query_lengths=queries,
    )
    for arm, source in sources.items():
        source_contracts[arm] = json.loads((source.parent / "manifest.json").read_text())
        if not args.cpu and source_contracts[arm].get("hardware_blocked_reason"):
            raise RuntimeError(source_contracts[arm]["hardware_blocked_reason"])
        report["sources"][arm] = source_contracts[arm]["csa_source"]
        if arm == "candidate":
            changed = source_contracts[arm].get("changed_files")
            assert isinstance(changed, dict) and changed, "candidate must identify its changed sources"
            for name, expected in changed.items():
                assert hashlib.sha256((source / package / name).read_bytes()).hexdigest() == expected
            report["candidate_changed_files"] = changed
        name = mount_package(source / package, "vllm_ascend.ops.pypto._paired_" + arm)
        modules[arm] = {
            n: importlib.import_module(name + "." + n) for n in ("entry", "policy", "binding", "native_adapter")
        }
    qualification = None
    if args.qualification_reference:
        assert args.candidate and not args.cpu and not args.clock
        qualification = json.loads(args.qualification_reference.read_text())
        gain = qualification["candidate_gain"]
        assert gain["fraction"] >= 0.01 and gain["faster_pairs"] >= 24 and gain["pairs"] == 30
        assert qualification["candidate_precision_vs_frozen"] is True
        assert qualification["sources"] == report["sources"], "precision reference belongs to another candidate"
        assert qualification["candidate_changed_files"] == report["candidate_changed_files"], (
            "precision reference did not test these candidate files"
        )
        report["qualification_reference"] = {
            "path": str(args.qualification_reference),
            "sha256": hashlib.sha256(args.qualification_reference.read_bytes()).hexdigest(),
            "geometry": qualification["geometry"],
            "candidate_changed_files": report["candidate_changed_files"],
        }
    if args.cpu:
        arm = args.cpu
        if hasattr(modules[arm]["binding"], "precompile_static_weights"):
            report["static_weight_artifact"] = modules[arm]["binding"].precompile_static_weights()
        policy = modules[arm]["policy"].select_policy(
            tokens=tokens, requests=requests, max_query_tokens=6, max_seq_len=8198
        )
        artifact, _, _ = modules[arm]["entry"].decode_csa_spmd._resolve_compiled(
            (),
            policy.constexprs(),
            allow_signature_mode=True,
            _kernel=True,
            _kernel_config=KernelConfig("a2a3", "tensormap_and_ringbuffer", 0, ring_task_window=2048),
        )
        artifact.load()
        assert not torch_npu.npu.is_initialized()
        report.update(cpu_only=True, artifact=artifact.identity().hex(), compiled_dir=str(artifact.output_dir))
    else:
        assert os.environ["TASK_DEVICE"] in ("8", "9", "10", "11", "12", "13", "15")
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == os.environ["TASK_DEVICE"]
        report["physical_device"] = os.environ["TASK_DEVICE"]
        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        torch.use_deterministic_algorithms(True)
        torch_npu.npu.set_deterministic_level(1)
        torch.set_num_threads(4)
        current_platform.pre_register_and_update()
        from vllm_ascend.utils import enable_custom_op

        assert enable_custom_op(), "Native custom operators must be registered before metadata construction"
        checkpoint = "/data/model/DeepSeek-V4-Flash-0731-w8a8"
        config = EngineArgs(
            model=checkpoint,
            tokenizer_mode="deepseek_v4",
            trust_remote_code=True,
            dtype="bfloat16",
            quantization="ascend",
            tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128},
            max_model_len=16384,
            max_num_seqs=40,
            max_num_batched_tokens=256,
            block_size=32,
            enable_prefix_caching=False,
            enforce_eager=True,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        with native_session(config, 0), torch.inference_mode():
            layer, details = make_layer(config, Path(checkpoint), torch.device("npu:0"), 4)
            report["weights"] = details
            if tokens == actual_tokens and requests == args.batch:
                fixture = make_ragged_fixture(
                    config, layer.self_attn, args.batch, 8192, 1024, torch.device("npu:0"), queries
                )
            else:
                fixture = make_fixture(
                    config,
                    layer.self_attn,
                    args.batch,
                    8192,
                    1024,
                    torch.device("npu:0"),
                    query_lengths=queries,
                    token_capacity=tokens,
                    request_capacity=requests,
                )
            report["query_bounds"] = groups_bounds = (
                fixture["metadata"][fixture["groups"]["indexer"]["prefix"]].decode.query_start_loc.cpu().tolist()
            )
            assert groups_bounds[-1] == tokens
            groups = {n: (fixture["metadata"][g["prefix"]], g["views"]) for n, g in fixture["groups"].items()}
            if any(m.get("same_activity_domains_only") for m in source_contracts.values()):
                reference = groups["indexer"][0].decode
                bounds_cpu, lengths_cpu = reference.query_start_loc.cpu(), reference.seq_lens.cpu()
                report["same_activity_domains"] = {
                    name: torch.equal(value[0].decode.query_start_loc.cpu(), bounds_cpu)
                    and torch.equal(value[0].decode.seq_lens.cpu(), lengths_cpu)
                    for name, value in groups.items()
                }
                assert all(report["same_activity_domains"].values()), "candidate activity domain proof failed"
            for name, values in fixture["compact"].items():
                for index, value in enumerate(values[:2]):
                    fixture["readonly"][f"{name}.compact_rope.{index}"] = (value, value.cpu().clone())
            main = groups["compressed"][0].decode
            layer_name = fixture["groups"]["compressed"]["prefix"]
            for name, table in (("cos", main.cos), ("sin", main.sin)):
                value = table[layer_name]
                fixture["readonly"][f"rope.{layer_name}.{name}"] = (value, value.cpu().clone())
            from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.native_adapter import prepare_weights

            pypto.torch.init(
                ring_heap=(256 << 20, 512 << 20, 256 << 20, 256 << 20),
                ring_task_window=2048,
                enable_chip_swimlane=1,
                enable_dep_gen=False,
                output_dir=str(args.output / "dfx"),
            )
            fixture["readonly"]["hidden"] = (fixture["hidden"], fixture["hidden"].cpu().clone())
            graphs, outputs, controls, owners = {}, {}, {}, {}

            def snapshot(output):
                return {
                    "output": output[:actual_tokens].cpu().clone(),
                    **{n: g["allocation"].cpu().clone() for n, g in fixture["groups"].items()},
                }

            def check_readonly():
                for arm, (entry, _native, _weights) in owners.items():
                    checker = getattr(modules[arm]["binding"], "check_static_weights", None)
                    if checker is not None:
                        checker(entry)
                for value, initial in fixture["readonly"].values():
                    assert torch.equal(
                        value.cpu().contiguous().view(torch.uint8), initial.contiguous().view(torch.uint8)
                    )

            def check_guards():
                check_readonly()
                assert all(row["status"] == "PASS" for row in guard_checks(fixture).values()), "invalid write domain"

            for arm, module in modules.items():
                output = torch.empty_like(fixture["hidden"])
                outputs[arm] = output
                weights = prepare_weights(
                    layer.self_attn, groups["indexer"][0].hadamard, layer, root_function=module["entry"].decode_csa_spmd
                )
                # 实验两臂统一使用可追踪版本的冻结权重；无数值转换，准备不计时。
                with torch.inference_mode(False):
                    weights["hc_attn_fn"] = weights["hc_attn_fn"].clone()
                value = weights["hc_attn_fn"]
                fixture["readonly"][f"{arm}.hc_fn"] = (value, value.cpu().clone())
                native = module["native_adapter"].bind_native_inputs(
                    weights,
                    fixture["hidden"],
                    fixture["positions"],
                    groups,
                    layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                    compact_metadata=fixture["compact"],
                    output=output,
                )
                entry = module["binding"].PreparedDynamicCSA(native, max_query_tokens=6, max_seq_len=8198)
                if "hc_prepacked_weight" in entry.buffers:
                    value = entry.buffers["hc_prepacked_weight"]
                    fixture["readonly"][f"{arm}.hc_prepacked_weight"] = (value, value.cpu().clone())
                owners[arm] = (entry, native, weights)  # 保活所有图外权重/scale/绑定，直到末次replay。
                restore(fixture)
                entry()
                torch.npu.synchronize()
                check_guards()
                controls[arm] = snapshot(output)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    entry()
                graphs[arm] = graph
            samples = {arm: [] for arm in modules}
            for window in range(35):
                for arm in list(modules) if window % 2 == 0 else list(reversed(modules)):
                    restore(fixture)
                    torch.npu.synchronize()
                    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    begin.record()
                    graphs[arm].replay()
                    end.record()
                    torch.npu.synchronize()
                    check_readonly()
                    if window >= 5:
                        samples[arm].append(begin.elapsed_time(end) * 1000)
            report["self_replay_equal"] = {}
            for arm, graph in graphs.items():
                restore(fixture)
                graph.replay()
                torch.npu.synchronize()
                check_guards()
                state = snapshot(outputs[arm])
                report["self_replay_equal"][arm] = all(
                    torch.equal(state[n].contiguous().view(torch.uint8), v.contiguous().view(torch.uint8))
                    for n, v in controls[arm].items()
                )
            assert all(report["self_replay_equal"].values()), "graph/eager self control failed"
            assert all(len(v) == 30 and all(math.isfinite(x) and x > 0 for x in v) for v in samples.values())
            report["timing_us"] = {
                a: dict(samples=v, median=statistics.median(v), mean=statistics.mean(v)) for a, v in samples.items()
            }
            if args.candidate:
                paired = [a < b for a, b in zip(samples["candidate"], samples["ours"], strict=True)]
                gain = 1 - statistics.median(samples["candidate"]) / statistics.median(samples["ours"])
                report["candidate_gain"] = dict(fraction=gain, faster_pairs=sum(paired), pairs=len(paired))
                report["performance_threshold_passed_current_case"] = gain >= 0.01 and sum(paired) >= 24
                if report["performance_threshold_passed_current_case"] or qualification is not None:
                    report["candidate_precision_vs_frozen"] = all(
                        torch.equal(
                            v.contiguous().view(torch.uint8), controls["ours"][k].contiguous().view(torch.uint8)
                        )
                        for k, v in controls["candidate"].items()
                    )
                    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                    assert report["candidate_precision_vs_frozen"], "first candidate output/cache difference"
                    if args.native_check:
                        from csa_threeway_20261029.candidate_native_check import check_native

                        report["native_check"] = check_native(
                            fixture, layer, config, controls["candidate"], args.output
                        )
                        check_readonly()
                else:
                    report["candidate_precision_vs_frozen"] = "SKIPPED_NO_STABLE_PERFORMANCE_GAIN"
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            report["dfx"] = {}
            for index, (arm, graph) in enumerate(graphs.items()):
                restore(fixture)
                torch.npu.synchronize()
                pypto.torch.begin_dfx()
                try:
                    graph.replay()
                finally:
                    pypto.torch.end_dfx()
                torch.npu.synchronize()
                source = args.output / "dfx"
                if index:
                    source /= f"window_{index}"
                destination = args.output / f"dfx_{arm}.json"
                shutil.copyfile(source / "chip_swimlane_records.json", destination)
                assert json.loads(destination.read_text())["metadata"]["clock_freq_hz"] == 50_000_000
                report["dfx"][arm] = str(destination)
                check_guards()
            if args.clock:
                from csa_threeway_20261029.phase_clock import artifact_receipt, load_tools

                report["phase_clock"] = {}
                for arm, (entry, native, weights) in owners.items():
                    name = mount_package(
                        args.root
                        / (
                            "candidates/hc-detail-clock-20261102-v1"
                            if args.hc_detail
                            else "candidates/phase-clock-20261102-v2"
                        )
                        / arm,
                        "vllm_ascend.ops.pypto._phase_" + arm,
                    )
                    module = importlib.import_module(name + ".entry")
                    if args.hc_detail:
                        assert entry.policy.constexprs()["OUTPUT_GROUP_PIPELINE"], "40/41 slots must have no consumer"
                    signal = torch.zeros(
                        module.PHASE_COUNT * module.CORES * module.SIGNAL_SLOT_WORDS,
                        dtype=torch.int32,
                        device="npu",
                    )
                    assert signal.data_ptr() % 64 == 0
                    artifact, _, _ = module.decode_csa_spmd._resolve_compiled(
                        (),
                        {**entry.buffers, "clock_signal": signal, **entry.policy.constexprs()},
                        _kernel=True,
                        _kernel_config=KernelConfig("a2a3", "tensormap_and_ringbuffer", 0, ring_task_window=2048),
                    )
                    artifact.load()  # 真实形状和dtype；只加载不执行，通过审计才允许首次调用。
                    cpu = args.root / f"results/phase-clock-cpu-{arm}-{args.clock_cpu_label}"
                    sealed = json.loads((cpu / "sealed.json").read_text())
                    current = artifact_receipt(artifact, cpu / "codegen")
                    assert all(sealed[k] == v for k, v in current.items()), "clock artifact changed after CPU audit"

                    def clock_call(entry=entry, module=module, signal=signal):
                        module.decode_csa_spmd(**entry.buffers, clock_signal=signal, **entry.policy.constexprs())

                    restore(fixture)
                    clock_call()
                    torch.npu.synchronize()
                    check_guards()
                    observed = snapshot(outputs[arm])
                    assert all(
                        torch.equal(observed[k].view(torch.uint8), v.view(torch.uint8))
                        for k, v in controls[arm].items()
                    ), "clock eager changed output/cache"
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        clock_call()
                    restore(fixture)
                    graph.replay()
                    torch.npu.synchronize()
                    check_guards()
                    observed = snapshot(outputs[arm])
                    assert all(
                        torch.equal(observed[k].view(torch.uint8), v.view(torch.uint8))
                        for k, v in controls[arm].items()
                    ), "clock FULL changed output/cache"
                    abi = importlib.import_module(name + ".runtime_abi")
                    phases = {v: k for k, v in vars(abi).items() if k.startswith("PHASE_") and isinstance(v, int)}
                    if args.hc_detail:
                        phases[40], phases[41] = "HC_DETAIL_1", "HC_DETAIL_2"
                    for prefix in ("PHASE_OA_GROUP_BASE", "PHASE_O_QUANT_GROUP_BASE"):
                        for group in range(8):
                            phases[getattr(abi, prefix) + group] = f"{prefix}_{group}"
                    runs = []
                    for trial in range(5):
                        restore(fixture)
                        torch.npu.synchronize()
                        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                        begin.record()
                        graph.replay()
                        end.record()
                        torch.npu.synchronize()
                        check_guards()
                        observed = snapshot(outputs[arm])
                        assert all(
                            torch.equal(observed[k].view(torch.uint8), v.view(torch.uint8))
                            for k, v in controls[arm].items()
                        ), "clock repeated FULL changed output/cache"
                        row = load_tools(args.root).export_clock(
                            signal.cpu().view(module.PHASE_COUNT, module.CORES, module.SIGNAL_SLOT_WORDS),
                            args.output / f"phase_{arm}_{trial}",
                            phases,
                        )
                        row["diagnostic_event_us"] = begin.elapsed_time(end) * 1000
                        runs.append(row)
                    report["phase_clock"][arm] = {"self_equal": True, "runs": runs}
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        destination = Path(sys.argv[2])
        if destination.is_dir():
            (destination / "failure.json").write_text(json.dumps({"passed": False, "error": repr(error)}) + "\n")
        raise
