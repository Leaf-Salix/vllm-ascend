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
    from dsv4_csa_single_layer import guard_checks, make_layer, restore
    from nal_native_alignment_20261027.ragged_fixture import make_ragged_fixture
    from pypto.runtime.kernel.abi import KernelConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu", choices=("ours", "nalinaly", "candidate"))
    parser.add_argument("--clock", action="store_true")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--native-check", action="store_true")
    parser.add_argument("--batch", type=int, choices=range(1, 41), default=8)
    parser.add_argument("--clock-cpu-label", default="20261102-v2")
    args = parser.parse_args()
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
    report = {"scope": f"single real layer4, synthetic common hidden/history, B{args.batch}/S6/8K", "sources": {}}
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for arm, source in sources.items():
        report["sources"][arm] = json.loads((source.parent / "manifest.json").read_text())["csa_source"]
        name = mount_package(source / package, "vllm_ascend.ops.pypto._paired_" + arm)
        modules[arm] = {
            n: importlib.import_module(name + "." + n) for n in ("entry", "policy", "binding", "native_adapter")
        }
    if args.cpu:
        arm = args.cpu
        policy = modules[arm]["policy"].select_policy(
            tokens=args.batch * 6, requests=args.batch, max_query_tokens=6, max_seq_len=8198
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
            fixture = make_ragged_fixture(
                config, layer.self_attn, args.batch, 8192, 1024, torch.device("npu:0"), [6] * args.batch
            )
            groups = {n: (fixture["metadata"][g["prefix"]], g["views"]) for n, g in fixture["groups"].items()}
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
                    "output": output.cpu().clone(),
                    **{n: g["allocation"].cpu().clone() for n, g in fixture["groups"].items()},
                }

            def check_readonly():
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
                if gain >= 0.01 and sum(paired) >= 24:
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
                        args.root / "candidates/phase-clock-20261102-v2" / arm,
                        "vllm_ascend.ops.pypto._phase_" + arm,
                    )
                    module = importlib.import_module(name + ".entry")
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
