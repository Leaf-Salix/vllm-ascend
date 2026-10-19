# SPDX-License-Identifier: Apache-2.0
"""短历史Key预取相对当前动态CSA的真实S1/S5回归，不将S补成6，不测试性能。

两侧绑定相同Native输入及正式权重，分别持有工作区/代际，并执行同一组
A→B→A输入及图重放。检查输出、Top-K/score、完整allocation、保护与地址。
参考是旧动态SPMD；不将本次PASS扩写成新增的Native/整网token精度结论。
"""

import importlib
import json
import os
import traceback
from dataclasses import replace
from pathlib import Path


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--query-tokens", type=int, choices=(1, 5), required=True)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    args = parser.parse_args()
    if "TASK_DEVICE" not in os.environ:
        raise RuntimeError("必须通过队列分配设备")
    if args.device != int(os.environ["TASK_DEVICE"]):
        raise RuntimeError("设备参数必须与队列分配一致")
    here = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(stage="initializing", passed=False, performance_measured=False,
                  batch=args.batch, history=args.history, query_tokens=args.query_tokens,
                  reference="旧正式动态SPMD，同形状同输入精确回归", rounds=[])

    def save():
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("stage", "passed")}), flush=True)

    try:
        import dsv4_csa_env

        dsv4_csa_env.activate()
        import pypto.torch
        import torch
        import torch_npu
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, make_layer, restore
        from dsv4_csa_validation import compare_tensor
        from pypto import passes
        from pypto.runtime import CompileOptions
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        from compile_diagnostics import compilation_probe
        from query_inputs import bind_native_inputs
        from source_loader import load_package
        from compiler_context import record_invocations
        record_invocations(output / "COMPILER_INVOCATIONS.jsonl")

        current_platform.pre_register_and_update()
        device_id = int(os.environ["TASK_DEVICE"])
        device = torch.device(f"npu:{device_id}")
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        from vllm_ascend.utils import enable_custom_op

        assert enable_custom_op(), "fixture需要既有自定义算子包"
        config = EngineArgs(
            model="/data/model/DeepSeek-V4-Flash-0731-w8a8", tokenizer_mode="deepseek_v4",
            trust_remote_code=True, dtype="bfloat16", quantization="ascend", tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128}, max_model_len=max(16384, args.history + 128),
            max_num_seqs=40, max_num_batched_tokens=256, block_size=32, enable_prefix_caching=False,
            enforce_eager=True,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa import native_adapter as adapter
        from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs

        modules = {}
        for role in ("reference", "candidate"):
            package = load_package(here / ("source_" + role), "query_" + role)
            modules[role] = importlib.import_module(package.name + ".binding")
        pypto.torch.init(device=device_id, platform="a2a3", runtime="tensormap_and_ringbuffer",
                         **ring_sizing_kwargs())
        options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        with native_session(config, device_id), torch.inference_mode():
            layer, report["weights"] = make_layer(config, Path(config.model_config.model), device, 4)
            fixture = make_fixture(config, layer.self_attn, args.batch, args.history, 1024, device,
                                   query_tokens=args.query_tokens)
            scale = fixture["groups"]["indexer"]["views"][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            fixture["groups"]["indexer"]["initial"] = fixture["groups"]["indexer"]["allocation"].cpu()
            hidden_a = fixture["hidden"].clone()
            hidden_b = (hidden_a.float() * 0.875).to(hidden_a.dtype)
            result_tensor = torch.empty_like(hidden_a)
            root = bind_native_inputs(adapter, layer, fixture, result_tensor, torch)
            allocations = {role: {} for role in modules}

            def allocator(role):
                def allocate(name, shape, dtype, device):
                    count = 1
                    for size in shape:
                        count *= size
                    guard = 512 // torch.empty((), dtype=dtype).element_size()
                    storage = torch.full((count + guard * 2,), 37, dtype=dtype, device=device)
                    allocations[role][name] = storage, guard
                    return storage[guard:-guard].view(shape)
                return allocate

            entries = {}
            for role, module in modules.items():
                policy = module.select_policy(tokens=fixture["tokens"], requests=args.batch,
                                              max_query_tokens=args.query_tokens,
                                              max_seq_len=args.history + args.query_tokens)
                if role == "candidate":
                    policy = replace(policy, prefetch=1)
                entries[role] = module.PreparedDynamicCSA(
                    root, max_query_tokens=args.query_tokens, max_seq_len=args.history + args.query_tokens,
                    allocator=allocator(role), policy=policy)
            report["policies"] = {role: entry.policy.constexprs() for role, entry in entries.items()}
            report["sources"] = {role: str(here / ("source_" + role)) for role in entries}
            addresses = {role: {name: value.data_ptr() for name, value in entry.buffers.items()}
                         for role, entry in entries.items()}

            def prepare(role, label):
                restore(fixture)
                fixture["hidden"].copy_(hidden_a if label == "A" else hidden_b)
                result_tensor.fill_(float("nan"))
                root["idx_topk"].fill_(-777)
                root["idx_topk_scores"].fill_(float("nan"))
                for name, value in entries[role].allocations.items():
                    if name not in ("signal", "ffts"):
                        value.fill_(37)

            def collect():
                values = collect_state(fixture, result_tensor, root["idx_topk"])
                values["scores"] = root["idx_topk_scores"].cpu()
                values.update({"allocation." + name: group["allocation"].cpu()
                               for name, group in fixture["groups"].items()})
                return values

            def checked(role, before):
                signals = entries[role].buffers["signal"].cpu().view(60, 72, 128)
                protocol = bool((signals[0, :, 0] == before + 1).all() and (signals[:, :, 1:] == 0).all())
                guards = guard_checks(fixture)
                scratch = {name: bool((base[:g].cpu() == 37).all() and (base[-g:].cpu() == 37).all())
                           for name, (base, g) in allocations[role].items()}
                same = addresses[role] == {name: value.data_ptr() for name, value in entries[role].buffers.items()}
                assert protocol and same and all(scratch.values()) and all(v["status"] == "PASS" for v in guards.values())
                return dict(protocol=True, same_addresses=True, guards=guards, scratch=scratch)

            graphs = {}
            eager = {}
            for role, entry in entries.items():
                report["stage"] = "compile_" + role
                save()
                prepare(role, "A")
                before = entry.buffers["signal"].cpu().view(60, 72, 128)[0, :, 0].clone()
                with compilation_probe(output / ("compile_" + role), wait_backoff=True):
                    entry(config=options)
                torch.npu.synchronize()
                eager[role] = collect()
                checked(role, before)
                before_capture = entry.buffers["signal"].cpu().clone()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    entry(config=options)
                torch.npu.synchronize()
                assert torch.equal(before_capture, entry.buffers["signal"].cpu()), "capture不应推进代际"
                graphs[role] = graph
            initial = {name: compare_tensor(value, eager["reference"][name], 0, 0)
                       for name, value in eager["candidate"].items()}
            assert all(v["status"] == "PASS" for v in initial.values()), "真实query初轮精确回归失败"
            report["eager_comparisons"] = initial
            for step, label in enumerate(("A", "B", "A")):
                values = {}
                checks = {}
                for role in (("reference", "candidate") if step % 2 == 0 else ("candidate", "reference")):
                    prepare(role, label)
                    before = entries[role].buffers["signal"].cpu().view(60, 72, 128)[0, :, 0].clone()
                    graphs[role].replay()
                    torch.npu.synchronize()
                    values[role] = collect()
                    checks[role] = checked(role, before)
                compared = {name: compare_tensor(value, values["reference"][name], 0, 0)
                            for name, value in values["candidate"].items()}
                passed = all(v["status"] == "PASS" for v in compared.values())
                report["rounds"].append(dict(label=label, passed=passed, comparisons=compared, checks=checks))
                save()
                assert passed, "真实query图重放精确回归失败"
            report.update(stage="complete", passed=True)
    except BaseException as error:
        report.update(error=str(error)[:1600], traceback=traceback.format_exc()[-8000:])
    finally:
        save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
