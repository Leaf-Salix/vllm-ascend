# SPDX-License-Identifier: Apache-2.0
"""固定真 TND 输入的 SPMD / MPMD 单卡对照；只使用完整 L1。

CPU 模式只验证源码依赖和编译，不初始化 NPU。设备模式必须由 task-submit
分配设备；fixture/权重恢复、校验与 D2H 全部在测量区间之外。
"""

import argparse
import ast
import copy
import importlib
import importlib.util
import inspect
import json
import os
import statistics
import sys
import traceback
import types
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def mount_package(source, name):
    importlib.import_module("vllm_ascend.ops.pypto")
    if name in sys.modules:
        raise RuntimeError(f"包已加载，不能中途替换：{name}")
    spec = importlib.util.spec_from_file_location(
        name, source / "__init__.py", submodule_search_locations=[str(source)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return name


def check_case(case):
    b, bc = case["active_requests"], case["request_capacity"]
    q, h = case["query_lengths"], case["history_lengths"]
    assert len(q) == len(h) == b and bc >= b
    assert all(1 <= n <= 6 for n in q) and all(n >= 0 for n in h)
    assert sum(q) == case["valid_tokens"] <= case["token_capacity"] <= 6 * bc
    assert len(case["query_start_loc"]) == bc + 1
    assert case["query_start_loc"][-1] == case["token_capacity"]
    assert case["seq_lens"] == [a + n for a, n in zip(h, q)] + [0] * (bc - b)


def select_case(manifest, name, boundary_query_length=None):
    """主八档原样返回；边界仅来自已冻结的b1_boundaries，不改验收输入。"""
    case = copy.deepcopy(next(c for c in manifest["cases"] if c["name"] == name))
    if boundary_query_length is None:
        return case
    if name != "h131072_b1":
        raise ValueError("固定S1～S6边界仅对应128K/B1，不能修改其它主性能场景")
    boundary = next(c for c in manifest["b1_boundaries"] if c["query_lengths"] == [boundary_query_length])
    case.update(copy.deepcopy(boundary))
    assert case["request_capacity"] == 1 and case["token_capacity"] == boundary_query_length
    case.update(
        name=f"{name}_s{boundary_query_length}",
        valid_tokens=boundary_query_length,
        query_start_loc=[0, boundary_query_length],
        seq_lens=[case["history_lengths"][0] + boundary_query_length],
        dummy_query_lengths=[],
    )
    return case


def export_l1(source, destination, side, kernel_config, *, allow_spmd_prologue=False):
    """使用 incore begin/end 字段，不用派发时间，也不累加重叠 task。"""
    import subprocess

    destination.mkdir()
    raw_path = source / "chip_swimlane_records.json"
    raw = json.loads(raw_path.read_text())
    frequency, rows = raw["metadata"]["clock_freq_hz"], raw["aicore_tasks"]
    if raw["chip_swimlane_level"] != 1 or not rows or frequency <= 0:
        raise ValueError("缺少有效 L1 记录")
    if any(len(r) != 6 or r[4] < r[3] or r[3] <= 0 for r in rows):
        raise ValueError("incore 时间戳非法")
    task_ids = {r[1] for r in rows}
    if side == "spmd":
        if allow_spmd_prologue:
            # 内部工作区版先清零signal；完整L1必须把这些初始化task也计入。
            if not any(len({r[0] for r in rows if r[1] == task}) == 72 for task in task_ids):
                raise ValueError("SPMD记录缺少覆盖72核的主体task")
        elif len(rows) != 72 or len(task_ids) != 1:
            raise ValueError("冻结 SPMD 基线不再是完整 MIX SPMD")
    if side.startswith("mpmd") and len(task_ids) <= 1:
        raise ValueError("MPMD 必须包含多个独立调度 task")
    names = {}
    for node in ast.parse(kernel_config.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KERNELS" for t in node.targets):
            for item in node.value.elts:
                values = {ast.literal_eval(k): v for k, v in zip(item.keys, item.values)}
                names[str(ast.literal_eval(values["func_id"]))] = ast.literal_eval(values["name"])
    (destination / raw_path.name).write_bytes(raw_path.read_bytes())
    # L1 的原始记录只有 task id；函数名和依赖必须由同一窗口的 deps.json 解析。
    has_deps = (source / "deps.json").exists()
    if has_deps:
        (destination / "deps.json").write_bytes((source / "deps.json").read_bytes())
    write_json(destination / "names.json", {"callable_id_to_name": names})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "simpler_setup.tools.swimlane_converter",
            str(destination / raw_path.name),
            "--func-names",
            str(destination / "names.json"),
            "-o",
            str(destination / "swimlane.json"),
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.environ["HCA_SIMPLER_ROOT"] + os.pathsep + os.environ.get("PYTHONPATH", ""),
        },
    )
    (destination / "converter.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    return {
        "side": side,
        "incore_span_us": (max(r[4] for r in rows) - min(r[3] for r in rows)) * 1e6 / frequency,
        "records": len(rows),
        "task_groups": len(task_ids),
        "cores": len({r[0] for r in rows}),
        "dependency_capture": has_deps,
        "raw": str(destination / raw_path.name),
        "trace": str(destination / "swimlane.json"),
    }


def main():
    helpers = Path(__file__).with_name("helpers")
    if helpers.is_dir():
        sys.path.insert(0, str(helpers))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--boundary-query-length", type=int, choices=range(1, 7), help="使用已冻结B1边界，排除于八档G")
    parser.add_argument("--state-validation", action="store_true", help="独立同址metadata/连续状态验证，不测性能")
    parser.add_argument("--service-validation", action="store_true", help="真实MPMD模型钩子/custom-op/框架图，不测性能")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control-mpmd", type=Path, help="同进程增加已接受的多 task 控制组")
    parser.add_argument("--mpmd-only", action="store_true", help="只比较已验收MPMD控制组与候选，不重复运行SPMD")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    dep_flags = parser.add_mutually_exclusive_group()
    dep_flags.add_argument("--dep-gen", dest="dep_gen", action="store_true", help="额外采集依赖，仅用于调度诊断")
    dep_flags.add_argument("--no-dep-gen", dest="dep_gen", action="store_false", help="只采完整 L1，默认行为")
    parser.set_defaults(dep_gen=False)
    parser.add_argument("--device", type=int, help="task-submit 自动追加；须与 TASK_DEVICE 一致")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    args = parser.parse_args()
    if args.mpmd_only and (
        not args.control_mpmd or args.baseline_only or args.state_validation or args.service_validation
    ):
        parser.error("--mpmd-only须指定已验收控制组，且不能用于状态、服务或baseline-only验证")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.cases.read_text())
    case = select_case(manifest, args.case, args.boundary_query_length)
    check_case(case)
    if args.service_validation:
        if args.state_validation or args.boundary_query_length or args.control_mpmd or args.baseline_only:
            raise ValueError("服务验证是独立两侧功能模式")
        os.environ["PTO_CSA_VARIANT"] = "mpmd_tnd"
    if args.state_validation:
        if (
            args.case not in ("h131072_b16", "h131072_b28")
            or args.boundary_query_length
            or args.control_mpmd
            or args.baseline_only
        ):
            raise ValueError("状态矩阵只从长B16/B28构造SPMD/MPMD两侧功能场景，不能混入其它对照")
        # 独立功能场景预留连续decode的页表容量；固定八档清单完全不修改。
        case["name"] += "_state_validation"
        case["table_history"] += 128
    max_query_tokens = max(case["query_lengths"])
    report = {
        "case": case,
        "stage": "initializing",
        "passed": False,
        "device_execution": False,
        "reference_commit": manifest["reference_commit"],
        "scope": "full CSA L1",
        "max_query_tokens": max_query_tokens,
        "enable_chip_swimlane": 1,
        "enable_dep_gen": args.dep_gen,
    }
    if args.boundary_query_length is not None:
        report["scope"] = "full CSA L1, fixed B1 boundary excluded from eight-case G"
    if args.state_validation:
        report["scope"] = "same-address metadata and continuous state; excluded from performance matrix"
        report["performance_measured"] = False
    if args.service_validation:
        report["scope"] = "real MPMD model hook/custom-op/ACLGraphWrapper; excluded from performance matrix"
        report["performance_measured"] = False

    def save(stage=None):
        if stage:
            report["stage"] = stage
        write_json(args.output / "result.json", report)
        print(json.dumps({k: report[k] for k in ("stage", "passed", "device_execution")}), flush=True)

    try:
        import dsv4_csa_env

        dsv4_csa_env.activate()
        from pypto import passes
        from pypto.runtime import CompileOptions

        prefix = "vllm_ascend.ops.pypto."
        packages = args.frozen / "vllm_ascend/ops/pypto"
        if args.service_validation:
            # 冻结参考的BSH数学依赖放在私有父命名空间，真实服务仍使用当前框架代码。
            namespace = prefix + "_service_reference"
            parent = types.ModuleType(namespace)
            parent.__path__ = [str(packages)]
            parent.__package__ = namespace
            parent.__spec__ = importlib.util.spec_from_loader(namespace, loader=None, is_package=True)
            sys.modules[namespace] = parent
            mount_package(packages / "deepseek_v4_flash_csa", namespace + ".deepseek_v4_flash_csa")
            baseline_name = mount_package(packages / "deepseek_v4_flash_csa_spmd_tnd", namespace + ".reference")
            candidate_name = mount_package(args.candidate, prefix + "deepseek_v4_flash_csa_tnd")
        else:
            mount_package(packages / "deepseek_v4_flash_csa", prefix + "deepseek_v4_flash_csa")
            baseline_name = mount_package(packages / "deepseek_v4_flash_csa_spmd_tnd", prefix + "_mpmd_tnd_reference")
            candidate_name = mount_package(args.candidate, prefix + "_mpmd_tnd_candidate")
        control_name = (
            mount_package(args.control_mpmd, prefix + "_mpmd_tnd_control") if args.control_mpmd is not None else None
        )
        report["sources"] = {
            "spmd": str(packages / "deepseek_v4_flash_csa_spmd_tnd"),
            "mpmd": str(args.candidate),
            "mpmd_control": str(args.control_mpmd),
        }
        options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        if args.cpu:
            module = importlib.import_module(candidate_name + ".entry")
            policy = importlib.import_module(candidate_name + ".policy").select_policy(
                tokens=case["token_capacity"],
                requests=case["request_capacity"],
                max_query_tokens=max_query_tokens,
                max_seq_len=max(case["seq_lens"]),
            )
            module.decode_csa_tnd._get_dep_graph()
            report["policy"] = policy.constexprs()
            save("source_dependencies_parsed")
            if args.compile:
                module.decode_csa_tnd.compile(**policy.constexprs(), config=options)
            report["passed"] = True
            save("cpu_backend_compiled" if args.compile else "cpu_parse_passed")
            return

        import pypto.torch
        import torch
        import torch_npu
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import make_layer, restore
        from dsv4_csa_validation import compare_tensor
        from scenarios import make_fixture
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        from vllm_ascend.utils import enable_custom_op

        device_id = int(os.environ["TASK_DEVICE"])
        report["device_id"] = device_id
        if args.device is not None and args.device != device_id:
            raise ValueError("命令行设备与队列分配不一致")
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        current_platform.pre_register_and_update()
        if not enable_custom_op():
            raise RuntimeError("Native 自定义算子未注册")
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
        adapter = importlib.import_module(prefix + "deepseek_v4_flash_csa.native_adapter")
        native_bind = importlib.import_module(baseline_name + ".native_adapter").bind_native_inputs
        with native_session(config, device_id), torch.inference_mode():
            layer, report["weights"] = make_layer(config, Path(manifest["model"]), device, manifest["layer"])
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
                name: (fixture["metadata"][group["prefix"]], group["views"])
                for name, group in fixture["groups"].items()
            }
            weights = adapter.prepare_weights(layer.self_attn, groups["indexer"][0].hadamard, layer)
            output = torch.empty_like(fixture["hidden"])
            native = native_bind(
                weights,
                fixture["hidden"],
                fixture["positions"],
                groups,
                layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                compact_metadata=fixture["compact"],
                output=output,
                topk=torch.empty((fixture["tokens"], 512), dtype=torch.int32, device=device),
                topk_scores=torch.empty((fixture["tokens"], 512), dtype=torch.float32, device=device),
            )
            # Assert fixture Host construction matches its recorded contract.
            report["fixture_checks"] = {
                "q": native["idx_query_start_loc"].cpu().tolist() == case["query_start_loc"],
                "l": native["kv_seq_lens"].cpu().tolist() == case["seq_lens"],
                "active_tokens": fixture["actual_tokens"] == case["valid_tokens"],
            }
            if not all(report["fixture_checks"].values()):
                raise AssertionError("Native fixture 与固定清单不一致")

            entries, guards = {}, {}
            variants = [] if args.mpmd_only else [("spmd", baseline_name)]
            if args.mpmd_only:
                report["scope"] = "accepted MPMD control versus candidate, full CSA L1; no new SPMD measurement"
                report["numerical_reference"] = str(args.control_mpmd)
            if control_name is not None:
                variants.append(("mpmd_control", control_name))
            variants.append(("mpmd", candidate_name))
            for side, package in variants:
                if side.startswith("mpmd") and args.baseline_only:
                    continue

                def allocate(name, shape, dtype, device, side=side):
                    count = 1
                    for dim in shape:
                        count *= dim
                    padding = 512 // torch.empty((), dtype=dtype).element_size()
                    storage = torch.full((count + 2 * padding,), 37, dtype=dtype, device=device)
                    guards[side, name] = storage, padding
                    return storage[padding:-padding].view(shape)

                prepared = importlib.import_module(package + ".binding").PreparedDynamicCSA
                allocator_options = (
                    {"allocator": allocate} if "allocator" in inspect.signature(prepared).parameters else {}
                )
                entries[side] = prepared(
                    native,
                    max_query_tokens=max_query_tokens,
                    max_seq_len=max(case["seq_lens"]),
                    **allocator_options,
                )
                report.setdefault("workspace_checks", {})[side] = {
                    "storage": "external_guarded" if allocator_options else "simpler_internal",
                    "external_scratch_guard_count": len(entries[side].allocations),
                    "topk": "explicit_diagnostic_outputs",
                    "scope": "Native输入/cache保护区始终检查；内部scratch不冒充已做外部哨兵检查",
                }
            report["policies"] = {name: entry.policy.constexprs() for name, entry in entries.items()}
            report["descriptors"] = {
                name: {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in entry.buffers.items()}
                for name, entry in entries.items()
            }
            if args.service_validation:
                from service_validation import preflight, prepare

                service_runtime = prepare(layer)
                report["service_preflight"] = preflight(service_runtime, fixture, config)
                save("service_preflight")
                if not report["service_preflight"]["eligible"]:
                    raise AssertionError("真实服务预检查拒绝：详见service_preflight中的Native描述符")
            else:
                pypto.torch.init(
                    device=device_id,
                    platform="a2a3",
                    runtime="tensormap_and_ringbuffer",
                    enable_chip_swimlane=1,
                    enable_dep_gen=args.dep_gen,
                    output_dir=str(args.output / "dfx"),
                )
            save("fixture_ready")

            slot_indices = {}
            for name, group in fixture["groups"].items():
                slots = (
                    fixture["compact"][name][2] if name in fixture["compact"] else groups[name][0].decode.slot_mapping
                )
                pairs = sorted({(p, r) for p, r in slots.cpu().tolist() if p >= 0 and r >= 0})
                slot_indices[name] = ([p for p, _ in pairs], [r for _, r in pairs])

            def snapshot(side):
                state = {
                    "output": output.cpu(),
                    "topk": native["idx_topk"].cpu(),
                    "scores": native["idx_topk_scores"].cpu(),
                }
                checks = {}
                for name, group in fixture["groups"].items():
                    raw = group["allocation"].cpu()
                    checks[name] = bool(((raw == group["initial"]) | group["allowed"]).all())
                    for index, view in enumerate(group["views"]):
                        cpu_view = raw.view(view.dtype).as_strided(view.shape, view.stride(), view.storage_offset())
                        state[f"{name}.{index}"] = cpu_view[slot_indices[name]].clone()
                for name, (value, initial) in fixture["readonly"].items():
                    checks["readonly/" + name] = torch.equal(value.cpu(), initial)
                for (owner, name), (storage, padding) in guards.items():
                    if owner == side:
                        checks["scratch/" + name] = bool(
                            (storage[:padding].cpu() == 37).all() and (storage[-padding:].cpu() == 37).all()
                        )
                return state, checks

            def compare(actual, expected):
                result = {}
                for name, value in actual.items():
                    rule = (
                        manifest["correctness"]["floating_output"]
                        if name == "output"
                        else manifest["correctness"]["topk_scores"]
                        if name == "scores"
                        else manifest["correctness"]["floating_cache_state"]
                    )
                    atol, rtol = (rule["atol"], rule["rtol"]) if value.is_floating_point() else (0, 0)
                    if value.numel():
                        result[name] = compare_tensor(value, expected[name], atol, rtol)
                        result[name]["bit_equal"] = torch.equal(
                            value.view(torch.uint8), expected[name].view(torch.uint8)
                        )
                    else:
                        result[name] = {"status": "PASS", "elements": 0, "bit_equal": True}
                return result

            expected, tables, report["checks"] = None, {}, {}
            build = Path(os.environ["PYPTO_PROG_BUILD_DIR"])
            for side, entry in entries.items():
                save("compile_and_eager_" + side)
                restore(fixture)
                before = set(build.glob("*/kernel_config.py"))
                entry(config=options)
                torch.npu.synchronize()
                report["device_execution"] = True
                created = set(build.glob("*/kernel_config.py")) - before
                if len(created) != 1:
                    raise RuntimeError(f"{side} 需要唯一 root kernel 配置：{created}")
                tables[side] = created.pop()
                state, check = snapshot(side)
                if expected is None:
                    expected = state
                numerical = compare(state, expected)
                report["checks"][side + "_eager"] = {"guards": check, "numerical": numerical}
                save()
                if not all(check.values()) or not all(v["status"] == "PASS" for v in numerical.values()):
                    raise AssertionError(f"{side} eager 正确性/保护区失败")

            if args.service_validation:
                from service_validation import validate

                report["service_validation"] = validate(
                    runtime=service_runtime,
                    layer=layer,
                    config=config,
                    fixture=fixture,
                    output=output,
                    expected=expected,
                    compare=compare,
                    save=save,
                )
                if not report["service_validation"]["passed"]:
                    raise AssertionError("真实MPMD服务/框架图验证失败")
                report["passed"] = True
                save("complete")
                return
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
                    state, check = snapshot(side)
                    numerical = compare(state, expected)
                    report["checks"][side + "_graph"] = {"guards": check, "numerical": numerical}
                    if not all(check.values()) or not all(v["status"] == "PASS" for v in numerical.values()):
                        raise AssertionError(f"{side} 图重放失败")
                if args.state_validation:
                    from state_replay import validate

                    result = validate(
                        config=config,
                        layer=layer,
                        fixture=fixture,
                        root=native,
                        native_bind=native_bind,
                        entries=entries,
                        graphs=graphs,
                        guards=guards,
                        compare=compare,
                        case=case,
                        report=report,
                        save=save,
                    )
                    if not result["passed"]:
                        raise AssertionError("同址metadata或连续状态验证失败")
                    report["passed"] = True
                    save("complete")
                    return
                for sample in range(3):
                    order = list(graphs) if sample % 2 == 0 else list(reversed(graphs))
                    for side in order:
                        restore(fixture)
                        graphs[side].replay()
                        stream.synchronize()
                records = []
                save("paired_l1")
                for sample in range(5):
                    order = list(graphs) if sample % 2 == 0 else list(reversed(graphs))
                    for side in order:
                        restore(fixture)
                        stream.synchronize()
                        pypto.torch.begin_dfx()
                        try:
                            graphs[side].replay()
                        finally:
                            pypto.torch.end_dfx()
                        source = args.output / "dfx"
                        if records:
                            source /= f"window_{len(records)}"
                        records.append(export_l1(source, args.output / f"l1_{sample:02d}_{side}", side, tables[side]))
                # No host comparisons between the final paired windows.
                for side, graph in graphs.items():
                    restore(fixture)
                    graph.replay()
                    stream.synchronize()
                    state, check = snapshot(side)
                    numerical = compare(state, expected)
                    report["checks"][side + "_post_l1"] = {"guards": check, "numerical": numerical}
                    if not all(check.values()) or not all(v["status"] == "PASS" for v in numerical.values()):
                        raise AssertionError(f"{side} 性能采样后状态检查失败")
            report["windows"] = records
            report["statistics_us"] = {}
            for side in entries:
                samples = [r["incore_span_us"] for r in records if r["side"] == side]
                report["statistics_us"][side] = {
                    "min": min(samples),
                    "mean": statistics.mean(samples),
                    "max": max(samples),
                    "samples": samples,
                }
            if "mpmd" in entries and "spmd" in entries:
                report["ratio"] = report["statistics_us"]["mpmd"]["mean"] / report["statistics_us"]["spmd"]["mean"]
            if "mpmd_control" in entries:
                report["candidate_over_control"] = (
                    report["statistics_us"]["mpmd"]["mean"] / report["statistics_us"]["mpmd_control"]["mean"]
                )
            report["passed"] = True
            save("complete")
    except BaseException:
        report["error"] = traceback.format_exc()
        save("failed")
        raise


if __name__ == "__main__":
    main()
