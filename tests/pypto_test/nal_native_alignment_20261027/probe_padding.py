# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind the candidate independently from the frozen Native control.
"""生产55入口的bucket/dummy FULL资格；先满档控制，首差停止。"""

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import pypto.torch
import torch
import torch_npu
from bind_package import bind
from padding_fixture import load_fixture_module, refresh_fixture

source = bind()
sys.path.insert(0, str(Path(os.environ["SOURCE_ROOT"]) / "tests/pypto_test"))
from dsv4_csa_native_case import native_session
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.ops.dsv4_csa import _native_attention_half
from vllm_ascend.utils import enable_custom_op

fixture_module = load_fixture_module()
service_module = importlib.import_module("nal_alignment.deepseek_v4_flash_csa_spmd_tnd.service")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--mixed", action="store_true")
mode.add_argument("--projection", action="store_true")
mode.add_argument("--full-batch", type=int, choices=(1, 40))
parser.add_argument("--query-rows", type=int, choices=range(1, 7), default=6)
options = parser.parse_args()
if options.query_rows != 6 and options.full_batch is None:
    parser.error("--query-rows requires --full-batch")
out = options.output
mixed_requested, projection_requested = options.mixed, options.projection
record = {
    "complete": False,
    "pass": False,
    "scope": "single-layer raw FULL replay and TND service contract",
    "physical_device": int(os.environ["TASK_DEVICE"]),
    "steps": [],
}
payload = {}
start = time.monotonic()
HISTORY = 8192
ACTIVE_REQUESTS = options.full_batch or 4
INITIAL_QUERY_ROWS = options.query_rows
TOKEN_CAPACITY = ACTIVE_REQUESTS * INITIAL_QUERY_ROWS
REQUEST_CAPACITY = options.full_batch or 5
record["full_batch"] = options.full_batch
record["initial_query_rows"] = INITIAL_QUERY_ROWS
EXTERNAL_ORDER = ("swa.0", "state.0", "compressed.0", "indexer_state.0", "indexer.0", "indexer.1", "x_out")


class FirstDifference(Exception):
    pass


def save():
    (out / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def metric(actual, expected):
    actual, expected = actual.cpu().contiguous(), expected.cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if actual.is_floating_point():
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    integer = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[actual.element_size()]
    different = actual.view(integer) != expected.view(integer)
    return {
        "bit_equal": not bool(different.any()),
        "different_elements": int(different.sum()),
        "max_abs": float((actual.double() - expected.double()).abs().max()) if actual.numel() else 0.0,
        "first_coordinates": different.nonzero()[:6].tolist(),
    }


def tensor_inputs(fixture):
    result = {"hidden": fixture["hidden"], "positions": fixture["positions"]}
    layer_name = fixture["groups"]["compressed"]["prefix"]
    for name, group in fixture["groups"].items():
        for owner, item in (
            ("common", group["common"]),
            ("outer", fixture["metadata"][group["prefix"]]),
            ("decode", fixture["metadata"][group["prefix"]].decode),
        ):
            for field, value in vars(item).items():
                if isinstance(value, torch.Tensor):
                    if owner == "decode" and field in ("input_positions", "slot_mapping"):
                        columns = 1 if field == "input_positions" else 2
                        value = service_module.token_capacity_view(value.view(-1, columns), TOKEN_CAPACITY, columns)
                        if columns == 1:
                            value = value.view(-1)
                    result[f"{name}.{owner}.{field}"] = value
                if field in ("cos", "sin") and value is not None:
                    try:
                        rope = value[layer_name]
                    except (KeyError, TypeError, IndexError):
                        continue
                    if isinstance(rope, torch.Tensor):
                        rope = service_module.token_capacity_view(rope, TOKEN_CAPACITY, 64)
                        result[f"{name}.{owner}.{field}.rows"] = rope
    return result


def descriptors(inputs):
    return {
        name: {
            "pointer": tensor.data_ptr(),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
        for name, tensor in inputs.items()
        if tensor.device.type != "cpu"
    }


try:
    stage_reference = Path(
        os.environ.get("PADDING_STAGE_REFERENCE", out.parent / "native-indexpool-t18-observe-v1/result.json")
    )
    previous = json.loads(stage_reference.read_text())
    record["stage_reference"] = str(stage_reference)
    record["stage_source_sha256"] = {
        name: hashlib.sha256((source / "deepseek_v4_flash_csa_spmd_tnd" / name).read_bytes()).hexdigest()
        for name in previous["stage_source_sha256"]
    }
    trim_sources = {
        "main_native_projection_stages.py": "7dd653b5c1d0256a9e0a6ddbbef0064d675a7dee648a8bc9e8281fae8106b433",
        "indexer_native_projection_stages.py": "eb3562260703304a11c1db5ddc8cd0fd9666ff7b6dcf1099666f0ea80d1b8f25",
    }
    trim_expected = {**previous["stage_source_sha256"], **trim_sources}
    if "PADDING_STAGE_REFERENCE" in os.environ:
        # 新候选必须绑定明确通过的同版本分模块记录，不继承旧版本的两文件例外。
        assert previous["complete"] and previous["pass"] and not previous.get("error")
        assert {"output_projection_a_stages.py", "indexer_topk_stages.py", "kv_native_stages.py"}.issubset(
            record["stage_source_sha256"]
        )
        assert record["stage_source_sha256"] == previous["stage_source_sha256"]
    else:
        # 保留旧检查点的复现入口；其历史允许变动范围不扩张。
        assert len(record["stage_source_sha256"]) == 27
        assert previous["pass"] and record["stage_source_sha256"] in (previous["stage_source_sha256"], trim_expected)
    record["tail_zero_projection_fix"] = record["stage_source_sha256"] == trim_expected
    record["harness_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(__file__).with_name("padding_fixture.py"), Path(fixture_module.__file__))
    }
    record["native_source"] = os.environ["SOURCE_ROOT"]
    record["sdk_release"] = os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"]
    current_platform.pre_register_and_update()
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true" and enable_custom_op()
    checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    config = EngineArgs(
        model=str(checkpoint),
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        quantization="ascend",
        hf_overrides={"sliding_window": 128},
        max_model_len=16384,
        max_num_seqs=REQUEST_CAPACITY,
        max_num_batched_tokens=256,
        enable_prefix_caching=False,
        enforce_eager=False,
        block_size=32,
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [TOKEN_CAPACITY],
            "cache_dir": str(out / "vllm-cache"),
        },
        additional_config={
            "ascend_compilation_config": {
                "enable_npugraph_ex": True,
                "enable_static_kernel": True,
                "fuse_norm_quant": True,
                "align_decode_capture_sizes": False,
            },
            "enable_cpu_binding": False,
            "multistream_overlap_shared_expert": True,
            "weight_nz_mode": 2,
            "enable_kv_nz": False,
            "enable_dsa_cp": False,
            "multistream_dsv4_dsa_overlap": True,
        },
    ).create_engine_config()
    with native_session(config, 0), torch.inference_mode():
        layer, details = fixture_module.make_layer(config, checkpoint, torch.device("npu:0"), 4)
        record["weights"] = details
        f = fixture_module.make_fixture(
            config,
            layer.self_attn,
            ACTIVE_REQUESTS,
            HISTORY,
            1024,
            torch.device("npu:0"),
            query_lengths=[INITIAL_QUERY_ROWS] * ACTIVE_REQUESTS,
            token_capacity=TOKEN_CAPACITY,
            request_capacity=REQUEST_CAPACITY,
        )
        wrapper = layer.self_attn.dsa_attn
        wrapper._pto_csa_runtime = None
        output_owner = torch.full(
            (TOKEN_CAPACITY + 2, *f["hidden"].shape[1:]),
            17.0,
            dtype=f["hidden"].dtype,
            device=f["hidden"].device,
        )
        output = output_owner[1:-1]
        for name, value in (("output_before", output_owner[:1]), ("output_after", output_owner[-1:])):
            f["readonly"][name] = (value, value.cpu().clone())
        metadata_inputs = tensor_inputs(f)
        capture_inputs = dict(metadata_inputs)
        captured_descriptors = descriptors(metadata_inputs)
        record["capture_descriptors"] = captured_descriptors
        f["readonly"].update({name: (value, value.cpu().clone()) for name, value in metadata_inputs.items()})
        record["initial_geometry"] = {
            name: {
                "bounds": f["metadata"][group["prefix"]].decode.query_start_loc.cpu().tolist(),
                "lengths": f["metadata"][group["prefix"]].decode.seq_lens.cpu().tolist(),
                "num_reqs_actual": f["metadata"][group["prefix"]].decode.num_reqs_actual,
                "num_decode_tokens": f["metadata"][group["prefix"]].num_decode_tokens,
                "num_decodes": f["metadata"][group["prefix"]].num_decodes,
            }
            for name, group in f["groups"].items()
        }
        expected_bounds = [i * INITIAL_QUERY_ROWS for i in range(ACTIVE_REQUESTS + 1)] + [TOKEN_CAPACITY] * (
            REQUEST_CAPACITY - ACTIVE_REQUESTS
        )
        assert all(
            row["bounds"] == expected_bounds
            and row["lengths"]
            == [HISTORY + INITIAL_QUERY_ROWS] * ACTIVE_REQUESTS + [0] * (REQUEST_CAPACITY - ACTIVE_REQUESTS)
            and row["num_reqs_actual"] == ACTIVE_REQUESTS
            and row["num_decode_tokens"] == TOKEN_CAPACITY
            and row["num_decodes"] == REQUEST_CAPACITY
            for row in record["initial_geometry"].values()
        )
        payload.update(
            hidden=f["hidden"].cpu(),
            positions=f["positions"].cpu(),
            initial={name: group["initial"] for name, group in f["groups"].items()},
            metadata={name: value.cpu().clone() for name, value in metadata_inputs.items()},
        )

        @contextmanager
        def context(capturing=False):
            # Raw NPUGraph does not set vLLM's Python capture marker by itself.
            with set_ascend_forward_context(
                f["metadata"], config, num_tokens=TOKEN_CAPACITY, num_actual_tokens=f["actual_tokens"]
            ):
                ctx = get_forward_context()
                ctx.capturing = capturing
                try:
                    yield ctx
                finally:
                    ctx.capturing = False

        def collect():
            return {
                "x_out": output[: f["actual_tokens"]].cpu().clone(),
                **{
                    f"{name}.{index}": value.cpu().clone()
                    for name, group in f["groups"].items()
                    for index, value in enumerate(group["views"])
                },
            }

        def reset():
            fixture_module.restore(f)
            output.fill_(float("nan"))

        def checked_guards(label):
            checked = fixture_module.guard_checks(f)
            record.setdefault("guards", {})[label] = checked
            save()
            assert all(row["status"] == "PASS" for row in checked.values()), label

        def native_call(capturing=False):
            with context(capturing):
                _native_attention_half(wrapper, f["hidden"], f["positions"], output)

        reset()
        normed_capture = {}

        def capture_norm(_module, _inputs, value):
            if isinstance(value, tuple):
                value = value[0]
            normed_capture["value"] = value.cpu().clone()
            normed_capture["calls"] = normed_capture.get("calls", 0) + 1

        norm_hook = layer.input_layernorm.register_forward_hook(capture_norm) if projection_requested else None
        try:
            native_call()
            torch.npu.synchronize()
        finally:
            if norm_hook is not None:
                norm_hook.remove()
        if projection_requested:
            assert normed_capture["calls"] == 1
        eager = collect()
        payload["native_eager"] = eager
        checked_guards("native_warmup")
        reset()
        native_graph, native_stream = torch.npu.NPUGraph(), torch.npu.Stream()
        native_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(native_graph, stream=native_stream):
            native_call(capturing=True)
        native_states = []
        for repeat in range(2):
            reset()
            native_graph.replay()
            torch.npu.synchronize()
            native_states.append(collect())
            payload[f"native_full_{repeat}"] = native_states[-1]
            checked_guards(f"native_full_{repeat}")
        record["native_full_repeat"] = {
            name: metric(native_states[1][name], native_states[0][name]) for name in EXTERNAL_ORDER
        }
        record["native_full_eager"] = {name: metric(native_states[0][name], eager[name]) for name in EXTERNAL_ORDER}
        save()
        assert all(row["bit_equal"] for row in record["native_full_repeat"].values())
        assert all(row["bit_equal"] for row in record["native_full_eager"].values())
        payload["native_full"] = native_states[1]

        if projection_requested:
            from padding_projection import compare_projection

            compare_projection(
                f, layer, normed_capture["value"], native_states[1], metadata_inputs, out, record, payload, metric, save
            )
            record.update(complete=True, **{"pass": True}, projection_only=True, mixed_replay_validated=False)
            raise SystemExit(0)

        # T240's production live set exceeds the default 256 MiB on scope ring 1.
        # Use the public runtime capacity option; leave all other rings unchanged.
        ring_heap = (256 << 20, 512 << 20, 256 << 20, 256 << 20)
        record["runtime_ring_heap_bytes"] = ring_heap
        save()
        pypto.torch.init(ring_heap=ring_heap)
        runtime = service_module.CSAServiceRuntime(layer.self_attn, None, REQUEST_CAPACITY, layer=layer)
        wrapper._pto_csa_runtime = runtime
        record["runtime_source"] = service_module.__file__

        def pto_call(capturing=False):
            with context(capturing) as ctx:
                assert runtime.eligible(ctx, f["hidden"], f["positions"]), "production service must not fall back"
                before = runtime.submissions
                torch.ops.vllm.dsv4_csa_forward(f["hidden"], f["positions"], output, wrapper.prefix)
                assert runtime.submissions == before + 1
                if capturing:
                    contract = ctx.additional_kwargs["pto_csa_tnd_graph_contracts"][runtime.layer_name]
                    assert contract.signature is not None and contract.allows_replay(ctx)
                    return contract

        reset()
        pto_call()
        torch.npu.synchronize()
        pto_eager = collect()
        payload["pto_eager"] = pto_eager
        assert not runtime.last_binding._diagnostic and "topk_indices" not in runtime.last_binding.buffers
        record["policy"] = runtime.last_binding.policy.constexprs()
        from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.entry import decode_csa_spmd
        from pypto.ir._kernel_compile import kernel_abi_for_program

        abi = kernel_abi_for_program(
            decode_csa_spmd.specialize(**record["policy"]),
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
        )
        record["production_root_tensors"] = len(abi.parameters)
        assert record["production_root_tensors"] == 55
        checked_guards("pto_warmup")
        reset()
        pto_graph, pto_stream = torch.npu.NPUGraph(), torch.npu.Stream()
        pto_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(pto_graph, stream=pto_stream):
            contract = pto_call(capturing=True)
        submitted = runtime.submissions
        step = {"state": f"A_full{TOKEN_CAPACITY}_capacity{REQUEST_CAPACITY}", "comparisons": {}, "full_eager": {}}
        record["steps"].append(step)
        for repeat in range(2):
            reset()
            with context() as ctx:
                assert runtime.eligible(ctx, f["hidden"], f["positions"]) and contract.allows_replay(ctx)
                pto_graph.replay()
            torch.npu.synchronize()
            actual = collect()
            payload[f"pto_full_{repeat}"] = actual
            checked_guards(f"pto_full_{repeat}")
            assert runtime.submissions == submitted, "FULL replay must not re-enter Python"
            assert descriptors(tensor_inputs(f)) == captured_descriptors
            step["full_eager"][str(repeat)] = {name: metric(actual[name], pto_eager[name]) for name in EXTERNAL_ORDER}
            save()
            assert all(row["bit_equal"] for row in step["full_eager"][str(repeat)].values())
            for name in EXTERNAL_ORDER:
                row = metric(actual[name], native_states[1][name])
                step["comparisons"][name] = row
                payload["actual_full"] = actual
                save()
                if not row["bit_equal"]:
                    record["first_difference"] = name
                    raise FirstDifference(name)
        step["pass"] = True
        if mixed_requested:
            initial_hidden = f["hidden"].clone()
            tables = {name: group["common"].block_table_tensor.clone() for name, group in f["groups"].items()}
            for label, queries, hidden in (
                ("B_mixed18_bucket24_dummy6", [1, 5, 6, 6], -initial_hidden),
                ("A_return24_capacity5", [6] * 4, initial_hidden),
            ):
                f["hidden"].copy_(hidden)
                refresh_fixture(f, layer.self_attn, queries, tables, fixture_module)
                current_inputs = tensor_inputs(f)
                step = {
                    "state": label,
                    "actual_tokens": f["actual_tokens"],
                    "physical_tokens": TOKEN_CAPACITY,
                    "queries": queries,
                    "metadata": {
                        name: {
                            "bounds": f["metadata"][group["prefix"]].decode.query_start_loc.cpu().tolist(),
                            "lengths": f["metadata"][group["prefix"]].decode.seq_lens.cpu().tolist(),
                            "num_decode_tokens": f["metadata"][group["prefix"]].num_decode_tokens,
                            "num_reqs_actual": f["metadata"][group["prefix"]].decode.num_reqs_actual,
                        }
                        for name, group in f["groups"].items()
                    },
                    "comparisons": {},
                    "current_descriptors": descriptors(current_inputs),
                    "capture_descriptors": descriptors(capture_inputs),
                }
                record["steps"].append(step)
                payload[label] = {
                    "hidden": f["hidden"].cpu().clone(),
                    "positions": f["positions"].cpu().clone(),
                    "metadata": {name: value.cpu().clone() for name, value in current_inputs.items()},
                }
                save()
                assert step["current_descriptors"] == captured_descriptors
                assert step["capture_descriptors"] == captured_descriptors
                assert all(row["num_decode_tokens"] == sum(queries) for row in step["metadata"].values())
                f["readonly"] = {name: (value, value.cpu().clone()) for name, value in current_inputs.items()}
                for name, value in (("output_before", output_owner[:1]), ("output_after", output_owner[-1:])):
                    f["readonly"][name] = (value, value.cpu().clone())
                with context() as ctx:
                    assert runtime.eligible(ctx, f["hidden"], f["positions"])
                    assert contract.allows_replay(ctx)
                references = []
                for repeat in range(2):
                    reset()
                    native_graph.replay()
                    torch.npu.synchronize()
                    references.append(collect())
                    payload[label][f"native_full_{repeat}"] = references[-1]
                    checked_guards(f"{label}.native_full_{repeat}")
                step["native_full_repeat"] = {
                    name: metric(references[1][name], references[0][name]) for name in EXTERNAL_ORDER
                }
                save()
                assert all(row["bit_equal"] for row in step["native_full_repeat"].values())
                if label.startswith("B_"):
                    assert not torch.equal(references[1]["x_out"], native_states[1]["x_out"][: sum(queries)])
                else:
                    step["return_native_a"] = {
                        name: metric(references[1][name], native_states[1][name]) for name in EXTERNAL_ORDER
                    }
                    save()
                    assert all(row["bit_equal"] for row in step["return_native_a"].values())
                for repeat in range(2):
                    reset()
                    with context() as ctx:
                        assert runtime.eligible(ctx, f["hidden"], f["positions"]) and contract.allows_replay(ctx)
                        pto_graph.replay()
                    torch.npu.synchronize()
                    actual = collect()
                    payload[label][f"pto_full_{repeat}"] = actual
                    checked_guards(f"{label}.pto_full_{repeat}")
                    assert runtime.submissions == submitted
                    assert descriptors(tensor_inputs(f)) == captured_descriptors
                    for name in EXTERNAL_ORDER:
                        row = metric(actual[name], references[1][name])
                        step["comparisons"][f"{repeat}.{name}"] = row
                        save()
                        if not row["bit_equal"]:
                            record["first_difference"] = f"{label}.{name}"
                            raise FirstDifference(record["first_difference"])
                step["pass"] = True
                save()
        record.update(
            complete=True,
            **{"pass": True},
            capacity_control_only=not mixed_requested,
            mixed_replay_validated=mixed_requested,
        )
except FirstDifference:
    record["complete"] = True
except SystemExit as error:
    if error.code:
        raise
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    save()
    if not record["pass"]:
        torch.save(payload, out / "failure.pt")
