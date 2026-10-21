# SPDX-License-Identifier: Apache-2.0
"""TND服务入参的CPU合同检查；不把描述符通过当作设备数学通过。"""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.native_adapter import bind_native_inputs
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.policy import select_policy
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.service import CSAServiceRuntime
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.service_config import supported_descriptor


@pytest.mark.parametrize(
    "tokens,requests,max_query,length,expected",
    [
        (6, 1, 6, 131078, True),
        (5, 1, 5, 131077, True),
        (4, 1, 4, 131076, False),
        (6, 2, 6, 131078, False),
        (6, 1, 6, 8198, False),
        (6, 1, 6, 262148, False),
        (63, 16, 6, 131078, False),
        (64, 16, 6, 131078, True),
        (65, 16, 6, 131078, True),
        (96, 16, 6, 131078, True),
        (97, 17, 6, 131078, False),
        (64, 16, 5, 131078, False),
        (96, 16, 6, 8198, False),
        (192, 32, 6, 8198, False),
    ],
)
def test_output_balance_capacity_domains(tokens, requests, max_query, length, expected):
    """The adopted ownership change is confined to its validated resource domains."""
    policy = select_policy(
        tokens=tokens, requests=requests, max_query_tokens=max_query, max_seq_len=length
    )
    assert policy.output_work_balance is expected
    assert policy.constexprs()["OUTPUT_WORK_BALANCE"] is expected


@pytest.mark.parametrize(
    "tokens,requests,max_query,length,topk_prefix,half_score_ub",
    [
        (31, 6, 6, 131078, False, False),
        (32, 6, 6, 131078, True, False),
        (41, 12, 6, 131078, True, False),
        (48, 8, 6, 131078, True, False),
        (63, 11, 6, 131078, True, False),
        (64, 11, 6, 131078, False, False),
        (48, 10, 5, 131077, False, False),
        (6, 1, 6, 131078, False, False),
        (168, 28, 6, 131078, False, False),
        (9, 3, 6, 8198, False, True),
        (65, 16, 6, 8198, False, True),
        (96, 16, 6, 8198, False, True),
        (96, 17, 6, 8198, False, False),
        (80, 16, 5, 8197, False, True),
        (80, 17, 5, 8197, False, False),
        (192, 32, 6, 8198, False, False),
        (216, 36, 6, 8198, False, False),
        (96, 16, 6, 32771, False, True),
        (96, 16, 6, 32772, False, False),
    ],
)
def test_borrowed_score_optimization_domains(tokens, requests, max_query, length, topk_prefix, half_score_ub):
    """Cover graph capacities, work upper bounds and inactive resource branches."""
    policy = select_policy(tokens=tokens, requests=requests, max_query_tokens=max_query, max_seq_len=length)
    assert policy.short_topk_prefix is topk_prefix
    assert policy.short_half_score_ub is half_score_ub
    assert policy.constexprs()["SHORT_TOPK_PREFIX"] is topk_prefix
    assert policy.constexprs()["SHORT_HALF_SCORE_UB"] is half_score_ub


@pytest.mark.parametrize(
    "tokens,requests,max_query,length,wide_n,branch_groups",
    [
        (1, 1, 1, 131073, True, 2),
        (6, 1, 6, 131078, True, 2),
        (8, 2, 6, 131078, True, 2),
        (9, 2, 6, 131078, True, 4),
        (23, 4, 6, 131078, True, 4),
        (24, 4, 6, 131078, False, 4),
        (31, 6, 6, 131078, False, 4),
        (32, 6, 6, 131078, True, 4),
        (41, 12, 6, 131078, True, 4),
        (48, 8, 6, 131078, True, 4),
        (63, 11, 6, 131078, True, 4),
        (64, 11, 6, 131078, False, 4),
        (96, 16, 6, 131078, False, 4),
        (6, 1, 6, 8198, False, 4),
        (48, 8, 6, 32771, False, 4),
        (48, 8, 6, 32772, True, 4),
        (192, 32, 6, 8198, False, 4),
    ],
)
def test_qb_width_and_prefix_resource_boundaries(tokens, requests, max_query, length, wide_n, branch_groups):
    """Preserve the measured negative QA domain and the two-Hadamard work bound."""
    policy = select_policy(tokens=tokens, requests=requests, max_query_tokens=max_query, max_seq_len=length)
    assert policy.qb_wide_n is wide_n
    assert policy.prefix_branch_groups == branch_groups
    assert policy.constexprs()["QB_WIDE_N"] is wide_n
    assert policy.constexprs()["PREFIX_BRANCH_GROUPS"] == branch_groups


def native_case():
    """真实混长[2,6,1]描述符与Native共享key/scale页，不分配NPU。"""
    tokens, requests, pages = 9, 3, 4
    hidden = torch.empty((tokens, 4, 4096), dtype=torch.bfloat16)
    positions = torch.tensor([8192, 8193, *range(130, 136), 32767], dtype=torch.int64)
    groups, compact = {}, {}
    layer_name = "model.layers.4.attn"
    for name in ("swa", "compressed", "state", "indexer", "indexer_state"):
        bounds = torch.tensor([0, 2, 8, 9], dtype=torch.int32)
        lengths = torch.tensor([8194, 136, 32768], dtype=torch.int32)
        cos = torch.ones((tokens, 64), dtype=torch.float32)
        req = NS(
            seq_lens=lengths,
            query_start_loc=bounds,
            block_table=torch.zeros((requests, 8), dtype=torch.int32),
            slot_mapping=torch.zeros((tokens, 2), dtype=torch.int32),
            cos={layer_name: cos},
            sin={layer_name: torch.zeros_like(cos)},
        )
        if name == "indexer":
            storage = torch.empty((pages, 4160), dtype=torch.int8)
            key = storage.as_strided((pages, 32, 1, 128), (4160, 128, 128, 1))
            scale = storage.view(torch.float16).as_strided((pages, 32, 1, 1), (2080, 1, 1, 1), 2048)
            views = (key, scale)
        elif "state" in name:
            width = 512 if name == "indexer_state" else 2048
            storage = torch.empty((pages, 1040 if name == "indexer_state" else 8192), dtype=torch.float32)
            views = (storage.as_strided((pages, 2, 1, width), (storage.shape[1], width, width, 1)),)
        else:
            views = (torch.empty((pages, 32, 1, 512), dtype=torch.bfloat16),)
        groups[name] = (NS(decode=req), views)
        if name in ("compressed", "indexer"):
            # 只校验零复制绑定；具体C4有效关闭行由设备读取Q/L决定。
            compact[name] = (cos[:3], torch.zeros_like(cos[:3]), torch.zeros((3, 2), dtype=torch.int32))
    return dict(
        weights={},
        hidden=hidden,
        positions=positions,
        groups=groups,
        layer_name=layer_name,
        compact_metadata=compact,
        output=torch.empty_like(hidden),
        topk=torch.empty((tokens, 512), dtype=torch.int32),
        topk_scores=torch.empty((tokens, 512), dtype=torch.float32),
    )


def test_mixed_tnd_uses_original_metadata_without_host_reads(monkeypatch):
    case = native_case()

    def forbidden(*args, **kwargs):
        raise AssertionError("生产绑定不能回读Tensor内容")

    for name in ("item", "cpu", "tolist", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    root = bind_native_inputs(**case)
    assert root["x_hc"] is case["hidden"]
    assert root["x_out"] is case["output"]
    assert root["x_hc"].shape[0] == 9
    for name, key in (("compressed", "cmp_query_start_loc"), ("indexer", "idx_query_start_loc")):
        assert root[key] is case["groups"][name][0].decode.query_start_loc
    assert root["cmp_query_start_loc"] is not root["idx_query_start_loc"]
    for name, key in (
        ("swa", "ori_slot_mapping"),
        ("state", "state_slot_mapping"),
        ("indexer_state", "inner_state_slot_mapping"),
    ):
        assert root[key] is case["groups"][name][0].decode.slot_mapping
    for name, key in (
        ("state", "compress_state"),
        ("indexer_state", "inner_compress_state"),
        ("indexer", "idx_native_kv_cache"),
    ):
        assert root[key].data_ptr() == case["groups"][name][1][0].data_ptr()
    for name, prefix in (("compressed", "cmp"), ("indexer", "inner")):
        assert root[prefix + "_freqs_cos"].data_ptr() == case["compact_metadata"][name][0].data_ptr()
    assert root["idx_topk"] is case["topk"]


@pytest.mark.parametrize(
    "fault", ["request_capacity", "bounds_dtype", "slot_capacity", "rope_capacity", "output_dtype"]
)
def test_invalid_native_descriptors_fail_before_launch(fault):
    case = native_case()
    if fault == "request_capacity":
        case["groups"]["state"][0].decode.seq_lens = torch.ones(2, dtype=torch.int32)
    elif fault == "bounds_dtype":
        case["groups"]["compressed"][0].decode.query_start_loc = torch.zeros(4, dtype=torch.int64)
    elif fault == "slot_capacity":
        case["groups"]["swa"][0].decode.slot_mapping = torch.zeros((8, 2), dtype=torch.int32)
    elif fault == "rope_capacity":
        case["groups"]["compressed"][0].decode.cos[case["layer_name"]] = torch.empty((8, 64))
    else:
        case["output"] = case["output"].float()
    with pytest.raises(ValueError):
        bind_native_inputs(**case)


@pytest.mark.parametrize(
    "tokens,requests,actual,max_q,length,expected",
    [
        (9, 3, 9, 6, 8198, True),  # [2,6,1]，T不整除6
        (21, 6, 21, 6, 131078, True),  # [1,2,3,4,5,6]
        (5, 5, 5, 1, 8193, True),  # 均匀S1
        (16, 3, 9, 6, 8198, True),  # token容量与实际token数独立
        (16, 1, 1, 1, 8193, True),  # graph容量可超过有效请求的query上界之和
        (0, 0, 0, 0, 0, False),  # 空调用不能初始化全核等待
        (9, 3, 9, 7, 8198, False),  # 超出首版query上界
        (9, 3, 9, 6, 8191, False),  # 短历史缺口显式回退
        (9, 3, 9, 6, 1048577, False),  # 超出模型候选容量
        (9, 3, 10, 6, 8198, False),  # 容量不足
        (9, 3, 9, 1, 8198, False),  # Native Host字段不一致
    ],
)
def test_host_support_contract(tokens, requests, actual, max_q, length, expected):
    assert (
        supported_descriptor(
            tokens=tokens,
            requests=requests,
            actual_tokens=actual,
            max_query_tokens=max_q,
            max_seq_len=length,
            batch_capacity=40,
        )
        is expected
    )


def service_case(runtime_type=CSAServiceRuntime):
    case = native_case()
    runtime = runtime_type.__new__(runtime_type)
    runtime.prefixes = {name: name for name in case["groups"]}
    runtime.batch_capacity = 40
    runtime.layer_name = case["layer_name"]
    runtime.empty_calls = 0
    for metadata, _ in case["groups"].values():
        metadata.num_actual_tokens = 9
        metadata.num_decodes = 3
        metadata.num_prefills = 0
        req = metadata.decode
        req.max_seqlen_q = 6
        req.max_seq_lens = 32768
        req.num_reqs_actual = 3
        req.ori_win_right = 0
        req.ori_win_left = 127
        req.dspark_swa_indices = None
        req.input_positions = case["positions"]
    context = NS(
        attn_metadata={name: value[0] for name, value in case["groups"].items()},
        is_draft_model=False,
    )
    return runtime, context, case


def test_real_tnd_service_gate_accepts_packed_mixed_requests(monkeypatch):
    runtime, context, case = service_case()

    def forbidden(*args, **kwargs):
        raise AssertionError("服务判定不能从Tensor读回Host")

    for name in ("item", "cpu", "tolist", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    assert runtime.eligible(context, case["hidden"], case["positions"])


def test_service_accepts_native_rope_proxy(monkeypatch):
    from vllm_ascend.ops.rope_dsv4 import _ROPE_STATE, RopeDataProxy

    runtime, context, case = service_case()
    monkeypatch.setitem(_ROPE_STATE.layer_info, runtime.layer_name, ("tnd_test", ["default"]))
    for metadata in context.attn_metadata.values():
        req = metadata.decode
        data = {
            "tnd_test": {
                "default": (
                    req.cos[runtime.layer_name],
                    req.sin[runtime.layer_name],
                )
            }
        }
        req.cos = RopeDataProxy(data, is_cos=True)
        req.sin = RopeDataProxy(data, is_cos=False)
    assert runtime.eligible(context, case["hidden"], case["positions"])


def test_padding_binds_native_capacity_behind_short_decode_views(monkeypatch):
    runtime, context, case = service_case()
    roots = {}
    for name in ("swa", "state", "indexer_state"):
        req = case["groups"][name][0].decode
        roots[name] = req.slot_mapping
        req.slot_mapping = req.slot_mapping[:5]
    req = case["groups"]["compressed"][0].decode
    roots["cos"], roots["sin"] = req.cos[runtime.layer_name], req.sin[runtime.layer_name]
    req.cos[runtime.layer_name] = roots["cos"][:5]
    req.sin[runtime.layer_name] = roots["sin"][:5]

    def forbidden(*args, **kwargs):
        raise AssertionError("容量绑定不得回读Tensor值")

    for name in ("item", "cpu", "tolist", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    assert runtime.eligible(context, case["hidden"], case["positions"])
    result = bind_native_inputs(**case)
    for name, field in (
        ("swa", "ori_slot_mapping"),
        ("state", "state_slot_mapping"),
        ("indexer_state", "inner_state_slot_mapping"),
        ("cos", "freqs_cos"),
        ("sin", "freqs_sin"),
    ):
        assert result[field].shape == roots[name].shape
        assert result[field].data_ptr() == roots[name].data_ptr()
        assert result[field].untyped_storage().data_ptr() == roots[name].untyped_storage().data_ptr()


@pytest.mark.parametrize("fault", ["prefill", "draft", "capacity", "query", "history", "group_mismatch", "rope"])
def test_tnd_service_gate_rejects_unsupported_native_calls(fault):
    runtime, context, case = service_case()
    main = context.attn_metadata["compressed"]
    if fault == "prefill":
        main.num_prefills = 1
    elif fault == "draft":
        context.is_draft_model = True
    elif fault == "capacity":
        main.num_actual_tokens = 10
    elif fault == "query":
        main.decode.max_seqlen_q = 7
    elif fault == "history":
        for metadata in context.attn_metadata.values():
            metadata.decode.max_seq_lens = 8191
    elif fault == "group_mismatch":
        main.num_decodes = 2
    else:
        main.decode.cos[runtime.layer_name] = torch.empty((8, 64))
    assert not runtime.eligible(context, case["hidden"], case["positions"])


@pytest.mark.parametrize("variant", ["spmd_tnd", "mpmd_tnd"])
def test_custom_op_routes_mixed_tnd_and_reuses_prepared_workspace(monkeypatch, variant):
    """真实custom-op Python边界接入新service；CPU桩只替代设备提交。"""
    import importlib

    from vllm_ascend import memcache_comm_fence
    from vllm_ascend.attention import utils as attention_utils
    from vllm_ascend.ops import dsa, dsv4_csa

    package = "deepseek_v4_flash_csa_spmd_tnd" if variant == "spmd_tnd" else "deepseek_v4_flash_csa_tnd"
    service = importlib.import_module(f"vllm_ascend.ops.pypto.{package}.service")

    runtime, context, case = service_case(service.CSAServiceRuntime)
    events, prepared_calls = [], []
    hadamard = torch.eye(128, dtype=torch.bfloat16)
    context.attn_metadata["indexer"].hadamard = hadamard
    context.additional_kwargs = {}
    by_request = {
        id(context.attn_metadata[name].decode): case["compact_metadata"][name] for name in ("compressed", "indexer")
    }
    wrapper = NS(
        _pto_csa_runtime=runtime,
        dsa_attn=NS(impl=NS(_compute_compressor_metadata=lambda req: by_request[id(req)])),
    )
    context.no_compile_layers = {"layer": wrapper}
    runtime.wrapper = wrapper
    runtime.weights = {}
    runtime._hadamard = hadamard
    runtime._outputs = {}
    runtime._bindings = {}
    runtime.submissions = 0
    runtime.last_binding = None

    class Prepared:
        def __init__(self, native, **kwargs):
            self.native = native
            self.rebinds = 0
            prepared_calls.append(self)

        def rebind(self, native):
            self.native = native
            self.rebinds += 1

        def __call__(self):
            events.append(variant)

    monkeypatch.setattr(service, "PreparedDynamicCSA", Prepared)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(dsv4_csa, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        dsa,
        "_build_kv_cache",
        lambda *args: (
            case["groups"]["compressed"][1][0],
            case["groups"]["swa"][1][0],
            case["groups"]["state"][1][0],
            case["groups"]["indexer_state"][1][0],
            *case["groups"]["indexer"][1],
        ),
    )
    monkeypatch.setattr(attention_utils, "wait_for_kv_layer_from_connector", lambda *args: events.append("wait"))
    monkeypatch.setattr(attention_utils, "notify_kv_cache_written", lambda *args: events.append("notify"))
    monkeypatch.setattr(attention_utils, "maybe_save_kv_layer_to_connector", lambda *args: events.append("save"))
    monkeypatch.setattr(memcache_comm_fence, "record_attention_compute_start", lambda: events.append("start"))
    monkeypatch.setattr(dsv4_csa, "_native_attention_half", lambda *args: events.append("native"))
    dsv4_csa.dsv4_csa_forward(case["hidden"], case["positions"], case["output"], "layer")
    # capture的output地址可能不同；描述符相同时沿用工作区。
    second_output = torch.empty_like(case["output"])
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    dsv4_csa.dsv4_csa_forward(case["hidden"], case["positions"], second_output, "layer")
    assert events == ["wait", "start", variant, "notify", "save"] * 2
    assert runtime.submissions == 2 and len(prepared_calls) == 1
    assert prepared_calls[0].rebinds == 1
    assert prepared_calls[0].native["x_out"] is second_output
    assert prepared_calls[0].native["cmp_query_start_loc"] is context.attn_metadata["compressed"].decode.query_start_loc

    # 超出支持范围必须走Native，不能把回退算作SPMD提交。
    context.attn_metadata["compressed"].num_prefills = 1
    dsv4_csa.dsv4_csa_forward(case["hidden"], case["positions"], second_output, "layer")
    assert events[-1] == "native" and runtime.submissions == 2


@pytest.mark.parametrize("zero_capacity", [False, True])
def test_empty_call_skips_spmd_and_native_cache_work(monkeypatch, zero_capacity):
    from vllm_ascend.ops import dsa, dsv4_csa

    runtime, context, case = service_case()
    for item in context.attn_metadata.values():
        item.num_actual_tokens = 0
    hidden = case["hidden"][:0] if zero_capacity else torch.ones_like(case["hidden"])
    positions = case["positions"][: hidden.shape[0]]
    output = torch.zeros_like(hidden)
    context.no_compile_layers = {"layer": NS(_pto_csa_runtime=runtime)}
    monkeypatch.setattr(dsv4_csa, "get_forward_context", lambda: context)

    def unexpected(*args, **kwargs):
        raise AssertionError("空调用不能启动Native/CSA或读取KV存储")

    monkeypatch.setattr(dsv4_csa, "_native_attention_half", unexpected)
    monkeypatch.setattr(dsa, "_build_kv_cache", unexpected)
    dsv4_csa.dsv4_csa_forward(hidden, positions, output, "layer")
    assert torch.equal(output, hidden) and runtime.empty_calls == 1


def test_graph_contract_allows_values_but_rejects_changed_roots_and_policy(monkeypatch):
    from vllm_ascend.ops import dsa
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.graph_contract import TNDGraphContract

    runtime, context, case = service_case()
    runtime.wrapper = NS()
    context.attn_metadata["indexer"].hadamard = torch.eye(128, dtype=torch.bfloat16)
    cache = (
        case["groups"]["compressed"][1][0],
        case["groups"]["swa"][1][0],
        case["groups"]["state"][1][0],
        case["groups"]["indexer_state"][1][0],
        *case["groups"]["indexer"][1],
    )
    monkeypatch.setattr(dsa, "_build_kv_cache", lambda *args: cache)
    signature = runtime.graph_signature(context, 9)
    assert signature is not None
    captured = TNDGraphContract(runtime, 9, signature)
    assert captured.allows_replay(context)
    for metadata in context.attn_metadata.values():
        # 内容更新不能导致Host解析Q；正确的位置/有效范围另由设备测试验证。
        metadata.decode.query_start_loc.copy_(torch.tensor([0, 6, 8, 9], dtype=torch.int32))
        metadata.decode.seq_lens.add_(1)
    assert captured.allows_replay(context)
    assert not TNDGraphContract(runtime, 9, None).allows_replay(context)
    original = context.attn_metadata["swa"].decode.block_table
    context.attn_metadata["swa"].decode.block_table = original.clone()
    assert not captured.allows_replay(context)
    context.attn_metadata["swa"].decode.block_table = original
    for metadata in context.attn_metadata.values():
        metadata.decode.max_seq_lens = 131078
    assert not captured.allows_replay(context)


def test_model_graph_contract_mismatch_redispatches_outside_graph(monkeypatch):
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor

    from vllm_ascend.compilation import acl_graph

    descriptor = BatchDescriptor(9)
    context = NS(batch_descriptor=descriptor, cudagraph_runtime_mode=CUDAGraphMode.FULL)
    replay_calls, dispatch_modes = [], []
    entry = acl_graph.ACLGraphEntry(
        batch_descriptor=descriptor,
        aclgraph=NS(replay=lambda: replay_calls.append(True)),
        csa_tnd_contracts={"layer": NS(allows_replay=lambda ctx: False)},
    )
    wrapper = acl_graph.ACLGraphWrapper.__new__(acl_graph.ACLGraphWrapper)
    wrapper.runtime_mode = CUDAGraphMode.FULL
    wrapper.concrete_aclgraph_entries = {descriptor: entry}

    def run(value):
        dispatch_modes.append(context.cudagraph_runtime_mode)
        return value + 1

    wrapper.runnable = run
    monkeypatch.setattr(acl_graph, "get_forward_context", lambda: context)
    assert wrapper(3) == 4
    assert dispatch_modes == [CUDAGraphMode.NONE] and not replay_calls
    assert context.cudagraph_runtime_mode == CUDAGraphMode.FULL


@pytest.mark.parametrize(
    "history,expected_graph",
    [(8192, True), (8185, False), (8186, True), (0, False), (1048575, False)],
)
@pytest.mark.parametrize("variant", ["spmd_tnd", "mpmd_tnd"])
def test_tnd_runner_preserves_framework_mixed_graph_choice(monkeypatch, history, expected_graph, variant):
    from unittest.mock import Mock

    import numpy as np
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor

    from vllm_ascend.worker import model_runner_v1 as module

    monkeypatch.setenv("PTO_CSA_VARIANT", variant)
    monkeypatch.setattr(module, "enable_sp", lambda _: False)
    config = NS(
        model_config=NS(hf_config=NS(architectures=["PyptoCSADeepseekV4ForCausalLM"])),
        parallel_config=NS(data_parallel_size=1),
        observability_config=NS(cudagraph_metrics=False),
    )
    dispatch = Mock(return_value=(CUDAGraphMode.FULL, BatchDescriptor(12)))
    runner = NS(
        _pad_for_sequence_parallelism=lambda n: n,
        input_batch=NS(num_computed_tokens_cpu=np.full(3, history), lora_id_to_lora_request={}),
        speculative_config=NS(method="dspark"),
        uniform_decode_query_len=6,
        model_config=NS(is_encoder_decoder=False),
        vllm_config=config,
        cudagraph_dispatcher=NS(dispatch=dispatch),
    )
    mode, descriptor, *_ = module.NPUModelRunner._determine_batch_execution_and_padding(
        runner,
        9,
        3,
        np.array([2, 6, 1]),
        6,
        False,
    )
    if expected_graph:
        assert mode == CUDAGraphMode.FULL and descriptor.num_tokens == 12
        # Mixed inputs must never be relabeled uniform just to obtain a graph.
        assert not dispatch.call_args.kwargs["uniform_decode"]
    else:
        assert mode == CUDAGraphMode.NONE and descriptor.num_tokens == 9
        dispatch.assert_not_called()
