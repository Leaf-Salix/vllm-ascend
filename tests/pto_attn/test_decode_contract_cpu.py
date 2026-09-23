"""Native forward contract, producer ownership, and zero-copy cache binding."""

import inspect
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.ops  # noqa: F401 -- initialize native operator registry before backend imports
from vllm_ascend.attention import pto_attn as pto
from vllm_ascend.attention.dsa_v1 import AscendDSABackend, AscendDSAImpl


@pytest.fixture
def call(monkeypatch):
    impl = object.__new__(pto.PyptoDSAImpl)
    impl.compress_ratio = 4
    impl.skip_topk = impl.use_index_cache = False
    impl.vllm_config = NS(
        parallel_config=NS(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        speculative_config=NS(method="dspark", num_speculative_tokens=5),
        kv_transfer_config=None,
        lora_config=None,
        model_config=NS(enforce_eager=True),
        scheduler_config=NS(max_num_seqs=4),
    )
    monkeypatch.setattr(pto, "get_ascend_device_type", lambda: pto.AscendDeviceType.A3)
    monkeypatch.setattr(pto, "olora_tp_enable", lambda: False)
    monkeypatch.setattr(pto, "oproj_tp_enable", lambda: False)
    monkeypatch.setattr(pto, "get_forward_context", lambda: NS(is_draft_model=False))
    monkeypatch.setattr(pto, "wait_for_kv_layer_from_connector", Mock())
    monkeypatch.setattr(pto, "maybe_save_kv_layer_to_connector", Mock())
    monkeypatch.setattr(pto, "notify_kv_cache_written", Mock())
    monkeypatch.setattr(pto, "record_attention_compute_start", Mock())
    hidden = torch.zeros(24, 4096, dtype=torch.bfloat16)
    output = torch.empty_like(hidden)
    pool = torch.zeros(4 * 32768, dtype=torch.uint8)
    main = pool.view(torch.float32).as_strided((4, 2, 1, 2048), (8192, 2048, 2048, 1))
    compressed = pool.view(torch.bfloat16).view(4, 32, 1, 512)
    raw = torch.zeros_like(compressed)
    idx = torch.zeros(4 * 4160, dtype=torch.int8)
    key = idx.as_strided((4, 32, 1, 128), (4160, 128, 128, 1))
    scale = idx.view(torch.float16).as_strided((4, 32, 1, 1), (2080, 1, 1, 1), 2048)
    inner = idx.view(torch.float32).as_strided((4, 2, 1, 512), (1040, 512, 512, 1))
    cache = (compressed, raw, main, inner, key, scale)
    hadamard = torch.eye(128)
    metadata = []
    for i in range(5):
        d = NS(
            query_start_loc_cpu=torch.arange(0, 25, 6, dtype=torch.int32),
            query_start_loc=torch.arange(0, 25, 6, dtype=torch.int32),
            seq_lens=torch.full((4,), 134, dtype=torch.int32),
            num_reqs_actual=4,
            ori_win_right=0,
            ori_win_left=127,
            dspark_swa_indices=None,
            block_table=torch.full((4, 8), i, dtype=torch.int32),
            slot_mapping=torch.full((24, 2), i, dtype=torch.int32),
            input_positions=torch.arange(128, 134).repeat(4),
            cos={"layer": torch.ones(24, 64)},
            sin={"layer": torch.zeros(24, 64)},
        )
        metadata.append(
            NS(decode=d, num_prefills=0, num_decodes=4, num_decode_tokens=24, num_actual_tokens=24, hadamard=hadamard)
        )
    impl._pto_hadamard = hadamard
    impl._pto_weights = dict.fromkeys(
        [
            "wq_a",
            "wq_b",
            "wq_b_scale",
            "wkv",
            "gamma_cq",
            "gamma_ckv",
            "cmp_wkv",
            "cmp_wgate",
            "cmp_ape",
            "cmp_norm_w",
            "idx_wq_b",
            "idx_wq_b_scale",
            "weights_proj",
            "hadamard_idx",
            "inner_wkv",
            "inner_wgate",
            "inner_ape",
            "inner_norm_w",
            "attn_sink",
            "wo_a",
            "wo_b",
            "wo_b_scale",
        ],
        torch.zeros(1),
    )
    impl._pto_scores = torch.zeros(24, 512)
    impl._pto_topk = torch.zeros(24, 512, dtype=torch.int32)
    impl._pto_calls = 0
    impl._pto_operator = Mock(side_effect=lambda *args: args[-1].fill_(3))
    compact = [(torch.ones(8, 64), torch.zeros(8, 64), torch.full((8, 2), i, dtype=torch.int32)) for i in (7, 9)]
    impl._compute_compressor_metadata = Mock(side_effect=compact)
    native = Mock(return_value=output)
    monkeypatch.setattr(AscendDSAImpl, "forward", native)
    return NS(impl=impl, hidden=hidden, output=output, cache=cache, metadata=metadata, compact=compact, native=native)


def test_signature_is_identical():
    assert inspect.signature(pto.PyptoDSAImpl.forward) == inspect.signature(AscendDSAImpl.forward)


def test_native_tensors_and_compact_producers_are_used_directly(call):
    c = call
    assert c.impl.forward("layer", c.hidden, c.cache, c.metadata, False, c.output) is c.output
    assert torch.all(c.output == 3)
    c.native.assert_not_called()
    calls = c.impl._compute_compressor_metadata.call_args_list
    assert calls[0].args[0] is c.metadata[0].decode
    assert calls[1].args[0] is c.metadata[3].decode
    args = c.impl._pto_operator.call_args.args
    assert len(args) == 52 and args[0] is c.hidden and args[-1] is c.output
    for arg, expected in (
        (29, c.cache[1]),
        (30, c.cache[0]),
        (34, c.metadata[4].decode.slot_mapping),
        (36, c.compact[0][2]),
        (37, c.compact[1][2]),
        (38, c.metadata[1].decode.slot_mapping),
        (39, c.metadata[2].decode.slot_mapping),
        (40, c.metadata[0].decode.input_positions),
        (42, c.metadata[0].decode.query_start_loc),
        (44, c.metadata[3].decode.query_start_loc),
    ):
        assert args[arg] is expected
    for arg, expected in (
        (17, c.cache[2]),
        (27, c.cache[3]),
        (32, c.cache[4]),
        (9, c.compact[0][0]),
        (11, c.compact[1][0]),
    ):
        assert args[arg].data_ptr() == expected.data_ptr()
    pto.wait_for_kv_layer_from_connector.assert_called_once_with("layer")
    pto.maybe_save_kv_layer_to_connector.assert_called_once_with("layer", list(c.cache))


@pytest.mark.parametrize(
    "reason",
    ["profiling", "gather", "prefill", "ragged", "dummy", "draft", "ratio", "tp", "graph", "spec", "cache", "window"],
)
def test_unsupported_calls_preserve_native_arguments(call, reason):
    c = call
    metadata, gather = c.metadata, False
    if reason == "profiling":
        metadata = None
    elif reason == "gather":
        gather = True
    elif reason == "prefill":
        metadata[0].num_prefills = 1
    elif reason == "ragged":
        metadata[0].decode.query_start_loc_cpu = torch.tensor([0, 5, 12, 18, 24])
    elif reason == "dummy":
        metadata[0].decode.num_reqs_actual = 3
    elif reason == "draft":
        metadata[0].decode.dspark_swa_indices = torch.ones(1)
    elif reason == "ratio":
        c.impl.compress_ratio = 128
    elif reason == "tp":
        c.impl.vllm_config.parallel_config.tensor_parallel_size = 2
    elif reason == "graph":
        c.impl.vllm_config.model_config.enforce_eager = False
    elif reason == "spec":
        c.impl.vllm_config.speculative_config.method = "mtp"
    elif reason == "cache":
        c.cache = (torch.zeros(4, 128, 1, 512), *c.cache[1:])
    elif reason == "window":
        metadata[-1].decode.ori_win_left = 132
    assert c.impl.forward("layer", c.hidden, c.cache, metadata, gather, c.output) is c.output
    c.native.assert_called_once_with("layer", c.hidden, c.cache, metadata, gather, c.output)
    c.impl._pto_operator.assert_not_called()
    c.impl._compute_compressor_metadata.assert_not_called()


def test_kernel_failure_never_falls_back(call):
    call.impl._pto_operator.side_effect = RuntimeError("failed after cache writes")
    with pytest.raises(RuntimeError, match="after cache writes"):
        call.impl.forward("layer", call.hidden, call.cache, call.metadata, False, call.output)
    call.native.assert_not_called()
    pto.notify_kv_cache_written.assert_not_called()
    pto.maybe_save_kv_layer_to_connector.assert_not_called()


def test_fused_lifecycle_order(call):
    order = Mock()
    for name in (
        "wait_for_kv_layer_from_connector",
        "record_attention_compute_start",
        "notify_kv_cache_written",
        "maybe_save_kv_layer_to_connector",
    ):
        order.attach_mock(getattr(pto, name), name)
    order.attach_mock(call.impl._pto_operator, "kernel")
    call.impl.forward("layer", call.hidden, call.cache, call.metadata, False, call.output)
    assert [c[0] for c in order.mock_calls] == [
        "wait_for_kv_layer_from_connector",
        "record_attention_compute_start",
        "kernel",
        "notify_kv_cache_written",
        "maybe_save_kv_layer_to_connector",
    ]


def test_quantized_input_projection_preserves_native_path(call):
    impl = call.impl
    impl.n_local_heads, impl.n_local_groups = 64, 8
    impl.wq_a = NS(weight=torch.empty((4096, 1024), dtype=torch.int8, device="meta"))
    impl.wkv = NS(weight=torch.empty((4096, 512), dtype=torch.int8, device="meta"))
    impl.process_weights_after_loading(torch.bfloat16)
    assert impl._pto_weights is None
    assert impl.forward("layer", call.hidden, call.cache, call.metadata, False, call.output) is call.output
    call.native.assert_called_once()


def test_full_model_loaded_weight_layout(call, monkeypatch):
    import torch_npu

    monkeypatch.setattr(torch_npu, "get_npu_format", lambda tensor: 2)

    def module(shape, dtype=torch.bfloat16, channels=None):
        value = NS(weight=torch.empty(shape, dtype=dtype, device="meta"))
        if channels is not None:
            value.weight_scale = torch.empty(channels, dtype=torch.bfloat16, device="meta")
        return value

    def compressor(width):
        return NS(
            wkv=module((width * 2, 4096)),
            wgate=module((width * 2, 4096)),
            ape=torch.empty((4, width * 2), device="meta"),
            norm=module((width,)),
        )

    impl = call.impl
    impl.n_local_heads, impl.n_local_groups = 64, 8
    impl.wq_a = module((1024, 4096))
    impl.wq_b = module((1024, 32768), torch.int8, 32768)
    impl.wkv = module((512, 4096))
    impl.q_norm, impl.kv_norm = module((1024,)), module((512,))
    impl.compressor = compressor(512)
    impl.indexer = NS(
        compressor=compressor(128), wq_b=module((1024, 8192), torch.int8, 8192), weights_proj=module((64, 4096))
    )
    impl.attn_sink = torch.empty(64, device="meta")
    impl.wo_a, impl.wo_b = module((8, 4096, 1024)), module((8192, 4096), torch.int8, 4096)
    result = impl._prepare_weights(None)
    assert result["wq_a"].shape == (4096, 1024)
    assert result["wkv"].shape == (4096, 512)
    assert result["wo_b"].shape == (4096, 8192)
    assert result["wq_b"].dtype == torch.int8
    assert result["wq_b_scale"].dtype == torch.float32


def test_padded_table_view_aliases_storage():
    owner = torch.arange(32, dtype=torch.int32).view(4, 8)
    table = owner[:, :5]
    view = pto.PyptoDSAImpl._table_view(table)
    assert view.data_ptr() == table.data_ptr() and view.shape == (4, 8)


def test_physical_page_rejects_incomplete_final_page():
    owner = torch.zeros(22)
    view = owner.as_strided((2, 2, 1, 3), (16, 3, 3, 1))
    with pytest.raises(ValueError, match="final page"):
        pto.PyptoDSAImpl._physical_pages(view)


def test_postload_rebuilds_weights_and_clears_runtime(call, monkeypatch):
    c = call
    prepare = Mock(side_effect=[{"new": 1}, {"new": 2}])
    monkeypatch.setattr(c.impl, "_prepare_weights", prepare)
    c.impl.process_weights_after_loading(torch.bfloat16)
    assert c.impl._pto_weights == {"new": 1} and not hasattr(c.impl, "_pto_operator")
    c.impl.process_weights_after_loading(torch.bfloat16)
    assert c.impl._pto_weights == {"new": 2}


@pytest.mark.parametrize("enabled", [False, True])
def test_backend_selects_implementation(monkeypatch, enabled):
    from vllm_ascend import utils

    monkeypatch.setattr(utils, "enable_dsa_cp", lambda: False)
    monkeypatch.setenv("VLLM_ASCEND_PYPTO_DSV4_CSA", str(int(enabled)))
    assert AscendDSABackend.get_impl_cls() is (pto.PyptoDSAImpl if enabled else AscendDSAImpl)
