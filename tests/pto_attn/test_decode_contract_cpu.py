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
    pool = torch.zeros(4 * 131072, dtype=torch.uint8)
    main = pool.view(torch.float32).as_strided((4, 8, 1, 2048), (32768, 2048, 2048, 1))
    compressed = pool.view(torch.bfloat16).view(4, 128, 1, 512)
    raw = torch.zeros_like(compressed)
    idx = torch.zeros(4 * 16640, dtype=torch.int8)
    key = idx.as_strided((4, 128, 1, 128), (16640, 128, 128, 1))
    scale = idx.view(torch.float16).as_strided((4, 128, 1, 1), (8320, 1, 1, 1), 8192)
    inner = idx.view(torch.float32).as_strided((4, 8, 1, 512), (4160, 512, 512, 1))
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
    impl._pto_attn_weights = dict.fromkeys(
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
    impl._pto_operator = Mock(side_effect=lambda *args: args[-1].fill_(3))
    monkeypatch.setattr(pto, "_registered", lambda: impl._pto_operator)
    monkeypatch.setattr(
        pto,
        "kernel",
        lambda: (
            NS(
                B=64,
                S=6,
                D=4096,
                VLLM_COMPRESS_STATE_PAGE_ROWS=16,
                MAIN_STATE_DIM=2048,
                VLLM_KV_PAGE_ROWS=128,
                HEAD_DIM=512,
                VLLM_INDEX_PAGE_ROWS=130,
                IDX_HEAD_DIM=128,
            ),
            NS(),
        ),
    )
    rope = (torch.ones(256, 64), torch.zeros(256, 64))
    monkeypatch.setattr(pto, "_native_rope_tables", lambda layer: rope)
    native = Mock(return_value=output)
    monkeypatch.setattr(AscendDSAImpl, "forward", native)
    return NS(impl=impl, hidden=hidden, output=output, cache=cache, metadata=metadata, rope=rope, native=native)


def test_signature_is_identical():
    assert inspect.signature(pto.PyptoDSAImpl.forward) == inspect.signature(AscendDSAImpl.forward)


def test_original_kernel_abi_and_native_storage(call):
    c = call
    assert c.impl.forward("layer", c.hidden, c.cache, c.metadata, False, c.output) is c.output
    assert torch.all(c.output == 3)
    c.native.assert_not_called()
    args = dict(zip(pto.ARG_ORDER, c.impl._pto_operator.call_args.args, strict=True))
    assert len(args) == 40
    for name, expected in (
        ("x_normed", c.hidden),
        ("attn_out", c.output),
        ("compress_state_pages", c.cache[2]),
        ("kv_cache_pages", c.cache[1]),
        ("cmp_kv_pages", c.cache[0]),
        ("inner_index_pages", c.cache[4]),
        ("position_ids", c.metadata[0].decode.input_positions),
        ("kv_seq_lens", c.metadata[0].decode.seq_lens),
        ("freqs_cos", c.rope[0]),
        ("cmp_freqs_cos", c.rope[0]),
    ):
        assert args[name].data_ptr() == expected.data_ptr()
    for name, i in (
        ("cmp_block_table", 0),
        ("compress_state_block_table", 1),
        ("inner_compress_state_block_table", 2),
        ("index_block_table", 3),
        ("ori_block_table", 4),
    ):
        assert args[name].data_ptr() == c.metadata[i].decode.block_table.data_ptr()
    assert torch.equal(args["token_valid"], torch.ones(24, dtype=torch.int32))


@pytest.mark.parametrize("seq", [1, 2, 3, 4, 5, 6])
def test_uniform_query_lengths(call, seq):
    c = call
    tokens = 4 * seq
    c.hidden, c.output = c.hidden[:tokens], c.output[:tokens]
    for m in c.metadata:
        m.num_actual_tokens = m.num_decode_tokens = tokens
        m.decode.input_positions = torch.arange(128, 128 + seq).repeat(4)
        m.decode.query_start_loc_cpu = torch.arange(0, tokens + 1, seq)
    if seq == 1:
        c.impl.vllm_config.speculative_config = None
    assert c.impl.forward("layer", c.hidden, c.cache, c.metadata, False, c.output) is c.output
    c.impl._pto_operator.assert_called_once()
    c.native.assert_not_called()


@pytest.mark.parametrize(
    "reason",
    [
        "profiling",
        "gather",
        "prefill",
        "ragged",
        "dummy",
        "draft",
        "ratio",
        "tp",
        "spec",
        "cache",
        "window",
        "short_lens",
        "position_rows",
        "group_tokens",
        "storage",
    ],
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
        metadata[-1].decode.dspark_swa_indices = torch.ones(1)
    elif reason == "ratio":
        c.impl.compress_ratio = 128
    elif reason == "tp":
        c.impl.vllm_config.parallel_config.tensor_parallel_size = 2
    elif reason == "spec":
        c.impl.vllm_config.speculative_config.method = "mtp"
    elif reason == "cache":
        c.cache = (torch.zeros(4, 32, 1, 512), *c.cache[1:])
    elif reason == "window":
        metadata[-1].decode.ori_win_left = 132
    elif reason == "short_lens":
        metadata[0].decode.seq_lens = metadata[0].decode.seq_lens[:3]
    elif reason == "position_rows":
        metadata[0].decode.input_positions = metadata[0].decode.input_positions[:18]
    elif reason == "group_tokens":
        metadata[2].num_decode_tokens = 18
    elif reason == "storage":
        c.cache = (*c.cache[:3], c.cache[3].clone(), *c.cache[4:])
    assert c.impl.forward("layer", c.hidden, c.cache, metadata, gather, c.output) is c.output
    c.native.assert_called_once_with("layer", c.hidden, c.cache, metadata, gather, c.output)
    c.impl._pto_operator.assert_not_called()


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


def test_padded_output_tail_is_not_written(call):
    output = torch.full((32, 4096), -12.5, dtype=torch.bfloat16)
    assert call.impl.forward("layer", call.hidden, call.cache, call.metadata, False, output) is output
    assert torch.all(output[:24] == 3) and torch.all(output[24:] == -12.5)


def test_postload_refreshes_weights(call, monkeypatch):
    def prepare():
        assert not hasattr(call.impl, "_pto_attn_weights")
        call.impl._pto_attn_weights = {"new": 1}

    monkeypatch.setattr(call.impl, "_prepare_weights", prepare)
    call.impl.process_weights_after_loading(torch.bfloat16)
    assert call.impl._pto_attn_weights == {"new": 1}


def test_incomplete_physical_page_rejected():
    cache = torch.zeros(22).as_strided((2, 2, 3), (16, 3, 1))
    with pytest.raises(pto.NativeLayoutError, match="complete final page"):
        pto._full_page_view(cache, 4, (4,))


@pytest.mark.parametrize("enabled", [False, True])
def test_backend_selects_implementation(monkeypatch, enabled):
    from vllm_ascend import utils

    monkeypatch.setattr(utils, "enable_dsa_cp", lambda: False)
    monkeypatch.setenv("VLLM_ASCEND_PYPTO_DSV4_CSA", str(int(enabled)))
    assert AscendDSABackend.get_impl_cls() is (pto.PyptoDSAImpl if enabled else AscendDSAImpl)
