"""CPU contract tests; neither vLLM initialization nor NPU access is required."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/attention/pto_attn.py"
spec = importlib.util.spec_from_file_location("pto_attn_contract", SOURCE)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def metadata(batch=2):
    tokens = batch * 6

    def req(block):
        return NS(
            storage_block_size=block,
            block_table=torch.ones((batch, 4), dtype=torch.int32),
            input_positions=torch.arange(tokens, dtype=torch.int64),
            seq_lens=torch.full((batch,), 6, dtype=torch.int32),
            start_pos=torch.zeros(batch, dtype=torch.int32),
            query_start_loc=torch.arange(0, tokens + 1, 6, dtype=torch.int32),
            slot_mapping=torch.zeros((tokens, 2), dtype=torch.int32),
            cos={"layer": torch.ones((tokens, 64), dtype=torch.float32)},
            sin={"layer": torch.zeros((tokens, 64), dtype=torch.float32)},
            ori_win_left=127,
            ori_win_right=0,
            dspark_swa_indices=None,
        )

    def group(r):
        return NS(
            req_metadata=r,
            num_prefills=0,
            num_decode_tokens=tokens,
            num_decodes=batch,
            hadamard=torch.eye(128, dtype=torch.bfloat16),
        )

    return NS(
        attention=group(req(128)),
        swa=group(req(128)),
        compressor=NS(state=group(req(8)), cache=group(req(128))),
        indexer=NS(compressor=NS(state=group(req(8)), cache=group(req(128)))),
    )


@pytest.fixture
def case(monkeypatch):
    md = metadata()
    k = NS(
        S=6,
        B=64,
        D=4,
        ROPE_HEAD_DIM=64,
        MAIN_STATE_DIM=2048,
        VLLM_COMPRESS_STATE_PAGE_ROWS=16,
        VLLM_KV_PAGE_ROWS=128,
        HEAD_DIM=512,
        VLLM_INDEX_PAGE_ROWS=130,
        IDX_HEAD_DIM=128,
    )
    monkeypatch.setattr(a, "kernel", lambda: (k, None))
    monkeypatch.setattr(a, "prepare_weights", lambda impl, hadamard: {name: torch.empty(0) for name in a.ARG_ORDER})
    compact = (torch.ones((4, 64)), torch.zeros((4, 64)), torch.zeros((4, 2), dtype=torch.int32))
    inner = (torch.ones((5, 64)), torch.zeros((5, 64)), torch.zeros((5, 2), dtype=torch.int32))
    calls = []

    def compute(req, result):
        calls.append(req)
        return result

    impl = NS(
        compressor=NS(_compute_metadata=lambda req: compute(req, compact)),
        indexer=NS(compressor=NS(_compute_metadata=lambda req: compute(req, inner))),
        compress_ratio=4,
        vllm_config=NS(
            speculative_config=NS(method="dspark", num_speculative_tokens=5), parallel_config=NS(tensor_parallel_size=1)
        ),
    )
    main = torch.zeros(2 * 131072, dtype=torch.uint8)
    idx = torch.zeros(2 * 16640, dtype=torch.uint8)
    state = main.view(torch.float32).view(2, 16, 2048)
    cmp = main.view(torch.bfloat16).view(2, 128, 1, 512)
    ik = idx.view(torch.int8).view(2, 130, 128)
    inner_state = idx.view(torch.float32).view(2, 4160)
    scales = torch.as_strided(idx.view(torch.float16), (2, 128), (8320, 1), 8192)
    caches = (cmp, torch.zeros_like(cmp), state, inner_state, ik, scales)
    hidden = torch.zeros((12, 4), dtype=torch.bfloat16)
    return impl, md, caches, hidden, compact, inner, calls


def test_direct_native_metadata_and_compact_producer(case):
    impl, md, caches, hidden, compact, inner, calls = case
    args, _ = a.build_args(impl, hidden, caches, md, 6, "layer", output=hidden)
    bound = dict(zip(a.ARG_ORDER, args))
    assert calls == [md.compressor.cache.req_metadata, md.indexer.compressor.cache.req_metadata]
    for key, original in (
        ("ori_slot_mapping", md.swa.req_metadata.slot_mapping),
        ("state_slot_mapping", md.compressor.state.req_metadata.slot_mapping),
        ("cmp_slot_mapping", compact[2]),
        ("idx_slot_mapping", inner[2]),
        ("cmp_start_pos", md.compressor.cache.req_metadata.start_pos),
        ("idx_start_pos", md.indexer.compressor.cache.req_metadata.start_pos),
        ("freqs_cos", md.attention.req_metadata.cos["layer"]),
        ("cmp_freqs_cos", compact[0]),
        ("inner_freqs_cos", inner[0]),
    ):
        assert bound[key].data_ptr() == original.data_ptr()
    assert "token_valid" not in bound
    assert bound["cmp_norm_w"].numel() == 0  # Fixture replaces only weight preparation.
    assert bound["compress_state_pages"].data_ptr() == caches[2].data_ptr()


def test_page_descriptors_reused_and_invalidated(case):
    impl, md, caches, hidden, *_ = case
    first, _ = a.build_args(impl, hidden, caches, md, 6, "layer", output=hidden)
    second, _ = a.build_args(impl, hidden, caches, md, 6, "layer", output=hidden)
    i = a.ARG_ORDER.index("kv_cache_pages")
    assert first[i] is second[i]
    changed = (*caches[:1], caches[1].clone(), *caches[2:])
    third, _ = a.build_args(impl, hidden, changed, md, 6, "layer", output=hidden)
    assert third[i] is not first[i]
    assert third[i].data_ptr() == changed[1].data_ptr()


def test_block32_rejected_before_metadata_producer(case):
    impl, md, caches, hidden, _, _, calls = case
    md.swa.req_metadata.storage_block_size = 32
    with pytest.raises(a.NativeLayoutError, match="logical block size"):
        a.build_args(impl, hidden, caches, md, 6, "layer")
    assert calls == []


def test_padding_slots_are_not_rebuilt(case):
    impl, md, caches, hidden, *_ = case
    md.swa.req_metadata.slot_mapping[6:] = -1
    args, _ = a.build_args(impl, hidden, caches, md, 6, "layer", output=hidden)
    slots = args[a.ARG_ORDER.index("ori_slot_mapping")]
    assert torch.equal(slots[6:], torch.full((6, 2), -1, dtype=torch.int32))


@pytest.mark.parametrize("dtype", [torch.int8, torch.float32])
def test_dense_weight_never_dequantizes_or_rounds(dtype):
    with pytest.raises(a.NativeLayoutError, match="retain native"):
        a._dense(NS(weight=torch.ones((4, 8), dtype=dtype)), (8, 4))


def test_dense_transpose_preserves_values():
    w = torch.arange(32).reshape(4, 8).to(torch.bfloat16)
    assert torch.equal(a._dense(NS(weight=w), (8, 4)), w.T)


@pytest.mark.parametrize("flag", ["skip_topk", "use_index_cache"])
def test_declines_native_indexer_reuse(case, flag):
    impl, md, caches, hidden, *_ = case
    setattr(impl.indexer, flag, True)
    assert not a.substitute(impl, "layer", hidden, caches, md, hidden)


def test_declines_prepared_cache(case):
    impl, md, caches, hidden, *_ = case
    assert not a.substitute(impl, "layer", hidden, caches, md, hidden, cache_is_prepared=True)


@pytest.mark.parametrize("field,value", [("ori_win_left", 128), ("ori_win_right", 1), ("dspark_swa_indices", object())])
def test_declines_changed_window(case, field, value):
    impl, md, caches, hidden, *_ = case
    setattr(md.swa.req_metadata, field, value)
    assert not a.substitute(impl, "layer", hidden, caches, md, hidden)


def test_kernel_abi_matches_adapter():
    tree = ast.parse((SOURCE.parent / "pto_kernels/dspark/decode_csa.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_decode_csa_attn_tp1")
    assert tuple(arg.arg for arg in fn.args.args) == a.ARG_ORDER


def test_hot_binding_has_no_device_tensor_derivation():
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_args")
    forbidden = {"to", "gather", "clamp", "div", "item", "cpu", "float"}
    assert not [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden
    ]


@pytest.mark.parametrize("starts", [(0, 1), (2, 3), (126, 127), (1021, 1024)])
def test_kernel_compact_rows_and_padding(starts):
    """Execute the actual scalar addressing code with CPU tile operations.

    This checks addressing only, not generated NPU code or numerical precision.
    The oracle enumerates native compression boundaries request by request.
    """
    tree = ast.parse((SOURCE.parent / "pto_kernels/dspark/decode_csa.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_decode_csa_attn_tp1")
    block = next(n for n in fn.body if isinstance(n, ast.With) and n.items[0].optional_vars.id == "rope_tid")
    code = compile(ast.Module(body=block.body, type_ignores=[]), "kernel_metadata_cpu", "exec")
    t, b, d = 18, 3, 64  # two real requests, then a padded request
    pos = torch.tensor([s + i for s in starts for i in range(6)] + [0] * 6)
    bounds = torch.tensor([0, 6, 12, 12], dtype=torch.int32)
    slots = torch.tensor([[3, i] for i in range(12)] + [[-1, -1]] * 6, dtype=torch.int32)
    expected_rows = {}
    for request, start in enumerate(starts):
        for step in range(6):
            if (start + step + 1) % 4 == 0:
                expected_rows[request * 6 + step] = len(expected_rows)
    n = len(expected_rows)
    compact = torch.arange(1, n + 1).float()[:, None].expand(n, d).contiguous()

    def cast(x, dtype=None, target_type=None, **_):
        dtype = dtype if dtype is not None else target_type
        if dtype == "index":
            return int(x)
        return x.to(dtype) if isinstance(x, torch.Tensor) else int(x)

    fake = NS(
        FP32=torch.float32,
        INT32=torch.int32,
        INDEX="index",
        range=range,
        full=lambda shape, dtype, value: torch.full(shape, value, dtype=dtype),
        arange=lambda start, shape, dtype: torch.arange(start, start + shape[-1], dtype=dtype)[None, :],
        col_expand_mul=torch.mul,
        mul=torch.mul,
        add=torch.add,
        sub=torch.sub,
        cast=cast,
        read=lambda x, idx: x[tuple(idx)].item(),
        write=lambda x, idx, v: x.__setitem__(tuple(idx), v),
    )
    env = dict(
        pl=fake,
        ROPE_HEAD_DIM=d,
        COMPRESS_RATIO=4,
        t_dim=t,
        b_dim=b,
        s_dim=6,
        position_ids=pos,
        ori_slot_mapping=slots,
        cmp_query_start_loc=bounds,
        idx_query_start_loc=bounds,
        cmp_start_pos=torch.tensor([*starts, 0], dtype=torch.int32),
        idx_start_pos=torch.tensor([*starts, 0], dtype=torch.int32),
        cmp_row_offsets=torch.empty(b, dtype=torch.int32),
        idx_row_offsets=torch.empty(b, dtype=torch.int32),
        token_valid=torch.empty(t, dtype=torch.int32),
        positions_i32=torch.empty(t, dtype=torch.int32),
        freqs_cos=torch.full((t, d), 2.0),
        freqs_sin=torch.full((t, d), 3.0),
        cmp_freqs_cos=compact,
        cmp_freqs_sin=compact,
        inner_freqs_cos=compact + 100,
        inner_freqs_sin=compact + 100,
    )
    for name in [
        "idx_cos_il",
        "idx_sin_signed",
        "cmp_cos_il",
        "cmp_sin_signed",
        "inner_cos_il",
        "inner_sin_signed",
        "rope_swap_idx",
    ]:
        env[name] = torch.empty((t, d))
    exec(code, env)
    assert env["token_valid"].tolist() == [1] * 12 + [0] * 6
    assert env["positions_i32"][12:].tolist() == [-1] * 6
    # Execute the actual consumer helper body, with CPU equivalents of tile IO.
    helper_tree = ast.parse((SOURCE.parent / "pto_kernels/dspark/native_rope.py").read_text())
    helper = next(n for n in helper_tree.body if isinstance(n, ast.FunctionDef))
    helper.decorator_list = []
    for arg in helper.args.args:
        arg.annotation = None
    helper.returns = None
    reads = []

    def gather_row(dst, src, dst_at, src_at, shape):
        reads.append(src_at[0])
        result = dst.clone()
        result[dst_at[0] : dst_at[0] + shape[0], :] = src[src_at[0] : src_at[0] + shape[0], :]
        return result

    fake.tile = NS(full=fake.full)
    fake.gather_row = gather_row
    helper_env = dict(pl=fake, TILE_ROWS=16, ROPE_DIM=d, DECODE_SEQ=6)
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "native_consumer_cpu", "exec"), helper_env)
    load = helper_env["load_compact_rope"]
    for begin in range(0, t, 16):
        rows = min(16, t - begin)
        cosine, sine = load(
            compact, compact + 10, env["positions_i32"], env["token_valid"], env["cmp_row_offsets"], begin, rows
        )
        inner_cos, _ = load(
            compact + 100, compact + 110, env["positions_i32"], env["token_valid"], env["idx_row_offsets"], begin, rows
        )
        for local in range(16):
            token = begin + local
            if local < rows and token in expected_rows:
                row = expected_rows[token]
                assert torch.equal(cosine[local], compact[row])
                assert torch.equal(sine[local], compact[row] + 10)
                assert torch.equal(inner_cos[local], compact[row] + 100)
            else:
                assert torch.equal(cosine[local], torch.ones(d))
                assert torch.equal(sine[local], torch.zeros(d))
    assert len(reads) == 4 * len(expected_rows)


def test_removed_gm_adapters_are_not_recreated():
    tree = ast.parse((SOURCE.parent / "pto_kernels/dspark/decode_csa.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_decode_csa_attn_tp1")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert names.isdisjoint(
        {"idx_cos_il", "cmp_cos_il", "cmp_sin_signed", "inner_cos_il", "inner_sin_signed", "cmp_out", "idx_out"}
    )


def test_int8_rejects_asymmetric_offset_before_format_conversion():
    linear = NS(weight=torch.zeros((4, 8), dtype=torch.int8), weight_offset=torch.ones(8))
    with pytest.raises(a.NativeLayoutError, match="symmetric"):
        a._int8(linear, (4, 8), 8)


def test_bias_is_not_silently_dropped():
    linear = NS(weight=torch.zeros((4, 8), dtype=torch.bfloat16), bias=torch.ones(4))
    with pytest.raises(a.NativeLayoutError, match="bias"):
        a._dense(linear, (8, 4))


def test_prepare_weights_preserves_fp32_compressor_norm(monkeypatch):
    k = NS(
        D=4,
        Q_LORA=4,
        HEAD_DIM=4,
        H=2,
        IDX_N_HEADS=2,
        IDX_HEAD_DIM=2,
        O_GROUPS=1,
        O_LORA=2,
        MAIN_OUT_DIM=8,
        INNER_OUT_DIM=4,
    )
    monkeypatch.setattr(a, "kernel", lambda: (k, None))
    monkeypatch.setattr(a, "capture_active", lambda: False)
    monkeypatch.setattr(a, "_to_nd", lambda tensor: tensor)

    def dense(*shape):
        return NS(weight=torch.ones(shape, dtype=torch.bfloat16))

    def quant(shape, scales):
        return NS(weight=torch.ones(shape, dtype=torch.int8), weight_scale=torch.ones(scales))

    def compressor(out, width):
        return NS(
            wkv=dense(out, 4),
            wgate=dense(out, 4),
            ape=torch.ones((4, out)),
            norm=NS(weight=torch.full((width,), 1.0001)),
        )

    impl = NS(
        wq_a=dense(4, 4),
        wq_b=quant((4, 8), 8),
        wkv=dense(4, 4),
        q_norm=dense(4),
        kv_norm=dense(4),
        wo_a=dense(1, 8, 2),
        wo_b=quant((4, 2), 4),
        attn_sink=torch.ones(2),
        compressor=compressor(8, 4),
        indexer=NS(wq_b=quant((4, 4), 4), weights_proj=dense(4, 2), compressor=compressor(4, 2)),
    )
    weights = a.prepare_weights(impl, torch.eye(2, dtype=torch.bfloat16))
    for name, original in (
        ("cmp_norm_w", impl.compressor.norm.weight),
        ("inner_norm_w", impl.indexer.compressor.norm.weight),
    ):
        assert weights[name].dtype == torch.float32
        assert weights[name].data_ptr() == original.data_ptr()
        assert torch.equal(weights[name], original)
        assert not torch.equal(weights[name], original.bfloat16().float())
    assert a.prepare_weights(impl, None) is weights


@pytest.mark.parametrize("inner", [False, True])
def test_native_consumer_rms_rope_math_and_tail(inner):
    """Run the actual consumer body with CPU tile IO against a scalar oracle."""
    kernel_dir = SOURCE.parent / "pto_kernels/dspark"
    filename = "decode_indexer_compressor.py" if inner else "decode_compressor_ratio4.py"
    fn_name = "indexer_compressor_pool_projected_vllm" if inner else "compressor_ratio4_cache_write_vllm"
    hint = "indexer_rmsnorm_rope_vllm" if inner else "rmsnorm_rope_cache_write_vllm"
    tree = ast.parse((kernel_dir / filename).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    block = next(
        n
        for n in fn.body
        if isinstance(n, ast.With)
        and any(k.arg == "name_hint" and k.value.value == hint for k in n.items[0].context_expr.keywords)
    )
    body = block.body
    if not inner:
        end = next(
            i
            for i, n in enumerate(body)
            if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "inner"
        )
        body = body[:end]  # cache addressing has a separate native-slot contract.
    code = compile(ast.Module(body=body, type_ignores=[]), "native_rms_cpu", "exec")
    width, count = (128 if inner else 512), 18
    torch.manual_seed(417)
    pooled = torch.randn(32, width)
    weight = torch.randn(1, width)
    output = torch.full((32, width), float("nan"), dtype=torch.bfloat16 if inner else torch.float32)
    cosine, sine = torch.ones(32, 64), torch.zeros(32, 64)
    # Only compression boundaries consume the random native compact rows.
    for token in [3, 7, 11, 15]:
        cosine[token] = torch.randn(64)
        sine[token] = torch.randn(64)

    def cast(x, dtype=None, target_type=None, **_):
        return x.to(dtype if dtype is not None else target_type)

    def load(x, at, shape, valid_shape=None):
        value = torch.zeros(shape, dtype=x.dtype)
        rows, cols = valid_shape or shape
        value[:rows, :cols] = x[at[0] : at[0] + rows, at[1] : at[1] + cols]
        return value

    def store(value, at, target):
        target[at[0] : at[0] + value.shape[0], at[1] : at[1] + value.shape[1]] = value

    fake = NS(
        FP32=torch.float32,
        INT32=torch.int32,
        BF16=torch.bfloat16,
        MemorySpace=NS(Vec=None),
        range=range,
        unroll=range,
        min=min,
        cast=cast,
        load=load,
        store=store,
        mul=torch.mul,
        add=torch.add,
        sub=torch.sub,
        sqrt=torch.sqrt,
        recip=torch.reciprocal,
        reshape=lambda x, shape: x.reshape(shape),
        col_expand_mul=torch.mul,
        row_expand_mul=torch.mul,
        row_sum=lambda x, tmp: x.sum(-1, keepdim=True),
        transpose=lambda x, axis1, axis2: x.transpose(axis1, axis2),
        create_tile=lambda shape, dtype, **kw: torch.empty(shape, dtype=dtype),
    )
    fake.tile = NS(
        full=lambda shape, dtype, value: torch.full(shape, value, dtype=dtype),
        arange=lambda start, shape, dtype: torch.arange(start, start + shape[-1], dtype=dtype)[None, :],
        gather=lambda src, indices, tmp: torch.take(src, indices.long()),
    )

    def compact_loader(cos, sin, pos, valid, offsets, begin, rows):
        return load(cos, [begin, 0], [16, 64], [rows, 64]), load(sin, [begin, 0], [16, 64], [rows, 64])

    env = dict(
        pl=fake,
        RMS_PAD_TILE=16,
        HEAD_DIM=width,
        HEAD_DIM_INV=1.0 / width,
        HEAD_TILE=64,
        ROPE_HEAD_DIM=64,
        NOPE_HEAD_DIM=width - 64,
        EPS=1e-6,
        tokens=count,
        pooled_kv=pooled,
        norm_w_2d=weight,
        normed_kv=output,
        cos=cosine,
        sin=sine,
        position_ids=None,
        token_valid=None,
        compact_offsets=None,
        load_compact_rope=compact_loader,
    )
    for block_index in [0, 1]:
        fake.tile.get_block_idx = lambda block_index=block_index: block_index
        exec(code, env)
    normed = pooled[:count] / torch.sqrt(pooled[:count].square().mean(-1, keepdim=True) + 1e-6) * weight
    rope = normed[:, -64:]
    swapped = rope.reshape(count, 32, 2).flip(-1).reshape(count, 64)
    expected = normed.clone()
    expected[:, -64:] = rope * cosine[:count] + swapped * sine[:count] * torch.tensor([-1, 1] * 32)
    if inner:
        expected = expected.bfloat16()
    torch.testing.assert_close(
        output[:count], expected, rtol=2e-5 if not inner else 0.008, atol=2e-6 if not inner else 0.008
    )
