"""Validate build_args with vLLM-shaped stand-ins.

build_args is the piece with the most ways to be quietly wrong -- a misspelled
attribute, a transposed weight, a table whose width does not match the kernel's
compile-time column count. None of that needs a device to surface.

Set ``TEST_DEVICE=npu:<id>`` to additionally register and execute the assembled
kernel once. The default remains a CPU-only contract test.
"""
import os
import sys
import types

import torch

TEST_DEVICE = os.environ.get("TEST_DEVICE", "cpu")
EXECUTE_NPU = TEST_DEVICE.startswith("npu")
TEST_GRAPH = os.environ.get("TEST_GRAPH", "0") == "1"
TEST_REPEAT = int(os.environ.get("TEST_REPEAT", "1"))
if TEST_REPEAT < 1:
    raise ValueError(f"TEST_REPEAT must be positive, got {TEST_REPEAT}")
if EXECUTE_NPU:
    import torch_npu  # noqa: F401

    torch.npu.set_device(TEST_DEVICE)
    torch.set_default_device("npu")

# Keep the lifecycle/capture fixture deterministic and within the numerical
# range of a trained checkpoint.  Unit-variance weights and unrelated random
# sine/cosine tables can overflow this whole-layer kernel before the property
# under test (repeatability) is reached.
torch.manual_seed(0)

sys.argv = [sys.argv[0], "--tp", "1"]

import vllm_ascend.attention.pto_attn as pa  # noqa: E402

# torch_npu's format cast is a device op; on CPU the identity is the right stand-in.
pa._to_nd = lambda w: w

from vllm_ascend.attention.pto_kernels.dspark import config as C  # noqa: E402
from vllm_ascend.attention.pto_kernels.dspark import decode_csa as K  # noqa: E402

M = C.FLASH
D = M.hidden_size
TARGET_BATCHES = (4, 8, 16, 24, 32, 40)
NREQ = int(os.environ.get("TEST_BATCH", "4"))
HOST_SEQ = int(os.environ.get("TEST_SEQ", "6"))
ROPE_ROWS = 1024
rope_cos = torch.ones(ROPE_ROWS, 64)
rope_sin = torch.zeros(ROPE_ROWS, 64)
pa._native_rope_tables = lambda layer: (rope_cos, rope_sin)
if NREQ not in TARGET_BATCHES:
    raise ValueError(f"TEST_BATCH must be one of {TARGET_BATCHES}, got {NREQ}")
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


class Lin:
    def __init__(self, *shape, dtype=torch.bfloat16, scale=None):
        self.weight = (torch.randn(*shape) * 0.02).to(dtype) if dtype != torch.int8 \
            else torch.randint(-127, 127, shape, dtype=torch.int8)
        if scale is not None:
            self.weight_scale_fp32 = torch.rand(scale) + 0.5


class Norm:
    def __init__(self, n):
        self.weight = torch.ones(n, dtype=torch.bfloat16)


class Cmp:
    def __init__(self, out_dim, head_dim):
        self.wkv, self.wgate = Lin(out_dim, D), Lin(out_dim, D)
        self.ape = torch.randn(
            C.COMPRESS_RATIO if hasattr(C, "COMPRESS_RATIO") else 4, out_dim,
        ) * 0.02
        self.norm = Norm(head_dim)


class Indexer:
    def __init__(self, quantized=True):
        self.head_dim = K.IDX_HEAD_DIM
        self.wq_b = (Lin(M.q_lora_rank, K.IDX_N_HEADS * K.IDX_HEAD_DIM,
                         dtype=torch.int8, scale=K.IDX_N_HEADS * K.IDX_HEAD_DIM)
                     if quantized else Lin(K.IDX_N_HEADS * K.IDX_HEAD_DIM, M.q_lora_rank))
        self.weights_proj = Lin(K.IDX_N_HEADS, D)
        self.compressor = Cmp(K.INNER_OUT_DIM, K.IDX_HEAD_DIM)


class Impl:
    def __init__(self, quantized: bool):
        self.layer_name = "model.layers.2.self_attn.attn"
        if quantized:
            self.wq_a = Lin(D, M.q_lora_rank, dtype=torch.int8, scale=M.q_lora_rank)
            self.wq_b = Lin(M.q_lora_rank, K.H * K.HEAD_DIM, dtype=torch.int8,
                            scale=K.H * K.HEAD_DIM)
            self.wkv = Lin(D, K.HEAD_DIM, dtype=torch.int8, scale=K.HEAD_DIM)
        else:
            # An unquantized checkpoint keeps torch's [out, in].
            self.wq_a = Lin(M.q_lora_rank, D)
            self.wq_b = Lin(K.H * K.HEAD_DIM, M.q_lora_rank)
            self.wkv = Lin(K.HEAD_DIM, D)
        self.q_norm, self.kv_norm = Norm(M.q_lora_rank), Norm(K.HEAD_DIM)
        self.compressor_wkv = Lin(K.MAIN_OUT_DIM, D)
        self.compressor_wgate = Lin(K.MAIN_OUT_DIM, D)
        self.compressor_ape = torch.randn(4, K.MAIN_OUT_DIM) * 0.02
        self.compressor_norm = Norm(K.HEAD_DIM)
        self.indexcom_wkv = Lin(K.INNER_OUT_DIM, D)
        self.indexcom_wgate = Lin(K.INNER_OUT_DIM, D)
        self.indexcom_ape = torch.randn(4, K.INNER_OUT_DIM) * 0.02
        self.indexcom_norm = Norm(K.IDX_HEAD_DIM)
        self.indexer = Indexer(quantized)
        self.inderxer_wq_b = self.indexer.wq_b
        self.weights_proj = self.indexer.weights_proj
        self.attn_sink = torch.zeros(K.H)
        self.wo_a = Lin(K.O_GROUPS, K.O_GROUP_IN, K.O_LORA)
        self.wo_b = Lin(D, K.O_GROUPS * K.O_LORA)


def md(page, ncols, nrows, cos_key, group_index):
    starts = torch.arange(NREQ, dtype=torch.int64) * 7 + 300
    pos = (starts[:, None] + torch.arange(HOST_SEQ)).reshape(-1)
    o = types.SimpleNamespace()
    o.input_positions = pos
    # Every metadata group addresses its own native cache pool, except the
    # state/compressed and inner-state/index pairs that deliberately share the
    # same pool. Block-table page ids start at one because zero is the native
    # graph-padding sentinel; slot mappings below deliberately include valid
    # physical page zero, whose invalid sentinel is -1.
    # State and cache views share an allocator parent, not live physical pages.
    # Giving both roles the same page ids makes FP32 state writes appear as
    # arbitrary BF16 KV (including NaNs) when attention reads the cache view.
    # The native owner assigns disjoint live pages within each shared pool.
    base = 4 * NREQ + 1 if group_index in (1, 2) else 1
    o.block_table = (
        base
        + torch.arange(NREQ, dtype=torch.int32).view(-1, 1) * 4
        + torch.arange(ncols, dtype=torch.int32).view(1, -1) % 4
    )
    o.seq_lens = (starts + HOST_SEQ).to(torch.int32)
    boundary_count = int((((pos + 1) % 4) == 0).sum())
    slot_rows = max(boundary_count, NREQ)
    sm = torch.stack(
        [torch.arange(slot_rows) % nrows, torch.arange(slot_rows) % page],
        1,
    ).to(torch.int32)
    o.slot_mapping = sm
    o.query_start_loc = torch.arange(NREQ + 1, dtype=torch.int32) * HOST_SEQ
    o.query_start_loc_cpu = o.query_start_loc.detach().cpu().clone()
    t = torch.ones(NREQ, 1, 1, 64)
    o.cos = {cos_key: t}
    o.sin = {cos_key: t.clone()}
    o.compress_cos = {cos_key: torch.ones(max(NREQ // 4, 1), 1, 1, 64)}
    o.compress_sin = {cos_key: torch.zeros(max(NREQ // 4, 1), 1, 1, 64)}
    return types.SimpleNamespace(
        decode=o,
        num_prefills=0,
        num_decode_tokens=NREQ * HOST_SEQ,
    )


def strided(nblk, rows, dim, pad):
    raw = torch.zeros(nblk * pad)
    return torch.as_strided(raw, (nblk, rows, dim), (pad, dim, 1))


QUANT = os.environ.get("TEST_QUANT", "0") == "1"
print(f"== checkpoint: {'W8A8' if QUANT else 'BF16'} ==")
impl = Impl(QUANT)
L = impl.layer_name
metas = [md(128, 64, 4096, L, group) for group in range(5)]
PREFIX = "model.layers.2.self_attn"
metadata_by_name = {
    f"{PREFIX}.attn": metas[0],
    f"{PREFIX}.compressor.state_cache": metas[1],
    f"{PREFIX}.indexer.compressor.state_cache": metas[2],
    f"{PREFIX}.indexer.k_cache": metas[3],
    f"{PREFIX}.swa_cache": metas[4],
}
metadata = pa.resolve_native_metadata(PREFIX, metadata_by_name)
check("metadata resolved by native names", metadata.compressed is metas[0])
try:
    pa.resolve_native_metadata(
        PREFIX,
        {**metadata_by_name, f"{PREFIX}.unexpected": object()},
    )
    check("unexpected metadata rejected", False)
except pa.NativeLayoutError:
    check("unexpected metadata rejected", True)
# Native raw-cache slots have one row per submitted token. Compressed/index
# slots are compact boundary rows; build_args must preserve both layouts.
raw_slots = torch.stack(
    [torch.arange(NREQ * HOST_SEQ, dtype=torch.int32) // 128,
     torch.arange(NREQ * HOST_SEQ, dtype=torch.int32) % 128],
    1,
)
metas[4].decode.slot_mapping = raw_slots
def strided_t(nblk, rows, dim, pad, dtype):
    raw = torch.zeros(nblk * pad, dtype=dtype)
    return torch.as_strided(raw, (nblk, rows, 1, dim), (pad, dim, dim, 1))


# Shapes, strides and dtypes as the live probe/code contract records them.  Main
# state and compressed KV share a 131072-byte page pool; raw KV is separate.
# Inner state/index key/index scale share one 16640-byte allocation.
main_state_dim = 2 * K.MAIN_OUT_DIM
inner_state_dim = 2 * K.INNER_OUT_DIM
PAGE_COUNT = max(64, 8 * NREQ + 2)
main_state_parent = torch.zeros(
    PAGE_COUNT, 16, main_state_dim, dtype=torch.float32,
)
main_state = torch.as_strided(
    main_state_parent,
    (PAGE_COUNT, 8, 1, main_state_dim),
    (32768, main_state_dim, main_state_dim, 1),
    0,
)
raw_parent = torch.zeros(
    PAGE_COUNT, 128, 1, K.HEAD_DIM, dtype=torch.bfloat16,
)
cmp_parent = main_state_parent.view(torch.bfloat16).view(
    PAGE_COUNT, 128, 1, K.HEAD_DIM,
)
index_parent = torch.zeros(PAGE_COUNT * 16640, dtype=torch.int8)
inner_state = torch.as_strided(
    index_parent.view(torch.float32),
    (PAGE_COUNT, 8, 1, inner_state_dim),
    (4160, inner_state_dim, inner_state_dim, 1),
    0,
)
index_key = torch.as_strided(
    index_parent, (PAGE_COUNT, 128, 1, K.IDX_HEAD_DIM),
    (16640, K.IDX_HEAD_DIM, K.IDX_HEAD_DIM, 1), 0,
)
index_scale = torch.as_strided(
    index_parent.view(torch.float16), (PAGE_COUNT, 128, 1, 1),
    (8320, 1, 1, 1), 8192,
)
kvc = (
    cmp_parent,
    raw_parent,
    main_state,
    inner_state,
    index_key,
    index_scale,
)
hs = (torch.randn(NREQ * HOST_SEQ, D) * 0.02).to(torch.bfloat16)
output = torch.empty_like(hs)

print("== build_args ==")
check("TP1 specialization", K.TP_SIZE == 1, str(K.TP_SIZE))
check("S=6 specialization", K.S == 6, str(K.S))
check("B=64 capacity", K.B == 64, str(K.B))
check("T=384 capacity", K.T == 384 and K.T_PAD == 384, f"{K.T}/{K.T_PAD}")
check(
    "target runtime batches fit",
    all(
        batch <= K.B
        and batch * K.S <= K.T
        and (batch * K.S) % 4 == 0
        for batch in TARGET_BATCHES
    ),
    str([(batch, batch * K.S) for batch in TARGET_BATCHES]),
)
try:
    args, plan = pa.build_args(impl, hs, kvc, metadata, L, output=output)
    check("returned native ABI", len(args) == len(pa.ARG_ORDER), str(len(args)))
except Exception:
    import traceback
    traceback.print_exc()
    check("build_args ran", False)
    print("\nFAILED")
    sys.exit(1)

print("== shapes vs the kernel signature ==")
# Runtime rows follow the payload, not the kernel's maximum S capacity.
RT = NREQ * HOST_SEQ
want = {
    "x_normed": [RT, D], "attn_out": [RT, D],
    "freqs_cos": [ROPE_ROWS, 64], "cmp_freqs_cos": [ROPE_ROWS, 64],
    "position_ids": [RT], "query_start_loc": [NREQ + 1],
    "ori_slot_mapping": [RT, 2],
    "cmp_slot_mapping": [metas[0].decode.slot_mapping.shape[0], 2],
    "index_slot_mapping": [metas[3].decode.slot_mapping.shape[0], 2],
    "compress_state_pages": [PAGE_COUNT, 16, main_state_dim],
    "kv_cache_pages": [PAGE_COUNT, 128, 1, K.HEAD_DIM],
    "cmp_kv_pages": [PAGE_COUNT, 128, 1, K.HEAD_DIM],
    "inner_index_pages": [PAGE_COUNT, 130, K.IDX_HEAD_DIM],
    "compress_state_block_table": [NREQ, 64],
    "inner_compress_state_block_table": [NREQ, 64],
    "ori_block_table": [NREQ, 64],
    "cmp_block_table": [NREQ, 64],
    "index_block_table": [NREQ, 64],
    "wo_a": [K.O_GROUPS, K.O_LORA, K.O_GROUP_IN],
    "weights_proj": [D, K.IDX_N_HEADS],
    "hadamard_idx": [K.IDX_HEAD_DIM, K.IDX_HEAD_DIM],
    # The three INT8 slots do not share a scale axis.
    "wq_b": [M.q_lora_rank, K.H * K.HEAD_DIM], "wq_b_scale": [K.H * K.HEAD_DIM],
    "idx_wq_b": [M.q_lora_rank, K.IDX_N_HEADS * K.IDX_HEAD_DIM],
    "idx_wq_b_scale": [K.IDX_N_HEADS * K.IDX_HEAD_DIM],
    "wo_b": [D, K.O_GROUPS * K.O_LORA], "wo_b_scale": [D],
    "wq_a": [D, M.q_lora_rank], "wkv": [D, K.HEAD_DIM],
    "cmp_ape": [4, K.MAIN_OUT_DIM], "inner_ape": [4, K.INNER_OUT_DIM],
    "attn_sink": [K.H],
}
by = dict(zip(pa.ARG_ORDER, args))
for n, w in want.items():
    got = list(by[n].shape)
    check(n, got == w, f"{got} want {w}")

print("== dtypes the signature fixes ==")
check(
    "inner_index_pages is INT8",
    by["inner_index_pages"].dtype == torch.int8,
    str(by["inner_index_pages"].dtype),
)
check(
    "shared page aliases allocator parent",
    by["inner_index_pages"].data_ptr() == inner_state.data_ptr(),
)
check("raw KV aliases its parent",
      by["kv_cache_pages"].data_ptr() == raw_parent.data_ptr())
check("compressed KV aliases its parent",
      by["cmp_kv_pages"].data_ptr() == cmp_parent.data_ptr())
check("main state aliases its native FP32 parent",
      by["compress_state_pages"].data_ptr() == main_state.data_ptr())
check("main-state/compressed-KV exact byte span is exposed twice",
      by["compress_state_pages"].data_ptr() == by["cmp_kv_pages"].data_ptr()
      and by["compress_state_pages"].numel()
      * by["compress_state_pages"].element_size()
      == by["cmp_kv_pages"].numel()
      * by["cmp_kv_pages"].element_size())
check("input rows alias vLLM", by["x_normed"].data_ptr() == hs.data_ptr())
check("output rows alias vLLM", by["attn_out"].data_ptr() == output.data_ptr())
check("positions alias vLLM INT64", by["position_ids"].data_ptr() == metas[0].decode.input_positions.data_ptr()
      and by["position_ids"].dtype == torch.int64)
check("query starts alias vLLM", by["query_start_loc"].data_ptr()
      == metas[0].decode.query_start_loc.data_ptr())
check("raw slots alias vLLM", by["ori_slot_mapping"].data_ptr()
      == metas[4].decode.slot_mapping.data_ptr())
check("compressed slots alias vLLM", by["cmp_slot_mapping"].data_ptr()
      == metas[0].decode.slot_mapping.data_ptr())
check("index slots alias vLLM", by["index_slot_mapping"].data_ptr()
      == metas[3].decode.slot_mapping.data_ptr())
check("native page zero remains a valid slot",
      by["ori_slot_mapping"][0, 0].item() == 0
      and by["cmp_slot_mapping"][0, 0].item() == 0
      and by["index_slot_mapping"][0, 0].item() == 0)
check("RoPE aliases persistent FP32 table", by["freqs_cos"].data_ptr() == rope_cos.data_ptr()
      and by["freqs_cos"].dtype == torch.float32)
check("compressed RoPE reuses persistent table", by["cmp_freqs_cos"].data_ptr() == rope_cos.data_ptr())

print("== contiguity (the binding rejects anything else) ==")
bad = [n for n, a in by.items() if not a.is_contiguous()]
check("all contiguous", not bad, str(bad))

print("== only native token rows ==")
check("no rectangular repetition", by["x_normed"].shape[0] == hs.shape[0])

print("== unsupported request shape is rejected ==")
metas[0].decode.query_start_loc_cpu[-1] -= 1
try:
    pa.build_args(impl, hs, kvc, metadata, L)
    check("non-uniform S6 rejected", False)
except pa.NativeLayoutError:
    check("non-uniform S6 rejected", True)
metas[0].decode.query_start_loc_cpu[-1] += 1

if EXECUTE_NPU:
    print("== NPU register/execute ==")
    try:
        op = pa._registered()

        mutable_names = (
            "attn_out",
            "compress_state_pages",
            "kv_cache_pages",
            "inner_index_pages",
        )

        def reset_mutable():
            # compress_state_pages and cmp_kv_pages are exact aliases, so one
            # reset covers both native views of that physical allocation.
            for name in mutable_names:
                by[name].zero_()

        def snapshot_mutable():
            return {name: by[name].clone() for name in mutable_names}

        def compare_snapshot(label, expected):
            mismatched = []
            details = []
            for name, reference in expected.items():
                if not torch.equal(by[name], reference):
                    mismatched.append(name)
                    if by[name].is_floating_point():
                        delta = (by[name].float() - reference.float()).abs()
                        details.append(
                            f"{name}:max={delta.max().item():.8g},"
                            f"mean={delta.mean().item():.8g}"
                        )
            check(label, not mismatched, ", ".join(details) or str(mismatched))

        reset_mutable()
        op(*args)
        torch.npu.synchronize()
        check("native ABI kernel executed", True)
        output_finite = torch.isfinite(by["attn_out"])
        bad_per_row = (~output_finite).sum(dim=1).cpu().tolist()
        bad_rows = [
            f"{row}:{count}" for row, count in enumerate(bad_per_row) if count
        ]
        check(
            "native ABI output is finite",
            output_finite.all().item(),
            "bad(row:elements)=" + ",".join(bad_rows[:16]),
        )
        for name in mutable_names[1:]:
            if by[name].is_floating_point():
                check(
                    f"{name} is finite",
                    torch.isfinite(by[name]).all().item(),
                )
        eager_reference = snapshot_mutable()

        for repeat in range(1, TEST_REPEAT):
            reset_mutable()
            op(*args)
            torch.npu.synchronize()
            compare_snapshot(f"eager repeat {repeat} is deterministic", eager_reference)

        if TEST_GRAPH:
            # Registration and specialization warmup have already completed.
            # Capture records only the prepared native launch.
            reset_mutable()
            graph = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(graph):
                op(*args)
            torch.npu.synchronize()

            reset_mutable()
            graph.replay()
            torch.npu.synchronize()
            compare_snapshot("ACLGraph replay matches eager", eager_reference)

            first_replay = by["attn_out"].clone()
            original_input = by["x_normed"].clone()
            by["x_normed"].copy_(original_input * 0.5)
            reset_mutable()
            graph.replay()
            torch.npu.synchronize()
            check(
                "ACLGraph replay follows changed input",
                not torch.equal(by["attn_out"], first_replay),
            )
            by["x_normed"].copy_(original_input)
    except Exception as error:
        check("native ABI kernel executed", False, repr(error))

print()
print("FAILED:" if fails else "ALL PASS", fails or "")
sys.exit(1 if fails else 0)
