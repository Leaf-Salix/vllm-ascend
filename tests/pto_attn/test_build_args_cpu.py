"""Dry-run build_args on CPU with vLLM-shaped stand-ins.

build_args is the piece with the most ways to be quietly wrong -- a misspelled
attribute, a transposed weight, a table whose width does not match the kernel's
compile-time column count. None of that needs a device to surface.
"""
import os, sys, types
import torch

ROOT = "/data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/vllm-ascend-v0.20.2rc1"
sys.path.insert(0, ROOT)
sys.argv = [sys.argv[0], "--tp", "4"]

import vllm_ascend.attention.pto_attn as pa

# torch_npu's format cast is a device op; on CPU the identity is the right stand-in.
pa._to_nd = lambda w: w

from vllm_ascend.attention.pto_kernels.dspark import decode_csa as K
from vllm_ascend.attention.pto_kernels.dspark import config as C

M = C.FLASH
D = M.hidden_size
NREQ, HOST_SEQ = 4, 1
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


class Lin:
    def __init__(self, *shape, dtype=torch.bfloat16, scale=None):
        self.weight = torch.randn(*shape).to(dtype) if dtype != torch.int8 \
            else torch.randint(-127, 127, shape, dtype=torch.int8)
        if scale is not None:
            self.weight_scale_fp32 = torch.rand(scale) + 0.5


class Norm:
    def __init__(self, n):
        self.weight = torch.randn(n).to(torch.bfloat16)


class Cmp:
    def __init__(self, out_dim, head_dim):
        self.wkv, self.wgate = Lin(out_dim, D), Lin(out_dim, D)
        self.ape = torch.randn(C.COMPRESS_RATIO if hasattr(C, "COMPRESS_RATIO") else 4, out_dim)
        self.norm = Norm(head_dim)


class Indexer:
    def __init__(self):
        self.head_dim = K.IDX_HEAD_DIM
        self.wq_b = Lin(M.q_lora_rank, K.IDX_N_HEADS * K.IDX_HEAD_DIM,
                        dtype=torch.int8, scale=K.IDX_N_HEADS * K.IDX_HEAD_DIM)
        self.weights_proj = Lin(K.IDX_N_HEADS, D)
        self.compressor = Cmp(K.INNER_OUT_DIM, K.IDX_HEAD_DIM)


class Impl:
    def __init__(self):
        self.layer_name = "model.layers.2.self_attn.attn"
        self.wq_a = Lin(D, M.q_lora_rank, dtype=torch.int8, scale=M.q_lora_rank)
        self.wq_b = Lin(M.q_lora_rank, K.H * K.HEAD_DIM, dtype=torch.int8,
                        scale=K.H * K.HEAD_DIM)
        self.wkv = Lin(D, K.HEAD_DIM, dtype=torch.int8, scale=K.HEAD_DIM)
        self.q_norm, self.kv_norm = Norm(M.q_lora_rank), Norm(K.HEAD_DIM)
        self.compressor_wkv = Lin(K.MAIN_OUT_DIM, D)
        self.compressor_wgate = Lin(K.MAIN_OUT_DIM, D)
        self.compressor_ape = torch.randn(4, K.MAIN_OUT_DIM)
        self.compressor_norm = Norm(K.HEAD_DIM)
        self.indexcom_wkv = Lin(K.INNER_OUT_DIM, D)
        self.indexcom_wgate = Lin(K.INNER_OUT_DIM, D)
        self.indexcom_ape = torch.randn(4, K.INNER_OUT_DIM)
        self.indexcom_norm = Norm(K.IDX_HEAD_DIM)
        self.indexer = Indexer()
        self.inderxer_wq_b = self.indexer.wq_b
        self.weights_proj = self.indexer.weights_proj
        self.attn_sink = torch.rand(K.H)
        self.wo_a = Lin(K.O_GROUPS, K.O_GROUP_IN, K.O_LORA)
        self.wo_b = Lin(D, K.O_GROUPS * K.O_LORA)


def md(page, ncols, nrows, cos_key):
    pos = torch.arange(NREQ, dtype=torch.int32) * 7 + 300
    o = types.SimpleNamespace()
    o.input_positions = pos
    o.block_table = torch.arange(NREQ * ncols, dtype=torch.int32).view(NREQ, ncols) + 1
    o.seq_lens = pos.to(torch.int32) + 1
    sm = torch.stack([torch.arange(NREQ) % nrows, torch.arange(NREQ) % page], 1).to(torch.int32)
    o.slot_mapping = sm
    t = torch.randn(NREQ, 1, 1, 64)
    o.cos = {cos_key: t}
    o.sin = {cos_key: t.clone()}
    o.compress_cos = {cos_key: torch.randn(max(NREQ // 4, 1), 1, 1, 64)}
    o.compress_sin = {cos_key: torch.randn(max(NREQ // 4, 1), 1, 1, 64)}
    return types.SimpleNamespace(decode=o)


def strided(nblk, rows, dim, pad):
    raw = torch.zeros(nblk * pad)
    return torch.as_strided(raw, (nblk, rows, dim), (pad, dim, 1))


impl = Impl()
L = impl.layer_name
metas = [md(128, 64, 4096, L) for _ in range(5)]
kvc = (
    torch.zeros(64, 128, 1, K.HEAD_DIM, dtype=torch.bfloat16),           # compress kv
    torch.zeros(64, 128, 1, K.HEAD_DIM, dtype=torch.bfloat16),           # swa kv
    strided(64, 8, K.MAIN_STATE_DIM, 2 * 8 * K.MAIN_STATE_DIM),          # main state
    strided(64, 8, K.INNER_STATE_DIM, 8 * K.INNER_STATE_DIM + 64),       # inner state
    torch.zeros(64, 128, 1, K.IDX_HEAD_DIM, dtype=torch.int8),           # indexer k
    torch.zeros(64, 128, 1, 1, dtype=torch.float32),                     # indexer scale
)
hs = torch.randn(NREQ, D).to(torch.bfloat16)

print("== build_args ==")
try:
    args, plan = pa.build_args(impl, hs, kvc, metas, HOST_SEQ)
    check("returned 46", len(args) == 46, str(len(args)))
except Exception:
    import traceback
    traceback.print_exc()
    check("build_args ran", False)
    print("\nFAILED")
    sys.exit(1)

print("== shapes vs the kernel signature ==")
# T_DYN is a dynamic axis, so the runtime token count is n_real * S, not the
# module's compile-time T.
RT = NREQ * K.S
want = {
    "x_normed": [RT, D], "attn_out": [RT, D],
    "freqs_cos": [RT, 64], "cmp_freqs_cos": [RT, 64],
    "position_ids": [RT], "window_swa_indices": [RT, K.WIN],
    "ori_slot_mapping": [RT], "state_slot_mapping": [RT],
    "compress_state_block_table": [NREQ, K.MAIN_STATE_MAX_BLOCKS],
    "cmp_block_table": [NREQ, K.CMP_MAX_BLOCKS],
    "wo_a": [K.O_GROUPS, K.O_LORA, K.O_GROUP_IN],
    "weights_proj": [D, K.IDX_N_HEADS],
    "hadamard_idx": [K.IDX_HEAD_DIM, K.IDX_HEAD_DIM],
}
by = dict(zip(pa.ARG_ORDER, args))
for n, w in want.items():
    got = list(by[n].shape)
    check(n, got == w, f"{got} want {w}")

print("== contiguity (the binding rejects anything else) ==")
bad = [n for n, a in by.items() if not a.is_contiguous()]
check("all contiguous", not bad, str(bad))

print("== padding slots are inert ==")
for n in ("ori_slot_mapping", "cmp_slot_mapping", "idx_slot_mapping", "state_slot_mapping"):
    v = by[n].view(NREQ, K.S)
    check(f"{n} padding = -1", bool((v[:, 1:] == -1).all()), str(v[0].tolist()[:4]))

print()
print("FAILED:" if fails else "ALL PASS", fails or "")
sys.exit(1 if fails else 0)
