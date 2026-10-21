# SPDX-License-Identifier: Apache-2.0
"""仅用 PyTorch 准备固定 TND 场景及正式单层权重，不加载推理引擎或自定义算子。"""

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


def load_weights(checkpoint: Path, layer: int, device):
    """读取 ModelSlim 的原始张量，显式表达矩阵方向、dtype 和四张 NZ 根权重。"""
    import torch_npu

    index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    prefix = f"layers.{layer}."
    records = []

    def read(suffix, dtype=None):
        name = prefix + suffix
        with safe_open(checkpoint / index[name], framework="pt", device="cpu") as reader:
            value = reader.get_tensor(name)
        records.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)})
        return value.to(dtype=dtype) if dtype else value

    def weight(suffix, dtype=torch.bfloat16, transpose=False):
        value = read("attn." + suffix, dtype)
        return (value.T if transpose else value).contiguous().to(device)

    def scale(name):
        offset = read("attn." + name + ".weight_offset")
        if torch.count_nonzero(offset):
            raise ValueError(f"{name}: 当前算子只支持对称 INT8 权重")
        return read("attn." + name + ".weight_scale", torch.float32).flatten().contiguous().to(device)

    result = {
        "hc_attn_fn": read("hc_attn_fn", torch.float32).to(device),
        "hc_attn_scale": read("hc_attn_scale", torch.float32).to(device),
        "hc_attn_base": read("hc_attn_base", torch.float32).to(device),
        "attn_norm_w": read("attn_norm.weight", torch.bfloat16).to(device),
        "wq_a": weight("wq_a.weight"),
        "wq_b": weight("wq_b.weight", torch.int8, True),
        "wq_b_scale": scale("wq_b"),
        "wkv": weight("wkv.weight", transpose=True),
        "gamma_cq": weight("q_norm.weight"),
        "gamma_ckv": weight("kv_norm.weight"),
        "cmp_wkv": weight("compressor.wkv.weight"),
        "cmp_wgate": weight("compressor.wgate.weight"),
        "cmp_ape": weight("compressor.ape", torch.float32),
        "cmp_norm_w": weight("compressor.norm.weight"),
        "attn_sink": weight("attn_sink", torch.float32),
        "wo_a": read("attn.wo_a.weight", torch.bfloat16).reshape(8, 1024, 4096)
        .transpose(1, 2).contiguous().to(device),
        "wo_b": weight("wo_b.weight", torch.int8, True),
        "wo_b_scale": scale("wo_b"),
        "idx_wq_b": weight("indexer.wq_b.weight", torch.int8, True),
        "idx_wq_b_scale": scale("indexer.wq_b"),
        "weights_proj": weight("indexer.weights_proj.weight", transpose=True),
        "inner_wkv": weight("indexer.compressor.wkv.weight"),
        "inner_wgate": weight("indexer.compressor.wgate.weight"),
        "inner_ape": weight("indexer.compressor.ape", torch.float32),
        "inner_norm_w": weight("indexer.compressor.norm.weight"),
    }
    for name in ("wq_a", "wq_b", "wo_a", "wo_b"):
        result[name] = torch_npu.npu_format_cast(result[name], 29)
    hadamard = torch.ones((1, 1), dtype=torch.float32)
    while hadamard.shape[0] < 128:
        hadamard = torch.cat((torch.cat((hadamard, hadamard), 1),
                              torch.cat((hadamard, -hadamard), 1)), 0)
    result["hadamard_idx"] = hadamard.to(device=device, dtype=torch.bfloat16)
    return result, records


def rope_rows(positions, config, device):
    """C4 的 YaRN FP32 cos/sin；偶奇相邻列重复同一频率，压缩行使用组起点。"""
    scaling = config.get("rope_parameters", config.get("rope_scaling"))
    base, dim = config["compress_rope_theta"], config["qk_rope_head_dim"]
    original, factor = scaling["original_max_position_embeddings"], scaling["factor"]

    def correction(rotations):
        return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction(scaling["beta_fast"])), 0)
    high = min(math.ceil(correction(scaling["beta_slow"])), dim - 1)
    if low == high:
        high += 0.001
    frequencies = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    mask = 1 - ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    inv = (1 / (factor * frequencies)) * (1 - mask) + (1 / frequencies) * mask
    angles = positions.to(device=device, dtype=torch.float32)[:, None] * inv.to(device)[None, :]
    return angles.cos().repeat_interleave(2, -1), angles.sin().repeat_interleave(2, -1)


def make_inputs(case, checkpoint, layer, device):
    """保留固定 Q/L、页顺序、padding、cache dtype/物理步长；仅替换输入准备方式。"""
    native, weights = load_weights(checkpoint, layer, device)
    config = json.loads((checkpoint / "config.json").read_text())
    torch.manual_seed(case["seed"])
    torch.npu.manual_seed(case["seed"])
    batch, requests, tokens = case["active_requests"], case["request_capacity"], case["token_capacity"]
    hidden = torch.randn((tokens, 4, 4096), dtype=torch.bfloat16, device=device)
    positions_cpu = torch.zeros(tokens, dtype=torch.int64)
    for request, (history, query) in enumerate(zip(case["history_lengths"], case["query_lengths"])):
        start = case["query_start_loc"][request]
        positions_cpu[start:start + query] = torch.arange(history, history + query)
    bounds = torch.tensor(case["query_start_loc"], dtype=torch.int32, device=device)
    lengths = torch.tensor(case["seq_lens"], dtype=torch.int32, device=device)
    positions = positions_cpu.to(device)
    native.update(x_hc=hidden, x_out=torch.zeros_like(hidden), position_ids=positions,
                  kv_seq_lens=lengths, cmp_seq_lens=lengths.clone(),
                  cmp_query_start_loc=bounds, idx_query_start_loc=bounds.clone(),
                  idx_topk=torch.full((tokens, 512), -1, dtype=torch.int32, device=device),
                  idx_topk_scores=torch.zeros((tokens, 512), dtype=torch.float32, device=device))
    native["freqs_cos"], native["freqs_sin"] = rope_rows(positions, config, device)
    groups, readonly = {}, {"positions": positions, "Q": bounds, "L": lengths,
                            "cmp_L": native["cmp_seq_lens"], "idx_Q": native["idx_query_start_loc"]}
    # block rows, compression ratio, physical bytes/page, element width, dtype, root name, table prefix
    layouts = {
        "swa": (32, 1, 32768, 512, torch.bfloat16, "kv_cache", "ori"),
        "compressed": (32, 4, 32768, 512, torch.bfloat16, "cmp_kv", "cmp"),
        "state": (2, 1, 32768, 2048, torch.float32, "compress_state", "state"),
        "indexer": (32, 4, 4160, 128, torch.int8, "idx_native_kv_cache", "idx"),
        "indexer_state": (2, 1, 4160, 512, torch.float32, "inner_compress_state", "inner_state"),
    }
    compact_capacity = min(case["valid_tokens"], case["valid_tokens"] // 4 + requests)
    for name, (block, ratio, page_bytes, width, dtype, root_name, prefix) in layouts.items():
        columns = (case["table_history"] + 6 + block * ratio - 1) // (block * ratio) + 1
        per_request = columns if ratio == 4 else 9
        pages = requests * per_request + 1
        storage = torch.full((pages * page_bytes + 256,), 37, dtype=torch.int8, device=device)
        raw = storage[128:-128]
        size = torch.empty((), dtype=dtype).element_size()
        view = storage.view(dtype).as_strided((pages, block, width),
                                            (page_bytes // size, width, 1), 128 // size)
        views = [view]
        if dtype == torch.int8:
            view.random_(-64, 64)
            scales = storage.view(torch.float16).as_strided((pages, 32, 1),
                                                           (page_bytes // 2, 1, 1), (128 + 4096) // 2)
            rows = torch.arange(scales.numel(), dtype=torch.int64, device=device)
            scales.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).half().reshape(scales.shape))
            views.append(scales)
            native[root_name] = raw.view(pages, page_bytes)
        else:
            view.normal_(0, 0.5)
            native[root_name] = (raw.view(dtype).view(pages, page_bytes // size)
                                 if "state" in name else view)
        table = torch.zeros((requests, columns), dtype=torch.int32)
        for request in range(batch):
            owner = batch - 1 - request if case["reverse_pages"] else request
            table[request] = 1 + owner * per_request + (per_request - 1 - torch.arange(columns)) % per_request
        native[prefix + "_block_table"] = table.to(device)
        slots = torch.full((compact_capacity if ratio == 4 else tokens, 2), -1, dtype=torch.int32)
        compact_positions = torch.zeros(compact_capacity, dtype=torch.int64)
        compact_row = 0
        for request, (history, query) in enumerate(zip(case["history_lengths"], case["query_lengths"])):
            if ratio == 4:
                for pos in range(history // 4, (history + query) // 4):
                    slots[compact_row] = torch.tensor([table[request, pos // block], pos % block])
                    compact_positions[compact_row] = pos * 4
                    compact_row += 1
            else:
                for offset in range(query):
                    pos = history + offset
                    slots[case["query_start_loc"][request] + offset] = torch.tensor(
                        [table[request, pos // block], pos % block])
        native[prefix + "_slot_mapping"] = slots.to(device)
        if ratio == 4:
            cos, sin = rope_rows(compact_positions, config, device)
            cos[compact_row:].zero_()
            sin[compact_row:].zero_()
            rope_prefix = "cmp" if name == "compressed" else "inner"
            native[rope_prefix + "_freqs_cos"], native[rope_prefix + "_freqs_sin"] = cos, sin
        allowed = torch.zeros(storage.numel(), dtype=torch.bool)
        for page, row in slots.tolist():
            if page < 0:
                continue
            for item in views:
                first = (item.storage_offset() + page * item.stride(0) + row * item.stride(1)) * item.element_size()
                allowed[first:first + item.shape[-1] * item.element_size()] = True
        groups[name] = {"allocation": storage, "views": views, "allowed": allowed,
                        "initial": storage.cpu(), "slots": slots}
        readonly[name + ".table"] = native[prefix + "_block_table"]
        readonly[name + ".slots"] = native[prefix + "_slot_mapping"]
    readonly.update({name: value for name, value in native.items() if "freqs_" in name})
    return native, groups, {name: (value, value.cpu()) for name, value in readonly.items()}, weights


def restore(native, groups):
    for group in groups.values():
        group["allocation"].copy_(group["initial"])
    native["x_out"].zero_()
    native["idx_topk"].fill_(-1)
    native["idx_topk_scores"].zero_()
