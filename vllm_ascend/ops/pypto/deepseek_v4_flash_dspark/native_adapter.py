# SPDX-License-Identifier: Apache-2.0
"""Prepared Native-storage invocation of the reference-derived CSA chain.

The caller retains Native metadata waits and cache lifecycle hooks. Allocation,
weight preparation and operator registration happen before graph capture.
"""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import logger

from .config import DECODE_BATCH
from .decode_csa import _decode_csa_tp1_layer, decode_csa_tp1_layer_test
from .native_storage import indexer_storage, physical_pages, table_storage
from .nz_mode import root_weight_layouts

_NZ_C0_BYTES = 32
_NZ_FRACTAL_ROWS = 16


def _base_weight_format(value: torch.Tensor) -> torch.Tensor:
    """仅为 ND 参数或离线打包承载张量在原设备上归一化基础格式。"""
    if value.device.type == "npu":
        import torch_npu

        if torch_npu.get_npu_format(value) not in (0, 2):
            return torch_npu.npu_format_cast(value, 2)
    return value


def _pack_nz(value: "torch.Tensor") -> "torch.Tensor":
    """把逻辑行主序的最后两维重排成 pto-isa 的 NZ 分形序，形状不变。

    NZ 是「列块在外、行分形在内」：c0 个连续元素构成一条 C0 线，16 行构成一个
    16 x c0 的分形，行轴上走 R/16 个分形，列轴上跨 C/c0 个列块——也就是
    BlockNzTensorViews 给 GM 视图的分块形状 [C/c0, R/16, 16, c0]。
    重排后仍按逻辑形状返回，只有字节次序变了，元素个数不变。
    `pl.NZ` 是对「GM 里的字节已经是这个次序」的断言，不是一个转换请求，
    按 pypto-lib/deepseek_v4_flash_dspark/utils.py 的 pack_nz 规则，在原设备重排。
    本 A3 对齐 BF16/INT8 权重的 Native 格式 29 与这里采用相同 fractal 规则；差异是描述符管理，
    以及各算子期待的矩阵方向/分组。本函数仅供离线回放布局转换；生产接入直接借用格式 29。
    """
    rows, cols = value.shape[-2], value.shape[-1]
    c0 = _NZ_C0_BYTES // value.element_size()
    if rows % _NZ_FRACTAL_ROWS or cols % c0:
        raise ValueError(f"NZ 需要 {_NZ_FRACTAL_ROWS} 行分形与整条 {c0} 元素的 C0 线，实到 {rows}x{cols}")
    logical = _base_weight_format(value.detach())
    packed = (
        logical.reshape(-1, rows // _NZ_FRACTAL_ROWS, _NZ_FRACTAL_ROWS, cols // c0, c0)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .reshape(value.shape)
    )
    return _base_weight_format(packed)


def _unpack_nz(value: "torch.Tensor") -> "torch.Tensor":
    """把已知为 PTO NZ 分形序的权重还原为逻辑 ND，供跨 mode 回放。"""
    rows, cols = value.shape[-2:]
    c0 = _NZ_C0_BYTES // value.element_size()
    if rows % _NZ_FRACTAL_ROWS or cols % c0:
        raise ValueError(f"不合法的 NZ 权重形状：{rows}x{cols}")
    unpacked = (
        value.reshape(-1, cols // c0, rows // _NZ_FRACTAL_ROWS, _NZ_FRACTAL_ROWS, c0)
        .permute(0, 2, 3, 1, 4)
        .contiguous().reshape(value.shape)
    )
    return _base_weight_format(unpacked)


def repack_weights(tensors: dict, source_layouts: dict, target_layouts: dict, target_shapes=None) -> dict:
    """离线快照显式转换：解包，迁移旧根矩阵方向，再按目标布局打包。"""
    result = dict(tensors)
    for name, target in target_layouts.items():
        source = source_layouts.get(name)
        if source not in ("ND", "NZ") or target not in ("ND", "NZ"):
            raise ValueError(f"未知权重布局：{name} {source} -> {target}")
        value = result[name]
        target_shape = tuple(target_shapes[name]) if target_shapes else tuple(value.shape)
        if source == target and tuple(value.shape) == target_shape:
            continue
        if source == "NZ":
            value = _unpack_nz(value)
        if tuple(value.shape) != target_shape:
            # 2026-09-26 前的 schema=2 快照使用 pypto-lib 矩阵方向。
            # 只迁移已知的三个转置矩阵，不能用 numel 相同猜测 reshape/分组。
            swapped = (*value.shape[:-2], value.shape[-1], value.shape[-2])
            if name not in ("wq_a", "wo_a", "wo_b") or swapped != target_shape:
                raise ValueError(f"不支持权重形状转换：{name} {tuple(value.shape)} -> {target_shape}")
            value = _base_weight_format(value.transpose(-1, -2).contiguous())
        result[name] = _pack_nz(value) if target == "NZ" else value
    return result


@dataclass(frozen=True)
class CSAOperators:
    attention: Any

    @classmethod
    def register(cls, kernel=decode_csa_tp1_layer_test) -> "CSAOperators":
        import pypto.torch

        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.reduction import validate_reduction_mode

        validate_reduction_mode()

        def register(kernel, name):
            return pypto.torch.register(kernel, f"dsv4_csa::{name}")

        return cls(
            register(kernel, "attention"),
        )


# torch_npu 的 acl format 取值：0=NCHW、2=ND，二者都是 PyPTO 根入参接受的基础格式。
_ACL_FORMAT_NCHW = 0
_ACL_FORMAT_ND = 2


def prepare_weights(attention, hadamard: torch.Tensor | None, layer=None, *,
                    root_function=_decode_csa_tp1_layer) -> dict[str, torch.Tensor]:
    """Prepare the TP1 ABI from already-loaded Native parameters exactly once."""
    import torch_npu

    if attention.compress_ratio != 4 or attention.n_local_heads != 64 or attention.n_local_groups != 8:
        raise ValueError("CSA specialization requires C4 and TP1 with 64 heads / 8 output groups")

    layouts = root_weight_layouts(root_function)

    def _recast(name, value, current, target):
        # 声明的布局与 Native 实际存法不符时只能另开一份私有副本。这条路径静默走过
        # 一次就意味着每层多占一份权重，而且只会体现在 worker 的 weights 汇总里——
        # 2026-09-28 的 CANN 9.0 -> 9.2 升级就是这样让 wo_a 悄悄多吃了 1.31 GiB
        # （9.0 的 Native 把三维 BF16 留在 ND，9.2 改存 NZ）。所以显式报出来。
        logger.warning(
            "PTO_CSA_WEIGHT_RECAST %s: kernel 声明 %s 但 Native 存的是 format=%d，"
            "复制一份私有副本 %.2f MiB/层；请核对 nz_mode 的布局判据是否还跟得上当前 CANN",
            name, layouts[name], current, value.numel() * value.element_size() / (1 << 20),
        )
        return torch_npu.npu_format_cast(value, target)

    def root_weight(name, shape, dtype):
        # 根矩阵方向与 Native 相同。格式已匹配时借用原存储，禁止解包、转置或重新打包。
        value = getattr(attention, name).weight.detach()
        if tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous():
            raise ValueError(f"Unexpected Native {name} weight: {value.shape}/{value.dtype}")
        current = int(torch_npu.get_npu_format(value))
        if layouts[name] == "NZ":
            return value if current == 29 else _recast(name, value, current, 29)
        if current in (_ACL_FORMAT_NCHW, _ACL_FORMAT_ND):
            return value
        return _recast(name, value, current, _ACL_FORMAT_ND)

    def weight(module, shape, dtype, transpose=False):
        value = module.weight.detach()
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"Unexpected loaded weight: {value.shape}/{value.dtype}; expected {shape}/{dtype}")
        if torch_npu.get_npu_format(value) not in (_ACL_FORMAT_NCHW, _ACL_FORMAT_ND):
            # 这些非目标权重仍由根签名声明 ND，按其数学方向在加载期准备一次。
            # 四张目标权重由 root_weight 独立绑定，已有 NZ 存储直接复用。
            value = torch_npu.npu_format_cast(value, _ACL_FORMAT_ND)
        if transpose:
            value = value.transpose(-1, -2)
        return _base_weight_format(value.contiguous())

    def scale(module, width):
        result = module.weight_scale.detach().reshape(-1)
        if result.numel() != width:
            raise ValueError("Unexpected quantized channel-scale count")
        offset = getattr(module, "weight_offset", None)
        if offset is not None and bool(torch.count_nonzero(offset).cpu()):
            raise ValueError("The reference CSA chain requires symmetric INT8 weights")
        return result.float().contiguous()

    bf16, int8 = torch.bfloat16, torch.int8
    main, indexer = attention.compressor, attention.indexer
    inner = indexer.compressor
    # mHC 的门控权重与 attention 的 input_layernorm 挂在 DeepseekV4DecoderLayer 上。
    hc = {}
    if layer is not None:
        hc = {
            "hc_attn_fn": layer.hc_attn_fn.detach().float().contiguous(),
            "hc_attn_scale": layer.hc_attn_scale.detach().float().contiguous(),
            "hc_attn_base": layer.hc_attn_base.detach().float().contiguous(),
            "attn_norm_w": layer.input_layernorm.weight.detach().to(bf16).contiguous(),
        }
    weights = {
        **hc,
        "wq_a": root_weight("wq_a", (1024, 4096), bf16),
        "wq_b": root_weight("wq_b", (1024, 32768), int8),
        "wq_b_scale": scale(attention.wq_b, 32768),
        "wkv": weight(attention.wkv, (512, 4096), bf16, True),
        "gamma_cq": weight(attention.q_norm, (1024,), bf16),
        "gamma_ckv": weight(attention.kv_norm, (512,), bf16),
        "cmp_wkv": weight(main.wkv, (1024, 4096), bf16),
        "cmp_wgate": weight(main.wgate, (1024, 4096), bf16),
        "cmp_ape": main.ape.detach().float().contiguous(),
        # Match Native A3 storage; the RMS task widens loaded BF16 tiles.
        "cmp_norm_w": weight(main.norm, (512,), bf16),
        "idx_wq_b": weight(indexer.wq_b, (1024, 8192), int8),
        "idx_wq_b_scale": scale(indexer.wq_b, 8192),
        "weights_proj": weight(indexer.weights_proj, (64, 4096), bf16, True),
        **({"hadamard_idx": hadamard.detach().T.to(bf16).contiguous()} if hadamard is not None else {}),
        "inner_wkv": weight(inner.wkv, (256, 4096), bf16),
        "inner_wgate": weight(inner.wgate, (256, 4096), bf16),
        "inner_ape": inner.ape.detach().float().contiguous(),
        "inner_norm_w": weight(inner.norm, (128,), bf16),
        "attn_sink": attention.attn_sink.detach().contiguous(),
        "wo_a": root_weight("wo_a", (8, 4096, 1024), bf16),
        "wo_b": root_weight("wo_b", (8192, 4096), int8),
        "wo_b_scale": scale(attention.wo_b, 4096),
    }
    return weights


class NativeCSACall:
    """Fixed-address eager/graph call for uniform six-token target requests.

    整档中可以含补位请求：它们由 kernel 内的 seq_lens 判据屏蔽，此处只校验真实
    部分仍是完整六行请求。模型前向的派发仍由各自的闸门决定。
    compact_metadata contains the release Native producer's device tensors;
    this descriptor binds them without materializing an expanded buffer.
    """

    def __init__(self, ops, weights, hidden, positions, groups, *, layer_name: str, compact_metadata, buffers=None,
                 kernel=decode_csa_tp1_layer_test):
        # Each entry contains its own metadata and Native cache views. No shared
        # synthetic page table can stand in for another cache group.
        self.ops = ops
        self.groups = groups
        self.positions = positions
        self.req = {name: value[0].decode for name, value in groups.items()}
        self.views = {name: value[1] for name, value in groups.items()}
        batch = self.req["swa"].seq_lens.numel()
        tokens = hidden.shape[0]
        if not 1 <= batch <= DECODE_BATCH or tokens != batch * 6:
            raise ValueError(f"CSA requires 1 <= batch <= {DECODE_BATCH} and six unpadded rows per request")
        # kernel 现在从 mHC 的残差流进、也从它出，入参是层间的 [T, HC_MULT, D]。
        if tuple(hidden.shape[1:]) != (4, 4096) or hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
            raise ValueError("CSA expects a contiguous BF16 hc residual stream [T, 4, 4096]")
        if positions.dtype != torch.int64 or tuple(positions.shape) != (tokens,):
            raise ValueError("CSA expects the Native INT64 target position vector")
        for name, (metadata, _) in groups.items():
            # 补位档位下 num_actual_tokens 是**实际** token 数，小于 hidden 的整档
            # 行数（实测 18/24）；补位请求本身由 kernel 内的 seq_lens 判据屏蔽，
            # 这里只要求真实部分是完整的六行请求。
            if metadata.num_prefills or metadata.num_actual_tokens > tokens:
                raise ValueError(f"{name}: requires target decode metadata without prefill rows")
            if metadata.num_actual_tokens % 6:
                raise ValueError(f"{name}: CSA requires whole six-token target requests")
            req = self.req[name]
            if req.query_start_loc.numel() != batch + 1 or req.seq_lens.numel() != batch:
                raise ValueError(f"{name}: inconsistent Native request capacity")
            if req.ori_win_right not in (None, 0):
                raise ValueError("Noncausal drafter windows are outside the target CSA contract")

        def empty(name, shape, dtype):
            if buffers is None:
                return torch.empty(shape, dtype=dtype, device=hidden.device)
            value = buffers[name]
            if tuple(value.shape) != shape or value.dtype != dtype or value.device != hidden.device:
                raise ValueError(f"Invalid prepared CSA buffer {name}")
            return value

        for name, width in (("state", 2048), ("indexer_state", 512)):
            view = self.views[name][0]
            if view.dtype != torch.float32 or tuple(view.shape[1:]) != (2, 1, width):
                raise ValueError(f"{name}: CSA expects Native FP32 two-token state pages with row width {width}")
        self.state_storage = {name: physical_pages(self.views[name][0]) for name in ("state", "indexer_state")}
        self.tables = {name: table_storage(req.block_table) for name, req in self.req.items()}
        self.args = dict(weights)
        self.args.update(
            x_hc=hidden,
            kv_cache=self.views["swa"][0],
            cmp_kv=self.views["compressed"][0],
            **self._indexer_cache_arguments(),
            cmp_block_table=table_storage(self.req["compressed"].block_table),
            idx_block_table=table_storage(self.req["indexer"].block_table),
            kv_seq_lens=self.req["indexer"].seq_lens,
            position_ids=positions,
            ori_slot_mapping=self.req["swa"].slot_mapping,
            state_slot_mapping=self.req["state"].slot_mapping,
            inner_state_slot_mapping=self.req["indexer_state"].slot_mapping,
            ori_block_table=table_storage(self.req["swa"].block_table),
            compress_state=self.state_storage["state"],
            inner_compress_state=self.state_storage["indexer_state"],
            state_block_table=self.tables["state"],
            inner_state_block_table=self.tables["indexer_state"],
            idx_topk_scores=empty("idx_topk_scores", (tokens, 512), torch.float32),
            idx_topk=empty("idx_topk", (tokens, 512), torch.int32),
            x_out=empty("x_out", (tokens, 4, 4096), torch.bfloat16),
        )
        for name, slot_name, rope_name in (
            ("compressed", "cmp_slot_mapping", "cmp_freqs"),
            ("indexer", "idx_slot_mapping", "inner_freqs"),
        ):
            cos, sin, slots = compact_metadata[name]
            self.args[slot_name] = slots
            self.args[f"{rope_name}_cos"] = cos.view(-1, 64)
            self.args[f"{rope_name}_sin"] = sin.view(-1, 64)
        self.args["cmp_query_start_loc"] = self.req["compressed"].query_start_loc
        self.args["cmp_seq_lens"] = self.req["compressed"].seq_lens
        self.args["idx_query_start_loc"] = self.req["indexer"].query_start_loc
        for name in ("kv_cache", "cmp_kv"):
            if not self.args[name].is_contiguous() or self.args[name].dtype != torch.bfloat16:
                raise ValueError(f"{name}: Native BF16 32-token pages must have no additional page padding")
        main = self.req["compressed"]
        # Native 的 decode RoPE 是常驻缓冲的切片视图，按**实际** token 数切
        # （`vllm_ascend/ops/rope_dsv4.py` 的 use_cache 分支）。图捕获发生在无补位的
        # 满档 dummy 上，捕获到的是整档视图；补位步只回写前若干行，尾部保留上一步
        # 的值，与 positions 同机制，不是越界。这里显式要求视图覆盖整档，
        # 免得在 view 上抛出难以定位的形状错误。
        cos, sin = main.cos[layer_name], main.sin[layer_name]
        if cos.shape[0] < tokens or sin.shape[0] < tokens:
            raise ValueError("CSA requires Native decode RoPE rows covering the whole padded bucket")
        self.native_cos = cos[:tokens].view(tokens, 64)
        self.native_sin = sin[:tokens].view(tokens, 64)
        # All CSA consumers use Native interleaved FP32 frequency columns.
        # Keep the Native buffers and their producer waits; no device conversion.
        self.args["freqs_cos"] = self.native_cos
        self.args["freqs_sin"] = self.native_sin
        self.core_args = tuple(self.args[name] for name in kernel.param_names)

    def _indexer_cache_arguments(self):
        return {"idx_kv_cache": indexer_storage(*self.views["indexer"])}

    def __call__(self):
        self.ops.attention(*self.core_args)
        return self.args["x_out"]
