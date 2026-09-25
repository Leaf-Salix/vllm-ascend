# SPDX-License-Identifier: Apache-2.0
"""Prepared Native-storage invocation of the reference-derived CSA chain.

The caller retains Native metadata waits and cache lifecycle hooks. Allocation,
weight preparation and operator registration happen before graph capture.
"""

from dataclasses import dataclass
from typing import Any

import torch

from .decode_csa import decode_csa_tp1_layer_test
from .config import DECODE_BATCH
from .native_storage import indexer_storage, physical_pages, table_storage


@dataclass(frozen=True)
class CSAOperators:
    attention: Any

    @classmethod
    def register(cls) -> "CSAOperators":
        import pypto.torch

        def register(kernel, name):
            return pypto.torch.register(kernel, f"dsv4_csa::{name}")

        return cls(
            register(decode_csa_tp1_layer_test, "attention"),
        )


# NZ 的字节重排实现在精度版，这里只引用，保证两版用的是同一份。
from ..deepseek_v4_flash_dspark.native_adapter import _maybe_pack_nz, _pack_nz  # noqa: F401

# torch_npu 的 acl format 取值：0=NCHW、2=ND，二者都是 PyPTO 根入参接受的基础格式。
_ACL_FORMAT_NCHW = 0
_ACL_FORMAT_ND = 2


def prepare_weights(attention, hadamard: torch.Tensor | None, layer=None) -> dict[str, torch.Tensor]:
    """Prepare the TP1 ABI from already-loaded Native parameters exactly once."""
    import torch_npu

    if attention.compress_ratio != 4 or attention.n_local_heads != 64 or attention.n_local_groups != 8:
        raise ValueError("CSA specialization requires C4 and TP1 with 64 heads / 8 output groups")

    def weight(module, shape, dtype, transpose=False):
        value = module.weight.detach()
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"Unexpected loaded weight: {value.shape}/{value.dtype}; expected {shape}/{dtype}")
        if torch_npu.get_npu_format(value) not in (_ACL_FORMAT_NCHW, _ACL_FORMAT_ND):
            # weight_nz_mode>=1 时 Native 会把量化权重转成 FRACTAL_NZ（见
            # vllm_ascend/utils.py 的 maybe_trans_nz 与各 w8a8 method 的
            # process_weights_after_loading），而 PyPTO 的根入参只接受
            # NCHW(0) 或 ND(2)。这里转回 ND。
            #
            # 两个 NZ 不是一回事：Native 的是张量的 npu format，PTO 的 pl.NZ 要的是
            # 「字节按 pto-isa 的分形序摆好、format 仍是 ND」。所以即便将来 PTO 这边
            # 也想用 NZ，也不能直接拿 Native 转过的这份，仍要先回到 ND。
            #
            # 这一步在 process_weights_after_loading 里每层只做一次（见
            # models/pypto_deepseek_v4.py 的 prepare_csa_model），发生在权重加载之后、
            # aclgraph capture 之前，不在 decode 路径上，因此不影响 replay。
            value = torch_npu.npu_format_cast(value, _ACL_FORMAT_ND)
        if transpose:
            value = value.transpose(-1, -2)
        return value.contiguous()

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
    # mHC 的门控权重与 attention 的 input_layernorm 挂在 DeepseekV4DecoderLayer 上，
    # 不在 self_attn 上，所以要由调用方把 layer 一并传进来。
    hc = {}
    if layer is not None:
        hc = {
            "hc_attn_fn": layer.hc_attn_fn.detach().float().contiguous(),
            "hc_attn_scale": layer.hc_attn_scale.detach().float().contiguous(),
            "hc_attn_base": layer.hc_attn_base.detach().float().contiguous(),
            "attn_norm_w": layer.input_layernorm.weight.detach().to(bf16).contiguous(),
        }
    return {
        **hc,
        # wq_a 保持 ND：它的 NZ kernel 版在当前 PyPTO 上编不过（可证判据不支持整除），
        # 见 qkv_proj_rope.q_proj_qa 的说明。主机侧打包必须与 kernel 标注同步，所以这里
        # 也不能提前打包——只开一侧不报错、只算错。
        "wq_a": weight(attention.wq_a, (1024, 4096), bf16, True),
        # NZ 序存放（mode>=1 即开）：Native 在 mode>=1 下已把它转成 FRACTAL_NZ，
        # weight() 里先 npu_format_cast 回 ND，这里再按 pto-isa 的分形序重排。
        # 两个 NZ 不是一回事，见 _pack_nz 的说明。
        "wq_b": _maybe_pack_nz(weight(attention.wq_b, (1024, 32768), int8), int8),
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
        # NZ 序存放：这一份本来就是 transpose 出来的独立副本，不额外占显存。
        "wo_a": _maybe_pack_nz(weight(attention.wo_a, (8, 4096, 1024), bf16, True), bf16),
        # wo_b 保持 ND，见 decode_csa.py 里它的签名说明。
        "wo_b": weight(attention.wo_b, (8192, 4096), int8, True),
        "wo_b_scale": scale(attention.wo_b, 4096),
    }


class NativeCSACall:
    """Fixed-address eager/graph call for uniform six-token target requests.

    整档中可以含补位请求：它们由 kernel 内的 seq_lens 判据屏蔽，此处只校验真实
    部分仍是完整六行请求。模型前向的派发仍由各自的闸门决定。
    compact_metadata contains the release Native producer's device tensors;
    this descriptor binds them without materializing an expanded buffer.
    """

    def __init__(self, ops, weights, hidden, positions, groups, *, layer_name: str, compact_metadata, buffers=None):
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
        # kernel 现在从 mHC 的残差流进、也从它出（hc_pre + input_layernorm +
        # attention + hc_post 都在算子内），入参因此是层间的 [T, HC_MULT, D]，
        # 不再是归一化后的 [T, D]。
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
            idx_kv_cache=indexer_storage(*self.views["indexer"]),
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
        self.core_args = tuple(self.args[name] for name in decode_csa_tp1_layer_test.param_names)

    def __call__(self):
        self.ops.attention(*self.core_args)
        return self.args["x_out"]
