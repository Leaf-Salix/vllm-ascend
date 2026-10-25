# SPDX-License-Identifier: Apache-2.0
"""Native TND零拷贝绑定；临时Tensor由PTO编排经Simpler内部申请。

本层只保存Native输入/输出、描述符和策略；不读回metadata值，
不分配临时工作区或信号。显式提供Top-K输出时使用诊断入口。
"""

from collections.abc import Mapping

import torch

from vllm_ascend import envs

from .policy import CSASPMDPolicy, select_policy

REQUIRED_CUBE_CORES = 24
REQUIRED_VECTOR_CORES = 2 * REQUIRED_CUBE_CORES


def validate_core_geometry(cube_core_num: int, vector_core_num: int) -> None:
    """固定CSA核角色映射要求整卡24C48V；硬件不匹配时在提交前拒绝。"""
    if (cube_core_num, vector_core_num) != (REQUIRED_CUBE_CORES, REQUIRED_VECTOR_CORES):
        raise NotImplementedError(
            "CSA单SPMD的核内清零/硬同步要求24个AIC和48个AIV；"
            f"当前为{cube_core_num}个AIC和{vector_core_num}个AIV，不启动其他task路径"
        )


class PreparedDynamicCSA:
    """准备Native根参数；无外部scratch，调用内由Simpler管理临时内存。"""

    def __init__(
        self,
        native: Mapping[str, torch.Tensor],
        *,
        max_query_tokens: int,
        max_seq_len: int,
        policy: CSASPMDPolicy | None = None,
    ):
        if envs.VLLM_ASCEND_ENABLE_NZ != 2:
            raise NotImplementedError("完整动态SPMD当前先接Native NZ2，其他布局尚未完成整链验证")
        self.native = dict(native)
        hidden = native["x_hc"]
        self.tokens = t = hidden.shape[0]
        self.requests = b = native["kv_seq_lens"].numel()
        if hidden.shape != (t, 4, 4096) or hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
            raise ValueError("要求Native BF16连续hidden[T,4,4096]")
        if native["cmp_block_table"].ndim != 2:
            raise ValueError("cmp_block_table must be two-dimensional")
        self.policy = policy or select_policy(
            tokens=t,
            requests=b,
            max_query_tokens=max_query_tokens,
            max_seq_len=max_seq_len,
            compressed_table_columns=native["cmp_block_table"].shape[1],
        )
        device = hidden.device
        if device.type == "npu":
            import torch_npu

            # 只查询硬件属性，不读取metadata或Tensor值；绑定准备时执行一次。
            properties = torch_npu.npu.get_device_properties(device)
            validate_core_geometry(properties.cube_core_num, properties.vector_core_num)
        self.buffers: dict[str, torch.Tensor] = {}
        # 名称变更只对应接口职责，Tensor保持原对象；同一组内Q/L分别保留。
        direct = {
            "hc_fn": "hc_attn_fn",
            "hc_scale": "hc_attn_scale",
            "hc_base": "hc_attn_base",
            "norm_w": "attn_norm_w",
            "wq_a": "wq_a",
            "gamma_cq": "gamma_cq",
            "wq_b": "wq_b",
            "wq_b_scale": "wq_b_scale",
            "rope_cos": "freqs_cos",
            "rope_sin": "freqs_sin",
            "wkv": "wkv",
            "gamma_ckv": "gamma_ckv",
            "native_slots": "ori_slot_mapping",
            "main_wkv": "cmp_wkv",
            "main_wgate": "cmp_wgate",
            "main_state": "compress_state",
            "main_state_table": "state_block_table",
            "main_ape": "cmp_ape",
            "positions": "position_ids",
            "main_bounds": "cmp_query_start_loc",
            "main_lengths": "cmp_seq_lens",
            "main_state_slots": "state_slot_mapping",
            "main_gamma": "cmp_norm_w",
            "main_cos": "cmp_freqs_cos",
            "main_sin": "cmp_freqs_sin",
            "main_slots": "cmp_slot_mapping",
            "idx_wkv": "inner_wkv",
            "idx_wgate": "inner_wgate",
            "idx_state": "inner_compress_state",
            "idx_state_table": "inner_state_block_table",
            "idx_ape": "inner_ape",
            "idx_bounds": "idx_query_start_loc",
            "idx_lengths": "kv_seq_lens",
            "idx_state_slots": "inner_state_slot_mapping",
            "idx_gamma": "inner_norm_w",
            "idx_cos": "inner_freqs_cos",
            "idx_sin": "inner_freqs_sin",
            "idx_slots": "idx_slot_mapping",
            "hadamard": "hadamard_idx",
            "idx_cache": "idx_native_kv_cache",
            "original_table": "ori_block_table",
            "compressed_table": "cmp_block_table",
            "indexer_table": "idx_block_table",
            "idx_wq_b": "idx_wq_b",
            "idx_wq_b_scale": "idx_wq_b_scale",
            "weights_proj": "weights_proj",
            "topk_scores": "idx_topk_scores",
            "topk_indices": "idx_topk",
            "wo_a": "wo_a",
            "wo_b": "wo_b",
            "wo_b_scale": "wo_b_scale",
        }
        self._diagnostic = "idx_topk" in native
        if self._diagnostic != ("idx_topk_scores" in native):
            raise ValueError("诊断Top-K索引与分数必须同时提供")
        if not self._diagnostic:
            direct.pop("topk_indices")
            direct.pop("topk_scores")
        self._direct_names = tuple(direct.items())
        self.buffers.update({target: native[source] for target, source in direct.items()})
        for name in ("main_bounds", "idx_bounds"):
            if self.buffers[name].shape != (b + 1,) or self.buffers[name].dtype != torch.int32:
                raise ValueError(f"{name}须为Native INT32[B+1]")
        for name in ("main_lengths", "idx_lengths"):
            if self.buffers[name].shape != (b,) or self.buffers[name].dtype != torch.int32:
                raise ValueError(f"{name}须为Native INT32[B]")
        for name, width in (
            ("norm_w", 4096),
            ("gamma_cq", 1024),
            ("gamma_ckv", 512),
            ("main_gamma", 512),
            ("idx_gamma", 128),
        ):
            value = self.buffers[name]
            if value.shape != (width,) or value.dtype != torch.bfloat16:
                raise ValueError(f"{name}必须保持Native BF16[{width}]")
        for name in ("wq_a", "wq_b", "wo_a", "wo_b"):
            if device.type == "npu":
                import torch_npu

                if torch_npu.get_npu_format(self.buffers[name]) != 29:
                    raise ValueError(f"{name}须由Native已有加载流程提供NZ格式")
        for name, value in self.buffers.items():
            if value.device != device or not value.is_contiguous():
                raise ValueError(f"{name}须在同一设备并满足连续GM描述符")

        self.buffers.update(
            x=hidden.view(t, 16384),
            output=native["x_out"].view(t, 16384),
            hc_base_2d=native["hc_attn_base"].view(1, 24),
            cache_rows=native["kv_cache"].view(-1, 512),
            main_cache=native["cmp_kv"].view(-1, 512),
            attn_sink=native["attn_sink"].view(64, 1),
        )
        self.allocations: dict[str, torch.Tensor] = {}  # 外部工作区计数为零。
        cache = self.buffers["idx_cache"]
        if cache.dtype != torch.int8 or cache.ndim != 2 or cache.shape[1] < 4160 or cache.shape[1] % 64:
            raise ValueError("Indexer cache须为Native对齐INT8页，页内至少4096B key+64B scale")

    def rebind(self, native: Mapping[str, torch.Tensor]) -> None:
        """替换同描述符的Native根参数；内部临时内存由每次PTO调用管理。"""
        roots = {target: native[source] for target, source in self._direct_names}
        roots.update(
            x=native["x_hc"].view(self.tokens, 16384),
            output=native["x_out"].view(self.tokens, 16384),
            hc_base_2d=native["hc_attn_base"].view(1, 24),
            cache_rows=native["kv_cache"].view(-1, 512),
            main_cache=native["cmp_kv"].view(-1, 512),
            attn_sink=native["attn_sink"].view(64, 1),
        )
        for name, value in roots.items():
            previous = self.buffers[name]
            if (
                value.shape != previous.shape
                or value.stride() != previous.stride()
                or value.dtype != previous.dtype
                or value.device != previous.device
            ):
                raise ValueError(f"{name}的Native描述符变化，需要重新准备TND绑定")
        self.native = dict(native)
        self.buffers.update(roots)

    def __call__(self, *, config=None):
        """直接走PyPTO native prepare/enqueue；不依赖Torch dispatcher的参数数量上限。"""
        from .entry import decode_csa_spmd, decode_csa_spmd_debug

        options = {} if config is None else {"config": config}
        entry = decode_csa_spmd_debug if self._diagnostic else decode_csa_spmd
        entry(**self.buffers, **self.policy.constexprs(), **options)
        return self.native["x_out"]
