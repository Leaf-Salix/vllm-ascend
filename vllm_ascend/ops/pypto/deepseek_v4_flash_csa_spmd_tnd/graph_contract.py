# SPDX-License-Identifier: Apache-2.0
"""TND模型图的捕获合同：重放前只比较Host描述符、策略和输入地址。

同址Q/L/positions内容可以变化，Tensor值不进入签名。请求槽容量、根输入
描述符、页存储地址或策略改变时，旧图不能被直接重放。捕获到Native/空路径的
图也不能假定后续仍走同一路径，统一回到正常Python派发重新判定。
"""

from dataclasses import dataclass
from typing import Any

TND_PADDED_REDISPATCH = "pto_csa_tnd_padded_redispatch"
CSA_COMPRESS_RATIO = 4


def compact_metadata_capacity(tokens, requests):
    """Native C4输出容量上界；实际关闭行数仍由设备Q/start_pos决定。"""
    return min(tokens, tokens // CSA_COMPRESS_RATIO + requests)


def graph_compact_request(request, tokens):
    """固定图容量，用设备L屏蔽padding，不捕获某一步的实际请求数。

    Native compressor_metadata对start_pos<0的请求产生零行。只修改传给该
    Native算子的浅拷贝；原metadata及其start_pos存储不变。where在图内执行，
    每次replay都读取本步L，且不从设备读回有效请求数。
    """
    from copy import copy

    import torch

    requests = request.seq_lens.numel()
    if request.start_pos.shape != (requests,) or request.start_pos.dtype != torch.int32:
        raise ValueError("Native C4 start_pos须为INT32请求容量向量")
    adapted = copy(request)
    adapted.num_reqs_actual = requests
    adapted.num_compressed_tokens = compact_metadata_capacity(tokens, requests)
    adapted.start_pos = torch.where(request.seq_lens > 0, request.start_pos, -1)
    return adapted


def native_decode_capacity_views(metadata, hidden_tokens, compress_ratio):
    """TND图拒绝重放后，Native仍须按原图容量消费已补齐的Q前缀。

    例如有效42行、容量96行时，Native builder的Q前缀末项是96。
    普通eager路径按num_decode_tokens=42截短query会越界。此处只为
    当前Native调用建立容量视图，复用已初始化的positions/slots/RoPE；
    保留真实num_actual_tokens和num_reqs_actual，不修改全局metadata、
    不读取device Q/L，也不重做metadata算子。PTO继续使用原始TND合同。
    """
    from copy import copy

    from .native_adapter import token_capacity_view

    result = []
    for item in metadata:
        if item.num_prefills or item.decode is None or item.num_decode_tokens == hidden_tokens:
            result.append(item)
            continue
        if item.num_input_tokens != hidden_tokens or item.num_decode_tokens > hidden_tokens:
            raise ValueError("TND图回退的Native输入与已准备的padding容量不一致")
        decoded = copy(item.decode)
        decoded.input_positions = token_capacity_view(decoded.input_positions.view(-1, 1), hidden_tokens, 1).view(-1)
        if decoded.slot_mapping is not None:
            decoded.slot_mapping = token_capacity_view(decoded.slot_mapping, hidden_tokens, 2)
        # 顶层RoPE由Native按num_input_tokens准备；decode上的view只覆盖有效token。
        decoded.cos, decoded.sin = item.cos, item.sin
        if getattr(decoded, "full_compress_cos", None) is not None:
            # 与Native builder的_num_compressor_metadata_rows保持同一容量规则。
            # 实际compact行仍由原num_reqs_actual/start_pos/Q在设备上决定。
            if compress_ratio <= 1:
                raise ValueError("压缩metadata必须对应真实的Native压缩比")
            decoded.num_compressed_tokens = min(hidden_tokens, hidden_tokens // compress_ratio + item.num_decodes)
        adapted = copy(item)
        adapted.decode = decoded
        adapted.num_decode_tokens = hidden_tokens
        result.append(adapted)
    return result


@dataclass(frozen=True)
class TNDGraphContract:
    runtime: Any
    token_capacity: int
    signature: tuple | None

    def allows_replay(self, context) -> bool:
        if self.signature is None:
            return False
        return self.runtime.graph_signature(context, self.token_capacity) == self.signature


def tensor_descriptor(value):
    if value is None:
        return None
    return (
        tuple(value.shape),
        value.stride(),
        value.dtype,
        value.device,
        value.data_ptr(),
    )
