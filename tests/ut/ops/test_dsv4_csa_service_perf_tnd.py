# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the performance build's TND (variable-length request) contract.

精度版仍要求每请求恰好六行，性能版按 query_start_loc 取每请求的 token 区间。
这里只校验宿主侧的判据与 ABI 绑定，不跑 kernel。
"""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.service import CSAServiceRuntime
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.service_config import QUERY_TOKENS

GROUPS = ("swa", "compressed", "state", "indexer", "indexer_state")
LAYER = "model.layers.2.attn"


def _bounds(lengths):
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    return torch.tensor(starts, dtype=torch.int32)


def _context(lengths, bucket):
    """一批请求，每个 lengths[i] 行；bucket 是档位容量，尾部是图填充行。"""
    batch = len(lengths)
    runtime = CSAServiceRuntime.__new__(CSAServiceRuntime)
    runtime.batch_capacity = 40
    runtime.layer_name = LAYER
    runtime.prefixes = {name: name for name in GROUPS}
    cos = torch.ones((bucket, 64), dtype=torch.float32)
    items = {}
    for prefix in GROUPS:
        req = NS(
            seq_lens=torch.full((batch,), 128, dtype=torch.int32),
            query_start_loc=_bounds(lengths),
            num_reqs_actual=batch,
            max_seqlen_q=max(lengths),
            ori_win_right=0,
            ori_win_left=127,
            dspark_swa_indices=None,
            cos={LAYER: cos},
            sin={LAYER: cos},
        )
        items[prefix] = NS(num_prefills=0, num_actual_tokens=sum(lengths), num_decodes=batch, decode=req)
    return runtime, NS(attn_metadata=items, is_draft_model=False)


def _hidden(bucket):
    return (torch.empty((bucket, 4, 4096), dtype=torch.bfloat16), torch.empty(bucket, dtype=torch.int64))


@pytest.mark.parametrize(
    "lengths,bucket,expected",
    [
        # 满档等长：与定长档位完全一样，必须仍然接受。
        ([6, 6, 6, 6], 24, True),
        # 变长：这正是定长判据 tokens % QUERY_TOKENS 会误拒的形状。
        ([6, 3, 1], 24, True),
        # 每请求一行（全部只接受了一个 token）。
        ([1, 1, 1, 1], 24, True),
        # 单请求不满长。
        ([4], 6, True),
        # 超过 S：spec decode 一步最多 QUERY_TOKENS 行。
        ([7, 6], 24, False),
        # 请求数超过档位能放下的行数。
        ([1] * 8, 4, False),
    ],
)
def test_perf_gate_accepts_variable_length_requests(lengths, bucket, expected):
    runtime, context = _context(lengths, bucket)
    assert runtime.eligible(context, *_hidden(bucket)) is expected


def test_perf_gate_rejects_groups_that_disagree_on_request_count():
    """各 cache group 必须看到同一批请求，否则共用一份 query_start_loc 就是错的。"""
    runtime, context = _context([6, 3, 1], 24)
    context.attn_metadata["indexer"].num_decodes = 2
    assert runtime.eligible(context, *_hidden(24)) is False


def test_perf_gate_still_rejects_prefill_and_draft():
    for mutate in (
        lambda c: setattr(c.attn_metadata["compressed"], "num_prefills", 1),
        lambda c: setattr(c, "is_draft_model", True),
        lambda c: setattr(c.attn_metadata["swa"].decode, "ori_win_left", 255),
    ):
        runtime, context = _context([6, 3, 1], 24)
        mutate(context)
        assert runtime.eligible(context, *_hidden(24)) is False


def _release_groups(lengths, bucket):
    """用 release 的 metadata dataclass 搭一组最小视图，只为走通 ABI 绑定。"""
    from vllm_ascend.attention.dsa_v1 import AscendDSADecodeMetadata, AscendDSAMetadata

    batch, pages = len(lengths), 4
    bounds = _bounds(lengths)
    seq_lens = torch.full((batch,), 128, dtype=torch.int32)
    positions = torch.arange(bucket, dtype=torch.int64)
    cos = torch.ones((bucket, 64), dtype=torch.float32)
    sin = torch.zeros_like(cos)
    groups, compact = {}, {}
    for name in GROUPS:
        table = torch.zeros((batch, 8), dtype=torch.int32)
        slots = None if name in ("compressed", "indexer") else torch.zeros((bucket, 2), dtype=torch.int32)
        req = AscendDSADecodeMetadata(
            input_positions=positions,
            block_table=table,
            seq_lens=seq_lens,
            max_seqlen_kv=128,
            max_seqlen_q=max(lengths),
            seq_lens_list=[128] * batch,
            max_seq_lens=128,
            slot_mapping=slots,
            block_size=2 if "state" in name else 32,
            query_start_loc=bounds,
            query_start_loc_cpu=bounds,
            num_reqs_actual=batch,
            cos={LAYER: cos},
            sin={LAYER: sin},
        )
        metadata = AscendDSAMetadata(
            num_actual_tokens=sum(lengths),
            slot_mapping=slots,
            query_start_loc=bounds,
            seq_lens=seq_lens,
            block_tables=table,
            sin=sin,
            cos=cos,
            num_decodes=batch,
            num_decode_tokens=sum(lengths),
            num_prefills=0,
            decode=req,
        )
        if name == "indexer":
            storage = torch.empty((pages, 4160), dtype=torch.int8)
            key = storage.as_strided((pages, 32, 1, 128), (4160, 128, 128, 1))
            scale = storage.view(torch.float16).as_strided((pages, 32, 1, 1), (2080, 1, 1, 1), 2048)
            views = (key, scale)
        elif "state" in name:
            width = 512 if name == "indexer_state" else 2048
            storage = torch.empty((pages, 1040 if width == 512 else 8192), dtype=torch.float32)
            views = (storage.as_strided((pages, 2, 1, width), (storage.shape[1], width, width, 1)),)
        else:
            views = (torch.empty((pages, 32, 1, 512), dtype=torch.bfloat16),)
        groups[name] = (metadata, views)
        if name in ("compressed", "indexer"):
            compact[name] = (cos[:2], sin[:2], torch.zeros((2, 2), dtype=torch.int32))
    return groups, compact, bounds, positions


def test_perf_abi_binds_token_level_query_start_loc():
    """token 级边界绑的是 swa 组：ori_slot_mapping 与 position_ids 都按这条流索引。"""
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_csa import decode_csa_tp1_layer_test
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.native_adapter import NativeCSACall

    lengths, bucket = [6, 3, 1], 24
    groups, compact, bounds, positions = _release_groups(lengths, bucket)
    hidden = torch.empty((bucket, 4, 4096), dtype=torch.bfloat16)
    weights = dict.fromkeys(decode_csa_tp1_layer_test.param_names, torch.empty(0))
    call = NativeCSACall(
        NS(attention=Mock()), weights, hidden, positions, groups, layer_name=LAYER, compact_metadata=compact
    )
    assert call.args["query_start_loc"] is bounds
    # 三个边界张量语义不同，不能互相顶替：这两个是压缩行/indexer 行的边界。
    assert call.args["cmp_query_start_loc"] is groups["compressed"][0].decode.query_start_loc
    assert call.args["idx_query_start_loc"] is groups["indexer"][0].decode.query_start_loc
    assert "query_start_loc" in decode_csa_tp1_layer_test.param_names


def test_precision_abi_still_requires_uniform_requests():
    """性能版放开的是它自己那条路；精度版 kernel 没做 TND，约束必须还在。"""
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_adapter import NativeCSACall as PrecisionCall

    lengths, bucket = [6, 3, 1], 24
    groups, compact, _, positions = _release_groups(lengths, bucket)
    hidden = torch.empty((bucket, 4, 4096), dtype=torch.bfloat16)
    weights = {}
    with pytest.raises(ValueError, match=f"{QUERY_TOKENS} unpadded rows per request"):
        PrecisionCall(
            NS(attention=Mock()), weights, hidden, positions, groups, layer_name=LAYER, compact_metadata=compact
        )
