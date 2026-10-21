# SPDX-License-Identifier: Apache-2.0
"""局部Score到Attention的独立代际信号；不复用前段或输出链的工作区。"""

import pypto.language as pl

from .query_phase_sync import SIGNAL_SLOT_WORDS

# 显式再导出供中段依赖/编排使用，不能按本文件未引用来删除。
from .query_phase_sync import begin_invocation as begin_invocation
from .query_phase_sync import publish_phase as publish_phase
from .query_phase_sync import wait_row_owners as wait_row_owners
from .runtime_abi import CORES
from .runtime_abi import GROUPS as GROUPS
from .runtime_abi import VECTORS as VECTORS

SIGNAL_WORDS = pl.dynamic("CSA_MIDDLE_SIGNAL_WORDS")
SIGNAL_COUNT = 4 * CORES * SIGNAL_SLOT_WORDS
PHASE_SCORE_VECTOR = 1
PHASE_TOPK = 2
PHASE_ATTN_PLAN = 3
