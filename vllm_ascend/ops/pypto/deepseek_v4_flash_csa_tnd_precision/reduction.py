# SPDX-License-Identifier: Apache-2.0
"""CSA/HCA 共用的编译期规约配置；固定规约用单 K 分片消除跨核累加。"""

import pypto.language as pl

from vllm_ascend import envs

ATOMIC_ADD = envs.VLLM_ASCEND_PTO_CSA_ATOMIC_ADD
if ATOMIC_ADD not in (0, 1):
    raise ValueError("VLLM_ASCEND_PTO_CSA_ATOMIC_ADD 只接受 0/1")
STORE_ATOMIC = pl.AtomicType.Add if ATOMIC_ADD else pl.AtomicType.None_


def validate_reduction_mode():
    """初始化后不能让环境开关与已编译的策略不一致。"""
    if ATOMIC_ADD != 0:
        raise ValueError("TND精度版要求 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0")
    if envs.VLLM_ASCEND_PTO_CSA_ATOMIC_ADD != ATOMIC_ADD:
        raise ValueError("PTO CSA 规约配置已在导入时固定，切换 atomic add 需要新进程")
