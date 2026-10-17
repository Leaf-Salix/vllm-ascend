# SPDX-License-Identifier: Apache-2.0
"""一次性从当前高层decode_indexer抽取源级helper；不读取冻结IR，不覆盖维护版。

该脚本只用于记录迁移来源，正式模块不会导入测试文件。所有数学表达式按原
AST搬运，只有worker、动态容量和外层分工改变；后续手写中文注释不影响数学。
"""

import ast
import copy
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
SOURCE = REPOSITORY / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_indexer.py"
TARGET = REPOSITORY / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_query_stages.py"

HEADER = '''# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""动态CSA的Indexer query前处理：六个显式worker的核内数学阶段。

一、范围与阅读顺序
    query_project→dequant_rope→hadamard→query_quant形成query分支；
    weights_project→weights_reduce形成独立head权重分支。只复用当前高层
    decode_indexer.py数学，不导入历史IR，也不包含历史QK/TopK MIX。

二、布局与动态容量
    QR是INT8[T,1024]，Indexer投影权重是ND INT8[1024,8192]，输出INT32
    投影至少pad64(T)行。RoPE后的BF16为[pad16(T),8192]，外层还须对同一
    数据建立[pad16(T)*64,128]的GM别名供Hadamard使用；helper不在incore内
    reshape整张GM。Hadamard矩阵为BF16[128,128]，FP32中间量至少T*64行，
    INT8结果和FP32 scale各有独立动态行容量，至少T*64，后续消费者可要求
    pad16(T)*64。weights分支输入x_normed BF16[T,4096]和ND BF16[4096,64]，
    partial至少4*pad16(T)行，输出FP32至少pad16(T)行。
    T取真实Tensor轴；Hadamard/量化/weights_reduce的tokens由外层相同T传入，
    不能用较大的容量推导有效token。此范围不使用S/请求边界或Host取值。

三、分工与同步合同
    所有helper只有普通核内range/pipeline，没有spmd/submit或task依赖。
    query投影按N256列块轮转AIC、整条K1024权重供所有M64行块复用；反量化
    按8行×4head轮转AIV，T不足8或最后尾行沿用逐行数学。Hadamard与量化
    按64个head行一块，因每token正好64heads，任意整数T均没有head行尾块。
    weights按M16行块×4个K1024 split轮转AIC，reduce按M16轮转AIV。
    外层负责真实producer通知/等待、业务fence和缓存维护；return仅表达GM
    别名。本核全部wave结束后才可发布phase，缓冲在消费者完成前不能覆盖。

四、数值顺序与当前边界
    以当前canonical源码为准，dequant严格保持(acc*activation_scale)*weight_scale，
    不混入历史“先合并scale”的变体。Hadamard后乘128**-0.5，amax先下限再
    两个64列块，scale=127/amax→recip，量化rint→FP16 round→INT8 trunc。
    weights四份partial按原左折叠相加，再乘WEIGHTS_SCALE，保留FP32输出。
    gamma/权重不做Host转换；数学等价审计和CPU后端不能证明设备精度、同步或性能。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.config import FLASH as M
from ..deepseek_v4_flash_csa.config import INT8_AMAX_EPS, INT8_SCALE_MAX

D = M.hidden_size
Q_LORA = M.q_lora_rank
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
WEIGHTS_SCALE = M.index_weights_scale
HADAMARD_SCALE = IDX_HEAD_DIM**-0.5
QR_MM_ROW_TILE = 64
QR_MM_N_TILE = 256
DEQUANT_T_TILE = 8
DQ_ROPE_H_TILE = 4
QH_MM_TILE = 64
QH_QUANT_TILE = 64
QH_HEAD_DIM_TILE = 64
MM_ROW_TILE = 16
WEIGHTS_OK = 4
WEIGHTS_K_TILE = D // WEIGHTS_OK
D_TILE = 512

T_DYN = pl.dynamic("CSA_INDEX_QUERY_T")
PROJ_ROWS_DYN = pl.dynamic("CSA_INDEX_PROJ_PAD64")
ROPE_ROWS_DYN = pl.dynamic("CSA_INDEX_ROPE_PAD16")
ROPE_HEAD_ROWS_DYN = pl.dynamic("CSA_INDEX_ROPE_HEAD_ROWS")
HADAMARD_ROWS_DYN = pl.dynamic("CSA_INDEX_HADAMARD_ROWS")
QUANT_ROWS_DYN = pl.dynamic("CSA_INDEX_QUANT_ROWS")
SCALE_ROWS_DYN = pl.dynamic("CSA_INDEX_SCALE_ROWS")
WEIGHTS_ROWS_DYN = pl.dynamic("CSA_INDEX_WEIGHTS_PAD16")
WEIGHTS_PARTIAL_ROWS_DYN = pl.dynamic("CSA_INDEX_WEIGHTS_PARTIAL_ROWS")
'''

SIGNATURES = {
    "indexer_query_project_stage": """qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    qr_acc_pad: pl.Tensor[[PROJ_ROWS_DYN, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT32],
    worker_id: pl.Scalar[pl.INDEX], WORKERS: pl.constexpr,""",
    "indexer_query_dequant_rope_stage": """qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    qr_acc_pad: pl.Tensor[[PROJ_ROWS_DYN, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT32],
    qr_bf16_2d: pl.Tensor[[ROPE_ROWS_DYN, IDX_N_HEADS * IDX_HEAD_DIM], pl.BF16],
    worker_id: pl.Scalar[pl.INDEX], WORKERS: pl.constexpr,""",
    "indexer_query_hadamard_stage": """qr_bf16: pl.Tensor[[ROPE_HEAD_ROWS_DYN, IDX_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    qh_acc_gm: pl.Tensor[[HADAMARD_ROWS_DYN, IDX_HEAD_DIM], pl.FP32],
    tokens: pl.Scalar[pl.INDEX], worker_id: pl.Scalar[pl.INDEX], WORKERS: pl.constexpr,""",
    "indexer_query_quant_stage": """qh_acc_gm: pl.Tensor[[HADAMARD_ROWS_DYN, IDX_HEAD_DIM], pl.FP32],
    qr_hadamard_i8: pl.Tensor[[QUANT_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    qr_hadamard_scale_dq: pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32],
    tokens: pl.Scalar[pl.INDEX], worker_id: pl.Scalar[pl.INDEX], WORKERS: pl.constexpr,""",
    "indexer_weights_project_stage": """x: pl.Tensor[[T_DYN, D], pl.BF16],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    weights_partial: pl.Tensor[[WEIGHTS_PARTIAL_ROWS_DYN, IDX_N_HEADS], pl.FP32],
    worker_id: pl.Scalar[pl.INDEX], WORKERS: pl.constexpr,""",
    "indexer_weights_reduce_stage": """weights_partial: pl.Tensor[[WEIGHTS_PARTIAL_ROWS_DYN, IDX_N_HEADS], pl.FP32],
    weights: pl.Tensor[[WEIGHTS_ROWS_DYN, IDX_N_HEADS], pl.FP32],
    tokens: pl.Scalar[pl.INDEX], worker_id: pl.Scalar[pl.INDEX], WORKERS: pl.constexpr,""",
}


class Rename(ast.NodeTransformer):
    def visit_Name(self, node):
        mapping = {
            "QR_PROJ_WORKERS": "WORKERS",
            "QH_WORKERS": "WORKERS",
            "QH_QUANT_WORKERS": "WORKERS",
            "weights_workers": "WORKERS",
            "T_PAD": "t_pad",
        }
        return ast.copy_location(ast.Name(id=mapping.get(node.id, node.id), ctx=node.ctx), node)


def code(nodes):
    return "\n".join(ast.unparse(Rename().visit(copy.deepcopy(node))) for node in nodes)


def emit(name, doc, body):
    raw = f"@pl.jit.inline(auto_scope=False)\ndef {name}(\n    {SIGNATURES[name]}\n):\n"
    raw += '    """' + doc + '"""\n'
    return raw + "\n".join("    " + line if line else "" for line in body.splitlines()) + "\n"


def extract():
    tree = ast.parse(SOURCE.read_text())
    funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    qr = funcs["indexer_qr_rope"]
    mm = next(node for node in qr.body if isinstance(node, ast.With))
    dq = next(node for node in qr.body if isinstance(node, ast.For))
    qh = funcs["indexer_qr_hadamard_mm"]
    had, quant = [node for node in qh.body if isinstance(node, ast.With)]
    weights = funcs["indexer_weights_project"]
    wp, wr = [node for node in weights.body if isinstance(node, ast.With)]
    result = [HEADER]
    result.append(
        emit(
            "indexer_query_project_stage",
            "AIC：按N256条带投影，先等QR；返回后发布query投影完成。",
            "bs = pl.tensor.dim(qr, 0)\nrow_blocks = (bs + QR_MM_ROW_TILE - 1) // QR_MM_ROW_TILE\n"
            "qr_proj_worker = worker_id\n" + code(mm.body[1:]) + "\nreturn qr_acc_pad",
        )
    )
    result.append(
        emit(
            "indexer_query_dequant_rope_stage",
            "AIV：等query INT32投影与QR scale，原反量化/RoPE后写BF16，外层发布供Hadamard消费。",
            "bs = pl.tensor.dim(qr_scale, 0)\n"
            "dq_rope_units = ((bs + DEQUANT_T_TILE - 1) // DEQUANT_T_TILE) * (IDX_N_HEADS // DQ_ROPE_H_TILE)\n"
            "dq_rope_workers = pl.min(dq_rope_units, WORKERS)\n"
            "if worker_id < dq_rope_workers:\n    dq_rope_worker = worker_id\n"
            + "\n".join("    " + line for line in code(dq.body).splitlines())
            + "\nreturn qr_bf16_2d",
        )
    )
    result.append(
        emit(
            "indexer_query_hadamard_stage",
            "AIC：先等BF16 query所有head；64行一块即一个token，保持原完整K128矩阵乘。",
            "bs_heads = tokens * IDX_N_HEADS\nqh_worker = worker_id\n" + code(had.body[1:]) + "\nreturn qh_acc_gm",
        )
    )
    result.append(
        emit(
            "indexer_query_quant_stage",
            "AIV：先等Hadamard FP32，逐64head行归一比例并量化；输出INT8与独立FP32 scale。",
            "bs_heads = tokens * IDX_N_HEADS\nqh_quant_worker = worker_id\n"
            + code(quant.body[1:])
            + "\nreturn qr_hadamard_i8, qr_hadamard_scale_dq",
        )
    )
    result.append(
        emit(
            "indexer_weights_project_stage",
            "AIC：等x_normed；M16×四段K1024，每段两个K512原顺序，写动态四份partial。",
            "bs = pl.tensor.dim(x, 0)\nrow_blocks = (bs + MM_ROW_TILE - 1) // MM_ROW_TILE\n"
            "t_pad = row_blocks * MM_ROW_TILE\nx_flat = x\nw_worker = worker_id\n"
            + code(wp.body[1:])
            + "\nreturn weights_partial",
        )
    )
    result.append(
        emit(
            "indexer_weights_reduce_stage",
            "AIV：等当前M16的四个投影split，原左折叠后乘模型scale，保持FP32给score消费。",
            "row_blocks = (tokens + MM_ROW_TILE - 1) // MM_ROW_TILE\nt_pad = row_blocks * MM_ROW_TILE\n"
            "for w_rb in pl.range(worker_id, row_blocks, WORKERS):\n"
            + "\n".join("    " + line for line in code(wr.body[1:]).splitlines())
            + "\nreturn weights",
        )
    )
    return "\n\n".join(result)


if __name__ == "__main__":
    assert not TARGET.exists(), "正式维护文件已存在，不允许覆盖；需要审计时调用extract()比较数学"
    TARGET.write_text(extract())
    (HERE / "reference_decode_indexer.py").write_text(SOURCE.read_text())
