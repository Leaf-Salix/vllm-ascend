# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""本次高层源码抽取的可复现记录；不读取历史 IR 或生成 entry。

只在维护时执行，正式 hc_stages.py 不依赖此脚本。抽取源文件中的数学语句，
把外层任务改为显式 worker 的动态范围循环；未创建静态 shape 白名单。
"""

import ast
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa"
TARGET = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/hc_stages.py"


def funcs(path):
    return {n.name: n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef)}


def math_loop(function, hint):
    for node in ast.walk(function):
        if isinstance(node, (ast.For, ast.With)):
            call = node.iter if isinstance(node, ast.For) else node.items[0].context_expr
            if isinstance(call, ast.Call) and any(
                k.arg == "name_hint" and isinstance(k.value, ast.Constant) and k.value.value == hint
                for k in call.keywords
            ):
                return copy.deepcopy(node)
    raise ValueError(hint)


class Rename(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(self.mapping.get(node.id, node.id), node.ctx), node)


class DivisionPath(ast.NodeTransformer):
    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == "row_recip":
            return [self.visit(n) for n in node.orelse]
        return self.generic_visit(node)


def wrapped_math(body, index, worker, workers, count):
    loop = ast.parse(f"""
for wave in pl.range(({count} + {workers} - 1) // {workers}):
    {index} = {worker} + wave * {workers}
    if {index} < {count}:
        pass
""").body[0]
    loop.body[1].body = body
    return ast.unparse(ast.fix_missing_locations(loop))


def section(signature, doc, setup, body, index, worker, workers, count, returned):
    text = f'@pl.jit.inline(auto_scope=False)\ndef {signature}:\n    """{doc}"""\n'
    text += "\n".join("    " + line for line in setup.splitlines()) + "\n"
    text += "\n".join("    " + line for line in wrapped_math(body, index, worker, workers, count).splitlines())
    return text + f"\n    return {returned}\n\n\n"


def main():
    hc = funcs(SOURCE / "hc_pre.py")
    root = funcs(SOURCE / "decode_csa.py")["_decode_csa_tp1_layer"]
    gates = hc["hc_pre_gates_from_rms"]
    widen = math_loop(root, "hc_widen_rms").body[0].body
    widen = [Rename({"HC_WIDEN_T_TILE": "T_TILE", "HC_WIDEN_D_TILE": "RMS_K_TILE"}).visit(n) for n in widen]
    project = math_loop(gates, "hc_pre_linear").body[0].body
    reduce = math_loop(gates, "hc_pre_linear_reduce").body
    split = math_loop(gates, "split_pre_post").body[-1].body
    comb = [DivisionPath().visit(n) for n in math_loop(gates, "comb_sinkhorn").body]
    comb = [part for n in comb for part in (n if isinstance(n, list) else [n])]
    mix = math_loop(hc["hc_mix_norm"], "mix_x_rms_norm").body
    mix[0].value.left = ast.Name("token_block", ast.Load())

    header = '''# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""完整动态 CSA 的 HC 核内阶段；保留原算术，不创建独立任务。

这里是可维护的高层 PyPTO 源码，不是固定 B16 的展开 IR。T 从输入张量的
动态轴读取；显式 group/vector worker 决定每核范围，所有波次在核内循环。
本文件没有 pl.spmd、submit、全核屏障或核间通知。完整 CSA 的调度层负责：
HC_INPUT → HC_PARTIAL → HC_RAW → {HC_GATES、HC_COMB}，HC_GATES → HC_NORM。
发布须在该核 payload 写回完成后，消费者须等待对应 producer 并维护数据缓存。

首版保留原六个数学阶段和 GM 中间量，供新的单 SPMD 控制层嵌入；没有宣称
已完成尾段融合或性能优化。FP32 投影输入、BF16 norm 权重与所有舍入点不变。
原数学取自相邻 deepseek_v4_flash_csa/{decode_csa,hc_pre,rmsnorm}.py。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.rmsnorm import rms_norm_apply, rms_norm_inverse

T_DYN = pl.dynamic("HC_STAGE_T")
PAD_ROWS_DYN = pl.dynamic("HC_STAGE_PAD16_ROWS")
PARTIAL_ROWS_DYN = pl.dynamic("HC_STAGE_PARTIAL_ROWS")
MIX_ROWS_DYN = pl.dynamic("HC_STAGE_PAD8_ROWS")

# DSV4 Flash 模型维度及已验证数学分块；这些不是 batch/sequence 白名单。
D = 4096
HC_MULT = 4
HC_DIM = HC_MULT * D
HC_DIM_INV = 1.0 / HC_DIM
MIX_HC = 24
MIX_PAD = 32
HC_PAD = 8
T_TILE = 8
COMB_T_TILE = T_TILE
LINEAR_T_TILE = 16
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
LINEAR_K_TILE = 256
RMS_K_TILE = 512
D_TILE = 256
HC_SINKHORN_ITER = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6


def workspace_contract(tokens):
    """Host 只按已有 shape 描述分配空间，不读取设备元素。

    tokens 是本次实际 T；若使用容量缓冲，应传下列实际形状的连续子视图。
    group_workers/vector_workers 必须为正，worker 为零起点且小于对应数量。
    所有核按相同分块公式分工；一个 token 行块只归属一个 worker。
    四份 tail scratch 各仅被全局唯一的最后短块写入，不能用于改变行分工后
    多个 worker 同时拥有短块的方案；并发执行整层必须拥有独立工作区。
    """
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("HC 工作区要求正整数实际 token 数")
    padded16 = (tokens + LINEAR_T_TILE - 1) // LINEAR_T_TILE * LINEAR_T_TILE
    padded8 = (tokens + T_TILE - 1) // T_TILE * T_TILE
    return {
        "x_fp32": {"shape": [tokens, HC_DIM], "dtype": "FP32"},
        "inv_rms": {"shape": [padded16, 1], "dtype": "FP32"},
        "widen_tail": {"shape": [T_TILE, HC_DIM], "dtype": "FP32"},
        "mixes_partials": {"shape": [LINEAR_OK * padded16, MIX_PAD], "dtype": "FP32"},
        "mixes_raw": {"shape": [padded16, MIX_PAD], "dtype": "FP32"},
        "pre_val_store": {"shape": [padded16, HC_PAD], "dtype": "FP32"},
        "post_tail_store": {"shape": [T_TILE, HC_PAD], "dtype": "FP32"},
        "comb_tail_store": {"shape": [COMB_T_TILE, HC_PAD * HC_MULT], "dtype": "FP32"},
        "x_mixed": {"shape": [padded8, D], "dtype": "BF16"},
        "x_normed_tail_store": {"shape": [T_TILE, D], "dtype": "BF16"},
    }


'''
    text = header
    text += section(
        "hc_widen_rms_range(\n    x: pl.Tensor[[T_DYN, HC_DIM], pl.BF16],\n    x_fp32: pl.Out[pl.Tensor[[T_DYN, HC_DIM], pl.FP32]],\n    inv_rms: pl.Out[pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32]],\n    widen_tail: pl.InOut[pl.Tensor[[T_TILE, HC_DIM], pl.FP32]],\n    vector_worker: pl.Scalar[pl.INDEX], vector_workers: pl.Scalar[pl.INDEX],\n)",
        "AIV：BF16 无损扩展并按原 32 个 K512 顺序累加输入平方和。\n\n    调用方在本核返回后发布 HC_INPUT；AIC 投影及后续门值消费者等待所需行。\n    尾行显式恢复 valid_shape 并清零后归约，避免 cast 丢失有效行标记。",
        "t_dim = pl.tensor.dim(x, 0)\nx_hc_flat = x\nx_hc32_flat = x_fp32\nwiden_rows = (t_dim + T_TILE - 1) // T_TILE",
        widen,
        "widen_blk",
        "vector_worker",
        "vector_workers",
        "widen_rows",
        "x_fp32, inv_rms",
    )
    text += section(
        "hc_project_range(\n    x: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],\n    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],\n    mixes_partials: pl.Out[pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32]],\n    group_worker: pl.Scalar[pl.INDEX], group_workers: pl.Scalar[pl.INDEX],\n)",
        "AIC：M16、四段 K4096，每段仍按原 16 次 K256 顺序累加。\n\n    调用前等待 HC_INPUT，确保所有本核输入行的 FP32 写回完成；返回后发布\n    HC_PARTIAL。逻辑 task=(行块×4+K段)，多波次属于同一组核，不另派任务。",
        "t_dim = pl.tensor.dim(x, 0)\nt_linear = (t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE * LINEAR_T_TILE\nx_flat = x\nlinear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK",
        project,
        "task",
        "group_worker",
        "group_workers",
        "linear_units",
        "mixes_partials",
    )
    text += section(
        "hc_reduce_range(\n    mixes_partials: pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32],\n    mixes_raw: pl.Out[pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32]],\n    vector_worker: pl.Scalar[pl.INDEX], vector_workers: pl.Scalar[pl.INDEX],\n)",
        "AIV：原四份投影按 ((p0+p1)+p2)+p3 顺序归约。\n\n    调用前等待所需四段 HC_PARTIAL；返回后发布 HC_RAW。按完整 M16\n    归约含 padding 的输出行，不把四份左折叠改成树形 col_sum。",
        "t_linear = pl.tensor.dim(mixes_raw, 0)\nlinear_blocks = t_linear // LINEAR_T_TILE",
        reduce,
        "linear_block",
        "vector_worker",
        "vector_workers",
        "linear_blocks",
        "mixes_raw",
    )
    text += section(
        "hc_gates_range(\n    inv_rms: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],\n    mixes_raw: pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32],\n    hc_scale: pl.Tensor[[3], pl.FP32], hc_base: pl.Tensor[[MIX_HC], pl.FP32],\n    pre_val_store: pl.Out[pl.Tensor[[PAD_ROWS_DYN, HC_PAD], pl.FP32]],\n    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],\n    post_tail_store: pl.InOut[pl.Tensor[[T_TILE, HC_PAD], pl.FP32]],\n    vector_worker: pl.Scalar[pl.INDEX], vector_workers: pl.Scalar[pl.INDEX],\n)",
        "AIV：保留原 pre/post sigmoid、scale 乘法和 eps 顺序。\n\n    调用前等待 HC_INPUT 与 HC_RAW；返回后发布 HC_GATES，供 mix/norm\n    和最终 HCpost 消费。pre 按八列物理布局保留，post 只有四列有效。",
        "t_dim = pl.tensor.dim(post, 0)\ntoken_tiles = (t_dim + T_TILE - 1) // T_TILE\nscale0 = pl.read(hc_scale, [0])\nscale1 = pl.read(hc_scale, [1])",
        split,
        "ob",
        "vector_worker",
        "vector_workers",
        "token_tiles",
        "pre_val_store, post",
    )
    text += section(
        "hc_comb_range(\n    inv_rms: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],\n    mixes_raw: pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32],\n    hc_scale: pl.Tensor[[3], pl.FP32], hc_base_2d: pl.Tensor[[1, MIX_HC], pl.FP32],\n    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],\n    comb_tail_store: pl.InOut[pl.Tensor[[COMB_T_TILE, HC_PAD * HC_MULT], pl.FP32]],\n    vector_worker: pl.Scalar[pl.INDEX], vector_workers: pl.Scalar[pl.INDEX],\n)",
        "AIV：CSA 原除法路径的二十轮 Sinkhorn，保持行列归约树。\n\n    调用前等待 HC_INPUT 与 HC_RAW；与 hc_gates_range/mix 独立，返回后\n    发布 HC_COMB。不引入 HCA 的 recip 替代路径，不改变迭代次数。",
        "t_dim = pl.tensor.dim(comb, 0)\ntoken_tiles = (t_dim + COMB_T_TILE - 1) // COMB_T_TILE",
        comb,
        "ob",
        "vector_worker",
        "vector_workers",
        "token_tiles",
        "comb",
    )
    text += section(
        "hc_mix_norm_range(\n    x: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],\n    pre_val_store: pl.Tensor[[PAD_ROWS_DYN, HC_PAD], pl.FP32],\n    norm_w: pl.Tensor[[D], pl.BF16],\n    x_mixed: pl.InOut[pl.Tensor[[MIX_ROWS_DYN, D], pl.BF16]],\n    x_normed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],\n    x_normed_tail_store: pl.InOut[pl.Tensor[[T_TILE, D], pl.BF16]],\n    vector_worker: pl.Scalar[pl.INDEX], vector_workers: pl.Scalar[pl.INDEX],\n)",
        "AIV：四路混合、BF16 舍入和完整 D4096 RMSNorm，原序不变。\n\n    调用前等待 HC_GATES 以及本核输入行；返回后发布 HC_NORM。每行始终\n    由一个 worker 完成全部 D，保留16个K256平方和的顺序，不新增跨核RMS。\n    norm_w 直接接 Native BF16，不在初始化时转换为 FP32。",
        "t_dim = pl.tensor.dim(x, 0)\nx_flat = x\ntoken_tiles = (t_dim + T_TILE - 1) // T_TILE",
        mix,
        "token_block",
        "vector_worker",
        "vector_workers",
        "token_tiles",
        "x_mixed, x_normed",
    )
    ast.parse(text)
    if TARGET.exists():
        raise FileExistsError("拒绝覆盖已维护的源文件：" + str(TARGET))
    TARGET.write_text(text)
    print(TARGET)


if __name__ == "__main__":
    main()
