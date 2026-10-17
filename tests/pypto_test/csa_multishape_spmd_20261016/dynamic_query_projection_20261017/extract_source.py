# SPDX-License-Identifier: Apache-2.0
"""从原高层数学源码抽取显式 worker helper；只生成源码，不导入编译器。

生成物保留动态 Tensor 轴，不能输入历史 IR 或实际 B。生产文件生成一次后独立
维护，运行时不依赖本脚本。复现时 --output 必须指向新文件，不覆盖已有实现。
"""

import argparse
import ast
import copy
from pathlib import Path


HEADER = '''"""动态 CSA 的 Q_B 投影、query 反量化/RMS/RoPE 与 RoPE 准备阶段。

输入输出和动态变量
    Q_B 读取 QR INT8 暂存与 Native INT8 wq_b[1024,32768]，输出 INT32 投影。
    tokens 是外层从真实输入 Tensor 维度取得的设备 Scalar，不是 Host 固定 B。
    反量化阶段接收每行 QR scale、每列权重 scale、cos/sin/swap，写 BF16
    q_flat[T,32768]；它必须与下游 q[T,64,512] 共用同一块 GM。
    T、QR 暂存行数和投影工作区行数均为动态 Tensor 轴。

数学 tile 与分工
    NZ 保留 M128/N256/K256，ND 保留 M64/N512/K128；按实际 T 计算
    ceil(T/128)*128 个工作区行，不按缓存容量扫描额外行。每个显式 AIC
    worker 负责列块 n=worker+k*WORKERS，沿 K 顺序做原 INT32 累加。
    反量化仍用八行、4/16/64 个 head 的原分组策略；原阈值以 48 AIV 为
    基准，WORKERS 只决定谁执行，不擅自改变数学分组。RoPE 准备按八行条带。

数据流、同步与所有权
    三个 helper 内没有 spmd/submit/notify。外层须等 QR 的 INT8 写回再运行
    Q_B；反量化同时依赖相关 Q_B 列块、QR scale 和 RoPE 准备写回。数据
    fence、通知和等待由统一 SPMD 编排负责，返回 Tensor 不代表跨核完成。
    Q_B 列块 owner=(column//N_TILE)%WORKERS。反量化的八行/head 组编号
    work=(row//8)*(64/head_tile)+head//head_tile，其 owner=work%WORKERS；
    每个 head 完整包含两个 NZ N256 列块。RoPE owner=(row//8)%active，
    active=min(ceil(T/8),WORKERS)。各阶段只写自己负责的独立行/列片段。

GM 视图与工作区合同
    q_flat 是外层创建的无损二维 GM 别名；不得在 incore helper 内把三维 GM
    reshape 后当作另一个 GM Tensor 再 load/store。QR 和投影暂存至少预留
    ceil(T/128)*128 行，scale 暂存至少 ceil(T/8)*8 行；有效输出只写真实 T。
    ND 会计算暂存的填充行，这些行不参与有效输出；未写容量尾部不保证清零。
    外层须直接在 spmd 体设置 wq_b 的 BYPASS 属性，并验证工作区不互相覆盖。

数值与范围边界
    数学来自同仓 q_projection.py/qkv_proj_rope.py，不使用冻结 IR。保留
    (INT32→FP32 * activation_scale) * weight_scale、512 列规约和原 RoPE
    运算顺序；完整八行用 rsqrt，尾行沿用 recip(sqrt)，不混成另一套公式。
    NZ 仅按 Native 同一模式声明，不转换权重、不执行三维权重格式探针。
    本文件只补齐 query 分支，不包含 KV、Indexer、Attention 或完整 CSA；
    源码/CPU 编译检查不能代替真实设备精度、跨核同步或性能验证。
"""

import pypto.language as pl

from vllm_ascend import envs

from ..deepseek_v4_flash_csa.config import FLASH as M

WEIGHT_NZ_MODE = envs.VLLM_ASCEND_ENABLE_NZ
if WEIGHT_NZ_MODE not in (0, 1, 2):
    raise ValueError("VLLM_ASCEND_ENABLE_NZ 只接受 0/1/2")
QUANT_WEIGHT_NZ = WEIGHT_NZ_MODE >= 1
QUANT_WEIGHT_LAYOUT = pl.NZ if QUANT_WEIGHT_NZ else None

H = M.num_attention_heads
HEAD_DIM = M.head_dim
Q_LORA = M.q_lora_rank
ROPE_DIM = M.qk_rope_head_dim
ROPE_DIM_SCALE = float(ROPE_DIM)
NOPE_DIM = HEAD_DIM - ROPE_DIM
EPS = M.rms_norm_eps
Q_PROJ_TILE = 128
QPROJ_MM_N_TILE = 256 if QUANT_WEIGHT_NZ else 512
QPROJ_M_TILE = 64
QPROJ_N_BLOCKS = H * HEAD_DIM // QPROJ_MM_N_TILE
QPROJ_PIPE_M_TILE = 128
QPROJ_PIPE_K_TILE = 256
Q_ROPE_T_TILE = 8
Q_ROPE_H_TILE = 4
Q_DEQUANT_WORKERS = 48

T_DYN = pl.dynamic("CSA_QUERY_T_DYN")
PAD_ROWS_DYN = pl.dynamic("CSA_QUERY_PAD_ROWS_DYN")
QPROJ_ROWS_DYN = pl.dynamic("CSA_QUERY_PROJ_ROWS_DYN")
'''


class Names(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self.mapping.get(node.id, node.id), ctx=node.ctx), node)


def function(signature, doc, body):
    result = ast.parse("@pl.jit.inline(auto_scope=False)\n" + signature + "\n    pass\n").body[0]
    result.body = [ast.Expr(value=ast.Constant(value=doc))] + body
    return ast.fix_missing_locations(result)


def build(repo):
    source_dir = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa"
    qprojection = (source_dir / "q_projection.py").read_text()
    qkv = (source_dir / "qkv_proj_rope.py").read_text()
    functions = {node.name: node for text in (qprojection, qkv) for node in ast.parse(text).body
                 if isinstance(node, ast.FunctionDef)}
    result = []
    for layout in ("nd", "nz"):
        original = functions[f"_q_proj_q_matmul_{layout}"]
        scope = next(node for node in original.body if isinstance(node, ast.With))
        body = [copy.deepcopy(node) for node in scope.body if not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "pl.set_cache_policy") and not (
            isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "qproj_worker")]
        prefix = ast.parse("qproj_worker = pl.max(worker_id, 0)\n"
                           "qproj_t_matmul = ((tokens + QPROJ_PIPE_M_TILE - 1) // QPROJ_PIPE_M_TILE) * QPROJ_PIPE_M_TILE\n").body
        if layout == "nd":
            prefix += ast.parse("qproj_full_rows = qproj_t_matmul").body
        body = [Names({"QPROJ_WORKERS": "WORKERS", "tile_rows": "tokens"}).visit(node) for node in body]
        body = prefix + body + ast.parse("return q_proj_i32").body
        signature = f'''def _q_b_{layout}_stage(
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    qr_i8_matmul: pl.Tensor[[PAD_ROWS_DYN, Q_LORA], pl.INT8],
    q_proj_i32: pl.Tensor[[QPROJ_ROWS_DYN, H * HEAD_DIM], pl.INT32],
    tokens: pl.Scalar[pl.INDEX], worker_id: pl.Scalar[pl.INDEX], WORKERS: pl.constexpr,
):'''
        result.append(function(signature, f"{layout.upper()} Q_B：显式 AIC 按列块条带领取，原 K 流水不变；调用前必须等待 QR。", body))

    original = functions["q_proj_q_dequant"]
    scope = next(node for node in original.body if isinstance(node, ast.For))
    body = copy.deepcopy(scope.body)
    mapping = {"dq_worker": "worker_id", "tile_rows": "tokens", "out_tg": "tg"}
    body = [Names(mapping).visit(node) for node in body]
    work_loop = next(node for node in body if isinstance(node, ast.For))
    # 只有工作分配步长改成参数；原 head 分组阈值的 48 仍保留。
    work_loop.iter.args[-1] = ast.Name(id="WORKERS", ctx=ast.Load())
    work_loop.body = [node for node in work_loop.body if not (isinstance(node, ast.Assign)
                      and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "tg"
                      and isinstance(node.value, ast.BinOp) and isinstance(node.value.left, ast.Name)
                      and node.value.left.id == "tile_base")]
    body = ast.parse("tokens = pl.tensor.dim(q_flat, 0)").body + body + ast.parse("return q_flat").body
    result.append(function('''def q_dequant_rope_stage(
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q_flat: pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16],
    qr_scale_pad_store: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
    q_proj_i32: pl.Tensor[[QPROJ_ROWS_DYN, H * HEAD_DIM], pl.INT32],
    worker_id: pl.Scalar[pl.INDEX], WORKERS: pl.constexpr,
):''', "显式 AIV 反量化、RMS、NOPE/RoPE；q_flat 必须是外层建立的 GM 二维别名。\n\n"
       "完整八行与尾块沿用不同的既有 RMS 算术；每次工作只写八行中的一个 head 组。", body))

    original = functions["rope_prepare"]
    scope = next(node for node in original.body if isinstance(node, ast.For))
    body = [Names({"qrp_worker": "worker_id", "Q_ROPE_WORKERS": "WORKERS",
                   "rope_sin_view": "rope_sin", "rope_sin_signed_view": "rope_sin_signed",
                   "rope_swap_idx_view": "rope_swap_idx"}).visit(copy.deepcopy(node)) for node in scope.body]
    guarded = ast.If(test=ast.parse("worker_id < pl.min(WORKERS, token_tiles)", mode="eval").body, body=body, orelse=[])
    body = ast.parse("t_dim = pl.tensor.dim(rope_sin, 0)\ntoken_tiles = (t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE").body
    body += [guarded] + ast.parse("return rope_sin_signed, rope_swap_idx").body
    result.append(function('''def q_rope_prepare_stage(
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    worker_id: pl.Scalar[pl.INDEX], WORKERS: pl.constexpr,
):''', "显式 AIV 构造 interleaved sin 符号与行内交换索引；只处理有效 token tile。\n\n"
       "这段只依赖本轮 RoPE 输入，可与 QR/Q_B 并行；反量化消费前必须等待发布。", body))
    license_text = qprojection[:qprojection.index('"""')]
    return license_text + HEADER + "\n\n" + "\n\n".join(ast.unparse(node) for node in result) + (
        "\n\n# 布局在进程配置时确定，避免同一 JIT 函数解析互斥的 ND/NZ 数学分支。\n"
        "q_b_stage = _q_b_nz_stage if QUANT_WEIGHT_NZ else _q_b_nd_stage\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    source = build(repo)
    ast.parse(source)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(source)


if __name__ == "__main__":
    main()
