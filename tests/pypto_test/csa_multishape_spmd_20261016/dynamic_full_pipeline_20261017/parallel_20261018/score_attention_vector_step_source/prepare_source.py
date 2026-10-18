#!/usr/bin/env python3
"""从正式动态数学体提取 G6 AIV 一步；只生成可读源码并核对 AST，不加载 PyPTO。"""

import ast
import copy
import json
import re
import subprocess
import textwrap
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "vllm_ascend").is_dir())
SOURCE_PATH = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_score_stages.py"
REFERENCE_REVISION = "66882634"
RENAMES = {"query_group_size": "G6_QUERY_GROUP", "score_tile": "G6_SCORE_TILE"}


def extract_lines(source, nodes):
    lines = source.splitlines(keepends=True)
    body = textwrap.dedent("".join(lines[nodes[0].lineno - 1 : nodes[-1].end_lineno]))
    for old, new in RENAMES.items():
        body = re.sub(rf"\b{old}\b", new, body)
    return textwrap.indent(body, "    ")


class NormalizeNames(ast.NodeTransformer):
    def visit_Name(self, node):
        inverse = {value: key for key, value in RENAMES.items()}
        return ast.copy_location(ast.Name(id=inverse.get(node.id, node.id), ctx=node.ctx), node)


def normalized(nodes):
    tree = ast.Module(body=copy.deepcopy(nodes), type_ignores=[])
    return ast.dump(NormalizeNames().visit(tree), include_attributes=False)


def main():
    reference = subprocess.check_output(
        ["git", "show", f"{REFERENCE_REVISION}:{SOURCE_PATH}"], cwd=REPO, text=True
    )
    tree = ast.parse(reference)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "indexer_score_vector_lane_stage")
    step = next(node for node in ast.walk(function) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "single_step")
    balanced = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Name) and node.test.id == "balance_leaves")
    initialization = [node for node in balanced.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id in
                              {"single_roots", "single_segments"} for target in node.targets)]
    finalize = balanced.body[balanced.body.index(step) + 1 :]
    assert len(initialization) == 2 and len(finalize) == 1

    header = '''# SPDX-License-Identifier: Apache-2.0
"""动态 CSA 长档 G6 Score 的 AIV 可暂停一步：源码准备，尚未接入调度。

整体思路与阅读顺序
    leaf_state 创建本叶的 roots/segments；vector_step 完整消费一个1024候选
    块；leaf_publish 在全部step之后写回真实query的根。三个helper均不创建
    task，不决定wave，不缓存上次调用metadata，也不执行跨wave发布。

动态输入与固定数学tile
    G6=6是现有Indexer矩阵分组策略，单AIV负责三条query，score tile=1024。
    这两个常量不代表B、T、S、历史长度白名单。Tensor行数/页数/列数保留
    原pl.dynamic，实际query、请求、leaf、step及causal/页边界均为设备Scalar
    或device Tensor。只有原G6/balanced策略可以调用本模块；其它策略沿原路。

跨步状态与核间同步
    roots FP32[3,1024]保存已合并的512对(score,index)，segments FP32[6,1024]
    保存两轮各1024候选。偶数step留下的半段必须保持到下一步；调用者必须接住
    两个返回Tile，并在外层循环作为显式状态携带，不得每step重新初始化。
    两个状态均由原tile.full创建，要求完整静态valid_shape。真实query和候选
    有效范围仍由Scalar/mask控制；当前不把动态valid_shape Tile作为此接口合同。
    同组AIC与双AIV必须处于同一FFTS域。READY后MTE2读取双槽，随后ACK仅表示
    FIFO已读取，仍有scale、排序、归并未完成。暂停点是整个vector_step返回；
    根写回及业务发布在leaf_publish之后，不能用ACK代替根就绪。

数值与生命周期
    复用正式668的原始数学：FP16 scale按位解释、FP32乘法、causal有效行、
    sort32/mrgsort顺序、末步512/1024/2048分支和incoming优先平局顺序均不变。
    有工作但空query lane仍load并ACK，只跳过不存在query的数学/根写回。
    roots/segments的物理分配与异步消费者完成尚待联合lowering核对；显式Tile
    入参/返回需要当前正在独立overlay验证的JIT注解修复。源码AST相同不代表
    编译、真机精度或性能通过。此文件不是独立性能候选，也不替代完整CSA。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.config import FP32_NEG_INF
from .indexer_group_stages import TOKENS_DYN, indexer_query_visibility
from .indexer_score_stages import (
    BLOCK_SIZE, BUFFERED_LONG_SCORE_TILE, CACHE_PAGES_DYN, INDEXER_KEY_BYTES,
    PAGE_BYTES_DYN, TABLE_COLUMNS_DYN, TOPK_PAIR_WIDTH, TOPK_ROWS_PER_QUERY,
    TRANSFER_COLS_DYN, TRANSFER_ROWS_DYN,
)
from .indexer_topk_stages import PAIR_ROWS_DYN
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN

G6_QUERY_GROUP = 6
G6_SCORE_TILE = 1024


@pl.jit.inline(auto_scope=False)
def score_vector_leaf_state_g6():
    """只在进入有效叶时初始化一次；不在每个step或Attention切回时重新创建。"""
'''
    signature = '''

@pl.jit.inline(auto_scope=False)
def score_vector_step_g6(
    idx_native_kv_cache: pl.Tensor[[CACHE_PAGES_DYN, PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_COLUMNS_DYN], pl.INT32],
    position_ids: pl.Tensor[[TOKENS_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    buf_score_transfer: pl.Tensor[[TRANSFER_ROWS_DYN, TRANSFER_COLS_DYN], pl.FP32],
    single_roots: pl.Tile[[3, 1024], pl.FP32, pl.MemorySpace.Vec],
    single_segments: pl.Tile[[6, 1024], pl.FP32, pl.MemorySpace.Vec],
    single_begin: pl.Scalar[pl.INDEX],
    single_step: pl.Scalar[pl.INDEX],
    single_steps: pl.Scalar[pl.INDEX],
    single_cache_len: pl.Scalar[pl.INDEX],
    single_batch: pl.Scalar[pl.INDEX],
    single_query: pl.Scalar[pl.INDEX],
    buf_worker: pl.Scalar[pl.INDEX],
    buf_score_lane: pl.Scalar[pl.INDEX],
    buf_live: pl.Scalar[pl.INDEX],
    SCORE_READY_EVENT: pl.constexpr,
    SCORE_CONSUMED_EVENT: pl.constexpr,
):
    """完成一整个原Score step，返回仍跨步存活的roots与segments。"""
'''
    final_signature = '''

@pl.jit.inline(auto_scope=False)
def score_vector_leaf_publish_g6(
    single_roots: pl.Tile[[3, 1024], pl.FP32, pl.MemorySpace.Vec],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
    buf_score_lane: pl.Scalar[pl.INDEX],
    buf_live: pl.Scalar[pl.INDEX],
    single_query: pl.Scalar[pl.INDEX],
    single_leaf: pl.Scalar[pl.INDEX],
):
    """所有step数学结束后写根；外层必须在此后再发布本波根完成。"""
'''
    candidate = (
        header + extract_lines(reference, initialization)
        + "    return single_roots, single_segments\n"
        + signature + extract_lines(reference, step.body)
        + "    return single_roots, single_segments\n"
        + final_signature + extract_lines(reference, finalize)
        + "    return pair_arena\n"
    )
    parsed = ast.parse(candidate)
    functions = {node.name: node for node in parsed.body if isinstance(node, ast.FunctionDef)}
    checks = {}
    globals_available = set(functions)
    for node in parsed.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            globals_available.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Assign):
            globals_available.update(target.id for target in node.targets if isinstance(target, ast.Name))
    for name, function_node in functions.items():
        locals_available = {arg.arg for arg in function_node.args.args}
        locals_available.update(node.id for node in ast.walk(function_node)
                                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
        loads = {node.id for node in ast.walk(function_node)
                 if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        missing = loads - locals_available - globals_available
        assert not missing, (name, missing)
        checks[name + "_all_dependencies_explicit"] = True
    for name, original in (
        ("score_vector_leaf_state_g6", initialization),
        ("score_vector_step_g6", step.body),
        ("score_vector_leaf_publish_g6", finalize),
    ):
        body = functions[name].body[1:-1]  # only the docstring and explicit return are new
        checks[name + "_body_ast_equal"] = normalized(body) == normalized(original)
    assert all(checks.values()), checks
    compile(candidate, "score_vector_steps_g6.py", "exec")
    reference_dir = HERE / "source_reference"
    candidate_dir = HERE / "source_candidate"
    reference_dir.mkdir(exist_ok=True)
    candidate_dir.mkdir(exist_ok=True)
    (reference_dir / "indexer_score_stages.py").write_text(reference)
    (candidate_dir / "score_vector_steps_g6.py").write_text(candidate)
    checks.update({
        "passed": True,
        "reference_revision": REFERENCE_REVISION,
        "original_step_lines": [step.lineno, step.end_lineno],
        "only_math_policy_constants_bound": {"query_group_size": 6, "score_tile": 1024},
        "input_tensor_axes_kept_dynamic": True,
        "python_syntax_passed": True,
        "pypto_imported": False,
        "cpu_lowering_run": False,
        "npu_run": False,
        "performance_claim": False,
    })
    (HERE / "source_audit.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
