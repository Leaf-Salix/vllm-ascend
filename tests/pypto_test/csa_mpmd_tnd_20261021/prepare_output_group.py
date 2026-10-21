# SPDX-License-Identifier: Apache-2.0
"""将冻结SPMD的逐组O_A/quant/O_B/HCpost流水移入一个局部MIX，保留前段多task。"""

import ast
import copy
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_priority_fusion import scope_name


class RenameSignal(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "signal":
            node.id = "output_signal"
        return node


def root_function(source, name):
    return next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)


def make_function(name, parameters, arguments, inouts, body, decorator):
    args = []
    for key in parameters:
        arg = copy.deepcopy(arguments[key])
        if key not in inouts and isinstance(arg.annotation, ast.Subscript):
            if ast.unparse(arg.annotation.value) == "pl.InOut":
                arg.annotation = arg.annotation.slice
        args.append(arg)
    return ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[], args=args, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
        ),
        body=body + ast.parse("return (" + ", ".join(inouts) + ")").body,
        decorator_list=[ast.parse(decorator, mode="eval").body],
        returns=None,
        type_comment=None,
    )


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    frozen = here / "sources/84751e34/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    cube = root_function((frozen / "cube_schedule.py").read_text(), "cube_schedule")
    vector = root_function((frozen / "vector_schedule.py").read_text(), "vector_schedule")
    cube_begin = next(
        i for i, n in enumerate(cube.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "OUTPUT_GROUP_PIPELINE"
    )
    vector_begin = next(
        i for i, n in enumerate(vector.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "OUTPUT_GROUP_PIPELINE"
    )
    # AIC保留逐组OA、量化owner等待、OB及完成发布；AIV保留逐组量化、OB等待和HCpost。
    # 外部Attention/HC依赖由TMR承担，不复制它们的旧phase，也不发布无人消费的OUT。
    cube_body = copy.deepcopy(cube.body[cube_begin : cube_begin + 2])
    vector_body = copy.deepcopy(vector.body[vector_begin : vector_begin + 3])
    assert "PHASE_OB" in ast.unparse(cube_body[-1])
    assert "o_dequant_hc_post_stage" in ast.unparse(vector_body[-1])

    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    entry_path = target / "entry.py"
    source = entry_path.read_text()
    root = root_function(source, "decode_csa_tnd")
    arguments = {arg.arg: arg for arg in root.args.args}
    arguments["output_signal"] = (
        ast.parse("def f(output_signal: pl.InOut[pl.Tensor[[OUTPUT_SIGNAL_WORDS], pl.INT32]]): pass")
        .body[0]
        .args.args[0]
    )
    arguments["OUTPUT_WORK_BALANCE"] = ast.parse("def f(OUTPUT_WORK_BALANCE: pl.constexpr): pass").body[0].args.args[0]

    cube_parameters = [
        "positions",
        "packed",
        "wo_a",
        "o_quant",
        "wo_b",
        "o_r",
        "o_partials",
        "output_signal",
        "OA_COLS",
        "OB_ROWS",
        "OUTPUT_WORK_BALANCE",
    ]
    vector_parameters = [
        "positions",
        "x",
        "post",
        "comb",
        "wo_b_scale",
        "o_r",
        "o_partials",
        "o_quant",
        "o_scales",
        "output",
        "output_signal",
        "OA_COLS",
        "OB_ROWS",
    ]
    cube_outputs = ["o_r", "o_partials", "output_signal"]
    vector_outputs = ["o_quant", "o_scales", "output", "output_signal"]
    common_outputs = ["o_r", "o_quant", "o_scales", "o_partials", "output", "output_signal"]
    common_parameters = [
        "positions",
        "packed",
        "wo_a",
        "wo_b",
        "wo_b_scale",
        "x",
        "post",
        "comb",
        *common_outputs,
        "OA_COLS",
        "OB_ROWS",
        "OUTPUT_WORK_BALANCE",
    ]
    cube_body = ast.parse(
        "worker = pl.tile.get_block_idx()\n"
        "tokens = pl.tensor.dim(positions, 0)\n"
        "(output_signal, generation) = begin_invocation(output_signal, worker)"
    ).body + [RenameSignal().visit(n) for n in cube_body]
    vector_body = ast.parse(
        "pl.func_attr({'split_aiv': True})\n"
        "worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()\n"
        "core = GROUPS + worker\n"
        "tokens = pl.tensor.dim(positions, 0)\n"
        "(output_signal, generation) = begin_invocation(output_signal, core)"
    ).body + [RenameSignal().visit(n) for n in vector_body]
    functions = [
        make_function("output_chain_cube", cube_parameters, arguments, cube_outputs, cube_body, "pl.jit.incore"),
        make_function(
            "output_chain_vector", vector_parameters, arguments, vector_outputs, vector_body, "pl.jit.incore"
        ),
    ]
    group_body = ast.parse(
        "with pl.cluster(name_hint='output_group_chain'):\n"
        "    (" + ", ".join(cube_outputs) + ") = output_chain_cube(" + ", ".join(cube_parameters) + ")\n"
        "    (" + ", ".join(vector_outputs) + ") = output_chain_vector(" + ", ".join(vector_parameters) + ")\n"
    ).body
    functions.append(
        make_function(
            "output_group_chain",
            common_parameters,
            arguments,
            common_outputs,
            group_body,
            "pl.jit.inline(auto_scope=False)",
        )
    )
    header = '''# SPDX-License-Identifier: Apache-2.0
"""冻结84751e34的输出组流水，独立于Attention与其它CSA task。

24 AIC/48 AIV每次各推进一次代际；AIC每组写完OA后发布，AIV只等真实owner，
量化通知包含INT8尾零。OB完成后HCpost仍按原八组左折叠规约。
局部MIX使用新分配且首次清零的20阶段signal，不能复用query阶段的代际布局。
跨task依赖由TMR表达；这里只复用输出尾段的核内通知，未合并完整CSA。
"""
import pypto.language as pl
from .output_projection_a_stages import (
    o_projection_a_groups_stage_nz, o_projection_a_stage_nz,
    o_projection_groups_quant_stage, o_projection_group_quant_stage,
)
from .output_projection_b_stages import o_b_groups_stage, o_b_stage, o_dequant_hc_post_stage
from .query_phase_sync import begin_invocation, publish_phase, wait_row_owners
from .qkv_stages import BF16_WEIGHT_LAYOUT
from .query_projection_stages import QUANT_WEIGHT_LAYOUT
from .runtime_abi import T, PACK_ROWS, O_MM_ROWS, O_SCALE_ROWS, GROUPS, VECTORS, CORES

OUTPUT_SIGNAL_WORDS = pl.dynamic("CSA_OUTPUT_SIGNAL_WORDS")
OUTPUT_SIGNAL_COUNT = 20 * CORES * 128
OUTPUT_GROUP_PIPELINE = True
PHASE_OA = 1
PHASE_O_QUANT = 2
PHASE_OB = 3
PHASE_OA_GROUP_BASE = 4
PHASE_O_QUANT_GROUP_BASE = 12
OUTPUT_BALANCE_HEAVY_CORES = 16
OUTPUT_BALANCE_VIRTUAL_WORKERS = 32
OUTPUT_BALANCE_LAST_RANK = 31
OUTPUT_BALANCE_COLUMNS_PER_GROUP = 8

'''
    (target / "output_group.py").write_text(
        header + "\n\n".join(ast.unparse(ast.fix_missing_locations(f)) for f in functions) + "\n"
    )

    for gate in root.body:
        if not isinstance(gate, ast.If) or ast.unparse(gate.test) != "FUSED_DAG":
            continue
        for statements in (gate.body, gate.orelse):
            first = next(i for i, node in enumerate(statements) if scope_name(node) == "oproj_a")
            assert [scope_name(node) for node in statements[first : first + 4]] == [
                "oproj_a",
                "oproj_quant",
                "oproj_b",
                "hc_post",
            ]
            replacement = ast.parse(
                "with pl.spmd(GROUPS, name_hint='output_group_chain', "
                "allow_early_resolve=True, sync_start=True, deps=[attn_tid]):\n"
                "    (" + ", ".join(common_outputs) + ") = output_group_chain(" + ", ".join(common_parameters) + ")\n"
            ).body
            statements[first : first + 4] = replacement
    signal_position = next(i for i, arg in enumerate(root.args.args) if arg.arg == "query_signal") + 1
    root.args.args.insert(signal_position, copy.deepcopy(arguments["output_signal"]))
    root.args.args.append(copy.deepcopy(arguments["OUTPUT_WORK_BALANCE"]))
    root.args.defaults.append(ast.Constant(False))
    assert isinstance(root.body[-1], ast.Return)
    root.body[-1].value.elts.append(ast.Name(id="output_signal", ctx=ast.Load()))
    start = min([root.lineno] + [n.lineno for n in root.decorator_list]) - 1
    lines = source.splitlines(keepends=True)
    lines[start : root.end_lineno] = [ast.unparse(ast.fix_missing_locations(root)) + "\n"]
    source = "".join(lines).replace(
        "import pypto.language as pl\n",
        "import pypto.language as pl\nfrom .output_group import OUTPUT_SIGNAL_WORDS, output_group_chain\n",
        1,
    )
    entry_path.write_text(source)
    binding_path = target / "binding.py"
    binding = binding_path.read_text().replace(
        "from .policy import", "from .output_group import OUTPUT_SIGNAL_COUNT\nfrom .policy import", 1
    )
    needle = '            "query_signal": ((QUERY_SIGNAL_COUNT if self.policy.fused_dag else 1,), torch.int32),\n'
    assert binding.count(needle) == 1
    binding = binding.replace(needle, needle + '            "output_signal": ((OUTPUT_SIGNAL_COUNT,), torch.int32),\n')
    binding = binding.replace('if name == "query_signal":', 'if name in ("query_signal", "output_signal"):')
    binding_path.write_text(binding)
    policy_path = target / "policy.py"
    policy = policy_path.read_text()
    needle = "    fused_dag: bool = True\n"
    assert policy.count(needle) == 1
    policy = policy.replace(needle, needle + "    output_work_balance: bool = False\n")
    needle = '            "FUSED_DAG": self.fused_dag,\n'
    assert policy.count(needle) == 1
    policy = policy.replace(needle, needle + '            "OUTPUT_WORK_BALANCE": self.output_work_balance,\n')
    # 逐字提取冻结参考的区间表达式，保持它的已验证适用范围。
    reference_policy = ast.parse((frozen / "policy.py").read_text())
    value = next(
        k.value
        for n in ast.walk(reference_policy)
        if isinstance(n, ast.Call)
        for k in n.keywords
        if k.arg == "output_work_balance"
    )
    needle = "    return CSAMPMDPolicy(\n"
    policy = policy.replace(needle, needle + "        output_work_balance=" + ast.unparse(value) + ",\n", 1)
    policy_path.write_text(policy)
    print(target)


if __name__ == "__main__":
    main()
