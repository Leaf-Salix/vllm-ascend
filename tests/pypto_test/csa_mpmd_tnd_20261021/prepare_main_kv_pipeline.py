# SPDX-License-Identifier: Apache-2.0
"""把KV三阶段纳入Main局部MIX；保留Indexer及后续独立task，不改数学。"""

import ast
import copy
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_priority_fusion import scope_name


def expr(source):
    return ast.parse(source, mode="eval").body


def extend_return(function, names):
    statement = function.body[-1]
    assert isinstance(statement, ast.Return) and isinstance(statement.value, ast.Tuple)
    statement.value.elts.extend(expr(name) for name in names)


def extend_call(statement, inputs, outputs):
    assert isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
    assert ast.unparse(statement.value.args[-1]) in ("FRONT_GROUPS", "MAIN_FRONT_GROUPS")
    statement.value.args[-1:-1] = [expr(name) for name in inputs]
    assert isinstance(statement.targets[0], ast.Tuple)
    statement.targets[0].elts.extend(ast.Name(id=name, ctx=ast.Store()) for name in outputs)


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    entry_path = target / "entry.py"
    entry_source = entry_path.read_text()
    root = next(
        n for n in ast.parse(entry_source).body if isinstance(n, ast.FunctionDef) and n.name == "decode_csa_tnd"
    )
    annotations = {a.arg: a for a in root.args.args}

    def argument(name, readonly=False):
        arg = copy.deepcopy(annotations[name])
        if readonly and isinstance(arg.annotation, ast.Subscript):
            if ast.unparse(arg.annotation.value) == "pl.InOut":
                arg.annotation = arg.annotation.slice
        return arg

    front_path = target / "front_pipelines.py"
    front = ast.parse(front_path.read_text())
    functions = {n.name: n for n in front.body if isinstance(n, ast.FunctionDef)}
    cube = functions["main_front_cube"]
    vector = functions["main_front_vector"]
    chain = functions["main_front_chain"]
    cube_inputs = ["wkv", "kv_fp32"]
    vector_inputs = ["kv_fp32", "gamma_ckv", "native_slots", "kv", "cache_rows"]
    chain_inputs = ["wkv", "gamma_ckv", "native_slots", "kv_fp32", "kv", "cache_rows"]
    for function, names in ((cube, cube_inputs), (vector, vector_inputs), (chain, chain_inputs)):
        assert function.args.args[-1].arg == "FRONT_GROUPS"
        function.args.args[-1:-1] = [argument(n, readonly=function is vector and n == "kv_fp32") for n in names]

    # AIC先完成KV，再进入已有Main压缩/Q_B流水。每阶段沿用原完整K归约。
    cube_start = next(
        i
        for i, n in enumerate(cube.body)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "compressor_project_stage"
    )
    cube.body[cube_start:cube_start] = ast.parse("""
kv_fp32 = kv_project_stage(normed, wkv, kv_fp32, worker, FRONT_GROUPS)
main_front_signal = publish_phase(main_front_signal, M_KV_PROJECT, core, 3 * FRONT_GROUPS, generation)
""").body
    extend_return(cube, ["kv_fp32"])

    # KV投影最多12个logical owner，所有有效生产者发布后才能读取整512列。
    # RMS每32行一owner，cache写每8行一owner；二者不能按同一worker直接串接。
    # 独立发布RMS完成阶段，写者等待所有有效RMS owner，保留原8行写回分工。
    vector_start = next(
        i
        for i, n in enumerate(vector.body)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "wait_row_owners"
    )
    vector.body[vector_start:vector_start] = ast.parse("""
main_front_signal = wait_row_owners(main_front_signal, M_KV_PROJECT, 3 * FRONT_GROUPS, generation,
    0, KV_HEAD_DIM // KV_N_TILE * KV_OM, 1, FRONT_GROUPS, 0, 1)
kv = kv_norm_rope_stage(kv_fp32, gamma_ckv, rope_cos, rope_signed, rope_swap, kv, worker, 2 * FRONT_GROUPS)
main_front_signal = publish_phase(main_front_signal, M_KV_NORM, core, 3 * FRONT_GROUPS, generation)
main_front_signal = wait_row_owners(main_front_signal, M_KV_NORM, 3 * FRONT_GROUPS, generation,
    0, (tokens + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE, 1, 2 * FRONT_GROUPS, FRONT_GROUPS, 1)
cache_rows = original_kv_writeback_stage(kv, native_slots, cache_rows, worker, 2 * FRONT_GROUPS)
""").body
    extend_return(vector, ["kv", "cache_rows"])
    cluster = next(n for n in chain.body if isinstance(n, ast.With))
    for node in cluster.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            label = ast.unparse(node.value.func)
            if label == "main_front_cube":
                extend_call(node, cube_inputs, ["kv_fp32"])
            elif label == "main_front_vector":
                extend_call(node, vector_inputs, ["kv", "cache_rows"])
    extend_return(chain, ["kv_fp32", "kv", "cache_rows"])
    front.body[1:1] = ast.parse("from .kv_stages import KV_RMS_T_TILE\n").body
    for node in front.body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "MAIN_FRONT_PHASES":
            assert ast.literal_eval(node.value) == 4
            node.value = ast.Constant(6)
    front.body += ast.parse("M_KV_PROJECT = 4\nM_KV_NORM = 5\n").body
    front_path.write_text(
        "# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(ast.fix_missing_locations(front)) + "\n"
    )

    gate = next(n for n in root.body if isinstance(n, ast.If) and ast.unparse(n.test) == "FUSED_DAG")
    gate.body = [n for n in gate.body if scope_name(n) not in ("kv_proj", "kv_norm_rope", "csa_cache_writeback")]
    for node in gate.body:
        if scope_name(node) == "main_front_chain":
            extend_call(node.body[0], chain_inputs, ["kv_fp32", "kv", "cache_rows"])
        if scope_name(node) == "index_score_topk":
            deps = next(k for k in node.items[0].context_expr.keywords if k.arg == "deps")
            assert isinstance(deps.value, ast.List)
            deps.value.elts = [n for n in deps.value.elts if ast.unparse(n) != "kv_cache_tid"]
    # 小容量else分支的原独立KV链完全保留。
    start = min([root.lineno] + [n.lineno for n in root.decorator_list]) - 1
    lines = entry_source.splitlines(keepends=True)
    lines[start : root.end_lineno] = [ast.unparse(ast.fix_missing_locations(root)) + "\n"]
    entry_path.write_text("".join(lines))
    binding_path = target / "binding.py"
    binding = binding_path.read_text()
    needle = "4 * 3 * self.policy.main_front_groups * 128"
    assert binding.count(needle) == 1
    binding_path.write_text(binding.replace(needle, "6 * 3 * self.policy.main_front_groups * 128"))
    for file in (front_path, entry_path, binding_path):
        ast.parse(file.read_text())
    print(target)


if __name__ == "__main__":
    main()
