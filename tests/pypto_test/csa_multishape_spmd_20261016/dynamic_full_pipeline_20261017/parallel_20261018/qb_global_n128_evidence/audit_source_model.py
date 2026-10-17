"""纯标准库审计：逆变换 AST、实际循环的整数模型；不导入或编译 PyPTO。"""

import ast
import copy
import json
from pathlib import Path

from prepare_source import REPLACEMENTS


HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "source_reference"
CANDIDATE = HERE / "source_candidate"


def dump(node):
    return ast.dump(node, include_attributes=False)


def assigned(module, name):
    return next(node.value for node in module.body if isinstance(node, ast.Assign)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name)


def integer(expr, env):
    """只解释源码中少数整数表达式，不执行候选模块或 Tensor 运算。"""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (bool, int)):
        return expr.value
    if isinstance(expr, ast.Name):
        return env[expr.id]
    if isinstance(expr, ast.IfExp):
        return integer(expr.body if integer(expr.test, env) else expr.orelse, env)
    if isinstance(expr, ast.BinOp):
        left, right = integer(expr.left, env), integer(expr.right, env)
        operators = {ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
                     ast.Mult: lambda: left * right, ast.FloorDiv: lambda: left // right}
        return operators[type(expr.op)]()
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        assert isinstance(expr.func.value, ast.Name) and expr.func.value.id == "pl"
        assert expr.func.attr in ("min", "max") and not expr.keywords
        return {"min": min, "max": max}[expr.func.attr](*(integer(arg, env) for arg in expr.args))
    raise AssertionError(ast.dump(expr))


def functions(module):
    return {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}


def inverse_audit():
    names = sorted(path.name for path in REFERENCE.glob("*.py"))
    assert len(names) == 33 and names == sorted(path.name for path in CANDIDATE.glob("*.py"))
    changed = [name for name in names if (REFERENCE / name).read_bytes() != (CANDIDATE / name).read_bytes()]
    assert changed == sorted(REPLACEMENTS)
    for name in names:
        original_text = (REFERENCE / name).read_text()
        candidate_text = (CANDIDATE / name).read_text()
        # 先要求完整文本确实只包含事先列明的有限替换，再作独立结构逆变换。
        restored_text = candidate_text
        for old, new in reversed(REPLACEMENTS.get(name, [])):
            assert restored_text.count(new) == 1
            restored_text = restored_text.replace(new, old)
        assert restored_text == original_text
        original = ast.parse(original_text)
        restored = ast.parse(candidate_text)
        if name == "query_projection_stages.py":
            assert all(dump(node) == dump(functions(original)[key]) for key, node in functions(restored).items())
            value = assigned(restored, "QPROJ_MM_N_TILE")
            assert isinstance(value, ast.IfExp) and value.body.value == 128 and value.orelse.value == 512
            value.body = ast.Constant(value=256)
            # 两处总览文案已经由上面的精确文本逆变换约束，不忽略其它任意字符串。
            restored.body[0] = copy.deepcopy(original.body[0])
        elif name == "vector_schedule.py":
            imports = [node for node in restored.body if isinstance(node, ast.ImportFrom)
                       and node.module == "query_projection_stages"]
            assert len(imports) == 1
            aliases = imports[0].names
            assert sum(alias.name == "QPROJ_N_BLOCKS" for alias in aliases) == 1
            imports[0].names = [alias for alias in aliases if alias.name != "QPROJ_N_BLOCKS"]
            calls = [node for node in ast.walk(restored) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "wait_row_owners"
                     and isinstance(node.args[1], ast.Name) and node.args[1].id == "PHASE_QB"]
            assert len(calls) == 1 and isinstance(calls[0].args[5], ast.Name)
            assert calls[0].args[5].id == "QPROJ_N_BLOCKS"
            calls[0].args[5] = ast.Constant(value=128)
        assert dump(original) == dump(restored), name
    return {"file_count": len(names), "changed_files": changed, "unchanged_files_direct_bytes": 31,
            "inverse_text_exact": True, "inverse_ast_exact": True,
            "all_query_function_asts_unchanged": True}


def source_env(module, nz):
    env = {"H": 64, "HEAD_DIM": 512, "Q_LORA": 1024, "QUANT_WEIGHT_NZ": nz, "WORKERS": 24}
    for name in ("QPROJ_MM_N_TILE", "QPROJ_N_BLOCKS", "QPROJ_PIPE_M_TILE", "QPROJ_PIPE_K_TILE"):
        env[name] = integer(assigned(module, name), env)
    return env


def main():
    source_check = inverse_audit()
    modules = {role: ast.parse((path / "query_projection_stages.py").read_text())
               for role, path in (("reference", REFERENCE), ("candidate", CANDIDATE))}
    roles = {}
    examples = []
    for role, module in modules.items():
        env = source_env(module, True)
        function = functions(module)["_q_b_nz_stage"]
        rounds = next(node for node in function.body if isinstance(node, ast.For))
        row_loop = next(node for node in rounds.body if isinstance(node, ast.For))
        n_expr = next(node.value for node in rounds.body if isinstance(node, ast.Assign)
                      and node.targets[0].id == "qproj_n_idx")
        rows_expr = next(node.value for node in function.body if isinstance(node, ast.Assign)
                         and node.targets[0].id == "qproj_t_matmul")
        valid_expr = next(node.value for node in row_loop.body if isinstance(node, ast.Assign)
                          and node.targets[0].id == "m_rows")
        # 动态条带来自真实 loop bound/index；不另写另一套调度算法。
        owners = []
        for worker in range(24):
            worker_env = dict(env, qproj_worker=worker)
            round_count = integer(rounds.iter.args[1], worker_env)
            owners.append([integer(n_expr, dict(worker_env, qproj_round=r)) for r in range(round_count)])
        flat = [n for tasks in owners for n in tasks]
        assert sorted(flat) == list(range(env["QPROJ_N_BLOCKS"]))
        assert len(flat) == len(set(flat))
        assert all(n % 24 == worker for worker, tasks in enumerate(owners) for n in tasks)
        width = env["QPROJ_MM_N_TILE"]
        assert [(n * width, (n + 1) * width) for n in sorted(flat)] == [
            (start, start + width) for start in range(0, 32768, width)
        ]
        assert all(n * width * 4 % 64 == 0 for n in flat)
        for tokens in range(4097):
            token_env = dict(env, tokens=tokens)
            padded = integer(rows_expr, token_env)
            row_env = dict(token_env, qproj_t_matmul=padded)
            bounds = [integer(arg, row_env) for arg in row_loop.iter.args]
            starts = list(range(*bounds))
            spans = [(start, integer(valid_expr, dict(row_env, t0=start))) for start in starts]
            assert len(spans) == (tokens + 127) // 128
            assert sum(valid for _, valid in spans) == tokens
            assert all(0 < valid <= 128 and start + valid <= tokens for start, valid in spans)
            assert all(start == index * 128 for index, (start, _) in enumerate(spans))
            # 唯一列块 × 不相交连续有效行块，给出全部 (token,column) 的唯一写覆盖。
            assert not spans or spans[-1][0] + spans[-1][1] == tokens
            if tokens in (0, 1, 24, 48, 96, 127, 128, 129, 144, 168, 192, 255, 256, 257, 4096):
                task_counts = [len(tasks) for tasks in owners]
                examples.append({
                    "role": role, "T": tokens, "N": width, "m_valid_rows": [valid for _, valid in spans],
                    "n_blocks_per_worker": task_counts,
                    "mma_per_worker": [count * len(spans) * 4 for count in task_counts],
                    "fix_per_worker": [count * len(spans) for count in task_counts],
                    "logical_qr_load_bytes": env["QPROJ_N_BLOCKS"] * tokens * 1024,
                    "logical_weight_load_bytes": 32768 * 1024 * len(spans),
                    "logical_valid_output_bytes": tokens * 32768 * 4,
                })
        nd_env = source_env(module, False)
        assert nd_env["QPROJ_MM_N_TILE"] == 512 and nd_env["QPROJ_N_BLOCKS"] == 64
        # wait_row_owners(first=0,rows_per_tile=1) 原算法；NZ 与 ND 都覆盖全部 24 核。
        assert min(env["QPROJ_N_BLOCKS"], 24) == min(nd_env["QPROJ_N_BLOCKS"], 24) == 24
        roles[role] = {"nz_n_tile": width, "nz_n_blocks": env["QPROJ_N_BLOCKS"],
                       "nd_n_tile": 512, "nd_n_blocks": 64, "owners": owners,
                       "max_columns_per_worker": max(map(len, owners)) * width,
                       "min_columns_per_worker": min(map(len, owners)) * width}
    result = {
        "passed": True, "source_audit": source_check,
        "model": {"T_inclusive": [0, 4096], "T_case_count_each_role": 4097, "workers": 24,
                  "unique_column_tasks": True, "valid_row_partition_exact": True,
                  "no_valid_output_overlap_or_gap": True, "nz_nd_wait_all_24": True,
                  "T0_is_integer_model_only_not_entry_support_claim": True},
        "roles": roles, "examples": examples,
        "limitations": "整数模型和源码 AST 不证明设备精度、真实双槽、事件覆盖、资源规划或性能。逻辑载荷不等于实测 HBM。",
        "compiled": False, "device_execution": False,
    }
    (HERE / "SOURCE_MODEL_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "changed_files": source_check["changed_files"],
                      "T_case_count_each_role": 4097, "source_candidate": str(CANDIDATE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
