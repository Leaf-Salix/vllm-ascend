#!/usr/bin/env python3
"""准备 OB32 小 T OA pair 独立候选；仅复制、文本编辑和 AST 审计，不编译。"""
from __future__ import annotations

import argparse
import ast
import copy
import difflib
import json
from pathlib import Path
import shutil

FROZEN = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/oa_pair_manual_evidence/source_candidate")
REFERENCE = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/oa_pair_manual_evidence/source_reference")
DOCUMENTED = Path("/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_oa_pair_documented_20261018/source_documented")
FOUR = ("cube_schedule.py", "vector_schedule.py", "output_projection_a_stages.py", "phase_sync.py")
SCHEDULES = ("cube_schedule.py", "vector_schedule.py")
OLD = "OA_COLS == 128 and tokens > 0 and tokens <= 96"
NEW = "OA_COLS == 128 and OB_ROWS == 32 and tokens > 0 and tokens <= 32"
UNKNOWN = object()


def dump(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


class RemoveDocs(ast.NodeTransformer):
    def visit_scope(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        return node
    visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = visit_scope


def parsed(text):
    return RemoveDocs().visit(ast.parse(text, type_comments=True))


class RestorePredicate(ast.NodeTransformer):
    def __init__(self):
        self.count = 0
    def visit_If(self, node):
        self.generic_visit(node)
        if dump(node.test) == dump(ast.parse(NEW, mode="eval").body):
            node.test = ast.parse(OLD, mode="eval").body
            self.count += 1
        return node


# 复用旧 closed_branch_review 的有限求值方法：未知 device 表达式保留。
def value(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, UNKNOWN)
    if isinstance(node, ast.BoolOp):
        children = [value(n, env) for n in node.values]
        if isinstance(node.op, ast.And):
            if any(n is not UNKNOWN and not n for n in children):
                return False
            return UNKNOWN if any(n is UNKNOWN for n in children) else True
        if any(n is not UNKNOWN and n for n in children):
            return True
        return UNKNOWN if any(n is UNKNOWN for n in children) else False
    if isinstance(node, ast.Compare):
        children = [value(node.left, env)] + [value(n, env) for n in node.comparators]
        if any(n is UNKNOWN for n in children):
            return UNKNOWN
        operations = {ast.Eq: lambda a, b: a == b, ast.Lt: lambda a, b: a < b,
                      ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b,
                      ast.GtE: lambda a, b: a >= b}
        if any(type(op) not in operations for op in node.ops):
            return UNKNOWN
        return all(operations[type(op)](a, b) for a, op, b in zip(children, node.ops, children[1:]))
    return UNKNOWN


class Fold(ast.NodeTransformer):
    def __init__(self, env):
        self.env = env
    def visit_If(self, node):
        truth = value(node.test, self.env)
        if truth is UNKNOWN:
            return self.generic_visit(node)
        result = []
        for child in node.body if truth else node.orelse:
            folded = self.visit(child)
            result.extend(folded if isinstance(folded, list) else [folded])
        return result


class Reachability(ast.NodeVisitor):
    def __init__(self):
        self.guards, self.calls = [], []
    def visit_If(self, node):
        self.guards.append(ast.unparse(node.test))
        for child in node.body:
            self.visit(child)
        self.guards.pop()
        self.guards.append("NOT(" + ast.unparse(node.test) + ")")
        for child in node.orelse:
            self.visit(child)
        self.guards.pop()
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in {
            "o_projection_a_pair_group_stage_nz", "wait_oa_pair_group"
        }:
            self.calls.append({"name": node.func.id, "line": node.lineno, "guards": self.guards[:]})
        self.generic_visit(node)


def small_text(name, text):
    if name in SCHEDULES:
        assert text.count(OLD) == 1, (name, "旧谓词不唯一")
        text = text.replace(OLD, NEW, 1)
    # 下列文字替换仅更新本副本说明，后续完整 AST 审计会拒绝额外逻辑改动。
    if name == "output_projection_a_stages.py":
        text = text.replace(
            "输出组流水开启、OA_COLS=128 且\n    设备实际 0<T<=96 时",
            "输出组流水开启、OA_COLS=128、OB_ROWS=32 且\n    设备实际 0<T<=32 时",
        )
        text = text.replace(
            "仅输出组流水开启、N128 且 0<T<=96",
            "仅输出组流水开启、N128、OB_ROWS=32 且 0<T<=32",
        )
    elif name in SCHEDULES:
        text = text.replace("N128 且 0<T<=96", "N128、OB_ROWS=32 且 0<T<=32")
        indent = "        " if name == "cube_schedule.py" else "                "
        text = text.replace(
            indent + "if " + NEW + ":",
            indent + "# OB32 沿用既有 T<=32 区间，每组量化仅一个 32 行任务，不按 B/S 选点。\n"
            + indent + "# 在此域共享输入是待实测假说，不能由边界或搬运次数预记完整 L1 收益。\n"
            + indent + "if " + NEW + ":",
            1,
        )
    elif name == "phase_sync.py":
        text = text.replace("输出组流水开启的 N128 小 T", "输出组流水开启、OB_ROWS=32 的 N128 小 T")
        text = text.replace("组流水开启、0<T<=96、", "组流水开启、OB_ROWS=32、0<T<=32、")
        text = text.replace("不可用于旧logical%24或T>96分工", "不可用于旧logical%24、OB_ROWS!=32或T>32分工")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen96", type=Path, default=FROZEN)
    parser.add_argument("--reference668", type=Path, default=REFERENCE)
    parser.add_argument("--documented96", type=Path, default=DOCUMENTED)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = args.output.resolve()
    source_dir, reference_dir = output / "source_small_t", output / "source_reference668"
    inputs = (args.frozen96.resolve(), args.reference668.resolve(), args.documented96.resolve())
    assert all(destination != src and src not in destination.parents
               for src in inputs for destination in (source_dir, reference_dir))
    frozen = {p.name: p.read_text(encoding="utf-8") for p in sorted(args.frozen96.glob("*.py"))}
    reference = {p.name: p.read_text(encoding="utf-8") for p in sorted(args.reference668.glob("*.py"))}
    assert len(frozen) == len(reference) == 33 and frozen.keys() == reference.keys()
    documented = {name: (args.documented96 / name).read_text(encoding="utf-8") for name in FOUR}
    for name in FOUR:
        assert dump(parsed(frozen[name])) == dump(parsed(documented[name])), (name, "文档版执行 AST 改变")
    base = dict(frozen)
    base.update(documented)
    candidate = {name: small_text(name, text) for name, text in base.items()}
    differences, unchanged_ast = [], []
    for name in base:
        old_tree, new_tree = parsed(base[name]), parsed(candidate[name])
        if dump(old_tree) == dump(new_tree):
            unchanged_ast.append(name)
        else:
            assert name in SCHEDULES, (name, "非调度 AST 改变")
            restore = RestorePredicate()
            restored = restore.visit(copy.deepcopy(new_tree))
            assert restore.count == 1 and dump(restored) == dump(old_tree), (name, "不止谓词改变")
            differences.append({
                "file": name, "changed_node": "If.test", "count": restore.count,
                "old_predicate": OLD, "new_predicate": NEW,
                "complete_ast_equal_after_restoring_only_predicate": True,
            })
    assert sorted(d["file"] for d in differences) == sorted(SCHEDULES)
    reference_same = [name for name in reference if reference[name] == candidate[name]]
    assert len(reference_same) == 29
    # 检查原有数学/同步函数的全部 AST；仅允许新 pair / wait 两个定义。
    common_functions = {}
    for name, added in (
        ("output_projection_a_stages.py", {"o_projection_a_pair_group_stage_nz"}),
        ("phase_sync.py", {"wait_oa_pair_group"}),
    ):
        before = {n.name: n for n in parsed(reference[name]).body if isinstance(n, ast.FunctionDef)}
        after = {n.name: n for n in parsed(candidate[name]).body if isinstance(n, ast.FunctionDef)}
        assert set(after) - set(before) == added
        assert all(dump(n) == dump(after[key]) for key, n in before.items())
        common_functions[name] = list(before)

    # 两项主证明不提供 T/OA/GROUP 常量：仅 OB96/128 即消去新分支。
    cases = [
        {"OB_ROWS": 96},
        {"OB_ROWS": 128},
        {"OA_COLS": 256},
        {"OUTPUT_GROUP_PIPELINE": False},
        # 防御性动态边界的源码核对，不据此生成静态 kernel。
        {"OA_COLS": 128, "OB_ROWS": 32, "OUTPUT_GROUP_PIPELINE": True, "tokens": 0},
        {"OA_COLS": 128, "OB_ROWS": 32, "OUTPUT_GROUP_PIPELINE": True, "tokens": 33},
    ]
    closed = []
    for env in cases:
        for name in SCHEDULES:
            function_name = name.removesuffix(".py")
            funcs = []
            for texts in (reference, candidate):
                fn = next(n for n in parsed(texts[name]).body
                          if isinstance(n, ast.FunctionDef) and n.name == function_name)
                funcs.append(Fold(env).visit(fn))
            assert dump(funcs[0]) == dump(funcs[1]), (env, name, "关闭后未恢复原调度")
            reachable = Reachability()
            reachable.visit(funcs[1])
            assert not reachable.calls
            closed.append({"known_values": env, "file": name,
                           "entire_schedule_function_ast_equal_to_reference668": True,
                           "pair_and_special_wait_unreachable": True})
    calls = []
    for name in SCHEDULES:
        visitor = Reachability()
        visitor.visit(ast.parse(candidate[name], type_comments=True))
        assert len(visitor.calls) == 1
        for call in visitor.calls:
            assert "OUTPUT_GROUP_PIPELINE" in call["guards"]
            assert any(dump(ast.parse(guard, mode="eval").body) == dump(ast.parse(NEW, mode="eval").body)
                       for guard in call["guards"] if not guard.startswith("NOT("))
            calls.append({"file": name, **call})
    predicate = ast.parse(NEW, mode="eval").body
    enabled = [t for t in range(0, 98) if value(predicate, {"OA_COLS": 128, "OB_ROWS": 32, "tokens": t})]
    assert enabled == list(range(1, 33))

    for destination in (source_dir, reference_dir):
        destination.mkdir(parents=True, exist_ok=True)
    for name in frozen:
        shutil.copy2(args.frozen96 / name, source_dir / name)
        (source_dir / name).write_text(candidate[name], encoding="utf-8")
        shutil.copy2(args.reference668 / name, reference_dir / name)
        assert (source_dir / name).read_text(encoding="utf-8") == candidate[name]
        assert (reference_dir / name).read_text(encoding="utf-8") == reference[name]
    assert all((args.frozen96 / name).read_text(encoding="utf-8") == text for name, text in frozen.items())
    assert all((args.reference668 / name).read_text(encoding="utf-8") == text for name, text in reference.items())
    assert all((args.documented96 / name).read_text(encoding="utf-8") == text for name, text in documented.items())
    for label, before in (("small_t_vs_documented96.patch", base), ("small_t_vs_reference668.patch", reference)):
        diff = []
        for name in frozen:
            diff.extend(difflib.unified_diff(before[name].splitlines(keepends=True),
                        candidate[name].splitlines(keepends=True), fromfile="a/" + name, tofile="b/" + name))
        (output / label).write_text("".join(diff), encoding="utf-8")
    audit = {
        "status": "PASS",
        "scope": "只进行源码文本与 AST 审计，未编译或上卡；不声称生成 CPP/分配/事件已核对",
        "inputs": {"frozen96": str(args.frozen96), "documented96": str(args.documented96), "reference668": str(args.reference668)},
        "outputs": {"candidate": str(source_dir), "reference668": str(reference_dir)},
        "files_each": 33,
        "complete_ast_method": "剥除全部 module/class/function/async docstring，保留其余所有 AST 节点、类型注释、导入及顺序；忽略行列位置",
        "candidate_vs_documented96_ast_changes": differences,
        "candidate_vs_documented96_unchanged_ast_files": unchanged_ast,
        "candidate_vs_reference668_byte_identical_files": reference_same,
        "reference668_copy_all_files_byte_identical": True,
        "frozen_inputs_text_unchanged": True,
        "common_math_and_sync_functions_ast_unchanged": common_functions,
        "closed_branch_comparisons": closed,
        "closed_branch_limit": "只比较完整调度函数及原有 helper；Python 模块仍有新增 import 和未调用的 pair/wait 定义，不把此结果称为整个模块或实际后端生成物相同",
        "calls_with_ancestor_guards": calls,
        "effective_predicate": "OUTPUT_GROUP_PIPELINE && " + NEW,
        "OB32_dynamic_T_checked": [0, 97],
        "OB32_enabled_T": [1, 32],
        "source_policy_unchanged": True,
        "policy_basis": "现行 policy.py: OB_ROWS=32 iff tokens<=32；量化每组只有一个32行任务。该边界是待验证假说，不是收益保证",
    }
    (output / "source_ast_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths = []
    for root in (source_dir, reference_dir):
        for file in sorted(root.glob("*.py")):
            paths.append({"path": str(file), "size_bytes": file.stat().st_size})
    (output / "SOURCE_FILES.json").write_text(json.dumps(paths, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "files_each": 33, "changed_predicates": 2,
                      "other_ast_files_unchanged": len(unchanged_ast), "closed_function_comparisons": len(closed),
                      "reference_copy_identical": True, "candidate": str(source_dir),
                      "reference668": str(reference_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
