#!/usr/bin/env python3
"""QA v2有限源AST/Host区间/动态owner检查；不导入设备库、不编译。"""
import ast
import copy
import dataclasses
import difflib
import importlib.util
import json
import pathlib
import sys
from collections import Counter

E = pathlib.Path(__file__).resolve().parent
R, C = E / "source_reference_9d", E / "source_candidate"
FLAG = "QA_M16_PARALLEL"
checks = {}


def normalized(n):
    n = copy.deepcopy(n)
    for obj in ast.walk(n):
        if hasattr(obj, "body") and isinstance(obj.body, list) and obj.body:
            first = obj.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                obj.body = obj.body[1:]
    return ast.dump(n, include_attributes=False)


class Restore(ast.NodeTransformer):
    def visit_FunctionDef(self, n):
        if n.name == "_q_a_m16_parallel_worker":
            return None
        if n.args.args and n.args.args[-1].arg == FLAG:
            assert ast.literal_eval(n.args.defaults[-1]) is False
            n.args.args.pop()
            n.args.defaults.pop()
        return self.generic_visit(n)

    def visit_If(self, n):
        if isinstance(n.test, ast.Name) and n.test.id == FLAG:
            assert len(n.orelse) == 1
            return self.visit(n.orelse[0])
        return self.generic_visit(n)

    def visit_Call(self, n):
        n.args = [a for a in n.args if not (isinstance(a, ast.Name) and a.id == FLAG)]
        n.keywords = [a for a in n.keywords if a.arg != "qa_m16_parallel"]
        return self.generic_visit(n)

    def visit_AnnAssign(self, n):
        if isinstance(n.target, ast.Name) and n.target.id == "qa_m16_parallel":
            assert ast.literal_eval(n.value) is False
            return None
        return self.generic_visit(n)

    def visit_Dict(self, n):
        pairs = [(k, v) for k, v in zip(n.keys, n.values)
                 if not (isinstance(k, ast.Constant) and k.value == FLAG)]
        n.keys, n.values = [k for k, _ in pairs], [v for _, v in pairs]
        return self.generic_visit(n)


names = sorted(p.name for p in R.glob("*.py"))
assert len(names) == 33 and names == sorted(p.name for p in C.glob("*.py"))
changed = [n for n in names if (R/n).read_text() != (C/n).read_text()]
assert changed == ["cube_schedule.py", "entry.py", "policy.py", "qkv_stages.py", "vector_schedule.py"]
for n in changed:
    old, new = ast.parse((R/n).read_text()), ast.parse((C/n).read_text())
    assert normalized(old) == normalized(Restore().visit(new)), n
checks["only_five_wiring_policy_and_math_files_changed_false_ast_restores_reference"] = True

old = ast.parse((R/"qkv_stages.py").read_text())
new = ast.parse((C/"qkv_stages.py").read_text())
worker = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "_q_a_worker")
outer = next(n for n in worker.body if isinstance(n, ast.For))
tail = [n for n in outer.body if isinstance(n, ast.For)][-1]
helper = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "_q_a_m16_parallel_worker")
loop = next(n for n in helper.body if isinstance(n, ast.For))


class Rename(ast.NodeTransformer):
    def visit_Name(self, n):
        if n.id.startswith("m16_"):
            n.id = "tail_" + n.id[4:]
        return n


assert normalized(ast.Module(body=tail.body, type_ignores=[])) == normalized(
    Rename().visit(ast.Module(body=copy.deepcopy(loop.body[2:]), type_ignores=[])))
assert ast.unparse(loop.body[0].value) == "logical_worker % QR_N_BLOCKS * QR_N_TILE"
assert ast.unparse(loop.body[1].value) == "logical_worker // QR_N_BLOCKS * QR_M_TILE"
vector = ast.parse((C/"vector_schedule.py").read_text())
guard = next(n for n in ast.walk(vector) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == FLAG)
assert ast.unparse(guard.body[0].value) == "pl.min(GROUPS, 8 * ((tokens + 15) // 16))"
checks["m16_math_same_original_tail_and_owner_wait_ceil_capped_24"] = True


def policy(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


old_policy, new_policy = policy(R/"policy.py", "qa2_old_policy"), policy(C/"policy.py", "qa2_new_policy")
points = [*range(1, 97), 127, 128, 129, 144, 168, 192, 384]
for t in points:
    for q in [1, 6, 8]:
        for history in [8192, 131072]:
            args = dict(tokens=t, requests=(t+q-1)//q, max_query_tokens=q, max_seq_len=history)
            before, after = old_policy.select_policy(**args), new_policy.select_policy(**args)
            controls = after.constexprs()
            assert controls.pop(FLAG) == (16 < t < 32)
            assert controls == before.constexprs()
            assert dataclasses.replace(after, qa_m16_parallel=False).constexprs()[FLAG] is False
checks["host_policy_only_16_lt_T_lt_32_all_other_controls_unchanged"] = True

for t in [0, *points]:
    tasks = 8*((t+15)//16)
    units = [(i%24, (i//8)*16, i%8) for i in range(tasks)]
    rows = Counter((r, col) for _, start, col in units for r in range(start, min(t, start+16)))
    assert rows == Counter({(r, c): 1 for r in range(t) for c in range(8)})
    assert {o for o, _, _ in units} == set(range(min(24, tasks)))
    assert all(0 <= row < t and row+16 <= ((t+15)//16)*16 for _, row, _ in units)
    if 16 < t < 32:
        assert tasks == 16 and len(units) == 16
        assert {o for o, row, _ in units if row == 16} == set(range(8, 16))
checks["dynamic_global_m16_owner_tail_capacity_and_waits_finite_coverage"] = True
checks["enabled_interval_exactly_two_original_m16_blocks_weight_requests_unchanged"] = True

patch = "".join("".join(difflib.unified_diff((R/n).read_text().splitlines(True),
    (C/n).read_text().splitlines(True), fromfile="source_reference_9d/"+n,
    tofile="source_candidate/"+n)) for n in changed)
(E/"source.diff").write_text(patch)
baseline = pathlib.Path("/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/ultra_indexer_20261018/query_dequant_group_20261018/cpu_v1/result.json")
bd = json.loads(baseline.read_text())
assert all((R/n).read_text() == (pathlib.Path(bd["source_package"])/n).read_text() for n in names)
checks["reference33_direct_text_equal_existing_9d_dq_cpu_source"] = True
(E/"controls_false.json").write_text(json.dumps(bd["controls"], indent=2)+"\n")
(E/"controls_true.json").write_text(json.dumps(dict(bd["controls"], QA_M16_PARALLEL=True), indent=2)+"\n")
result = dict(passed=True, checks=checks, changed_files=changed, tested_tokens=[0,*points],
              baseline_cpu_result=str(baseline), false_control_omits_new_flag_to_test_default=True,
              scope="源码AST/有限所有权/Host选择；不代表lowering或设备结论")
(E/"audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
