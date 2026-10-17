#!/usr/bin/env python3
"""小T QA候选的有限AST与所有权审计；不导入设备库、不编译。"""
import ast
import copy
import difflib
import json
from collections import Counter
from pathlib import Path

E = Path(__file__).resolve().parent
R, C = E / "source_reference", E / "source_candidate"
checks = {}

def normalized(node):
    node = copy.deepcopy(node)
    for obj in ast.walk(node):
        if hasattr(obj, "body") and isinstance(obj.body, list) and obj.body:
            first = obj.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                obj.body = obj.body[1:]
    return ast.dump(node, include_attributes=False)

files = sorted(p.name for p in R.glob("*.py"))
assert files == sorted(p.name for p in C.glob("*.py")) and len(files) == 33
changed = [n for n in files if (R / n).read_text() != (C / n).read_text()]
assert changed == ["qkv_stages.py", "vector_schedule.py"], changed
checks["only_two_source_files_changed"] = changed

r = ast.parse((R / "qkv_stages.py").read_text())
c = ast.parse((C / "qkv_stages.py").read_text())
rf = {n.name: n for n in r.body if isinstance(n, ast.FunctionDef)}
cf = {n.name: n for n in c.body if isinstance(n, ast.FunctionDef)}
old_worker = rf["_q_a_worker"]
old_outer = next(n for n in old_worker.body if isinstance(n, ast.For))
old_tail = [n for n in old_outer.body if isinstance(n, ast.For)][-1]
new_helper = cf["_q_a_small_m16_worker"]
new_outer = next(n for n in new_helper.body if isinstance(n, ast.For))

class Rename(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id.startswith("tail_"):
            node.id = "small_" + node.id[5:]
        return node

old_math = Rename().visit(ast.Module(body=copy.deepcopy(old_tail.body), type_ignores=[]))
new_math = ast.Module(body=copy.deepcopy(new_outer.body[2:]), type_ignores=[])
assert normalized(old_math) == normalized(new_math)
checks["small_math_identical_to_original_m16_tail_except_local_names"] = True
new_strategy = next(n for n in cf["q_a_stage"].body if isinstance(n, ast.If))
assert ast.unparse(new_strategy.test) == "tokens <= QR_FIXED_M_GROUPS * QR_M_TILE"
assert len(new_strategy.orelse) == 1 and isinstance(new_strategy.orelse[0], ast.If)
cf["q_a_stage"].body[cf["q_a_stage"].body.index(new_strategy)] = new_strategy.orelse[0]
c.body.remove(new_helper)
assert normalized(r) == normalized(c)
checks["all_existing_qkv_math_and_t_above_48_branch_ast_unchanged"] = True

r = ast.parse((R / "vector_schedule.py").read_text())
c = ast.parse((C / "vector_schedule.py").read_text())
def find_qa_if(tree):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and n.body and isinstance(n.body[0], ast.Assign)
                and isinstance(n.body[0].targets[0], ast.Name)
                and n.body[0].targets[0].id == "qa_tasks")
new_wait = find_qa_if(c)
assert ast.unparse(new_wait.test) == "tokens <= 48"
assert ast.unparse(new_wait.body[0].value) == "8 * ((tokens + 15) // 16)"
assert normalized(new_wait.body[1]) == normalized(find_qa_if(r).body[1])
assert len(new_wait.orelse) == 1 and isinstance(new_wait.orelse[0], ast.If)
class RestoreWait(ast.NodeTransformer):
    def visit_If(self, node):
        if node is new_wait:
            return node.orelse[0]
        return self.generic_visit(node)
c = RestoreWait().visit(c)
assert normalized(r) == normalized(c)
checks["vector_ast_only_adds_small_t_qa_owner_branch"] = True

def jobs(t, candidate):
    result = []
    if candidate and t <= 48:
        for logical in range(8 * ((t + 15) // 16)):
            result.append((logical % 24, (logical // 8) * 16, 16, logical % 8))
    else:
        m = 32 if t < 128 else 64
        groups = min(3, max(1, t // m))
        full = (t // m) * m
        pad = ((t + 15) // 16) * 16
        for logical in range(8 * groups):
            owner, col, group = logical % 24, logical % 8, logical // 8
            for row in range(group * m, full, groups * m):
                result.append((owner, row, m, col))
            for row in range(full + group * 16, pad, groups * 16):
                result.append((owner, row, 16, col))
    return result

def waited(t, candidate):
    if candidate and t <= 48:
        count = 8 * ((t + 15) // 16)
    else:
        count = 8 * min(3, max(1, t // (32 if t < 128 else 64)))
    return set(range(min(count, 24)))

cases = []
for t in [*range(97), 127, 128, 129, 192, 384]:
    before, after = jobs(t, False), jobs(t, True)
    for candidate, units in [(False, before), (True, after)]:
        valid = Counter((row, col) for _, start, height, col in units
                        for row in range(start, min(start + height, t)))
        assert valid == Counter({(row, col): 1 for row in range(t) for col in range(8)})
        assert {u[0] for u in units} <= waited(t, candidate)
        pad = ((t + 15) // 16) * 16
        for _, start, height, _ in units:
            assert 0 <= start < t and start + height <= pad
            assert 1 <= min(height, t - start) <= height
    if t > 48:
        assert before == after and waited(t, False) == waited(t, True)
    if t <= 48:
        assert max(Counter(u[0] for u in after).values(), default=0) <= 1
    if t in [0,1,15,16,17,24,31,32,33,47,48,49,64,96,127,128,129]:
        cases.append(dict(t=t,reference_jobs=before,candidate_jobs=after,
                          reference_waited=sorted(waited(t,False)),candidate_waited=sorted(waited(t,True))))
checks["t_0_to_96_and_extra_boundaries_owner_coverage_and_waits"] = True
checks["t49_and_all_t_above48_tested_owner_formula_unchanged"] = True
checks["t24_tail_row16_owned_by_cores8_to15"] = sorted(u[0] for u in jobs(24,True) if u[1] == 16)
assert checks["t24_tail_row16_owned_by_cores8_to15"] == list(range(8,16))

patch = ""
for name in changed:
    patch += "".join(difflib.unified_diff((R/name).read_text().splitlines(True),
                                        (C/name).read_text().splitlines(True),
                                        fromfile="source_reference/"+name,tofile="source_candidate/"+name))
(E/"source.diff").write_text(patch)
result = dict(passed=True,checks=checks,scope="AST及有限整数所有权检查；不代表lowering、设备精度或性能",
              source_files=[dict(name=n,reference_bytes=(R/n).stat().st_size,candidate_bytes=(C/n).stat().st_size) for n in files],
              boundary_cases=cases)
(E/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(dict(passed=True,checks=checks),ensure_ascii=False,indent=2))

