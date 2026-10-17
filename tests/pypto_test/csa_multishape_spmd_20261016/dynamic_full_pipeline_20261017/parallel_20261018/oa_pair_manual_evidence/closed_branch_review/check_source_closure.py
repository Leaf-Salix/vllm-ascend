"""对冻结源码做有限constexpr折叠与可达性核对；不导入PyPTO，不编译。"""

import ast
import copy
import json
from pathlib import Path


here = Path(__file__).resolve().parent
before, after = here.parent / "source_reference", here.parent / "source_candidate"
UNKNOWN = object()


def value(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, UNKNOWN)
    if isinstance(node, ast.BoolOp):
        children = [value(n, env) for n in node.values]
        if isinstance(node.op, ast.And):
            if any(n is not UNKNOWN and not n for n in children): return False
            return UNKNOWN if any(n is UNKNOWN for n in children) else True
        if any(n is not UNKNOWN and n for n in children): return True
        return UNKNOWN if any(n is UNKNOWN for n in children) else False
    if isinstance(node, ast.Compare):
        children = [value(node.left, env)] + [value(n, env) for n in node.comparators]
        if any(n is UNKNOWN for n in children): return UNKNOWN
        functions = {ast.Eq: lambda a,b:a==b, ast.NotEq:lambda a,b:a!=b,
                     ast.Lt:lambda a,b:a<b, ast.LtE:lambda a,b:a<=b,
                     ast.Gt:lambda a,b:a>b, ast.GtE:lambda a,b:a>=b}
        if any(type(op) not in functions for op in node.ops): return UNKNOWN
        return all(functions[type(op)](a,b) for a,op,b in zip(children,node.ops,children[1:]))
    return UNKNOWN


class Fold(ast.NodeTransformer):
    def __init__(self, env): self.env = env

    def visit_If(self, node):
        truth = value(node.test, self.env)
        if truth is UNKNOWN:
            return self.generic_visit(node)
        chosen = node.body if truth else node.orelse
        result = []
        for child in chosen:
            folded = self.visit(child)
            result.extend(folded if isinstance(folded,list) else [folded])
        return result


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)


comparisons=[]
cases=[{"OA_COLS":256,"OUTPUT_GROUP_PIPELINE":True},
       {"OA_COLS":128,"OUTPUT_GROUP_PIPELINE":False},
       {"OA_COLS":256,"OUTPUT_GROUP_PIPELINE":False},
       # 最后一项只验证防御性谓词的源级分支，不把T97按N128声明成正式策略。
       {"OA_COLS":128,"OUTPUT_GROUP_PIPELINE":True,"tokens":97}]
for env in cases:
    for filename,name in [("cube_schedule.py","cube_schedule"),("vector_schedule.py","vector_schedule")]:
        a=Fold(env).visit(copy.deepcopy(function(before/filename,name)))
        b=Fold(env).visit(copy.deepcopy(function(after/filename,name)))
        same=ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
        assert same, (env,filename)
        comparisons.append({"controls":env,"file":filename,"folded_function_ast_identical":same})
unchanged=[]
for filename in ["policy.py","entry.py","binding.py","runtime_abi.py","signals.py","wait_codegen.py"]:
    assert (before/filename).read_bytes()==(after/filename).read_bytes()
    unchanged.append(filename)

class Reachability(ast.NodeVisitor):
    def __init__(self): self.guards=[];self.calls=[]
    def visit_If(self,node):
        self.guards.append(ast.unparse(node.test))
        for n in node.body:self.visit(n)
        self.guards.pop()
        self.guards.append("NOT("+ast.unparse(node.test)+")")
        for n in node.orelse:self.visit(n)
        self.guards.pop()
    def visit_Call(self,node):
        if ast.unparse(node.func) in ("o_projection_a_pair_group_stage_nz","wait_oa_pair_group"):
            self.calls.append({"function":ast.unparse(node.func),"guards":self.guards[:],"line":node.lineno})
        self.generic_visit(node)
calls=[]
for filename in ["cube_schedule.py","vector_schedule.py"]:
    visitor=Reachability();visitor.visit(ast.parse((after/filename).read_text()));calls.extend(visitor.calls)
assert len(calls)==2
for call in calls:
    assert "OUTPUT_GROUP_PIPELINE" in call["guards"]
    assert "OA_COLS == 128 and tokens > 0 and (tokens <= 96)" in call["guards"] or "OA_COLS == 128 and tokens > 0 and tokens <= 96" in call["guards"]
predicate=ast.parse("OA_COLS == 128 and tokens > 0 and tokens <= 96",mode="eval").body
accepted=[t for t in range(-1,385) if value(predicate,{"OA_COLS":128,"tokens":t})]
assert accepted==list(range(1,97))
assert all((t+127)//128==1 for t in accepted)
assert not any(value(predicate,{"OA_COLS":256,"tokens":t}) for t in range(-1,385))
(here/"source_closure.json").write_text(json.dumps({
    "scope":"源级AST有限constexpr折叠，不替代真实后端；没有冻结真实运行T",
    "comparisons":comparisons,"byte_identical_host_abi_files":unchanged,
    "candidate_calls_and_ancestor_guards":calls,"accepted_T_range":[1,96],
    "single_M128_block":True,"new_runtime_roots":0,"new_phase_definitions":0,
    "coverage_evidence":"../owner_evidence/rotated_owner_model.json：T1..96全部64个N128列块唯一写、五owner等待及phase DAG",
},ensure_ascii=False,indent=2)+"\n")
print("closed branches: 8 folded function comparisons PASS; host policy/ABI byte-identical")
