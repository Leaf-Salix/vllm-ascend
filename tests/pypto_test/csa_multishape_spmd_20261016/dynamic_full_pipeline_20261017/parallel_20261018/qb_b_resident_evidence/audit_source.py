"""检查constexpr=False回原AST、四文件改动与Host区间；不加载PyPTO。"""

import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


class FoldDisabled(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == 'QB_B_RESIDENT':
            return [self.visit(n) for n in node.orelse]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        defaults = dict(zip([a.arg for a in node.args.args[-len(node.args.defaults):]], node.args.defaults)) if node.args.defaults else {}
        node.args.args = [a for a in node.args.args if a.arg != 'QB_B_RESIDENT']
        node.args.defaults = [defaults[a.arg] for a in node.args.args if a.arg in defaults]
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.args = [x for x in node.args if not (isinstance(x, ast.Name) and x.id == 'QB_B_RESIDENT')]
        return self.generic_visit(node)


def load_policy(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    reference, candidate = HERE/'source_reference', HERE/'source_candidate'
    changes = sorted(p.name for p in reference.glob('*.py') if p.read_text() != (candidate/p.name).read_text())
    ast_checks = {}
    for name in ('query_projection_stages.py', 'entry.py', 'cube_schedule.py'):
        left, right = [FoldDisabled().visit(ast.parse((p/name).read_text())) for p in (reference, candidate)]
        ast_checks[name] = ast.dump(left) == ast.dump(right)
        assert ast_checks[name], name
    ref = load_policy(reference/'policy.py', 'qb_resident_reference_policy')
    cand = load_policy(candidate/'policy.py', 'qb_resident_candidate_policy')
    checked = enabled = 0
    examples = []
    for t in range(1, 513):
        for q in (1, 5, 6):
            for h in (8192, 131072):
                args = dict(tokens=t, requests=(t+q-1)//q, max_query_tokens=q, max_seq_len=h)
                a, b = ref.select_policy(**args).constexprs(), cand.select_policy(**args).constexprs()
                value = b.pop('QB_B_RESIDENT')
                assert a == b and value == (128 < t <= 256)
                checked += 1
                enabled += value
                if t in (128, 129, 144, 168, 192, 256, 257) and q == 6 and h == 131072:
                    examples.append(dict(tokens=t, enabled=value, QA_M16_PARALLEL=b['QA_M16_PARALLEL']))
    controls = json.loads((HERE/'controls.json').read_text())
    assert controls['QB_B_RESIDENT'] is True and controls['QA_M16_PARALLEL'] is False
    # 编译前配置与目标长档描述符完全一致；CPU本身不提供任何样例维度。
    assert controls == cand.select_policy(tokens=144, requests=24, max_query_tokens=6, max_seq_len=131072).constexprs()
    summary = dict(passed=True, changed_files=changes, disabled_ast_equal=ast_checks,
                   policy_cases=checked, policy_enabled_cases=enabled, examples=examples,
                   controls_equal_long_t144_policy=True,
                   no_false_kernel_compilation=True,
                   scope='False经常量折叠后的源AST等价；不是新False二进制等价证明')
    (HERE/'disabled_and_policy_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
