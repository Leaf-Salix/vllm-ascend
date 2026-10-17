"""只用标准库核对私有 PV constexpr 接线与区间策略，不加载 PyPTO/NPU。

数学核比较只消去新传参、旧模块常量与中文 docstring；不能消去任意运算差异。
策略用真实输入描述符计算的既有 long_history/G6 域，另外核对默认值与非法值。
输出 source_audit.json，不修改源码，不能代替完整 CPU 或设备验证。
"""

import ast
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path


HERE = Path(__file__).resolve().parent


class RemoveNewWiring(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.args.args and node.args.args[-1].arg == "PV_N_TILE":
            assert ast.literal_eval(node.args.defaults[-1]) == 128
            node.args.args.pop()
            node.args.defaults.pop()
        return self.generic_visit(node)

    def visit_Call(self, node):
        if node.args and isinstance(node.args[-1], ast.Name) and node.args[-1].id == "PV_N_TILE":
            node.args.pop()
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "PV_N_TILE":
            assert ast.literal_eval(node.value) == 128
            return None
        return self.generic_visit(node)


def main():
    checks = {}
    for name in ("entry.py", "cube_schedule.py", "attention_stream_stages.py"):
        old = RemoveNewWiring().visit(ast.parse((HERE / "source_before" / name).read_text()))
        new = RemoveNewWiring().visit(ast.parse((HERE / "source_candidate" / name).read_text()))
        checks[name] = ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
        assert checks[name], name
    spec = importlib.util.spec_from_file_location("pv_policy_candidate", HERE / "source_candidate/policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cases = []
    for history in (8192, 32768, 32769, 131072):
        for query in (1, 5, 6):
            for batch in (4, 8, 16, 20, 24, 28, 32):
                policy = module.select_policy(tokens=batch * query, requests=batch,
                    max_query_tokens=query, max_seq_len=history + query)
                expected = 256 if (history + query) // 4 > 8192 and policy.query_group == 6 else 128
                assert policy.pv_cols == expected == policy.constexprs()["PV_N_TILE"]
                cases.append(dict(history=history, query=query, batch=batch, pv_cols=expected,
                    query_group=policy.query_group))
    policy = module.select_policy(tokens=96, requests=16, max_query_tokens=6, max_seq_len=131078)
    rejected = []
    for value in (False, True, 0, 64, 129, 512, 128.0):
        try:
            replace(policy, pv_cols=value)
        except ValueError:
            rejected.append(repr(value))
        else:
            raise AssertionError(value)
    kwargs = {key: getattr(policy, key) for key in ("query_group", "qk_panel", "score_tile", "prefetch",
        "balanced", "roots", "oa_cols", "ob_rows")}
    assert module.CSASPMDPolicy(**kwargs).pv_cols == 128
    report = dict(source_math_after_removing_only_new_constexpr_wiring=checks,
        policy_cases=cases, rejected=rejected, manual_policy_default=128,
        no_runtime_source_changed=True)
    (HERE / "source_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("源码接线和策略检查通过：", len(cases), "个区间输入，", len(rejected), "个非法值/类型")


if __name__ == "__main__":
    main()
