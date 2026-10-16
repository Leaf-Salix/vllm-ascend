# SPDX-License-Identifier: Apache-2.0
"""保留旧静态版本，给已有 128K/B16 生成临时信号 API 候选。"""

import ast
import copy
import json
from pathlib import Path


def parse_statements(text):
    return ast.parse(text).body


def call_name(node):
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return ast.unparse(node.value.func)
    return None


def main():
    root = Path(__file__).resolve().parent
    source = root / "static/h131072_b16"
    target = root / "static_local_signal/h131072_b16"
    target.mkdir(parents=True, exist_ok=False)
    module = ast.parse((source / "entry_jit.py").read_text())
    module.body.insert(1, ast.ImportFrom(
        module="local_signal_adapter",
        names=[ast.alias(name="csa_signal_notify"), ast.alias(name="csa_signal_wait")], level=0))
    audit = {}
    for role in ("aic", "aiv"):
        fn = next(x for x in module.body if isinstance(x, ast.FunctionDef)
                  and x.name == "csa_static_" + role)
        begin = next(i for i, x in enumerate(fn.body)
                     if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "own_flag")
        assert isinstance(fn.body[begin + 7], ast.Assign)
        assert ast.unparse(fn.body[begin + 7].targets[0]) == "record_0"
        fn.body[begin:begin + 7] = parse_statements("""
own_flag = core_id * 128
pl.system.cacheinvalid(progress, [1], [own_flag])
pl.system.fence()
prior_progress = pl.tensor.read(progress, [own_flag])
generation_base = (prior_progress // 1024 + 1) * 1024
progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base, pl.INT32))
""")
        records = []
        for stmt in fn.body:
            if not isinstance(stmt, ast.If) or not ast.unparse(stmt.test).startswith("mapped_block_"):
                continue
            number = int(stmt.test.left.id.removeprefix("mapped_block_"))
            assert isinstance(stmt.body[5], ast.For)
            assert [call_name(x) for x in stmt.body[6:8]] == [
                "pl.system.fence", "pl.system.cacheinvalid"]
            assert [call_name(x) for x in stmt.body[-4:-1]] == [
                "pl.system.fence", "pl.system.cacheinvalid", "pl.system.fence"]
            math = stmt.body[8:-4]
            original_math = ast.dump(ast.Module(body=copy.deepcopy(math), type_ignores=[]))
            dependency = stmt.body[5]
            assert len(dependency.body) == 8
            assert ast.unparse(dependency.body[4]).startswith(f"flag_{number} =")
            dependency.body[4] = parse_statements(f"flag_{number} = other_{number} * 128")[0]
            dependency.body[5:] = parse_statements(
                f"progress = csa_signal_wait(progress, flag_{number}, pl.cast(target_{number}, pl.INT32))")
            publication = stmt.body[-1]
            assert isinstance(publication, ast.If)
            publication.body = parse_statements(
                f"progress = csa_signal_notify(progress, own_flag, "
                f"pl.cast(generation_base + done_{number}, pl.INT32))")
            # 等待返回后先处理 payload 缓存，再等待缓存维护完成；原数学体不变。
            stmt.body = (stmt.body[:6]
                         + parse_statements("pl.system.fence()\npl.system.cacheinvalid()\npl.system.fence()")
                         + math + stmt.body[-4:])
            assert ast.dump(ast.Module(body=math, type_ignores=[])) == original_math
            records.append({"batch": number, "math_statements": len(math), "math_unchanged": True})
        audit[role] = records

    class SignalAnnotation(ast.NodeTransformer):
        def __init__(self):
            self.count = 0

        def visit_Subscript(self, node):
            if ast.unparse(node) == "pl.Tensor[[4608], pl.INT64]":
                self.count += 1
                return ast.parse("pl.Tensor[[9216], pl.INT32]", mode="eval").body
            return self.generic_visit(node)

    annotations = SignalAnnotation()
    module = annotations.visit(module)
    assert annotations.count == 8
    ast.fix_missing_locations(module)
    (target / "entry_jit.py").write_text(
        "# 临时信号候选：旧 static/h131072_b16 保留；只替换控制协议。\n"
        + ast.unparse(module) + "\n")
    layout = json.loads((source / "layout.json").read_text())
    layout["protocol"].update(progress_words_per_core=128, signal_api="temporary_local_set_ge",
                              signal_dtype="INT32", max_generation=(2**31 - 1) // 1024 - 1)
    progress = layout["workspace"]["sections"]["progress"]
    progress.update(shape=[9216], dtype="INT32")
    assert progress["bytes"] == 9216 * 4
    # 保留整个 arena 大小与其他 allocation 的原偏移，信号仍每 worker 独占 512 B。
    (target / "layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n")
    bindings = json.loads((source / "entry.bindings.json").read_text())
    for item in bindings["_entry_contract"]["ordered_inputs"]:
        if item["name"] == "progress":
            item["reference"].update(shape=[9216], dtype="INT32")
    (target / "entry.bindings.json").write_text(json.dumps(bindings, ensure_ascii=False, indent=2) + "\n")
    report = {"source": str(source), "candidate": str(target), "math_audit": audit,
              "annotation_replacements": annotations.count, "original_retained": True,
              "signal_stride_bytes": 512, "new_global_barriers": 0,
              "signal_only_contract": True, "device_validated": False}
    (target / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate": str(target), "math_batches": {k: len(v) for k, v in audit.items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
