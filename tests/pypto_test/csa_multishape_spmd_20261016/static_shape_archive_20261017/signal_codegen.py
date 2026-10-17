# SPDX-License-Identifier: Apache-2.0
"""将参数化变体接到已验证的临时 notify/wait；不修改数学体。

沿用静态 128K/B16 修复：每核独占 512 B、INT32 代际信号、发布前排空
本核流水、读写侧维护缓存。此处没有全局 syncall，也不依赖 Scalar cache 一致性。
"""

import ast
import copy


def statements(source):
    return ast.parse(source).body


def call_name(node):
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return ast.unparse(node.value.func)
    return None


def use_local_signals(source, layout, bindings):
    module = ast.parse(source)
    module.body.insert(
        1,
        ast.ImportFrom(
            module="vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.signals",
            names=[ast.alias(name="csa_signal_notify"), ast.alias(name="csa_signal_wait")],
            level=0,
        ),
    )
    audit = {}
    for role in ("aic", "aiv"):
        fn = next(
            node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "csa_static_" + role
        )
        begin = next(
            i
            for i, node in enumerate(fn.body)
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "own_flag"
        )
        assert ast.unparse(fn.body[begin + 7].targets[0]) == "record_0"
        fn.body[begin : begin + 7] = statements("""
own_flag = core_id * 128
pl.system.cacheinvalid(progress, [1], [own_flag])
pl.system.fence()
prior_progress = pl.tensor.read(progress, [own_flag])
generation_base = (prior_progress // 1024 + 1) * 1024
progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base, pl.INT32))
""")
        batches = 0
        for stmt in fn.body:
            if not isinstance(stmt, ast.If) or not ast.unparse(stmt.test).startswith("mapped_block_"):
                continue
            number = int(stmt.test.left.id.removeprefix("mapped_block_"))
            assert isinstance(stmt.body[5], ast.For)
            assert [call_name(node) for node in stmt.body[6:8]] == ["pl.system.fence", "pl.system.cacheinvalid"]
            assert [call_name(node) for node in stmt.body[-4:-1]] == [
                "pl.system.fence",
                "pl.system.cacheinvalid",
                "pl.system.fence",
            ]
            math = stmt.body[8:-4]
            before = ast.dump(ast.Module(body=copy.deepcopy(math), type_ignores=[]))
            dependency = stmt.body[5]
            assert len(dependency.body) == 8
            dependency.body[4] = statements(f"flag_{number} = other_{number} * 128")[0]
            dependency.body[5:] = statements(
                f"progress = csa_signal_wait(progress, flag_{number}, pl.cast(target_{number}, pl.INT32))"
            )
            publication = stmt.body[-1]
            assert isinstance(publication, ast.If)
            publication.body = statements(
                f"progress = csa_signal_notify(progress, own_flag, pl.cast(generation_base + done_{number}, pl.INT32))"
            )
            stmt.body = (
                stmt.body[:6]
                + statements("pl.system.fence()\npl.system.cacheinvalid()\npl.system.fence()")
                + math
                + stmt.body[-4:]
            )
            assert ast.dump(ast.Module(body=math, type_ignores=[])) == before
            batches += 1
        audit[role] = {"batches": batches, "math_unchanged": True}

    class SignalType(ast.NodeTransformer):
        count = 0

        def visit_Subscript(self, node):
            if ast.unparse(node) == "pl.Tensor[[4608], pl.INT64]":
                self.count += 1
                return ast.parse("pl.Tensor[[9216], pl.INT32]", mode="eval").body
            return self.generic_visit(node)

    transform = SignalType()
    module = transform.visit(module)
    assert transform.count == 8
    ast.fix_missing_locations(module)
    layout["protocol"].update(
        progress_words_per_core=128,
        signal_api="temporary_local_set_ge",
        signal_dtype="INT32",
        max_generation=(2**31 - 1) // 1024 - 1,
    )
    progress = layout["workspace"]["sections"]["progress"]
    progress.update(shape=[9216], dtype="INT32")
    assert progress["bytes"] == 9216 * 4
    for item in bindings["_entry_contract"]["ordered_inputs"]:
        if item["name"] == "progress":
            item["reference"].update(shape=[9216], dtype="INT32")
    return (
        "# 参数化 CSA 单 SPMD 变体；数学体与所选分支一致。\n" + ast.unparse(module) + "\n",
        layout,
        bindings,
        {"roles": audit, "signal_stride_bytes": 512, "new_global_barriers": 0},
    )
