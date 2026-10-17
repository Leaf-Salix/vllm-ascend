# SPDX-License-Identifier: Apache-2.0
"""按 Scalar 的实际 GM 访问保留业务数据缓存维护。

AIC/AIV 的 Tile DMA 不通过 Scalar DCache；仍在阶段边界完整排空本核流水。
存在 Scalar GM 写的阶段保留写后 DCCI+DSB；存在 Scalar GM 读的阶段保留读前
DCCI+DSB。每轮入口额外失效整份 Scalar DCache，避免上轮输入残留。
信号本身的发布/等待维护完全不变，不依赖核间缓存一致性。
"""

import ast
import copy


def call(node):
    return ast.unparse(node.value.func) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) else None


def specialize_payload_cache(source):
    tree = ast.parse(source)
    audit = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name not in {"csa_static_aic", "csa_static_aiv"}:
            continue
        role = fn.name.removeprefix("csa_static_")
        initial = [node for node in fn.body if call(node) == "pl.system.cacheinvalid"]
        if len(initial) != 1 or not initial[0].value.args:
            raise ValueError("入口进度缓存维护边界改变")
        # 原先只维护本核进度；本方案在每次调用开始维护全部 Scalar 输入缓存。
        initial[0].value.args = []
        for phase in fn.body:
            if not isinstance(phase, ast.If) or not ast.unparse(phase.test).startswith("mapped_block_"):
                continue
            index = int(phase.test.left.id.removeprefix("mapped_block_"))
            if [call(node) for node in phase.body[6:9]] != [
                    "pl.system.fence", "pl.system.cacheinvalid", "pl.system.fence"]:
                raise ValueError("阶段读前维护边界改变")
            if [call(node) for node in phase.body[-4:-1]] != [
                    "pl.system.fence", "pl.system.cacheinvalid", "pl.system.fence"]:
                raise ValueError("阶段写后维护边界改变")
            mathematical_body = copy.deepcopy(phase.body[9:-4])
            reads, writes = set(), set()
            for node in ast.walk(ast.Module(body=mathematical_body, type_ignores=[])):
                if not isinstance(node, ast.Call):
                    continue
                name = ast.unparse(node.func)
                if name in {"pl.tensor.read", "pl.read"}:
                    reads.add(ast.unparse(node.args[0]))
                elif name in {"pl.tensor.write", "pl.write"}:
                    writes.add(ast.unparse(node.args[0]))
                elif ("read" in name or "write" in name) and name not in {"pl.tile.read", "pl.tile.write"}:
                    raise ValueError("未分析的读写接口：" + name)
            before = phase.body[6:9] if reads else phase.body[6:7]
            after = phase.body[-4:-1] if writes else phase.body[-4:-3]
            phase.body = phase.body[:6] + before + mathematical_body + after + phase.body[-1:]
            audit.append({"role": role, "stage": index, "scalar_gm_reads": sorted(reads),
                          "scalar_gm_writes": sorted(writes), "entry_dcci_kept": bool(reads),
                          "exit_dcci_kept": bool(writes), "payload_pipeline_fences_kept": True,
                          "signal_cache_maintenance_unchanged": True, "math_unchanged": True})
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n", audit
