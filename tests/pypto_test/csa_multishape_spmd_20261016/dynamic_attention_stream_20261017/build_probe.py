"""仅生成重复的动态Tensor签名，数学与循环全部调用正式源码helper。"""

import ast
from pathlib import Path

here = Path(__file__).resolve().parent
p = here.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/attention_stream_stages.py"
tree = ast.parse(p.read_text())
funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
cn = "attention_qk_pv_cube_stage"
vn = "attention_qk_pv_vector_lane_stage"
cargs = funcs[cn].args.args[:-3]
vargs = funcs[vn].args.args[:-4]
allargs = {}
for a in cargs + vargs:
    if a.arg not in allargs or "InOut" in ast.unparse(a.annotation):
        allargs[a.arg] = a
outputs = [
    "score_transfer",
    "pv_transfer",
    "kv_transfer",
    "probability_transfer",
    "mi_transfer",
    "li_transfer",
    "o_packed_heads",
]
imports = sorted({n.id for a in allargs.values() for n in ast.walk(a.annotation) if isinstance(n, ast.Name)} - {"pl"})
header = (
    '''# SPDX-License-Identifier: Apache-2.0
"""动态Attention成对MIX CPU探针，仅编译，不执行NPU。

输入按正式helper的真实动态描述符绑定：Q_ROWS=64*T，所有容量独立dynamic；
Q/L/positions/两页表/页容量及plan为同一不可变批次。plan由已验证准备阶段生成，
本探针把它作为已完成输入，不复制HC、Indexer或Attention计划数学。

阶段职责：同一24组MIX并发24AIC+48AIV。AIC执行QK(tick)、PV(tick-2)，
每组两个AIV执行采KV(tick)、softmax(tick-1)、归并/packed发布(tick-3)。
每组3槽跨query连续轮转，末尾排空；事件0..3分别KV/SCORE/PROB/PV，
不能先完成某角色再调用另一个，不能用阶段barrier替代逐块FFTS握手。
外层没有额外业务依赖，因此不加入phase信号。实际完整CSA须先等待Q/plan/cache。

cluster返回原GM别名，避免条件返回SSA使输出写回被消除；后端另外检查
环槽/packed真实TSTORE与动态ABI。CPU编译/事件模型不证明A3可见性或性能。
"""
import pypto.language as pl
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.attention_stream_stages import (
'''
    + "".join("    " + x + ",\n" for x in imports + [cn, vn])
    + ")\n"
)


def sig(args):
    return "".join("    " + a.arg + ": " + ast.unparse(a.annotation) + ",\n" for a in args)


def call(name, args):
    return name + "(" + ",".join(a.arg for a in args) + ")"


def ret(names):
    return "    return (" + ",".join(names) + ",)\n"


s = header
for name, args, helper, outs, extra in [
    ("cube", cargs, cn, outputs[:2], "worker,24,0"),
    ("vector", vargs, vn, outputs[2:], "worker,lane,24,0"),
]:
    s += "\n@pl.jit.incore\ndef " + name + "(\n" + sig(args) + "):\n"
    if name == "vector":
        s += '    pl.func_attr({"split_aiv":True})\n'
    for n in outs:
        s += "    " + n + "_gm=" + n + "\n"
    s += "    worker=pl.tile.get_block_idx()\n"
    if name == "vector":
        s += "    lane=pl.tile.get_subblock_idx()\n"
    s += "    " + ",".join(outs) + "=" + helper + "(" + ",".join(a.arg for a in args) + "," + extra + ")\n"
    s += ret([x + "_gm" for x in outs])
s += "\n@pl.jit.inline(auto_scope=False)\ndef group(\n" + sig(allargs.values()) + "):\n    with pl.cluster():\n"
s += "        " + ",".join(outputs[:2]) + "=" + call("cube", cargs) + "\n"
s += "        " + ",".join(outputs[2:]) + "=" + call("vector", vargs) + "\n" + ret(outputs)
s += "\n@pl.jit\ndef entry(\n" + sig(allargs.values()) + "):\n    with pl.spmd(24, sync_start=True):\n"
s += "        " + ",".join(outputs) + "=" + call("group", list(allargs.values())) + "\n" + ret(outputs)
(here / "probe.py").write_text(s)
