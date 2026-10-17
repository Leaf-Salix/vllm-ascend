# SPDX-License-Identifier: Apache-2.0
"""从当前原动态 CSA 源码生成可观察的分 task 前段基线。

数学来源仅为 decode_csa.py 的 widen task、hc_pre.py 的两个动态 helper，以及
原 qkv_proj_rope.py 的 Q_A/QR。只把 HC 内部六项 create_tensor 改为调用者传入
的工作区，以便逐阶段比较；原循环、spmd、浮点运算及返回语义均保留。这里既不
读取历史 IR，也不使用新单 SPMD helper 作为参考。输出是高层动态源码，实际 T
仍由设备 Tensor 形状确定；没有按 T168/T169 展开计算。
"""

import ast
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MATH = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa"
PROBE = HERE.parent / "dynamic_front_pipeline_20261017/probe.py"


def dump(node):
    return ast.dump(node, include_attributes=False)


def observed_function(original, new_name, buffers):
    """限定变换为顶层工作区外置；逆变换后逐节点核对原函数。"""
    result = copy.deepcopy(original)
    result.name = new_name
    removed = []
    for name, annotation in buffers.items():
        assignments = [node for node in result.body if isinstance(node, ast.Assign)
                       and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id == name]
        assert len(assignments) == 1, (original.name, name)
        assignment = assignments[0]
        assert isinstance(assignment.value, ast.Call)
        assert ast.unparse(assignment.value.func) == "pl.create_tensor"
        removed.append((result.body.index(assignment), copy.deepcopy(assignment)))
        result.body.remove(assignment)
        result.args.args.append(ast.arg(arg=name, annotation=ast.parse(annotation, mode="eval").body))
    restored = copy.deepcopy(result)
    restored.name = original.name
    restored.args.args = restored.args.args[:len(original.args.args)]
    # 每次移除后的索引按逆序恢复，避免相邻 create_tensor 的位置偏移。
    for index, assignment in reversed(removed):
        restored.body.insert(index, assignment)
    assert dump(restored) == dump(original), original.name
    return result, {"source_function": original.name, "observed_function": new_name,
                    "only_externalized_allocations": list(buffers), "inverse_ast_identical": True}


def build(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hc_tree = ast.parse((MATH / "hc_pre.py").read_text())
    decode_tree = ast.parse((MATH / "decode_csa.py").read_text())
    functions = {node.name: node for node in hc_tree.body if isinstance(node, ast.FunctionDef)}
    gates, gates_audit = observed_function(functions["hc_pre_gates_from_rms"], "observed_gates", {
        "mixes_partials": "pl.Tensor[[PARTIAL, 32], pl.FP32]",
        "mixes_raw": "pl.Tensor[[P16, 32], pl.FP32]",
        "post_tail_store": "pl.Tensor[[8, 8], pl.FP32]",
        "comb_tail_store": "pl.Tensor[[8, 32], pl.FP32]",
    })
    mix, mix_audit = observed_function(functions["hc_mix_norm"], "observed_mix", {
        "x_mixed": "pl.Tensor[[P8, 4096], pl.BF16]",
        "x_normed_tail_store": "pl.Tensor[[8, 4096], pl.BF16]",
    })
    widen = [node for node in ast.walk(decode_tree) if isinstance(node, ast.With)
             and any(isinstance(item.context_expr, ast.Call)
                     and any(kw.arg == "name_hint" and isinstance(kw.value, ast.Constant)
                             and kw.value.value == "hc_widen_rms"
                             for kw in item.context_expr.keywords) for item in node.items)]
    assert len(widen) == 1
    constants = [node for node in decode_tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id.startswith("HC_WIDEN_")
                         for target in node.targets)]
    assert len(constants) == 3
    probe_tree = ast.parse(PROBE.read_text())
    entry = copy.deepcopy(next(node for node in probe_tree.body
                               if isinstance(node, ast.FunctionDef) and node.name == "entry"))
    # 原 QR staging 合同为固定容量 512；对有效 T 与实际写入 ceil(T/8) 分别核对。
    for argument in entry.args.args:
        if argument.arg in ("qr_matmul", "qr_scale_pad"):
            argument.annotation = ast.parse(ast.unparse(argument.annotation).replace("P8", "512"), mode="eval").body
    entry.body = ast.parse("""
t_dim = pl.tensor.dim(x, 0)
x_hc_flat = x
x_hc32_flat = x_fp32
x_hc32 = pl.reshape(x_fp32, [t_dim, HC_MULT, D])
widen_rows = (t_dim + HC_WIDEN_T_TILE - 1) // HC_WIDEN_T_TILE
inv_rms = inv
""").body + [copy.deepcopy(widen[0])] + ast.parse("""
pre = observed_gates(x_hc32, hc_fn, hc_scale, hc_base, pre, post, comb, False, inv,
                     partials, raw, post_tail, comb_tail)
observed_mix(x_hc32, pre, norm_w, normed, mixed, norm_tail)
q_proj_qa(normed, wq_a, qa, 0, t_dim)
q_proj_qr_normalize(qa, gamma_cq, qr, qr_scale, qr_matmul, qr_scale_pad, 0, t_dim)
return partials, qa, x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale, qr_matmul, qr_scale_pad
""").body
    header = '''# SPDX-License-Identifier: Apache-2.0
"""当前原动态分 task 前段的观察入口：只外置工作区，不改数学与任务划分。

本文件由 prepare_reference.py 从当前高层源码产生。阅读顺序为 entry 的原 widen
task、observed_gates、observed_mix，再到直接引用的原 Q_A/QR。每个阶段仍交给
原 TMR 数据依赖调度；不使用单 SPMD 的 signal 或新 hc_stages/qkv_stages。
T/P16/P8/PARTIAL 是实际 Tensor 维度，512 只是原 QR 暂存容量，不是固定 T。
"""
import pypto.language as pl
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.hc_pre import *
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.qkv_proj_rope import q_proj_qa, q_proj_qr_normalize
T = pl.dynamic("REFERENCE_T")
P16 = pl.dynamic("REFERENCE_P16")
P8 = pl.dynamic("REFERENCE_P8")
PARTIAL = pl.dynamic("REFERENCE_PARTIAL")
WORDS = pl.dynamic("REFERENCE_SIGNAL_WORDS")
'''
    tree = ast.fix_missing_locations(ast.Module(body=constants + [gates, mix, entry], type_ignores=[]))
    text = header + "\n" + ast.unparse(tree) + "\n"
    ast.parse(text)
    # 原 widen 节点未经重写；此证据与 HC 两个逆变换审计一起界定参考来源。
    generated_widen = next(node for node in ast.walk(ast.parse(text))
                           if isinstance(node, ast.With) and dump(node) == dump(widen[0]))
    assert dump(generated_widen) == dump(widen[0])
    path = output / "reference_source.py"
    path.write_text(text)
    audit = {"passed": True, "math_source": str(MATH), "probe_source": str(PROBE),
             "source_kind": "当前高层动态源码；不是历史 IR/静态 entry", "hc_functions": [gates_audit, mix_audit],
             "widen_task_ast_identical": True,
             "qa_qr": "直接调用当前 qkv_proj_rope.q_proj_qa / q_proj_qr_normalize",
             "guard_contract": "外置原中间容量；数学有效区与 pad 可写区分开检查",
             "device_execution": False}
    (output / "reference_source_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return path, audit


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path, audit = build(args.output)
    print(json.dumps({"source": str(path), "passed": audit["passed"]}, ensure_ascii=False))
