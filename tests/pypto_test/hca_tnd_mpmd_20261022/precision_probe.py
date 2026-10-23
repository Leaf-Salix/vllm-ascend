# SPDX-License-Identifier: Apache-2.0
"""HCA差异归因：复用生产数学采集中间值，并以Native尾链做诊断拼接。

不新增一套QKV/attention计算，不改变容差，不采性能。trace入口只把已有
normalized/post/comb/packed工作区改绑到输出，并增加等q_ready的Q复制任务。
trace与正式入口的最终输出及cache/state必须精确一致，才解释中间值。
"""

import ast
import copy
import importlib
import json
from pathlib import Path


def generate_entry(package):
    package = Path(package)
    tree = ast.parse((package / "entry.py").read_text())
    root = copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                              and n.name == "_decode_hca_tp1_layer"))
    root.name = "trace_hca"
    root.decorator_list = [ast.parse("pl.jit(auto_scope=False)", mode="eval").body]
    fields = {
        "normalized": "[T_DYN, D], pl.BF16",
        "post": "[T_DYN, HC_MULT], pl.FP32",
        "comb": "[T_DYN, HC_MULT * HC_MULT], pl.FP32",
        "packed": "[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16",
        "q": "[T_DYN, H * HEAD_DIM], pl.BF16",
    }
    for name, spec in fields.items():
        root.args.args.append(ast.arg(arg="debug_" + name, annotation=ast.parse(
            f"pl.Out[pl.Tensor[{spec}]]", mode="eval").body))
    replaced = []

    class Capture(ast.NodeTransformer):
        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in fields and node.targets[0].id != "q"
                    and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == "pl.create_tensor"):
                name = node.targets[0].id
                replaced.append(name)
                node.value = ast.Name(id="debug_" + name, ctx=ast.Load())
            return self.generic_visit(node)

    root = Capture().visit(root)
    if sorted(replaced) != ["comb", "normalized", "packed", "post"]:
        raise ValueError(f"生产工作区定义改变，须重审采集点：{replaced}")
    tap = ast.parse('''
with pl.spmd(48, name_hint="diagnostic_q_copy", deps=[q_ready]):
    for qt in pl.range(pl.tile.get_block_idx(), tokens, 48):
        debug_q[qt:qt + 1, :] = q[qt:qt + 1, :]
''').body
    # q在包含qkv_proj_rope的pl.scope内产生新的SSA别名；采集也留在该scope内。
    # 不让诊断额外引入一个跨编排函数的临时别名消费。
    owner = [node for node in ast.walk(root) if isinstance(node, ast.With)
             and any(isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
                     and ast.unparse(stmt.value.func) == "qkv_proj_rope" for stmt in node.body)]
    if len(owner) != 1:
        raise ValueError("Q生产者scope改变，须重审采集点")
    owner[0].body.extend(tap)
    ast.fix_missing_locations(root)
    tail = '''

@pl.jit(auto_scope=False)
def trace_output(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    ready = pl.system.task_dummy(deps=[])
    output = o_proj_hc_post(packed, wo_a, wo_b, wo_scale, residual, post, comb, output, ready)
    return output
'''
    extra_trace = ""
    if any(isinstance(n, ast.FunctionDef) and n.name == "_make_decode_hca" for n in tree.body):
        tall_root = copy.deepcopy(root)
        tall_root.name = "trace_hca_tall"
        for node in ast.walk(tall_root):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "qkv_proj_rope":
                node.func.id = "qkv_proj_rope_tall"
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "o_proj_hc_post" and "o_proj_hc_post_tall" in ast.unparse(tree)):
                node.func.id = "o_proj_hc_post_tall"
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "sparse_attn_hca_tp1" and "sparse_attn_hca_tp1_long" in ast.unparse(tree)):
                node.func.id = "sparse_attn_hca_tp1_long"
        extra_trace = "\n\n" + ast.unparse(tall_root)
    if "sparse_attn_hca_tp1_long" in ast.unparse(tree):
        long_root = copy.deepcopy(root)
        long_root.name = "trace_hca_long"
        for node in ast.walk(long_root):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sparse_attn_hca_tp1":
                node.func.id = "sparse_attn_hca_tp1_long"
        extra_trace += "\n\n" + ast.unparse(long_root)
    if "o_proj_hc_post_tall" in ast.unparse(tree):
        tail += tail.replace("def trace_output(", "def trace_output_tall(").replace(
            "output = o_proj_hc_post(", "output = o_proj_hc_post_tall(")
    (package / "trace_entry.py").write_text(
        '# SPDX-License-Identifier: Apache-2.0\n'
        '"""自动提取生产入口，只增加中间工作区采集；不用于性能计时。"""\n'
        'from .entry import *\n\n' + ast.unparse(root) + extra_trace + '\n' + tail)


def capture_native(layer):
    """只在独立诊断进程中包装Native实际方法；返回恢复函数。"""
    from vllm_ascend.device.device_op import DeviceOperator

    data = {}
    original_pre = layer.hc_pre
    impl = layer.self_attn.dsa_attn.dsa_attn.impl
    original_output = impl._forward_o_proj
    original_selector = DeviceOperator.get_dsa_sparse_attn_op
    attention_op = original_selector()

    def pre(*args, **kwargs):
        values = original_pre(*args, **kwargs)
        for name, value in zip(("mixed", "post", "comb"), values):
            data[name] = value.detach().clone()
        return values

    def norm(module, inputs, value):
        data["normalized"] = value.detach().clone()

    def output(value, target):
        data["packed"] = value.detach().clone()
        result = original_output(value, target)
        data["projected"] = result.detach().clone()
        return result

    def attention(query, *args, **kwargs):
        data["q"] = query.detach().clone()
        return attention_op(query, *args, **kwargs)

    layer.hc_pre = pre
    impl._forward_o_proj = output
    DeviceOperator.get_dsa_sparse_attn_op = staticmethod(lambda: attention)
    hook = layer.input_layernorm.register_forward_hook(norm)

    def release():
        layer.hc_pre = original_pre
        impl._forward_o_proj = original_output
        DeviceOperator.get_dsa_sparse_attn_op = staticmethod(original_selector)
        hook.remove()

    return data, release


def run_probe(*, package, options, call, fixture, layer, config, output, baseline, native,
              native_tensors, restore, collect, output_dir):
    import pypto.torch
    import torch
    from dsv4_csa_validation import compare_tensor
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    module = importlib.import_module(package + ".trace_entry")
    tokens = fixture["tokens"]
    active = fixture["actual_tokens"]
    captures = {
        "debug_normalized": torch.empty((tokens, 4096), dtype=torch.bfloat16, device=output.device),
        "debug_post": torch.empty((tokens, 4), dtype=torch.float32, device=output.device),
        "debug_comb": torch.empty((tokens, 16), dtype=torch.float32, device=output.device),
        "debug_packed": torch.empty((8 * module.T_PAD, 4096), dtype=torch.bfloat16, device=output.device),
        "debug_q": torch.empty((tokens, 32768), dtype=torch.bfloat16, device=output.device),
    }
    policy = getattr(call, "policy", "default")
    trace_root = module.trace_hca
    if policy == "tall_projection":
        trace_root = module.trace_hca_tall
    elif policy == "long_attention":
        trace_root = module.trace_hca_long
    trace = pypto.torch.register(trace_root, "hca_tnd_stage_probe::attention", config=options)
    tail_root = (module.trace_output_tall if getattr(call, "policy", "default") == "tall_projection"
                 else module.trace_output)
    tail = pypto.torch.register(tail_root, "hca_tnd_stage_probe::output", config=options)
    restore(fixture)
    trace(**call.args, **captures)
    torch.npu.synchronize()
    actual = collect()
    exact = {"output": compare_tensor(actual["output"], baseline["output"], 0, 0)}
    for name in fixture["groups"]:
        exact[name] = [compare_tensor(a, b, 0, 0) for a, b in zip(actual[name], baseline[name])]
    checks = [q for v in exact.values() for q in (v if isinstance(v, list) else [v])]
    result = {"production_exact": exact, "passed_capture_equivalence": all(q["status"] == "PASS" for q in checks),
              "performance_measured": False, "aclgraph_replay": False}
    if not result["passed_capture_equivalence"]:
        return result
    packed = captures["debug_packed"].view(8, module.T_PAD, 4096)[:, :tokens].transpose(0, 1).contiguous()
    packed = packed.view(tokens, 64, 512)
    pto_values = {"normalized": captures["debug_normalized"], "post": captures["debug_post"],
                  "comb": captures["debug_comb"], "q": captures["debug_q"].view(tokens, 64, 512),
                  "packed": packed}
    result["intermediates"] = {}
    for name, value in pto_values.items():
        reference = native_tensors[name].reshape_as(value)
        value_cpu, ref_cpu = value.cpu(), reference.cpu()
        result["intermediates"][name] = {
            "all": compare_tensor(value_cpu[:active], ref_cpu[:active], 0.0, 0.0),
            "token143": compare_tensor(value_cpu[143:144], ref_cpu[143:144], 0.0, 0.0) if active > 143 else None,
        }
    final_reference = native["output"][:active]
    impl = layer.self_attn.dsa_attn.dsa_attn.impl
    native_projected_from_pto = torch.empty((tokens, 4096), dtype=torch.bfloat16, device=output.device)
    with set_ascend_forward_context(fixture["metadata"], config, num_tokens=tokens, num_actual_tokens=active):
        impl._forward_o_proj(packed, native_projected_from_pto)
    alternatives = {}
    for label, projected, post, comb in (
        ("native_tail_pto_gates", native_projected_from_pto, captures["debug_post"], captures["debug_comb"]),
        ("native_tail_native_gates", native_projected_from_pto, native_tensors["post"], native_tensors["comb"]),
        ("native_attention_pto_gates", native_tensors["projected"], captures["debug_post"], captures["debug_comb"]),
    ):
        alternatives[label] = layer.hc_post(projected, fixture["hidden"], post, comb.view(tokens, 4, 4))
    # 同一生产O尾链，输入替换成Native attention与门控，隔离上游误差。
    native_packed = torch.zeros_like(captures["debug_packed"])
    native_packed.view(8, module.T_PAD, 4096)[:, :tokens].copy_(
        native_tensors["packed"].view(tokens, 8, 4096).transpose(0, 1))
    tail_result = torch.empty_like(output)
    tail(native_packed, call.args["wo_a"], call.args["wo_b"], call.args["wo_b_scale"],
         fixture["hidden"], native_tensors["post"], native_tensors["comb"].reshape(tokens, 16), tail_result)
    alternatives["pto_tail_native_attention_and_gates"] = tail_result
    torch.npu.synchronize()
    result["splices"] = {name: compare_tensor(value[:active].cpu(), final_reference, 0.01, 0.01)
                         for name, value in alternatives.items()}
    # 只存诊断激活；权重不重复落盘，结果不纳入Git。
    dump = Path(output_dir) / "stage_values.pt"
    torch.save({"native": {k: v.cpu() for k, v in native_tensors.items()},
                "pto": {k: v.cpu() for k, v in pto_values.items()},
                "splices": {k: v.cpu() for k, v in alternatives.items()},
                "output_native": native["output"], "output_pto": baseline["output"]}, dump)
    result["local_values"] = str(dump)
    (Path(output_dir) / "precision_probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
