# SPDX-License-Identifier: Apache-2.0
"""TND attention独立FP64诊断；复用历史合成fixture，全部直接执行，不使用图。

参考是密集FP64 softmax+sink+逆RoPE，不复制被测块归约算法。
报告误差而不根据实测反设容差；此诊断不代替真实层Native输出或模型token验收。
"""

import argparse
import ast
import copy
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def generate_entry(package, *, long_attention=False):
    package = Path(package)
    tree = ast.parse((package / "attention.py").read_text())
    root = copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                              and n.name == "sparse_attn_hca_tp1"))
    root.name = "attention_probe"
    root.decorator_list = [ast.parse("pl.jit(auto_scope=False)", mode="eval").body]
    root.returns = None
    excluded = {"token_geometry", "raw_cache_ready_dep", "cmp_cache_ready_dep", "q_ready"}
    root.args.args = [arg for arg in root.args.args if arg.arg not in excluded]
    for arg in root.args.args:
        if arg.arg == "o_packed_heads":
            arg.annotation = ast.parse("pl.Out[pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16]]", mode="eval").body
    root.body = ast.parse('''
q.bind_dynamic(0, T_DYN)
position_ids.bind_dynamic(0, T_DYN)
kv_seq_lens.bind_dynamic(0, B_DYN)
query_bounds.bind_dynamic(0, BOUNDS_DYN)
ori_kv.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
ori_block_table.bind_dynamic(0, B_DYN)
ori_block_table.bind_dynamic(1, ORI_TABLE_COLUMNS_DYN)
cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
cmp_block_table.bind_dynamic(0, B_DYN)
cmp_block_table.bind_dynamic(1, CMP_TABLE_BLOCKS_DYN)
freqs_cos.bind_dynamic(0, T_DYN)
freqs_sin.bind_dynamic(0, T_DYN)
tokens = pl.tensor.dim(q, 0)
geometry = pl.create_tensor([tokens, 16], dtype=pl.INT32)
ready = build_token_geometry(query_bounds, kv_seq_lens, geometry)
q_ready = pl.array.create(4, pl.TASK_ID)
for group in pl.unroll(4):
    q_ready[group] = pl.system.task_invalid()
packed, done = sparse_attn_hca_tp1(
    q, ori_kv, ori_block_table, cmp_kv, cmp_block_table, position_ids, kv_seq_lens,
    query_bounds, geometry, attn_sink, freqs_cos, freqs_sin, o_packed_heads,
    ready, ready, q_ready,
)
return packed
''').body
    if long_attention:
        for node in ast.walk(root):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sparse_attn_hca_tp1":
                node.func.id = "sparse_attn_hca_tp1_long"
    ast.fix_missing_locations(root)
    (package / "standalone_attention_probe.py").write_text(
        '# SPDX-License-Identifier: Apache-2.0\n'
        '"""只包装实际生产/候选attention与实际TND geometry，不复制计算体。"""\n'
        'from .attention import *\n'
        'from .request_geometry import build_token_geometry\n\n' + ast.unparse(root) + '\n')


def load_variant(name, source):
    package = "vllm_ascend.ops.pypto._hca_attention_" + name
    path = source / "deepseek_v4_flash_hca_tnd"
    spec = importlib.util.spec_from_file_location(package, path / "__init__.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    return importlib.import_module(package + ".standalone_attention_probe")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int)
    parser.add_argument("--queries", help="独立合成B5的各请求Q长度；选择历史S6 fixture内的前几行")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(args.output / "build")
    os.environ["PYPTO_CACHE_DIR"] = str(args.output / "cache")
    from dsv4_csa_env import activate

    activate()
    import pypto.torch
    import torch
    import torch_npu
    from pypto import passes
    from pypto.runtime import CompileOptions
    import vllm_ascend.ops.pypto as parent
    from hca_online_softmax_20260930.probe import fixture, metrics

    parent.__path__ = [str(args.baseline.resolve())]
    modules = {name: load_variant(name, path.resolve())
               for name, path in (("control", args.baseline), ("candidate", args.candidate))}
    device = int(os.environ["TASK_DEVICE"]) if args.device is None else args.device
    if device != int(os.environ["TASK_DEVICE"]):
        raise ValueError("独立探针只能使用task-submit分配的设备")
    torch.set_num_threads(4)
    torch.npu.set_device(device)
    torch_npu.npu.set_deterministic_level(0)
    torch.npu.set_compile_mode(jit_compile=False)
    pypto.torch.init(device=device, platform="a2a3", runtime="tensormap_and_ringbuffer")
    options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
    operators = {name: pypto.torch.register(module.attention_probe, f"hca_attention_{name}::probe", config=options)
                 for name, module in modules.items()}
    module = modules["candidate"]
    report = {"scope": "合成分页attention对密集FP64；非真实层/模型精度验收", "aclgraph_replay": False,
              "shared_toolchain": os.environ.get("CSA_SHARED_TOOLCHAIN_RELEASE"), "cases": {},
              "candidate_attention_source": str(args.candidate.resolve() / "deepseek_v4_flash_hca_tnd/attention.py")}
    for history, edge in ((131072, False), (16507, True)):
        inputs, reference = fixture(module, history, edge)
        queries = [6] * 5 if args.queries is None else [int(v) for v in args.queries.split(",")]
        if len(queries) != 5 or any(not 1 <= q <= 6 for q in queries):
            raise ValueError("独立fixture为B5，每请求须有1..6行")
        ids = torch.tensor([i * 6 + j for i, q in enumerate(queries) for j in range(q)])
        # 同一请求只取前若干query；原FP64结果也选同样行，不复制/修改参考数学。
        for index in (0, 5, 8, 9):  # q / position / cos / sin，插入bounds前的参数位置
            inputs[index] = inputs[index].index_select(0, ids)
        reference = reference.index_select(0, ids)
        tokens = sum(queries)
        bounds = torch.tensor([0, *torch.tensor(queries).cumsum(0).tolist()], dtype=torch.int32)
        inputs.insert(7, bounds)
        device_inputs = [x.to(f"npu:{device}") for x in inputs]
        outputs = {}
        checks = {}
        for name, operator in operators.items():
            packed = torch.full((module.O_GROUPS * module.T_PAD, module.O_GROUP_IN), 17,
                                dtype=torch.bfloat16, device=f"npu:{device}")
            operator(*device_inputs, packed)
            torch.npu.synchronize()
            first = packed.cpu()
            operator(*device_inputs, packed)
            torch.npu.synchronize()
            actual = packed.cpu()
            if not torch.equal(first, actual):
                raise AssertionError("重复直接调用结果不一致")
            shaped = actual.reshape(module.O_GROUPS, module.T_PAD, module.O_GROUP_IN)
            if not torch.all(shaped[:, tokens:] == 17):
                raise AssertionError("独立输出保护区被覆盖")
            outputs[name] = shaped[:, :tokens].permute(1, 0, 2).reshape(tokens, 64, 512)
            checks[name] = metrics(outputs[name], reference)
            if checks[name]["nonfinite"] or (edge and torch.count_nonzero(outputs[name][-queries[-1]:])):
                raise AssertionError("有效输出非有限或失活请求输出非零")
            checks[name].update(repeat_eager_exact=True, output_padding_guard=True)
        report["cases"][str(history)] = {"edge": edge, "queries": queries, "vs_dense_fp64": checks,
                                        "candidate_vs_control": metrics(outputs["candidate"], outputs["control"])}
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report["status"] = "DIAGNOSTIC_COMPLETE"
    report["acceptance"] = "误差如实报告，后续真实层仍使用冻结Native输出容差，不由此推导新容差"
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
