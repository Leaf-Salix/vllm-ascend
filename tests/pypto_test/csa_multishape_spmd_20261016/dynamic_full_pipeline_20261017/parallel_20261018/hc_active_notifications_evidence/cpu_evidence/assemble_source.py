#!/usr/bin/env python3
"""把已有源级动态阶段连成正式源码；只生成接口重复项，不读历史IR/shape/DAG。

此脚本是迁移过程记录，正式运行不调用它。数学函数仍为ops/pypto中的动态
helper；编排体为本目录两个人工编写的.txt源片段。输入规格完全不在生成参数中。
"""

import ast
import json
import re
import textwrap
from pathlib import Path

from advance_hc_norm import advance
from prioritize_indexer import prioritize_cube, prioritize_vector

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PKG = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
CASE = HERE.parent


def read_function(path, name):
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return source, node


def rename(source, names):
    return re.sub(r"\b(" + "|".join(sorted(names, key=len, reverse=True)) + r")\b", lambda m: names[m[0]], source)


def base_body(path, name):
    source, fn = read_function(path, name)
    body = fn.body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return textwrap.dedent("\n".join(source.splitlines()[body[0].lineno - 1 : body[-1].lineno - 1]))


def main():
    args = {}
    dims = set()

    def add(name, annotation):
        if name in args:
            assert args[name] == annotation, (name, args[name], annotation)
        args[name] = annotation

    # 先取已通过完整后端的前段ABI；仅将硬编码NZ注解替换成同一Native布局规则。
    front = CASE / "dynamic_qkv_front_pipeline_20261017/probe.py"
    _, entry = read_function(front, "entry")
    for arg in entry.args.args:
        annotation = ast.unparse(arg.annotation)
        if arg.arg == "wq_a":
            annotation = annotation.replace("pl.NZ", "BF16_WEIGHT_LAYOUT")
        if arg.arg == "wq_b":
            annotation = annotation.replace("pl.NZ", "QUANT_WEIGHT_LAYOUT")
        add(arg.arg, annotation)

    comp = CASE / "dynamic_compressor_pipeline_20261017/probe.py"
    _, comp_entry = read_function(comp, "entry")
    replacements = {
        "MT": "T",
        "IT": "T",
        "MB": "B",
        "IB": "B",
        "main_x": "normed",
        "idx_x": "normed",
        "main_positions": "positions",
        "idx_positions": "positions",
        "main_table": "main_state_table",
        "idx_table": "idx_state_table",
    }
    for arg in comp_entry.args.args:
        name = replacements.get(arg.arg, arg.arg)
        if name in args:
            continue
        add(name, rename(ast.unparse(arg.annotation), replacements))

    def tensor(name, shape, dtype, writes=False, layout=""):
        annotation = f"pl.Tensor[[{shape}], pl.{dtype}{', ' + layout if layout else ''}]"
        add(name, f"pl.InOut[{annotation}]" if writes else annotation)

    # Native视图必须在Host按已有shape/stride无损绑定；没有device数据取值。
    tensor("q_heads", "QUERY_HEAD_ROWS, 512", "BF16")
    tensor("original_table", "B, ORIGINAL_COLS", "INT32")
    tensor("compressed_table", "B, COMPRESSED_COLS", "INT32")
    tensor("indexer_table", "B, INDEXER_COLS", "INT32")
    tensor("idx_wq_b", "1024, 8192", "INT8")
    tensor("idx_wq_b_scale", "8192", "FP32")
    tensor("idx_query_acc", "IDX_PROJ_ROWS, 8192", "INT32", True)
    tensor("idx_query_rope", "IDX_ROPE_ROWS, 8192", "BF16", True)
    tensor("idx_query_rope_heads", "IDX_ROPE_HEAD_ROWS, 128", "BF16")
    tensor("idx_query_had", "IDX_HEAD_ROWS, 128", "FP32", True)
    tensor("idx_query_i8", "IDX_HEAD_ROWS, 128", "INT8", True)
    tensor("idx_query_scale", "IDX_HEAD_ROWS, 1", "FP32", True)
    tensor("idx_query_scale_heads", "IDX_SCALE_T_ROWS, 64", "FP32")
    tensor("weights_proj", "4096, 64", "BF16")
    tensor("weights_partial", "WEIGHTS_PARTIAL_ROWS, 64", "FP32", True)
    tensor("head_weights", "WEIGHTS_ROWS, 64", "FP32", True)
    tensor("group_offsets", "GROUP_OFFSET_ROWS", "INT32", True)
    tensor("group_plan", "8", "INT32", True)
    tensor("coefficients", "COEFF_ROWS, COEFF_COLS", "FP16", True)
    tensor("idx_key0", "KEY0_ROWS, 128", "INT8")
    tensor("idx_key64", "KEY64_ROWS, 128", "INT8")
    tensor("indexer_transfer", "INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS", "FP32", True)
    tensor("score_arena", "SCORE_ROWS, 8192", "FP32", True)
    tensor("pair_arena", "PAIR_ROWS, 1024", "FP32", True)
    tensor("topk_scores", "T, 512", "FP32", True)
    tensor("topk_indices", "T, 512", "INT32", True)
    tensor("attention_indices", "ATTN_PLAN_ROWS, 512", "INT32", True)
    tensor("attention_bias", "ATTN_PLAN_ROWS, 640", "FP32", True)
    tensor("attention_mask", "ATTN_PLAN_ROWS, 16", "INT32", True)
    tensor("inverse_sin", "T, 64", "FP32", True)
    tensor("attn_sink", "64, 1", "FP32")
    tensor("kv_ring", "KV_RING_ROWS, 512", "BF16", True)
    tensor("score_ring", "HEAD_RING_ROWS, 128", "FP32", True)
    tensor("probability_ring", "HEAD_RING_ROWS, 128", "BF16", True)
    tensor("pv_ring", "HEAD_RING_ROWS, 512", "FP32", True)
    tensor("mi_ring", "HEAD_RING_ROWS, 1", "FP32", True)
    tensor("li_ring", "HEAD_RING_ROWS, 1", "FP32", True)
    tensor("packed", "PACK_ROWS, 4096", "BF16", True)
    tensor("ffts", "256", "INT64")
    tensor("wo_a", "8, 4096, 1024", "BF16", layout="BF16_WEIGHT_LAYOUT")
    tensor("o_r", "O_MM_ROWS, 8192", "FP32", True)
    tensor("o_quant", "O_MM_ROWS, 8192", "INT8", True)
    tensor("o_scales", "8, O_SCALE_ROWS", "FP32", True)
    tensor("wo_b", "8192, 4096", "INT8", layout="QUANT_WEIGHT_LAYOUT")
    tensor("o_partials", "O_MM_ROWS, 32768", "INT32", True)
    tensor("wo_b_scale", "4096", "FP32")
    tensor("output", "T, 16384", "BF16", True)

    # 输入顺序稳定，signal置末；所有容量轴仍为dynamic，不传样例的B/T/L。
    args["signal"] = args.pop("signal")
    controls = ["QUERY_GROUP", "QK_PANEL", "SCORE_TILE", "PREFETCH", "TO_L0", "BALANCED", "ROOTS", "OA_COLS", "OB_ROWS"]
    for ann in args.values():
        dims.update(re.findall(r"\b[A-Z][A-Z_0-9]*\b", ann))
    dims -= {"FP32", "BF16", "FP16", "INT8", "INT32", "INT64", "NZ", "BF16_WEIGHT_LAYOUT", "QUANT_WEIGHT_LAYOUT"}

    phases = [
        "INPUT",
        "PARTIAL",
        "RAW",
        "GATES",
        "COMB",
        "NORM",
        "QA",
        "QR",
        "ROPE",
        "QB",
        "QUERY",
        "KV_PROJECT",
        "KV_NORM",
        "ORIGINAL_CACHE",
        "META",
        "IDX_QUERY_PROJECT",
        "IDX_QUERY_ROPE",
        "IDX_QUERY_HAD",
        "IDX_QUERY_QUANT",
        "WEIGHTS_PROJECT",
        "WEIGHTS",
        "MAIN_PROJECT",
        "IDX_PROJECT",
        "MAIN_POOL",
        "MAIN_COMMIT",
        "MAIN_OUTPUT",
        "IDX_POOL",
        "IDX_COMMIT",
        "IDX_NORM",
        "IDX_HADAMARD",
        "IDX_QUANT",
        "IDX_SCALE",
        "COEFF",
        "SCORE_CUBE",
        "SCORE_VECTOR",
        "TOPK",
        "ATTN_PLAN",
        "ATTN_CUBE",
        "ATTN_VECTOR",
        "OA",
        "O_QUANT",
        "OB",
        "OUT",
    ]
    abi = '''# SPDX-License-Identifier: Apache-2.0
"""动态CSA统一ABI与阶段编号：不含具体batch、历史长度或冻结执行图。

T为实际输入行容量（允许Native dummy行），B为Native请求槽数。MQ/IQ是两套
query前缀Tensor的独立长度，实际有效边界由device Q/L/position解释。其余
dynamic轴描述独立GM容量；不能把较大的容量当成有效行数。详细绑定合同在
entry.py；各数学文件说明最小容量和尾行。核组数为A3启动配置，不是shape档位。

阶段0保存每核调用代际；其余阶段一个核每调用只发布一次。阶段编号只标识数据，
不强制数字顺序。META由一个AIV写，其他完成槽归各核；512B隔离步长见phase_sync。
同一signal工作区必须固定本文件阶段含义、72参与者、单流顺序生命周期。
"""

import pypto.language as pl

GROUPS = 24
VECTORS = 48
CORES = 72
'''
    abi += "\n".join(f'{d} = pl.dynamic("CSA_FULL_{d}")' for d in sorted(dims)) + "\n\n"
    abi += "\n".join(f"PHASE_{name} = {i}" for i, name in enumerate(phases, 1)) + f"\nPHASE_COUNT = {len(phases) + 1}\n"
    (PKG / "runtime_abi.py").write_text(abi)

    cube = base_body(front, "cube")
    vector = base_body(front, "vector")
    vector = advance(vector, "")
    # 统一在函数入口保存全部写缓冲的原GM别名，避免条件返回值被当作新分配。
    cube = re.sub(r"^\w+_gm = \w+\n", "", cube, flags=re.M)
    cube = cube.replace(
        "signal, generation = begin_invocation(signal, worker)",
        "signal, generation = begin_invocation(signal, worker)\npl.system.set_ffts(ffts)",
    )
    vector = vector.replace('pl.func_attr({"split_aiv": True})\n', "")
    vector = vector.replace(
        "signal, generation = begin_invocation(signal, core)",
        "signal, generation = begin_invocation(signal, core)\npl.system.set_ffts(ffts)",
    )
    marker = "signal = publish_phase(signal, PHASE_INPUT, core, CORES, generation)\n"
    early = (HERE / "metadata_schedule.txt").read_text()
    vector = vector.replace(marker, marker + "\n" + early + "\n")
    vector = vector.replace(
        "signal = publish_phase(signal, PHASE_ROPE, core, CORES, generation)",
        "inverse_sin = attention_inverse_rope_sign_stage(rope_sin, inverse_sin, worker, VECTORS)\n"
        "signal = publish_phase(signal, PHASE_ROPE, core, CORES, generation)",
    )
    cube += "\n" + (HERE / "cube_schedule.txt").read_text()
    vector += "\n" + (HERE / "vector_schedule.txt").read_text()
    cube = prioritize_cube(cube, "")
    vector = prioritize_vector(vector, "")

    # 正式helper导入来自自身源文件，不从tests导入任何数学或运行时生成器。
    helper_modules = {}
    for f in PKG.glob("*_stages.py"):
        for node in ast.parse(f.read_text()).body:
            if isinstance(node, ast.FunctionDef):
                helper_modules[node.name] = f.stem
    helper_modules.update(
        {
            "q_b_stage": "query_projection_stages",
            "o_b_stage": "output_projection_b_stages",
            "begin_invocation": "phase_sync",
            "wait_producer": "phase_sync",
            "wait_row_owners": "phase_sync",
            "publish_phase": "phase_sync",
        }
    )

    cube_writes = set(
        [
            "partials",
            "qa",
            "q_proj",
            "kv_fp32",
            "idx_query_acc",
            "weights_partial",
            "main_value",
            "main_score",
            "idx_value",
            "idx_score",
            "idx_query_had",
            "idx_had",
            "indexer_transfer",
            "score_ring",
            "pv_ring",
            "o_r",
            "o_partials",
            "signal",
        ]
    )
    vector_writes = {k for k, v in args.items() if "InOut" in v} - cube_writes | {"signal"}
    # 业务写者唯一；跨角色读取的别名在入口显式绑定同一GM，InOut只用于实际写角色。
    roles = {}
    for role, body, writes in (("cube", cube, cube_writes), ("vector", vector, vector_writes)):
        tree = ast.parse(body)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        role_args = {
            k: v if k in writes else v.removeprefix("pl.InOut[").removesuffix("]") if "pl.InOut[" in v else v
            for k, v in args.items()
            if k in used
        }
        assert writes <= role_args.keys(), (role, writes - role_args.keys())
        roles[role] = (role_args, [k for k in args if k in writes])
        functions = sorted(
            {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            & helper_modules.keys()
        )
        head = (HERE / (role + "_overview.txt")).read_text() + "\nimport pypto.language as pl\n\n"
        all_names = used | set().union(*(set(re.findall(r"\b\w+\b", v)) for v in role_args.values()))
        constants = sorted((set(dims) | {"PHASE_" + n for n in phases} | {"GROUPS", "VECTORS", "CORES"}) & all_names)
        head += "from .runtime_abi import (\n" + "".join(f"    {n},\n" for n in constants) + ")\n"
        if "BF16_WEIGHT_LAYOUT" in all_names:
            head += "from .qkv_stages import BF16_WEIGHT_LAYOUT\n"
        if "QUANT_WEIGHT_LAYOUT" in all_names:
            head += "from .query_projection_stages import QUANT_WEIGHT_LAYOUT\n"
        modules = {}
        for name in functions:
            modules.setdefault(helper_modules[name], []).append(name)
        for mod, names in sorted(modules.items()):
            head += f"from .{mod} import (" + ", ".join(names) + ")\n"
        head += f"\n\n@pl.jit.incore\ndef {role}_schedule(\n"
        head += "".join(f"    {k}: {v},\n" for k, v in role_args.items())
        head += "".join(f"    {c}: pl.constexpr,\n" for c in controls) + "):\n"
        if role == "vector":
            head += '    pl.func_attr({"split_aiv": True})\n'
        else:
            cache_params = [(list(role_args).index(n), 1) for n in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")]
            head += "    # 预定义incore的cache_policy按当前函数参数序号声明；保持原五份权重BYPASS。\n"
            head += f'    pl.func_attr({{"cache_policy": {cache_params!r}}})\n'
        head += "".join(f"    {k}_gm = {k}\n" for k in roles[role][1])
        head += textwrap.indent(body, "    ").rstrip() + "\n"
        head += "    return (" + ", ".join(k + "_gm" for k in roles[role][1]) + ")\n"
        (PKG / (role + "_schedule.py")).write_text(head)

    entry_head = (HERE / "entry_overview.txt").read_text() + "\nimport pypto.language as pl\n\n"
    entry_head += "from .runtime_abi import (\n" + "".join(f"    {d},\n" for d in sorted(dims)) + "    GROUPS,\n)\n"
    entry_head += (
        "from .qkv_stages import BF16_WEIGHT_LAYOUT\nfrom .query_projection_stages import QUANT_WEIGHT_LAYOUT\n"
    )
    entry_head += "from .cube_schedule import cube_schedule\nfrom .vector_schedule import vector_schedule\n"
    writes = [k for k, v in args.items() if "InOut" in v]
    signature = "".join(f"    {k}: {v},\n" for k, v in args.items()) + "".join(
        f"    {c}: pl.constexpr,\n" for c in controls
    )
    entry_head += "\n\n@pl.jit.inline(auto_scope=False)\ndef _csa_core_group(\n" + signature + "):\n"
    entry_head += "    # cluster将两种角色合成同一个MIX任务；这里的源码顺序不把AIC/AIV串行执行。\n"
    entry_head += '    with pl.cluster(name_hint="dynamic_csa_full"):\n'
    for role, (a, w) in roles.items():
        entry_head += "        (" + ", ".join(w) + ") = " + role + "_schedule(\n"
        entry_head += "".join("            " + k + ",\n" for k in [*a, *controls]) + "        )\n"
    entry_head += "    return (" + ", ".join(writes) + ")\n"
    runtime_args = {k: v for k, v in args.items() if k not in ("idx_key0", "idx_key64")}
    entry_signature = "".join(f"    {k}: {v},\n" for k, v in runtime_args.items()) + "".join(
        f"    {c}: pl.constexpr,\n" for c in controls
    )
    entry_head += "\n\n@pl.jit\ndef decode_csa_spmd(\n" + entry_signature + "):\n"
    entry_head += """    # Native cache只作为一个根入参传入。两份Key是编排内的零拷贝描述符，
    # 避免把“可写cache+部分重叠只读视图”当作独立外部入参触发Torch别名拒绝。
    # 这里只读取Tensor shape并建立GM view，没有Host取device值、复制cache或额外task。
    pages = pl.tensor.dim(idx_cache, 0)
    page_bytes = pl.tensor.dim(idx_cache, 1)
    cache_flat = pl.reshape(idx_cache, [pages * page_bytes])
    # 两个视图均只包含完整128字节行；它们重叠但只读。cache写入仍走原始页布局。
    # 切片先命名再reshape，让JIT沿张量维度表达式传递类型；不创建额外GM allocation。
    prefix = cache_flat[0:(pages * page_bytes // 128) * 128]
    shifted = cache_flat[64:64 + ((pages * page_bytes - 64) // 128) * 128]
    idx_key0 = pl.reshape(prefix, [pages * page_bytes // 128, 128])
    idx_key64 = pl.reshape(shifted, [(pages * page_bytes - 64) // 128, 128])
"""
    entry_head += "    # 所有数学/设备metadata/跨核等待均在这一次SPMD内；不再创建前置meta任务。\n"
    entry_head += "    # sync_start只约束本次MIX核组准入，不代表各核硬件同周期起跑。\n"
    entry_head += "    with pl.spmd(GROUPS, sync_start=True):\n"
    entry_head += "        # 预定义MIX的权重缓存策略在cube_schedule函数属性声明。\n"
    entry_head += "        (" + ", ".join(writes) + ") = _csa_core_group(\n"
    entry_head += "".join("            " + k + ",\n" for k in [*args, *controls]) + "        )\n"
    entry_head += "    return (" + ", ".join(writes) + ")\n"
    (PKG / "entry.py").write_text(entry_head)
    report = {
        "runtime_tensors": runtime_args,
        "internal_view_tensors": {k: args[k] for k in ("idx_key0", "idx_key64")},
        "dynamic_axes": sorted(dims),
        "controls": controls,
        "phase_count": len(phases) + 1,
        "phases": {n: i for i, n in enumerate(phases, 1)},
        "roles": {r: {"arguments": list(a), "writes": w} for r, (a, w) in roles.items()},
        "sample_shape": None,
        "historical_ir_used": False,
        "device_validated": False,
    }
    (HERE / "source_interface.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "root_tensor_count": len(runtime_args),
                "internal_views": 2,
                "dynamic_axes": len(dims),
                "phases": len(phases) + 1,
            }
        )
    )


if __name__ == "__main__":
    main()
