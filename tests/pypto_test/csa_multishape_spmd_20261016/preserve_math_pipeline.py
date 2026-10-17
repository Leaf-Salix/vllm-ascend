# SPDX-License-Identifier: Apache-2.0
"""归因用候选：保留原 PyPTO 数学核的局部流水，仍只派发一个融合任务。

仅在当前进程的 C++ 生成边界替换选定数学区。调用的是同次动态 PTO 编译得到的
device 函数，没有增加 Host/AICPU 调度，也不修改依赖仓库。此实验用于确认全函数
内存规划/自动同步造成的退步；不是最终 PyPTO 接口实现。
"""

import ast
import contextlib
import json
import re
from pathlib import Path

HOT_KERNELS = {"hc_widen_rms", "mix_x_rms_norm", "qproj_matmul", "idx_qr_proj_matmul"}


def canonical_name(name):
    name = re.sub(r"_inline\d+", "", name)
    return re.sub(r"__(?:ssa|rv|idx|iter|phi|tile)_v\d+", "", name)


def view_key(ref):
    return (ref["allocation"], tuple(ref["shape"]), tuple(ref["strides"]),
            ref["offset"], ref["dtype"], ref["layout"])


def replace_math(source, case, dynamic_build, selected, *, all_stages=False):
    case = Path(case)
    schedule = json.loads((case.parent.parent / "bindings" / (case.name + "_prepared.json")).read_text())
    layout = json.loads((case / "layout.json").read_text())
    contract = json.loads((case / "entry.bindings.json").read_text())["_entry_contract"]
    jit = ast.parse((case / "entry_jit.py").read_text())
    refs = {row["name"]: row["reference"] for row in
            contract["ordered_inputs"] + contract["derived_native_views"]}
    helpers, audit = {}, []
    spans = list(re.finditer(r"^AICORE void csa_static_(aic|aiv)\(([^\n]+)\) \{", source, re.M))
    if len(spans) != 2:
        raise ValueError("找不到两个融合物理函数")
    for match in reversed(spans):
        role = match.group(1)
        fn = next(node for node in jit.body if isinstance(node, ast.FunctionDef)
                  and node.name == "csa_static_" + role)
        params = [part.strip().split()[-1].lstrip("*") for part in match.group(2).split(",")]
        pointers = {view_key(refs[arg.arg]): params[index] for index, arg in enumerate(fn.args.args)
                    if arg.arg in refs and "allocation" in refs[arg.arg]}
        end = source.find("\n}", match.end())
        body = source[match.end():end]
        blocks = list(re.finditer(r"^  if \(([^\n]+) >= [^\n]+\) \{\n", body, re.M))
        if len(blocks) != len(layout["batches"][role]):
            raise ValueError("物理控制段与布局不匹配")
        for index in reversed(range(len(blocks))):
            batch = layout["batches"][role][index]
            name = batch["kernel"]
            if name not in selected:
                continue
            if any(row["logical_block_num"] != row["logical_block_stride"]
                   for row in batch["members"]):
                raise ValueError("本归因实验暂不替换额外分片循环")
            original = (Path(dynamic_build) / "kernels" / role / (name + ".cpp")).read_text()
            # 保留实际动态 PTO 的完整 math 函数，不从另一规格拼接数学源码。
            definition = re.search(r"^static __aicore__ void " + re.escape(name) +
                                   r"\([^\n]+\) \{.*?\n}", original, re.M | re.S)
            if not definition:
                raise ValueError("原动态 C++ 函数不存在：" + name)
            helper_name = "csa_preserved_" + name
            helpers[name] = definition.group().replace(
                "static __aicore__ void " + name,
                ("static __aicore__ __attribute__((noinline)) void " if all_stages else
                 "static __aicore__ __attribute__((always_inline)) inline void ") + helper_name, 1)
            wrapper = original.split("// --- Kernel entry point ---")[1]
            call = schedule["call_bindings"][batch["task"]]
            values = {}
            for row in call["arguments"]:
                key = canonical_name(row["parameter"])
                if key in values:
                    raise ValueError("参数去掉编译器编号后不唯一：" + key)
                values[key] = row["value"]
            arguments, tensor_refs = {}, {}
            for kind, arg in re.findall(r"// Unpack (tensor|scalar): (\w+)", wrapper):
                value = values[canonical_name(arg)]
                if kind == "tensor":
                    arguments[arg] = pointers[view_key(value)]
                    tensor_refs[arg] = value
                else:
                    if isinstance(value, dict):
                        raise ValueError("Scalar 参数类型不一致")
                    arguments[arg] = str(value).lower() if isinstance(value, bool) else repr(value)
            for arg, tensor, dim in re.findall(
                    r"int64_t (\w+) = static_cast<int64_t>\((\w+)_tensor->shapes\[(\d+)\]\);", wrapper):
                arguments[arg] = str(tensor_refs[tensor]["shape"][int(dim)])
            for arg, tensor, dim in re.findall(
                    r"int64_t (\w+) = static_cast<int64_t>\((\w+)_tensor->strides\[(\d+)\]\);", wrapper):
                arguments[arg] = str(tensor_refs[tensor]["strides"][int(dim)])
            for arg, tensor, dim, divisor in re.findall(
                    r"int64_t (\w+) = \(static_cast<int64_t>\((\w+)_tensor->shapes\[(\d+)\]\) / (\d+)\);", wrapper):
                extent, factor = tensor_refs[tensor]["shape"][int(dim)], int(divisor)
                if extent % factor:
                    raise ValueError("动态维恢复不是精确整除：" + arg)
                arguments[arg] = str(extent // factor)
            arguments["__pypto_spmd_block_idx"] = blocks[index].group(1)
            arguments["__pypto_spmd_block_num"] = str(call["block_num"])
            if role == "aiv":
                # MIX 分配已经检查 AIC/AIV0/AIV1 的物理配对；原函数若读取
                # lane，必须传真实 lane。纯 AIV 不会因为此映射新增输入。
                arguments["__pypto_spmd_subblock_idx"] = params[-1]
            forward = re.search(r"// Forward to ptoas-generated function\s+" +
                                re.escape(name) + r"\(([^;]+)\);", wrapper)
            actuals = [arguments[arg.strip()] for arg in forward.group(1).split(",")]
            invocation = f"    {helper_name}({', '.join(actuals)});\n"
            if all_stages:
                # 原动态物理核包装器每次调用先清除继承的 atomic 配置。
                # 全部原核体汇入一个任务后，仍保留这个计算入口状态约定。
                invocation = "    set_atomic_none();\n" + invocation
            close = body.find("\n  }", blocks[index].end())
            segment = body[blocks[index].end():close]
            sites = list(re.finditer(
                r"^    dcci\(\(__gm__ void\*\)0, cache_line_t::ENTIRE_DATA_CACHE\);\n"
                r"(?:    pipe_barrier\([^\n]+\);\n)*    dsb\(DSB_DDR\);\n", segment, re.M))
            if len(sites) != 2:
                raise ValueError("无法唯一识别两处 payload 缓存维护")
            # 第二个 DCCI 前的 barrier/DSB 一起作为数学尾部替换；新函数自己的
            # ptoas_auto_sync_tail 排空计算，外层写回 DCCI/DSB 与信号发布仍保留。
            after = sites[1].start()
            prefix = segment[:after]
            while prefix.endswith("    pipe_barrier(PIPE_ALL);\n") or prefix.endswith("    dsb(DSB_DDR);\n"):
                prefix = prefix[:prefix.rfind("\n", 0, len(prefix) - 1) + 1]
            after = len(prefix)
            segment = segment[:sites[0].end()] + invocation + segment[after:]
            body = body[:blocks[index].end()] + segment + body[close:]
            audit.append({"role": role, "stage": index, "kernel": name, "arguments": actuals,
                          "original_source": str(Path(dynamic_build) / "kernels" / role / (name + ".cpp"))})
        if all_stages:
            if any(batch["kernel"] not in selected for batch in layout["batches"][role]):
                raise ValueError("只有全部数学阶段都替换后才能清理外层 auto-sync")
            # 外层已经只含控制、数据缓存维护和信号。独立数学核各自管理
            # auto-sync 生命周期，外层原数学的事件初始化/回收必须成对清理。
            flags = list(re.finditer(r"^  (set_flag|wait_flag)\(([^\n]+)\);\n", body, re.M))
            before = sorted(m.group(2) for m in flags if m.group(1) == "set_flag")
            after = sorted(m.group(2) for m in flags if m.group(1) == "wait_flag")
            if before != after:
                raise ValueError("外层原数学 auto-sync 事件不成对")
            for flag in reversed(flags):
                body = body[:flag.start()] + body[flag.end():]
            if re.search(r"\b(?:set_flag|wait_flag)\(", body):
                raise ValueError("外层残留未识别的事件生命周期")
            audit.append({"role": role, "all_math_outlined": True,
                          "removed_outer_auto_sync_pairs": before,
                          "math_and_mixed_events_unchanged": True})
        source = source[:match.end()] + body + source[end:]
    if {item["kernel"] for item in audit if "kernel" in item} != selected:
        raise ValueError("请求替换的数学核未全部覆盖")
    start = source.index("AICORE void csa_static_")
    source = source[:start] + "\n".join(helpers[name] for name in sorted(helpers)) + "\n\n" + source[start:]
    return source, audit


@contextlib.contextmanager
def preserved_pipeline(case, dynamic_build, output, selected=HOT_KERNELS, *, all_stages=False):
    from pypto.backend import pto_backend

    original = pto_backend._compile_pto_module
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if all_stages:
        layout = json.loads((Path(case) / "layout.json").read_text())
        selected = {batch["kernel"] for batches in layout["batches"].values() for batch in batches}

    def compile_preserved(pto_code, unit_name, output_dir, *args, **kwargs):
        source = original(pto_code, unit_name, output_dir, *args, **kwargs)
        if unit_name != "csa_static_group":
            return source
        transformed, audit = replace_math(source, case, dynamic_build, set(selected), all_stages=all_stages)
        (output / "original.cpp").write_text(source)
        (output / "preserved.cpp").write_text(transformed)
        (output / "bindings.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
        return transformed

    pto_backend._compile_pto_module = compile_preserved
    try:
        yield
    finally:
        pto_backend._compile_pto_module = original
