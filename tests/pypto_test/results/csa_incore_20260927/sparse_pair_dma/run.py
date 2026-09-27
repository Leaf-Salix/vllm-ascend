"""Prove Native-style paired DMA in generated code before extending the DSL API."""

import argparse
import difflib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(REPO / "tests/pypto_test"))
from dsv4_csa_env import activate
from dsv4_csa_sparse_diagnostic import build_kernel
from dsv4_csa_validation import compare_tensor


def closing_brace(text, start):
    depth = 0
    for pos in range(start, len(text)):
        depth += (text[pos] == "{") - (text[pos] == "}")
        if depth == 0:
            return pos
    raise ValueError("Unclosed generated loop")


def patch_dma(text, chunked=False):
    func = text.index("static __aicore__ void qk_pv_aiv(")
    start = text.index("{", func)
    end = closing_brace(text, start)
    body = text[start:end]
    loops = []
    for match in re.finditer(r"for \(int64_t (v\d+) = v\d+; \1 < v\d+; \1 \+= v\d+\) \{", body):
        loop_end = closing_brace(body, match.end() - 1)
        section = body[match.end():loop_end]
        if "qk_ridx_" in section and "TLOAD" in section and not re.search(r"\bfor \(", section):
            loops.append((match, loop_end, section))
    if len(loops) != 1:
        raise ValueError(f"Expected one compressed gather loop, got {len(loops)}")
    match, loop_end, section = loops[0]
    row = match[1]
    assignments = re.findall(r"int64_t (v\d+) = ([^;]+);", section)
    col, col_expr = assignments[0]
    loads = re.findall(r"int32_t (v\d+) = \((v\d+)\)\[([^;]+)\];", section)
    if len(loads) != 2:
        raise ValueError("Expected one index and one page-table read")
    idx_var, indices, idx_expr = loads[0]
    page_var, table, table_expr = loads[1]
    idx_wide = next(v for v, expr in assignments if expr.strip() == f"(int64_t) {idx_var}")
    phys_var, phys_expr = next((v, expr) for v, expr in assignments if page_var in expr)
    ub_expr = re.search(r"uint64_t v\d+ = ([^;]+);\s*TASSIGN", section)[1]
    cache_view = re.search(r"PTOAS__GLOBAL_TENSOR_DATA\((v\d+)\)", section)[1]
    col_expr = re.sub(rf"\b{row}\b", "pair_row", col_expr)
    idx_expr = re.sub(rf"\b{col}\b", "pair_col", idx_expr)
    table_expr = re.sub(rf"\b{idx_wide}\b", "pair_idx", table_expr)
    phys_expr = re.sub(rf"\b{idx_wide}\b", "pair_idx", phys_expr)
    phys_expr = re.sub(rf"\b{page_var}\b", "pair_page", phys_expr)
    ub_expr = re.sub(rf"\b{row}\b", "pair_row", ub_expr)
    chunk_store = ""
    old_store = None
    if chunked:
        if "PIPE_MTE2, PIPE_MTE3, EVENT_ID7" in body:
            raise ValueError("Diagnostic event id already occupied")
        after_loop = body[loop_end + 1:]
        gm_row = re.search(r"int64_t v\d+ = ([^;]+);", after_loop)[1]
        gm_view = re.search(r"PTOAS__GLOBAL_TENSOR_DATA\((v\d+)\)", after_loop)[1]
        old_store = re.search(r"TSTORE\(v\d+, v\d+\);", after_loop)[0]
        tick = re.findall(r"\bv\d+\b", col_expr)[0]
        chunk_store = f"""
              if ((pair_row & 15) == 14) {{
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
                copy_ubuf_to_gm_align_b16(
                    PTOAS__GLOBAL_TENSOR_DATA({gm_view}) + (({gm_row}) + pair_row - 14) * 512,
                    pair_dst - 14 * 512, 0, 1, 16384, 0, 0, 0, 0);
              }}
"""
    replacement = f"""
            // Diagnostic only: two valid rows may be permuted by physical address.
            // Existing V->MTE2 and MTE2->MTE3 fences around this loop are retained.
            for (int64_t pair_row = 0; pair_row < 64; pair_row += 2) {{
              int64_t pair_col = {col_expr};
              int64_t pair_idx0 = {indices}[{idx_expr}];
              int64_t pair_idx1 = {indices}[({idx_expr}) + 1];
              int64_t pair_src0 = -1, pair_src1 = -1;
              if (pair_idx0 >= 0) {{
                int64_t pair_idx = pair_idx0;
                int32_t pair_page = {table}[{table_expr}];
                pair_src0 = {phys_expr};
                if (pair_src0 < 0) pair_src0 = 0; // baseline gather_row clamps
              }}
              if (pair_idx1 >= 0) {{
                int64_t pair_idx = pair_idx1;
                int32_t pair_page = {table}[{table_expr}];
                pair_src1 = {phys_expr};
                if (pair_src1 < 0) pair_src1 = 0;
              }}
              __ubuf__ bfloat16_t* pair_dst = reinterpret_cast<__ubuf__ bfloat16_t*>({ub_expr});
              __gm__ bfloat16_t* pair_pool = PTOAS__GLOBAL_TENSOR_DATA({cache_view});
              int64_t pair_low = pair_src0 < pair_src1 ? pair_src0 : pair_src1;
              int64_t pair_high = pair_src0 > pair_src1 ? pair_src0 : pair_src1;
              int64_t pair_gap = (pair_high - pair_low - 1) * 1024;
              if (pair_src0 >= 0 && pair_src1 >= 0 && pair_gap >= 0 && pair_gap < 2147483647) {{
                copy_gm_to_ubuf_align_b16(pair_dst, pair_pool + pair_low * 512,
                                        0, 2, 1024, 0, 0, static_cast<uint32_t>(pair_gap), 0);
              }} else {{
                if (pair_src0 >= 0)
                  copy_gm_to_ubuf_align_b16(pair_dst, pair_pool + pair_src0 * 512, 0, 1, 1024, 0, 0, 0, 0);
                if (pair_src1 >= 0)
                  copy_gm_to_ubuf_align_b16(pair_dst + 512, pair_pool + pair_src1 * 512, 0, 1, 1024, 0, 0, 0, 0);
              }}
              {chunk_store}
            }}"""
    result = text[:start + match.start()] + replacement + text[start + loop_end + 1:]
    if chunked:
        # SWA retains its original store; compressed rows were written in four
        # disjoint 16-row pieces, with the original final MTE3 fences retained.
        if result.count(old_store) != 1:
            raise ValueError("Ambiguous full KV store")
        result = result.replace(old_store, f"if ({tick} == 0) {{ {old_store} }}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--chunked", action="store_true", help="Publish each 16-row DMA group")
    args = parser.parse_args()
    output_root = ROOT / "chunked" if args.chunked else ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    activate()
    import torch
    from pypto.ir import CompiledProgram
    from pypto.runtime import RunConfig

    payload = torch.load(ROOT.parent / "sparse_pmu/native/native_sparse.pt", map_location="cpu", weights_only=True)
    names = ("q", "ori_kv", "ori_block_table", "cmp_kv", "cmp_block_table", "cmp_sparse_indices",
             "position_ids", "seqused_kv", "sinks")
    tensors = [payload[name].contiguous() for name in names]
    tensors[5] = tensors[5].view(-1, 512)
    tensors[6] = tensors[6].view(-1, 1)
    tokens = tensors[0].shape[0]
    cos = torch.ones((tokens, 64), dtype=torch.float32)
    sin = torch.zeros_like(cos)
    output = torch.full((3072, 4096), float("nan"), dtype=torch.bfloat16)
    config = RunConfig(platform="a2a3", device_id=0, enable_pmu=2,
                       save_kernels=True, save_kernels_dir=str(output_root / "build"))
    if args.prepare:
        module_name = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_sparse_attn_csa"
        source = "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_sparse_attn_csa.py"
        code = subprocess.run(["git", "show", f"cd910e2c:{source}"], cwd=REPO,
                              check=True, text=True, capture_output=True).stdout
        pinned = ROOT / "baseline_sparse.py"
        pinned.write_text(code)
        spec = importlib.util.spec_from_file_location(module_name, pinned)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        kernel, _ = build_kernel("performance")
        compiled = kernel.compile(*tensors, cos, sin, output, config=config)
        patches = []
        paths = list((output_root / "build/kernels").glob("*/qk_pv_*.cpp"))
        if len(paths) != 2:
            raise ValueError(f"Unexpected qk_pv kernels: {paths}")
        for path in paths:
            before = path.read_text()
            after = patch_dma(before, chunked=args.chunked)
            path.write_text(after)
            patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                fromfile=str(path.relative_to(ROOT)),
                                                tofile=str(path.relative_to(ROOT))))
        (output_root / "generated.patch").write_text("".join(patches))
        compiled.load()
        print("PAIR_DMA_CPU_COMPILE_OK")
        return
    compiled = CompiledProgram.from_dir(output_root / "build", platform="a2a3")
    compiled(*tensors, cos, sin, output, config=config)
    actual = output.view(8, 384, 4096)[:, :tokens].transpose(0, 1).contiguous().view(tokens, 64, 512)
    baseline = torch.load(ROOT.parent / "sparse_kv_early/gated_standalone/output.pt",
                          map_location="cpu", weights_only=True)
    checks = {"vs_native": compare_tensor(actual, payload["expected"].view_as(actual), 0, 0),
              "vs_current_pto": compare_tensor(actual, baseline, 0, 0)}
    if any(value["nonfinite"] for value in checks.values()):
        raise RuntimeError(f"Nonfinite output: {checks}")
    (output_root / "report.json").write_text(json.dumps({"base_revision": "cd910e2c", "scope": "generated-code standalone probe",
                                                "comparison": checks}, indent=2) + "\n")
    torch.save(actual, output_root / "output.pt")
    print(checks)


if __name__ == "__main__":
    main()
