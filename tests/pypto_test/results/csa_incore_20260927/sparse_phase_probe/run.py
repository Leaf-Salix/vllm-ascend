"""Probe generated Sparse Attention C++ with fixed Native inputs, off the production path."""

import argparse
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
from dsv4_csa_validation import compare_tensor


def matching_brace(text, start):
    depth = 0
    for pos in range(start, len(text)):
        depth += (text[pos] == "{") - (text[pos] == "}")
        if depth == 0:
            return pos
    raise ValueError("Unclosed generated function/loop")


def instrument(text, name):
    start = text.index(f"static __aicore__ void {name}(")
    brace = text.index("{", start)
    end = matching_brace(text, brace)
    signature, body = text[start:brace], text[brace + 1:end]
    params = re.findall(r"\b(v\d+)\b", signature)
    trace = params[0]
    is_aic = name.endswith("aic")
    block = params[-2] if is_aic else params[-3]
    slot = block if is_aic else f"(24 + 2 * {block} + {params[-1]})"
    # Both cross-core waits are stable semantic anchors, independent of SSA ids.
    waits = list(re.finditer(r"wait_flag_dev\(v\d+\);", body))
    signals = list(re.finditer(r"ffts_cross_core_sync\(PIPE_\w+, v\d+\);", body))
    if len(waits) != 2 or len(signals) != 2:
        raise ValueError(f"Unexpected sync topology in {name}")
    edits = []
    for idx, match in enumerate(waits):
        before = "uint64_t phase_wait_start = get_sys_cnt();\n"
        if not is_aic and idx == 0:
            before = "if (phase_tick < 5) phase_metric[1] += get_sys_cnt() - phase_tick_start;\n" + before
        metric = (1 + idx) if is_aic else (2 if idx == 0 else 4)
        after = f"\nphase_metric[{metric}] += get_sys_cnt() - phase_wait_start;\n"
        after += f"phase_metric[{8 + idx}] += 1;\nphase_stage[{idx}] = get_sys_cnt();"
        if not is_aic and idx == 1:
            after += "\nphase_merge_active = true;"
        edits.extend([(match.start(), before), (match.end(), after)])
    for idx, match in enumerate(signals):
        if is_aic:
            extra = f"\nphase_metric[{3 + idx}] += get_sys_cnt() - phase_stage[{idx}];"
        elif idx == 0:
            extra = "\nphase_metric[3] += get_sys_cnt() - phase_stage[0];"
        else:
            extra = "\nif (phase_tick == 0) phase_metric[1] += get_sys_cnt() - phase_tick_start;"
        edits.append((match.end(), extra))
    if not is_aic:
        constants = dict(re.findall(r"const int64_t (v\d+) = (\d+);", body))
        loops = [m for m in re.finditer(r"for \(int64_t (v\d+) = v\d+; \1 < (v\d+); \1 \+= v\d+\) \{", body)
                 if constants.get(m[2]) == "8"]
        if len(loops) != 1:
            raise ValueError("Expected one eight-tick AIV pipeline")
        loop = loops[0]
        loop_end = matching_brace(body, loop.end() - 1)
        edits.append((loop.end(), f"\nphase_tick = {loop[1]};\nphase_tick_start = get_sys_cnt();\nphase_merge_active = false;"))
        edits.append((loop_end, "\nif (phase_merge_active) phase_metric[5] += get_sys_cnt() - phase_stage[1];\n"))
    if_pos = body.index("\n", body.index("#if defined("))
    edits.append((if_pos, "\nuint64_t phase_start = get_sys_cnt();\nuint64_t phase_metric[16] = {};\n"
                  "uint64_t phase_stage[2] = {};\nuint64_t phase_tick_start = 0;\n"
                  "int64_t phase_tick = 0;\nbool phase_merge_active = false;\n"))
    tail = f"""
  uint64_t phase_end = get_sys_cnt();
  phase_metric[0] = phase_end - phase_start;
  phase_metric[10] = phase_start;
  phase_metric[11] = phase_end;
  __gm__ uint64_t* phase_row = reinterpret_cast<__gm__ uint64_t*>({trace}) + ({slot}) * 16;
  for (int phase_i = 0; phase_i < 16; ++phase_i) phase_row[phase_i] = phase_metric[phase_i];
  dcci(reinterpret_cast<__gm__ void*>(phase_row), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
  dcci(reinterpret_cast<__gm__ void*>(phase_row + 8), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
"""
    edits.append((body.rindex("#endif"), tail))
    for pos, value in sorted(edits, reverse=True):
        body = body[:pos] + value + body[pos:]
    return text[:brace + 1] + body + text[end:]


def prepare(args, tensors, cos, sin, output, trace):
    import pypto.language as pl
    from pypto.runtime import RunConfig

    source = "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_sparse_attn_csa.py"
    code = subprocess.run(["git", "show", f"21d99f8a:{source}"], cwd=REPO,
                          check=True, text=True, capture_output=True).stdout
    # Reuse the existing FFTS workspace binding for a larger, caller-owned probe
    # buffer. Original code merely passes this pointer to set_ffts; no extra
    # stores or synchronization are inserted by the DSL.
    code = code.replace("    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],",
                        "    phase_trace: pl.Tensor[[1152], pl.INT64],\n    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],")
    code = code.replace("ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)", "ffts_workspace = phase_trace")
    code = code.replace(") = sparse_attn_csa(\n        q,", ") = sparse_attn_csa(\n        phase_trace,\n        q,")
    probe = ROOT / "instrumented_sparse.py"
    probe.write_text(code)
    module_name = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.sparse_phase_probe"
    spec = importlib.util.spec_from_file_location(module_name, probe)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    sparse = mod.sparse_attn_csa_tp1
    batch, tokens, ori_pages, cmp_pages, ori_cols, cmp_cols = [pl.dynamic(n) for n in (
        "B", "T", "ORI_PAGES", "CMP_PAGES", "ORI_COLS", "CMP_COLS")]

    def root(
        q: pl.Tensor[[tokens, 64, 512], pl.BF16],
        ori_kv: pl.Tensor[[ori_pages, 32, 1, 512], pl.BF16],
        ori_table: pl.Tensor[[batch, ori_cols], pl.INT32],
        cmp_kv: pl.Tensor[[cmp_pages, 32, 1, 512], pl.BF16],
        cmp_table: pl.Tensor[[batch, cmp_cols], pl.INT32],
        topk: pl.Tensor[[tokens, 512], pl.INT32],
        positions: pl.Tensor[[tokens, 1], pl.INT64],
        seq_lens: pl.Tensor[[batch], pl.INT32],
        sink: pl.Tensor[[64], pl.FP32],
        cosine: pl.Tensor[[tokens, 64], pl.FP32],
        sine: pl.Tensor[[tokens, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[3072, 4096], pl.BF16]],
        phase_trace: pl.InOut[pl.Tensor[[1152], pl.INT64]],
    ):
        out, _ = sparse(phase_trace, q, ori_kv, ori_table, cmp_kv, cmp_table, topk,
                        positions, seq_lens, sink, cosine, sine, out)
        return out, phase_trace

    compiled = pl.jit(auto_scope=False)(root).compile(
        *tensors, cos, sin, output, trace,
        config=RunConfig(platform="a2a3", save_kernels=True, save_kernels_dir=str(ROOT / "build")),
    )
    orchestration = (ROOT / "build/orchestration/root.cpp").read_text()
    if not re.search(r"TaskArgs (params_\w+);\s+\1.add_input\(ext_phase_trace\);", orchestration):
        raise ValueError("Probe must be the first qk_pv tensor argument")
    for path in (ROOT / "build/kernels").glob("*/qk_pv_*.cpp"):
        original = path.read_text()
        if not re.search(r"// Forward to ptoas-generated function\s+qk_pv_ai[cv]\(ffts_workspace", original):
            raise ValueError("Probe tensor not present in generated ABI")
        changed = instrument(instrument(original, "qk_pv_aic"), "qk_pv_aiv")
        path.write_text(changed)
    compiled.load()  # CPU compilation only; validate the edited C++ before queuing a device.
    print("PHASE_PROBE_COMPILE_OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    activate()
    import torch
    from pypto.ir import CompiledProgram
    from pypto.runtime import RunConfig

    payload = torch.load(ROOT.parent / "sparse_pmu/native/native_sparse.pt", map_location="cpu", weights_only=True)
    names = ("q", "ori_kv", "ori_block_table", "cmp_kv", "cmp_block_table",
             "cmp_sparse_indices", "position_ids", "seqused_kv", "sinks")
    tensors = [payload[name].contiguous() for name in names]
    tensors[5] = tensors[5].view(-1, 512)
    tensors[6] = tensors[6].view(-1, 1)
    cos = torch.ones((240, 64), dtype=torch.float32)
    sin = torch.zeros_like(cos)
    output = torch.full((3072, 4096), float("nan"), dtype=torch.bfloat16)
    trace = torch.zeros(1152, dtype=torch.int64)
    if args.prepare:
        prepare(args, tensors, cos, sin, output, trace)
        return
    compiled = CompiledProgram.from_dir(ROOT / "build", platform="a2a3")
    compiled(*tensors, cos, sin, output, trace, config=RunConfig(platform="a2a3", device_id=0))
    actual = output.view(8, 384, 4096)[:, :240].transpose(0, 1).contiguous().view(240, 64, 512)
    baseline = torch.load(ROOT.parent / "sparse_pmu/standalone_pipe/output.pt", map_location="cpu", weights_only=True)
    check = compare_tensor(actual, baseline, 0, 0)
    records = trace.view(72, 16).tolist()
    if check["status"] != "PASS" or any(row[0] <= 0 or row[8:10] != [50, 50] for row in records):
        raise RuntimeError(f"Probe changed output or has incomplete event coverage: {check}")
    result = {"operator_revision": "21d99f8a", "scope": "instrumented standalone program; scalar issue/drain boundaries",
              "unit": "get_sys_cnt ticks; no cross-core subtraction", "output_vs_uninstrumented": check,
              "records": records}
    (ROOT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(check)
    for name, rows in (("aic", records[:24]), ("aiv", records[24:])):
        means = [sum(row[i] for row in rows) / len(rows) for i in range(10)]
        print(name, means, "percent", [100 * value / means[0] for value in means[1:6]])


if __name__ == "__main__":
    main()
