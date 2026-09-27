"""Expose the Q_A producer token and order only the two Compressor projections."""

import difflib
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
RELATIVE = "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf"
PACKAGE = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf"


def main():
    originals = {name: subprocess.check_output(["git", "show", f"7dc17735:{RELATIVE}/{name}.py"],
                                              cwd=REPO, text=True)
                 for name in ("qkv_proj_rope", "decode_csa")}
    q = originals["qkv_proj_rope"]
    q = q.replace('qr_fp32: pl.Out[pl.Tensor[[QPROJ_MM_T_DYN, Q_LORA], pl.FP32]]',
                  'qr_fp32: pl.Tensor[[QPROJ_MM_T_DYN, Q_LORA], pl.FP32]')
    q = q.replace('qr_i8_matmul: pl.Out[pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8]]',
                  'qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8]')
    q = q.replace('qr_scale_pad_store: pl.Out[pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32]]',
                  'qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32]')
    q = q.replace('for qbg_idx in pl.spmd(QR_N_BLOCKS * QR_OK, name_hint="qr_proj_matmul", allow_early_resolve=True):',
                  'with pl.spmd(QR_N_BLOCKS * QR_OK, name_hint="qr_proj_matmul", allow_early_resolve=True) as qa_tid:\n'
                  '        qbg_idx = pl.tile.get_block_idx()')
    q = q.replace('with pl.spmd(QR_N_BLOCKS * QR_OK, name_hint="qr_proj_matmul", allow_early_resolve=True):',
                  'with pl.spmd(QR_N_BLOCKS * QR_OK, name_hint="qr_proj_matmul", allow_early_resolve=True) as qa_tid:')
    q = q.replace('\n\n@pl.jit.inline(auto_scope=False)\ndef _q_proj_qa_nz',
                  '\n    return qa_tid\n\n\n@pl.jit.inline(auto_scope=False)\ndef _q_proj_qa_nz')
    q = q.replace('\n\nq_proj_qa =', '\n    return qa_tid\n\n\nq_proj_qa =')
    start = q.index('def q_proj_qr(')
    end = q.index('\n\n@pl.jit.inline(auto_scope=False)\ndef q_proj_qr_normalize', start)
    part = q[start:end]
    part = part.replace('    t_dim = pl.tensor.dim(x, 0)',
                        '    qa_tids = pl.array.create(1, pl.TASK_ID)\n'
                        '    qa_tids[0] = pl.system.task_invalid()\n'
                        '    t_dim = pl.tensor.dim(x, 0)')
    part = part.replace('            q_proj_qa(', '            qa_tids[0] = q_proj_qa(')
    q = q[:start] + part + '\n    return qa_tids[0]\n' + q[end:]
    q = q.replace('    q_proj_qr(x, wq_a, gamma_cq, qr, qr_scale, qr_i8_matmul, qr_scale_pad_store)',
                  '    qa_tid = q_proj_qr(x, wq_a, gamma_cq, qr, qr_scale, qr_i8_matmul, qr_scale_pad_store)')
    q = q.replace('\n\n@pl.jit.inline(auto_scope=False)\ndef kv_proj_rope',
                  '\n    return qa_tid\n\n\n@pl.jit.inline(auto_scope=False)\ndef kv_proj_rope')
    start = q.index('def qkv_proj_rope(')
    part = q[start:].replace('    q_proj_rope(', '    qa_tid = q_proj_rope(').replace('    return q', '    return q, qa_tid')
    q = q[:start] + part
    c = originals["decode_csa"].replace('        qkv_proj_rope(', '        q, qa_tid = qkv_proj_rope(')
    c = c.replace('        cmp_out = pl.create_tensor',
                  '        # Keep Q_A ahead of the two background Compressor projections.\n'
                  '        compressor_dep = pl.system.task_dummy(deps=[late_dep, qa_tid])\n'
                  '        cmp_out = pl.create_tensor')
    c = c.replace('            state_slot_mapping,\n            late_dep,',
                  '            state_slot_mapping,\n            compressor_dep,')
    c = c.replace('            inner_state_slot_mapping,\n            late_dep,',
                  '            inner_state_slot_mapping,\n            compressor_dep,')
    assert c.count('            compressor_dep,') == 2
    candidates = {"qkv_proj_rope": q, "decode_csa": c}
    diffs = []
    for name, candidate in candidates.items():
        (ROOT / f"{name}.py").write_text(candidate)
        diffs.extend(difflib.unified_diff(originals[name].splitlines(True), candidate.splitlines(True),
                                        fromfile=f"a/{RELATIVE}/{name}.py", tofile=f"b/{RELATIVE}/{name}.py"))
    (ROOT / "candidate.patch").write_text("".join(diffs))
    sys.path.insert(0, str(REPO / "tests/pypto_test"))
    from dsv4_csa_env import activate
    activate()
    for name in candidates:
        qualified = f"{PACKAGE}.{name}"
        spec = importlib.util.spec_from_file_location(qualified, ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    sys.argv = ["compile_contiguous.py", str(ROOT / "compile")]
    runpy.run_path(str(REPO / "tests/pypto_test/results/csa_split_optimization_20260927/compile_contiguous.py"),
                   run_name="__main__")


if __name__ == "__main__":
    main()
