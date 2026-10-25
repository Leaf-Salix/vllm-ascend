# SPDX-License-Identifier: Apache-2.0
"""B16保持M128/N256列所有权，仅将Q_B改为L1 K512、L0 K128两级流水。"""

import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path

from prepare import replace_once
from prepare_qb_tall import HELPER


def main():
    here = Path(__file__).resolve().parent
    parent = here / "frozen/pypto/deepseek_v4_flash_csa_spmd_tnd"
    target = here / "candidates/qb_l1_k512_v1"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    helper = HELPER.replace("q_b_tall_stage", "q_b_l1_stage").replace("QB_TALL", "QB_L1")
    helper = replace_once(helper, "QB_L1_M = 192", "QB_L1_M = 128")
    helper = replace_once(helper, "QB_L1_N = 128", "QB_L1_N = 256")
    begin, end = helper.index('"""'), helper.index('"""', helper.index('"""') + 3) + 3
    helper = helper[:begin] + '"""保持M128/N256及INT32点积；L1 K512与L0 K128分别双缓冲。"""' + helper[end:]
    ast.parse(helper)
    (target / "qb_l1.py").write_text(helper)
    path = target / "projection_overlap_stages.py"
    before = path.read_text()
    source = replace_once(
        before, "import pypto.language as pl\n", "import pypto.language as pl\n\nfrom .qb_l1 import q_b_l1_stage\n"
    )
    original = (
        "        q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, qb_worker, GROUPS - BRANCH_GROUPS, QB_WIDE_N)\n"
    )
    replacement = """        if not QB_WIDE_N and tokens >= 64 and tokens <= 128:
            q_proj = q_b_l1_stage(wq_b, qr_matmul, q_proj, tokens, qb_worker, GROUPS - BRANCH_GROUPS)
        else:
            q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, qb_worker, GROUPS - BRANCH_GROUPS, QB_WIDE_N)
"""
    source = replace_once(source, original, replacement)
    ast.parse(source)
    path.write_text(source)
    (here / "qb_l1_k512_v1.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                source.splitlines(True),
                fromfile="a/projection_overlap_stages.py",
                tofile="b/projection_overlap_stages.py",
            )
        )
    )
    report = {
        "source": str(target),
        "parent": "frozen_current_internal_workspace",
        "reference": "同仓HCA的显式L1/L0两级流水，保持CSA当前M128/N256",
        "mechanism": "完整K1024原4次K256的GM供数改为2次K512；L0仍8次K128，完整INT32点积",
        "tiles": {"M": 128, "N": 256, "L1_K": 512, "L0_K": 128},
        "preserved": ["N256列块与20核producer", "实际有效行", "原完成通知", "QR/反量化/RMS/RoPE", "INT32点积不溢出"],
        "tradeoff": "不减少权重总字节数；增加每槽L1占用以减少DMA发射和外层循环，须测完整L1",
        "planned_case": "C04",
        "independent_of_pv_candidate": True,
        "status": "CPU_gate_pending",
        "device_execution": False,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.glob("*.py"))},
    }
    (here / "qb_l1_k512_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
