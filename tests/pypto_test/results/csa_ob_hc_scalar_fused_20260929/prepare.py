"""保留原HC每worker四token、整D单行标量算术的收尾融合。"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIRST = ROOT.parent / "csa_ob_hc_fused_20260929"
PACKAGE = "dsv4_csa_ob_hc_scalar_fused_9a868d26"
PREFIX = ROOT.parents[4] / ".cache/csa-ob-hc-scalar-fused-9a868d26"


def main():
    spec = importlib.util.spec_from_file_location("fusion_scalar_prepare", FIRST / "prepare.py")
    first = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(first)
    first.ROOT = ROOT
    first.PREFIX = PREFIX
    first.PACKAGE = PACKAGE
    helper = first.HC_HELPER
    start = helper.index("    if ROWS == 1:\n")
    end = helper.index("    else:\n", start)
    scalar = helper[start + len("    if ROWS == 1:\n"):end]
    scalar = "\n".join(line[4:] if line.startswith("    ") else line for line in scalar.split("\n"))
    first.HC_HELPER = helper[:start] + scalar + "    return y_flat\n\n\n"
    original = first.change_o_proj

    def scalar_o_proj(before):
        after = original(before)
        changes = {
            "PROJ_B_ACT_TASK_T_TILE = 16": "PROJ_B_ACT_TASK_T_TILE = 4",
            "PROJ_B_ACT_T_TILE = 8": "PROJ_B_ACT_T_TILE = 1",
            "PROJ_B_ACT_N_TILE = 512": "PROJ_B_ACT_N_TILE = D",
            "                g_scale_row = act_scale_dq[act_g : act_g + 1, b_tb : b_tb + PROJ_B_ACT_T_TILE]\n"
            "                g_scale = pl.reshape(g_scale_row, [PROJ_B_ACT_T_TILE, 1])":
            "                g_scale_scalar = pl.read(act_scale_dq, [act_g, b_tb])",
            "pl.row_expand_mul(p_g_f32, g_scale)": "pl.mul(p_g_f32, g_scale_scalar)",
        }
        for old, new in changes.items():
            if after.count(old) != 1:
                raise ValueError(f"单行融合锚点发生变化：{old}")
            after = after.replace(old, new)
        return after

    first.change_o_proj = scalar_o_proj
    first.main()
    for name in ("compile_shared_hc.py", "collect.py"):
        text = (FIRST / name).read_text().replace(FIRST.name, ROOT.name)
        text = text.replace("csa-ob-hc-fused-9a868d26", PREFIX.name)
        text = text.replace("dsv4_csa_ob_hc_fused_9a868d26", PACKAGE)
        text = text.replace('((batch * 6 + 15) // 16) * 8', '(batch * 6 + 3) // 4')
        (ROOT / name).write_text(text)
    run = ROOT / "run.sh"
    run.write_text(run.read_text().replace('test -f "$root/compile_candidate.json"',
                                          'test -f "$root/compile_candidate.json"\n'
                                          'test -f "$root/compile_shared_hc.json"\n'
                                          'test -f "$root/static_evidence.json"'))
    source_path = ROOT / "source.json"
    source = json.loads(source_path.read_text())
    source.update(change="T4/D4096任务、内部T1，反量化scale和HC门控均用原单行标量乘；保留BF16边界",
                  expected_workers={"128K/B16": 24, "8K/B24": 36},
                  scope="全新私有CPU候选；保留原HC单行整D算术/分工，生产和首版冻结包不变")
    source_path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
