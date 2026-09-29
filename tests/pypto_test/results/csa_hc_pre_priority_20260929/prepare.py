"""冻结已保留HC_post的私有对照，只给非关键Sinkhorn增加pre/post任务依赖。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
REPO = ROOT.parents[3]
BASE = WORKSPACE / ".cache/csa-hc-post-resident-55b89ee2-candidate"
PREFIX = WORKSPACE / ".cache/csa-hc-pre-priority-d93bba14"
OLD_PACKAGE = "dsv4_csa_hc_post_resident_55b89ee2"
PACKAGE = "dsv4_csa_hc_pre_priority_d93bba14"


def main():
    if (ROOT / "task.txt").exists():
        raise RuntimeError("禁止修改已入队副本")
    for side in ("baseline", "candidate"):
        dest = Path(f"{PREFIX}-{side}")
        shutil.copytree(BASE, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        packages = dest / "vllm_ascend/ops/pypto"
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)
        # Adopted shared arithmetic; retain the frozen test wrapper, not a new Native config.
        shared = Path("vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/hc_post.py")
        shutil.copyfile(REPO / shared, dest / shared)
    relative = Path("vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/hc_pre.py")
    target = Path(f"{PREFIX}-candidate") / relative
    before = target.read_text()
    old = ('    for ob_worker in pl.spmd(pl.min(token_tiles, PRE_POST_WORKERS), '
           'name_hint="split_pre_post", allow_early_resolve=True):\n')
    new = ('    with pl.spmd(pl.min(token_tiles, PRE_POST_WORKERS), name_hint="split_pre_post",\n'
           '                 allow_early_resolve=True) as pre_post_tid:\n'
           '        ob_worker = pl.tile.get_block_idx()\n')
    anchor = '    for ob in pl.spmd(token_tiles, name_hint="comb_sinkhorn", allow_early_resolve=True):\n'
    replacement = ('    # Prioritize the attention pre-gate before the independent post-only Sinkhorn.\n'
                   '    for ob in pl.spmd(token_tiles, name_hint="comb_sinkhorn",\n'
                   '                      deps=[pre_post_tid], allow_early_resolve=True):\n')
    if before.count(old) != 1 or before.count(anchor) != 1:
        raise ValueError("HC task declaration changed")
    after = before.replace(old, new).replace(anchor, replacement)
    ast.parse(after)
    target.write_text(after)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile=f"a/{relative}", tofile=f"b/{relative}")))
    source = {"baseline": "d93bba14", "base_source": str(BASE), "source_prefix": str(PREFIX),
              "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
              "change": "comb_sinkhorn显式依赖split_pre_post；不改计算、任务数或数据布局",
              "evidence": "55b89ee2七档DFX：两者共用linear_reduce前置，长B24 comb启动比pre/post早5.515us",
              "inplace_pass": True, "scope": "复用现有PTO执行路径；Native不重测"}
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    origin = ROOT.parent / "csa_hc_post_resident_20260929"
    for name in ("compile.py", "run.sh", "run_side.sh"):
        text = (origin / name).read_text().replace(origin.name, ROOT.name)
        text = text.replace("csa-hc-post-resident-55b89ee2", "csa-hc-pre-priority-d93bba14")
        text = text.replace(OLD_PACKAGE, PACKAGE)
        text = text.replace("# Baseline 55b89ee2 already fuses the final Sparse publication.",
                            "# Baseline includes Sparse final publication and adopted HC_post residency.")
        text = text.replace("# Only test HC residual residency here; do not combine the pending first-PV candidate.",
                            "# Only test the explicit pre/post -> Sinkhorn dependency; keep arithmetic fixed.")
        (ROOT / name).write_text(text)


if __name__ == "__main__":
    main()
