"""只保留A的L1复用，沿已有AutoTileMatmulL0恢复K128双缓冲。"""

import ast
import difflib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "csa_ob_activation_l1_20260929"
WORKSPACE = ROOT.parents[4]
PREFIX = WORKSPACE / ".cache/csa-ob-activation-l1-k128-2ed8ae2e"
PACKAGE = "dsv4_csa_ob_activation_l1_k128_2ed8ae2e"


def main():
    if (ROOT / "task.txt").exists():
        raise RuntimeError("禁止修改已排队源码")
    spec = importlib.util.spec_from_file_location("ob_first_prepare", PREVIOUS / "prepare.py")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    baseline_source = Path(str(previous.PREFIX) + "-baseline")
    for side in ("baseline", "candidate"):
        destination = Path(str(PREFIX) + "-" + side)
        packages = destination / "vllm_ascend/ops/pypto"
        if not destination.exists():
            shutil.copytree(baseline_source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (packages / previous.PACKAGE).rename(packages / PACKAGE)
        elif not (packages / PACKAGE / "decode_o_proj.py").is_file():
            raise RuntimeError(f"已有私有包不完整，停止刷新：{destination}")
    path = Path(str(PREFIX) + "-candidate") / "vllm_ascend/ops/pypto" / PACKAGE / "decode_o_proj.py"
    # A CPU-only retry always starts from the unchanged private baseline.
    before = (Path(str(PREFIX) + "-baseline") / "vllm_ascend/ops/pypto" / PACKAGE / "decode_o_proj.py").read_text()
    after = previous.transform(before)
    old = '''                    b_act_left = pl.tile.extract(
                        activation_l1, 0, k0, [ROW_TILE, B_K_TILE], target_memory=pl.MemorySpace.Left,
                    )
                    b_weight_right = pl.tile.move(b_weight_l1, target_memory=pl.MemorySpace.Right)
                    acc_b = pl.tile.matmul_acc(acc_b, b_act_left, b_weight_right, init_cond=(kb == 0))
'''
    new = '''                    b_act_l1 = pl.tile.slice(activation_l1, [ROW_TILE, B_K_TILE], [0, k0])
                    # Keep both operands in Mat so the existing AutoTileMatmulL0
                    # pass retains the baseline K128 ping-pong L0 lowering.
                    acc_b = pl.tile.matmul_acc(acc_b, b_act_l1, b_weight_l1, init_cond=(kb == 0))
'''
    if after.count(old) != 1:
        raise ValueError("候选变换锚点发生变化")
    after = after.replace(old, new)
    ast.parse(after)
    path.write_text(after)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile="a/decode_o_proj.py", tofile="b/decode_o_proj.py")))
    source = {"baseline": "2ed8ae2e", "base_source": str(baseline_source), "source_prefix": str(PREFIX),
              "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 16]],
              "change": "只改ROW32/96的A加载复用；Mat切片交由原AutoTileMatmulL0恢复K128双缓冲",
              "reason": "第一版连带改变L0流水并使长档O-B核时回退15.190%；二版隔离这一因素",
              "short_case": "改用8K/B16，使长短两档均覆盖受影响的ROW96，避免只测不变路径",
              "scope": "性能版NZ O-B；Native、精度版、ND、ROW128及量化/任务/调度保持"}
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    for name in ("compile.py", "run.sh", "run_side.sh", "collect.py"):
        text = (PREVIOUS / name).read_text().replace(PREVIOUS.name, ROOT.name)
        text = text.replace(str(previous.PREFIX.name), PREFIX.name).replace(previous.PACKAGE, PACKAGE)
        text = text.replace("8192:24", "8192:16")
        text = text.replace("NZ O-B小中档完整激活L1复用，权重保留分段流水", "NZ O-B激活复用并保持原K128双缓冲")
        text = text.replace("长B16使用ROW96的新A复用路径；短B24为ROW128不变路径控制。",
                            "长短B16均覆盖ROW96的A复用，原L0 K128双缓冲保持。")
        text = text.replace("不把短档控制的波动当作新核内优化收益。", "量化与整数归约保持，核内与CSA分别按8:2评估。")
        (ROOT / name).write_text(text)


if __name__ == "__main__":
    main()
