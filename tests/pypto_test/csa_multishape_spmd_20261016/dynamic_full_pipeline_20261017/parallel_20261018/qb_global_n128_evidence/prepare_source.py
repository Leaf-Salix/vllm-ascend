"""从正式 668 冻结 33 源；只改变 NZ 的全局 N 粒度和对应等待描述。"""

import difflib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
MAIN = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1")
BASE = "66882634"
PACKAGE = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
REPLACEMENTS = {
    "query_projection_stages.py": [
        (
            "    NZ 保留 M128/N256/K256，ND 保留 M64/N512/K128；按实际 T 计算\n",
            "    NZ 使用 M128/N128/K256，ND 保留 M64/N512/K128；按实际 T 计算\n",
        ),
        (
            "    每个 head 完整包含两个 NZ N256 列块。RoPE owner=(row//8)%active，\n",
            "    每个 head 完整包含四个 NZ N128 列块。RoPE owner=(row//8)%active，\n",
        ),
        (
            "QPROJ_MM_N_TILE = 256 if QUANT_WEIGHT_NZ else 512\n",
            "# NZ 将全部 32768 列划成 256 个 N128 任务，仍按 n%WORKERS 全局分配；\n"
            "# 不在原 N256 owner 内拆两列。M128、四段 K256、动态尾行与 ND 路径不变。\n"
            "# Right 单块减至 32KiB 只说明容量允许双槽，是否生成双槽须检查实际产物。\n"
            "QPROJ_MM_N_TILE = 128 if QUANT_WEIGHT_NZ else 512\n",
        ),
    ],
    "vector_schedule.py": [
        (
            "from .query_projection_stages import q_dequant_rope_stage, q_rope_prepare_stage\n",
            "from .query_projection_stages import QPROJ_N_BLOCKS, q_dequant_rope_stage, q_rope_prepare_stage\n",
        ),
        (
            "    # 每个head占两个NZ N256列块。首版等全部128列块的24个真实owner。\n"
            "    signal = wait_row_owners(signal, PHASE_QB, CORES, generation, 0, 128, 1, GROUPS, 0, 1)\n",
            "    # 一个 head 包含四个 NZ N128 列块；ND 的 N512 列块划分保持不变。\n"
            "    # QB 的列块 owner=n%GROUPS，须获取全部列块对应的 24 个实际 AIC；\n"
            "    # 用派生任务数同步描述 NZ/ND，仍由 wait_row_owners 去重并统一维护缓存。\n"
            "    signal = wait_row_owners(signal, PHASE_QB, CORES, generation, 0, QPROJ_N_BLOCKS, 1, GROUPS, 0, 1)\n",
        ),
    ],
}


def main():
    reference, candidate = HERE / "source_reference", HERE / "source_candidate"
    if reference.exists() or candidate.exists():
        raise FileExistsError("冻结源码目录已存在，拒绝覆盖")
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASE, PACKAGE], cwd=MAIN, text=True
    ).splitlines()
    paths = [path for path in paths if path.endswith(".py")]
    assert len(paths) == 33
    assert all(Path(path).parent.as_posix() == PACKAGE for path in paths)
    contents = {
        Path(path).name: subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=MAIN)
        for path in paths
    }
    # 先直接比较正式工作区与指定提交，防止悄悄混入并行实验；不用摘要校验。
    assert all(contents[Path(path).name] == (MAIN / path).read_bytes() for path in paths)
    reference.mkdir()
    candidate.mkdir()
    for name, data in contents.items():
        (reference / name).write_bytes(data)
        (candidate / name).write_bytes(data)
    for name, replacements in REPLACEMENTS.items():
        path = candidate / name
        text = path.read_text()
        for old, new in replacements:
            assert text.count(old) == 1, (name, old)
            text = text.replace(old, new)
        path.write_text(text)
    diff = []
    for name in sorted(contents):
        diff.extend(difflib.unified_diff(
            (reference / name).read_text().splitlines(keepends=True),
            (candidate / name).read_text().splitlines(keepends=True),
            fromfile=f"source_reference/{name}", tofile=f"source_candidate/{name}",
        ))
    (HERE / "candidate.diff").write_text("".join(diff))
    record = {
        "base_revision": BASE,
        "formal_worktree_equal_to_reference_by_direct_bytes": True,
        "file_count_each": len(contents),
        "source_reference": str(reference),
        "source_candidate": str(candidate),
        "changed_files": sorted(REPLACEMENTS),
        "scope": "NZ 全局 N128；ND512/动态 M128/四 K256/全部数学/GM/阶段发布不变",
        "compiled": False,
        "device_execution": False,
        "files": [
            {"name": name, "reference_bytes": (reference / name).stat().st_size,
             "candidate_bytes": (candidate / name).stat().st_size}
            for name in sorted(contents)
        ],
    }
    (HERE / "SOURCE_PREPARATION.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(candidate)


if __name__ == "__main__":
    main()
