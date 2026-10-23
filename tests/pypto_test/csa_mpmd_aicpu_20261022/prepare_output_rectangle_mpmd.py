# SPDX-License-Identifier: Apache-2.0
"""将已提交SPMD的NZ矩形核内体移入独立MPMD实验，保持本组TaskId调度。"""

import ast
import difflib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REFERENCE = "c086ecd67"
NAME = "aicpu_v32_output_rectangle"


def main():
    subprocess.run(["python3", str(HERE / "prepare.py"), NAME], check=True)
    experiment = HERE / "experiments" / NAME
    operator = experiment / "operator"
    stages = operator / "output_projection_a_stages.py"
    original = stages.read_text()
    reference_path = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/output_projection_a_stages.py"
    source = subprocess.check_output(["git", "-C", str(REPO), "show", f"{REFERENCE}:{reference_path}"], text=True)
    tree = ast.parse(source)
    helpers = []
    for name in ("_rectangle_o_projection_a_tile_nz", "_rectangle_o_projection_a_groups_stage_nz"):
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        helpers.append("@pl.jit.inline(auto_scope=False)\n" + ast.get_source_segment(source, function))
    body = "\n\n\n".join(helpers)
    # SPMD用全局owner；MPMD每组task传入本组worker，不能复制其全局取模起点。
    old = "first = worker_id + (group_first + WORKERS - 1 - worker_id) // WORKERS * WORKERS"
    assert body.count(old) == 1
    body = body.replace(old, "first = group_first + worker_id")
    body = body.replace(
        "保持全局条带 owner 与原 K 链。",
        "本组task局部worker分工，保留原K链；跨组由AICPU调度。",
    )
    target = "@pl.jit.inline(auto_scope=False)\ndef o_projection_a_groups_stage_nz("
    assert original.count(target) == 1
    updated = original.replace(target, body + "\n\n\n" + target.replace("def o_", "def _original_o_"))
    wrapper = '''@pl.jit.inline(auto_scope=False)
def o_projection_a_groups_stage_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    A_COL_TILE: pl.constexpr,
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    if A_COL_TILE == 192:
        o_r_pad = _rectangle_o_projection_a_groups_stage_nz(
            o_packed, wo_a, o_r_pad, tokens, worker_id, WORKERS, A_COL_TILE, 160, group_begin, group_end
        )
    else:
        o_r_pad = _original_o_projection_a_groups_stage_nz(
            o_packed, wo_a, o_r_pad, tokens, worker_id, WORKERS, A_COL_TILE, group_begin, group_end
        )
    return o_r_pad


'''
    anchor = "@pl.jit.inline(auto_scope=False)\ndef o_projection_a_stage_nz("
    assert updated.count(anchor) == 1
    updated = updated.replace(anchor, wrapper + anchor)
    updated = updated.replace(
        "所有容量轴均 pl.dynamic；固定8组、模型维度和128/256列tile不是B/S白名单。",
        "所有容量轴均pl.dynamic；NZ增加M160/N192及静态N64尾块，适用容量128<T<=160。",
    )
    stages.write_text(updated)
    tasks = operator / "output_tasks.py"
    text = tasks.read_text()
    old = "    oa_units = (tokens + 127) // 128 * (1024 // OA_COLS)"
    assert text.count(old) == 1
    text = text.replace(
        old,
        "    if OA_COLS == 192:\n"
        "        oa_units = (tokens + 159) // 160 * 6\n"
        "    else:\n"
        "        oa_units = (tokens + 127) // 128 * (1024 // OA_COLS)",
    )
    tasks.write_text(text)
    policy = operator / "policy.py"
    text = policy.read_text()
    old = "        oa_cols=128 if tokens <= 96 else 256,"
    assert text.count(old) == 1
    text = text.replace(
        old,
        "        # 已提交SPMD的M160/N192+N64；只按已有token容量选核内tile，实际寻址仍在设备。\n"
        "        oa_cols=192 if 128 < tokens <= 160 else 128 if tokens <= 96 else 256,",
    )
    policy.write_text(text)
    changed = ["output_projection_a_stages.py", "output_tasks.py", "policy.py"]
    patches = []
    production = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd"
    for name in changed:
        patches.extend(
            difflib.unified_diff(
                (production / name).read_text().splitlines(True),
                (operator / name).read_text().splitlines(True),
                fromfile=f"a/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd/{name}",
                tofile=f"b/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd/{name}",
            )
        )
    (HERE / "scheduling_v32_output_rectangle.patch").write_text("".join(patches))
    manifest = json.loads((experiment / "manifest.json").read_text())
    manifest.update(
        {
            "candidate": NAME,
            "reference_commit_for_rectangle": REFERENCE,
            "changes": changed,
            "isolated_from_query_chunks": True,
            "owner_adaptation": "Each independent output group uses first=group_first+worker_id.",
            "fixed_inputs_unchanged": True,
            "frozen": False,
        }
    )
    (experiment / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(experiment)


if __name__ == "__main__":
    main()
