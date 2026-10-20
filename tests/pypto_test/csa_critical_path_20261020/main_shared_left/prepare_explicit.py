# SPDX-License-Identifier: Apache-2.0
"""保留原K512双组Mat预取和两个Right槽，显式复用每组的两份K256 Left。"""

import ast
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def step(index):
    suffix = str(index)
    init = "kb == 0" if index == 0 else "False"
    return f'''
        # 第{index + 1}个K512：两份32KiB Left只恢复一次，供value/gate共用。
        # 每条输出仍按K0、K256顺序累加；两个输出间交错不改变各自数学链。
        left{suffix}_0 = pl.tile.extract(x_l1_{suffix}, 0, 0, [CMP_M_TILE, SHARED_L0_K], target_memory=pl.MemorySpace.Left)
        left{suffix}_1 = pl.tile.extract(x_l1_{suffix}, 0, SHARED_L0_K, [CMP_M_TILE, SHARED_L0_K], target_memory=pl.MemorySpace.Left)
        kv_trans{suffix} = pl.tile.transpose_view(kv_l1_{suffix})
        gate_trans{suffix} = pl.tile.transpose_view(gate_l1_{suffix})
        kv_right{suffix}_0 = pl.tile.extract(kv_trans{suffix}, 0, 0, [SHARED_L0_K, CMP_N_TILE], target_memory=pl.MemorySpace.Right)
        gate_right{suffix}_0 = pl.tile.extract(gate_trans{suffix}, 0, 0, [SHARED_L0_K, CMP_N_TILE], target_memory=pl.MemorySpace.Right)
        kv_acc = pl.tile.matmul_acc(kv_acc, left{suffix}_0, kv_right{suffix}_0, init_cond={init})
        # 第一个value已发出后才恢复第二份value Right，避免三份Right同时活跃。
        # 原value槽复用必须等待其M读取结束；gate槽独立，编译器保留必要事件。
        kv_right{suffix}_1 = pl.tile.extract(kv_trans{suffix}, SHARED_L0_K, 0, [SHARED_L0_K, CMP_N_TILE], target_memory=pl.MemorySpace.Right)
        score_acc = pl.tile.matmul_acc(score_acc, left{suffix}_0, gate_right{suffix}_0, init_cond={init})
        gate_right{suffix}_1 = pl.tile.extract(gate_trans{suffix}, SHARED_L0_K, 0, [SHARED_L0_K, CMP_N_TILE], target_memory=pl.MemorySpace.Right)
        kv_acc = pl.tile.matmul_acc(kv_acc, left{suffix}_1, kv_right{suffix}_1)
        score_acc = pl.tile.matmul_acc(score_acc, left{suffix}_1, gate_right{suffix}_1)
'''


def main():
    source = HERE / "source_candidate_v2"
    source.mkdir(exist_ok=False)
    for path in (HERE.parent / "source_after_score_split").glob("*.py"):
        shutil.copyfile(path, source / path.name)
    path = source / "compressor_projection_stages.py"
    text = path.read_text()
    fn, = [node for node in ast.parse(text).body
           if isinstance(node, ast.FunctionDef) and node.name == "compressor_project_stage"]
    lines = text.splitlines(keepends=True)
    start = fn.body[0].lineno - 1
    prefix = '''    """主投影显式K512双组预取；共享Left并保持每条K256链和动态compact行距。"""
    bs = pl.tensor.dim(x, 0)
    t_matmul = ((bs + CMP_M_TILE - 1) // CMP_M_TILE) * CMP_M_TILE
    for idx in pl.range(worker_id, t_matmul * CMP_OUT_DIM // (CMP_M_TILE * CMP_N_TILE), WORKERS):
        global_row0 = (idx // (CMP_OUT_DIM // CMP_N_TILE)) * CMP_M_TILE
        o0 = (idx % (CMP_OUT_DIM // CMP_N_TILE)) * CMP_N_TILE
        x_rows = pl.min(CMP_M_TILE, bs - global_row0)
        # 显式描述后端原来生成的compact Acc合同，不以全64行种子推断尾行距。
        # create只分配句柄，首个K256的init_cond=True建立数值；不会读取未初始化Acc。
        kv_seed = pl.tile.create([CMP_M_TILE, CMP_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, compact=True)
        score_seed = pl.tile.create([CMP_M_TILE, CMP_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, compact=True)
        kv_acc = pl.set_validshape(kv_seed, x_rows, CMP_N_TILE)
        score_acc = pl.set_validshape(score_seed, x_rows, CMP_N_TILE)
        # 普通range保留原一次预取两个K512的窗口。不能套pipeline把后续Right
        # 再提前到当前MMA之前，否则Right可能超过64KiB。固定2块只属数学K分块，
        # 不是固定B/T/历史；GM读与最终写的有效行始终是本轮x_rows。
        for kb in pl.range(0, D // K_TILE, 2):
            k0 = kb * K_TILE
            k1 = (kb + 1) * K_TILE
            x_l1_0 = pl.load(x, [global_row0, k0], [CMP_M_TILE, K_TILE], valid_shape=[x_rows, K_TILE], target_memory=pl.MemorySpace.Mat)
            kv_l1_0 = pl.load(wkv, [o0, k0], [CMP_N_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)
            gate_l1_0 = pl.load(wgate, [o0, k0], [CMP_N_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)
            x_l1_1 = pl.load(x, [global_row0, k1], [CMP_M_TILE, K_TILE], valid_shape=[x_rows, K_TILE], target_memory=pl.MemorySpace.Mat)
            kv_l1_1 = pl.load(wkv, [o0, k1], [CMP_N_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)
            gate_l1_1 = pl.load(wgate, [o0, k1], [CMP_N_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)
'''
    # 两份展开只展开模型固有K512组，避免不同Mat对象跨phi形成非法Mat→Mat。
    for index in range(2):
        prefix += "\n".join("    " + line if line else "" for line in step(index).splitlines()) + "\n"
    prefix += '''        # Tile自身保留动态valid和compact信息，仅写真实行；容量仍是原pad64。
        kv_proj_pad = pl.tile.store(kv_acc, [global_row0, o0], kv_proj_pad)
        score_proj_pad = pl.tile.store(score_acc, [global_row0, o0], score_proj_pad)
    return kv_proj_pad, score_proj_pad
'''
    text = "".join(lines[:start]) + prefix + "".join(lines[fn.end_lineno:])
    text = text.replace("CMP_N_TILE = 64\n", "CMP_N_TILE = 64\nSHARED_L0_K = 256\n", 1)
    text = text.replace(
        '"""动态 CSA 的两路压缩投影：保留原数学、显式 AIC 分工的核内阶段。',
        '"""动态 CSA 的两路压缩投影：主投影共享Left，保留双组Mat和双Right。\n\n'
        '本候选总览\n'
        '    主M64/N64投影仍按两组K512预取。每组两个K256输入只搬入Left\n'
        '    一次，同时服务value/gate两条独立链；Right按KV0/Gate0/KV1/Gate1\n'
        '    交错恢复，保持最多两个32KiB槽。每条输出的原K顺序不变。\n'
        '    显式compact Acc及x_rows保留动态尾行。Indexer不改。完整生成\n'
        '    地址、事件与设备精度/完整收益尚待验证，正式ops未修改。', 1,
    )
    text = text.replace("stage=2 只保留原搬运流水。", "主投影显式保持两K512预取；Indexer仍用stage=2。")
    text = text.replace("主投影 M64 跨四个 16 行分形，动态尾行需要首个 pl.matmul 建立 compact\n    accumulator 的有效行距；不能改成 create_tensor+matmul_acc(init_cond)。",
                        "主投影 M64 跨四个16行分形，本候选显式设置compact=True及x_rows\n    有效行，再由首个init_cond建立数值；不能用固定64行距的普通零种子。")
    ast.parse(text)
    path.write_text(text)
    (HERE / "PREPARATION_V2.json").write_text(json.dumps({
        "reference": "source_after_score_split/13259f92", "modified_files": [path.name],
        "reason": "v1减少重复Left但Right退成单32KiB槽；v2显式保留双Right和原Mat预取窗口",
        "gm_k": 512, "l0_k": 256, "mat_groups": 2, "right_slots_expected": 2,
        "mat_bytes_expected": 393216, "left_bytes_expected": 65536,
        "right_bytes_expected": 65536, "hardware_proven": False,
    }, ensure_ascii=False, indent=2) + "\n")
    print(source)


if __name__ == "__main__":
    main()
