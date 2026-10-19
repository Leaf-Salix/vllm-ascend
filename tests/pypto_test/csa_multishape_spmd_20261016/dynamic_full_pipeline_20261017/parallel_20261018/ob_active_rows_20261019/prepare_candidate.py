# SPDX-License-Identifier: Apache-2.0
"""仅在NZ大M路径裁剪O_B算术padding行，动态T/原K链与其它阶段保持。"""
import ast
import json
import shutil
import subprocess
from pathlib import Path


def main():
    here=Path(__file__).resolve().parent
    repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=here,text=True).strip())
    formal=repo/'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
    base=here.parent/'attention_explicit_row_dma_20261019'
    saved={p.name:p.read_bytes() for p in formal.glob('*.py')}
    assert all(data==(base/'source_reference'/name).read_bytes() for name,data in saved.items())
    text=saved['output_projection_b_stages.py'].decode()
    text=text.replace('整体数据流与阅读顺序\n','整体数据流与阅读顺序\n'
        '    本候选只裁剪NZ大M128路径在真实T之后的算术padding行：最后一个\n'
        '    activation切片显式带min(128,T-t0)有效行，matmul和partial写回\n'
        '    都只覆盖这些行。保留所有T行（包括框架dummy），不按metadata跳算。\n'
        '    权重驻留、K1024整数点积、N256、组号及FP32反量化顺序保持。\n'
        '    工作区容量和上游INT8补零合同暂时不改；HCpost本来就只读取T行。\n',1)
    old='''                        b_act_full = o_r_i8_pad[t0 : t0 + ROW_TILE, col_g : col_g + O_LORA]
                        full_acc = pl.matmul(b_act_full, weight_resident, out_dtype=pl.INT32)
                        partials[t0 : t0 + ROW_TILE, g * D + n0 : g * D + n0 + PROJ_B_MM_N_TILE] = full_acc
'''
    new='''                        # 当前tb来自ceil(T/ROW_TILE)，所以有效行数在1..ROW_TILE。
                        # 这是删去T以外的算术padding，T内的dummy或inactive行仍完整计算。
                        # ND activation允许动态有效M；NZ权重的K/N范围没有变化。
                        active_rows = pl.min(ROW_TILE, tokens - t0)
                        b_act_full = pl.slice(
                            o_r_i8_pad, [ROW_TILE, O_LORA], [t0, col_g],
                            valid_shape=[active_rows, O_LORA],
                        )
                        full_acc = pl.matmul(b_act_full, weight_resident, out_dtype=pl.INT32)
                        # assemble沿用源Acc的有效行，不能仍按完整ROW_TILE写回。
                        # [T,P_MM)不再由此路径初始化；所有消费者必须只读取真实T。
                        partials = pl.assemble(partials, full_acc, [t0, g * D + n0])
'''
    assert text.count(old)==1
    text=text.replace(old,new,1)
    ast.parse(text)
    for name,data in saved.items():
        p=here/'source_reference'/name;p.parent.mkdir(exist_ok=True);p.write_bytes(data)
    try:
        (formal/'output_projection_b_stages.py').write_text(text)
        shutil.copytree(formal,here/'source_candidate',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    finally:
        for name,data in saved.items():(formal/name).write_bytes(data)
    for name in ['run_cpu.sh','run_device.sh','policy_candidate.json']:
        shutil.copyfile(base/name,here/name)
    controls=json.loads((base/'controls_on.json').read_text())
    controls.update(OA_COLS=256,OB_ROWS=128,LOCAL_ATTENTION_STATS=False,SCALE_PAGE_COALESCING=True)
    (here/'controls_on.json').write_text(json.dumps(controls,ensure_ascii=False,indent=2)+'\n')
    (here/'SOURCE_SCOPE.json').write_text(json.dumps(dict(
        reference_commit='289cc072',authored_in_ops=True,formal_source_restored=True,
        changed_files=['output_projection_b_stages.py'],scope='NZ ROW_TILE>96的大M路径',
        dropped_rows='仅实际Tensor形状T之后的算术补齐行；T内任何行都不跳过',
        input_capacity_and_quant_padding_unchanged=True,dependencies_changed=False,
        other_candidates_combined=False,cpu_validated=False,device_validated=False,
    ),ensure_ascii=False,indent=2)+'\n')
    print('prepared dynamic O_B valid-row candidate')


if __name__=='__main__':main()
