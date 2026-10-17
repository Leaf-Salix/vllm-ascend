"""冻结66882634的33源，仅构造独立O_A pair候选；不写正式仓或执行编译。"""

import ast
import difflib
import json
import subprocess
from pathlib import Path


here = Path(__file__).resolve().parent
repo = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1")
relative = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
reference, candidate = here / "source_reference", here / "source_candidate"
if reference.exists() or candidate.exists():
    raise ValueError("拒绝覆盖冻结源码")
reference.mkdir(); candidate.mkdir()
names = subprocess.check_output(["git", "ls-tree", "--name-only", "66882634:" + relative], cwd=repo, text=True).splitlines()
assert len(names) == 33 and all(name.endswith(".py") for name in names)
for name in names:
    contents = subprocess.check_output(["git", "show", "66882634:" + relative + "/" + name], cwd=repo)
    (reference / name).write_bytes(contents); (candidate / name).write_bytes(contents)

probe = here.parent / "oa_pair_explicit_mat_probe_20261018/v2/pair_four_k256.py"
probe_text = probe.read_text()
pair_math = probe_text[probe_text.index("        seed0 ="):probe_text.index("    return output")]
pair_math = pair_math.replace("lhs,", "o_packed,").replace("rhs,", "wo_a,").replace(", output)", ", o_r_pad)")
helper = '''

@pl.jit.inline(auto_scope=False)
def o_projection_a_pair_group_stage_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    group_id: pl.Scalar[pl.INDEX],
):
    """仅既有N128且1≤T≤96：每组3pair+2single，实际GM容量/有效行仍动态。

    pair p归3*g+p，写2p/2p+1；single s归(3*g+3+s)%24，写6+s。
    本组五个owner互异。两个判断独立，外层在本组所有本核写入完成后才发布。
    组间packed步长由Tensor求得，不沿用资源探针固定384；输入T可小于M128，
    valid_rows贯穿Tile的compact Acc/load/store，不能写实际GM的padding/guard。
    K1024普通range内部四K256局部Mat，只有两个FP32 Acc跨回边；每条链仍
    K128依次累加。MTE2/MTE1/M事件由后端保持，源码顺序不是异步保证。
    single保留现行Tensor matmul的首K256及后续15段pipeline，量化数学不动。
    WORKERS必须24；不含任务/通知，完整依赖由cube/vector两侧匹配公式维护。
    """
    pack_stride = pl.tensor.dim(o_packed, 0) // O_GROUPS
    g = pl.max(0, group_id)
    pair_first = g * 3
    if worker_id >= pair_first and worker_id < pair_first + 3:
        pair = pl.max(0, worker_id - pair_first)
        row_base = g * pack_stride
        n0 = pair * 256
        output_col = g * O_LORA + n0
        valid_rows = tokens
        group = g
'''+pair_math+'''
    # single交给下一组三核中的前两核，避免同组pair与single串行后才发布。
    # 该映射来自8组/24AIC结构，不依赖B/S。每核至多一个pair加一个single。
    single_first = ((g + 1) * 3) % WORKERS
    if worker_id >= single_first and worker_id < single_first + 2:
        single = pl.max(0, worker_id - single_first)
        single_n0 = 768 + single * 128
        single_src0 = g * pack_stride
        single_x_first = pl.slice(o_packed, [128, 256], [single_src0, 0], valid_shape=[tokens, 256])
        single_w_first = wo_a[g : g + 1, 0:256, single_n0 : single_n0 + 128]
        single_acc = pl.matmul(single_x_first, single_w_first, out_dtype=pl.FP32)
        for single_kb in pl.pipeline(1, 16, stage=2):
            single_k0 = single_kb * 256
            single_x = pl.slice(o_packed, [128, 256], [single_src0, single_k0], valid_shape=[tokens, 256])
            single_w = wo_a[g : g + 1, single_k0 : single_k0 + 256, single_n0 : single_n0 + 128]
            single_acc = pl.matmul_acc(single_acc, single_x, single_w)
        o_r_pad = pl.assemble(o_r_pad, single_acc, [0, g * O_LORA + single_n0])
    return o_r_pad
'''
file = candidate / "output_projection_a_stages.py"
text = file.read_text()
text = text.replace("阅读", "阅读", 1)  # 保留原总览原文；追加本轮明确边界。
end_doc = text.index('"""', text.index('"""') + 3)
text = text[:end_doc] + '''
独立候选补充
    pair_group helper仅用于1≤T≤96既有N128区间，三个pair共享各自Left，
    两个single沿原数学；五owner映射及阶段通知必须与cube/vector同时修改。
    T/GM容量仍动态，不扩大到T128，也不改变N256或关闭组流水时的原路径。
''' + text[end_doc:] + helper
file.write_text(text)

file = candidate / "phase_sync.py"
file.write_text(file.read_text() + '''

@pl.jit.inline(auto_scope=False)
def wait_oa_pair_group(
    signals: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    phase: pl.Scalar[pl.INDEX],
    participant_count: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
    group: pl.Scalar[pl.INDEX],
):
    """N128小T专用五owner并集；合同0≤group<8，24AIC，三pair和两single。

    pair归3g..3g+2；single归下一组前两核，并在g7回绕到0/1。两集合互异。
    所有信号按原代际读取后才统一维护业务cache，避免每owner多一次全局fence。
    本helper没有新增信号/phase，不可用于旧logical%24或T>96分工。
    """
    for pair in pl.range(3):
        owner = group * 3 + pair
        offset = (phase * participant_count + owner) * SIGNAL_SLOT_WORDS
        signals = csa_signal_wait(signals, offset, generation)
    for single in pl.range(2):
        owner = (group * 3 + 3 + single) % 24
        offset = (phase * participant_count + owner) * SIGNAL_SLOT_WORDS
        signals = csa_signal_wait(signals, offset, generation)
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    return signals
''')

file = candidate / "cube_schedule.py"
text = file.read_text().replace("from .output_projection_a_stages import o_projection_a_groups_stage_nz, o_projection_a_stage_nz",
    "from .output_projection_a_stages import o_projection_a_groups_stage_nz, o_projection_a_stage_nz, o_projection_a_pair_group_stage_nz")
start = text.index("        for oa_group in pl.range(8):")
end = text.index("        quant_per_group =", start)
old = text[start:end]
new = '''        # 只沿现有N128/T≤96区间共享A；同一谓词必须与AIV量化等待一致。
        # 全部本核OA仍先于OB等待，不能把未完成OA移到可能阻塞的quant之后。
        if OA_COLS == 128 and tokens > 0 and tokens <= 96:
            for oa_group in pl.range(8):
                pair_first = oa_group * 3
                single_first = ((oa_group + 1) * 3) % GROUPS
                if (worker >= pair_first and worker < pair_first + 3) or (worker >= single_first and worker < single_first + 2):
                    o_r = o_projection_a_pair_group_stage_nz(packed, wo_a, o_r, tokens, worker, GROUPS, oa_group)
                    signal = publish_phase(signal, PHASE_OA_GROUP_BASE + oa_group, worker, CORES, generation)
        else:
'''+"\n".join("    "+line for line in old.rstrip().splitlines())+"\n"
text = text[:start] + new + text[end:]
file.write_text(text)

file = candidate / "vector_schedule.py"
text = file.read_text().replace("from .phase_sync import begin_invocation, publish_phase, wait_producer, wait_row_owners",
    "from .phase_sync import begin_invocation, publish_phase, wait_producer, wait_row_owners, wait_oa_pair_group")
old = '''                signal = wait_row_owners(
                    signal, PHASE_OA_GROUP_BASE + quant_group, CORES, generation,
                    quant_group * oa_per_group, oa_per_group, 1, GROUPS, 0, 1
                )'''
new = '''                # pair写两列块，owner已不再是旧logical%24；必须等同代的
                # 三pair+两single共五核。仅一次末尾cache维护，量化/pad数学不动。
                if OA_COLS == 128 and tokens > 0 and tokens <= 96:
                    signal = wait_oa_pair_group(signal, PHASE_OA_GROUP_BASE + quant_group, CORES, generation, quant_group)
                else:
                    signal = wait_row_owners(
                        signal, PHASE_OA_GROUP_BASE + quant_group, CORES, generation,
                        quant_group * oa_per_group, oa_per_group, 1, GROUPS, 0, 1
                    )'''
assert text.count(old) == 1
file.write_text(text.replace(old, new))

diff, changed = [], []
for name in names:
    before, after = (reference/name).read_text(), (candidate/name).read_text()
    ast.parse(after)
    if before != after:
        changed.append(name)
        diff.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile="source_reference/"+name, tofile="source_candidate/"+name))
assert set(changed) == {"output_projection_a_stages.py", "cube_schedule.py", "vector_schedule.py", "phase_sync.py"}
(here / "candidate.patch").write_text("".join(diff))
(here / "source_audit.json").write_text(json.dumps({
    "baseline": "66882634", "files": len(names), "changed_files": changed,
    "unchanged_files": [n for n in names if n not in changed],
    "pair_math_source": str(probe), "new_runtime_arguments": 0,
    "new_phases": 0, "new_workspaces": 0,
    "candidate_predicate": "OUTPUT_GROUP_PIPELINE && OA_COLS == 128 && 0 < tokens <= 96",
    "notes": "AST parse仅语法；完整编译和设备正确性尚待验证。未修改主仓。",
}, ensure_ascii=False, indent=2)+"\n")
(here / "controls.json").write_text(json.dumps({"OA_COLS":128,"OB_ROWS":96,"QA_M16_PARALLEL":False},indent=2)+"\n")
print(json.dumps({"files": len(names), "changed_files": changed, "candidate": str(candidate)}))
