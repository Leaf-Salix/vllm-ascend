"""只从正式668复制Attention，文本移动AIC两段；不导入PyPTO、不编译、不上卡。"""
import ast
import copy
import difflib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1')
SOURCE_PATH = 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/attention_stream_stages.py'
REVISION = '66882634'


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
    return tree


def body_text(source, node):
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return ''.join(source.splitlines(keepends=True)[start - 1:node.end_lineno])


def main():
    source = subprocess.check_output(['git', '-C', str(REPO), 'show', REVISION + ':' + SOURCE_PATH], text=True)
    old = ast.parse(source)
    cube = function(old, 'attention_qk_pv_cube_stage')
    loop = next(n for n in cube.body if isinstance(n, ast.For))
    assert len(loop.body) == 2 and all(isinstance(n, ast.If) for n in loop.body)
    qk, pv = loop.body
    lines = source.splitlines(keepends=True)
    qk_text = ''.join(lines[qk.lineno-1:pv.lineno-1])
    pv_text = ''.join(lines[pv.lineno-1:loop.end_lineno])
    candidate = ''.join(lines[:qk.lineno-1]) + (
        '        # 先消耗两拍前的L1 KV：r与r-2同属一个%2槽，必须先PV后覆盖。\n'
        '        # 本段的GM概率/PV地址仍按%3，FFTS的PROB/PV事件沿用原条件。\n'
    ) + pv_text + '\n' + (
        '        # 再写入本拍KV并做QK。这里的源级先后只建立所需数据顺序；\n'
        '        # 后端仍须在最后PV MTE1读取与新KV MTE2覆盖间生成正确依赖。\n'
        '        # 不能把Scalar走过PV语句当作异步搬运已经完成。\n'
    ) + qk_text.rstrip('\n') + '\n' + ''.join(lines[loop.end_lineno:])
    old_create = 'qk_l1 = pl.create_tile([QK_TRANSFER_SLOTS * ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)'
    new_create = 'qk_l1 = pl.create_tile([QK_L1_SLOTS * ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)'
    assert candidate.count(old_create) == 1
    candidate = candidate.replace(old_create, new_create)
    for name, work in [('qk_l1_row', 'qk_tick'), ('pv_l1_row', 'pv_work')]:
        before = f'{name} = ({work} % QK_TRANSFER_SLOTS) * ATTN_K_TILE'
        after = f'{name} = ({work} % QK_L1_SLOTS) * ATTN_K_TILE'
        assert candidate.count(before) == 1
        candidate = candidate.replace(before, after)
    candidate = candidate.replace('QK_TRANSFER_SLOTS = 3\n', 'QK_TRANSFER_SLOTS = 3  # 所有GM ring仍三槽；不得与AIC内部L1槽数合并。\nQK_L1_SLOTS = 2  # 本候选仅AIC KV Mat双槽，依赖每tick先PV后QK。\n', 1)
    candidate = candidate.replace('"""动态 CSA Attention 的成对 AIC / 双 AIV 三槽流水与最终头向量发布。', '"""动态 CSA Attention 源码准备：GM三槽、AIC L1双槽与双AIV最终发布。\n\n候选边界\n    这是未来Score/Attention同核交错的源码准备，不是独立性能候选。\n    只把AIC内部KV Mat改为两槽，tick内先PV(tick-2)再QK(tick)；\n    GM全部ring、AIV函数正文、动态T/device mask和每块算术均保持正式668。\n    尚未lowering/上卡，不能以源码顺序代替MTE1→MTE2覆盖依赖或性能证据。', 1)
    candidate = candidate.replace('attention_qk_pv_cube_stage 执行QK与滞后两拍的PV；', 'attention_qk_pv_cube_stage 先执行滞后两拍PV，再执行本拍QK；', 1)
    candidate = candidate.replace('首版保留其5块/3槽/提前KV分支，不用全阶段barrier替代循环流水。', '保留其5块/GM三槽/提前KV分支，仅AIC L1改双槽；不加入全阶段barrier。', 1)
    candidate = candidate.replace('64头、512列、128候选块、5块和3槽是模型/流水', '64头、512列、128候选块、5块、GM三槽和AIC L1双槽是模型/流水', 1)
    candidate = candidate.replace('AIC第tick拍做QK(tick)、PV(tick-2)；AIV做gather(tick)、softmax(tick-1)、', 'AIC第tick拍先做PV(tick-2)、再QK(tick)，L1位置分别用work%2；\n    GM位置仍用core_id*3+work%3。AIV做gather(tick)、softmax(tick-1)、', 1)
    candidate = candidate.replace('"""AIC侧：QK领先PV两拍；必须与本组两个AIV helper并发。"""', '"""AIC侧：PV仍处理两拍前数据，但每tick先PV后QK以保护L1双槽。\n\n    首两拍仅QK，末两拍仅PV排空；每块仍使用原device mask。\n    r与r-2共享L1槽，必须完成旧PV的MTE1读取后才能覆盖新KV。\n    GM三槽保持不变，避免AIV的mi/li(tick-1)与归并(tick-3)撞槽。\n    与双AIV并发、局部复用事件及真实资源分配仍需后端与设备验证。\n    """', 1)
    candidate = candidate.replace('    # KV 只进一次 L1：QK 用它的转置视图，PV 用同一份的行切片。\n    # 原先 QK 与 PV 各自从 GM 搬一遍，同一块 KV 过两次 GM->L1。', '    # KV仍只进一次L1：QK用转置view，之后PV用同一份行切片。\n    # 本候选只缩AIC内部同dtype Mat，不改GM环，不做混类型槽别名。\n    # 原来三槽允许先QK后PV；双槽必须先消费旧PV，再覆盖同一槽。', 1)
    candidate = candidate.replace('    # 多走两拍排空最后PV；空组仅走控制分支，不读plan或环槽。', '    # 循环界仍为真实query块数+2：首两拍填充、末两拍排空；空组不读环槽。\n    # mask无效则该块两端均跳过事件/访存，不能为了双槽无条件读取旧KV。', 1)

    after = ast.parse(candidate)
    new_cube = function(after, cube.name)
    new_loop = next(n for n in new_cube.body if isinstance(n, ast.For))
    assert len(new_loop.body) == 2
    expected = copy.deepcopy(old)
    expected_cube = function(expected, cube.name)
    expected_loop = next(n for n in expected_cube.body if isinstance(n, ast.For))
    expected_loop.body.reverse()
    changed_assignments = []
    for n in ast.walk(expected_cube):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {'qk_l1', 'qk_l1_row', 'pv_l1_row'}:
            changed_assignments.append(n.targets[0].id)
            for v in ast.walk(n.value):
                if isinstance(v, ast.Name) and v.id == 'QK_TRANSFER_SLOTS':
                    v.id = 'QK_L1_SLOTS'
    idx = next(i for i,n in enumerate(expected.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'QK_TRANSFER_SLOTS')
    expected.body.insert(idx+1, ast.parse('QK_L1_SLOTS = 2').body[0])
    assert ast.dump(strip_docstrings(copy.deepcopy(expected)), include_attributes=False) == ast.dump(strip_docstrings(copy.deepcopy(after)), include_attributes=False)
    old_vec = function(old, 'attention_qk_pv_vector_lane_stage')
    new_vec = function(after, old_vec.name)
    aiv_equal = body_text(source, old_vec) == body_text(candidate, new_vec)
    assert aiv_equal
    qk_after = copy.deepcopy(new_loop.body[1]);pv_after = copy.deepcopy(new_loop.body[0])
    for block in [qk_after,pv_after]:
        for n in ast.walk(block):
            if isinstance(n, ast.Name) and n.id == 'QK_L1_SLOTS': n.id='QK_TRANSFER_SLOTS'
    assert ast.dump(qk,include_attributes=False) == ast.dump(qk_after,include_attributes=False)
    assert ast.dump(pv,include_attributes=False) == ast.dump(pv_after,include_attributes=False)
    old_assign = {n.targets[0].id:ast.dump(n.value,include_attributes=False) for n in ast.walk(cube) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    new_assign = {n.targets[0].id:ast.dump(n.value,include_attributes=False) for n in ast.walk(new_cube) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    gm_names=['qk_slot','qk_kv_row','qk_transfer_row','pv_slot','pv_transfer_row']
    gm_equal={n:old_assign[n]==new_assign[n] for n in gm_names}
    assert all(gm_equal.values())
    assert ast.dump(cube.args,include_attributes=False)==ast.dump(new_cube.args,include_attributes=False)
    compile(candidate, str(HERE/'source_candidate/attention_stream_stages.py'), 'exec')
    for sub, content in [('source_reference',source),('source_candidate',candidate)]:
        out=HERE/sub/'attention_stream_stages.py';out.parent.mkdir(exist_ok=True)
        if out.exists():
            assert out.read_text()==content, '拒绝覆盖已存在且不同的冻结源码: '+str(out)
        else:out.write_text(content)
    diff=''.join(difflib.unified_diff(source.splitlines(keepends=True),candidate.splitlines(keepends=True),fromfile='source_reference/attention_stream_stages.py',tofile='source_candidate/attention_stream_stages.py'))
    (HERE/'candidate.patch').write_text(diff)
    report={'passed':True,'reference_revision':REVISION,'reference_path':SOURCE_PATH,'aiv_decorator_through_body_byte_equal':aiv_equal,'module_ast_only_expected_changes':True,'expected_execution_changes':['新增QK_L1_SLOTS=2','qk_l1容量使用两槽','qk_l1_row/pv_l1_row仅内部L1取模改2','AIC每tick完整PV块移到完整QK块之前'],'l1_assignment_names':sorted(changed_assignments),'qk_subtree_equal_after_l1_modulus_normalization':True,'pv_subtree_equal_after_l1_modulus_normalization':True,'gm_address_ast_equal':gm_equal,'signature_equal':True,'dynamic_loop_bound_and_device_mask_preserved':True,'python_syntax_passed':True,'pypto_imported':False,'pypto_lowering_run':False,'npu_run':False,'performance_claim':False,'old_cube_lines':[cube.lineno,cube.end_lineno],'new_cube_lines':[new_cube.lineno,new_cube.end_lineno],'old_aiv_lines':[old_vec.lineno,old_vec.end_lineno],'new_aiv_lines':[new_vec.lineno,new_vec.end_lineno],'files':[{'path':str(HERE/x),'bytes':(HERE/x).stat().st_size} for x in ['source_reference/attention_stream_stages.py','source_candidate/attention_stream_stages.py','candidate.patch']]}
    (HERE/'source_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
