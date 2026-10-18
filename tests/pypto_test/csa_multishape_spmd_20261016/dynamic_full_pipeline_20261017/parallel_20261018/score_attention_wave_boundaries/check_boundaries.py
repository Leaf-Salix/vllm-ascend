"""纯Python同源AST边界检查；不导入PyPTO/Torch，不编译、不运行设备。"""
import ast
import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


class CheckedTensor:
    def __init__(self, values, allowed=None):
        self.values = list(values)
        self.shape = (len(values),)
        self.allowed = allowed
        self.reads = []

    def read(self, indices):
        assert len(indices) == 1
        i = indices[0]
        assert 0 <= i < len(self.values), ('越界读取', i, len(self.values))
        assert self.allowed is None or i in self.allowed, ('读取非本轮描述符/容量尾部', i)
        self.reads.append(i)
        return self.values[i]


def source_function():
    tree = ast.parse((HERE/'source_candidate/wave_boundaries.py').read_text())
    fn = copy.deepcopy(next(x for x in tree.body if isinstance(x, ast.FunctionDef)))
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    body = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(body)
    # 直接解释同一函数正文：仅将device的dim/read/cast替换为受检查整数操作。
    pl = SimpleNamespace(INDEX='INDEX', cast=lambda value, dtype: int(value), read=lambda tensor, indices: tensor.read(indices), tensor=SimpleNamespace(dim=lambda tensor, axis: tensor.shape[axis]))
    env = dict(pl=pl, GROUP_DESCRIPTOR_ALIGNMENT=16, GROUP_DESCRIPTOR_WORDS=4, MIN_GROUPS_PER_WAVE=8, PLAN_QUERY_ROWS=8)
    exec(compile(body, '<same_source_integer_boundary>', 'exec'), env)
    return env['next_wave_boundary'], tree


def make_fixture(sizes, active, group_size, leading=0, trailing=0):
    assert len(sizes) == len(active)
    b = len(sizes)
    first = leading
    descriptors = []
    prefix = []
    lengths = []
    for request, (query_len, valid) in enumerate(zip(sizes, active)):
        prefix.append(len(descriptors))
        lengths.append(10000+query_len if valid and query_len > 0 else 0)
        if valid and query_len > 0:
            for local in range(0, query_len, group_size):
                # max_visible=0也可属于合法非空group，不能因此跳过group计数。
                descriptors.append((request, first+local, min(group_size, query_len-local), 0 if request % 3 == 0 else 1024))
        first += query_len
    prefix.append(len(descriptors))
    tokens = first + trailing
    base = ((b+1+15)//16)*16
    values = prefix + [0x5A5A5A5A] * (base-len(prefix))
    for desc in descriptors:
        values.extend(desc)
    values.extend([0x5A5A5A5A] * (4*tokens-4*len(descriptors)+19))
    allowed = {b} | {base+4*g+i for g in range(len(descriptors)) for i in (1,2)}
    return CheckedTensor(values, allowed), CheckedTensor(lengths), tokens, descriptors


def validate_descriptors(desc, tokens, requests):
    end = 0
    previous_request = -1
    for request, first, live, visible in desc:
        assert 0 <= request < requests and request >= previous_request
        assert live > 0 and 0 <= first and first+live <= tokens
        assert end <= first, '描述符必须单调且不重叠'
        assert visible >= 0
        end = first+live
        previous_request = request
    assert len(desc) <= tokens


def check_case(helper, name, family, sizes, active, group_size, leading=0, trailing=0):
    offsets, lengths, tokens, desc = make_fixture(sizes, active, group_size, leading, trailing)
    validate_descriptors(desc, tokens, len(sizes))
    before = list(offsets.values)
    groups = len(desc)
    if not groups:
        assert helper(offsets, lengths, tokens, 0) == (0, tokens)
        return dict(name=name, family=family, tokens=tokens, groups=0, waves=[], fallback=True, reads=len(offsets.reads))
    waves = []
    first_group = 0
    query_begin = 0
    while first_group < groups:
        nxt, end = helper(offsets, lengths, tokens, first_group)
        # 独立oracle用过滤列表求首截点，不复用候选的Scalar while控制。
        choices = [(g, desc[g][1]) for g in range(first_group+8, groups) if desc[g][1] % 8 == 0]
        expected = choices[0] if choices else (groups, tokens)
        assert (nxt,end) == expected, (name, first_group, (nxt,end), expected)
        assert first_group < nxt <= groups
        assert query_begin < end <= tokens
        assert nxt == groups or (nxt-first_group >= 8 and end % 8 == 0)
        for g in range(first_group,nxt):
            _, first, live, _ = desc[g]
            assert query_begin <= first < first+live <= end
        waves.append(dict(first_group=first_group,next_group=nxt,query_begin=query_begin,query_end=end))
        first_group,query_begin=nxt,end
    assert waves[0]['query_begin']==0 and waves[-1]['query_end']==tokens
    assert all(a['next_group']==b['first_group'] and a['query_end']==b['query_begin'] for a,b in zip(waves,waves[1:]))
    assert len(waves) <= (groups+7)//8 <= (tokens+7)//8
    # 连续query域及M8 plan块都不能由两个wave各处理一部分。
    owners=[-1]*tokens
    for w,interval in enumerate(waves):
        for q in range(interval['query_begin'],interval['query_end']):
            assert owners[q]==-1
            owners[q]=w
    assert all(x>=0 for x in owners)
    assert all(len(set(owners[q:min(q+8,tokens)]))==1 for q in range(0,tokens,8))
    assert offsets.values==before, '不得写描述符或重新编号group'
    return dict(name=name,family=family,tokens=tokens,groups=groups,waves=waves,fallback=False,reads=len(offsets.reads))


def main():
    helper,tree=source_function()
    cases=[]
    for s in (1,5,6):
        for group in (1,2,6):
            for b in range(41):
                cases.append((f'S{s}_G{group}_B{b}',f'S{s}',[s]*b,[True]*b,group,0,0))
                cases.append((f'S{s}_G{group}_B{b}_gaps',f'S{s}_gaps',[s]*b,[(i%5)!=2 for i in range(b)],group,3,5))
    for t in list(range(130))+[143,144,145,167,168,169,255,256,257,383,384,385,1023,1024,1025,4095,4096]:
        for group in (1,2,6):
            cases.append((f'T{t}_G{group}','T_boundary',[t],[True],group,0,0))
    mixed=[1,5,6,13,2,0,7,9]
    for repeat in range(1,10):
        sizes=mixed*repeat
        for group in (1,2,6):
            for gaps in (False,True):
                cases.append((f'mixed_{repeat}_G{group}_gaps{gaps}','mixed_gaps' if gaps else 'mixed',sizes,[(i%4)!=1 if gaps else True for i in range(len(sizes))],group,7,9))
    cases += [('all_dummy','empty_groups',[6,1,5],[False]*3,6,4,11),('padding_only','empty_groups',[],[],6,0,17)]
    results=[check_case(helper,*case) for case in cases]
    # 这些只是合同反例检查；设备helper没有声称修复非法生产者输出。
    invalid=[([(0,0,0,0)],1,1), ([(0,0,2,0),(0,1,1,0)],3,1), ([(0,1,2,0)],2,1), ([(1,0,1,0)],1,1)]
    rejected=0
    for desc,t,b in invalid:
        try:validate_descriptors(desc,t,b)
        except AssertionError:rejected+=1
        else:raise AssertionError('非法描述符未被合同检查拒绝')
    report={'passed':True,'scope':'同源AST整数语义+独立边界oracle；无PyPTO编译/设备/性能模型','case_count':len(results),'families':dict(Counter(r['family'] for r in results)),'fallback_cases':sum(r['fallback'] for r in results),'largest_tokens':max(r['tokens'] for r in results),'largest_groups':max(r['groups'] for r in results),'largest_wave_count':max(len(r['waves']) for r in results),'contracts':{'ordered_group_exact_coverage':True,'query_exact_coverage_including_dummy_gaps':True,'non_final_query_boundaries_aligned_8':True,'M8_plan_block_does_not_cross_wave':True,'first_wave_query_begin_zero':True,'last_wave_query_end_actual_T':True,'wave_count_le_ceil_groups_8_le_ceil_T_8':True,'metadata_not_mutated':True,'no_capacity_tail_read':True,'global_group_ids_not_rebased':True},'invalid_contract_cases_rejected':rejected,'signal_capacity_contract':{'fixed_prefix_phases':60,'phases_per_wave':3,'wave_capacity':'ceil(T/8)','signal_phase_capacity':'60+3*ceil(T/8)','signal_words':'(60+3*ceil(T/8))*72*128','empty_groups':'原不交错路径，不增加实际wave'},'examples':[r for r in results if r['name'] in {'S6_G6_B16','S5_G6_B16','S1_G6_B16','S6_G6_B16_gaps','mixed_3_G6_gapsTrue','all_dummy','padding_only'}], 'source_syntax_parsed':True,'pypto_imported':False,'pypto_lowering_run':False,'device_execution':False,'performance_model':False}
    (HERE/'boundary_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='examples'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
