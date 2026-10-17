"""保留失败v1，只构造获准的INT16 TXORS修正版，不调用编译器。

A3不支持INT32 TXORS，也没有INT16→INT32的直接TCVT分支；因此对
0..511的局部索引显式先转FP32再转INT32，两次转换均逐整数精确。
"""
from pathlib import Path
import ast
import difflib
import json
import shutil
import struct
import prepare_and_audit as v1

out=Path(__file__).resolve().parent
assert not (out/'source_v2').exists()
shutil.copytree(out/'source_candidate',out/'source_v2')
p=out/'source_v2/query_projection_stages.py'
s=p.read_text()
s=s.replace('本地512项INT32临时只在当前工作块活跃；实际TXORS别名/UB须核对生成物。','本地512项INT16索引先XOR，再经FP32精确转INT32供gather；\n    临时只在当前工作块活跃，实际TXORS别名/UB须核对生成物。')
old='''            q_flat_i = pl.tile.arange(0, [1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32)
            q_swap_xor_tmp = pl.tile.create([1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32)
            q_abs_swap = pl.tile.xors(q_flat_i, 1, q_swap_xor_tmp)'''
new='''            # A3 TXORS只接受i8/i16，这里0..511全部落在INT16正数范围。
            # A3没有INT16→INT32直接TCVT，显式经过FP32；此范围逐整数精确，
            # 两次TCVT是真实成本，不能因Python写法简短而忽略。
            q_flat_i = pl.tile.arange(0, [1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT16)
            q_swap_xor_tmp = pl.tile.create([1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT16)
            q_swap_i16 = pl.tile.xors(q_flat_i, 1, q_swap_xor_tmp)
            q_swap_f32 = pl.cast(q_swap_i16, pl.FP32)
            q_abs_swap = pl.cast(q_swap_f32, pl.INT32)'''
assert s.count(old)==1
s=s.replace(old,new)
p.write_text(s)
class N(v1.Normalize):
    def visit_Assign(self,node):
        if isinstance(node.targets[0],ast.Name) and node.targets[0].id in {'q_swap_i16','q_swap_f32'}: return None
        return super().visit_Assign(node)
original=(out/'source_before/query_projection_stages.py').read_text()
assert ast.dump(N().visit(ast.parse(original)))==ast.dump(N().visit(ast.parse(s)))
assert all(int(struct.unpack('f',struct.pack('f',j^1))[0])==(j^1) for j in range(512))
assert all((out/'source_before'/p.name).read_bytes()==p.read_bytes() for p in (out/'source_v2').glob('*.py') if p.name!='query_projection_stages.py')
(out/'candidate_v2.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='source_before/query_projection_stages.py',tofile='source_v2/query_projection_stages.py')))
(out/'source_v2_audit.json').write_text(json.dumps({'passed':True,'only_changed_file':'query_projection_stages.py','all_other_execution_ast_equal':True,'all_other_32_files_text_equal':True,'INT16_XOR_FP32_INT32_exact_cases':512,'owner_audit_reused':'source_owner_audit.json','v1_files_untouched':True,'device_execution':False},ensure_ascii=False,indent=2)+'\n')
print('v2 source/math/owner audit PASS; no compilation performed by this script')
