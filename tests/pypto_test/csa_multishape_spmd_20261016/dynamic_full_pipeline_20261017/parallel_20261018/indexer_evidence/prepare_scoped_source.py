"""构造仅候选分支拥有UB生命周期的私有编译副本，不改冻结生产目录。"""
import ast
import copy
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE_REL = Path('vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd')
SOURCE = ROOT / PACKAGE_REL
TARGET = HERE / 'scoped_source_v7'
TARGET.mkdir()
for path in SOURCE.glob('*.py'):
    shutil.copy2(path, TARGET / path.name)
path = TARGET / 'indexer_score_stages.py'
text = path.read_text()
tree = ast.parse(text)
vector = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'indexer_score_vector_lane_stage')
branch = next(n for n in ast.walk(vector) if isinstance(n, ast.If) and ast.unparse(n.test) == 'balance_leaves')
new_body = copy.deepcopy(branch.orelse)

class ChooseLocal(ast.NodeTransformer):
    def visit_If(self, node):
        if 'LOCAL_HALF_QUERY_GROUP' in ast.unparse(node.test):
            return [self.visit(x) for x in node.body]
        return self.generic_visit(node)

new_body = ChooseLocal().visit(ast.Module(body=new_body, type_ignores=[])).body
helper_arguments = '''
    idx_native_kv_cache: pl.Tensor[[CACHE_PAGES_DYN, PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_COLUMNS_DYN], pl.INT32],
    position_ids: pl.Tensor[[TOKENS_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    buf_score_transfer: pl.Tensor[[TRANSFER_ROWS_DYN, TRANSFER_COLS_DYN], pl.FP32],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
    buf_score_lane: pl.Scalar[pl.INDEX],
    buf_worker: pl.Scalar[pl.INDEX],
    buf_batch_idx: pl.Scalar[pl.INDEX],
    buf_query: pl.Scalar[pl.INDEX],
    buf_live: pl.Scalar[pl.INDEX],
    buf_visible_count: pl.Scalar[pl.INDEX],
    buf_cache_len: pl.Scalar[pl.INDEX],
    buf_logical_begin: pl.Scalar[pl.INDEX],
    buf_leaf_capacity: pl.Scalar[pl.INDEX],
    buf_leaf: pl.Scalar[pl.INDEX],
    query_group_size: pl.constexpr,
    score_tile: pl.constexpr,
    SCORE_READY_EVENT: pl.constexpr,
    SCORE_CONSUMED_EVENT: pl.constexpr,
'''
helper_source = '''@pl.jit.inline(auto_scope=False)
def indexer_score_local_half_leaf(''' + helper_arguments + ''') -> None:
    """G2/768单叶的AIV私有UB流水；只由本模块候选constexpr分支调用。

    两query×4096候选按384列消费步长补齐为4224；A3每次完整搬运384列，
    所以物理布局为22×384，排序前reshape为2×4224。FIFO等待/ACK与原GM
    路径相同，尾组仍消费全部物理score，最终只有真实causal query发布根。
    query visibility在排序前检查一次；六个sort32/mrgsort分支逐句保留原树：
    1024为left/right，2560为right/left，3072为right0/right1/left，
    4096为right/left；这些顺序决定并列语义，不能交换或改成通用排序。
    所有参数均为Tensor/Scalar/策略常量，不跨helper传Tile。整个create和
    循环都在所选分支内联，未启用策略不会残留UB分配，避免影响其它数学tile。
    物理尺寸仍由G/score_tile表达，兼容当前前端裁剪前的分支类型检查。
    """
''' + '\n'.join('    '+line for statement in new_body for line in ast.unparse(statement).splitlines()) + '\n\n\n'
original = subprocess.check_output(['git', 'show', '593c5d25:' + str(PACKAGE_REL / 'indexer_score_stages.py')], cwd=ROOT, text=True)
original_tree = ast.parse(original)
original_vector = next(n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == vector.name)
original_branch = next(n for n in ast.walk(original_vector) if isinstance(n, ast.If) and ast.unparse(n.test) == 'balance_leaves')
original_lines = original.splitlines()
fallback = original_lines[original_branch.orelse[0].lineno-1:original_branch.orelse[-1].end_lineno]
args = [arg.arg for arg in ast.parse('def f('+helper_arguments+'):\n    pass').body[0].args.args]
replacement = '                if query_group_size == LOCAL_HALF_QUERY_GROUP and score_tile == LOCAL_HALF_SCORE_TILE:\n'
replacement += '                    indexer_score_local_half_leaf(\n'
replacement += ''.join('                        '+name+',\n' for name in args)
replacement += '                    )\n                else:\n'
replacement += '\n'.join('    '+line for line in fallback)+'\n'
lines = text.splitlines(keepends=True)
lines[branch.orelse[0].lineno-1:branch.orelse[-1].end_lineno] = [replacement]
text = ''.join(lines)
marker='@pl.jit.inline(auto_scope=False)\ndef indexer_score_vector_lane_stage('
assert text.count(marker)==1
text=text.replace(marker,helper_source+marker)
text=text.replace('''    G6、balanced及G2/1024仍沿用已有scratch/流水，不增加跨核依赖。''','''    G6、balanced及G2/1024仍沿用已有scratch/流水，不增加跨核依赖。
    UB create与流水完整封装在候选分支helper中，未启用策略不携带额外缓存。''')
path.write_text(text)
