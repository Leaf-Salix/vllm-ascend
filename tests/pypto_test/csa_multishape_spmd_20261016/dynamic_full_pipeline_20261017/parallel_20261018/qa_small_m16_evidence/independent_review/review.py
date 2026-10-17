"""独立只读审阅小T QA候选；不导入PyPTO、不编译、不使用NPU。

比较真实候选与其33文件基线，核对原M16算术体与所有权/容量模型。模型只
证明源级唯一写和等待集合，不能代替backend valid_shape或A3精度/协议验证。
"""

import ast
import collections
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CANDIDATE = Path("/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_20261018/source_candidate")
REFERENCE = CANDIDATE.parent / "source_reference"


class NormalizeNames(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id.startswith("small_"):
            node.id = "tail_" + node.id[len("small_"):]
        return node


def node_string(node):
    return ast.dump(node, include_attributes=False)


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


def original_blocks(tokens):
    dense = 32 if tokens < 128 else 64
    full = tokens // dense * dense
    groups = min(3, max(1, tokens // dense))
    blocks = []
    for logical in range(8 * groups):
        group, column = divmod(logical, 8)
        for row in range(group * dense, full, groups * dense):
            blocks.append((row, dense, column, logical % 24))
        for row in range(full + group * 16, (tokens + 15) // 16 * 16, groups * 16):
            blocks.append((row, min(16, tokens-row), column, logical % 24))
    return blocks


def main():
    before = sorted(REFERENCE.glob("*.py"))
    after = sorted(CANDIDATE.glob("*.py"))
    assert len(before) == len(after) == 33
    changed = [p.name for p in before if p.read_text() != (CANDIDATE/p.name).read_text()]
    assert changed == ["qkv_stages.py", "vector_schedule.py"]
    old, new = functions(REFERENCE / "qkv_stages.py"), functions(CANDIDATE / "qkv_stages.py")
    assert node_string(old["_q_a_worker"]) == node_string(new["_q_a_worker"])
    assert node_string(old["q_r_stage"]) == node_string(new["q_r_stage"])
    original = next(node for node in ast.walk(old["_q_a_worker"]) if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == "row"
        and any(isinstance(child, ast.Name) and child.id == "tail_acc" for child in ast.walk(node)))
    small = next(node for node in new["_q_a_small_m16_worker"].body if isinstance(node, ast.For))
    assert len(small.body) == len(original.body) + 2
    new_math = ast.Module(body=small.body[2:], type_ignores=[])
    assert node_string(ast.Module(body=original.body,type_ignores=[])) == node_string(NormalizeNames().visit(new_math))
    rows = []
    for tokens in (0,1,15,16,17,24,31,32,33,47,48,49):
        blocks = []
        if tokens <= 48:
            for logical in range(8 * ((tokens+15)//16)):
                row = logical//8*16
                blocks.append((row,min(16,tokens-row),logical%8,logical%24))
            wait_count = 8 * ((tokens+15)//16)
        else:
            blocks = original_blocks(tokens)
            wait_count = 8 * min(3,max(1,tokens//(32 if tokens<128 else 64)))
        written = collections.Counter()
        for row, valid, column, owner in blocks:
            assert 0<valid<=32 and row+valid<=tokens and 0<=column<8 and 0<=owner<24
            for token in range(row,row+valid):
                written[token,column]+=1
        assert set(written)=={(token,column) for token in range(tokens) for column in range(8)}
        assert all(count==1 for count in written.values())
        writers=sorted({block[3] for block in blocks})
        waited=sorted({logical%24 for logical in range(wait_count)})
        assert writers==waited
        qa_capacity=(tokens+15)//16*16
        qr_read_extent=(tokens+7)//8*8
        assert qr_read_extent<=qa_capacity
        for row,valid,_,_ in blocks:
            assert row+valid<=qa_capacity
        rows.append(dict(tokens=tokens,branch="新M16" if tokens<=48 else "原M32",
            blocks=blocks,producer_cores=writers,waited_cores=waited,
            unique_live_element_writes=True,qa_capacity=qa_capacity,qr_read_extent=qr_read_extent,
            qa_padding_rows_read_by_qr=list(range(tokens,qr_read_extent)),
            originally_active_cores=sorted({v[3] for v in original_blocks(tokens)})))
    report=dict(scope="独立源级审查，不包含CPU或设备验证",source_candidate=str(CANDIDATE),
        source_reference=str(REFERENCE),changed_files=changed,other_31_files_exactly_equal=True,
        original_q_a_worker_ast_identical=True,original_q_r_stage_ast_identical=True,
        small_math_equals_original_tail_after_local_rename=True,capacity_model=rows,
        limitations=["QR沿用读取pad8行的合同，qa中T之后内容仍不保证；不能称所有padding被清零",
            "源级valid_shape与assemble保留，实际写掩码和硬件舍入仍需后端及设备验证",
            "T0只检查helper空循环/空wait，不声称正式binding支持零T",
            "新增QA工作会推迟部分核的Indexer压缩投影，不保证完整L1改善"],
        inputs=[dict(path=str(p),bytes=p.stat().st_size) for p in before+after])
    (HERE/"review.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("源级审查通过；",len(rows),"个T边界；没有编译或NPU")


if __name__=="__main__":
    main()
