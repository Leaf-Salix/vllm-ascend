# SPDX-License-Identifier: Apache-2.0
"""在正式动态源码编写32行独立KV缓冲候选，冻结后恢复已采用源码；不运行设备。"""
import ast
import json
import shutil
import subprocess
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    repo = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=here, text=True,
    ).strip())
    formal = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    previous = here.parent / "attention_explicit_row_dma_20261019"
    saved = {p.name: p.read_bytes() for p in formal.glob("*.py")}
    assert all(data == (previous / "source_reference" / name).read_bytes()
               for name, data in saved.items()), "正式源已变化，不能覆盖其它会话修改"
    for name, data in saved.items():
        target = here / "source_reference" / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
    text = saved["attention_stream_stages.py"].decode()
    text = text.replace(
        '整体数据流与阅读顺序\n',
        '整体数据流与阅读顺序\n'
        '    本候选在整个Attention期间保留32KiB独立KV搬运缓冲，避免与softmax/PV\n'
        '    临时Tile复用地址。每lane的64个候选分为两段32行，按原顺序写入同一\n'
        '    GM环槽；两段全部写完后才执行原KV_READY通知，三槽生命周期不变。\n'
        '    索引先整段DMA到UB，再由Scalar读取；合法行DMA与无效行补零分两遍，\n'
        '    避免每一行在V和MTE2之间往返。该资源表达能否改善完整流水必须实测。\n',
        1,
    )
    text = text.replace('ATTN_K_TILE = 128\n',
                        'ATTN_K_TILE = 128\nKV_DMA_ROWS = 32  # 32KiB持久UB；每lane分两段覆盖64行，不改变Attention数学块。\n', 1)
    helper = '''@pl.jit.inline(auto_scope=False)
def attention_gather_physical_rows(
    target: pl.Tile[[KV_DMA_ROWS, HEAD_DIM], pl.BF16, pl.Mem.Vec],
    table: pl.Tensor[[COMPRESSED_ROWS_DYN, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[PLAN_CAP_DYN, CMP_TOPK], pl.INT32],
    query: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
) -> pl.Tile[[KV_DMA_ROWS, HEAD_DIM], pl.BF16, pl.Mem.Vec]:
    """向外层独占的32行缓冲完整写入一段KV，保留负值/越界的零填充语义。

    target在整个Attention循环内保持独立，不能与softmax/PV临时空间共用。
    本函数每行只写一次，不消费旧内容；上一段MTE3读与下一段MTE2写的冲突
    仍由编译器维护。indices已由本代plan发布并取得可见性，调用方保证32项
    位于同一query行内。本地索引Tile的MTE2→Scalar就绪也由编译器维护。
    """
    result = target
    table_rows = pl.tensor.dim(table, 0)
    index_values = pl.load(indices, [query, column], [1, KV_DMA_ROWS])
    invalid_found = pl.cast(0, pl.INT32)
    # 这里只发合法行DMA，防止不同pipe写同一动态父Tile时产生逐行回边同步。
    for row in pl.range(KV_DMA_ROWS):
        physical_row = pl.cast(pl.tile.read(index_values, [0, row]), pl.INDEX)
        if physical_row >= 0 and physical_row < table_rows:
            result = pl.gather_row(result, table, [row, 0], [physical_row, 0], [1, HEAD_DIM])
        else:
            invalid_found = pl.cast(1, pl.INT32)
    # 两遍读取同一不可变的UB索引，合法行无需重读；所有无效行完整写正零。
    if invalid_found != 0:
        for row in pl.range(KV_DMA_ROWS):
            physical_row = pl.cast(pl.tile.read(index_values, [0, row]), pl.INDEX)
            if physical_row < 0 or physical_row >= table_rows:
                zero_values = pl.tile.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
                result = pl.tile.assemble(result, zero_values, [row, 0])
    return result


'''
    pos = text.index('@pl.jit.inline(auto_scope=False)\ndef attention_qk_pv_cube_stage(')
    text = text[:pos] + helper + text[pos:]
    marker = '    # 多走三拍排空末块归并；循环携带的m/l/oi仅属于当前query，\n'
    text = text.replace(marker,
        '    # 32KiB缓冲在整个循环存活，避免KV MTE3读出与Vector临时写入因复用\n'
        '    # 同址而互相阻塞。关闭物理plan时无消费者，必须由constexpr消除。\n'
        '    kv_dma_buffer = pl.create_tile([KV_DMA_ROWS, HEAD_DIM], dtype=pl.BF16,\n'
        '                                   target_memory=pl.MemorySpace.Vec)\n' + marker, 1)
    begin = text.index('                    # 本轮plan已验证唯一owner')
    end = text.index('                else:\n                    # 只有原SWA', begin)
    text = text[:begin] + '''                    # 物理plan的容量和哨兵语义沿用原逻辑。32行是搬运分段，
                    # 不改变QK/PV的128候选块或softmax归约，也不增加跨核事件。
                    physical_column = (qk_sb - 1) * ATTN_K_TILE + qk_lane_kv
                    for kv_part in pl.unroll((ATTN_K_TILE // 2) // KV_DMA_ROWS):
                        kv_part_values = attention_gather_physical_rows(
                            kv_dma_buffer, cmp_kv_flat, cmp_sparse_indices, qk_t,
                            physical_column + kv_part * KV_DMA_ROWS,
                        )
                        pl.store(kv_part_values,
                                 [qk_kv_row + qk_lane_kv + kv_part * KV_DMA_ROWS, 0],
                                 kv_transfer)
''' + text[end:]
    old = '                pl.store(qk_kv_half, [qk_kv_row + qk_lane_kv, 0], kv_transfer)'
    assert text.count(old) == 1
    text = text.replace(old, '    ' + old, 1)
    tree = ast.parse(text)
    ref_tree = ast.parse(saved['attention_stream_stages.py'])
    for name in ['attention_qk_pv_cube_stage']:
        old_ast = next(n for n in ref_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new_ast = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(old_ast) == ast.dump(new_ast)
    try:
        (formal / 'attention_stream_stages.py').write_text(text)
        shutil.copytree(formal, here / 'source_candidate',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    finally:
        for name, data in saved.items():
            (formal / name).write_bytes(data)
    for name in ['run_cpu.sh', 'run_device.sh', 'run_gather_probe.sh',
                 'controls_on.json', 'controls_short.json', 'policy_candidate.json']:
        shutil.copyfile(previous / name, here / name)
    probe = (previous / 'gather_probe.py').read_text()
    probe = probe.replace(
        '    gathered = attention_gather_physical_rows(table, indices, first, first)\n'
        '    pl.store(gathered, [0, 0], output)',
        '    buffer = pl.create_tile([32, 512], dtype=pl.BF16, target_memory=pl.MemorySpace.Vec)\n'
        '    for part in pl.unroll(2):\n'
        '        gathered = attention_gather_physical_rows(buffer, table, indices, first,\n'
        '                                                    pl.cast(part * 32, pl.INDEX))\n'
        '        pl.store(gathered, [part * 32, 0], output)',
    )
    (here / 'gather_probe.py').write_text(probe)
    (here / 'SOURCE_SCOPE.json').write_text(json.dumps(dict(
        reference_commit='289cc072', authored_in_ops=True, formal_source_restored=True,
        changes=['32行独立KV UB缓冲', '64行分两段并按原位置写入GM环',
                 'UB批量索引与合法DMA/无效补零两遍'],
        cube_ast_unchanged=True, dependencies_changed=False,
        cpu_validated=False, device_validated=False,
    ), ensure_ascii=False, indent=2) + '\n')
    print('prepared', here)


if __name__ == '__main__':
    main()
