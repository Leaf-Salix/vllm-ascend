"""只读取两份已有 .pt 中的小 Top-K storage，不加载权重或大 KV payload。

只允许标准 Tensor 重建元数据的 pickle 全局对象；不导入 torch/PyPTO/NPU。
物理行不是设备快照：按当时保留的 make_fixture 源码确定性页表规则推导，
并核对 saved compressed cache shape、逻辑索引唯一性与请求页空间不重叠。
结论限定于随机 hidden/KV fixture，不能用来证明真实服务跨 query 重用价值。
"""

import collections
import io
import json
import math
import pickle
import statistics
import struct
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
C = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1')
RESULTS = C/'tests/pypto_test/results'
FROZEN_FIXTURE = Path('/data/pyptouser/qinchuanyu/pto-eager/.cache/csa-accept-9a983dae/tests/pypto_test/dsv4_csa_single_layer.py')


class TensorMetadata(pickle.Unpickler):
    """只重建字典/shape/stride/存储名，禁止执行其它 pickle 全局对象。"""

    def find_class(self, module, name):
        if module == 'torch' and name.endswith('Storage'):
            return name
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
            return lambda storage, offset, size, stride, *unused: {
                'storage': storage, 'offset': offset, 'shape': size, 'stride': stride}
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        raise ValueError(f'未允许的 pickle 对象: {module}.{name}')

    def persistent_load(self, value):
        assert value[0] == 'storage'
        return value


def load_only_indices(path):
    with zipfile.ZipFile(path) as archive:
        key = next(name for name in archive.namelist() if name.endswith('/data.pkl'))
        metadata_raw = archive.read(key)
        metadata = TensorMetadata(io.BytesIO(metadata_raw)).load()
        topk = metadata['idx_topk']
        assert topk['storage'][1] == 'IntStorage'
        rows, cols = topk['shape']
        assert cols == 512 and tuple(topk['stride']) == (512,1) and topk['offset'] == 0
        root = key.removesuffix('data.pkl')
        byteorder = archive.read(root+'byteorder').decode() if root+'byteorder' in archive.namelist() else 'little'
        assert byteorder == 'little'
        data_key = root+'data/'+topk['storage'][2]
        info = archive.getinfo(data_key)
        assert info.file_size == rows*cols*4 < 1024*1024
        raw = archive.read(data_key)
        values = struct.unpack('<'+'i'*(rows*cols),raw)
        arrays = [values[i*cols:(i+1)*cols] for i in range(rows)]
        record = {
            'path':str(path), 'file_bytes':path.stat().st_size,
            'metadata_entry':key, 'metadata_bytes_read':len(metadata_raw),
            'topk_storage_entry':data_key, 'topk_bytes_read':len(raw),
            'byteorder':byteorder, 'tensor_descriptors':metadata,
            'other_storage_payloads_read':False,
        }
        return arrays, metadata, record


def describe(values):
    data = sorted(values)
    assert data
    return {'count':len(data), 'min':data[0], 'mean':statistics.mean(data),
            'median':statistics.median(data), 'p95_nearest_rank':data[math.ceil(len(data)*.95)-1],
            'max':data[-1], 'sum':sum(data)}


def pair_statistics(pairs, logical, physical, logical_blocks, physical_blocks):
    counts = [len(logical[a]&logical[b]) for a,b in pairs]
    pcounts = [len(physical[a]&physical[b]) for a,b in pairs]
    ratios = [len(physical[a]&physical[b])/min(len(physical[a]),len(physical[b])) for a,b in pairs]
    same_block_counts = [len(physical_blocks[a][j]&physical_blocks[b][j]) for a,b in pairs for j in range(4)]
    matrix = [[statistics.mean(len(physical_blocks[a][i]&physical_blocks[b][j]) for a,b in pairs)
               for j in range(4)] for i in range(4)]
    logical_matrix = [[statistics.mean(len(logical_blocks[a][i]&logical_blocks[b][j]) for a,b in pairs)
                      for j in range(4)] for i in range(4)]
    return {'pairs':len(pairs), 'logical_intersection':describe(counts),
            'inferred_physical_intersection':describe(pcounts),
            'inferred_physical_overlap_over_smaller_valid_set':describe(ratios),
            'inferred_physical_same_128_block_intersection':describe(same_block_counts),
            'inferred_physical_block_4x4_mean_intersection':matrix,
            'logical_block_4x4_mean_intersection':logical_matrix,
            'physical_unique_pairs_nonzero':sum(n>0 for n in pcounts),
            'example_first_pair':{'queries':list(pairs[0]),'logical':counts[0],'inferred_physical':pcounts[0]}}


def analyze(history, batch):
    directory=RESULTS/'csa_accept_20260930'/f'h{history}_b{batch}'/'pto_a'
    report_path=directory/'report.json'
    report=json.loads(report_path.read_text())
    assert report['status']=='MEASURED' and report['history']==history and report['batch']==batch
    assert report['layer_index']==4 and 'synthetic history' in report['scope']
    assert Path(report['source']) == FROZEN_FIXTURE.parents[2]
    indices,meta,read_record=load_only_indices(directory/'states.pt')
    tokens=batch*6
    assert len(indices)==tokens
    # 原源码 compressed 分组 block_size=32、ratio=4。实际cache descriptor独立核对页总量。
    columns=(history+6+32*4-1)//(32*4)+1
    cache_shape=meta['compressed.0']['shape']
    assert tuple(cache_shape)==(batch*columns+1,32,1,512)
    guard_names=['metadata.positions','metadata.compressed.query_start_loc',
                 'metadata.compressed.seq_lens','metadata.compressed.block_table']
    guards={name:report['timing']['guards'][name] for name in guard_names}
    assert all(item['status']=='PASS' and item['mismatches']==0 for item in guards.values())
    assert guards['metadata.compressed.block_table']['shape']==[batch,columns]
    # 源码中给定每请求一个反序置换，物理页集合相互不交；先对全页列显式核对。
    tables=[[1+r*columns+(columns-1-col)%columns for col in range(columns)] for r in range(batch)]
    page_sets=[set(row) for row in tables]
    assert all(len(row)==columns and min(row)>=1 and max(row)<cache_shape[0] for row in page_sets)
    assert len(set.union(*page_sets))==batch*columns
    bounds=[6*r for r in range(batch+1)]
    lengths=[history+6]*batch
    positions=[history+t%6 for t in range(tokens)]
    logical=[];physical=[];logical_blocks=[];physical_blocks=[]
    duplicates=[];negative=[];causal_invalid=[];page_invalid=[];valid_counts=[]
    for q, row in enumerate(indices):
        request=q//6
        assert bounds[request]<=q<bounds[request+1]
        upper=(positions[q]+1)//4
        valid=[v for v in row if 0<=v<upper]
        duplicates.append(len(valid)-len(set(valid)))
        negative.append(sum(v<0 for v in row));causal_invalid.append(sum(v>=upper for v in row))
        page_invalid.append(sum(not 0<=v//32<columns for v in valid))
        valid_counts.append(len(valid))
        def addr(v):return tables[request][v//32]*32+v%32
        whole=set(valid)
        physical_set={addr(v) for v in whole}
        assert len(whole)==len(physical_set)
        logical.append(whole);physical.append(physical_set)
        local_blocks=[{v for v in row[i*128:(i+1)*128] if 0<=v<upper} for i in range(4)]
        logical_blocks.append(local_blocks)
        physical_blocks.append([{addr(v) for v in block} for block in local_blocks])
    assert sum(duplicates)==sum(negative)==sum(causal_invalid)==sum(page_invalid)==0
    same_request=[(r*6+s,r*6+s+1) for r in range(batch) for s in range(5)]
    same_core=[(q,q+24) for q in range(tokens-24)]
    assert all(a//6 != b//6 and a%24==b%24 for a,b in same_core)
    result={
        'history':history,'batch':batch,'query_tokens':6,'tokens':tokens,
        'input_provenance':'正式W8A8第4层权重；随机hidden与随机历史cache；不是真实整网层输入',
        'saved_arrays':'仅实际idx_topk；Q/L/positions/页表未保存在states.pt',
        'physical_scope':'按当时保留fixture源码推导的物理地址；不是设备页表快照测量',
        'report':str(report_path),'report_scope':report['scope'],'frozen_fixture_source':str(FROZEN_FIXTURE),
        'seed_from_saved_driver':1024,'compressed_columns':columns,'saved_compressed_cache_shape':cache_shape,
        'saved_metadata_readonly_checks':guards,
        'mapping_check':{'within_request_injective':True,'cross_request_page_sets_disjoint':True,
                         'all_saved_indices_valid_and_unique':True,'valid_indices_per_query':describe(valid_counts),
                         'logical_duplicates_total':sum(duplicates),'negative_total':sum(negative),
                         'causal_invalid_total':sum(causal_invalid),'page_invalid_total':sum(page_invalid)},
        'same_request_adjacent_query':pair_statistics(same_request,logical,physical,logical_blocks,physical_blocks),
        'same_core_next_query_q_plus_24':pair_statistics(same_core,logical,physical,logical_blocks,physical_blocks),
        'read_evidence':read_record,
    }
    return result


def main():
    source=FROZEN_FIXTURE.read_text()
    for fragment in [
        'tokens = batch * 6', 'bounds_cpu = torch.arange(batch + 1, dtype=torch.int32) * 6',
        'lengths_cpu = torch.full((batch,), history + 6, dtype=torch.int32)',
        'hidden = torch.randn((tokens, 4, 4096)', 'view.random_(-64, 64)', 'view.normal_(0, 0.5)',
        'columns = (history + 6 + spec.block_size * ratio - 1) // (spec.block_size * ratio) + 1',
        'per_request = columns if name in ("compressed", "indexer") else 9',
        '[1 + row * per_request + (per_request - 1 - col) % per_request for col in range(columns)]',
    ]:
        assert fragment in source, fragment
    lines=source.splitlines()
    (HERE/'fixture_source_excerpt.py.txt').write_text('\n'.join(f'{i}: {line}' for i,line in enumerate(lines,1) if 85<=i<=165)+'\n')
    output={'kind':'existing_saved_topk_only_readonly','pypto_compile':False,'device_execution':False,
            'new_weight_read':False,'cache_payload_read':False,'hash_computation':False,
            'current_20261018_success_jobs':'摘要有比较结果/shape，没有实际TopK数组；本次使用9/30历史S6快照',
            'real_service_data_limit':'优先检索canonical results：P→D bank含prefill cache/manifest，无本次decode TopK；未找到同时满足两目标S6档的真实服务层snapshot',
            'cases':[analyze(131072,16),analyze(8192,32)]}
    (HERE/'overlap.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'cases':[{k:v for k,v in row.items() if k in ['history','batch','same_request_adjacent_query','same_core_next_query_q_plus_24']} for row in output['cases']]},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
