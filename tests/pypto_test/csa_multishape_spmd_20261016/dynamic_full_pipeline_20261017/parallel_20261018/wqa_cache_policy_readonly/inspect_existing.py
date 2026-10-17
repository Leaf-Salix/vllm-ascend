"""只读提取已有 QA 缓存策略证据；不调用编译器、不导入设备框架、不计算 hash。"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/pyptouser/qinchuanyu/pto-eager')
REPO = ROOT / 'vllm-ascend-dsv4-pto-0251rc1'
D = REPO / 'tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017'
CLOCK = D / 'parallel_20261018/current_clock_h131072_b16_v1/build_output'
FUSED = CLOCK / 'decode_csa_spmd_jqlk5eey'
ORIGINAL = CLOCK / '_decode_csa_tp1_layer_0leidxnz'
SMALL = ROOT / '.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_20261018/cpu_v1/kernel'


def inspect(label, pto, cpp, location_filter):
    ir = pto.read_text()
    code = cpp.read_text()
    loads = [dict(line=i, text=line) for i, line in enumerate(ir.splitlines(), 1)
             if 'pto.tload ' in line and 'cache_policy' in line and location_filter in line]
    snippets = []
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if '// pto: %wq_a' in line and i + 1 < len(lines):
            tail = lines[i:i + 35]
            if any('TLOAD(' in text for text in tail):
                snippets.append(dict(line=i + 1, text='\n'.join(tail)))
                break
    return dict(label=label, pto_path=str(pto), pto_bytes=pto.stat().st_size,
                cpp_path=str(cpp), cpp_bytes=cpp.stat().st_size,
                matching_ir_bypass_load_count=len(loads), ir_bypass_loads=loads,
                explicit_ir_offset_count=sum(' offset =' in x['text'] for x in loads),
                cpp_all_TLoadL2Hint_count=code.count('TLoadL2Hint'),
                cpp_all_l2Cacheoffset_count=code.count('l2Cacheoffset'),
                cpp_all_TLOAD_template_count=code.count('TLOAD<'),
                representative_wqa_cpp=snippets)


def main():
    results = [
        inspect('已运行长B16完整SPMD', FUSED / 'ptoas/dynamic_csa_full.pto',
                FUSED / 'kernels/aic/cube_schedule.cpp', 'qkv_stages.py'),
        inspect('已编译小T候选，不新增编译或读取运行中结果', SMALL / 'ptoas/dynamic_csa_full.pto',
                SMALL / 'kernels/aic/cube_schedule.cpp', 'qkv_stages.py'),
        inspect('同作业原分task QA', ORIGINAL / 'ptoas/qr_proj_matmul_0.pto',
                ORIGINAL / 'ptoas/qr_proj_matmul_0.cpp', 'qkv_proj_rope.py'),
    ]
    for result in results:
        assert result['matching_ir_bypass_load_count'] > 0
        assert result['explicit_ir_offset_count'] == 0
        assert result['cpp_all_TLoadL2Hint_count'] == 0
        assert result['cpp_all_l2Cacheoffset_count'] == 0
        assert result['cpp_all_TLOAD_template_count'] == 0
    report = dict(scope='读取已有文本，不编译、不上卡、不改源', generated=results,
        prior_dynamic_audit=json.loads((D / 'kernel_weight_bypass_v2/cache_policy_audit.json').read_text()),
        limitations=['没有比较新增DEFAULT二进制；使用已有全权重声明前后C++等价证据',
                     '普通TLOAD不证明缓存命中率、HBM流量或L2容量',
                     '本地PTOAS源码不被冒充为已安装二进制精确源码版本'])
    (HERE / 'evidence.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for item in results:
        print(item['label'], 'IR BYPASS=', item['matching_ir_bypass_load_count'],
              'C++ hint=', item['cpp_all_TLoadL2Hint_count'])


if __name__ == '__main__':
    main()
