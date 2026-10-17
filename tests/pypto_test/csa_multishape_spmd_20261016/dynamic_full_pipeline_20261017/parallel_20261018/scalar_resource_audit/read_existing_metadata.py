#!/usr/bin/env python3
"""只读已有 d253 与同作业分 task 编译产物，记录资源证据是否存在。

不编译、不运行设备、不反汇编猜测未知指令、不复制二进制、不计算 hash。
只读取小型元数据/日志，二进制仅读 ELF 魔数与文件大小；资源未知不能记为零。
"""

import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
JOB = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/'
           'csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/'
           'hc_active_notifications_h131072_b16_v1')
ROLES = {'fused_d253':'decode_csa_spmd_3kuwaqat', 'dynamic_tasks':'_decode_csa_tp1_layer_uy0bge_t'}
RESOURCE_KEY = re.compile(r'stack|spill|register|scratch|local.?memory|text.?size', re.I)


def resource_fields(value, path=''):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + '/' + key
            if RESOURCE_KEY.search(key):
                found.append({'path':child_path, 'value':child})
            found.extend(resource_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(resource_fields(child, path + '/' + str(index)))
    return found


def main():
    result = dict(
        scope='只读现有机器码容器/元数据/日志；未新编译或测量',
        job=str(JOB), roles={}, register_count=None, stack_bytes=None,
        spill_bytes=None, local_scratch_bytes=None,
        proved_fusion_spill_regression=False,
        reason='现有产物没有保留对应资源元数据，也没有可解码的函数/栈证据。',
        new_compile=False, device_execution=False, hashes_computed=False, binary_copied=False,
    )
    for role, directory in ROLES.items():
        root = JOB / 'build_output' / directory
        record = dict(path=str(root), metadata={}, kernels=[], logs=[])
        for name in ('binary_manifest.json', 'compiled_meta.json'):
            path = root / name
            value = json.loads(path.read_text())
            record['metadata'][name] = dict(
                path=str(path), bytes=path.stat().st_size, top_level_keys=list(value),
                resource_fields=resource_fields(value),
            )
        manifest = json.loads((root / 'binary_manifest.json').read_text())
        record['kernel_metadata_keys'] = sorted({key for k in manifest['kernels'] for key in k})
        for kernel in manifest['kernels']:
            path = root / kernel['binary']['path']
            with path.open('rb') as stream:
                magic = stream.read(4)
            record['kernels'].append(dict(
                func_id=kernel['func_id'], name=kernel['name'], path=str(path),
                bytes=path.stat().st_size, has_elf_header=magic == b'\x7fELF',
                contains_resource_metadata='未知；无ELF头，manifest未给资源字段',
            ))
        record['preserved_object_or_assembly_paths'] = [
            str(path) for path in root.rglob('*') if path.suffix in ('.o','.elf','.s','.ll','.su')
        ]
        for path in sorted(root.rglob('*.log')):
            hits = [{'line':no, 'text':line} for no,line in enumerate(path.read_text().splitlines(),1)
                    if RESOURCE_KEY.search(line)]
            record['logs'].append(dict(path=str(path), bytes=path.stat().st_size, resource_hits=hits))
        record['kernel_count'] = len(record['kernels'])
        record['elf_kernel_count'] = sum(k['has_elf_header'] for k in record['kernels'])
        result['roles'][role] = record
    result['existing_machine_decoder_limitation'] = dict(
        reused_prior_evidence=True,
        path='/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-physical-plan-20261018/'
             'tests/pypto_test/ultra_physical_plan_20261018/A3_FFTS_WAIT_TIMING_READONLY.md',
        section='补查：已上卡长 B16 插桩产物的机器码（同日）',
        meaning='已有CANN工具审查未能输出A3指令助记符；本轮不重复包装raw bin或反汇编。',
    )
    (OUT / 'evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({role:{k:r[k] for k in ('kernel_count','elf_kernel_count','preserved_object_or_assembly_paths')}
                      for role,r in result['roles'].items()},ensure_ascii=False))


if __name__ == '__main__':
    main()
