# SPDX-License-Identifier: Apache-2.0
"""打包可独立交给组件维护者的文本复现材料；不含权重、快照或设备二进制。"""

import ast
import json
import shutil
import zipfile
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    results = here.parent / 'results' / here.name
    directory = results / 'component_repro_kit_20261016'
    directory.mkdir(exist_ok=True)
    copied = []

    def copy(source, relative):
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(relative)

    for filename in ('portable_memory_compile.py', 'repro_core_arg_capacity.cpp',
                     'repro_mixed_while_loop_state.py', 'MEMORY_REUSE_PHASES_REPRO.md',
                     'JIT_SINGLE_GROUP_NOTES.md', 'SIMPLER_CORE_ARGUMENT_CAPACITY.md'):
        copy(here / filename, filename)
    for name in ('memory_indexer', 'memory_oproj', 'memory_pair_compact'):
        for filename in ('entry.py', 'scope.json'):
            relative = f'static/{name}/{filename}'
            copy(here / relative, relative)
    for filename in ('memory_probe_summary.json', 'abi_capacity.log'):
        copy(here / 'static' / filename, 'reports/' + filename)
    for case in ('memory_indexer_PYPTO', 'memory_oproj_PYPTO', 'memory_pair_compact_PYPTO',
                 'memory_pair_compact_DSA_RP_v2', 'jit_group_wrapper_v1'):
        copy(results / case / 'result.json', 'reports/' + case + '.json')

    # 原算子和测试语义不变；仅移除本机 SDK 路径选择，使用维护者自己的安装环境。
    original = (here / 'repro_jit_group_wrapper.py').read_text()
    activation = 'import dsv4_csa_env\n\ndsv4_csa_env.activate()\n\n'
    if original.count(activation) != 1:
        raise ValueError('无法唯一定位本地环境激活段')
    portable = original.replace(activation, '')
    ast.parse(portable)
    target = directory / 'repro_jit_group_wrapper.py'
    target.write_text(portable)
    copied.append(target.name)
    readme = '''# CSA 单 SPMD：组件复现包

本包不包含模型权重、KV 快照、设备二进制或性能数据；只需已安装的 PyPTO/PTOAS，
容量复现另需 Simpler 源码头文件。所有 Python 命令只编译，不运行 NPU。

## 1. 两个独立可编译阶段，融合后片上规划失败

```bash
python portable_memory_compile.py --case indexer --output /tmp/csa_mem_indexer
python portable_memory_compile.py --case oproj --output /tmp/csa_mem_oproj
python portable_memory_compile.py --case pair --output /tmp/csa_mem_pair
python portable_memory_compile.py --case pair --planner DSA_RP --output /tmp/csa_mem_pair_dsa
```

本次前两项代码生成通过，后两项失败。融合复现只需13个Tensor参数，
独立于Simpler的32参数限制。详见 MEMORY_REUSE_PHASES_REPRO.md；本包已有结果在 reports/。
原说明中提到的完整工作树路径仅是历史证据位置，运行本包不需要那些路径。

## 2. JIT 单 Group 表达

```bash
python repro_jit_group_wrapper.py --output /tmp/csa_jit_group
```

这是5参数、无while的小例子。本次编译器未将opaque wrapper识别为可提交Group。
具体范围见 JIT_SINGLE_GROUP_NOTES.md，不据此断言所有JIT表达都不可行。
本包版本只删除了本机的SDK激活代码；计算、接口及编译检查保持原样。

## 3. CORE 参数容量

```bash
c++ -std=c++17 -I /path/to/simpler/src/common/task_interface \\
    repro_core_arg_capacity.cpp -o /tmp/csa_core_capacity
/tmp/csa_core_capacity
```

这是实际TaskArgs容器检查。本次CORE容量32，完整CSA原型需要132/133。
示例中CHIP的57参数仅用于区分CHIP与CORE容量，不是最终融合核参数数。
详见 SIMPLER_CORE_ARGUMENT_CAPACITY.md。

## 4. 已提交的 while 活跃性问题

https://github.com/hw-native-sys/pypto/issues/3009

```bash
python repro_mixed_while_loop_state.py --variant scalar --output /tmp/csa_while_scalar
python repro_mixed_while_loop_state.py --variant tile --output /tmp/csa_while_tile
```

无需为了复现读取权重或执行设备任务。修复后应核对while条件、初值、回边和carry，
不能只凭进程退出码判断。包内的Python驱动会保留失败记录并返回非零退出码。

这些复现不是完整CSA功能测试。恢复完整CSA时仍需解决参数承载与完整融合编译，
再按功能、精度、性能的顺序验证。
'''
    (directory / 'README.md').write_text(readme)
    copied.append('README.md')
    manifest = {'scope': '仅文本复现、说明和已有CPU结果', 'files': sorted(copied),
                'weights_included': False, 'device_binaries_included': False,
                'new_device_execution': False, 'checksums_generated': False}
    (directory / 'FILES.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    archive = directory.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for relative in sorted(copied + ['FILES.json']):
            bundle.write(directory / relative, directory.name + '/' + relative)
    print(json.dumps({'directory': str(directory), 'archive': str(archive),
                      'files': len(copied) + 1, 'archive_bytes': archive.stat().st_size}, ensure_ascii=False))


if __name__ == '__main__':
    main()
