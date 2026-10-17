"""生成归档路径/大小清单；只读既有文件，不读取设备、不计算内容摘要。"""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DESIGN = HERE.parent


def main():
    output = HERE / 'LOCAL_ARTIFACTS.json'
    archive, local = [], []
    sources = {}
    for path in sorted(DESIGN.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path == output:
            continue
        relative = path.relative_to(DESIGN)
        record = dict(path=str(path), relative_path=str(relative), bytes=path.stat().st_size)
        source = next((part for part in relative.parts if part in ('source_reference', 'source_candidate', 'source_candidate_v2')), None)
        if source:
            entry = sources.setdefault(source, dict(path=str(HERE / source), files=0, bytes=0))
            entry['files'] += 1
            entry['bytes'] += record['bytes']
        cpu_tree = any(part in ('cpu_v1', 'cpu_v2') for part in relative.parts)
        full_cpu_artifact = cpu_tree and path.name not in ('result.json', 'perf_hints.log')
        if source or full_cpu_artifact:
            local.append(record)
        else:
            archive.append(record)
    result = dict(
        root=str(DESIGN),
        final_status='不采用；v1 CPU失败、v2 CPU和两档设备完整精度PASS，但同轮性能均回退',
        copy_note='archive_files为建议复制文本；local_only_files保留原完整源/CPU产物，主进程已另外冻结复制设备用33源',
        sources=sources,
        archive_files=archive,
        local_only_files=local,
        totals=dict(archive_files=len(archive), archive_bytes=sum(x['bytes'] for x in archive),
                    local_only_files=len(local), local_only_bytes=sum(x['bytes'] for x in local)),
    )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['totals']))


if __name__ == '__main__':
    main()
