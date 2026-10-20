# SPDX-License-Identifier: Apache-2.0
"""私有编译进程复用冻结单SPMD的100 tick等待策略；限定被测局部MIX。"""

import argparse
import contextlib
import importlib.util
import json
import shutil
from pathlib import Path


@contextlib.contextmanager
def backoff_probe(directory, frozen_wait_source, unit_family="score"):
    from pypto.backend import pto_backend

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("_csa_backport_frozen_wait", frozen_wait_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = pto_backend._compile_pto_module
    changes = []
    prefixes = {
        "score": ("indexer_score_topk_native_pair",),
        "output": ("csa_output_groups",),
        "hc": ("hc_row_pipeline",),
    }
    allowed = prefixes[unit_family]

    def hooked(pto_code, unit_name, output_dir, *args, **kwargs):
        source = original(pto_code, unit_name, output_dir, *args, **kwargs)
        if "pto::comm::TWAIT(" not in source:
            return source
        if not unit_name.startswith(allowed):
            raise ValueError(f"等待策略不能修改{unit_family}范围外单元：{unit_name}")
        rewritten, audit = module.rewrite_wait_source(source)
        if audit["backoff_syscnt_ticks"] != 100 or not audit["writer_unchanged"]:
            raise ValueError("冻结等待策略合同改变")
        name = str(unit_name)
        (directory / f"{name}.original.cpp").write_text(source)
        (directory / f"{name}.modified.cpp").write_text(rewritten)
        changes.append(dict(unit=name, **audit))
        return rewritten

    pto_backend._compile_pto_module = hooked
    try:
        yield
        if not changes:
            raise AssertionError("没有实际改写等待单元；不能将cache命中或空变换当退避A/B")
    finally:
        pto_backend._compile_pto_module = original
        (directory / "audit.json").write_text(
            json.dumps(
                dict(
                    source=str(frozen_wait_source),
                    changes=changes,
                    unit_family=unit_family,
                    scope="仅本测试编译进程的指定局部MIX；未接入正式service或修改共享编译器",
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )


def main():
    """复制冻结融合包，以不同源码身份隔离本次编译消融的cache。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="fused_topk_v4")
    parser.add_argument("--candidate", default="fused_wait100")
    parser.add_argument("--unit-family", choices=("score", "output", "hc"), default="score")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    sources = Path(__file__).resolve().parent / "sources"
    target = sources / args.candidate
    shutil.copytree(sources / args.base, target)
    modules = {"score": "decode_indexer.py", "output": "output_group_mixed.py", "hc": "hc_row_mixed.py"}
    path = target / modules[args.unit_family]
    original = {
        "score": '"""同一24组MIX内，先发布全部本人根，再按实际生产者等待和归并。',
        "output": '"""私有候选：以冻结单SPMD的组通知流水承载多task的输出尾段。',
        "hc": '"""将冻结单SPMD的HC行依赖适配为多task前段局部MIX。',
    }[args.unit_family]
    replacement = original + "仅本测试另用100 tick等待。"
    source = path.read_text()
    if source.count(original) != 1:
        raise ValueError("需要唯一冻结融合helper")
    path.write_text(source.replace(original, replacement, 1))
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                algorithm_unchanged=True,
                wait_source="13259f92/wait_codegen.py",
                required_compile_option=(
                    "--candidate-wait-backoff cached100 --candidate-wait-units " + args.unit_family
                ),
                unit_family=args.unit_family,
                scope="测试进程局部MIX单元源码改写，尚未接入正式service",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
