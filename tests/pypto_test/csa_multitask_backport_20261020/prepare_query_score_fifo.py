# SPDX-License-Identifier: Apache-2.0
"""只扩大Q融合Score的GM通信环，保留双AIV、QK拆半和Top-K两步归并。"""

import argparse
import ast
import json
import shutil

from prepare import HERE
from prepare_query_ready import replace_once
from prepare_query_score import append_parameters, excerpt, function


def counted(source, before, after, count):
    if source.count(before) != count:
        raise ValueError(f"通信表达式应出现{count}次：{before}")
    return source.replace(before, after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="query_score_shared_v9")
    parser.add_argument("--candidate", default="query_score_fifo4_v10")
    args = parser.parse_args()
    if not all(n.isidentifier() for n in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / n for n in (args.base, args.candidate))
    source = (base / "decode_indexer.py").read_text()
    source = append_parameters(source, "indexer_score_topk_native_cube", ["score_slots: pl.constexpr"])
    node = function(source, "indexer_score_topk_native_cube")
    old = excerpt(source, node)
    core = old
    core = counted(
        core, "TOPK_SCORE_WORKERS * 2 * query_group_size", "TOPK_SCORE_WORKERS * score_slots * query_group_size", 1
    )
    core = counted(core, "if buf_score_step >= 2:", "if buf_score_step >= score_slots:", 2)
    # 同一乘法也出现在原score_arena的双AIV行布局中；只改四处传输行赋值。
    rows = [
        n
        for n in ast.walk(ast.parse(core))
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("buf_transfer_row", "single_transfer_row")
    ]
    if len(rows) != 4:
        raise ValueError("通信行赋值应为四处")
    lines = core.splitlines(keepends=True)
    for n in sorted(rows, key=lambda n: n.lineno, reverse=True):
        block = "".join(lines[n.lineno - 1 : n.end_lineno])
        block = counted(block, "buf_worker * 2 * query_group_size", "buf_worker * score_slots * query_group_size", 1)
        step = "single_step" if n.targets[0].id == "single_transfer_row" else "buf_score_step"
        block = counted(block, f"{step} % 2 * query_group_size", f"{step} % score_slots * query_group_size", 1)
        lines[n.lineno - 1 : n.end_lineno] = [block]
    core = "".join(lines)
    if core.count("buf_worker * 2 * query_group_size") != 4:
        raise ValueError("原score_arena的四处行地址必须保留双AIV布局")
    core = counted(core, "pl.min(buf_score_iters, 2)", "pl.min(buf_score_iters, score_slots)", 2)
    core = core.replace(
        "# 每个worker两个通信槽，每槽容纳整个query组的分数。",
        "# 每个worker有score_slots个通信槽；只有Q融合采用四槽，数学分块仍不变。",
    )
    core = core.replace("保持单MIX双槽READY/ACK。", "保持单MIX的READY/ACK；融合前段可用四槽。")
    source = replace_once(source, old, core)
    calls = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "indexer_score_topk_native_cube"
    ]
    if len(calls) != 6 or any(len(n.args) != 31 or n.keywords for n in calls):
        raise ValueError("需要六个已知的Score入口")
    lines = source.splitlines(keepends=True)
    selectable = 0
    for call in sorted(calls, key=lambda n: n.end_lineno, reverse=True):
        mode = ast.unparse(call.args[-1])
        if mode == "allow_query_fusion":
            slots = "2 + 2 * allow_query_fusion"
            selectable += 1
        elif mode == "False":
            slots = "2"
        else:
            raise ValueError(f"未知融合选择：{mode}")
        close = lines[call.end_lineno - 1]
        if close.strip() != ")":
            raise ValueError("需要独立闭括号")
        indent = " " * (len(close) - len(close.lstrip()) + 4)
        lines.insert(call.end_lineno - 1, indent + slots + ",\n")
    if selectable != 1:
        raise ValueError("只允许一个融合策略采用四槽")
    source = "".join(lines)
    tree = ast.parse(source)
    new_core = function(source, "indexer_score_topk_native_cube")
    # 传输槽的mod变化不能误改双AIV或Top-K每两步拼2048候选的规则。
    assert "pl.split_aiv(2, mode=pl.SplitMode.NONE)" in excerpt(source, new_core)
    for expression in ("single_local * 2 + single_step % 2", "single_step % 2 == 1", "single_step // 2"):
        assert expression in excerpt(source, new_core), expression
    assert sum(isinstance(n, ast.arg) and n.arg == "score_slots" for n in ast.walk(tree)) == 1
    shutil.copytree(base, target)
    (target / "decode_indexer.py").write_text(source)
    manifest = json.loads((base / "candidate.json").read_text())
    manifest.update(
        candidate=args.candidate,
        base=args.base,
        delta="Q融合的Score通信槽2→4：容量、生产准入、读写地址、末尾ACK排空一起修改；其它特化仍2槽",
        protocol="FIX完成后READY；两个AIV各在MTE2读完后ACK；生产复用前等对应ACK，末尾min(iterations,slots)排空",
        numerical_change=False,
        slot_count=4,
        extra_fused_gm_bytes=24 * 2 * 6 * 1024 * 4,
        invariants=[
            "未消费READY/ACK不超过4",
            "四槽均等两个AIV读完才复用",
            "每条消息严格各消费一次",
            "无工作叶不发事件；短于4轮只排空真实轮数",
            "双AIV、QK拆M/面板双缓冲及Top-K两步归并不变",
        ],
        api_reference="https://www.hiascend.com/doc_center/source/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0273.html",
        status="private_candidate_no_device_evidence",
    )
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
