# SPDX-License-Identifier: Apache-2.0
"""B1第二候选：同顺序逐根归并时，让累积根留在UB，最后才写回。

相比已经精确通过的4K分片，叶边界、每次mrgsort的两输入及顺序都不变。
循环只加载新根，省去累积根的重复GM写回/读入；最后仍写根0，保持可见结果。
"""

import ast
from pathlib import Path
import shutil


def main():
    here = Path(__file__).resolve().parent
    target = here / "source_b1_root_resident"
    shutil.copytree(here / "source_b1_leaf4k", target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    file = target / "indexer_publish_stages.py"
    text = file.read_text()
    old = """            else:
                # 原始短路径逐child向根0左折叠，且每次incoming先于旧root。
"""
    new = """            elif BALANCED and QUERY_GROUP == 2:
                # 单请求4K叶仍按child递增左折叠，incoming始终放在旧根之前。
                # 累积根在UB跨轮复用，不能换成另一棵归并树或交换同分优先级。
                root_pairs = pl.load(pair_arena, [arena_base, 0], [1, TOPK_PAIR_WIDTH])
                for child in pl.range(1, root_count):
                    next_pairs = pl.load(pair_arena, [arena_base + child, 0], [1, TOPK_PAIR_WIDTH])
                    resident_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
                    resident_merged = pl.tile.mrgsort(next_pairs, root_pairs, tmp=resident_tmp)
                    # 实取1024元素，避免把整个2048结果的生命周期带进下一轮。
                    root_pairs = pl.tile.extract(
                        resident_merged, 0, 0, [1, TOPK_PAIR_WIDTH],
                        target_memory=pl.MemorySpace.Vec,
                    )
                # 保留原根0最终内容；只消除无人消费的中间写回。
                pl.store(root_pairs, [arena_base, 0], pair_arena)
            else:
                # 原始短路径逐child向根0左折叠，且每次incoming先于旧root。
"""
    if text.count(old) != 1:
        raise ValueError("归并分支上下文改变")
    text = text.replace(old, new)
    ast.parse(text)
    file.write_text(text)
    print(target)


if __name__ == "__main__":
    main()
