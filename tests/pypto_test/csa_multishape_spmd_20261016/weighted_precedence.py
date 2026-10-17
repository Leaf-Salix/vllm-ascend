# SPDX-License-Identifier: Apache-2.0
"""按最长路径约简带时间差的DAG；逐对检查约束闭包保持。"""

import numpy as np

from member_schedule import ancestor_sets


def distances(parents, order):
    result = np.full((len(parents), len(parents)), -np.inf)
    for v in order:
        result[v, v] = 0.0
        for u, lag in parents[v].items():
            result[v] = np.maximum(result[v], result[u] + lag)
    return result


def reduce(parents):
    order, _ = ancestor_sets([set(p) for p in parents])
    position = {u: i for i, u in enumerate(order)}
    original = distances(parents, order)
    reduced = [dict() for _ in parents]
    for v in order:
        implied = np.full(len(parents), -np.inf)
        for u in sorted(parents[v], key=position.get, reverse=True):
            lag = parents[v][u]
            if implied[u] >= lag - 1e-9:
                continue
            reduced[v][u] = lag
            implied = np.maximum(implied, original[u] + lag)
    after = distances(reduced, order)
    reachable = np.isfinite(original)
    if not np.array_equal(reachable, np.isfinite(after)):
        raise ValueError("约简改变了可达性")
    error = float(np.max(np.abs(original[reachable] - after[reachable])))
    if error > 1e-7:
        raise ValueError("约简改变了路径时间差约束")
    return reduced, original, {
        "edges_before": sum(map(len, parents)), "edges_after": sum(map(len, reduced)),
        "all_pair_longest_lags_preserved": True, "maximum_error": error,
        "reachable_pairs": int(reachable.sum()),
    }
