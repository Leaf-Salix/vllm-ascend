import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])


def run(masks, seed):
    """Interleave one Cube and two AIV traces; reject event coalescing or live-buffer reuse."""
    valid = lambda k: 0 <= k[0] < len(masks) and bool(masks[k[0]][k[1]])

    def nxt(k):
        return (k[0] + (k[1] == 1), 1 - k[1])

    def prev(k):
        return (k[0] - (k[1] == 0), 1 - k[1])

    cube = []
    vectors = [[], []]
    for q, mask in enumerate(masks):
        pair = all(mask)
        for b in range(2):
            k = (q, b)
            if not valid(k):
                continue
            cube += [("waitall", "kv", k), ("qk", k)]
            if not pair or b == 1:
                for pb in range(2) if pair else (b,):
                    pk = (q, pb)
                    cube += [("waitall", "prob", pk), ("pv", pk)]
        for lane in range(2):
            for b in range(2):
                k = (q, b)
                if not valid(k):
                    continue
                if not valid(prev(k)):
                    vectors[lane] += [("gather", lane, k)]
                vectors[lane] += [("wait", "score", lane, k)]
                if b == 0 and valid(nxt(k)):
                    vectors[lane] += [("gather", lane, nxt(k))]
                vectors[lane] += [("softmax", lane, k)]
                if b == 1 and valid(nxt(k)):
                    vectors[lane] += [("gather", lane, nxt(k))]
                vectors[lane] += [("wait", "pv", lane, k), ("merge", lane, k)]
    threads = [cube, *vectors]
    pc = [0, 0, 0]
    ev = {}
    kv = {}
    score = {}
    prob = {}
    pv = {}
    qread = set()
    pvread = set()
    sread = set()
    pread = set()
    merged = set()
    rng = random.Random(seed)

    def publish(kind, lane, k):
        key = (kind, lane)
        assert key not in ev, ("event coalesced", key, ev.get(key), k)
        ev[key] = k

    def ready(a):
        if a[0] == "wait":
            return ev.get((a[1], a[2])) == a[3]
        if a[0] == "waitall":
            return all(ev.get((a[1], lane_index)) == a[2] for lane_index in range(2))
        return True

    while any(pc[i] < len(t) for i, t in enumerate(threads)):
        can = [i for i, t in enumerate(threads) if pc[i] < len(t) and ready(t[pc[i]])]
        assert can, ("deadlock", masks, pc, ev)
        i = rng.choice(can)
        a = threads[i][pc[i]]
        pc[i] += 1
        if a[0] == "wait":
            assert ev.pop((a[1], a[2])) == a[3]
        elif a[0] == "waitall":
            for lane_index in range(2):
                assert ev.pop((a[1], lane_index)) == a[2]
        elif a[0] == "gather":
            _, lane_index, k = a
            slot = (k[1], lane_index)
            if slot in kv:
                old = kv[slot]
                assert old in qread and old in pvread, ("KV overwritten", old, k)
            kv[slot] = k
            publish("kv", lane_index, k)
        elif a[0] == "qk":
            k = a[1]
            for lane_index in range(2):
                assert kv[(k[1], lane_index)] == k
                slot = (k[1], lane_index)
                if slot in score:
                    assert (score[slot], lane_index) in sread, ("score overwritten", score[slot], k)
                score[slot] = k
                publish("score", lane_index, k)
            qread.add(k)
        elif a[0] == "softmax":
            _, lane_index, k = a
            assert score[(k[1], lane_index)] == k
            sread.add((k, lane_index))
            if lane_index in prob:
                assert (prob[lane_index], lane_index) in pread, ("prob overwritten", prob[lane_index], k)
            prob[lane_index] = k
            publish("prob", lane_index, k)
        elif a[0] == "pv":
            k = a[1]
            for lane_index in range(2):
                assert kv[(k[1], lane_index)] == k and prob[lane_index] == k
                pread.add((k, lane_index))
                if lane_index in pv:
                    assert (pv[lane_index], lane_index) in merged, ("PV overwritten", pv[lane_index], k)
                pv[lane_index] = k
                publish("pv", lane_index, k)
            pvread.add(k)
        elif a[0] == "merge":
            _, lane_index, k = a
            assert pv[lane_index] == k
            if k[1] == 1 and masks[k[0]][0]:
                assert ((k[0], 0), lane_index) in merged
            merged.add((k, lane_index))
        else:
            raise AssertionError(a)
    assert not ev
    expected = {((q, b), lane_index) for q, m in enumerate(masks) for b in range(2) if m[b] for lane_index in range(2)}
    assert merged == expected
    return sum(pc)


count = steps = 0
for masks in itertools.product(((0, 0), (0, 1), (1, 0), (1, 1)), repeat=3):
    for seed in range(1000):
        steps += run(masks, seed)
        count += 1
record = {
    "pass": True,
    "device_execution": False,
    "source_sha256": hashlib.sha256((ROOT / "candidate_sparse.py").read_bytes()).hexdigest(),
    "queries": 3,
    "mask_patterns": 64,
    "random_interleavings": count,
    "steps": steps,
    "checks": [
        "binary event one publish/consume epoch",
        "KV two-slot QK/PV lifetime",
        "score double-slot lifetime",
        "probability/PV single-slot lifetime",
        "two AIV independent progress",
        "Native PV merge block order",
    ],
    "limitations": "abstract protocol only; UB/L1 codegen and device accuracy remain separate gates",
}
(ROOT / "PROTOCOL_MODEL.json").write_text(json.dumps(record, indent=2) + "\n")
print(record)
