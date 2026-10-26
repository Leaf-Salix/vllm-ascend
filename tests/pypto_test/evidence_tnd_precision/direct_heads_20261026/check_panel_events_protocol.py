import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

R = Path(sys.argv[1]).resolve()


def run(masks, seed):
    """Model every producer/consumer step, binary event epoch and scratch lifetime."""

    def valid(k):
        return 0 <= k[0] < len(masks) and bool(masks[k[0]][k[1]])

    def nxt(k):
        return (k[0] + (k[1] == 1), 1 - k[1])

    def prev(k):
        return (k[0] - (k[1] == 0), 1 - k[1])

    def parts(k):
        return range(1 if k[1] == 0 else 4)

    cube = []
    vectors = [[], []]
    for q, mask in enumerate(masks):
        pair = all(mask)
        for b in range(2):
            k = (q, b)
            if not valid(k):
                continue
            for part in parts(k):
                packet = (k, part)
                cube += [("waitall", "kv" + str(part), packet), ("load", packet), ("qkpart", packet)]
            cube += [("score", k)]
            if not pair or b == 1:
                for pb in range(2) if pair else (b,):
                    pk = (q, pb)
                    cube += [("waitall", "prob", pk), ("pv", pk)]
        for lane in range(2):

            def gather(k, lane=lane):
                for p in parts(k):
                    vectors[lane].extend([("gather", lane, (k, p))])

            for b in range(2):
                k = (q, b)
                if not valid(k):
                    continue
                if not valid(prev(k)):
                    gather(k)
                vectors[lane] += [("wait", "score", lane, k)]
                if b == 0 and valid(nxt(k)):
                    gather(nxt(k))
                vectors[lane] += [("softmax", lane, k)]
                if b == 1 and valid(nxt(k)):
                    gather(nxt(k))
                vectors[lane] += [("wait", "pv", lane, k), ("merge", lane, k)]
    threads = [cube, *vectors]
    pc = [0, 0, 0]
    ev = {}
    kv = {}
    score = {}
    prob = {}
    pv = {}
    loaded = set()
    computed = set()
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
            return all(ev.get((a[1], lane)) == a[2] for lane in range(2))
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
            for lane in range(2):
                assert ev.pop((a[1], lane)) == a[2]
        elif a[0] == "gather":
            _, lane, packet = a
            k, p = packet
            slot = (k[1], p, lane)
            if slot in kv:
                old = kv[slot]
                assert old in loaded and old[0] in pvread, ("KV overwritten", old, packet)
            kv[slot] = packet
            publish("kv" + str(p), lane, packet)
        elif a[0] == "load":
            packet = a[1]
            k, p = packet
            assert all(kv[(k[1], p, lane)] == packet for lane in range(2))
            loaded.add(packet)
        elif a[0] == "ack":
            packet = a[1]
            assert packet in loaded
            for lane in range(2):
                publish("ack", lane, packet)
        elif a[0] == "qkpart":
            packet = a[1]
            assert packet in loaded
            computed.add(packet)
        elif a[0] == "score":
            k = a[1]
            assert all((k, p) in computed for p in parts(k))
            for lane in range(2):
                slot = (k[1], lane)
                if slot in score:
                    assert (score[slot], lane) in sread, ("score overwritten", score[slot], k)
                score[slot] = k
                publish("score", lane, k)
        elif a[0] == "softmax":
            _, lane, k = a
            assert score[(k[1], lane)] == k
            sread.add((k, lane))
            if lane in prob:
                assert (prob[lane], lane) in pread, ("prob overwritten", prob[lane], k)
            prob[lane] = k
            publish("prob", lane, k)
        elif a[0] == "pv":
            k = a[1]
            for lane in range(2):
                assert all(kv[(k[1], p, lane)] == (k, p) for p in parts(k)) and prob[lane] == k
                pread.add((k, lane))
                if lane in pv:
                    assert (pv[lane], lane) in merged, ("PV overwritten", pv[lane], k)
                pv[lane] = k
                publish("pv", lane, k)
            pvread.add(k)
        elif a[0] == "merge":
            _, lane, k = a
            assert pv[lane] == k
            if k[1] == 1 and masks[k[0]][0]:
                assert ((k[0], 0), lane) in merged
            merged.add((k, lane))
        else:
            raise AssertionError(a)
    assert not ev
    expected = {((q, b), lane) for q, m in enumerate(masks) for b in range(2) if m[b] for lane in range(2)}
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
    "source_sha256": hashlib.sha256((R / "panel-events-v1/candidate_sparse.py").read_bytes()).hexdigest(),
    "queries": 3,
    "mask_patterns": 64,
    "random_interleavings": count,
    "steps": steps,
    "checks": [
        "fourdistinctbinaryKVepochsno-coalescing",
        "sameGMrowsreadbeforeoverwrite, no ACK needed",
        "twoAIVindependentprogress",
        "twoKVblocksnotreusedbeforePV",
        "twoSCOREslots",
        "oneprob/PVslot",
        "Native512SWAthenCMPmathorder",
    ],
    "limitations": "abstract logical protocol; actual pipe/codegen/resource and NPU accuracy separate",
}
(R / "PANEL_EVENTS_PROTOCOL_MODEL.json").write_text(json.dumps(record, indent=2) + "\n")
print(record)
