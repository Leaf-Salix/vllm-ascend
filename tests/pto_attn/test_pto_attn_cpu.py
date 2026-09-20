"""CPU checks for the vLLM -> kernel translation, no device and no PyPTO.

Every function under test is plain tensor arithmetic, so the shape and index
logic can be pinned down before spending a device slot on it. The state ring is
the interesting one: it has to round-trip through a padded, non-contiguous cache
built exactly the way vLLM builds it.
"""
import sys
import torch

sys.path.insert(0, "/data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/vllm-ascend-v0.20.2rc1")
import vllm_ascend.attention.pto_attn as pa

RING = 16
pa.state_ring_len = lambda: RING            # stand in for the kernel constant

B, S, DIM = 4, 1, 32
T = B * S
VP, VSP = pa.VLLM_PAGE, pa.VLLM_STATE_PAGE
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


def strided_state(nblk, rows, dim, pad_elems):
    """A cache laid out the way model_runner_v1 lays one out: padded page stride."""
    raw = torch.arange(nblk * pad_elems, dtype=torch.float32)
    return torch.as_strided(raw, size=(nblk, rows, dim), stride=(pad_elems, dim, 1))


print("== rope table ==")
t = torch.arange(T * 64, dtype=torch.float32).view(T, 1, 1, 64)
r = pa._rope_to_pto(t)
check("shape", list(r.shape) == [T, 64], str(list(r.shape)))
check("halves equal", torch.equal(r[:, :32], r[:, 32:]))

print("== window indices ==")
pos = torch.tensor([300, 301, 302, 303])[:T]
swa_bt = torch.arange(B * 8, dtype=torch.int32).view(B, 8)
w = pa._window_indices(pos, swa_bt, S, 128)
check("shape", list(w.shape) == [T, 128], str(list(w.shape)))
last = w[:, -1]
want = swa_bt.long()[torch.arange(T) // S, pos // VP] * VP + pos % VP
check("last col is current token", torch.equal(last.long(), want), f"{last.tolist()} vs {want.tolist()}")

print("== repage block table ==")
bt = torch.arange(B * 4, dtype=torch.int32).view(B, 4)
rb = pa._repage_block_table(bt, 16, 4, 10_000)
check("shape", list(rb.shape) == [B, 16], str(list(rb.shape)))
check("first row", rb[0, :8].tolist() == [0, 1, 2, 3, 4, 5, 6, 7], rb[0, :8].tolist())

print("== state ring round-trip ==")
nblk = 32
pad = VSP * DIM * 2                                   # page stride twice the content
cache = strided_state(nblk, VSP, DIM, pad)
check("cache is strided", not cache.is_contiguous(), str(cache.stride()))
before = cache.clone()

state_bt = torch.arange(B * 3, dtype=torch.int32).view(B, 3) + 1
pos = torch.tensor([9, 10, 11, 12])[:T]
plan = pa.state_ring_plan(pos, S, state_bt)
ring = pa.make_state_ring(cache, plan, B, DIM)
check("ring shape", list(ring.shape) == [B * RING // 2, 2, DIM], str(list(ring.shape)))
check("ring contiguous", ring.is_contiguous())

blk, intra, ring_rows, valid = plan
check("plan sizes", all(x.shape[0] == B * RING for x in (blk, intra, ring_rows, valid)))
check("ring rows cover 0..15 per request",
      sorted((ring_rows[:RING] % RING).tolist()) == list(range(RING)))

# every seeded row must equal the cache row it came from
flat = ring.reshape(-1, DIM)
ok = True
for i in range(B * RING):
    if not valid[i]:
        continue
    want_row = cache[blk[i], intra[i]]
    if not torch.equal(flat[ring_rows[i]], want_row):
        ok = False
        break
check("seeded rows match source", ok)

# write-back must be the identity when the ring is unchanged
pa.write_state_ring(cache, ring, plan, S, pos, DIM)
check("write-back is identity for untouched ring", torch.equal(cache, before))

# a modified ring must land in the right cache rows, and only there
vidx = int(valid.nonzero()[0, 0])
ring2 = ring.clone()
ring2.reshape(-1, DIM)[ring_rows[vidx]] = -7.0
pa.write_state_ring(cache, ring2, plan, S, pos, DIM)
check("modified row lands", bool((cache[blk[vidx], intra[vidx]] == -7.0).all()),
      f"valid idx {vidx} -> page {int(blk[vidx])} row {int(intra[vidx])}")
touched = (cache != before)
check("only one row changed", int(touched.any(-1).sum()) == 1,
      f"{int(touched.any(-1).sum())} rows differ")


print("== state slots ==")
slots = pa.state_slots(pos, S)
want = (torch.arange(T) // S) * RING + pos % RING
check("slots", torch.equal(slots, want.to(torch.int64)), f"{slots.tolist()}")

print()
print("FAILED:" if fails else "ALL PASS", fails or "")
sys.exit(1 if fails else 0)
