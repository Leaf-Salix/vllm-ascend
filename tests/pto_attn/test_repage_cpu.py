"""The strided-cache path of repage_kv, on CPU.

vLLM lays the indexer key and its scale into one padded page, so the key alone is
strided and has to be compacted into a private buffer before the kernel can take
it. Compaction moves every row, which makes two things the caller cannot see:
a slot computed against the original cache addresses a row the buffer does not
have, and a write the kernel makes in the buffer never reaches vLLM.

Both are silent -- the first shows up on device as an AICore fault far from here,
the second as an indexer that never remembers anything.
"""
import sys
import torch

ROOT = "/data/sunkaixuan/sunkaixuan_subdir/own_stack_20260918/vllm-ascend-v0.20.2rc1"
sys.path.insert(0, ROOT)
sys.argv = [sys.argv[0], "--tp", "4"]

import vllm_ascend.attention.pto_attn as pa

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


N_PAGES, PAGE, DIM = 64, 128, 32
PAD = PAGE * DIM + PAGE * 2          # vLLM's page holds the scale too
B, NCOLS, KCOLS = 4, 8, 8192


def strided_key():
    """The key view of a padded page, exactly as vLLM hands it over."""
    base = torch.arange(N_PAGES * PAD, dtype=torch.int32).remainder(97).to(torch.int8)
    return base.view(N_PAGES, PAD).as_strided(
        (N_PAGES, PAGE, DIM), (PAD, DIM, 1), 0)


print("== the fixture is what we think it is ==")
key = strided_key()
check("key is strided", not key.is_contiguous(), str(key.stride()))

# Pages scattered on purpose: a compacted buffer must not depend on their order.
torch.manual_seed(0)
bt = torch.randperm(N_PAGES)[: B * NCOLS].view(B, NCOLS).to(torch.int32)
pg = pa.repage_kv(key, bt, KCOLS)
check("compacted", pg.compacted)
check("view is contiguous", pg.view.is_contiguous())
check("view page is a quarter of vLLM's", pg.view.shape[1] == PAGE // 4,
      str(list(pg.view.shape)))

print("== a raw vLLM slot does not address this buffer ==")
req = torch.arange(B)
col = torch.tensor([3, 0, 7, 5])
intra = torch.tensor([11, 127, 0, 64])
flat = bt.long()[req, col] * PAGE + intra
rows = pg.view.shape[0] * pg.view.shape[1]
check("raw slot is out of range", bool((flat >= rows).any()),
      f"max slot {int(flat.max())} vs {rows} rows")

print("== remap puts it in range, at the row the block table implies ==")
moved = pg.remap(flat, req)
check("in range", bool(((moved >= 0) & (moved < rows)).all()), str(moved.tolist()))
want = (req * NCOLS + col) * PAGE + intra
check("lands where the compaction put the block", bool((moved == want).all()),
      f"{moved.tolist()} vs {want.tolist()}")

print("== the compacted rows carry the original bytes ==")
flatv = pg.view.reshape(-1, DIM)
src = key[bt.long()[req, col], intra]
check("seeded correctly", bool((flatv[moved] == src).all()))

print("== an unmapped slot stays inert ==")
absent = torch.tensor([-1, -1, -1, -1])
check("-1 survives remap", bool((pg.remap(absent, req) == -1).all()))

print("== commit publishes the kernel's writes back into vLLM's cache ==")
pg2 = pa.repage_kv(strided_key(), bt, KCOLS)
moved2 = pg2.remap(flat, req)
marker = torch.full((B, DIM), 42, dtype=torch.int8)
pg2.view.reshape(-1, DIM)[moved2] = marker
before = key.clone()
pg2.commit()
after = pg2._cache
check("written rows reached vLLM's page",
      bool((after[bt.long()[req, col], intra] == 42).all()))
touched = (after != before).any(dim=-1)
check("only those rows moved", int(touched.sum()) == B, f"{int(touched.sum())} rows")

print("== rejected slots write nothing, and nothing syncs ==")
# The kernel is handed -1 for a padding lane or a non-boundary token. commit has
# to leave those rows alone WITHOUT reading a mask on the host, so it redirects
# them onto (block 0, row 0) and writes back what is already there.
pg3 = pa.repage_kv(strided_key(), bt, KCOLS)
mixed = torch.stack([flat[0], torch.tensor(-1), flat[2], torch.tensor(-1)])
moved3 = pg3.remap(mixed, req)
pg3.view.reshape(-1, DIM)[moved3.clamp_min(0)] = torch.full((B, DIM), 7, dtype=torch.int8)
base = pg3._cache.clone()
pg3.commit()
after3 = pg3._cache
moved_rows = (after3 != base).any(dim=-1)
check("only the two real slots moved", int(moved_rows.sum()) == 2,
      f"{int(moved_rows.sum())} rows")
check("both real slots carry the write",
      bool((after3[bt.long()[req, col], intra][[0, 2]] == 7).all()))
check("block 0 row 0 untouched", bool((after3[0, 0] == base[0, 0]).all()))

print("== a real slot parked on row 0 is not lost to the rejected ones ==")
# Rejected rows have to go somewhere, and commit parks them on row 0. When a live
# token is written there too, every rejected row must write what that token
# writes -- otherwise the scatter picks among disagreeing duplicates and the real
# write is the one that can vanish. vLLM is not assumed to keep row 0 free.
bt0 = bt.clone()
bt0[0, 0] = 0                       # request 0 column 0 -> physical page 0
pg4 = pa.repage_kv(strided_key(), bt0, KCOLS)
with_zero = torch.stack([torch.tensor(0), torch.tensor(-1),
                         torch.tensor(-1), torch.tensor(-1)])
moved4 = pg4.remap(with_zero, req)
pg4.view.reshape(-1, DIM)[moved4.clamp_min(0)] = torch.full((B, DIM), 99, dtype=torch.int8)
base4 = pg4._cache.clone()
pg4.commit()
check("the real write at row 0 survived", bool((pg4._cache[0, 0] == 99).all()),
      f"got {pg4._cache[0, 0][:4].tolist()}")
moved_rows4 = (pg4._cache != base4).any(dim=-1)
check("and nothing else moved", int(moved_rows4.sum()) == 1,
      f"{int(moved_rows4.sum())} rows")

print("== commit does not read the device on the host ==")
import inspect
body = inspect.getsource(pa.Paged.commit)
for probe in ("bool(", "[ok]", ".item(", ".cpu(", ".tolist(", "nonzero"):
    check(f"no {probe!r}", probe not in body)

print("== a contiguous cache is left alone ==")
cont = torch.zeros(N_PAGES, PAGE, DIM, dtype=torch.int8)
pgc = pa.repage_kv(cont, bt, KCOLS)
check("not compacted", not pgc.compacted)
check("remap is identity", bool((pgc.remap(flat, req) == flat).all()))
check("view aliases the cache", pgc.view.data_ptr() == cont.data_ptr())
pgc.commit()  # must be a no-op, not a crash
check("commit is inert", bool((cont == 0).all()))

print()
if fails:
    print("FAILED:", ", ".join(fails))
    sys.exit(1)
print("ALL PASS")
