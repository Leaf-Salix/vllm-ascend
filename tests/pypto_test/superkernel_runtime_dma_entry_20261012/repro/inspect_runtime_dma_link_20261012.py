"""Reconstruct the former DMA executor and show both entries' symbol resolution."""
import json
import shlex
import subprocess
from pathlib import Path

private = Path(__file__).resolve().parent.parent
repo = private / "simpler"
output = private / "artifacts/runtime_dma_link_20261012"
output.mkdir(exist_ok=True)
build = repo / "build/cache/a2a3/onboard/tensormap_and_ringbuffer/aicore"
source = output / "former_executor.cpp"
source.write_text(subprocess.check_output([
    "git", "show", "3ea4375fe:src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp"
], cwd=repo, text=True))
makefile = (build / "CMakeFiles/aicore_kernel.dir/build.make").read_text().splitlines()
commands = []
for core in ("aic", "aiv"):
    old_target = str(build / f"aicore_executor_{core}.o")
    line = next(line for line in makefile if line.startswith("\t") and "bin/ccec " in line and old_target in line)
    cmd = shlex.split(line)
    cmd[cmd.index("-o") + 1] = str(output / f"former_executor_{core}.o")
    cmd[-1] = str(source)
    result = subprocess.run(cmd, cwd=repo, text=True, capture_output=True, timeout=120)
    (output / f"compile_{core}.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr[-3000:]
    commands.append(cmd)
    print("compiled", core, flush=True)
link_make = (build / "CMakeFiles/aicore_kernel_mode.dir/build.make").read_text().splitlines()
cmd = shlex.split(next(line for line in link_make if line.startswith("\t") and "bin/ld.lld " in line))
cmd[cmd.index("-o") + 1] = str(output / "former_kernel_mode.o")
for core in ("aic", "aiv"):
    old = str(build / f"aicore_executor_{core}.o")
    cmd[cmd.index(old)] = str(output / f"former_executor_{core}.o")
cmd.extend(["--cref", "-Map=" + str(output / "former_kernel_mode.map")])
result = subprocess.run(cmd, cwd=repo, text=True, capture_output=True, timeout=120)
(output / "link.log").write_text(result.stdout + result.stderr)
assert result.returncode == 0, result.stderr[-3000:]
commands.append(cmd)
for name in ("former_executor_aic.o", "former_executor_aiv.o", "former_kernel_mode.o"):
    result = subprocess.run(["readelf", "--wide", "--symbols", str(output / name)], capture_output=True, text=True, check=True)
    rows = [line for line in result.stdout.splitlines() if any(key in line for key in (
        "aicore_execute", "poll_late_handoff", "execute_dispatch_loop", "execute_worker"))]
    (output / f"{name}.symbols.txt").write_text("\n".join(rows) + "\n")
    print(name, "\n".join(rows), flush=True)
(output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
