#!/usr/bin/env python3
"""调用原版incore-profiling；只绑定生成harness的真实数据与72核启动。

生产算子、PyPTO与skill脚本均不改。调用完整PTOAS生成器后，替换独立用例
main以按storage一次分配并重建132个指针别名，动态shape标量来自真机wrapper。
模拟器经CPU队列运行，不占NPU。原始trace仅本地保留，逐核输出另做验证。
"""

import importlib.util
import json
import os
import re
import shlex
import sys
from pathlib import Path

from prepare_standalone import BUILD, CAPTURE, FUNCTION, HERE


REPO = HERE.parents[5]
WORKSPACE = REPO.parent
SKILL = HERE / "skill_snapshot/.claude/skills/incore-profiling"
ISA = WORKSPACE / ".worktrees/simpler-csa-core-args-256-20261016/build/pto-isa"
TOOLKIT = Path("/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2")


def host_main(original, bindings, capture):
    # 保留生成器的ACL/Host类型和Launch原型；仅替换独立程序的输入输出绑定。
    preamble = original[:original.index("int main()")]
    launch = "Launch" + FUNCTION[:1].upper() + FUNCTION[1:]
    prototype = re.search(r'extern "C" void ' + launch + r"\((.*?)\);", preamble, re.S)
    assert prototype
    host_types = {}
    for declaration in prototype.group(1).split(", "):
        match = re.fullmatch(r"(.*?)\s*\*\s*(v\d+)", declaration.strip())
        if match:
            host_types[match[2]] = match[1].strip()
    assert len(host_types) == 132
    storage_keys = list(capture["storages"])
    positions = {key: index for index, key in enumerate(storage_keys)}
    records = []
    for key, data in capture["storages"].items():
        records.append("    {" + str(data["bytes"]) + ', "' +
                       str(CAPTURE / "replay/before" / (key + ".bin")) + '", "' +
                       key + '.bin", ' + ("true" if data["written"] else "false") + "},")
    call_args = []
    for row in bindings["arguments"]:
        if row["kind"] == "pointer":
            value = f"static_cast<unsigned char*>(storage[{positions[row['storage']]}].device) + {row['offset_bytes']}"
            call_args.append(f"reinterpret_cast<{host_types[row['name']]}*>({value})")
        else:
            scalar_type = row["declaration"].rsplit(" ", 1)[0]
            call_args.append(f"static_cast<{scalar_type}>({row['value']})")
    call_args += ["ffts", "stream"]
    return preamble + "\n" + r'''
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int rtGetC2cCtrlAddr(uint64_t*, uint32_t*);
struct ReplayStorage {
  size_t bytes;
  const char* before;
  const char* output;
  bool written;
  void* host = nullptr;
  void* device = nullptr;
};
static void require_acl(int error, const char* operation) {
  if (error != ACL_SUCCESS) throw std::runtime_error(std::string(operation) + ": " + std::to_string(error));
}
int main() {
  std::vector<ReplayStorage> storage = {
''' + "\n".join(records) + r'''
  };
  aclrtStream stream = nullptr;
  bool initialized = false, device_set = false;
  int result = 0;
  try {
    require_acl(aclInit(nullptr), "aclInit"); initialized = true;
    require_acl(aclrtSetDevice(0), "aclrtSetDevice"); device_set = true;
    require_acl(aclrtCreateStream(&stream), "aclrtCreateStream");
    for (auto& item : storage) {
      if (std::filesystem::file_size(item.before) != item.bytes) throw std::runtime_error("input size mismatch");
      require_acl(aclrtMallocHost(&item.host, item.bytes), "aclrtMallocHost");
      require_acl(aclrtMalloc(&item.device, item.bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
      std::ifstream input(item.before, std::ios::binary);
      input.read(static_cast<char*>(item.host), item.bytes);
      if (!input || static_cast<size_t>(input.gcount()) != item.bytes) throw std::runtime_error("input read failed");
      require_acl(aclrtMemcpy(item.device, item.bytes, item.host, item.bytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
    }
    uint64_t ffts = 0;
    uint32_t ffts_length = 0;
    require_acl(rtGetC2cCtrlAddr(&ffts, &ffts_length), "rtGetC2cCtrlAddr");
''' + f"    {launch}(\n        " + ",\n        ".join(call_args) + r''');
    require_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    std::filesystem::create_directories("actual");
    for (auto& item : storage) {
      if (!item.written) continue;
      require_acl(aclrtMemcpy(item.host, item.bytes, item.device, item.bytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
      std::ofstream output(std::string("actual/") + item.output, std::ios::binary);
      output.write(static_cast<const char*>(item.host), item.bytes);
      if (!output) throw std::runtime_error("output write failed");
    }
    std::printf("CSA_REPLAY_COMPLETE blocks=24 aic=24 aiv=48 storages=%zu\n", storage.size());
  } catch (const std::exception& error) {
    std::fprintf(stderr, "CSA_REPLAY_ERROR %s\n", error.what()); result = 1;
  }
  if (stream) aclrtDestroyStream(stream);
  for (auto& item : storage) {
    if (item.device) aclrtFree(item.device);
    if (item.host) aclrtFreeHost(item.host);
  }
  if (device_set) aclrtResetDevice(0);
  if (initialized) aclFinalize();
  return result;
}
'''


def wire_case(case):
    bindings = json.loads((HERE / "WORKLOAD_BINDINGS.json").read_text())
    capture = json.loads((CAPTURE / "replay/capture.json").read_text())
    path = case / "main.cpp"
    original = path.read_text()
    (case / "main.generated.cpp.txt").write_text(original)
    path.write_text(host_main(original, bindings, capture))
    path = case / "launch.cpp"
    text = path.read_text()
    assert text.count(FUNCTION + "<<<1, nullptr, stream>>>(") == 2
    path.write_text(text.replace(FUNCTION + "<<<1, nullptr, stream>>>(", FUNCTION + "<<<24, nullptr, stream>>>("))
    path = case / "CMakeLists.txt"
    text = path.read_text().replace("-O2 -std=c++17", "-O3 -g -std=c++17")
    text = text.replace("    -s\n", "").replace("-DREGISTER_BASE", "-DMEMORY_BASE")
    text = text.replace("${ASCEND_HOME_PATH}/aarch64-linux/simulator/${SOC_VERSION}/lib",
                        "${ASCEND_HOME_PATH}/aarch64-linux/simulator/dav_2201/lib")
    path.write_text(text)
    # standalone的-xcce还会解析Host版本。真机仅设备编译的wait helper含
    # 设备专用pto::comm类型；只隐藏Host解析，设备函数体与原wrapper一致。
    path = case / f"{FUNCTION}_msprof_kernel.cpp"
    text = path.read_text()
    first = text.index("template <typename Signal>\nstatic AICORE inline void CSA_WAIT_LOCAL_GM")
    last = text.index("template <typename Tensor>", first)
    text = text[:first] + "#if defined(__CCE_AICORE__)\n" + text[first:last] + "#endif\n\n" + text[last:]
    path.write_text(text)
    (case / "workload_bindings.json").write_text(json.dumps(bindings, ensure_ascii=False, indent=2) + "\n")
    (HERE / "ACTIVE_CASE_PATH.txt").write_text(str(case) + "\n")


def prepare_inputs(case, env, log_path, timeout):
    capture = json.loads((CAPTURE / "replay/capture.json").read_text())
    assert capture["capture_complete"]
    for name, storage in capture["storages"].items():
        assert (CAPTURE / "replay/before" / (name + ".bin")).stat().st_size == storage["bytes"]
    log_path.write_text("使用完整CSA真机物理storage快照；不生成随机或零metadata；输出待与after快照逐字节核验。\n")


def main():
    spec = importlib.util.spec_from_file_location("upstream_incore_profile", SKILL / "incore_profile.py")
    profiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiler)
    original = profiler.run_cmd

    def run_cmd(command, **kwargs):
        if command[:3] == ["msprof", "op", "simulator"]:
            script = Path(kwargs["log_path"]).parent / "collect_queued.sh"
            env = kwargs["env"]
            lines = ["#!/usr/bin/env bash", "set -eo pipefail",
                     "source " + shlex.quote(str(WORKSPACE / "env-dsv4-0251rc1.sh")),
                     "cd " + shlex.quote(str(kwargs["cwd"]))]
            for key in ("PATH", "LD_LIBRARY_PATH", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
                if key in env:
                    value = env[key]
                    if key == "LD_LIBRARY_PATH":
                        value = str(TOOLKIT / "aarch64-linux/simulator/dav_2201/lib") + ":" + value
                    lines.append("export " + key + "=" + shlex.quote(value))
            lines += ["export ACL_DEVICE_ID=0", "exec " + shlex.join(command)]
            script.write_text("\n".join(lines) + "\n")
            return original(["task-submit", "--no-device", "--max-time", "1500", "--timeout", "1600", "--run",
                             "bash " + shlex.quote(str(script))],
                            log_path=kwargs["log_path"], timeout=1650, check=False)
        if command[:2] == ["cmake", "--build"]:
            command = [*command, "--parallel", "4"]
        result = original(command, **kwargs)
        if len(command) > 1 and str(command[1]).endswith("generate_testcase.py"):
            wire_case(Path(command[command.index("--output-root") + 1]) / f"ptoas/{FUNCTION}_msprof")
        return result

    profiler.run_cmd = run_cmd
    profiler.run_golden = prepare_inputs
    return profiler.main([
        "--build-dir", str(BUILD), "--target", "a2a3", "--func", FUNCTION,
        "--ptoas-root", str(WORKSPACE / "ptoas-src"), "--pto-isa-root", str(ISA),
        "--cann-set-env", str(TOOLKIT / "set_env.sh"),
        "--soc-version", "Ascend910_9392", "--aicore-arch", "dav-c220",
        "--output-root", str(HERE / "build_output"), "--name", "full_csa_all72",
        "--msprof-timeout", "900", "--step-timeout", "600", "--build-timeout", "900", "--no-keep-going",
    ])


if __name__ == "__main__":
    sys.exit(main())
