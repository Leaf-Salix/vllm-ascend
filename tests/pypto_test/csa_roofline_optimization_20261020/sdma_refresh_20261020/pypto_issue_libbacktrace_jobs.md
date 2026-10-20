### 问题

`cmake/libbacktrace.cmake:46` 当前使用：

```cmake
BUILD_COMMAND make -j${CMAKE_BUILD_PARALLEL_LEVEL}
```

`CMAKE_BUILD_PARALLEL_LEVEL` 通常是控制 `cmake --build` 的环境变量；
`cmake --build build --parallel 2` 不会自动给配置阶段设置同名 CMake 变量。
没有显式传 `-DCMAKE_BUILD_PARALLEL_LEVEL=2` 时，上述表达式变成裸 `make -j`，
使 libbacktrace 的 ExternalProject 忽略外层两个作业的限制。

2026-10-20 核对 `main` 和 `feat/kernel-mode-integration-test` 均仍有这一行。
这是构建并行度问题，与本次 SDMA 设备功能无关。

### 实际证据

本地全新构建使用明确的 `cmake --build build --parallel 2`。
生成的 `build/CMakeFiles/project_libbacktrace.dir/build.make` 第86行包含：

```text
cd .../build/3rdparty/libbacktrace/build && make -j
make[3]: warning: -j0 forced in submake: resetting jobserver mode.
```

本次构建没有因它失败；不能把该警告当作已经发生 OOM。问题是共享机器上的
并发资源限制没有被遵守，外部 make 被要求使用无限并行。

### 最小复现

在子模块已初始化的新源码树中执行：

```bash
cmake -S . -B build-job-repro -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-job-repro --parallel 2 --target project_libbacktrace
```

检查生成的 `project_libbacktrace.dir/build.make` 与构建输出，应能看到裸
`make -j` 和 jobserver 警告。同名环境变量也不能自动成为 CMake `${...}` 变量。

### 修改建议

避免把可能为空的值拼在 `-j` 后。可为外部项目提供经过正整数验证的明确预算，
默认使用保守值，并在配置阶段显式读取已有构建预算；例如：

```cmake
set(PYPTO_EXTERNAL_BUILD_JOBS "2" CACHE STRING "Parallel jobs for external projects")
if(NOT PYPTO_EXTERNAL_BUILD_JOBS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "PYPTO_EXTERNAL_BUILD_JOBS must be a positive integer")
endif()
# ExternalProject_Add 中：
BUILD_COMMAND make -j${PYPTO_EXTERNAL_BUILD_JOBS}
```

或者正确继承外层 GNU Make jobserver；若采用后者，需要兼顾项目当前的 CMake
最低版本和 Ninja 外层生成器。这里仅提供方向，不提交修改 PR。

建议检查：未设置预算、显式设置预算、同名环境变量、不同外层生成器等情况，
确认生成命令从不意外包含裸 `-j`。本地复现脚本目前仅用显式
`-DCMAKE_BUILD_PARALLEL_LEVEL=2` 绕过，没有修改共享安装。
