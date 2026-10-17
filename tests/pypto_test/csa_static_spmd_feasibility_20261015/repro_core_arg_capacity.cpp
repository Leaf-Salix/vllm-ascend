// SPDX-License-Identifier: Apache-2.0
// CPU 检查真实 Simpler 参数容器的容量；不构造 Worker，不初始化 NPU。
#include <iostream>
#include "task_args.h"

int main() {
    TaskArgsTpl<ChipTensor, uint64_t, CORE_MAX_TENSOR_ARGS, CORE_MAX_SCALAR_ARGS> core;
    ChipTensor tensor{};
    for (int i = 0; i < CORE_MAX_TENSOR_ARGS; ++i) core.add_tensor(tensor);
    bool rejected = false;
    try {
        core.add_tensor(tensor);
    } catch (const std::out_of_range &error) {
        rejected = true;
        std::cout << "core_limit=" << CORE_MAX_TENSOR_ARGS
                  << ", next_tensor_rejected=" << error.what() << '\n';
    }
    ChipStorageTaskArgs entry;
    for (int i = 0; i < 57; ++i) entry.add_tensor(tensor);
    std::cout << "chip_limit=" << CHIP_MAX_TENSOR_ARGS
              << ", root_entry_tensor_count=" << entry.tensor_count() << '\n';
    return rejected && entry.tensor_count() == 57 ? 0 : 1;
}
