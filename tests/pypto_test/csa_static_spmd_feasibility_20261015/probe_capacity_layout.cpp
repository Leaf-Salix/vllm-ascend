// SPDX-License-Identifier: Apache-2.0
// 用真实通用模板核对 CPU 容器与头部布局。绝不替换 CoreCallable 别名或加载核。
#include <iostream>
#include "callable.h"
#include "task_args.h"

template <int Capacity>
bool inspect() {
    using Storage = TaskArgsTpl<ChipTensor, uint64_t, Capacity, CORE_MAX_SCALAR_ARGS>;
    using Header = Callable<void, Capacity, 0>;
    Storage storage;
    ChipTensor placeholder{};  // 仅计数，不读取该占位描述符，不提交到 runtime。
    bool accepted = true;
    try {
        for (int i = 0; i < 133; ++i) storage.add_tensor(placeholder);
    } catch (const std::out_of_range &) {
        accepted = false;
    }
    std::cout << "{\"capacity\":" << Capacity
              << ",\"container_accepts_133\":" << (accepted ? "true" : "false")
              << ",\"callable_header_bytes\":" << sizeof(Header)
              << ",\"binary_offset\":" << Header::binary_data_offset()
              << ",\"transport_container_bytes\":" << sizeof(Storage)
              << ",\"current_dump_mask_fits\":"
              << (Capacity + CORE_MAX_SCALAR_ARGS <= 64 ? "true" : "false")
              << "}\n";
    return accepted == (Capacity >= 133);
}

int main() {
    bool valid = inspect<32>();
    valid = inspect<128>() && valid;
    valid = inspect<160>() && valid;
    valid = inspect<256>() && valid;
    return valid ? 0 : 1;
}
