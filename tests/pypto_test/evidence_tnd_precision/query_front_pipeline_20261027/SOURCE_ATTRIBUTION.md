# 源码来源与原始声明

归档内helper是实际执行的冻结字节，用于manifest和CPU/JIT身份复核；不为排版重写它们。部分诊断副本由AST展开生成，原来源声明在此一并保留。正式数学来源为已记录3399源快照及11b380e3f控制；新调度参考nalinaly1685da69，当前实测参考c346573a，两者区别见主报告。

其中`qa_base.py`等QKV派生来源为`deepseek_v4_flash_csa_tnd_precision/qkv_proj_rope.py`，原文件SHA256为`773f04fd3e5384b37d39fc096aec6c8327984a0d1132ce243d5885badf0eea4f`，原始文件头为：

```text
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
```

packet和其余文件所带SPDX/版权声明仍适用；复制或移植候选进入正式源码时保留原文件头。本证据目录不重新指定这些模块的许可证。
