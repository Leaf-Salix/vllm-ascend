# 动态CSA编译缺陷：上游issue与最小代码建议

2026-10-18。用户要求以issue和修改建议提供，不直接提修改PR，不添加标签。
四份issue均已创建，补丁直接附在正文末尾；远端正文回读与本地提交文本一致。

| Issue | 最小建议diff | 复现与验证 |
| --- | --- | --- |
| [PyPTO #3026](https://github.com/hw-native-sys/pypto/issues/3026) | [FloorMod](floor_mod_suggested.patch) | [独立复现](floor_mod_repro.py)；10项数值回归通过，相关codegen共176项唯一用例通过 |
| [PyPTO #3027](https://github.com/hw-native-sys/pypto/issues/3027) | [view别名传播](view_alias_suggested.patch) | [独立复现](view_alias_repro.py)；原样主线新增5例4失败/1通过 |
| [PyPTO #3028](https://github.com/hw-native-sys/pypto/issues/3028) | [While遍历](while_traversal_suggested.patch) | [独立复现](while_reuse_repro.py)；原样主线新增5例2失败/3通过 |
| [PTOAS #1622](https://github.com/hw-native-sys/PTOAS/issues/1622) | [并行赋值快照](ptoas_parallel_copy_suggested.patch) | [最小IR](../ptoas_parallel_copy_repro/parallel_copy.pto)；官方0.66/0.67各12/20失败，本地补丁20/20通过 |

PyPTO diff的基线为主线 `14736f65c04a7db06547edb656f515054950555c`；PTOAS diff的
基线为 `f5eff3ee249697f6157088f649c6434fcc9d7c5b`。源码版本用于定位实现，未做
文件hash校验。源码补丁保留在独立worktree，共享安装未替换，未推送修改PR。

## 验证范围

FloorMod的176项是首次175项通过，加上修正旧多缓冲测试的SSA依赖检查后剩余1项
通过，不能写成一次176项全绿。多缓冲IR另用官方PTOAS0.67实际编译通过。

两项内存diff已分别拆分，完整 `test_memory_reuse.py` 的154项通过来自二者联合
构建，没有分别重复整套测试；非零boxed Mat的通用物理偏移仍需组件侧继续评估。

PTOAS本地独立构建完成后，将新二进制生成的Scalar C++在CPU执行20例，保留所有
赋值与控制流原文，只替换设备前言并空实现屏障。结果见
[本地修正后的逐例结果](../ptoas_parallel_copy_repro/fixed_host_v1/result.json)。
构建版本仍显示0.66，不是官方0.67 wheel，也不是完整CSA/NPU同步或性能验收。

## 文件说明

- `ISSUE_<编号>_WITH_DIFF.md`：本次提交到远端的完整正文，包含最小复现和修改建议。
- `issue_<编号>_before_diff.json`、`issue_<编号>_after_diff.json`：编辑前后远端快照。
- `ISSUE_DIFFS_UPDATED.json`：远端正文核对结果及issue链接；四份标签均为空。
- `*_suggested.patch`：只包含对应缺陷的核心建议，不夹带本地构建辅助改动。
- `REPRO_VERIFICATION.json`及旧/新stdout、stderr：此前已执行的独立复现记录。

host可执行文件、构建对象和共享库仅本地保留，不纳入Git。
