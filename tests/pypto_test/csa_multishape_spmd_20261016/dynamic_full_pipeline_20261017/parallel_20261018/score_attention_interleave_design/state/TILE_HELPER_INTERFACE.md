# 当前 Tile helper 接口：底层支持，JIT 桥接缺口明确

2026-10-18。仅阅读当前依赖源码和现有测试；没有导入 PyPTO、运行测试、编译或修改依赖。下述 `Y` 为：

`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016`

结论：**`pl.function(type=Inline)` 可接收 Tile，IR splice 不因为参数传递就复制 Tile；当前 `@pl.jit.inline` 的 Tile 参数注解却会在特化时丢失。模块级 typed Inline 也不能直接由现有 JIT 入口发现/带入。**因此不能仅将原 helper 改为带 Tile 参数的 JIT step helper，也不能只换一个 decorator 就完成接入。固定 G6/G2 的算法 Tile 签名不影响这个桥接缺口。

| 层级 | 具体源码 / 现有测试 | 可以确认什么 |
| --- | --- | --- |
| DSL 类型解析 | `Y/python/pypto/language/parser/type_resolver.py:248–288`，尤其285；同文件 Tile annotation 解析 | 参数可以是 Tile；InOut 限制仅拒绝 scalar，错误提示明确 Tensor/Tile 允许。不能据此推导 JIT 已保留该参数注解。 |
| typed Inline 实际测试 | `Y/tests/ut/ir/transforms/test_inline_functions.py:264–308` 的 `test_eval_stmt_call_site_preserves_returned_builtin_store` | helper 参数 `t: pl.Tile[[64,64],FP32]`，caller load 后调用；Expected 只有原 load→store。结构等价测试明确没有因跨 helper 额外 load/copy。此轮只读测试，未重新执行。 |
| IR Inline 形参替换 | `Y/src/ir/transforms/inline_functions_pass.cpp:252–290` | `seed[formal]=actual`，参数 use/def 均替换为同一调用方 Expr/Var；只为 helper 自己定义的局部量建立新 Var。不能把 helper 内新建 Tile 也当旧 allocation。 |
| 返回与 tuple | 同文件382–425；`Y/python/pypto/language/parser/type_resolver.py:272–277` | 原样返回同一 Var 的 `lhs=lhs` 被省去；tuple 返回逐元素替换而非搬运。参数 tuple 不支持，状态接口应为各独立参数/多返回，而非未经支持的状态对象。 |
| 当前 JIT 参数分类 | `Y/python/pypto/jit/specializer.py:1679–1747` | 只归类 Tensor/DistributedTensor 与 Scalar；Tile 不在其中。 |
| 当前 JIT 额外注解保留 | 同文件2079–2111 | 尽管注释说额外 typed annotation，实际2099条件只接受 `_LangArray`，Tile 没进入 `extra_anns`。 |
| 当前 JIT 生成形参 | 同文件2331–2387 | 非 Tensor/Scalar 且无 extra annotation 时2387只输出裸名字；`x: pl.Tile[...]` 因此失去注解。 |
| 后续明确拒绝 | `Y/python/pypto/language/parser/ast_parser.py:1107–1122` | 1117报 `Parameter 'x' missing type annotation`。这是上述源码分支推导，不冒充本轮实际编译报错。 |
| 配套已有测试 | `Y/tests/ut/jit/test_specializer.py:1927–1964` | 测试分别固定 extra annotation 被保留和缺少 extra annotation 时生成裸参数；不是 Tile 端到端支持测试。 |

返回值须与参数分开说。正式668的 `indexer_topk_stages.py:77–106` 已存在 `@pl.jit.inline` helper 返回 `root` Tile，调用方直接接着用；JIT 单 Tile 返回可省注解再由 body 推导，不能说“Tile 返回不支持”。`specializer.py:1598–1638` 还保留完整显式 tuple 元素注解。问题是当前 live Tile **参数**无法经过上述 JIT 签名重建，而不是 IR 完全没有 Tile。

“同一 allocation”也有两层：Inline pass 对传入 Var 的别名替换没有隐含 GM/Tile copy，这是源码和结构测试支持的事实；联合循环的 SSA phi、MemoryReuse、Tile 合法化仍可能引入复制或重新分配。不能从这条单元测试保证 Score/Attention 全体 live Tile 保持现成物理地址或相同峰值。

## 模块级 `@pl.function(type=Inline)` 不能直接绕开 JIT

主会话提出按 G6/G2 各写固定数学 Tile 签名、让现有 `@pl.jit.incore` 引用。当前自动接线没有这条路径：

1. `Y/python/pypto/jit/decorator.py:3967–4005` 的 `_discover_dep_bindings` 在4002只接受 `isinstance(obj,JITFunction)`。模块级 `@pl.function` 得到的是 `ir.Function`，不在发现集合。
2. `Y/python/pypto/jit/specializer.py:1208–1228` 只把已发现 dep 改写成 `self.helper`，再纳入生成的 program；`_render_free_value` 350–379只序列化常量/枚举/list/tuple等，不把外部 Function 对象注入生成文本。
3. `Y/python/pypto/jit/decorator.py:3736–3742`、3782–3788仅调用 `pl.parse(source,filename,source_map)`，没有传回原 Python module globals。
4. `Y/python/pypto/language/parser/text_parser.py:162–166,244–261` 新建解析 namespace，只预置 pl/pld，再执行该 source；没有原模块的 typed helper。
5. 下层确实有外部 IR 函数支持：`Y/python/pypto/language/parser/ast_parser.py:6657–6666` 查 `closure_vars` 中的 `ir.Function` / AST `InlineFunction`。但本例 helper 对象没有跨过第1–4步，不能仅引用这段就宣称 JIT 已桥接。

同理，模块级 `@pl.inline` 的 AST 展开器在 `ast_parser.py:8634–8713` 可将实际 Expr 直接绑定并返回，但它也不是 JITFunction、不会自动进入当前 JIT 生成模块。此轮没有发现可直接采用且已有证据的模块级混接方式。

## 对本次实现路线的含义

在不改当前依赖、不改变入口组装机制的约束下，直接 typed Tile step helper 不是现成可用接口。可维护路线需要先作明确选择：

- 修复 JIT 对 typed Tile 参数的保留及相关静态形状/返回/loop carry 合同，再保留正常的、手写的小 helper；或
- 显式使用作者维护的 `@pl.program` / IR 函数组装路径，使 typed Inline helper 真正进入同一 program。这会触及入口与依赖图的组装，不只是改 step decorator；不要求重写数学，但当前33文件的 JIT 自动依赖链不能假定原样混入。

这两条均超出本轮只读任务，未实施。没有建议把已生成的600余行冻结大体或 C++ 搬回正式 ops，也没有构造冒充现有 API 的包装器。两槽 Mat 替代可独立改善资源表达，但不修复这条 JIT 接口桥。
