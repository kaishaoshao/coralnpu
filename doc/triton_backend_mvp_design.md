# CoralNPU Triton Backend MVP Design

本文档给出一个面向当前 `coralnpu` 仓库的 Triton 后端 MVP 设计稿，目标是把前一份建议文档落到可以开工的层面。

对应的背景建议见 [triton_backend_recommendations.md](/home/shaokai/Desktop/code/google-npu/coralnpu/doc/triton_backend_recommendations.md:1)。

## MVP 目标

第一阶段只解决这 4 件事：

1. 接收一个小范围 Triton kernel
2. 生成可被当前 `coralnpu` 仓库消费的离线产物
3. 通过 Bazel 工具链打包成 ELF
4. 在 simulator 上完成功能验证和 cycle 验证

第一阶段不解决：

- 完整 TorchInductor 集成
- 完整 Triton API 覆盖
- 完整动态 shape
- 完整 runtime / driver / memory manager
- 多设备透明调度

## 总体原则

MVP 必须遵守三个约束：

- 优先 AOT，而不是完整 JIT
- 优先 out-of-tree backend，而不是直接侵入 Triton 主仓库
- 优先复用当前仓库的交叉工具链、Bazel 规则和 simulator

## 推荐的最小架构

建议的第一阶段编译链如下：

```text
Python Triton kernel
-> Triton frontend / Triton IR
-> CoralNPU backend plugin
-> 中性 MLIR / backend-specific IR
-> Coral/RVV/scalar lowering
-> C/C++ stub or object or assembly
-> Bazel target in coralnpu repo
-> coralnpu_v2_binary
-> ELF
-> Verilator / hw_sim
```

这里故意把“最终 codegen 产物”说得宽一点，因为 MVP 不必从第一天就直接生成 Coral 原生二进制格式。第一版允许采用更保守的桥接策略。

## 为什么这样设计

当前仓库已经天然提供了三类重要能力：

- Coral ELF 构建规则 `coralnpu_v2_binary`
- Verilator simulator 运行路径
- 可嵌入式 simulator wrapper / shared library

参考位置：

- [`rules/coralnpu_v2.bzl`](/home/shaokai/Desktop/code/google-npu/coralnpu/rules/coralnpu_v2.bzl:220)
- [`tests/verilator_sim/BUILD`](/home/shaokai/Desktop/code/google-npu/coralnpu/tests/verilator_sim/BUILD:222)
- [`hw_sim/BUILD`](/home/shaokai/Desktop/code/google-npu/coralnpu/hw_sim/BUILD:68)

所以 MVP 最应该做的是把 Triton 编译结果“接入现有路径”，而不是再发明一套新的运行体系。

## 推荐目录结构

如果单独做一个 out-of-tree backend，我建议类似这样组织：

```text
triton-coralnpu/
  python/
    triton_coralnpu/
      __init__.py
      backend.py
      compiler.py
      target.py
      options.py
      runtime/
        __init__.py
        launcher.py
      lowering/
        passes.py
        legality.py
        coral_lowering.py
        rvv_lowering.py
        scalar_lowering.py
      debug/
        dump.py
  tools/
    emit_bazel_wrapper.py
    emit_standalone_launcher.py
  test/
    test_elementwise.py
    test_matmul.py
    test_rvv_fallback.py
  examples/
    vector_add.py
    matmul.py
```

如果短期内你希望直接放在当前仓库里孵化，也可以先建立一个更轻量的目录：

```text
third_party/triton_coralnpu_backend/
tools/triton_coralnpu/
tests/triton_backend/
examples/triton_backend/
```

但长期看，独立仓库会更健康。

## MVP 的最小算子集合

建议第一阶段只支持这三类：

- elementwise unary/binary
  例如 `add`、`mul`、`relu`
- contiguous vector/tensor load-store
- 小规模 matmul / dot

建议明确不支持：

- reduction 全家桶
- scan / sort
- atomic 的大范围支持
- 动态 shape kernel
- 复杂 block pointer
- 高级 pipeline primitive

原因很简单：这三类已经足够验证整条编译链、内存访问、算术 lowering 和 simulator 验证流程。

## MVP 的三条执行路径

从第一版开始，建议就定义三条后端路径：

1. `coral_native`
2. `rvv_fallback`
3. `scalar_fallback`

即使第一阶段只有 `rvv_fallback` 或 `scalar_fallback` 真正完成，也应先把接口层定义出来。

推荐理由：

- 上层 Triton frontend 保持统一
- 支持范围可以逐步扩展
- 不会把后端写死在单一路径上
- 后面更容易参考 `FlagTree` 走向多 backend 管理

## 路径一：`coral_native`

这是长期价值最高的路径，但不要求在 MVP 的第一个里程碑完整实现。

建议的第一阶段目标：

- 先识别最容易映射的固定形状 kernel
- 只支持少量已知可高效表达的模式
- 输出后端中间表示，哪怕还不直接生成最终最优代码

如果你当前还没有稳定的 Triton-to-Coral 原生指令 lowering，这条路径可以先做成：

- pattern match
- legality check
- 产出“暂不支持”的明确诊断

而不是强行半成品 codegen。

## 路径二：`rvv_fallback`

这是我最建议优先做通的路径。

原因：

- 当前仓库已经区分了 RVV simulator 目标，例如 [`rvv_core_mini_axi_sim`](/home/shaokai/Desktop/code/google-npu/coralnpu/tests/verilator_sim/BUILD:242)
- RVV 更接近 Triton 对 tile/vector 的表达方式
- 比 scalar 更能保留并行语义
- 更适合作为 matmul 和向量算子的过渡执行层

MVP 可以把很多 Triton kernel 先 lower 成 RVV-friendly 形式，再交给现有 Coral RVV 工具链产出 ELF。

## 路径三：`scalar_fallback`

这是保底路径。

建议用途：

- 数值正确性对照
- 编译链打通
- 不支持 RVV 的场景兜底
- 某些 legality 边界上的调试基线

它不追求性能，但非常适合保证“同一 Triton kernel 至少能运行”。

## 后端输出形式建议

第一版可以考虑三种输出形式，按实现难度从低到高排序：

1. 生成 C/C++ stub
2. 生成 LLVM IR / object
3. 直接生成 Coral 更原生的低层代码

我建议 MVP 先以“生成可纳入 Bazel 的 C/C++ 或 object”为主。

原因：

- 最容易复用 `coralnpu_v2_binary`
- 最容易和现有 simulator/ELF 路径结合
- 最容易做单步调试和 diff
- 不会一开始就把工作压死在复杂 codegen 上

## 与当前仓库的对接点

第一阶段至少建议利用以下现有能力。

## 对接点一：`coralnpu_v2_binary`

这是当前最自然的 ELF 生成入口。

参考：
- [`examples/BUILD.bazel`](/home/shaokai/Desktop/code/google-npu/coralnpu/examples/BUILD.bazel:19)
- [`rules/coralnpu_v2.bzl`](/home/shaokai/Desktop/code/google-npu/coralnpu/rules/coralnpu_v2.bzl:220)

建议做法：

- Triton backend 输出 `generated_kernel.cc`
- 用一个小的 Bazel target 包它
- 再由 `coralnpu_v2_binary` 生成 `.elf`

## 对接点二：Verilator simulator

这是最直接的回归验证路径。

参考：
- [`core_mini_axi_sim`](/home/shaokai/Desktop/code/google-npu/coralnpu/tests/verilator_sim/BUILD:222)
- [`rvv_core_mini_axi_sim`](/home/shaokai/Desktop/code/google-npu/coralnpu/tests/verilator_sim/BUILD:242)

建议做法：

- scalar/non-RVV kernel 优先走 `core_mini_axi_sim`
- RVV kernel 优先走 `rvv_core_mini_axi_sim`

## 对接点三：`hw_sim`

`hw_sim` 提供了 simulator library 和 wrapper，这意味着你后面不一定只能通过独立进程运行模拟器。

参考：
- [`core_mini_axi_simulator`](/home/shaokai/Desktop/code/google-npu/coralnpu/hw_sim/BUILD:68)
- [`libcoralnpu_simulator.so`](/home/shaokai/Desktop/code/google-npu/coralnpu/hw_sim/BUILD:111)

MVP 第一阶段不一定要立即用它，但建议在设计上预留：

- 后续可做 Python launcher
- 后续可做 standalone kernel harness
- 后续可做批量 kernel regression

## 推荐的 Bazel 集成方式

建议不要让 Triton backend 直接“入侵”现有 BUILD 文件，而是先采用生成式桥接。

推荐流程：

1. Triton backend 生成 `generated_kernel.cc`
2. Triton backend 生成一个小的 Bazel wrapper 文件
3. Bazel build 该 wrapper target
4. 生成 `.elf`

示意：

```text
tmp/
  generated/
    matmul_kernel.cc
    BUILD.bazel
```

`BUILD.bazel` 中只做最小包装，例如：

```python
load("//rules:coralnpu_v2.bzl", "coralnpu_v2_binary")

coralnpu_v2_binary(
    name = "matmul_kernel",
    srcs = ["matmul_kernel.cc"],
)
```

这样能把 Triton backend 和 `coralnpu` 主仓库规则边界保持清晰。

## 建议实现的核心接口

MVP 至少应实现以下模块。

## 1. Target 描述

需要一份统一 target 能力描述，例如：

- `supports_rvv`
- `supports_fp16`
- `supports_int8`
- `supports_native_matmul`
- `max_vector_bits`
- `preferred_tile_shapes`

这一步很重要，因为你迟早会有多个执行路径。

## 2. Legality Check

输入一个 Triton kernel 后，尽早判断：

- 是否支持
- 支持哪条路径
- 哪些限制不满足

输出必须明确，例如：

- `supported on rvv_fallback`
- `rejected: dynamic stride not supported`

## 3. Lowering Pipeline

MVP 的 lowering pipeline 至少要区分：

- 公共 lowering
- `coral_native` lowering
- `rvv_fallback` lowering
- `scalar_fallback` lowering

就算当前 `coral_native` 只做 stub，也应该留出模块边界。

## 4. Artifact Emitter

建议统一一个产物导出模块，负责：

- 导出 IR
- 导出 C/C++
- 导出 object
- 导出 standalone launcher
- 导出输入输出 tensor dump

这是后续调试的核心。

## 5. Simulator Runner

建议提供一个统一 runner，把：

- 生成 ELF
- 选择 simulator
- 运行 simulator
- 收集退出码
- 收集 stdout
- 收集 cycle

封成同一条工具路径。

## MVP 的最小调试能力

第一版至少提供这些开关：

- dump Triton IR
- dump lowering 后 IR
- dump generated C/C++
- dump Bazel wrapper
- dump final ELF path
- dump simulator stdout
- dump cycle result

如果没有这些调试材料，后端很难推进。

## MVP 的最小测试集合

建议第一阶段建立以下测试：

1. `vector_add`
2. `elementwise_relu`
3. `small_matmul`
4. `rvv_fallback_matmul`
5. `scalar_reference_matmul`

每个测试至少做三类校验：

- 编译成功
- 输出数值正确
- simulator 能正常退出

如果是性能基线测试，再额外记录：

- cycle

## 建议的 simulator 验证流程

推荐统一成下面的步骤：

1. 从 Triton kernel 生成 backend 产物
2. 产出 Bazel wrapper
3. Bazel 生成 ELF
4. 在对应 simulator 上运行
5. 读取 stdout / semihosting 输出
6. 比对参考结果
7. 记录 cycle

对于 non-RVV 路径：

```bash
bazel-bin/tests/verilator_sim/core_mini_axi_sim --binary <kernel>.elf
```

对于 RVV 路径：

```bash
bazel-bin/tests/verilator_sim/rvv_core_mini_axi_sim --binary <kernel>.elf
```

## 建议的性能验证策略

MVP 阶段只看两类性能数据：

- cycle
- 不同 lowering 路径之间的相对比较

不要在 MVP 阶段给出：

- TOPS
- 实芯片绝对时延结论
- 用宿主机 wall-clock 推导出的吞吐

Verilator/SystemC 的宿主机耗时不能作为芯片真实性能结论。

## 一个务实的里程碑划分

建议按以下 4 个里程碑推进。

## 里程碑 A：scalar 路径打通

完成标准：

- Triton elementwise kernel
- 生成可链接产物
- 生成 ELF
- 在 `core_mini_axi_sim` 跑通

## 里程碑 B：RVV 路径打通

完成标准：

- Triton matmul / vector kernel
- 落到 `rvv_fallback`
- 在 `rvv_core_mini_axi_sim` 跑通

## 里程碑 C：统一 runner 和调试导出

完成标准：

- 一条命令完成 compile + build + run + compare + cycle
- 能导出 IR / generated code / logs

## 里程碑 D：探索 `coral_native`

完成标准：

- 至少一个代表性 kernel 能走真正的 Coral 更原生路径
- 与 RVV/scalar 对比数值一致

## 关键风险

MVP 最大的几个风险如下：

- Triton 语义和 Coral 执行模型差距过大
- 过早承诺 `coral_native` 覆盖率
- 过早接入 PyTorch/Inductor
- 没有统一 legality check，导致调试困难
- 没有统一 artifact dump，导致问题不可定位

因此第一阶段最重要的不是“后端代码量”，而是：

- 范围控制
- 明确的路径分层
- 可验证的中间产物

## 最终建议

如果现在就要开始实现，我建议你按下面顺序干：

1. 先做 `scalar_fallback` 的 elementwise
2. 再做 `rvv_fallback` 的 matmul
3. 补齐 Bazel wrapper 生成和 simulator runner
4. 最后才去碰 `coral_native`

这是当前仓库条件下，风险最低、回报最高的启动顺序。
