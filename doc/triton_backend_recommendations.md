# Triton Backend Recommendations for CoralNPU

本文档总结如何为当前 `coralnpu` 仓库规划一个基于 OpenAI Triton language/compiler 的后端实现，并参考以下公开项目的设计思路：

- `verisilicon/triton-vsi-backend`
- `flagos-ai/FlagTree`
- `triton-lang/triton-ascend`
- `intel/intel-xpu-backend-for-triton`
- Triton 官方主仓库

## 目标

目标不是“让所有 Triton 程序立刻无修改跑在 CoralNPU 上”，而是分阶段建立一条可维护的后端路径：

1. 支持一个小而稳定的 Triton 子集
2. 产出可被当前 `coralnpu` 仓库消费的离线编译产物
3. 能在 simulator 或板级环境中验证功能和 cycle
4. 在此基础上再考虑 PyTorch / TorchInductor 集成

这条路线比直接追求完整 JIT 和完整生态集成更符合当前仓库的工具链形态。

## 参考项目的关键信号

## VeriSilicon 项目带来的启发

`triton-vsi-backend` 的 README 明确表明它是一个 Triton backend plugin，并依赖外部 SDK、编译器和运行时环境变量。这说明它不是简单修改 Triton 前端，而是围绕 Triton 的后端扩展点来做设备适配。

从它公开 README 可以提炼出几条非常实际的经验：

- 后端以插件形式存在，而不是一开始就侵入 Triton 主仓库
- 构建依赖外部设备 SDK 和专有编译组件
- 首先关注“单 kernel 编译和运行”
- 提供 standalone launcher 以脱离 Python runtime 调试编译结果
- TorchInductor 集成不是天然完备的，甚至需要修改 `torch/utils/_triton.py`

这对 CoralNPU 的意义很直接：

- 第一阶段应优先做 out-of-tree backend
- 应优先提供单 kernel 离线编译和独立运行能力
- 应尽早提供 standalone 调试路径
- 不要把 Inductor 当第一阶段目标

## Triton-Ascend 项目带来的启发

`triton-ascend` 的 README 展示的是更完整的平台化路线。它不只是“能编译”，还覆盖了：

- 版本兼容矩阵
- Quick Start
- Installation Guide
- Architecture Design
- Operator Development Guide
- Debugging Guide
- Profiling Guide
- Migration Guide

这说明当 Triton 后端走向产品化时，技术实现只是其中一部分，文档、兼容性、调试、迁移、性能方法论同样重要。

对 CoralNPU 来说，这意味着：

- 后端设计文档应从第一天就区分“功能支持”和“性能支持”
- 要尽早定义兼容矩阵
- 要单独准备调试和 profiling 路径
- 要有从 GPU Triton kernel 迁移到 CoralNPU Triton kernel 的约束说明

## FlagTree 项目带来的启发

`FlagTree` 的定位和前面几个项目不太一样。它不是单一设备后端，而是一个“统一多 AI 芯片后端”的 Triton 分支体系。它公开强调的目标包括：

- 单仓库多后端支持
- 尽量兼容已有适配方案
- 为上游模型用户提供统一编译能力
- 为下游芯片厂商提供 Triton 接入范例

它的一个非常重要的信号是：多后端并不是简单在同一个分支里堆代码，而是明确按 Triton 版本和 backend 组织维护分支、安装方式和 CI/CD。

从公开 README 能看到这些值得注意的点：

- 一个仓库里维护多个 backend 和多个 Triton 版本分支
- 为每个 backend 提供独立安装入口和 CI/CD
- 开始引入统一抽象，例如 `UnifiedHardware manager`
- 在标准 Triton 之外，又引入了更面向硬件控制的扩展层，例如 `TLE`、`FLIR`、`HINTS`

这对 CoralNPU 的启发不是“立刻照搬 FlagTree”，而是：

- 如果未来 CoralNPU 希望同时维护 NPU / RVV / scalar 多条后端路径，FlagTree 式的统一编排会比单后端散装维护更健康
- 如果未来需要在 Triton 语言上表达更多硬件特性，可能需要类似 `TLE` 这样的分层扩展思路
- 如果未来希望降低不同硬件执行路径之间的迁移成本，应从一开始就设计统一 backend 能力模型，而不是每条路径各自发展

但对 CoralNPU 当前阶段，我不建议直接把 FlagTree 当成第一阶段实现模板。原因是它更像“多后端平台化后的组织方式”，而不是最小可行后端的起点。

## Intel XPU 项目带来的启发

Intel 的 `intel-xpu-backend-for-triton` 是一个公开的 out-of-tree backend，并且将 Triton 后端作为独立仓库演进。这再次说明：

- out-of-tree backend 是现实可行路线
- 后端往往需要和特定 toolchain / runtime 深度绑定
- 真正进入 PyTorch 生态时，版本协同会成为重要成本

## 官方 Triton 项目带来的约束

Triton 官方仓库当前主流后端仍然围绕成熟设备栈来构建，并通过统一后端抽象组织编译链路。因此对 CoralNPU 这类嵌入式 / NPU 目标，更合理的做法是：

- 利用 Triton 前端和中间优化能力
- 尽量避免过早依赖 Triton 假设中的 GPU 运行时模型
- 优先把 Triton 当作“kernel DSL + compiler frontend”

## 对 CoralNPU 的总体建议

## 建议一：第一阶段只做 AOT，不做完整 JIT

当前 `coralnpu` 仓库天然更适合离线编译：

- 通过 Bazel 构建
- 通过已有交叉工具链生成 ELF
- 通过 simulator 或板级环境运行

因此第一阶段建议是：

```text
Triton kernel
-> Triton IR / MLIR
-> CoralNPU-specific lowering
-> object / assembly / intermediate artifact
-> 当前仓库工具链链接为 ELF
-> simulator / board
```

不建议第一阶段追求：

- Python 侧即时加载到设备
- PyTorch 无感知运行
- 完整 TorchInductor 自动接管

## 建议二：采用 out-of-tree backend

从 `triton-vsi-backend`、`triton-ascend`、`intel-xpu-backend-for-triton` 的方向看，CoralNPU 后端最适合独立演进。

原因：

- 便于快速试错
- 便于绑定 CoralNPU 私有工具链和 ABI
- 不会被 Triton 主线频繁变更立即拖垮
- 更容易维护本仓库自己的测试基线

如果未来成熟，再考虑上游化某些公共能力。

如果未来后端数量增加，例如同时长期维护：

- Coral NPU 原生后端
- RVV 后端
- scalar fallback
- simulator-only reference backend

那么可以进一步参考 `FlagTree` 的组织方式，把“多 backend 管理”提升为一级设计问题。

## 建议三：先支持一个很小的 Triton 子集

建议第一版只支持：

- 静态 shape
- contiguous memory
- `tl.load`
- `tl.store`
- 简单 elementwise
- 小范围 matmul / dot
- 少量 dtype：`fp32`、`fp16`、`int8`

先不要承诺：

- 动态 shape 大量覆盖
- 复杂 layout
- 高级 async pipeline
- 所有 atomic / scan / sort / block pointer 能力
- 所有 Triton Python API

这是最重要的范围控制。

## 建议四：明确执行目标层级

CoralNPU 后端不一定只有一种最终执行路径。建议一开始就允许多级降级：

1. 直接映射到 CoralNPU 专用算子 / 指令路径
2. 映射不到时降到 RVV
3. 再不行降到 scalar

这样你可以：

- 先实现可用性
- 再逐步提升 NPU 覆盖率
- 保持单一 Triton frontend，不必一开始就解决所有 lowerings

这部分其实最接近 `FlagTree` 的长期思路，只不过 CoralNPU 在第一阶段不需要把它做成一个“大一统平台”，只需要先把多路径执行模型设计正确。

## 建议五：优先建立 standalone 调试链路

这是 `triton-vsi-backend` 最值得借鉴的点之一。

建议你的后端在第一阶段就支持导出：

- lowering 后的 MLIR
- LLVM IR 或等价低层 IR
- 最终 codegen 产物
- standalone launcher
- 输入输出 tensor dump

这可以大幅降低排查难度，尤其适合当前 `coralnpu` 这种 simulator 优先的开发环境。

## 建议六：性能评估应以 cycle 为主

对 CoralNPU 而言，Verilator/SystemC 或类似环境中的宿主机 wall-clock 并不能代表芯片真实性能。

建议后端 autotuning 或 benchmark 优先使用：

- simulator cycle
- 板上 cycle counter
- 硬件 performance counter

不要把“主机上运行几秒”直接当成性能结论。

## 建议七：尽早建立 legality check

很多 Triton 程序默认带有 GPU 语义假设。CoralNPU 后端应在较早阶段就显式拒绝不支持模式，而不是继续模糊 lowering。

建议尽早检查：

- shape / stride 约束
- dtype 约束
- 地址空间约束
- 内存布局约束
- 不支持的控制流和 primitive
- 是否要求设备端 runtime 特性

让用户尽快得到“为什么不支持”的结论，比后期 codegen 崩溃要好得多。

## 建议八：把文档当成后端的一部分

`triton-ascend` 和 `FlagTree` 的经验都说明，后端一旦超过 demo 阶段，就必须补齐这些内容：

- 安装说明
- 版本兼容表
- Quick Start
- 调试指南
- Profiling 指南
- 从 GPU Triton kernel 迁移到本后端的约束说明
- backend / Triton 版本兼容矩阵
- 不同执行路径的支持范围说明

对 CoralNPU 项目，这些文档甚至应该和代码同步演进。

## 一个务实的实施顺序

推荐按下面 4 个阶段推进。

## 阶段 1：单 kernel AOT 打通

目标：

- 从 Triton kernel 生成 Coral 可消费的中间产物
- 能链接成 ELF
- 能在 simulator 跑通

验收建议：

- `vector add`
- `matmul`
- 简单 activation / elementwise

## 阶段 2：建立多后端 fallback

目标：

- 能在 NPU / RVV / scalar 三层中择优
- 为不支持算子提供明确退化路径

验收建议：

- 相同 Triton kernel 在多条后端路径上都能落地

## 阶段 3：完善调试与 profiling

目标：

- dump IR
- dump standalone launcher
- dump inputs / outputs
- 给出 cycle 指标

验收建议：

- 能把某个 kernel 的编译失败或数值错误快速定位到具体 lowering 阶段

## 阶段 4：再考虑 PyTorch / Inductor

目标：

- 在稳定后端基础上探索 TorchInductor 对接
- 明确哪些路径需要 patch、plugin 或上游支持

这里应特别注意 `triton-vsi-backend` README 中提到的现实问题：Inductor 对 out-of-tree backend 的接纳并不天然完整，可能需要额外适配。

如果 CoralNPU 后端未来扩展为多设备或多执行路径统一维护，则在这一阶段可以继续吸收 `FlagTree` 的经验：

- 把后端能力注册抽象化
- 把版本兼容和安装入口模块化
- 把 backend-specific extension 收束到统一命名和统一入口下

## 针对当前 `coralnpu` 仓库的具体落点

建议优先复用本仓库已经具备的能力：

- Bazel 构建系统
- CoralNPU 交叉工具链
- example ELF 生成路径
- Verilator simulator 运行路径
- RVV / scalar / variant 目标组织方式

一条更现实的集成路径是：

```text
Triton frontend
-> CoralNPU backend plugin
-> 生成当前仓库可链接/可封装的目标文件
-> Bazel 产出 ELF
-> simulator 运行并统计 cycle
```

而不是：

```text
PyTorch eager/Inductor
-> Triton JIT
-> 直接设备 runtime 加载
```

后者太早，风险和耦合都偏高。

## 最终建议

如果你现在要正式启动 CoralNPU Triton 后端，最好的第一步不是“写完整 backend”，而是先定义一个最小可行子集：

- 2 到 3 个代表性 kernel
- 1 条 AOT 编译链
- 1 套 simulator cycle 验证方法
- 1 份 legality / unsupported feature 规则

只要这四件事先站住，后面的 Triton 后端演进就会健康很多。

而 `FlagTree` 更适合作为第二阶段和第三阶段的参考对象：

- 当你开始维护多个 backend 路径时，参考它的统一组织方式
- 当你开始引入 backend-specific language extension 时，参考它的分层扩展思路
- 当你开始把单后端原型升级为可持续演进的平台时，参考它的版本矩阵和 CI 结构

## 参考链接

- Triton 官方仓库  
  https://github.com/triton-lang/triton

- VeriSilicon Triton backend README  
  https://github.com/verisilicon/triton-vsi-backend

- FlagTree  
  https://github.com/flagos-ai/FlagTree

- Triton-Ascend 仓库  
  https://github.com/triton-lang/triton-ascend

- Intel XPU backend for Triton  
  https://github.com/intel/intel-xpu-backend-for-triton
