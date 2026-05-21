# Verilator Simulator

本文档说明 `tests/verilator_sim` 下 Verilator 仿真器的编译条件、常用构建命令和运行参数。

## 适用范围

本文主要针对这些 Bazel 目标：

- `//tests/verilator_sim:core_mini_axi_sim`
- `//tests/verilator_sim:rvv_core_mini_axi_sim`
- `//tests/verilator_sim:core_mini_highmem_axi_sim`
- `//tests/verilator_sim:core_mini_verification_axi_sim`
- `//tests/verilator_sim:core_mini_itcm512kb_dtcm512kb_axi_sim`
- `//tests/verilator_sim:rvv_core_mini_highmem_axi_sim`
- `//tests/verilator_sim:rvv_core_mini_verification_axi_sim`
- `//tests/verilator_sim:rvv_core_mini_itcm512kb_dtcm512kb_axi_sim`

这些目标定义在 [tests/verilator_sim/BUILD](/home/shaokai/Desktop/code/google-npu/coralnpu/tests/verilator_sim/BUILD:222)。

## 编译条件

构建前至少需要满足以下条件：

- Bazel `7.4.1`
- Python `3.9` 到 `3.12`
- `SRecord`
- 可用的主机 C/C++ 编译环境
- 首次构建时可访问外网下载 Bazel 依赖和工具链

其中顶层仓库给出的基础要求见 [README.md](/home/shaokai/Desktop/code/google-npu/coralnpu/README.md:25)。

另外，这个仓库会在 `WORKSPACE` 中下载 Coral NPU 交叉工具链和其他第三方依赖，因此首次构建通常需要联网；例如 `toolchain_coralnpu_v2` 是通过 `http_archive` 拉取的，见 [WORKSPACE](/home/shaokai/Desktop/code/google-npu/coralnpu/WORKSPACE:154)。

## 构建说明

推荐的最小构建流程如下。

先编译一个可运行的 Coral ELF：

```bash
bazel build //examples:coralnpu_v2_hello_world_add_floats
```

再编译 non-RVV 仿真器：

```bash
bazel build //tests/verilator_sim:core_mini_axi_sim
```

如果要跑 RVV 程序，编译 RVV 仿真器：

```bash
bazel build //tests/verilator_sim:rvv_core_mini_axi_sim
```

README 中也给出了最小示例，见 [README.md](/home/shaokai/Desktop/code/google-npu/coralnpu/README.md:31)。

## 当前环境注意事项

如果当前环境下 Bazel 默认输出目录不可写，或 Bazel 常驻 server 运行不稳定，可以使用下面这组更稳妥的命令：

```bash
bazel --batch --output_user_root=/tmp/bazel_coralnpu build //examples:coralnpu_v2_hello_world_add_floats
bazel --batch --output_user_root=/tmp/bazel_coralnpu build //tests/verilator_sim:core_mini_axi_sim
```

这不是仓库强制要求，而是当前受限环境下更稳定的运行方式。

## 运行命令

构建完成后，可以直接运行 simulator：

```bash
bazel-bin/tests/verilator_sim/core_mini_axi_sim \
  --binary bazel-out/k8-fastbuild-ST-dd8dc713f32d/bin/examples/coralnpu_v2_hello_world_add_floats.elf
```

如果使用 RVV 版本：

```bash
bazel-bin/tests/verilator_sim/rvv_core_mini_axi_sim \
  --binary bazel-out/k8-fastbuild-ST-dd8dc713f32d/bin/<path>/<your_rvv_program>.elf
```

`--binary` 是必选参数；如果不传，程序会直接报错退出，见 [core_mini_axi_sim.cc](/home/shaokai/Desktop/code/google-npu/coralnpu/tests/verilator_sim/coralnpu/core_mini_axi_sim.cc:94)。

## RVV 程序如何跑起来

RVV 程序不是用 `core_mini_axi_sim` 跑，而是要走单独的 RVV simulator：

```bash
bazel build //tests/verilator_sim:rvv_core_mini_axi_sim
```

然后准备一个包含 RVV 指令的 ELF，再把它交给 `rvv_core_mini_axi_sim`。

仓库里已经有现成的 RVV 例子和测试程序，例如：

- [`//tests/cocotb/rvv:rvv_add`](/home/shaokai/Desktop/code/google-npu/coralnpu/tests/cocotb/rvv/BUILD:19)
- [`//tests/cocotb/rvv:vslide`](/home/shaokai/Desktop/code/google-npu/coralnpu/tests/cocotb/rvv/BUILD:19)
- [`//tests/cocotb/rvv/arithmetics:template_add_int8_m1`](/home/shaokai/Desktop/code/google-npu/coralnpu/tests/cocotb/rvv/arithmetics/BUILD:84)

一个最小可执行示例：

```bash
bazel build //tests/cocotb/rvv:rvv_add
bazel build //tests/verilator_sim:rvv_core_mini_axi_sim

bazel-bin/tests/verilator_sim/rvv_core_mini_axi_sim \
  --binary bazel-out/k8-fastbuild-ST-dd8dc713f32d/bin/tests/cocotb/rvv/rvv_add.elf
```

另一个使用 C/C++ 模板生成的 RVV 算术测试示例：

```bash
bazel build //tests/cocotb/rvv/arithmetics:template_add_int8_m1

bazel-bin/tests/verilator_sim/rvv_core_mini_axi_sim \
  --binary bazel-out/k8-fastbuild-ST-dd8dc713f32d/bin/tests/cocotb/rvv/arithmetics/template_add_int8_m1.elf
```

实际运行模型上，RVV simulator 和 non-RVV simulator 的入口参数一致，区别只在于：

- 程序本身必须是 RVV 版本 ELF
- simulator 必须选择 `rvv_core_mini_axi_sim`

如果拿 RVV ELF 去跑 `core_mini_axi_sim`，或者拿 non-RVV 假设的程序去跑 RVV simulator，结果通常没有意义。

## 支持的运行参数

`core_mini_axi_sim` 支持以下命令行参数，定义见 [core_mini_axi_sim.cc](/home/shaokai/Desktop/code/google-npu/coralnpu/tests/verilator_sim/coralnpu/core_mini_axi_sim.cc:36)：

- `--binary=<path>`
  作用：指定待运行的 ELF 文件。必选。
- `--cycles=<N>`
  作用：设置最大仿真周期数。
  默认值：`100000000`
- `--trace`
  作用：打开波形导出。
- `--debug_axi`
  作用：打印 AXI 访问调试信息。
- `--instr_trace`
  作用：将指令执行轨迹打印到控制台。
- `--backdoor_load`
  作用：对 TCM 区域启用更快的 backdoor 装载方式。

`rvv_core_mini_axi_sim` 复用同一份入口程序和参数定义，因此参数集合与 `core_mini_axi_sim` 相同。

在当前仓库版本中，simulator 正常结束前还会打印一行总周期数：

```text
Simulation cycles: <N>
```

这对 non-RVV 和 RVV simulator 都适用。

示例：

```bash
bazel-bin/tests/verilator_sim/core_mini_axi_sim \
  --binary bazel-bin/examples/coralnpu_v2_hello_world_add_floats.elf \
  --cycles=2000000 \
  --instr_trace \
  --backdoor_load
```

## 这个仿真器实际支持什么

这个 simulator 不是完整 SoC 模型，而是面向 `core_mini_axi` 平台的功能仿真器。

当前测试平台至少包含这些模型：

- 一块从 `0x20000000` 开始的内存，大小 `4 MiB`
- 一个 UART 地址 `0x54000000`

见 [tests/systemc/Xbar.h](/home/shaokai/Desktop/code/google-npu/coralnpu/tests/systemc/Xbar.h:146)。

程序加载方式支持 ELF 装载，并会查找 `tohost`、`fromhost`、`tohost_ready`、`fromhost_ready` 等符号来完成 semihosting/退出协议，见 [core_mini_axi_tb.cc](/home/shaokai/Desktop/code/google-npu/coralnpu/tests/verilator_sim/coralnpu/core_mini_axi_tb.cc:361)。

当前实现支持的宿主侧 syscall 主要包括：

- `openat`
- `close`
- `lseek`
- `read`
- `write`
- `pread`
- `pwrite`
- `exit`
- `getmainvars`

见 [core_mini_axi_tb.cc](/home/shaokai/Desktop/code/google-npu/coralnpu/tests/verilator_sim/coralnpu/core_mini_axi_tb.cc:530)。

因此它更适合：

- 跑裸机或 semihosting 风格 ELF
- 做功能验证
- 做 cycle 级相对比较

不应直接把宿主机上的运行秒数当作真实芯片性能。

## 常用命令清单

编译 example：

```bash
bazel build //examples:coralnpu_v2_hello_world_add_floats
```

编译 non-RVV simulator：

```bash
bazel build //tests/verilator_sim:core_mini_axi_sim
```

编译 RVV simulator：

```bash
bazel build //tests/verilator_sim:rvv_core_mini_axi_sim
```

编译一个最小 RVV ELF：

```bash
bazel build //tests/cocotb/rvv:rvv_add
```

运行 non-RVV example：

```bash
bazel-bin/tests/verilator_sim/core_mini_axi_sim \
  --binary bazel-out/k8-fastbuild-ST-dd8dc713f32d/bin/examples/coralnpu_v2_hello_world_add_floats.elf
```

运行 RVV example：

```bash
bazel-bin/tests/verilator_sim/rvv_core_mini_axi_sim \
  --binary bazel-out/k8-fastbuild-ST-dd8dc713f32d/bin/tests/cocotb/rvv/rvv_add.elf
```

带指令跟踪运行：

```bash
bazel-bin/tests/verilator_sim/core_mini_axi_sim \
  --binary bazel-out/k8-fastbuild-ST-dd8dc713f32d/bin/examples/coralnpu_v2_hello_world_add_floats.elf \
  --instr_trace
```

带指令跟踪运行 RVV example：

```bash
bazel-bin/tests/verilator_sim/rvv_core_mini_axi_sim \
  --binary bazel-out/k8-fastbuild-ST-dd8dc713f32d/bin/tests/cocotb/rvv/rvv_add.elf \
  --instr_trace
```

带波形运行：

```bash
bazel-bin/tests/verilator_sim/core_mini_axi_sim \
  --binary bazel-out/k8-fastbuild-ST-dd8dc713f32d/bin/examples/coralnpu_v2_hello_world_add_floats.elf \
  --trace
```
