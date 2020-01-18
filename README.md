# NVIDIA CUDA COURSE

此笔记是根据NVINDA CUDA COURSE总结而成的。

## Chapter 1: 
[使用 CUDA C/C++ 加速应用程序](https://github.com/jokmingwong/NVIDIA_CUDA_COURSE/tree/master/Chap1)

加速计算正在取代 CPU 计算，成为最佳计算做法。加速计算带来的层出不穷的突破性进展、对加速应用程序日益增长的需求、轻松编写加速计算的编程规范以及支持加速计算的硬件的不断改进，所有这一切都在推动计算方式必然会过渡到加速计算。

无论是从出色的性能还是易用性来看，`CUDA`计算平台均是加速计算的制胜法宝。CUDA 提供一种可扩展 C、C++、Python 和 Fortran 等语言的编码范式，能够在世界上性能超强劲的并行处理器 NVIDIA GPU 上运行大量经加速的并行代码。CUDA 可以毫不费力地大幅加速应用程序，具有适用于 `DNN`、`BLAS`、`图形分析`和 `FFT` 等的高度优化库生态系统，并且还附带功能强大的 `命令行` 和 `可视化分析器`。

学习 CUDA 将能助您加速自己的应用程序。加速应用程序的执行速度远远超过 CPU 应用程序，并且可以执行 CPU 应用程序受限于其性能而无法执行的计算。在本实验中, 您将学习使用 CUDA C/C++ 为加速应用程序编程的入门知识，这些入门知识足以让您开始加速自己的 CPU 应用程序以获得性能提升并助您迈入全新的计算领域。

### Prerequisites

To get the most out of this lab you should already be able to:

- 在 C 中声明变量、编写循环并使用 if/else 语句。
- 在 C 中定义和调用函数。
- 在 C 中分配数组。
---
## Chapter 2: 
[使用 CUDA C/C++ 统一内存和 nvprof 管理加速应用程序内存](https://github.com/jokmingwong/NVIDIA_CUDA_COURSE/tree/master/Chap2)

对于本实验和其他 CUDA 基础实验，我们强烈建议您遵循 [*CUDA 最佳实践指南*](http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)，其中推荐一种称为 **APOD** 的设计周期：**评估**、**并行化**、**优化**和**部署**。简言之，APOD 规定一个迭代设计过程，开发人员能够在该过程中对其加速应用程序性能施以渐进式改进，并发布代码。随着开发人员的 CUDA 编程能力愈渐增强，他们已能在加速代码库中应用更先进的优化技术。

本实验将支持这种迭代开发风格。您将使用 **NVIDIA 命令行分析器**定性衡量应用程序性能及确定优化机会，之后您将应用渐进式改进，最后您会学习新技术并重复该周期。需重点关注的是，您将在本实验中学习及应用的众多技术均会涉及 CUDA **统一内存**工作原理的具体细节。理解统一内存行为是 CUDA 开发人员的一项基本技能，同时也可作为多项更先进内存管理技术的先决条件。


### Prerequisites

如要充分利用本实验，您应已能胜任如下任务：

- 编写、编译及运行既可调用 CPU 函数也可启动 GPU 核函数的 C/C++ 程序。
- 使用执行配置控制并行线程层次结构。
- 重构串行循环以在 GPU 上并行执行其迭代。
- 分配和释放统一内存。
---

## Chapter 3: 
[异步流及使用 CUDA C/C++ 对加速应用程序开展可视化分析](https://github.com/jokmingwong/NVIDIA_CUDA_COURSE/tree/master/Chap3)



