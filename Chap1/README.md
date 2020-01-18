
# 使用 CUDA C/C++ 加速应用程序

加速计算正在取代 CPU 计算，成为最佳计算做法。加速计算带来的层出不穷的突破性进展、对加速应用程序日益增长的需求、轻松编写加速计算的编程规范以及支持加速计算的硬件的不断改进，所有这一切都在推动计算方式必然会过渡到加速计算。

无论是从出色的性能还是易用性来看，`CUDA`计算平台均是加速计算的制胜法宝。CUDA 提供一种可扩展 C、C++、Python 和 Fortran 等语言的编码范式，能够在世界上性能超强劲的并行处理器 NVIDIA GPU 上运行大量经加速的并行代码。CUDA 可以毫不费力地大幅加速应用程序，具有适用于 `DNN`、`BLAS`、`图形分析`和 `FFT` 等的高度优化库生态系统，并且还附带功能强大的 `命令行` 和 `可视化分析器`。

学习 CUDA 将能助您加速自己的应用程序。加速应用程序的执行速度远远超过 CPU 应用程序，并且可以执行 CPU 应用程序受限于其性能而无法执行的计算。在本实验中, 您将学习使用 CUDA C/C++ 为加速应用程序编程的入门知识，这些入门知识足以让您开始加速自己的 CPU 应用程序以获得性能提升并助您迈入全新的计算领域。

---
## 如要充分利用本实验,您应已能胜任如下任务：

To get the most out of this lab you should already be able to:

- 在 C 中声明变量、编写循环并使用 if/else 语句。
- 在 C 中定义和调用函数。
- 在 C 中分配数组。

无需 CUDA 预备知识。

---
## Objectives

当您在本实验完成学习后，您将能够：

- 编写、编译及运行既可调用 CPU 函数也可**启动** GPU **核函数*的 C/C++ 程序。
- 使用**执行配置**控制并行**线程层次结构**。
- 重构串行循环以在 GPU 上并行执行其迭代。
- 分配和释放可用于 CPU 和 GPU 的内存。
- 处理 CUDA 代码生成的错误。
- 加速 CPU 应用程序。


---
## Accelerated Systems

*加速系统*又称*异构系统*，由 CPU 和 GPU 组成。加速系统会运行 CPU 程序，这些程序也会转而启动将受益于 GPU 大规模并行计算能力的函数。本实验环境是一个包含 NVIDIA GPU 的加速系统。可以使用 `nvidia-smi` (*Systems Management Interface的缩写*) 命令行命令查询有关此 GPU 的信息。现在，可以在下方的代码执行单元上使用 `CTRL` + `ENTER` 发出 `nvidia-smi` 命令。无论您何时需要执行代码，均可在整个实验中找到这些单元。代码运行后，运行该命令的输出将打印在代码执行单元的正下方。在运行下方的代码执行块后，请注意在输出中找到并记录 GPU 的名称。


```shell
nvidia-smi
```

---
## GPU-accelerated Vs. CPU-only Applications
在`CPU应用程序`中，数据在CPU上进行分配，并且所有工作都在CPU上执行。

而在`加速应用程序`中，可以使用`cudaMallocManaged()`分配数据。其数据可由CPU进行访问和处理，并能自动迁移至可执行并行工作的GPU。GPU和CPU是异步工作的，通过`cudaDeviceSynchronize()`,CPU代码可与异步 GPU工作实现同步，并等待后者完成。经CPU访问的数据会自动迁移。

![CUDA和CPU对比](https://pic1.xuehuaimg.com/proxy/csdn/https://img-blog.csdn.net/20160328220721465)

---
## Writing Application Code for the GPU

CUDA 为许多常用编程语言提供扩展，而在本实验中，我们将会为 C/C++ 提供扩展。这些语言扩展可让开发人员在 GPU 上轻松运行其源代码中的函数。

以下是一个 `.cu` 文件（`.cu` 是 CUDA 加速程序的文件扩展名）。其中包含两个函数，第一个函数将在 CPU 上运行，第二个将在 GPU 上运行。请抽点时间找出这两个函数在定义方式和调用方式上的差异。

```cpp
void CPUFunction()
{
  printf("This function is defined to run on the CPU.\n");
}

__global__ void GPUFunction()
{
  printf("This function is defined to run on the GPU.\n");
}

int main()
{
  CPUFunction();

  GPUFunction<<<1, 1>>>();
  cudaDeviceSynchronize();
}
```

以下是一些需要特别注意的重要代码行，以及加速计算中使用的一些其他常用术语：

`__global__ void GPUFunction()`
  - `__global__` 关键字表明以下函数将在 GPU 上运行并可**全局**调用，而在此种情况下，则指由 CPU 或 GPU 调用。
  - 通常，我们将在 CPU 上执行的代码称为**主机**代码，而将在 GPU 上运行的代码称为**设备**代码，即`__device__`。
  - 注意返回类型为 `void`。使用 `__global__` 关键字定义的函数需要返回 `void` 类型。

`GPUFunction<<<1, 1>>>();`
  - 通常，当调用要在 GPU 上运行的函数时，我们将此种函数称为**已启动**的**核函数**。
  - 启动核函数时，我们必须提供**执行配置**，即在向核函数传递任何预期参数之前使用 `<<< ... >>>` 语法完成的配置。
  - 在宏观层面，程序员可通过执行配置为核函数启动指定**线程层次结构**，从而定义线程组（称为**线程块**）的数量，以及要在每个线程块中执行的**线程**数量。稍后将在本实验深入探讨执行配置，但现在请注意正在使用包含 `1` 线程（第二个配置参数）的 `1` 线程块（第一个执行配置参数）启动核函数。

`cudaDeviceSynchronize();`
  - 与许多 C/C++ 代码不同，核函数启动方式为**异步**：CPU 代码将继续执行*而无需等待核函数完成启动*。
  - 调用 CUDA 运行时提供的函数 `cudaDeviceSynchronize` 将导致主机 (CPU) 代码暂作等待，直至设备 (GPU) 代码执行完成，才能在 CPU 上恢复执行。

---
### Exercise: Write a Hello GPU Kernel

`01-hello-gpu`文件夹下的`01-hello-gpu.cu`包含已在运行的程序。其中包含两个函数，都有打印 `”Hello from the CPU\”` 消息。您的目标是重构源文件中的 `helloGPU()` 函数，以便该函数实际上在 GPU 上运行，并打印指示执行此操作的消息。参考答案也在同一个文件夹下面


```shell
nvcc -arch=sm_70 -o hello-gpu 01-hello/01-hello-gpu.cu -run
```

成功重构 `01-hello-gpu.cu` 后，请进行以下修改，并尝试在每次更改后编译并运行该应用程序。
若出现错误，请花时间仔细阅读错误内容：熟知错误内容会在您开始编写自己的加速代码时，给与很大的帮助。

- 从核函数定义中删除关键字 `__global__`。注意错误中的行号：您认为错误中的 `configured` 是什么意思？完成后，请替换 `__global__`。
- 移除执行配置：您对 `configured` 的理解是否仍旧合理？完成后，请替换执行配置。
- 移除对 `cudaDeviceSynchronize()` 的调用。在编译和运行代码之前，猜猜会发生什么情况，可以回顾一下核函数采取的是异步启动，且 `cudaDeviceSynchronize()` 会使主机执行暂作等待，直至核函数执行完成后才会继续。完成后，请替换对 `cudaDeviceSynchronize()` 的调用。
- 重构 `01-hello-gpu.cu`，以便 `Hello from the GPU` 在 `Hello from the CPU` **之前**打印。
- 重构 `01-hello-gpu.cu`，以便 `Hello from the GPU` 打印**两次**，一次是在 `Hello from the CPU` **之前**，另一次是在 `Hello from the CPU` **之后**。


---
### Compiling and Running Accelerated CUDA Code

本节包含上述为编译和运行 `.cu` 程序而调用的`nvcc` 命令的详细信息。
运行程序需要有可使用的cuda环境

曾使用过 `gcc` 的用户会对 `nvcc` 感到非常熟悉。例如，编译 `some-CUDA.cu` 文件就很简单：

`nvcc -arch=sm_70 -o out some-CUDA.cu -run`
  - `nvcc` 是使用 `nvcc` 编译器的命令行命令。
  - 将 `some-CUDA.cu` 作为文件传递以进行编译。
  - `o` 标志用于指定编译程序的输出文件。
  - `arch` 标志表示该文件必须编译为哪个**架构**类型。本示例中，`sm_70` 将用于专门针对本实验运行的 Volta GPU 进行编译，但有意深究的用户可以参阅有关 [`arch` 标志](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation)、[虚拟架构特性] (http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list) 和 [GPU特性](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list) 的文档。

  - 为方便起见，提供 `run` 标志将执行已成功编译的二进制文件。

---
## CUDA Thread Hierarchy
GPU在线程(thread)中执行工作。线程的集合叫做块(block)，块的数量很多，给的核函数启动相关联的块的集合称为网格(grid)。GPU函数为核函数，其通过执行配置启动并定义了网格的块数、每个块的线程数。在网格中，每个块包含相同数量的线程。


---
## Launching Parallel Kernels

程序员可通过执行配置指定有关如何启动核函数以在多个 GPU **线程**中并行运行的详细信息。更准确地说，程序员可通过执行配置指定线程组（称为**线程块**或简称为**块**）数量以及其希望每个线程块所包含的线程数量。执行配置的语法如下：

`<<< NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>`

**启动核函数时，核函数代码由每个已配置的线程块中的每个线程执行**。

因此，如果假设已定义一个名为 `someKernel` 的核函数，则下列情况为真：
  - `someKernel<<<1, 1>>()` 配置为在具有单线程的单个线程块中运行后，将只运行一次。
  - `someKernel<<<1, 10>>()` 配置为在具有 10 线程的单个线程块中运行后，将运行 10 次。
  - `someKernel<<<10, 1>>()` 配置为在 10 个线程块（每个均具有单线程）中运行后，将运行 10 次。
  - `someKernel<<<10, 10>>()` 配置为在 10 个线程块（每个均具有 10 线程）中运行后，将运行 100 次。

---
### Exercise: Launch Parallel Kernels

`01-first-parallel.cu`目前已作出十分基本的函数调用，即打印消息 `This should be running in parallel`。请按下列步骤首先进行重构使之在 GPU 上运行，然后在单个线程块中并行运行，最后则在多个线程块中运行。如您遇到问题，请参阅 `01-basic-parallel-solution.cu`。

- 重构 `firstParallel` 函数以便在 GPU 上作为 CUDA 核函数启动。在使用下方 `nvcc` 命令编译和运行 `01-basic-parallel.cu` 后，您应仍能看到函数的输出。
- 重构 `firstParallel` 核函数以便在 5 个线程中并行执行，且均在同一个线程块中执行。在编译和运行代码后，您应能看到输出消息已打印 5 次。
- 再次重构 `firstParallel` 核函数，并使其在 5 个线程块内并行执行（每个线程块均包含 5 个线程）。编译和运行之后，您应能看到输出消息现已打印 25 次。


```shell
nvcc -arch=sm_70 -o basic-parallel basic-parallel/01-basic-parallel.cu -run
```

---

## CUDA-Provided Thread Hierarchy Variables
- `gridDim.x`是网格中的块数
- `blockIdx.x`是网格中当前块的索引
- `blockDim.x`描述块中的线程数，网格中的所有块包含数量相同的线程，注意，没有`threadDim`这种东西
- `threadIdx.x`描述块中线程的索引



---
## Thread and Block Indices

每个线程在其线程块内部均会被分配一个索引，从 `0` 开始。此外，每个线程块也会被分配一个索引，并从 `0` 开始。正如线程组成线程块，线程块又会组成**网格**，而网格是 CUDA 线程层次结构中`级别最高`的实体。简言之，CUDA 核函数在由一个或多个线程块组成的网格中执行，且每个线程块中均包含相同数量的一个或多个线程。

CUDA 核函数可以访问能够识别如下两种索引的特殊变量：正在执行核函数的线程（位于线程块内）索引和线程所在的线程块（位于网格内）索引。这两种变量分别为 `threadIdx.x` 和 `blockIdx.x`。

---
### Exercise: Use Specific Thread and Block Indices

目前，`01-thread-and-block-idx.cu`文件包含一个正在打印失败消息的执行中的核函数。打开文件以了解如何更新执行配置，以便打印成功消息。重构后，请使用下方代码执行单元编译并运行代码以确认您的工作。如您遇到问题，请参阅 `01-thread-and-block-idx-solution.cu`。


```shell
!nvcc -arch=sm_70 thread-and-block-idx/01-thread-and-block-idx.cu -run
```

---
## Accelerating For Loops

对 CPU 应用程序中的循环进行加速的时机已经成熟：我们并非要顺次运行循环的每次迭代，而是让每次迭代都在自身线程中并行运行。考虑以下“for 循环”，尽管很明显，但还是请注意它控制着循环将执行的次数，并会界定循环的每次迭代将会发生的情况：

```cpp
int N = 2<<20;
for (int i = 0; i < N; ++i)
{
  printf("%d\n", i);
}
```

如要并行此循环，必须执行以下 2 个步骤：

- 必须编写完成**循环的单次迭代**工作的核函数。
- 由于核函数与其他正在运行的核函数无关，因此执行配置必须使核函数执行正确的次数，例如循环迭代的次数。

---
### Exercise: Accelerating a For Loop with a Single Block of Threads

目前，`01-single-block-loop.cu` 内的 `loop` 函数运行着一个“for 循环”并将连续打印 `0` 至 `9` 之间的所有数字。将 `loop` 函数重构为 CUDA 核函数，使其在启动后并行执行 `N` 次迭代。重构成功后，应仍能打印 `0` 至 `9` 之间的所有数字。如您遇到问题，请参阅 `01-single-block-loop-solution.cu`。


```shell
nvcc -arch=sm_70 -o single-block-loop 04-loops/01-single-block-loop.cu -run
```

---
## Coordinating Parallel Threads
GPU由于某种未知原因，必须映射每个进程以处理向量中的元素。每个线程都可以通过`blockDim.x`访问所在块大小并通过`blockIdx.x`访问网格内所在块的索引，`threadIdx.x`访问所在块自身的线程索引。

公式`threadIdx.x + blockIdx.x * blockDim.x`可把每个线程映射到向量的元素中。


---
## Using Block Dimensions for More Parallelization

线程块包含的线程具有数量限制：确切地说是 1024 个。为增加加速应用程序中的并行量，我们必须要能在多个线程块之间进行协调。

CUDA 核函数可以访问给出块中线程数的特殊变量：`blockDim.x`。通过将此变量与 `blockIdx.x` 和 `threadIdx.x` 变量结合使用，并借助惯用表达式 `threadIdx.x + blockIdx.x * blockDim.x` 在包含多个线程的多个线程块之间组织并行执行，并行性将得以提升。以下是详细示例。

执行配置 `<<<10, 10>>>` 将启动共计拥有 100 个线程的网格，这些线程均包含在由 10 个线程组成的 10 个线程块中。因此，我们希望每个线程（`0` 至 `99` 之间）都能计算该线程的某个唯一索引。

- 如果线程块 `blockIdx.x` 等于 `0`，则 `blockIdx.x * blockDim.x` 为 `0`。向 `0` 添加可能的 `threadIdx.x` 值（`0` 至 `9`），之后便可在包含 100 个线程的网格内生成索引 `0` 至 `9`。
- 如果线程块 `blockIdx.x` 等于 `1`，则 `blockIdx.x * blockDim.x` 为 `10`。向 `10` 添加可能的 `threadIdx.x` 值（`0` 至 `9`），之后便可在包含 100 个线程的网格内生成索引 `10` 至 `19`。
- 如果线程块 `blockIdx.x` 等于 `5`，则 `blockIdx.x * blockDim.x` 为 `50`。向 `50` 添加可能的 `threadIdx.x` 值（`0` 至 `9`），之后便可在包含 100 个线程的网格内生成索引 `50` 至 `59`。
- 如果线程块 `blockIdx.x` 等于 `9`，则 `blockIdx.x * blockDim.x` 为 `90`。向 `90` 添加可能的 `threadIdx.x` 值（`0` 至 `9`），之后便可在包含 100 个线程的网格内生成索引 `90` 至 `99`。


---
### Exercise: Accelerating a For Loop with Multiple Blocks of Threads

目前，`multi-block-loop\02-multi-block-loop.cu`内的 `loop` 函数运行着一个“for 循环”并将连续打印 `0` 至 `9` 之间的所有数字。将 `loop` 函数重构为 CUDA 核函数，使其在启动后并行执行 `N` 次迭代。重构成功后，应仍能打印 `0` 至 `9` 之间的所有数字。对于本练习，作为附加限制，请使用启动*至少 2 个线程块*的执行配置。如您遇到问题，请参阅`02-multi-block-loop-solution.cu`。


```shell
nvcc -arch=sm_70 multi-block-loop/02-multi-block-loop.cu -run
```

---
## Allocating Memory to be accessed on the GPU and the CPU

CUDA 的最新版本（版本 6 和更高版本）已能轻松分配可用于 CPU 主机和任意数量 GPU 设备的内存。尽管现今有许多适用于内存管理并可支持加速应用程序中最优性能的 [中高级技术](http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)，但我们现在要介绍的基础 CUDA 内存管理技术不但能够支持远超 CPU 应用程序的卓越性能，而且几乎不会产生任何开发人员成本。

如要分配和释放内存，并获取可在主机和设备代码中引用的指针，请使用 `cudaMallocManaged` 和 `cudaFree` 取代对 `malloc` 和 `free` 的调用，即把内存的分配和释放都从CPU迁移到GPU上，如下例所示：

```cpp
// CPU-only

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
a = (int *)malloc(size);

// Use `a` in CPU-only program.

free(a);
```

```cpp
// Accelerated

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
// Note the address of `a` is passed as first argument.
cudaMallocManaged(&a, size);

// Use `a` on the CPU and/or on any GPU in the accelerated system.

cudaFree(a);
```

---
### Exercise: Array Manipulation on both the Host and Device

`double-elements\01-double-elements.cu` 程序会分配一个数组、在主机上使用整数值对其进行初始化并尝试在 GPU 上对其中的每个值并行加倍，然后在主机上确认加倍操作是否成功。目前，程序将无法执行：因其正尝试在主机和设备上与指针 `a` 指向的数组进行交互，但仅分配可在主机上访问的数组（使用 `malloc`）。重构应用程序以满足以下条件，如您遇到问题，请参阅 `double-elements\01-double-elements-solution.cu`

- 指针 `a` 应可供主机和设备代码使用。
- 应该正确释放指针 `a` 的内存。


```shell
nvcc -arch=sm_70 -o double-elements 05-allocate/01-double-elements.cu -run
```

## Grid Size Work Amount Mismatch
在之前的场景中，网格中的线程数与元素数量完全匹配。如果线程数超过要完成的工作量，尝试访问不存在的元素会导致运行时错误，这个时候就需要用`if`条件语句，确保`threadIdx.x + blockIdx.x * blockDim.x`计算出的`dataIndex`小于数据元素数量N


---
## Handling Block Configuration Mismatches to Number of Needed Threads

可能会出现这样的情况，即无法表示会创建并行循环所需确切线程数量的执行配置。

常见示例与希望选择的最佳线程块大小有关。例如，鉴于 GPU 的硬件特性，所含线程的数量为 32 的倍数的线程块是为理想的选择，因其具备性能上的优势。假设我们要启动一些线程块且每个线程块中均包含 256 个线程（32 的倍数），并需运行 1000 个并行任务（此处使用极小的数量以便于说明），则任何数量的线程块均无法在网格中精确生成 1000 个总线程，因为没有任何整数值在乘以 32 后可以恰好等于 1000。

可以通过以下方式轻松解决这种情况：

- 编写执行配置，使其创建的线程数**超过**执行分配工作所需的线程数。
- 将值作为参数传递到核函数 (`N`) 中，以表示要处理的数据集总大小或完成工作所需的总线程数。
- 计算网格内的线程索引后（使用 `tid+bid*bdim`），请检查该索引是否超过 `N`，并且只在不超过的情况下执行与核函数相关的工作。


以下是编写执行配置的惯用方法示例，适用于 `N` 和线程块中的线程数已知，但无法保证网格中的线程数和 `N` 之间完全匹配的情况。如此一来，便可确保网格中至少始终拥有 `N` 所需的线程数，且超出的线程数至多仅可相当于 1 个线程块的线程数量：

```cpp
// Assume `N` is known
int N = 100000;

// Assume we have a desire to set `threads_per_block` exactly to `256`
size_t threads_per_block = 256;

// Ensure there are at least `N` threads in the grid, but only 1 block's worth extra
size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
// Better than N/threads_per_block + 1, considering  when N % threads_per_block = 0 

some_kernel<<<number_of_blocks, threads_per_block>>>(N);
```

由于上述执行配置致使网格中的线程数超过 `N`，因此需要注意 `some_kernel` 定义中的内容，以确保 `some_kernel` 在由其中一个 \”extra\” 线程执行时不会尝试访问超出范围的数据元素：

```cpp
__global__ some_kernel(int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < N) // Check to make sure `idx` maps to some value within `N`
  {
    // Only do work if it does
  }
}
```

---
### Exercise: Accelerating a For Loop with a Mismatched Execution Configuration

`mismatched-config-loop\02-mismatched-config-loop.cu`中的程序使用 `cudaMallocManaged` 为包含 1000 个元素的整数数组分配内存，然后试图使用 CUDA 核函数以并行方式初始化数组中的所有值。此程序假设 `N` 和 `threads_per_block` 的数量均为已知。您的任务是完成以下两个目标，如您遇到问题，请参阅 `mismatched-config-loop\02-mismatched-config-loop-solution.cu`：

- 为 `number_of_blocks` 分配一个值，以确保线程数至少与指针 `a` 中可供访问的元素数同样多。
- 更新 `initializeElementsTo` 核函数以确保不会尝试访问超出范围的数据元素。


```shell
!nvcc -arch=sm_70 -o mismatched-config-loop/02-mismatched-config-loop.cu -run
```

---
## Data Sets Larger then the Grid

或出于选择，为了要创建具有超高性能的执行配置，或出于需要，一个网格中的线程数量可能会小于数据集的大小。请思考一下包含 1000 个元素的数组和包含 250 个线程的网格（此处使用极小的规模以便于说明）。此网格中的每个线程将需使用 4 次。如要实现此操作，一种常用方法便是在核函数中使用**网格跨度循环**。

在网格跨度循环中，每个线程将在网格内使用 `tid+bid*bdim` 计算自身唯一的索引，并对数组内该索引的元素执行相应运算，然后将网格中的线程数添加到索引并重复此操作，直至超出数组范围。例如，对于包含 500 个元素的数组和包含 250 个线程的网格，网格中索引为 20 的线程将执行如下操作：

- 对包含 500 个元素的数组的元素 20 执行相应运算
- 将其索引增加 250，使网格的大小达到 270
- 对包含 500 个元素的数组的元素 270 执行相应运算
- 将其索引增加 250，使网格的大小达到 520
- 由于 520 现已超出数组范围，因此线程将停止工作


CUDA 提供一个可给出网格中线程块数的特殊变量：`gridDim.x`。然后计算网格中的总线程数，即网格中的线程块数乘以每个线程块中的线程数：`gridDim.x * blockDim.x`。带着这样的想法来看看以下核函数中网格跨度循环的详细示例：

```cpp
__global void kernel(int *a, int N)
{
  int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;

  for (int i = indexWithinTheGrid; i < N; i += gridStride)
  {
    // do work on a[i];
  }
}
```
由于grid是GPU执行计算执行核函数的单位，不可以直接跨过grid。
错误示范：
```cpp
__global void kernel(int *a, int N)
{
  int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  //int gridStride = gridDim.x * blockDim.x;

  if(indexWithinTheGrid<N)
  {
    // do work on a[i];
  }
}
```
这样子是无法处理那些超出范围的数据。

---
### Exercise: Use a Grid-Stride Loop to Manipulate an Array Larger than the Grid

重构 `grid-stride-double\03-grid-stride-double.cu` 以在 `doubleElements` 核函数中使用网格跨度循环，进而使小于 `N` 的网格可以重用线程以覆盖数组中的每个元素。程序会打印数组中的每个元素是否均已加倍，而当前该程序会准确打印出 `FALSE`。如您遇到问题，请参阅 `grid-stride-double\03-grid-stride-double-solution.cu`。


```shell
nvcc -arch=sm_70 grid-stride-double/03-grid-stride-double.cu -run
```

---
## Error Handling

与在任何应用程序中一样，加速 CUDA 代码中的错误处理同样至关重要。即便不是大多数，也有许多 CUDA 函数（例如，[内存管理函数](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY)）会返回类型为 `cudaError_t` 的值，该值可用于检查调用函数时是否发生错误。以下是对调用 `cudaMallocManaged` 函数执行错误处理的示例：

```cpp
cudaError_t err;
err = cudaMallocManaged(&a, N)                    // Assume the existence of `a` and `N`.

if (err != cudaSuccess)                           // `cudaSuccess` is provided by CUDA.
{
  printf("Error: %s\n", cudaGetErrorString(err)); // `cudaGetErrorString` is provided by CUDA.
}
```

启动定义为返回 `void` 的核函数后，将不会返回类型为 `cudaError_t` 的值。为检查启动核函数时是否发生错误（例如，如果启动配置错误），CUDA 提供 `cudaGetLastError` 函数，该函数会返回类型为 `cudaError_t` 的值。

```cpp
/*
 * This launch should cause an error, but the kernel itself
 * cannot return it.
 */

someKernel<<<1, -1>>>();  // -1 is not a valid number of threads.

cudaError_t err;
err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
if (err != cudaSuccess)
{
  printf("Error: %s\n", cudaGetErrorString(err));
}
```

最后，为捕捉异步错误（例如，在异步核函数执行期间），请务必检查后续同步 CUDA 运行时 API 调用所返回的状态（例如 `cudaDeviceSynchronize`）；如果之前启动的其中一个核函数失败，则将返回错误。

`cudaMallocManaged`之类的函数就不需要返回值了。主要是由上面两个函数完成错误处理。

---
### Exercise: Add Error Handling

目前，`01-add-error-handling.cu`会编译、运行并打印已加倍失败的数组元素。不过，该程序不会指明其中是否存在任何错误。重构应用程序以处理 CUDA 错误，以便您可以了解程序出现的问题并进行有效调试。您将需要调查在调用 CUDA 函数时可能出现的同步错误，以及在执行 CUDA 核函数时可能出现的异步错误。如您遇到问题，请参阅 `add-error-handling/01-add-error-handling-solution.cu`。


```shell
nvcc -arch=sm_70 -o add-error-handling/01-add-error-handling.cu -run
```

---
### CUDA Error Handling Function

创建一个包装 CUDA 函数调用的宏对于检查错误十分有用。以下是一个宏示例，您可以在余下练习中随时使用：

```cpp
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

int main()
{

/*
 * The macro can be wrapped around any function returning
 * a value of type `cudaError_t`.
 */

  checkCuda( cudaDeviceSynchronize() )
}
```

---
## Summary

至此，您已经完成以下列出的所有实验学习目标：

- 编写、编译及运行既可调用 CPU 函数也可**启动** GPU **核函数*的 C/C++ 程序。
- 使用**执行配置**控制并行**线程层次结构**。
- 重构串行循环以在 GPU 上并行执行其迭代。
- 分配和释放可用于 CPU 和 GPU 的内存。
- 处理 CUDA 代码生成的错误。


现在，您将完成实验的最终目标：

- 加速 CPU 应用程序。

---
### Final Exercise: Accelerate Vector Addition Application

下面的挑战将使您有机会运用在实验中学到的知识。其中涉及加速 CPU 向量加法程序，尽管该程序不甚复杂，但仍能让您有机会重点运用所学的借助 CUDA 加速 GPU 应用程序的相关知识。完成此练习后，如果您有富余时间并有意深究，可继续学习*高阶内容*部分以了解涉及更复杂代码库的一些挑战。

`vector-add\01-vector-add.cu`包含一个可正常运作的 CPU 向量加法应用程序。加速其 `addVectorsInto` 函数，使之在 GPU 上以 CUDA 核函数运行并使其并行执行工作。鉴于需发生以下操作，如您遇到问题，请参阅 `vector-add\01-vector-add-solution.cu`。

- 扩充 `addVectorsInto` 定义，使之成为 CUDA 核函数。
- 选择并使用有效的执行配置，以使 `addVectorsInto` 作为 CUDA 核函数启动。
- 更新内存分配，内存释放以反映主机和设备代码需要访问 3 个向量：`a`、`b` 和 `result`。
- 重构 `addVectorsInto` 的主体：将在单个线程内部启动，并且只需对输入向量执行单线程操作。确保线程从不尝试访问输入向量范围之外的元素，并注意线程是否需对输入向量的多个元素执行操作。
- 在 CUDA 代码可能以其他方式静默失败的位置添加错误处理。


```shell
nvcc -arch=sm_70 vector-add/01-vector-add.cu -run
```

---
## Advanced Content

以下练习为时间富余且有意深究的学习者提供额外挑战。这些挑战需要使用更先进的技术加以应对，并且其提供的有用知识也很少。因此，完成这些挑战着实不易，但您在此过程中亦会收获重大进步。

---
## Grids and Blocks of 2 and 3 Dimensions

可以将网格和线程块定义为最多具有 3 个维度。使用多个维度定义网格和线程块绝不会对其性能造成任何影响，但这在处理具有多个维度的数据时可能非常有用，例如 2D 矩阵。如要定义二维或三维网格或线程块，可以使用 CUDA 的 `dim3` 类型，即如下所示：

```cpp
dim3 threads_per_block(16, 16, 1);
dim3 number_of_blocks(16, 16, 1);
someKernel<<<number_of_blocks, threads_per_block>>>();
```

鉴于以上示例，`someKernel` 内部的变量 `gridDim.x`、`gridDim.y`、`blockDim.x` 和 `blockDim.y` 均将等于 `16`。

---
### Exercise: Accelerate 2D Matrix Multiply Application

文件 `01-matrix-multiply-2d.cu`包含一个功能齐全的主机函数 `matrixMulCPU`。您的任务是扩建 CUDA 核函数 `matrixMulGPU`。源代码将使用这两个函数执行矩阵乘法，并比较它们的答案以验证您编写的 CUDA 核函数是否正确。使用以下指南获得操作支持，如您遇到问题，请参阅`matrix-multiply/01-matrix-multiply-2d-solution.cu`：

- 您将需创建执行配置，其参数均为 `dim3` 值，且 `x` 和 `y` 维度均设为大于 `1`。
- 在核函数主体内部，您将需要按照惯例在网格内建立所运行线程的唯一索引，但应为线程建立两个索引：一个用于网格的 x 轴，另一个用于网格的 y 轴。


```shell
nvcc -arch=sm_70 matrix-multiply-2d/01-matrix-multiply-2d.cu -run
```

---
### Exercise: Accelerate A Thermal Conductivity Application

在下面的练习中，您将为模拟金属银二维热传导的应用程序执行加速操作。

将 `01-heat-conduction.cu`内的 `step_kernel_mod` 函数转换为在 GPU 上执行，并修改 `main` 函数以恰当分配在 CPU 和 GPU 上使用的数据。`step_kernel_ref` 函数在 CPU 上执行并用于检查错误。由于此代码涉及浮点计算，因此不同的处理器甚或同一处理器上的简单重排操作都可能导致结果略有出入。为此，错误检查代码会使用错误阈值，而非查找完全匹配。如您遇到问题，请参阅`heat-conduction\01-heat-conduction-solution.cu`。


```shell
nvcc -arch=sm_70 -o heat-conduction 09-heat/01-heat-conduction.cu -run
```

> 此任务中的原始热传导 CPU 源代码取自于休斯顿大学的文章 [An OpenACC Example Code for a C-based heat conduction code](http://docplayer.net/30411068-An-openacc-example-code-for-a-c-based-heat-conduction-code.html)（基于 C 的热传导代码的 OpenACC 示例代码）。
