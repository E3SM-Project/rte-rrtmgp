# Welcome to the C++ RRTMGP

RRTMGP++ is ported to C++ using Kokkos, and it was ported with readibility in mind, keeping it as similar to the original Fortran code as possible, including:
* Multi-dimensional arrays
* Column-major index ordering
* Array slicing

All of this was retained while also achieve excellent performance on Nvidia and AMD GPUs (with support for Intel GPUs coming soon). For instance, on the Summit supercomputer, the six Nvidia Tesla V100 GPUs on a node run RRTMGP++'s longwave solver 45x faster than the two IBM Power9 CPUs on a node including all data transfer costs. The shortwave solver runs 30x faster. Even at very small workloads, the GPUs give sizeable speed-up because all fo the RRTMGP++ code (the `load_and_init` function is the only exception) runs continuously on the GPU with minimal traffic between device and host memory and with all kernels launched asynchronously to hide kernel launch latencies. The use of a pool allocator was critical for performance as well.

To build and test the code, for example on Summit, please see the README.md for the rte-rrtmgp/cpp/test directory.

