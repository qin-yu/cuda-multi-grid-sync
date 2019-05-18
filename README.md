# Multi-grid/Multi-device Synchronization, Cooperative Group
This is my experiment on CUDA Multi-Device Synchronization, before my attempt of making this functionality available in Julia.

Firstly, on GitHub, there are only three (useful or not) CUDA files that actually use CUDA multi-grid synchronization:
- the official [conjugateGradientMultiDeviceCG.cu](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/conjugateGradientMultiDeviceCG/conjugateGradientMultiDeviceCG.cu) by NVIDIA.
- [an experiment](https://github.com/cwpearson/cuda-experiments/blob/master/system-atomics/main.cu) by another user.
- [a test](https://github.com/fabricatedmath/plastic/blob/master/src/multi_test.cu) by another user.

and now additionally:
- [this repo](https://github.com/qin-yu/cuda-multi-grid-sync) you are looking at.


To actually learn how should you use `cudaLaunchCooperativeKernelMultiDevice`, which is far different from `cudaLaunchCooperativeKernel` or `cudaLaunchKernel`, go to [C.4. Multi-Device Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-synchronization-cg) in CUDA Toolkit Documentation.

Before you get into the detail, make sure you have access to a node/host/machine with more than two GPUs installed. These GPUs must be identical and must have compute capability greater than 6.1.
