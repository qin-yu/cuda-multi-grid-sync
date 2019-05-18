/**
 * Copyright 2019 Qin Yu
 */

/**
 * Multi-grid Vector addition: C = A + B.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <set>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

extern "C" __global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
int
main(void)
{
	int num_of_gpus = 0;
	checkCudaErrors(cudaGetDeviceCount(&num_of_gpus));
	printf("No. of GPU on node %d\n", num_of_gpus);

	if (num_of_gpus != 2) {
		printf("Two GPUs are required to run this sample code\n");
		exit(EXIT_WAIVED);
	}

	// not necessary, just to remind myself i can do this:
	cudaSetDevice(0); cudaDeviceEnablePeerAccess(1,0);
	cudaSetDevice(1); cudaDeviceEnablePeerAccess(0,0);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 40000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    float *d_A2 = NULL;
    err = cudaMalloc((void **)&d_A2, size);
    float *d_B2 = NULL;
    err = cudaMalloc((void **)&d_B2, size);
    float *d_C2 = NULL;
    err = cudaMalloc((void **)&d_C2, size);
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A2, h_A, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B2, h_B, size, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
//    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
//    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A2, d_B2, d_C2, numElements);
//    err = cudaGetLastError();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 1);
    std::cout << "deviceProp.multiProcessorCount is " << deviceProp.multiProcessorCount << std::endl;

    int numBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, vectorAdd, threadsPerBlock, 0);
    std::cout << "numBlocksPerSm is " << numBlocksPerSm << std::endl;

    dim3 dimGrid(blocksPerGrid, 1, 1), dimBlock(threadsPerBlock, 1, 1);
    cudaLaunchParams launchParamsList[2];
    void *kernelArgs[] = {(void *)&d_A, (void *)&d_B, (void *)&d_C, (void *)&numElements};
    void *kernelArgs2[] = {(void *)&d_A2, (void *)&d_B2, (void *)&d_C2, (void *)&numElements};
	for (int i = 0; i < 2; i++) {
		cudaSetDevice(i);
		launchParamsList[i].func      = (void *)vectorAdd;
		launchParamsList[i].blockDim  = dimBlock;
		launchParamsList[i].gridDim   = dimGrid;
		launchParamsList[i].sharedMem = 0;
		cudaStreamCreate(&launchParamsList[i].stream);
	}
	launchParamsList[0].args = kernelArgs;
	launchParamsList[1].args = kernelArgs2;

	checkCudaErrors(cudaLaunchCooperativeKernelMultiDevice(
			launchParamsList,
			2,
			0
//			cudaCooperativeLaunchMultiDeviceNoPreSync | cudaCooperativeLaunchMultiDeviceNoPostSync
	));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaLaunchCooperativeKernelMultiDevice failed.\n");
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaStreamSynchronize(launchParamsList[0].stream));
	checkCudaErrors(cudaSetDevice(1));
	checkCudaErrors(cudaStreamSynchronize(launchParamsList[1].stream));

    // Copy the device result vector in device memory to the host result vector in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) {
    	std::cout << h_A[i] + h_B[i] << " vs " << h_C[i] << std::endl;
    }
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");
    err = cudaMemcpy(h_C, d_C2, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED2\n");

    // Free device global memory
    err = cudaFree(d_A);
    err = cudaFree(d_B);
    err = cudaFree(d_C);
    err = cudaFree(d_A2);
    err = cudaFree(d_B2);
    err = cudaFree(d_C2);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}

