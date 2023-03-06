/*
   Lab2: Histogram using Atomic operations, shared memory

   Name: Vamsee Krishna Tunuguntla
   CWID: 20009051

   Constraints: Input vector<VecDim> values should be in between 0 to 1023, VecD
im size is inarbitrary, <BinNum> should not exceed 2^8
   Code has been implemented taking the instructions into consideration and using multiple kernels
   */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void histo_kernel(unsigned int *input, long size, unsigned int *histo, int bin_num)
{
    // Shared memory to hold histogram bins
    __shared__ unsigned int s_histo[256];

    // Initialize shared memory bins to 0
    for (int i = threadIdx.x; i < bin_num; i += blockDim.x)
        s_histo[i] = 0;
    __syncthreads();

    // Calculate stride and thread index
    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < size)
    {
        // Calculate histogram bin for input value
        int bin = input[tid] / (1024 / bin_num);
        
        // Increment histogram bin using atomic operation
        atomicAdd(&(s_histo[bin]), 1);
        
        tid += stride;
    }
    __syncthreads();

    // Accumulate per-thread histogram bins into global histogram
    for (int i = threadIdx.x; i < bin_num; i += blockDim.x)
        atomicAdd(&(histo[i]), s_histo[i]);
}

int main(int argc, char *argv[])
{
    // Parse input arguments
    if (argc != 5)
    {
        printf("Usage: %s -i <BinNum> <VecDim> <BlockSize>\n", argv[0]);
        exit(1);
    }

    int bin_num = atoi(argv[2]);
    long vec_dim = atol(argv[3]);
    int block_size = atoi(argv[4]);

    // Allocate host memory
    unsigned int *h_input = (unsigned int *)malloc(vec_dim * sizeof(unsigned int));
    unsigned int *h_histo = (unsigned int *)calloc(bin_num, sizeof(unsigned int));

    // Generate random input vector
    for (int i = 0; i < vec_dim; i++)
        h_input[i] = rand() % 1024;

   /*
    // Print the input vector
    printf("Input vector:\n");
    for (int i = 0; i < vec_dim; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\n");
    */

    // Allocate device memory
    unsigned int *d_input, *d_histo;
    cudaMalloc((void **)&d_input, vec_dim * sizeof(unsigned int));
    cudaMalloc((void **)&d_histo, bin_num * sizeof(unsigned int));

    // Copy host memory to device
    cudaMemcpy(d_input, h_input, vec_dim * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Calculate grid dimensions for the first kernel launch
    dim3 block_dim_1(block_size, 1, 1);
    dim3 grid_dim_1((vec_dim / 2 + block_size - 1) / block_size, 1, 1);

    // Invoke the first CUDA kernel
    histo_kernel<<<grid_dim_1, block_dim_1>>>(d_input, vec_dim / 2, d_histo, bin_num);

    // Calculate grid dimensions for the second kernel launch
    dim3 block_dim_2(block_size, 1, 1);
    dim3 grid_dim_2(((vec_dim - vec_dim / 2) + block_size - 1) / block_size, 1, 1);

    // Invoke the second CUDA kernel
    histo_kernel<<<grid_dim_2, block_dim_2>>>(d_input + vec_dim / 2, vec_dim - vec_dim / 2, d_histo, bin_num);

    // Copy results from device to host
    cudaMemcpy(h_histo, d_histo, bin_num * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Deallocate device memory
    cudaFree(d_input);
    cudaFree(d_histo);

    // Print histogram results
    for (int i = 0; i < bin_num; i++)
        printf("%d: %u\n", i, h_histo[i]);

   // Count total number of elements from all bins
   int total_elements = 0;
   for (int i = 0; i < bin_num; i++){
         total_elements += h_histo[i];
   }
   //printf("Vector Dimension (Total elements): %ld\n", vec_dim);
   printf("Total Elements in bins: %d\n", total_elements);

    // Deallocate host memory
    free(h_input);
    free(h_histo);

    return 0;
}
