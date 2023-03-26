/*
   Lab3: Convolution(2D) routine using texture and shared memory

   Name: Vamsee Krishna Tunuguntla
   CWID: 20009051

   Constraints: Input vector <dimX>, <dimY> values should be in between 0 to 15, VecDim size is inarbitrary, <dimK> is the size of the 2D mask(<dimK>*<dimK>)
   Code has been implemented taking the instructions into consideration
   */



#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_SIZE 16

texture<unsigned char, 2, cudaReadModeElementType> tex;

//Convolution kernel
__global__ void convolution_2D_basic_kernel(unsigned char *out, unsigned char *in, unsigned char *mask, int maskwidth, int w, int h) {

   // Declare shared memory
    __shared__ unsigned char shared_mem[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int N_start_col = bx * blockDim.x - (maskwidth/2);
    int N_start_row = by * blockDim.y - (maskwidth/2);

    // Load data into shared memory
    int shared_row = ty;
    int shared_col = tx;
    int input_row = N_start_row + ty;
    int input_col = N_start_col + tx;
    if (input_row >= 0 && input_row < h && input_col >= 0 && input_col < w) {
        shared_mem[shared_row][shared_col] = in[input_row * w + input_col];
    } else {
        shared_mem[shared_row][shared_col] = 0;
    }
    __syncthreads();
    
    // Compute convolution
    if (tx < blockDim.x - maskwidth/2 && ty < blockDim.y - maskwidth/2 && bx * blockDim.x + tx < w && by * blockDim.y + ty < h) {
        int pixVal = 0;
        for (int j = 0; j < maskwidth; ++j) {
            for (int k = 0; k < maskwidth; ++k) {
                int shared_row_index = shared_row + j;
                int shared_col_index = shared_col + k;
                pixVal += shared_mem[shared_row_index][shared_col_index] * mask[j * maskwidth + k];
            }
        }
        out[(by * blockDim.y + ty) * w + bx * blockDim.x + tx] = (unsigned char)(pixVal);
    }
}

int main(int argc, char *argv[]) {
    int dimX, dimY, dimK;
    if (argc != 7 || strcmp(argv[1], "-i") != 0 || strcmp(argv[3], "-j") != 0 || strcmp(argv[5], "-k") != 0) {
        printf("Usage: ./convolution2D -i <dimX> -j <dimY> -k <dimK>\n");
        return 1;
    } else {
        dimX = atoi(argv[2]);
        dimY = atoi(argv[4]);
        dimK = atoi(argv[6]);
    }

    // Allocate host memory
    unsigned char *h_in = (unsigned char*) malloc(dimX * dimY * sizeof(unsigned char));
    unsigned char *h_mask = (unsigned char*) malloc(dimK * dimK * sizeof(unsigned char));
    unsigned char *h_out = (unsigned char*) malloc(dimX * dimY * sizeof(unsigned char));

    // Allocate device memory
    int input_size = dimX * dimY;
    int mask_size = dimK * dimK;
    int output_size = input_size;
    unsigned char *d_input, *d_mask, *d_output;
    cudaMalloc((void **)&d_input, input_size * sizeof(unsigned char));
    cudaMalloc((void **)&d_mask, mask_size * sizeof(unsigned char));
    cudaMalloc((void **)&d_output, output_size * sizeof(unsigned char));

    // Generate random input and mask data
    srand(time(NULL));
    for (int i = 0; i < dimX * dimY; i++) {
        h_in[i] = rand() % 16;
    }
    for (int i = 0; i < dimK * dimK; i++) {
        h_mask[i] = rand() % 16;
    }

    // Print input, mask matrices
    printf("Input:\n");
    for (int i = 0; i < dimY; i++) {
    	for (int j = 0; j < dimX; j++) {
        	printf("%d ", h_in[i * dimX + j]);
    	}
    	printf("\n");
    }
    printf("\nMask:\n");
    for (int i = 0; i < dimK; i++) {
    	for (int j = 0; j < dimK; j++) {
        	printf("%d ", h_mask[i * dimK + j]);
    	}
    	printf("\n");
    }

    // Copy host memory to device

    cudaMemcpy(d_input, h_in, input_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, mask_size * sizeof(unsigned char), cudaMemcpyHostToDevice);


    // Initialize thread block and kernel grid dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((dimX + TILE_SIZE - 1) / TILE_SIZE, (dimY + TILE_SIZE - 1) / TILE_SIZE);

    // Bind texture to input data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    cudaBindTexture2D(NULL, &tex, d_input, &channelDesc, dimX, dimY, dimX * sizeof(unsigned char));

    // Invoke CUDA kernel
    convolution_2D_basic_kernel<<<grid, block>>>(d_output, d_input, d_mask, dimK, dimX, dimY);

    // Copy results from device to host
    cudaMemcpy(h_out, d_output, output_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Print output matrices

    printf("\nOutput:\n");
    for (int i = 0; i < dimY; i++) {
    	for (int j = 0; j < dimX; j++) {
        	printf("%d ", h_out[i * dimX + j]);
    	}
    	printf("\n");
    }

    // Deallocate device memory
    cudaUnbindTexture(tex);
    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);

    // Deallocate host memory
    free(h_in);
    free(h_mask);
    free(h_out);

    return 0;
}
