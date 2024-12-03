#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Kernel for matrix transposition with shared memory and padding
__global__ void matrixTransposeSharedMemory(float* d_in, float* d_out, int width, int height) {
    __shared__ float sharedMem[BLOCK_SIZE][BLOCK_SIZE + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < width && y < height) {
        sharedMem[threadIdx.y][threadIdx.x] = d_in[y * width + x];
    }

    __syncthreads();

    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    if (x < height && y < width) {
        d_out[y * height + x] = sharedMem[threadIdx.x][threadIdx.y];
    }
}

// Kernel for parallel reduction using shared memory
__global__ void parallelReduction(float* d_in, float* d_out, int size) {
    __shared__ float sharedMem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    sharedMem[tid] = (index < size) ? d_in[index] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = sharedMem[0];
    }
}

// Function to initialize the matrix with random values
void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand() % 100) / 10.0f;
    }
}

// Main function
int main() {
    const int width = 1024;
    const int height = 1024;
    const int size = width * height;

    float* h_in = (float*)malloc(size * sizeof(float));
    float* h_out = (float*)malloc(size * sizeof(float));

    initializeMatrix(h_in, height, width);

    float* d_in, * d_out;
    cudaMalloc((void**)&d_in, size * sizeof(float));
    cudaMalloc((void**)&d_out, size * sizeof(float));

    cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch matrix transpose kernel
    matrixTransposeSharedMemory << <gridDim, blockDim >> > (d_in, d_out, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Matrix transposition completed successfully!\n");

    // Reduction example
    const int reductionSize = 1024;
    float* h_reduction = (float*)malloc(reductionSize * sizeof(float));
    float* h_reduction_result = (float*)malloc(sizeof(float));

    for (int i = 0; i < reductionSize; i++) {
        h_reduction[i] = static_cast<float>(rand() % 100);
    }

    float* d_reduction, * d_reduction_out;
    cudaMalloc((void**)&d_reduction, reductionSize * sizeof(float));
    cudaMalloc((void**)&d_reduction_out, sizeof(float));

    cudaMemcpy(d_reduction, h_reduction, reductionSize * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE;
    int blocks = (reductionSize + threadsPerBlock - 1) / threadsPerBlock;

    parallelReduction << <blocks, threadsPerBlock >> > (d_reduction, d_reduction_out, reductionSize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_reduction_result, d_reduction_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Reduction result: %f\n", h_reduction_result[0]);

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_reduction);
    cudaFree(d_reduction_out);

    free(h_in);
    free(h_out);
    free(h_reduction);
    free(h_reduction_result);

    return 0;
}
