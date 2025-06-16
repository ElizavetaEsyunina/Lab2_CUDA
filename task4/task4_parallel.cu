#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define NUM_RUNS 100
#define DEFAULT_BLOCK_SIZE 256

__global__ void addKernel(double* arr1, double* arr2, double* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = arr1[idx] + arr2[idx];
    }
}

__global__ void substractKernel(double* arr1, double* arr2, double* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = arr1[idx] - arr2[idx];
    }
}

__global__ void multiplyKernel(double* arr1, double* arr2, double* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = arr1[idx] * arr2[idx];
    }
}

__global__ void divideKernel(double* arr1, double* arr2, double* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = arr1[idx] / arr2[idx];
    }
}

int main() {
    double *arr1, *arr2, *sum, *difference, *product, *quotient;
    double *d_arr1, *d_arr2, *d_sum, *d_difference, *d_product, *d_quotient;
    int rows, cols, size;
    clock_t start, end;
    double total_time = 0.0, average_time;
    char* endptr;

    char* array_size_str = getenv("ARRAY_SIZE");
    if (array_size_str == NULL) {
        fprintf(stderr, "Error: ARRAY_SIZE environment variable not set.\n");
        return 1;
    }
    rows = strtol(array_size_str, &endptr, 10);
    if (*endptr != '\0' || rows <= 0) {
        fprintf(stderr, "Error: Incorrect ARRAY_SIZE value: %s\n", array_size_str);
        return 1;
    }
    cols = rows;
    size = rows * cols;

    int threadsPerBlock = DEFAULT_BLOCK_SIZE;
    char* block_size_str = getenv("BLOCK_SIZE");
    if (block_size_str != NULL) {
        threadsPerBlock = strtol(block_size_str, &endptr, 10);
        if (*endptr != '\0' || threadsPerBlock <= 0 || threadsPerBlock > 1024) {
            fprintf(stderr, "Error: Incorrect BLOCK_SIZE value (1-1024): %s\n", block_size_str);
            return 1;
        }
    }
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    printf("Matrix size: %d x %d (%d elements)\n", rows, cols, size);
    printf("CUDA configuration: %d blocks x %d threads\n", blocksPerGrid, threadsPerBlock);

    arr1 = (double*)malloc(size * sizeof(double));
    arr2 = (double*)malloc(size * sizeof(double));
    sum = (double*)malloc(size * sizeof(double));
    difference = (double*)malloc(size * sizeof(double));
    product = (double*)malloc(size * sizeof(double));
    quotient = (double*)malloc(size * sizeof(double));

    cudaMalloc((void**)&d_arr1, size * sizeof(double));
    cudaMalloc((void**)&d_arr2, size * sizeof(double));
    cudaMalloc((void**)&d_sum, size * sizeof(double));
    cudaMalloc((void**)&d_difference, size * sizeof(double));
    cudaMalloc((void**)&d_product, size * sizeof(double));
    cudaMalloc((void**)&d_quotient, size * sizeof(double));

    for (int run = 0; run < NUM_RUNS; ++run) {
        srand(time(NULL) + run);
        for (int i = 0; i < size; ++i) {
            arr1[i] = (double)rand() / RAND_MAX * 100.0;
            arr2[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
        }

        cudaMemcpy(d_arr1, arr1, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_arr2, arr2, size * sizeof(double), cudaMemcpyHostToDevice);

        start = clock();
        
        addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr1, d_arr2, d_sum, size);
        substractKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr1, d_arr2, d_difference, size);
        multiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr1, d_arr2, d_product, size);
        divideKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr1, d_arr2, d_quotient, size);
        
        cudaDeviceSynchronize();
        end = clock();

        cudaMemcpy(sum, d_sum, size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(difference, d_difference, size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(product, d_product, size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(quotient, d_quotient, size * sizeof(double), cudaMemcpyDeviceToHost);

        double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
        total_time += time_spent;

        printf("Run %d: %f sec\n", run + 1, time_spent);
    }

    average_time = total_time / NUM_RUNS;
    printf("\nAverage CUDA execution time (over %d iterations): %f sec\n", NUM_RUNS, average_time);
    printf("Total threads used: %d (%d blocks x %d threads)\n", blocksPerGrid * threadsPerBlock, blocksPerGrid, threadsPerBlock);

    free(arr1); free(arr2); free(sum); free(difference); free(product); free(quotient);
    cudaFree(d_arr1); cudaFree(d_arr2); cudaFree(d_sum); 
    cudaFree(d_difference); cudaFree(d_product); cudaFree(d_quotient);

    return 0;
}