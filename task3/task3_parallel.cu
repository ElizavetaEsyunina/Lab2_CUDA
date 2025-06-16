#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void add_kernel(double *arr1, double *arr2, double *result, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
                result[idx] = arr1[idx] + arr2[idx];
        }
}

__global__ void substract_kernel(double *arr1, double *arr2, double *result, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
                result[idx] = arr1[idx] - arr2[idx];
        }
}

__global__ void multiply_kernel(double *arr1, double *arr2, double *result, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
                result[idx] = arr1[idx] * arr2[idx];
        }
}

__global__ void divide_kernel(double *arr1, double *arr2, double *result, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
                result[idx] = arr1[idx] / arr2[idx];
        }
}

int main() {
        double *arr1, *arr2, *sum, *difference, *product, *quotient;
        double *d_arr1, *d_arr2, *d_sum, *d_difference, *d_product, *d_quotient;
        int i, j, array_size, num_iterations, threadsPerBlock, blocksPerGrid;
        cudaEvent_t start, stop;
        float gpu_time_used, total_time = 0.0, average_time;
        char *endptr;

        char *array_size_str = getenv("ARRAY_SIZE");
        if (array_size_str == NULL) {
                fprintf(stderr, "Error: ARRAY_SIZE environment variable not set. \n");
                return 1;
        }

        array_size = strtol(array_size_str, &endptr, 10);
        if (*endptr != '\0' || array_size <= 0) {
                fprintf(stderr, "Error: Incorrect ARRAY_SIZE value: %s\n", array_size_str);
                return 1;
        }

        char *num_iterations_str = getenv("NUM_ITERATIONS");
        if (num_iterations_str == NULL) {
                fprintf(stderr, "Error: NUM_ITERATIONS environment variable not set. \n");
                return 1;
        }

        num_iterations = strtol(num_iterations_str, &endptr, 10);
        if (*endptr != '\0' || num_iterations <= 0) {
                fprintf(stderr, "Error: Incorrect NUM_ITERATIONS value: %s\n", num_iterations_str);
                return 1;
        }

        char *threads_per_block_str = getenv("THREADS_PER_BLOCK");
        if (threads_per_block_str == NULL) {
                fprintf(stderr, "Error: THREADS_PER_BLOCK environment variable not set. \n");
                return 1;
        }

        threadsPerBlock = strtol(threads_per_block_str, &endptr, 10);
        if (*endptr != '\0' || threadsPerBlock <= 0) {
                fprintf(stderr, "Error: Incorrect THREADS_PER_BLOCK value: %s\n", threads_per_block_str);
                return 1;
        }

        arr1 = (double *)malloc(array_size * sizeof(double));
        arr2 = (double *)malloc(array_size * sizeof(double));
        sum = (double *)malloc(array_size * sizeof(double));
        difference = (double *)malloc(array_size * sizeof(double));
        product = (double *)malloc(array_size * sizeof(double));
        quotient = (double *)malloc(array_size * sizeof(double));

        cudaMalloc((void **)&d_arr1, array_size * sizeof(double));
        cudaMalloc((void **)&d_arr2, array_size * sizeof(double));
        cudaMalloc((void **)&d_sum, array_size * sizeof(double));
        cudaMalloc((void **)&d_difference, array_size * sizeof(double));
        cudaMalloc((void **)&d_product, array_size * sizeof(double));
        cudaMalloc((void **)&d_quotient, array_size * sizeof(double));

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        blocksPerGrid = (array_size + threadsPerBlock - 1) / threadsPerBlock;

        for (j = 0; j < num_iterations; ++j) {
                srand(time(NULL)+j);
                for (i = 0; i < array_size; ++i) {
                        arr1[i] = (double)rand() / RAND_MAX * 100.0;
                        arr2[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
                }

                cudaMemcpy(d_arr1, arr1, array_size * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_arr2, arr2, array_size * sizeof(double), cudaMemcpyHostToDevice);

                cudaEventRecord(start, 0);

                add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr1, d_arr2, d_sum, array_size);
                substract_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr1, d_arr2, d_difference, array_size);
                multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr1, d_arr2, d_product, array_size);
                divide_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr1, d_arr2, d_quotient, array_size);
                cudaDeviceSynchronize();

                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&gpu_time_used, start, stop);

                total_time += gpu_time_used / 1000.0;
        }

        average_time = total_time / num_iterations;
        printf("Average CUDA execution time (over %d iterations): %f sec\n", num_iterations, average_time);
        printf("Threads: %d, blocks: %f \n", threadsPerBlock, blocksPerGrid);

        free(arr1);
        free(arr2);
        free(sum);
        free(difference);
        free(product);
        free(quotient);

        cudaFree(d_arr1);
        cudaFree(d_arr2);
        cudaFree(d_sum);
        cudaFree(d_difference);
        cudaFree(d_product);
        cudaFree(d_quotient);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return 0;
}
