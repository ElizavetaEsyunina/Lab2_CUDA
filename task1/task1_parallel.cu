#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_VALUE 1000
#define NUM_RUNS 100
#define DEFAULT_BLOCK_SIZE 256

__global__ void sumArrayKernel(int *array, long long *sum, int array_size) {
    extern __shared__ long long shared_data[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = (i < array_size) ? array[i] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum[blockIdx.x] = shared_data[0];
    }
}

int main() {
    int array_size;
    int threadsPerBlock = DEFAULT_BLOCK_SIZE;

    char* array_size_str = getenv("ARRAY_SIZE");
    if (array_size_str == NULL) {
        fprintf(stderr, "Переменная окружения ARRAY_SIZE не установлена\n");
        return 1;
    }

    char* endptr;
    array_size = strtol(array_size_str, &endptr, 10);
    if (*endptr != '\0' || array_size <= 0) {
        fprintf(stderr, "Некорректное значение ARRAY_SIZE: %s\n", array_size_str);
        return 1;
    }

    char* block_size_str = getenv("BLOCK_SIZE");
    if (block_size_str != NULL) {
        threadsPerBlock = strtol(block_size_str, &endptr, 10);
        if (*endptr != '\0' || threadsPerBlock <= 0 || threadsPerBlock > 1024) {
            fprintf(stderr, "Некорректное значение BLOCK_SIZE: %s (должно быть 1-1024)\n", block_size_str);
            return 1;
        }
    }

    if ((threadsPerBlock & (threadsPerBlock - 1)) != 0) {
        fprintf(stderr, "Предупреждение: BLOCK_SIZE (%d) не является степенью двойки, что может снизить производительность\n", threadsPerBlock);
    }

    double total_time = 0.0;
    int blocksPerGrid = (array_size + threadsPerBlock - 1) / threadsPerBlock;
    int num_threads_used = blocksPerGrid * threadsPerBlock;

    printf("Конфигурация CUDA:\n");
    printf("  Размер массива: %d элементов\n", array_size);
    printf("  Потоков в блоке: %d\n", threadsPerBlock);
    printf("  Количество блоков: %d\n", blocksPerGrid);
    printf("  Всего потоков: %d\n\n", num_threads_used);

    for (int run = 0; run < NUM_RUNS; run++) {
        int *array = (int*)malloc(array_size * sizeof(int));
        if (array == NULL) {
            fprintf(stderr, "Ошибка выделения памяти\n");
            return 1;
        }

        srand(time(NULL) + run);
        for (int i = 0; i < array_size; i++) {
            array[i] = rand() % MAX_VALUE;
        }

        int *d_array;
        long long *d_sum;
        long long *h_sum = (long long*)malloc(blocksPerGrid * sizeof(long long));
        
        cudaMalloc((void**)&d_array, array_size * sizeof(int));
        cudaMalloc((void**)&d_sum, blocksPerGrid * sizeof(long long));

        cudaMemcpy(d_array, array, array_size * sizeof(int), cudaMemcpyHostToDevice);

        clock_t start = clock();
        sumArrayKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(long long)>>>(d_array, d_sum, array_size);
        cudaDeviceSynchronize();
        clock_t end = clock();

        cudaMemcpy(h_sum, d_sum, blocksPerGrid * sizeof(long long), cudaMemcpyDeviceToHost);

        long long total_sum = 0;
        for (int i = 0; i < blocksPerGrid; i++) {
            total_sum += h_sum[i];
        }

        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        total_time += time_spent;

        printf("Запуск %d:\n", run + 1);
        printf("  Сумма: %lld\n", total_sum);
        printf("  Время: %f секунд\n", time_spent);

        free(array);
        free(h_sum);
        cudaFree(d_array);
        cudaFree(d_sum);
    }

    double average_time = total_time / NUM_RUNS;
    printf("\nИтоговые результаты:\n");
    printf("  Среднее время выполнения: %f секунд\n", average_time);
    printf("  Использовано потоков: %d\n", num_threads_used);
    printf("  Конфигурация: %d блоков × %d потоков\n", blocksPerGrid, threadsPerBlock);

    return 0;
}