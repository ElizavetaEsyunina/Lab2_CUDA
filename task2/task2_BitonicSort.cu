#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdlib>   // Для atoi, rand, srand
#include <ctime>     // Для time
#include <cmath>     // Для log2
#include <iomanip>   // Для std::setprecision

// CUDA заголовочный файл
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t error = call;                                                \
        if (error != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error line %d: %s\n", __LINE__,               \
                    cudaGetErrorString(error));                                  \
            exit(1);                                                             \
        }                                                                        \
    } while (0)


// CUDA Kernel for Bitonic Sort Exchange
__global__ void bitonicSortKernel(int *arr, int n, int stage, int passOfStage) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int j = idx ^ (1 << passOfStage); 

        if (j > idx) {
            if (((idx >> (stage + 1)) & 1) == 0) {
                if (arr[idx] > arr[j]) {
                    int temp = arr[idx];
                    arr[idx] = arr[j];
                    arr[j] = temp;
                }
            } else {
                if (arr[idx] < arr[j]) {
                    int temp = arr[idx];
                    arr[idx] = arr[j];
                    arr[j] = temp;
                }
            }
        }
    }
}


int main(int argc, char* argv[]) {
    std::ofstream outputFile("bitonic_sort_results.txt", std::ios::app); // Открываем файл для добавления

    if (!outputFile.is_open()) {
        std::cerr << "Ошибка открытия файла для записи." << std::endl;
        return 1;
    }

    outputFile << "Начало сортировки (Bitonic Sort - CUDA): " << std::time(nullptr) << std::endl;

    if (argc != 3) {
        outputFile << "Использование: " << argv[0] << " <размер_массива> <количество_потоков>" << std::endl;
        std::cerr << "Использование: " << argv[0] << " <размер_массива> <количество_потоков>" << std::endl;
        outputFile.close();
        return 1;
    }

    int n = std::atoi(argv[1]); // Получаем размер массива из аргумента командной строки
    int numThreads = std::atoi(argv[2]); // Получаем количество потоков из аргумента командной строки

    if (n <= 0) {
        outputFile << "Размер массива должен быть положительным числом." << std::endl;
        std::cerr << "Размер массива должен быть положительным числом." << std::endl;
        outputFile.close();
        return 1;
    }

    if ((n & (n - 1)) != 0) {
        outputFile << "Размер массива должен быть степенью 2." << std::endl;
        std::cerr << "Размер массива должен быть степенью 2." << std::endl;
        outputFile.close();
        return 1;
    }

    if (numThreads <= 0 || (numThreads & (numThreads - 1)) != 0) {
        outputFile << "Количество потоков должно быть положительной степенью 2." << std::endl;
        std::cerr << "Количество потоков должно быть положительной степенью 2." << std::endl;
        outputFile.close();
        return 1;
    }

    outputFile << "Количество элементов: " << n << std::endl;
    outputFile << "Количество потоков: " << numThreads << std::endl;

    int *arr = new int[n];

    // Инициализируем генератор случайных чисел
    std::srand(std::time(nullptr));

    // Заполняем массив случайными числами
    for (int i = 0; i < n; i++) {
        arr[i] = std::rand() % 1000; // Случайные числа от 0 до 999
    }

    // CUDA: Allocate memory on the device
    int *arr_gpu;
    CUDA_CHECK(cudaMalloc((void**)&arr_gpu, n * sizeof(int)));

    // CUDA: Copy data from host to device
    CUDA_CHECK(cudaMemcpy(arr_gpu, arr, n * sizeof(int), cudaMemcpyHostToDevice));

    auto start = std::chrono::high_resolution_clock::now();

    // CUDA:  Bitonic Sort on GPU
    int numStages = log2(n);
    int blockSize = numThreads;
    int gridSize = (n + blockSize - 1) / blockSize;

    for (int stage = 0; stage < numStages; ++stage) {
        for (int passOfStage = stage; passOfStage >= 0; --passOfStage) {
            bitonicSortKernel<<<gridSize, blockSize>>>(arr_gpu, n, stage, passOfStage);
            CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to complete
        }
    }


    auto end = std::chrono::high_resolution_clock::now();

    // CUDA: Copy data from device to host
    CUDA_CHECK(cudaMemcpy(arr, arr_gpu, n * sizeof(int), cudaMemcpyDeviceToHost));

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double seconds = static_cast<double>(duration.count()) / 1000000.0; // Преобразуем в секунды

    outputFile << "Затраченное время (секунды): " << std::fixed << std::setprecision(6) << seconds << std::endl;
    std::cout << "Время выполнения: " << std::fixed << std::setprecision(6) << seconds << " секунд" << std::endl;

    // CUDA: Free memory on the device
    CUDA_CHECK(cudaFree(arr_gpu));

    delete[] arr;
    outputFile << "----------------------------------------" << std::endl;
    outputFile.close();

    return 0;
}