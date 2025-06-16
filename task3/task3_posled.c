#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void add(double *arr1, double *arr2, double *result, int size) {
        for (int i = 0; i < size; ++i) {
                result[i] = arr1[i] + arr2[i];
        }
}

void substract(double *arr1, double *arr2, double *result, int size) {
        for (int i = 0; i < size; ++i) {
                result[i] = arr1[i] - arr2[i];
        }
}

void multiply(double *arr1, double *arr2, double *result, int size) {
        for (int i = 0; i < size; ++i) {
                result[i] = arr1[i] * arr2[i];
        }
}

void divide(double *arr1, double *arr2, double *result, int size) {
        for (int i = 0; i < size; ++i) {
                result[i] = arr1[i] / arr2[i];
        }
}

int main() {
        double *arr1, *arr2, *sum, *difference, *product, *quotient;
        int i, j, array_size, num_iterations;
        clock_t start, end;
        double cpu_time_used, total_time = 0.0, average_time;
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

        arr1 = (double *)malloc(array_size * sizeof(double));
        arr2 = (double *)malloc(array_size * sizeof(double));
        sum = (double *)malloc(array_size * sizeof(double));
        difference = (double *)malloc(array_size * sizeof(double));
        product = (double *)malloc(array_size * sizeof(double));
        quotient = (double *)malloc(array_size * sizeof(double));

        if (arr1 == NULL || arr2 == NULL || sum == NULL || difference == NULL || product == NULL || quotient == NULL) {
                fprintf(stderr, "Ошибка выделения памяти\n");
                return 1;
        }

        for (j = 0; j < num_iterations; ++j) {
                srand(time(NULL)+j);
                for (i = 0; i < array_size; ++i) {
                        arr1[i] = (double)rand() / RAND_MAX * 100.0;
                        arr2[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
                }

                start = clock();

                add(arr1, arr2, sum, array_size);
                substract(arr1, arr2, difference, array_size);
                multiply(arr1, arr2, product, array_size);
                divide(arr1, arr2, quotient, array_size);

                end = clock();
                cpu_time_used = ((double)(end-start)) / CLOCKS_PER_SEC;
                total_time += cpu_time_used;
        }

        average_time = total_time / num_iterations;
        printf("Average sequential execution time (over %d iterations): %f sec\n", num_iterations, average_time);

        free(arr1);
        free(arr2);
        free(sum);
        free(difference);
        free(product);
        free(quotient);

        return 0;
}
