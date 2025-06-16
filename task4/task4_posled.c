#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void add(double** arr1, double** arr2, double** result, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                        result[i][j] = arr1[i][j] + arr2[i][j];
                }
        }
}

void substract(double** arr1, double** arr2, double** result, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                        result[i][j] = arr1[i][j] - arr2[i][j];
                }
        }
}

void multiply(double** arr1, double** arr2, double** result, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                        result[i][j] = arr1[i][j] * arr2[i][j];
                }
        }
}

void divide(double** arr1, double** arr2, double** result, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                        result[i][j] = arr1[i][j] / arr2[i][j];
                }
        }
}

#define NUM_RUNS 100
int main() {
        double** arr1, ** arr2, ** sum, ** difference, ** product, ** quotient;
        int i, j, rows, cols;
        clock_t start, end;
        double cpu_time_used, total_time = 0.0, average_time;
        char* endptr;

        char* array_size_str = getenv("ARRAY_SIZE");
        if (array_size_str == NULL) {
                fprintf(stderr, "Error: ARRAY_SIZE environment variable not set. \n");
                return 1;
        }
        rows = strtol(array_size_str, &endptr, 10);
        if (*endptr != '\0' || rows <= 0) {
                fprintf(stderr, "Error: Incorrect ARRAY_SIZE value: %s\n", array_size_str);
                return 1;
        }
        cols = rows;

        arr1 = (double**)malloc(rows * sizeof(double*));
        arr2 = (double**)malloc(rows * sizeof(double*));
        sum = (double**)malloc(rows * sizeof(double*));
        difference = (double**)malloc(rows * sizeof(double*));
        product = (double**)malloc(rows * sizeof(double*));
        quotient = (double**)malloc(rows * sizeof(double*));arr1 = (double**)malloc(rows * sizeof(double*));

        for (int i = 0; i < rows; ++i) {
                arr1[i] = (double*)malloc(cols * sizeof(double));
                arr2[i] = (double*)malloc(cols * sizeof(double));
                sum[i] = (double*)malloc(cols * sizeof(double));
                difference[i] = (double*)malloc(cols * sizeof(double));
                product[i] = (double*)malloc(cols * sizeof(double));
                quotient[i] = (double*)malloc(cols * sizeof(double));
        }
        if (arr1 == NULL || arr2 == NULL || sum == NULL || difference == NULL || product == NULL || quotient == NULL) {
                fprintf(stderr, "Ошибка выделения памяти\n");
                return 1;
        }

        for (j = 0; j < NUM_RUNS; ++j) {
                srand(time(NULL) + j);
                for (i = 0; i < rows; ++i) {
                        for (int k = 0; k < cols; ++k) {
                                arr1[i][k] = (double)rand() / RAND_MAX * 100.0;
                                arr2[i][k] = (double)rand() / RAND_MAX * 100.0 + 1.0;
                        }
                }
                start = clock();
                add(arr1, arr2, sum, rows, cols);
                substract(arr1, arr2, difference, rows, cols);
                multiply(arr1, arr2, product, rows, cols);
                divide(arr1, arr2, quotient, rows, cols);
                end = clock();
                cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
                total_time += cpu_time_used;
        }

        average_time = total_time / NUM_RUNS;
        printf("Average sequential execution time (over %d iterations): %f sec\n", NUM_RUNS, average_time);


        for (int i = 0; i < rows; ++i) {
                free(arr1[i]);
                free(arr2[i]);
                free(sum[i]);
                free(difference[i]);
                free(product[i]);
                free(quotient[i]);
        }

        free(arr1);
        free(arr2);
        free(sum);
        free(difference);
        free(product);
        free(quotient);
        return 0;
}
