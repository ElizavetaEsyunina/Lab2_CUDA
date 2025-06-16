#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdlib>   // Для atoi, rand, srand
#include <ctime>     // Для time
#include <iomanip>   // Для std::setprecision

// Последовательная сортировка слиянием
void merge(std::vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    // Создаем временные массивы
    std::vector<int> L(n1);
    std::vector<int> R(n2);

    // Копируем данные во временные массивы
    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    // Слияние временных массивов обратно в arr[l..r]
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Копируем оставшиеся элементы из L, если есть
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Копируем оставшиеся элементы из R, если есть
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Рекурсивная функция сортировки слиянием
void mergeSort(std::vector<int>& arr, int l, int r) {
    if (l < r) {
        // Находим середину
        int m = l + (r - l) / 2;

        // Сортируем первую и вторую половины
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        // Слияние отсортированных половин
        merge(arr, l, m, r);
    }
}

int main(int argc, char* argv[]) {
    std::ofstream outputFile("merge_sort_results.txt", std::ios::app); // Открываем файл для добавления

    if (!outputFile.is_open()) {
        std::cerr << "Ошибка открытия файла для записи." << std::endl;
        return 1;
    }

    outputFile << "Начало сортировки (Merge Sort): " << std::time(nullptr) << std::endl;

    if (argc != 2) {
        outputFile << "Использование: " << argv[0] << " <размер_массива>" << std::endl;
        std::cerr << "Использование: " << argv[0] << " <размер_массива>" << std::endl;
        outputFile.close();
        return 1;
    }

    int n = std::atoi(argv[1]); // Получаем размер массива из аргумента командной строки

    if (n <= 0) {
        outputFile << "Размер массива должен быть положительным числом." << std::endl;
        std::cerr << "Размер массива должен быть положительным числом." << std::endl;
        outputFile.close();
        return 1;
    }

    outputFile << "Количество элементов: " << n << std::endl;

    std::vector<int> arr(n);

    // Инициализируем генератор случайных чисел
    std::srand(std::time(nullptr));

    // Заполняем массив случайными числами
    for (int i = 0; i < n; i++) {
        arr[i] = std::rand() % 1000; // Случайные числа от 0 до 999
    }

    auto start = std::chrono::high_resolution_clock::now();
    mergeSort(arr, 0, n - 1);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double seconds = static_cast<double>(duration.count()) / 1000000.0; // Преобразуем в секунды

    outputFile << "Затраченное время (секунды): " << std::fixed << std::setprecision(6) << seconds << std::endl;
    std::cout << "Время выполнения: " << std::fixed << std::setprecision(6) << seconds << " секунд" << std::endl;

    outputFile << "----------------------------------------" << std::endl;
    outputFile.close();

    return 0;
}
