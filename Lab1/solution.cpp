#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <string>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

// ============================================================================
// Константы
// ============================================================================
const int IMAGE_SIZE = 10;                      // Размер образа 10x10
const int N = IMAGE_SIZE * IMAGE_SIZE;          // 100 нейронов
const int NUM_PATTERNS = 3;                     // Количество эталонных образов
const int MAX_ITERATIONS = 1000;                // Максимум итераций воспроизведения
const int TESTS_PER_NOISE_LEVEL = 10;           // Тестов на каждый уровень шума

// Пути к папкам
const string PATTERNS_DIR = "patterns/";
const string TESTS_DIR = "tests/";

// ============================================================================
// Типы данных
// ============================================================================
using Pattern = array<int, N>;                  // Биполярный вектор {-1, 1}
using WeightMatrix = array<array<int, N>, N>;   // Матрица весов 100x100

// Названия букв для вывода
const string PATTERN_NAMES[NUM_PATTERNS] = {"D", "N", "X"};
const string PATTERN_NAMES_RU[NUM_PATTERNS] = {"Д", "Н", "Х"};

// ============================================================================
// Глобальные переменные
// ============================================================================
Pattern patterns[NUM_PATTERNS];     // Эталонные образы
WeightMatrix weights;               // Матрица весов
mt19937 rng;                        // Генератор случайных чисел

// ============================================================================
// Загрузка паттерна из файла
// ============================================================================
bool loadPattern(const string& filename, Pattern& pattern) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: не удалось открыть файл " << filename << endl;
        return false;
    }
    
    string line;
    int idx = 0;
    
    while (getline(file, line) && idx < N) {
        // Пропускаем комментарии
        if (line.empty() || line[0] == '#') continue;
        
        istringstream iss(line);
        int value;
        while (iss >> value && idx < N) {
            // Преобразуем 0/1 в биполярный формат -1/1
            pattern[idx++] = (value == 1) ? 1 : -1;
        }
    }
    
    file.close();
    return idx == N;
}

// ============================================================================
// Сохранение паттерна в файл
// ============================================================================
void savePattern(const string& filename, const Pattern& pattern, const string& header = "") {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: не удалось создать файл " << filename << endl;
        return;
    }
    
    if (!header.empty()) {
        file << "# " << header << endl;
    }
    
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            int idx = i * IMAGE_SIZE + j;
            file << (pattern[idx] == 1 ? 1 : 0);
            if (j < IMAGE_SIZE - 1) file << " ";
        }
        file << endl;
    }
    
    file.close();
}

// ============================================================================
// Инициализация эталонных образов из файлов
// ============================================================================
bool initPatterns() {
    for (int i = 0; i < NUM_PATTERNS; i++) {
        string filename = PATTERNS_DIR + PATTERN_NAMES[i] + ".txt";
        if (!loadPattern(filename, patterns[i])) {
            cerr << "Ошибка загрузки паттерна " << PATTERN_NAMES[i] << endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// Обучение сети по правилу Хебба
// ============================================================================
void train() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            weights[i][j] = 0;
        }
    }
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j) {
                for (int k = 0; k < NUM_PATTERNS; k++) {
                    weights[i][j] += patterns[k][i] * patterns[k][j];
                }
            }
        }
    }
}

// ============================================================================
// Синхронное воспроизведение
// ============================================================================
Pattern recallSync(const Pattern& input, int& iterations) {
    Pattern current = input;
    Pattern next;
    iterations = 0;
    
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        iterations++;
        
        for (int i = 0; i < N; i++) {
            int sum = 0;
            for (int j = 0; j < N; j++) {
                sum += weights[i][j] * current[j];
            }
            next[i] = (sum >= 0) ? 1 : -1;
        }
        
        if (next == current) break;
        current = next;
    }
    
    return current;
}

// ============================================================================
// Асинхронное воспроизведение
// ============================================================================
Pattern recallAsync(const Pattern& input, int& iterations) {
    Pattern current = input;
    iterations = 0;
    
    vector<int> indices(N);
    for (int i = 0; i < N; i++) indices[i] = i;
    
    bool changed = true;
    while (changed && iterations < MAX_ITERATIONS * N) {
        changed = false;
        shuffle(indices.begin(), indices.end(), rng);
        
        for (int idx : indices) {
            iterations++;
            int sum = 0;
            for (int j = 0; j < N; j++) {
                sum += weights[idx][j] * current[j];
            }
            int newValue = (sum >= 0) ? 1 : -1;
            if (newValue != current[idx]) {
                current[idx] = newValue;
                changed = true;
            }
        }
        if (!changed) break;
    }
    
    return current;
}

// ============================================================================
// Добавление шума к образу
// ============================================================================
Pattern addNoise(const Pattern& original, int noisePercent) {
    Pattern noisy = original;
    int numPixelsToFlip = (N * noisePercent) / 100;
    
    vector<int> indices(N);
    for (int i = 0; i < N; i++) indices[i] = i;
    shuffle(indices.begin(), indices.end(), rng);
    
    for (int i = 0; i < numPixelsToFlip; i++) {
        noisy[indices[i]] *= -1;
    }
    
    return noisy;
}

// ============================================================================
// Сравнение образов
// ============================================================================
int findClosestPattern(const Pattern& test) {
    int bestMatch = -1;
    int maxSimilarity = -N - 1;
    
    for (int k = 0; k < NUM_PATTERNS; k++) {
        int similarity = 0;
        for (int i = 0; i < N; i++) {
            if (test[i] == patterns[k][i]) similarity++;
        }
        if (similarity > maxSimilarity) {
            maxSimilarity = similarity;
            bestMatch = k;
        }
    }
    
    return bestMatch;
}

// ============================================================================
// Вывод образа в консоль
// ============================================================================
void printPattern(const Pattern& p, const string& title = "") {
    if (!title.empty()) cout << title << ":" << endl;
    
    for (int i = 0; i < IMAGE_SIZE; i++) {
        cout << "  ";
        for (int j = 0; j < IMAGE_SIZE; j++) {
            int idx = i * IMAGE_SIZE + j;
            cout << (p[idx] == 1 ? "##" : "  ");
        }
        cout << endl;
    }
    cout << endl;
}

// ============================================================================
// Вычисление процента сходства
// ============================================================================
double calculateSimilarity(const Pattern& a, const Pattern& b) {
    int matches = 0;
    for (int i = 0; i < N; i++) {
        if (a[i] == b[i]) matches++;
    }
    return (100.0 * matches) / N;
}

// ============================================================================
// Генерация тестовых образов и сохранение в папку tests/
// ============================================================================
void generateTestPatterns() {
    cout << "Генерация тестовых образов..." << endl;
    
    int noiseLevels[] = {10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    for (int p = 0; p < NUM_PATTERNS; p++) {
        // Создаём подпапку для каждой буквы
        string letterDir = TESTS_DIR + PATTERN_NAMES[p] + "/";
        fs::create_directories(letterDir);
        
        for (int n = 0; n < numNoiseLevels; n++) {
            int noise = noiseLevels[n];
            string noiseDir = letterDir + "noise_" + to_string(noise) + "/";
            fs::create_directories(noiseDir);
            
            for (int t = 0; t < TESTS_PER_NOISE_LEVEL; t++) {
                Pattern noisy = addNoise(patterns[p], noise);
                string filename = noiseDir + "test_" + to_string(t + 1) + ".txt";
                string header = "Буква " + PATTERN_NAMES_RU[p] + ", шум " + to_string(noise) + "%, тест " + to_string(t + 1);
                savePattern(filename, noisy, header);
            }
        }
    }
    
    cout << "Тестовые образы сохранены в папку " << TESTS_DIR << endl;
}

// ============================================================================
// Структура результатов теста
// ============================================================================
struct TestResult {
    int correctSync;
    int correctAsync;
    int total;
    double avgIterSync;
    double avgIterAsync;
    double avgSimilaritySync;
    double avgSimilarityAsync;
};

// ============================================================================
// Тестирование с загрузкой из файлов
// ============================================================================
TestResult runTestFromFiles(int patternIdx, int noisePercent) {
    TestResult result = {0, 0, 0, 0.0, 0.0, 0.0, 0.0};
    
    string noiseDir = TESTS_DIR + PATTERN_NAMES[patternIdx] + "/noise_" + to_string(noisePercent) + "/";
    
    for (int t = 0; t < TESTS_PER_NOISE_LEVEL; t++) {
        string filename = noiseDir + "test_" + to_string(t + 1) + ".txt";
        Pattern noisy;
        
        if (!loadPattern(filename, noisy)) {
            continue;
        }
        
        result.total++;
        
        // Синхронное воспроизведение
        int iterSync;
        Pattern resultSync = recallSync(noisy, iterSync);
        result.avgIterSync += iterSync;
        result.avgSimilaritySync += calculateSimilarity(resultSync, patterns[patternIdx]);
        
        if (findClosestPattern(resultSync) == patternIdx) {
            result.correctSync++;
        }
        
        // Асинхронное воспроизведение
        int iterAsync;
        Pattern resultAsync = recallAsync(noisy, iterAsync);
        result.avgIterAsync += iterAsync;
        result.avgSimilarityAsync += calculateSimilarity(resultAsync, patterns[patternIdx]);
        
        if (findClosestPattern(resultAsync) == patternIdx) {
            result.correctAsync++;
        }
    }
    
    if (result.total > 0) {
        result.avgIterSync /= result.total;
        result.avgIterAsync /= result.total;
        result.avgSimilaritySync /= result.total;
        result.avgSimilarityAsync /= result.total;
    }
    
    return result;
}

// ============================================================================
// Сохранение результатов в файл
// ============================================================================
void saveResults(const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) return;
    
    int noiseLevels[] = {10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    file << "Буква,Шум%,Sync_Успех,Sync_Итер,Sync_Сходство%,Async_Успех,Async_Итер,Async_Сходство%" << endl;
    
    for (int p = 0; p < NUM_PATTERNS; p++) {
        for (int n = 0; n < numNoiseLevels; n++) {
            TestResult res = runTestFromFiles(p, noiseLevels[n]);
            
            file << PATTERN_NAMES_RU[p] << ","
                 << noiseLevels[n] << ","
                 << res.correctSync << "/" << res.total << ","
                 << fixed << setprecision(1) << res.avgIterSync << ","
                 << res.avgSimilaritySync << ","
                 << res.correctAsync << "/" << res.total << ","
                 << res.avgIterAsync << ","
                 << res.avgSimilarityAsync << endl;
        }
    }
    
    file.close();
}

// ============================================================================
// Главная функция
// ============================================================================
int main() {
    setlocale(LC_ALL, "Russian");
    
    // Инициализация генератора случайных чисел
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    rng.seed(seed);
    
    cout << "========================================================" << endl;
    cout << "    ЛАБОРАТОРНАЯ РАБОТА №1: НЕЙРОННАЯ СЕТЬ ХОПФИЛДА" << endl;
    cout << "    Вариант 5: буквы Д, Н, Х" << endl;
    cout << "========================================================" << endl << endl;
    
    // Создание директорий
    fs::create_directories(PATTERNS_DIR);
    fs::create_directories(TESTS_DIR);
    
    // Загрузка эталонных образов
    cout << "1. Загрузка эталонных образов из " << PATTERNS_DIR << "..." << endl;
    if (!initPatterns()) {
        cerr << "Ошибка загрузки паттернов!" << endl;
        return 1;
    }
    
    // Вывод эталонных образов
    cout << "\nЭталонные образы (10x10):" << endl;
    cout << "------------------------" << endl;
    for (int i = 0; i < NUM_PATTERNS; i++) {
        printPattern(patterns[i], "Буква " + PATTERN_NAMES_RU[i]);
    }
    
    // Обучение сети
    cout << "2. Обучение сети по правилу Хебба..." << endl;
    train();
    cout << "   Матрица весов вычислена (" << N << "x" << N << ")" << endl << endl;
    
    // Генерация тестовых образов
    cout << "3. Генерация тестовых образов с разным уровнем шума..." << endl;
    generateTestPatterns();
    cout << endl;
    
    // Демонстрация работы
    cout << "4. Демонстрация распознавания (30% шума):" << endl;
    cout << "----------------------------------------" << endl;
    
    Pattern demo = addNoise(patterns[0], 30);
    cout << "Зашумленный образ буквы Д:" << endl;
    printPattern(demo);
    
    int iterDemo;
    Pattern recovered = recallSync(demo, iterDemo);
    cout << "Восстановленный образ (синхронный режим, " << iterDemo << " итераций):" << endl;
    printPattern(recovered);
    
    int recognized = findClosestPattern(recovered);
    cout << "Распознано как: " << PATTERN_NAMES_RU[recognized] << endl;
    cout << "Сходство с эталоном: " << fixed << setprecision(1) 
         << calculateSimilarity(recovered, patterns[0]) << "%" << endl << endl;
    
    // Массовое тестирование из файлов
    cout << "5. Тестирование распознавания из файлов:" << endl;
    cout << "=========================================" << endl << endl;
    
    int noiseLevels[] = {10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    // Заголовок таблицы
    cout << "┌────────┬───────┬─────────────────────────┬─────────────────────────┐" << endl;
    cout << "│ Буква  │  Шум  │    Синхронный режим     │   Асинхронный режим     │" << endl;
    cout << "│        │   %   │  Успех   │  Ср.итер.   │  Успех   │  Ср.итер.    │" << endl;
    cout << "├────────┼───────┼──────────┼─────────────┼──────────┼──────────────┤" << endl;
    
    int totalCorrectSync[12] = {0};
    int totalCorrectAsync[12] = {0};
    int totalTests[12] = {0};
    
    for (int p = 0; p < NUM_PATTERNS; p++) {
        for (int n = 0; n < numNoiseLevels; n++) {
            TestResult res = runTestFromFiles(p, noiseLevels[n]);
            
            totalCorrectSync[n] += res.correctSync;
            totalCorrectAsync[n] += res.correctAsync;
            totalTests[n] += res.total;
            
            cout << "│   " << PATTERN_NAMES_RU[p] << "    │  " 
                 << setw(3) << noiseLevels[n] << "  │  "
                 << setw(2) << res.correctSync << "/" << setw(2) << res.total << "   │   "
                 << setw(6) << fixed << setprecision(1) << res.avgIterSync << "    │  "
                 << setw(2) << res.correctAsync << "/" << setw(2) << res.total << "   │   "
                 << setw(7) << fixed << setprecision(1) << res.avgIterAsync << "    │" << endl;
        }
        if (p < NUM_PATTERNS - 1) {
            cout << "├────────┼───────┼──────────┼─────────────┼──────────┼──────────────┤" << endl;
        }
    }
    
    cout << "└────────┴───────┴──────────┴─────────────┴──────────┴──────────────┘" << endl;
    
    // Общая статистика
    cout << endl << "6. Сводная статистика по уровням шума:" << endl;
    cout << "=======================================" << endl << endl;
    
    cout << "┌───────┬─────────────────┬─────────────────┐" << endl;
    cout << "│  Шум  │   Синхронный    │  Асинхронный    │" << endl;
    cout << "│   %   │   % успеха      │   % успеха      │" << endl;
    cout << "├───────┼─────────────────┼─────────────────┤" << endl;
    
    for (int n = 0; n < numNoiseLevels; n++) {
        double successSync = totalTests[n] > 0 ? (100.0 * totalCorrectSync[n]) / totalTests[n] : 0;
        double successAsync = totalTests[n] > 0 ? (100.0 * totalCorrectAsync[n]) / totalTests[n] : 0;
        
        cout << "│  " << setw(3) << noiseLevels[n] << "  │      "
             << setw(5) << fixed << setprecision(1) << successSync << "%      │      "
             << setw(5) << fixed << setprecision(1) << successAsync << "%      │" << endl;
    }
    
    cout << "└───────┴─────────────────┴─────────────────┘" << endl;
    
    // Сохранение результатов
    cout << endl << "7. Сохранение результатов..." << endl;
    saveResults(TESTS_DIR + "results.csv");
    cout << "   Результаты сохранены в " << TESTS_DIR << "results.csv" << endl;
    
    // Выводы
    cout << endl << "8. Выводы:" << endl;
    cout << "==========" << endl;
    cout << "- Сеть Хопфилда успешно обучена на 3 образах (Д, Н, Х)" << endl;
    cout << "- При низком уровне шума (до 30%) распознавание работает хорошо" << endl;
    cout << "- При шуме выше 40-50% качество распознавания значительно падает" << endl;
    cout << "- Асинхронный режим обычно требует больше итераций" << endl;
    
    return 0;
}
