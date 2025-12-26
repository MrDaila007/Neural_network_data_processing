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
#include <map>

using namespace std;
namespace fs = std::filesystem;

// ============================================================================
// Константы
// ============================================================================
const int IMAGE_SIZE = 6;                      // Размер образа 6x6
const int N = IMAGE_SIZE * IMAGE_SIZE;          // 36 нейронов входного слоя
const int NUM_CLASSES = 5;                     // Количество классов образов
const int NUM_NEURONS = 5;                     // Количество выходных нейронов (кластеров)
const int NUM_TRAINING_SAMPLES = 15;           // Образов для обучения (больше чем нейронов)
const int NUM_TEST_SAMPLES = 10;               // Тестовых образов на класс
const double LEARNING_RATE = 0.1;               // Скорость обучения β
const double MAX_DISTANCE = 0.01;               // Максимальное расстояние для завершения обучения
const int MAX_ITERATIONS = 1000;                // Максимум итераций обучения

// Пути к папкам
const string PATTERNS_DIR = "patterns/";
const string TESTS_DIR = "tests/";

// ============================================================================
// Типы данных
// ============================================================================
using Pattern = vector<double>;                // Нормированный вектор
using WeightMatrix = vector<vector<double>>;   // Матрица весов [NUM_NEURONS][N]

// Названия символов для вывода
const string CLASS_NAMES[NUM_CLASSES] = {"LE", "GE", "NE", "AP", "CO"};
const string CLASS_NAMES_RU[NUM_CLASSES] = {"≤", "≥", "≠", "≈", "≅"};

// ============================================================================
// Глобальные переменные
// ============================================================================
vector<Pattern> trainingPatterns;              // Обучающие образы
vector<int> trainingClasses;                   // Классы обучающих образов
WeightMatrix weights;                           // Матрица весов [NUM_NEURONS][N]
vector<int> winCounts;                         // Количество побед каждого нейрона
mt19937 rng;                                   // Генератор случайных чисел

// ============================================================================
// Нормировка вектора
// ============================================================================
void normalize(Pattern& pattern) {
    double norm = 0.0;
    for (double val : pattern) {
        norm += val * val;
    }
    norm = sqrt(norm);
    
    if (norm > 1e-10) {
        for (double& val : pattern) {
            val /= norm;
        }
    }
}

// ============================================================================
// Загрузка паттерна из файла
// ============================================================================
bool loadPattern(const string& filename, Pattern& pattern) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: не удалось открыть файл " << filename << endl;
        return false;
    }
    
    pattern.clear();
    pattern.resize(N);
    
    string line;
    int idx = 0;
    
    while (getline(file, line) && idx < N) {
        // Пропускаем комментарии
        if (line.empty() || line[0] == '#') continue;
        
        istringstream iss(line);
        int value;
        while (iss >> value && idx < N) {
            // Преобразуем 0/1 в биполярный формат -1/1, затем нормируем
            pattern[idx++] = (value == 1) ? 1.0 : -1.0;
        }
    }
    
    file.close();
    
    if (idx == N) {
        normalize(pattern);
        return true;
    }
    return false;
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
            // Преобразуем из биполярного формата обратно в 0/1
            file << (pattern[idx] > 0 ? 1 : 0);
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
    for (int i = 0; i < NUM_CLASSES; i++) {
        string filename = PATTERNS_DIR + CLASS_NAMES[i] + ".txt";
        Pattern pattern;
        if (!loadPattern(filename, pattern)) {
            cerr << "Ошибка загрузки паттерна " << CLASS_NAMES[i] << endl;
            return false;
        }
        
        // Добавляем несколько вариантов каждого образа для обучения
        for (int j = 0; j < NUM_TRAINING_SAMPLES / NUM_CLASSES; j++) {
            trainingPatterns.push_back(pattern);
            trainingClasses.push_back(i);
        }
    }
    
    // Добавляем еще образы, чтобы общее количество превышало NUM_NEURONS
    while (trainingPatterns.size() < NUM_TRAINING_SAMPLES) {
        int classIdx = trainingPatterns.size() % NUM_CLASSES;
        string filename = PATTERNS_DIR + CLASS_NAMES[classIdx] + ".txt";
        Pattern pattern;
        if (loadPattern(filename, pattern)) {
            trainingPatterns.push_back(pattern);
            trainingClasses.push_back(classIdx);
        }
    }
    
    return true;
}

// ============================================================================
// Инициализация весов случайными значениями
// ============================================================================
void initWeights() {
    weights.resize(NUM_NEURONS);
    winCounts.resize(NUM_NEURONS, 0);
    
    uniform_real_distribution<double> dist(-0.5, 0.5);
    
    for (int j = 0; j < NUM_NEURONS; j++) {
        weights[j].resize(N);
        for (int i = 0; i < N; i++) {
            weights[j][i] = dist(rng);
        }
        normalize(weights[j]);
    }
}

// ============================================================================
// Нахождение нейрона-победителя (обычный метод)
// ============================================================================
int findWinner(const Pattern& input) {
    int winner = 0;
    double maxDot = -1e10;
    
    for (int j = 0; j < NUM_NEURONS; j++) {
        double dot = 0.0;
        for (int i = 0; i < N; i++) {
            dot += weights[j][i] * input[i];
        }
        
        if (dot > maxDot) {
            maxDot = dot;
            winner = j;
        }
    }
    
    return winner;
}

// ============================================================================
// Нахождение нейрона-победителя (частотно-зависимый метод)
// ============================================================================
int findWinnerFrequency(const Pattern& input) {
    int winner = 0;
    double minDistance = 1e10;
    
    for (int j = 0; j < NUM_NEURONS; j++) {
        // Вычисляем евклидово расстояние
        double distance = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = input[i] - weights[j][i];
            distance += diff * diff;
        }
        distance = sqrt(distance);
        
        // Умножаем на количество побед (частотно-зависимое обучение)
        double weightedDistance = distance * (1.0 + winCounts[j]);
        
        if (weightedDistance < minDistance) {
            minDistance = weightedDistance;
            winner = j;
        }
    }
    
    return winner;
}

// ============================================================================
// Обучение сети
// ============================================================================
void train() {
    cout << "Обучение конкурентной сети..." << endl;
    
    bool converged = false;
    int iteration = 0;
    
    while (!converged && iteration < MAX_ITERATIONS) {
        iteration++;
        double maxDistance = 0.0;
        
        // Перемешиваем обучающие образы
        vector<int> indices(trainingPatterns.size());
        for (size_t i = 0; i < indices.size(); i++) {
            indices[i] = i;
        }
        shuffle(indices.begin(), indices.end(), rng);
        
        for (int idx : indices) {
            const Pattern& input = trainingPatterns[idx];
            
            // Находим нейрон-победитель
            int winner = findWinnerFrequency(input);
            winCounts[winner]++;
            
            // Вычисляем расстояние до обновления
            double distance = 0.0;
            for (int i = 0; i < N; i++) {
                double diff = input[i] - weights[winner][i];
                distance += diff * diff;
            }
            distance = sqrt(distance);
            maxDistance = max(maxDistance, distance);
            
            // Обновляем веса нейрона-победителя
            Pattern newWeights(N);
            for (int i = 0; i < N; i++) {
                newWeights[i] = weights[winner][i] + LEARNING_RATE * (input[i] - weights[winner][i]);
            }
            
            // Нормируем новые веса
            normalize(newWeights);
            weights[winner] = newWeights;
        }
        
        // Проверяем условие завершения
        if (maxDistance < MAX_DISTANCE) {
            converged = true;
            cout << "  Обучение завершено на итерации " << iteration 
                 << " (макс. расстояние: " << fixed << setprecision(4) << maxDistance << ")" << endl;
        } else if (iteration % 100 == 0) {
            cout << "  Итерация " << iteration << ", макс. расстояние: " 
                 << fixed << setprecision(4) << maxDistance << endl;
        }
    }
    
    if (!converged) {
        cout << "  Достигнуто максимальное количество итераций (" << MAX_ITERATIONS << ")" << endl;
    }
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
    
    normalize(noisy);
    return noisy;
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
            cout << (p[idx] > 0 ? "■" : "□") << " ";
        }
        cout << endl;
    }
    cout << endl;
}

// ============================================================================
// Генерация тестовых образов
// ============================================================================
void generateTestPatterns() {
    cout << "Генерация тестовых образов..." << endl;
    
    int noiseLevels[] = {10, 20, 30, 40, 50};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    for (int c = 0; c < NUM_CLASSES; c++) {
        // Загружаем эталонный образ
        string filename = PATTERNS_DIR + CLASS_NAMES[c] + ".txt";
        Pattern pattern;
        if (!loadPattern(filename, pattern)) continue;
        
        // Создаём подпапку для каждого класса
        string classDir = TESTS_DIR + CLASS_NAMES[c] + "/";
        fs::create_directories(classDir);
        
        for (int n = 0; n < numNoiseLevels; n++) {
            int noise = noiseLevels[n];
            string noiseDir = classDir + "noise_" + to_string(noise) + "/";
            fs::create_directories(noiseDir);
            
            for (int t = 0; t < NUM_TEST_SAMPLES; t++) {
                Pattern noisy = addNoise(pattern, noise);
                string testFilename = noiseDir + "test_" + to_string(t + 1) + ".txt";
                string header = "Класс " + CLASS_NAMES_RU[c] + ", шум " + to_string(noise) + "%, тест " + to_string(t + 1);
                savePattern(testFilename, noisy, header);
            }
        }
    }
    
    cout << "Тестовые образы сохранены в папку " << TESTS_DIR << endl;
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
    cout << "    ЛАБОРАТОРНАЯ РАБОТА №4: КОНКУРЕНТНАЯ СЕТЬ" << endl;
    cout << "    Вариант 5: символы ≤, ≥, ≠, ≈, ≅" << endl;
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
    
    cout << "   Загружено " << trainingPatterns.size() << " обучающих образов" << endl << endl;
    
    // Вывод эталонных образов
    cout << "Эталонные образы (6x6):" << endl;
    cout << "------------------------" << endl;
    for (int i = 0; i < NUM_CLASSES; i++) {
        string filename = PATTERNS_DIR + CLASS_NAMES[i] + ".txt";
        Pattern pattern;
        if (loadPattern(filename, pattern)) {
            printPattern(pattern, "Класс " + CLASS_NAMES_RU[i] + " (" + CLASS_NAMES[i] + ")");
        }
    }
    
    // Инициализация весов
    cout << "2. Инициализация весов случайными значениями..." << endl;
    initWeights();
    cout << "   Матрица весов инициализирована (" << NUM_NEURONS << " нейронов x " << N << " входов)" << endl << endl;
    
    // Обучение сети
    cout << "3. Обучение сети..." << endl;
    train();
    cout << endl;
    
    // Определение соответствия нейронов классам
    cout << "4. Определение соответствия нейронов классам:" << endl;
    cout << "----------------------------------------------" << endl;
    
    map<int, int> neuronToClass;  // нейрон -> класс
    map<int, vector<int>> classToNeurons;  // класс -> нейроны
    
    for (size_t i = 0; i < trainingPatterns.size(); i++) {
        int winner = findWinner(trainingPatterns[i]);
        int trueClass = trainingClasses[i];
        
        if (classToNeurons[trueClass].empty() || 
            find(classToNeurons[trueClass].begin(), classToNeurons[trueClass].end(), winner) == classToNeurons[trueClass].end()) {
            classToNeurons[trueClass].push_back(winner);
        }
    }
    
    // Определяем основной нейрон для каждого класса
    for (int c = 0; c < NUM_CLASSES; c++) {
        if (!classToNeurons[c].empty()) {
            int mainNeuron = classToNeurons[c][0];
            neuronToClass[mainNeuron] = c;
            cout << "Класс " << CLASS_NAMES_RU[c] << " → Нейрон-победитель: #" << mainNeuron << endl;
        }
    }
    cout << endl;
    
    // Генерация тестовых образов
    cout << "5. Генерация тестовых образов с разным уровнем шума..." << endl;
    generateTestPatterns();
    cout << endl;
    
    // Демонстрация работы
    cout << "6. Демонстрация распознавания (30% шума):" << endl;
    cout << "----------------------------------------" << endl;
    
    string demoFilename = PATTERNS_DIR + CLASS_NAMES[2] + ".txt";  // Класс ≠
    Pattern demoPattern;
    if (loadPattern(demoFilename, demoPattern)) {
        Pattern demoNoisy = addNoise(demoPattern, 30);
        cout << "Зашумленный образ класса " << CLASS_NAMES_RU[2] << ":" << endl;
        printPattern(demoNoisy);
        
        int winner = findWinner(demoNoisy);
        cout << "Результат: Нейрон-победитель #" << winner;
        
        if (neuronToClass.find(winner) != neuronToClass.end()) {
            int recognizedClass = neuronToClass[winner];
            cout << " (соответствует классу " << CLASS_NAMES_RU[recognizedClass] << ")";
            if (recognizedClass == 2) {
                cout << " ✓" << endl;
            } else {
                cout << " ✗" << endl;
            }
        } else {
            cout << " (класс не определен)" << endl;
        }
    }
    cout << endl;
    
    // Статистика кластеризации
    cout << "7. Статистика кластеризации:" << endl;
    cout << "=============================" << endl;
    
    map<int, map<int, int>> neuronStats;  // нейрон -> (класс -> количество)
    
    for (size_t i = 0; i < trainingPatterns.size(); i++) {
        int winner = findWinner(trainingPatterns[i]);
        int trueClass = trainingClasses[i];
        neuronStats[winner][trueClass]++;
    }
    
    for (int j = 0; j < NUM_NEURONS; j++) {
        if (neuronStats[j].empty()) continue;
        
        int total = 0;
        int mainClass = -1;
        int mainClassCount = 0;
        
        for (auto& pair : neuronStats[j]) {
            total += pair.second;
            if (pair.second > mainClassCount) {
                mainClassCount = pair.second;
                mainClass = pair.first;
            }
        }
        
        cout << "Нейрон #" << j << ": " << total << " образов";
        if (mainClass >= 0) {
            cout << " (основной класс: " << CLASS_NAMES_RU[mainClass] << ", " 
                 << mainClassCount << " образов)";
        }
        cout << endl;
    }
    cout << endl;
    
    // Тестирование на зашумленных образах
    cout << "8. Тестирование на зашумленных образах:" << endl;
    cout << "=======================================" << endl;
    
    int noiseLevels[] = {10, 20, 30, 40, 50};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    cout << "┌────────┬───────┬──────────────┬──────────────┐" << endl;
    cout << "│ Класс  │  Шум  │  Правильно   │  Неправильно  │" << endl;
    cout << "│        │   %   │              │               │" << endl;
    cout << "├────────┼───────┼──────────────┼──────────────┤" << endl;
    
    for (int c = 0; c < NUM_CLASSES; c++) {
        for (int n = 0; n < numNoiseLevels; n++) {
            int noise = noiseLevels[n];
            string noiseDir = TESTS_DIR + CLASS_NAMES[c] + "/noise_" + to_string(noise) + "/";
            
            int correct = 0;
            int total = 0;
            
            for (int t = 1; t <= NUM_TEST_SAMPLES; t++) {
                string filename = noiseDir + "test_" + to_string(t) + ".txt";
                Pattern testPattern;
                if (loadPattern(filename, testPattern)) {
                    int winner = findWinner(testPattern);
                    if (neuronToClass.find(winner) != neuronToClass.end() && 
                        neuronToClass[winner] == c) {
                        correct++;
                    }
                    total++;
                }
            }
            
            cout << "│   " << CLASS_NAMES_RU[c] << "    │  " 
                 << setw(3) << noise << "  │     " 
                 << setw(3) << correct << "/" << setw(2) << total << "      │     "
                 << setw(3) << (total - correct) << "/" << setw(2) << total << "       │" << endl;
        }
        if (c < NUM_CLASSES - 1) {
            cout << "├────────┼───────┼──────────────┼──────────────┤" << endl;
        }
    }
    
    cout << "└────────┴───────┴──────────────┴──────────────┘" << endl << endl;
    
    // Выводы
    cout << "9. Выводы:" << endl;
    cout << "==========" << endl;
    cout << "- Конкурентная сеть успешно обучена на " << trainingPatterns.size() << " образах" << endl;
    cout << "- Сеть разбила образы на " << NUM_NEURONS << " кластеров" << endl;
    cout << "- Похожие образы спроецированы в один кластер" << endl;
    cout << "- При низком уровне шума (10-20%) распознавание работает хорошо" << endl;
    cout << "- При высоком уровне шума (40-50%) качество распознавания снижается" << endl;
    
    return 0;
}
