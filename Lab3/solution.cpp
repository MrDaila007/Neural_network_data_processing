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
#include <limits>

using namespace std;
namespace fs = std::filesystem;

// ============================================================================
// Константы
// ============================================================================
const int IMAGE_SIZE = 6;                       // Размер образа 6x6
const int INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE; // 36 входов
const int NUM_CLASSES = 5;                      // Количество классов
const int NUM_PATTERNS_PER_CLASS = 3;           // Тестовых образов на класс
const int MAX_TRAINING_ITERATIONS = 10000;      // Максимум итераций обучения
const double LEARNING_RATE = 0.1;               // Скорость обучения
const double EPSILON = 1e-6;                    // Точность для остановки обучения

// Пути к папкам
const string PATTERNS_DIR = "patterns/";
const string TESTS_DIR = "tests/";

// ============================================================================
// Типы данных
// ============================================================================
using Pattern = array<double, INPUT_SIZE>;      // Входной вектор (0 или 1)
using RBFNeuron = struct {
    Pattern center;                              // Центр РБФ ячейки
    double sigma;                                // Параметр разброса
};
using WeightMatrix = vector<vector<double>>;     // Веса выходного слоя

// Названия классов для вывода
const string CLASS_NAMES[NUM_CLASSES] = {"N", "F", "I", "P", "D"};
const string CLASS_NAMES_RU[NUM_CLASSES] = {"N", "F", "I", "P", "D"};

// ============================================================================
// Глобальные переменные
// ============================================================================
Pattern patterns[NUM_CLASSES];                  // Идеальные образы для обучения
vector<RBFNeuron> rbfNeurons;                   // РБФ ячейки (по одной на класс)
WeightMatrix outputWeights;                     // Веса выходного слоя
mt19937 rng;                                    // Генератор случайных чисел

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
    
    while (getline(file, line) && idx < INPUT_SIZE) {
        // Пропускаем комментарии
        if (line.empty() || line[0] == '#') continue;
        
        istringstream iss(line);
        int value;
        while (iss >> value && idx < INPUT_SIZE) {
            // Преобразуем 0/1 в формат 0.0/1.0
            pattern[idx++] = (value == 1) ? 1.0 : 0.0;
        }
    }
    
    file.close();
    return idx == INPUT_SIZE;
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
            file << (pattern[idx] > 0.5 ? 1 : 0);
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
        if (!loadPattern(filename, patterns[i])) {
            cerr << "Ошибка загрузки паттерна " << CLASS_NAMES[i] << endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// Вычисление евклидова расстояния
// ============================================================================
double euclideanDistance(const Pattern& a, const Pattern& b) {
    double sum = 0.0;
    for (int i = 0; i < INPUT_SIZE; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// ============================================================================
// Инициализация РБФ ячеек
// ============================================================================
void initRBFNeurons() {
    rbfNeurons.clear();
    rbfNeurons.resize(NUM_CLASSES);
    
    // Центры РБФ ячеек = идеальные образы классов
    for (int i = 0; i < NUM_CLASSES; i++) {
        rbfNeurons[i].center = patterns[i];
    }
    
    // Вычисление параметров разброса sigma
    // sigma_j = половина расстояния до ближайшего центра другого класса
    for (int i = 0; i < NUM_CLASSES; i++) {
        double minDist = numeric_limits<double>::max();
        
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (i != j) {
                double dist = euclideanDistance(patterns[i], patterns[j]);
                if (dist < minDist) {
                    minDist = dist;
                }
            }
        }
        
        // Если все центры одинаковые (не должно быть), используем значение по умолчанию
        if (minDist < EPSILON) {
            rbfNeurons[i].sigma = 1.0;
        } else {
            rbfNeurons[i].sigma = minDist / 2.0;
        }
    }
}

// ============================================================================
// Вычисление выхода РБФ ячейки (гауссов колокол)
// ============================================================================
double rbfOutput(const Pattern& input, const RBFNeuron& neuron) {
    double distSquared = 0.0;
    for (int i = 0; i < INPUT_SIZE; i++) {
        double diff = input[i] - neuron.center[i];
        distSquared += diff * diff;
    }
    
    double sigmaSquared = neuron.sigma * neuron.sigma;
    return exp(-distSquared / sigmaSquared);
}

// ============================================================================
// Вычисление выхода сети РБФ
// ============================================================================
vector<double> computeRBFOutput(const Pattern& input) {
    vector<double> rbfOutputs(NUM_CLASSES);
    for (int i = 0; i < NUM_CLASSES; i++) {
        rbfOutputs[i] = rbfOutput(input, rbfNeurons[i]);
    }
    return rbfOutputs;
}

// ============================================================================
// Вычисление выхода выходного слоя
// ============================================================================
vector<double> computeOutputLayer(const vector<double>& rbfOutputs) {
    vector<double> outputs(NUM_CLASSES, 0.0);
    
    for (int k = 0; k < NUM_CLASSES; k++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            outputs[k] += outputWeights[j][k] * rbfOutputs[j];
        }
    }
    
    return outputs;
}

// ============================================================================
// Обучение выходного слоя градиентным спуском
// ============================================================================
int trainOutputLayer() {
    // Инициализация весов случайными значениями
    outputWeights.assign(NUM_CLASSES, vector<double>(NUM_CLASSES, 0.0));
    uniform_real_distribution<double> dist(-0.1, 0.1);
    for (int j = 0; j < NUM_CLASSES; j++) {
        for (int k = 0; k < NUM_CLASSES; k++) {
            outputWeights[j][k] = dist(rng);
        }
    }
    
    int iterations = 0;
    double prevError = numeric_limits<double>::max();
    
    for (int iter = 0; iter < MAX_TRAINING_ITERATIONS; iter++) {
        iterations++;
        double totalError = 0.0;
        
        // Проход по всем обучающим образам
        for (int classIdx = 0; classIdx < NUM_CLASSES; classIdx++) {
            // Вычисляем выходы РБФ слоя
            vector<double> rbfOuts = computeRBFOutput(patterns[classIdx]);
            
            // Вычисляем выходы выходного слоя
            vector<double> outputs = computeOutputLayer(rbfOuts);
            
            // Ожидаемый выход: 1.0 для правильного класса, 0.0 для остальных
            vector<double> targets(NUM_CLASSES, 0.0);
            targets[classIdx] = 1.0;
            
            // Коррекция весов для каждого выходного нейрона
            for (int k = 0; k < NUM_CLASSES; k++) {
                double error = targets[k] - outputs[k];
                totalError += error * error;
                
                // Градиентный спуск: w_jk := w_jk + α * d_k * g_j
                for (int j = 0; j < NUM_CLASSES; j++) {
                    outputWeights[j][k] += LEARNING_RATE * error * rbfOuts[j];
                }
            }
        }
        
        totalError /= NUM_CLASSES;
        
        // Проверка на сходимость
        if (abs(prevError - totalError) < EPSILON) {
            break;
        }
        prevError = totalError;
    }
    
    return iterations;
}

// ============================================================================
// Добавление шума к образу
// ============================================================================
Pattern addNoise(const Pattern& original, int noisePercent) {
    Pattern noisy = original;
    int numPixelsToFlip = (INPUT_SIZE * noisePercent) / 100;
    
    vector<int> indices(INPUT_SIZE);
    for (int i = 0; i < INPUT_SIZE; i++) indices[i] = i;
    shuffle(indices.begin(), indices.end(), rng);
    
    for (int i = 0; i < numPixelsToFlip; i++) {
        // Инверсия пикселя: 0 -> 1, 1 -> 0
        noisy[indices[i]] = (noisy[indices[i]] > 0.5) ? 0.0 : 1.0;
    }
    
    return noisy;
}

// ============================================================================
// Распознавание образа
// ============================================================================
vector<double> recognize(const Pattern& input) {
    vector<double> rbfOuts = computeRBFOutput(input);
    vector<double> outputs = computeOutputLayer(rbfOuts);
    
    // Преобразуем выходы в положительные значения (softmax-like normalization)
    // Используем экспоненту для нормализации, чтобы получить проценты
    double sum = 0.0;
    vector<double> expOutputs(NUM_CLASSES);
    
    // Находим максимум для численной стабильности
    double maxOutput = *max_element(outputs.begin(), outputs.end());
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        expOutputs[i] = exp(outputs[i] - maxOutput);
        sum += expOutputs[i];
    }
    
    // Нормализация в проценты
    vector<double> percentages(NUM_CLASSES);
    if (sum > EPSILON) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            percentages[i] = (expOutputs[i] / sum) * 100.0;
        }
    } else {
        // Если сумма нулевая, равномерное распределение
        for (int i = 0; i < NUM_CLASSES; i++) {
            percentages[i] = 100.0 / NUM_CLASSES;
        }
    }
    
    return percentages;
}

// ============================================================================
// Вывод образа в консоль
// ============================================================================
void printPattern(const Pattern& p, const string& title = "") {
    if (!title.empty()) cout << title << ":" << endl;
    
    for (int i = 0; i < IMAGE_SIZE; i++) {
        cout << "│   ";
        for (int j = 0; j < IMAGE_SIZE; j++) {
            int idx = i * IMAGE_SIZE + j;
            cout << (p[idx] > 0.5 ? "■" : "□") << " ";
        }
        cout << "                                  │" << endl;
    }
}

// ============================================================================
// Генерация тестовых образов
// ============================================================================
void generateTestPatterns() {
    cout << "Генерация тестовых образов..." << endl;
    
    int noiseLevels[] = {10, 20, 30, 40, 50};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    for (int c = 0; c < NUM_CLASSES; c++) {
        // Создаём подпапку для каждого класса
        string classDir = TESTS_DIR + CLASS_NAMES[c] + "/";
        fs::create_directories(classDir);
        
        for (int n = 0; n < numNoiseLevels; n++) {
            int noise = noiseLevels[n];
            string noiseDir = classDir + "noise_" + to_string(noise) + "/";
            fs::create_directories(noiseDir);
            
            for (int t = 0; t < NUM_PATTERNS_PER_CLASS; t++) {
                Pattern noisy = addNoise(patterns[c], noise);
                string filename = noiseDir + "test_" + to_string(t + 1) + ".txt";
                string header = "Класс " + CLASS_NAMES[c] + ", шум " + to_string(noise) + "%, тест " + to_string(t + 1);
                savePattern(filename, noisy, header);
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
    cout << "    ЛАБОРАТОРНАЯ РАБОТА №3: СЕТЬ РБФ" << endl;
    cout << "    Вариант 2: классы N, F, I, P, D" << endl;
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
    cout << "\nИдеальные образы для обучения (6x6):" << endl;
    cout << "-----------------------------------" << endl;
    for (int i = 0; i < NUM_CLASSES; i++) {
        cout << "Класс " << (i + 1) << " (" << CLASS_NAMES[i] << "):" << endl;
        for (int row = 0; row < IMAGE_SIZE; row++) {
            cout << "  ";
            for (int col = 0; col < IMAGE_SIZE; col++) {
                int idx = row * IMAGE_SIZE + col;
                cout << (patterns[i][idx] > 0.5 ? "■" : "□") << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    
    // Инициализация РБФ ячеек
    cout << "2. Инициализация РБФ ячеек..." << endl;
    initRBFNeurons();
    cout << "   Количество РБФ ячеек: " << NUM_CLASSES << endl;
    for (int i = 0; i < NUM_CLASSES; i++) {
        cout << "   Класс " << (i + 1) << " (" << CLASS_NAMES[i] << "): sigma = " 
             << fixed << setprecision(3) << rbfNeurons[i].sigma << endl;
    }
    cout << endl;
    
    // Обучение выходного слоя
    cout << "3. Обучение выходного слоя градиентным спуском..." << endl;
    int trainingSteps = trainOutputLayer();
    cout << "   Обучение завершено за " << trainingSteps << " шагов" << endl << endl;
    
    // Генерация тестовых образов
    cout << "4. Генерация тестовых образов с разным уровнем шума..." << endl;
    generateTestPatterns();
    cout << endl;
    
    // Тестирование
    cout << "5. Тестирование распознавания:" << endl;
    cout << "=========================================" << endl << endl;
    
    int noiseLevels[] = {10, 20, 30, 40, 50};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    for (int c = 0; c < NUM_CLASSES; c++) {
        for (int n = 0; n < numNoiseLevels; n++) {
            for (int t = 0; t < NUM_PATTERNS_PER_CLASS; t++) {
                string filename = TESTS_DIR + CLASS_NAMES[c] + "/noise_" + 
                                  to_string(noiseLevels[n]) + "/test_" + to_string(t + 1) + ".txt";
                
                Pattern testPattern;
                if (!loadPattern(filename, testPattern)) {
                    continue;
                }
                
                // Распознавание
                vector<double> percentages = recognize(testPattern);
                
                // Вывод результатов
                cout << "┌─────────────────────────────────────────────────┐" << endl;
                cout << "│ Распознаваемый образ (6×6):                     │" << endl;
                cout << "│                                                 │" << endl;
                printPattern(testPattern);
                cout << "│                                                 │" << endl;
                cout << "├─────────────────────────────────────────────────┤" << endl;
                cout << "│ Процент подобия (выход РБФ):                    │" << endl;
                
                int bestClass = 0;
                double maxPercent = percentages[0];
                for (int i = 0; i < NUM_CLASSES; i++) {
                    if (percentages[i] > maxPercent) {
                        maxPercent = percentages[i];
                        bestClass = i;
                    }
                }
                
                for (int i = 0; i < NUM_CLASSES; i++) {
                    string marker = (i == bestClass) ? "  ◄── Распознан как \"" + CLASS_NAMES[i] + "\"" : "";
                    cout << "│   Класс " << (i + 1) << " (" << CLASS_NAMES[i] << "): " 
                         << setw(5) << fixed << setprecision(1) << percentages[i] << "%" << marker << endl;
                }
                
                cout << "│                                                 │" << endl;
                cout << "├─────────────────────────────────────────────────┤" << endl;
                cout << "│ Шагов обучения: " << setw(30) << trainingSteps << " │" << endl;
                cout << "└─────────────────────────────────────────────────┘" << endl << endl;
            }
        }
    }
    
    // Выводы
    cout << "6. Выводы:" << endl;
    cout << "==========" << endl;
    cout << "- Сеть РБФ успешно обучена на 5 классах образов (N, F, I, P, D)" << endl;
    cout << "- Обучение завершено за " << trainingSteps << " шагов" << endl;
    cout << "- Сеть способна распознавать зашумленные образы" << endl;
    cout << "- Процент подобия показывает степень соответствия каждому классу" << endl;
    
    return 0;
}
