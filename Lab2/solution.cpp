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
const int IMAGE_SIZE = 6;                       // Размер образа 6x6
const int INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE; // 36 входов
const int NUM_CLASSES = 5;                      // Количество классов
const int HIDDEN_SIZE = 20;                     // Количество нейронов скрытого слоя
const double LEARNING_RATE_ALPHA = 0.5;         // Скорость обучения выходного слоя
const double LEARNING_RATE_BETA = 0.5;          // Скорость обучения скрытого слоя
const double MAX_ERROR = 0.01;                  // Максимальная допустимая ошибка
const int MAX_EPOCHS = 10000;                    // Максимум эпох обучения
const int NUM_NOISY_TESTS = 3;                  // Количество зашумленных тестов на класс

// Пути к папкам
const string PATTERNS_DIR = "patterns/";

// Названия классов
const string CLASS_NAMES[NUM_CLASSES] = {"N", "F", "I", "P", "D"};

// ============================================================================
// Типы данных
// ============================================================================
using InputVector = vector<double>;              // Входной вектор (36 элементов)
using HiddenVector = vector<double>;             // Вектор скрытого слоя
using OutputVector = vector<double>;             // Выходной вектор (5 элементов)
using WeightMatrix = vector<vector<double>>;     // Матрица весов

// ============================================================================
// Класс многослойного персептрона
// ============================================================================
class MultilayerPerceptron {
private:
    // Веса скрытого слоя (INPUT_SIZE x HIDDEN_SIZE)
    WeightMatrix weights_hidden;
    // Веса выходного слоя (HIDDEN_SIZE x NUM_CLASSES)
    WeightMatrix weights_output;
    // Пороги скрытого слоя
    vector<double> thresholds_hidden;
    // Пороги выходного слоя
    vector<double> thresholds_output;
    
    mt19937 rng;
    
    // Сигмоидная функция активации
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }
    
    // Производная сигмоидной функции
    double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }
    
    // Инициализация весов случайными значениями из [-1, 1]
    void initializeWeights() {
        uniform_real_distribution<double> dist(-1.0, 1.0);
        
        // Инициализация весов скрытого слоя
        weights_hidden.resize(INPUT_SIZE);
        for (int i = 0; i < INPUT_SIZE; i++) {
            weights_hidden[i].resize(HIDDEN_SIZE);
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                weights_hidden[i][j] = dist(rng);
            }
        }
        
        // Инициализация порогов скрытого слоя
        thresholds_hidden.resize(HIDDEN_SIZE);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            thresholds_hidden[j] = dist(rng);
        }
        
        // Инициализация весов выходного слоя
        weights_output.resize(HIDDEN_SIZE);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_output[j].resize(NUM_CLASSES);
            for (int k = 0; k < NUM_CLASSES; k++) {
                weights_output[j][k] = dist(rng);
            }
        }
        
        // Инициализация порогов выходного слоя
        thresholds_output.resize(NUM_CLASSES);
        for (int k = 0; k < NUM_CLASSES; k++) {
            thresholds_output[k] = dist(rng);
        }
    }
    
public:
    MultilayerPerceptron() {
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        rng.seed(seed);
        initializeWeights();
    }
    
    // Прямой проход (вычисление выходов)
    pair<HiddenVector, OutputVector> forward(const InputVector& input) {
        // Вычисление выхода скрытого слоя
        HiddenVector hidden(HIDDEN_SIZE);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double sum = thresholds_hidden[j];
            for (int i = 0; i < INPUT_SIZE; i++) {
                sum += weights_hidden[i][j] * input[i];
            }
            hidden[j] = sigmoid(sum);
        }
        
        // Вычисление выхода выходного слоя
        OutputVector output(NUM_CLASSES);
        for (int k = 0; k < NUM_CLASSES; k++) {
            double sum = thresholds_output[k];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += weights_output[j][k] * hidden[j];
            }
            output[k] = sigmoid(sum);
        }
        
        return {hidden, output};
    }
    
    // Обучение на одном примере
    double trainExample(const InputVector& input, const OutputVector& target) {
        // Прямой проход
        auto [hidden, output] = forward(input);
        
        // Вычисление ошибок выходного слоя
        vector<double> delta_output(NUM_CLASSES);
        for (int k = 0; k < NUM_CLASSES; k++) {
            double error = target[k] - output[k];
            delta_output[k] = error * sigmoidDerivative(output[k]);
        }
        
        // Вычисление ошибок скрытого слоя
        vector<double> delta_hidden(HIDDEN_SIZE);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double sum = 0.0;
            for (int k = 0; k < NUM_CLASSES; k++) {
                sum += delta_output[k] * weights_output[j][k];
            }
            delta_hidden[j] = sum * sigmoidDerivative(hidden[j]);
        }
        
        // Коррекция весов выходного слоя
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            for (int k = 0; k < NUM_CLASSES; k++) {
                weights_output[j][k] += LEARNING_RATE_ALPHA * delta_output[k] * hidden[j];
            }
        }
        
        // Коррекция порогов выходного слоя
        for (int k = 0; k < NUM_CLASSES; k++) {
            thresholds_output[k] += LEARNING_RATE_ALPHA * delta_output[k];
        }
        
        // Коррекция весов скрытого слоя
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                weights_hidden[i][j] += LEARNING_RATE_BETA * delta_hidden[j] * input[i];
            }
        }
        
        // Коррекция порогов скрытого слоя
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            thresholds_hidden[j] += LEARNING_RATE_BETA * delta_hidden[j];
        }
        
        // Вычисление среднеквадратичной ошибки
        double error = 0.0;
        for (int k = 0; k < NUM_CLASSES; k++) {
            double diff = target[k] - output[k];
            error += diff * diff;
        }
        return error / 2.0;
    }
    
    // Обучение на наборе примеров
    int train(const vector<pair<InputVector, OutputVector>>& trainingSet) {
        int epoch = 0;
        
        for (epoch = 0; epoch < MAX_EPOCHS; epoch++) {
            double maxError = 0.0;
            
            // Проход по всем примерам
            for (const auto& [input, target] : trainingSet) {
                double error = trainExample(input, target);
                maxError = max(maxError, abs(error));
            }
            
            // Проверка условия завершения
            bool allCorrect = true;
            for (const auto& [input, target] : trainingSet) {
                auto [hidden, output] = forward(input);
                for (int k = 0; k < NUM_CLASSES; k++) {
                    if (abs(target[k] - output[k]) >= MAX_ERROR) {
                        allCorrect = false;
                        break;
                    }
                }
                if (!allCorrect) break;
            }
            
            if (allCorrect) {
                break;
            }
        }
        
        return epoch + 1;
    }
    
    // Получение выхода сети (для тестирования)
    OutputVector predict(const InputVector& input) {
        auto [hidden, output] = forward(input);
        return output;
    }
};

// ============================================================================
// Загрузка паттерна из файла
// ============================================================================
bool loadPattern(const string& filename, InputVector& pattern) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: не удалось открыть файл " << filename << endl;
        return false;
    }
    
    string line;
    int idx = 0;
    pattern.resize(INPUT_SIZE);
    
    while (getline(file, line) && idx < INPUT_SIZE) {
        // Пропускаем комментарии
        if (line.empty() || line[0] == '#') continue;
        
        istringstream iss(line);
        int value;
        while (iss >> value && idx < INPUT_SIZE) {
            // Преобразуем 0/1 в формат [0, 1] для сигмоиды
            pattern[idx++] = (value == 1) ? 1.0 : 0.0;
        }
    }
    
    file.close();
    return idx == INPUT_SIZE;
}

// ============================================================================
// Создание целевого вектора для класса
// ============================================================================
OutputVector createTargetVector(int classIndex) {
    OutputVector target(NUM_CLASSES, 0.0);
    target[classIndex] = 1.0;
    return target;
}

// ============================================================================
// Добавление шума к образу
// ============================================================================
InputVector addNoise(const InputVector& original, int noisePercent) {
    InputVector noisy = original;
    int numPixelsToFlip = (INPUT_SIZE * noisePercent) / 100;
    
    vector<int> indices(INPUT_SIZE);
    for (int i = 0; i < INPUT_SIZE; i++) indices[i] = i;
    
    mt19937 rng(chrono::system_clock::now().time_since_epoch().count());
    shuffle(indices.begin(), indices.end(), rng);
    
    for (int i = 0; i < numPixelsToFlip; i++) {
        // Инверсия пикселя
        noisy[indices[i]] = (noisy[indices[i]] == 1.0) ? 0.0 : 1.0;
    }
    
    return noisy;
}

// ============================================================================
// Вывод образа в консоль
// ============================================================================
void printPattern(const InputVector& p, const string& title = "") {
    if (!title.empty()) cout << title << ":" << endl;
    
    cout << "┌─────────────────────────────────┐" << endl;
    cout << "│ Распознаваемый образ (6×6):    │" << endl;
    cout << "│                                 │" << endl;
    cout << "│   ";
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            int idx = i * IMAGE_SIZE + j;
            cout << (p[idx] > 0.5 ? "■" : "□") << " ";
        }
        if (i < IMAGE_SIZE - 1) {
            cout << "  │" << endl << "│   ";
        }
    }
    cout << "  │" << endl;
    cout << "│                                 │" << endl;
}

// ============================================================================
// Вывод процента подобия
// ============================================================================
void printSimilarity(const OutputVector& output, const string& title = "") {
    if (!title.empty()) cout << title << endl;
    
    cout << "├─────────────────────────────────┤" << endl;
    cout << "│ Процент подобия:               │" << endl;
    
    // Находим максимальный процент
    int maxIndex = 0;
    double maxValue = output[0];
    for (int k = 1; k < NUM_CLASSES; k++) {
        if (output[k] > maxValue) {
            maxValue = output[k];
            maxIndex = k;
        }
    }
    
    for (int k = 0; k < NUM_CLASSES; k++) {
        double percent = output[k] * 100.0;
        string marker = (k == maxIndex) ? "  ◄── Распознан" : "";
        cout << "│   Класс " << (k + 1) << " (" << CLASS_NAMES[k] << "): " 
             << fixed << setprecision(1) << setw(5) << percent << "%" << marker << endl;
    }
    cout << "│                                 │" << endl;
}

// ============================================================================
// Главная функция
// ============================================================================
int main() {
    setlocale(LC_ALL, "Russian");
    
    cout << "========================================================" << endl;
    cout << "    ЛАБОРАТОРНАЯ РАБОТА №2: МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН" << endl;
    cout << "    Вариант 2: буквы N, F, I, P, D" << endl;
    cout << "========================================================" << endl << endl;
    
    // Создание директории для паттернов
    fs::create_directories(PATTERNS_DIR);
    
    // Загрузка эталонных образов
    cout << "1. Загрузка эталонных образов из " << PATTERNS_DIR << "..." << endl;
    vector<InputVector> patterns(NUM_CLASSES);
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        string filename = PATTERNS_DIR + CLASS_NAMES[i] + ".txt";
        if (!loadPattern(filename, patterns[i])) {
            cerr << "Ошибка загрузки паттерна " << CLASS_NAMES[i] << endl;
            return 1;
        }
    }
    cout << "   Загружено " << NUM_CLASSES << " паттернов" << endl << endl;
    
    // Создание обучающей выборки
    cout << "2. Создание обучающей выборки..." << endl;
    vector<pair<InputVector, OutputVector>> trainingSet;
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        trainingSet.push_back({patterns[i], createTargetVector(i)});
    }
    cout << "   Создано " << trainingSet.size() << " обучающих примеров" << endl << endl;
    
    // Создание и обучение сети
    cout << "3. Обучение многослойного персептрона..." << endl;
    MultilayerPerceptron network;
    
    int trainingSteps = network.train(trainingSet);
    cout << "   Обучение завершено за " << trainingSteps << " эпох" << endl << endl;
    
    // Тестирование на идеальных образах
    cout << "4. Тестирование на идеальных образах:" << endl;
    cout << "======================================" << endl << endl;
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        OutputVector output = network.predict(patterns[i]);
        printPattern(patterns[i], "Класс " + to_string(i + 1) + " (" + CLASS_NAMES[i] + ")");
        printSimilarity(output);
        cout << "└─────────────────────────────────┘" << endl << endl;
    }
    
    // Тестирование на зашумленных образах
    cout << "5. Тестирование на зашумленных образах:" << endl;
    cout << "=======================================" << endl << endl;
    
    int noiseLevels[] = {10, 20, 30, 40, 50};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    for (int classIdx = 0; classIdx < NUM_CLASSES; classIdx++) {
        cout << "Класс " << (classIdx + 1) << " (" << CLASS_NAMES[classIdx] << "):" << endl;
        cout << "─────────────────────────────────────" << endl;
        
        for (int noiseIdx = 0; noiseIdx < numNoiseLevels; noiseIdx++) {
            int noise = noiseLevels[noiseIdx];
            
            for (int test = 0; test < NUM_NOISY_TESTS; test++) {
                InputVector noisy = addNoise(patterns[classIdx], noise);
                OutputVector output = network.predict(noisy);
                
                printPattern(noisy, "Шум " + to_string(noise) + "%, тест " + to_string(test + 1));
                printSimilarity(output);
                cout << "└─────────────────────────────────┘" << endl << endl;
            }
        }
    }
    
    // Вывод информации о шагах обучения
    cout << "6. Информация об обучении:" << endl;
    cout << "==========================" << endl;
    cout << "┌─────────────────────────────────┐" << endl;
    cout << "│ Шагов обучения: " << setw(15) << trainingSteps << " │" << endl;
    cout << "└─────────────────────────────────┘" << endl << endl;
    
    cout << "7. Выводы:" << endl;
    cout << "==========" << endl;
    cout << "- Многослойный персептрон успешно обучен на " << NUM_CLASSES << " классах" << endl;
    cout << "- Сеть способна распознавать зашумленные образы" << endl;
    cout << "- Процент подобия показывает уверенность сети в каждом классе" << endl;
    
    return 0;
}
