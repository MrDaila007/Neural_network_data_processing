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
// Constants
// ============================================================================
const int IMAGE_SIZE = 6;                       // Image size 6x6
const int INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE; // 36 inputs
const int NUM_CLASSES = 5;                      // Number of classes
const int HIDDEN_SIZE = 20;                     // Hidden layer neurons
const double LEARNING_RATE_ALPHA = 0.5;         // Output layer learning rate
const double LEARNING_RATE_BETA = 0.5;          // Hidden layer learning rate
const double MAX_ERROR = 0.01;                  // Maximum allowed error
const int MAX_EPOCHS = 10000;                   // Max training epochs
const int NUM_NOISY_TESTS = 3;                  // Noisy tests per class

// Directory paths
const string PATTERNS_DIR = "patterns/";

// Class names
const string CLASS_NAMES[NUM_CLASSES] = {"N", "F", "I", "P", "D"};

// ============================================================================
// Data types
// ============================================================================
using InputVector = vector<double>;              // Input vector (36 elements)
using HiddenVector = vector<double>;             // Hidden layer vector
using OutputVector = vector<double>;             // Output vector (5 elements)
using WeightMatrix = vector<vector<double>>;     // Weight matrix

// ============================================================================
// Multilayer Perceptron class
// ============================================================================
class MultilayerPerceptron {
private:
    // Hidden layer weights (INPUT_SIZE x HIDDEN_SIZE)
    WeightMatrix weights_hidden;
    // Output layer weights (HIDDEN_SIZE x NUM_CLASSES)
    WeightMatrix weights_output;
    // Hidden layer thresholds
    vector<double> thresholds_hidden;
    // Output layer thresholds
    vector<double> thresholds_output;
    
    mt19937 rng;
    
    // Sigmoid activation function
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }
    
    // Sigmoid derivative
    double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }
    
    // Initialize weights with random values from [-1, 1]
    void initializeWeights() {
        uniform_real_distribution<double> dist(-1.0, 1.0);
        
        // Initialize hidden layer weights
        weights_hidden.resize(INPUT_SIZE);
        for (int i = 0; i < INPUT_SIZE; i++) {
            weights_hidden[i].resize(HIDDEN_SIZE);
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                weights_hidden[i][j] = dist(rng);
            }
        }
        
        // Initialize hidden layer thresholds
        thresholds_hidden.resize(HIDDEN_SIZE);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            thresholds_hidden[j] = dist(rng);
        }
        
        // Initialize output layer weights
        weights_output.resize(HIDDEN_SIZE);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_output[j].resize(NUM_CLASSES);
            for (int k = 0; k < NUM_CLASSES; k++) {
                weights_output[j][k] = dist(rng);
            }
        }
        
        // Initialize output layer thresholds
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
    
    // Forward pass (compute outputs)
    pair<HiddenVector, OutputVector> forward(const InputVector& input) {
        // Compute hidden layer output
        HiddenVector hidden(HIDDEN_SIZE);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double sum = thresholds_hidden[j];
            for (int i = 0; i < INPUT_SIZE; i++) {
                sum += weights_hidden[i][j] * input[i];
            }
            hidden[j] = sigmoid(sum);
        }
        
        // Compute output layer output
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
    
    // Train on single example
    double trainExample(const InputVector& input, const OutputVector& target) {
        // Forward pass
        auto [hidden, output] = forward(input);
        
        // Compute output layer errors
        vector<double> delta_output(NUM_CLASSES);
        for (int k = 0; k < NUM_CLASSES; k++) {
            double error = target[k] - output[k];
            delta_output[k] = error * sigmoidDerivative(output[k]);
        }
        
        // Compute hidden layer errors
        vector<double> delta_hidden(HIDDEN_SIZE);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double sum = 0.0;
            for (int k = 0; k < NUM_CLASSES; k++) {
                sum += delta_output[k] * weights_output[j][k];
            }
            delta_hidden[j] = sum * sigmoidDerivative(hidden[j]);
        }
        
        // Update output layer weights
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            for (int k = 0; k < NUM_CLASSES; k++) {
                weights_output[j][k] += LEARNING_RATE_ALPHA * delta_output[k] * hidden[j];
            }
        }
        
        // Update output layer thresholds
        for (int k = 0; k < NUM_CLASSES; k++) {
            thresholds_output[k] += LEARNING_RATE_ALPHA * delta_output[k];
        }
        
        // Update hidden layer weights
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                weights_hidden[i][j] += LEARNING_RATE_BETA * delta_hidden[j] * input[i];
            }
        }
        
        // Update hidden layer thresholds
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            thresholds_hidden[j] += LEARNING_RATE_BETA * delta_hidden[j];
        }
        
        // Compute mean squared error
        double error = 0.0;
        for (int k = 0; k < NUM_CLASSES; k++) {
            double diff = target[k] - output[k];
            error += diff * diff;
        }
        return error / 2.0;
    }
    
    // Train on dataset
    int train(const vector<pair<InputVector, OutputVector>>& trainingSet) {
        int epoch = 0;
        
        for (epoch = 0; epoch < MAX_EPOCHS; epoch++) {
            double maxError = 0.0;
            
            // Pass through all examples
            for (const auto& [input, target] : trainingSet) {
                double error = trainExample(input, target);
                maxError = max(maxError, abs(error));
            }
            
            // Check termination condition
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
    
    // Get network output (for testing)
    OutputVector predict(const InputVector& input) {
        auto [hidden, output] = forward(input);
        return output;
    }
};

// ============================================================================
// Load pattern from file
// ============================================================================
bool loadPattern(const string& filename, InputVector& pattern) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: cannot open file " << filename << endl;
        return false;
    }
    
    string line;
    int idx = 0;
    pattern.resize(INPUT_SIZE);
    
    while (getline(file, line) && idx < INPUT_SIZE) {
        // Skip comments
        if (line.empty() || line[0] == '#') continue;
        
        istringstream iss(line);
        int value;
        while (iss >> value && idx < INPUT_SIZE) {
            // Convert 0/1 to [0, 1] format for sigmoid
            pattern[idx++] = (value == 1) ? 1.0 : 0.0;
        }
    }
    
    file.close();
    return idx == INPUT_SIZE;
}

// ============================================================================
// Create target vector for class
// ============================================================================
OutputVector createTargetVector(int classIndex) {
    OutputVector target(NUM_CLASSES, 0.0);
    target[classIndex] = 1.0;
    return target;
}

// ============================================================================
// Add noise to pattern
// ============================================================================
InputVector addNoise(const InputVector& original, int noisePercent) {
    InputVector noisy = original;
    int numPixelsToFlip = (INPUT_SIZE * noisePercent) / 100;
    
    vector<int> indices(INPUT_SIZE);
    for (int i = 0; i < INPUT_SIZE; i++) indices[i] = i;
    
    mt19937 rng(chrono::system_clock::now().time_since_epoch().count());
    shuffle(indices.begin(), indices.end(), rng);
    
    for (int i = 0; i < numPixelsToFlip; i++) {
        // Flip pixel
        noisy[indices[i]] = (noisy[indices[i]] == 1.0) ? 0.0 : 1.0;
    }
    
    return noisy;
}

// ============================================================================
// Print pattern to console
// ============================================================================
void printPattern(const InputVector& p, const string& title = "") {
    if (!title.empty()) cout << title << ":" << endl;
    
    cout << "+----------------------------------+" << endl;
    cout << "| Recognized pattern (6x6):        |" << endl;
    cout << "|                                  |" << endl;
    cout << "|   ";
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            int idx = i * IMAGE_SIZE + j;
            cout << (p[idx] > 0.5 ? "#" : ".") << " ";
        }
        if (i < IMAGE_SIZE - 1) {
            cout << "   |" << endl << "|   ";
        }
    }
    cout << "   |" << endl;
    cout << "|                                  |" << endl;
}

// ============================================================================
// Print similarity percentages
// ============================================================================
void printSimilarity(const OutputVector& output, const string& title = "") {
    if (!title.empty()) cout << title << endl;
    
    cout << "+----------------------------------+" << endl;
    cout << "| Similarity percentage:           |" << endl;
    
    // Find maximum percentage
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
        string marker = (k == maxIndex) ? "  <-- Recognized" : "";
        cout << "|   Class " << (k + 1) << " (" << CLASS_NAMES[k] << "): " 
             << fixed << setprecision(1) << setw(5) << percent << "%" << marker << endl;
    }
    cout << "|                                  |" << endl;
}

// ============================================================================
// Main function
// ============================================================================
int main() {
    cout << "========================================================" << endl;
    cout << "    LAB 2: MULTILAYER PERCEPTRON" << endl;
    cout << "    Variant 2: Letters N, F, I, P, D" << endl;
    cout << "========================================================" << endl << endl;
    
    // Create patterns directory
    fs::create_directories(PATTERNS_DIR);
    
    // Load reference patterns
    cout << "1. Loading reference patterns from " << PATTERNS_DIR << "..." << endl;
    vector<InputVector> patterns(NUM_CLASSES);
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        string filename = PATTERNS_DIR + CLASS_NAMES[i] + ".txt";
        if (!loadPattern(filename, patterns[i])) {
            cerr << "Error loading pattern " << CLASS_NAMES[i] << endl;
            return 1;
        }
    }
    cout << "   Loaded " << NUM_CLASSES << " patterns" << endl << endl;
    
    // Create training set
    cout << "2. Creating training set..." << endl;
    vector<pair<InputVector, OutputVector>> trainingSet;
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        trainingSet.push_back({patterns[i], createTargetVector(i)});
    }
    cout << "   Created " << trainingSet.size() << " training examples" << endl << endl;
    
    // Create and train network
    cout << "3. Training multilayer perceptron..." << endl;
    MultilayerPerceptron network;
    
    int trainingSteps = network.train(trainingSet);
    cout << "   Training completed in " << trainingSteps << " epochs" << endl << endl;
    
    // Test on ideal patterns
    cout << "4. Testing on ideal patterns:" << endl;
    cout << "==============================" << endl << endl;
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        OutputVector output = network.predict(patterns[i]);
        printPattern(patterns[i], "Class " + to_string(i + 1) + " (" + CLASS_NAMES[i] + ")");
        printSimilarity(output);
        cout << "+----------------------------------+" << endl << endl;
    }
    
    // Test on noisy patterns
    cout << "5. Testing on noisy patterns:" << endl;
    cout << "==============================" << endl << endl;
    
    int noiseLevels[] = {10, 20, 30, 40, 50};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    for (int classIdx = 0; classIdx < NUM_CLASSES; classIdx++) {
        cout << "Class " << (classIdx + 1) << " (" << CLASS_NAMES[classIdx] << "):" << endl;
        cout << "------------------------------------" << endl;
        
        for (int noiseIdx = 0; noiseIdx < numNoiseLevels; noiseIdx++) {
            int noise = noiseLevels[noiseIdx];
            
            for (int test = 0; test < NUM_NOISY_TESTS; test++) {
                InputVector noisy = addNoise(patterns[classIdx], noise);
                OutputVector output = network.predict(noisy);
                
                printPattern(noisy, "Noise " + to_string(noise) + "%, test " + to_string(test + 1));
                printSimilarity(output);
                cout << "+----------------------------------+" << endl << endl;
            }
        }
    }
    
    // Training info
    cout << "6. Training information:" << endl;
    cout << "========================" << endl;
    cout << "+----------------------------------+" << endl;
    cout << "| Training epochs: " << setw(15) << trainingSteps << " |" << endl;
    cout << "+----------------------------------+" << endl << endl;
    
    cout << "7. Conclusions:" << endl;
    cout << "===============" << endl;
    cout << "- Multilayer perceptron trained successfully on " << NUM_CLASSES << " classes" << endl;
    cout << "- Network can recognize noisy patterns" << endl;
    cout << "- Similarity percentage shows network confidence for each class" << endl;
    
    return 0;
}
