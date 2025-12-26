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
// Constants
// ============================================================================
const int IMAGE_SIZE = 6;                       // Image size 6x6
const int INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE; // 36 inputs
const int NUM_CLASSES = 5;                      // Number of classes
const int NUM_PATTERNS_PER_CLASS = 3;           // Test patterns per class
const int MAX_TRAINING_ITERATIONS = 10000;      // Max training iterations
const double LEARNING_RATE = 0.1;               // Learning rate
const double EPSILON = 1e-6;                    // Precision for stopping training

// Directory paths
const string PATTERNS_DIR = "patterns/";
const string TESTS_DIR = "tests/";

// ============================================================================
// Data types
// ============================================================================
using Pattern = array<double, INPUT_SIZE>;      // Input vector (0 or 1)
using RBFNeuron = struct {
    Pattern center;                              // RBF cell center
    double sigma;                                // Spread parameter
};
using WeightMatrix = vector<vector<double>>;     // Output layer weights

// Class names for output
const string CLASS_NAMES[NUM_CLASSES] = {"N", "F", "I", "P", "D"};

// ============================================================================
// Global variables
// ============================================================================
Pattern patterns[NUM_CLASSES];                  // Ideal training patterns
vector<RBFNeuron> rbfNeurons;                   // RBF cells (one per class)
WeightMatrix outputWeights;                     // Output layer weights
mt19937 rng;                                    // Random number generator

// ============================================================================
// Load pattern from file
// ============================================================================
bool loadPattern(const string& filename, Pattern& pattern) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: cannot open file " << filename << endl;
        return false;
    }
    
    string line;
    int idx = 0;
    
    while (getline(file, line) && idx < INPUT_SIZE) {
        // Skip comments
        if (line.empty() || line[0] == '#') continue;
        
        istringstream iss(line);
        int value;
        while (iss >> value && idx < INPUT_SIZE) {
            // Convert 0/1 to 0.0/1.0 format
            pattern[idx++] = (value == 1) ? 1.0 : 0.0;
        }
    }
    
    file.close();
    return idx == INPUT_SIZE;
}

// ============================================================================
// Save pattern to file
// ============================================================================
void savePattern(const string& filename, const Pattern& pattern, const string& header = "") {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: cannot create file " << filename << endl;
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
// Initialize reference patterns from files
// ============================================================================
bool initPatterns() {
    for (int i = 0; i < NUM_CLASSES; i++) {
        string filename = PATTERNS_DIR + CLASS_NAMES[i] + ".txt";
        if (!loadPattern(filename, patterns[i])) {
            cerr << "Error loading pattern " << CLASS_NAMES[i] << endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// Compute Euclidean distance
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
// Initialize RBF neurons
// ============================================================================
void initRBFNeurons() {
    rbfNeurons.clear();
    rbfNeurons.resize(NUM_CLASSES);
    
    // RBF cell centers = ideal class patterns
    for (int i = 0; i < NUM_CLASSES; i++) {
        rbfNeurons[i].center = patterns[i];
    }
    
    // Compute spread parameters sigma
    // sigma_j = half distance to nearest center of another class
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
        
        // If all centers are the same (shouldn't happen), use default value
        if (minDist < EPSILON) {
            rbfNeurons[i].sigma = 1.0;
        } else {
            rbfNeurons[i].sigma = minDist / 2.0;
        }
    }
}

// ============================================================================
// Compute RBF cell output (Gaussian bell)
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
// Compute RBF network output
// ============================================================================
vector<double> computeRBFOutput(const Pattern& input) {
    vector<double> rbfOutputs(NUM_CLASSES);
    for (int i = 0; i < NUM_CLASSES; i++) {
        rbfOutputs[i] = rbfOutput(input, rbfNeurons[i]);
    }
    return rbfOutputs;
}

// ============================================================================
// Compute output layer output
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
// Train output layer using gradient descent
// ============================================================================
int trainOutputLayer() {
    // Initialize weights with random values
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
        
        // Pass through all training patterns
        for (int classIdx = 0; classIdx < NUM_CLASSES; classIdx++) {
            // Compute RBF layer outputs
            vector<double> rbfOuts = computeRBFOutput(patterns[classIdx]);
            
            // Compute output layer outputs
            vector<double> outputs = computeOutputLayer(rbfOuts);
            
            // Expected output: 1.0 for correct class, 0.0 for others
            vector<double> targets(NUM_CLASSES, 0.0);
            targets[classIdx] = 1.0;
            
            // Weight correction for each output neuron
            for (int k = 0; k < NUM_CLASSES; k++) {
                double error = targets[k] - outputs[k];
                totalError += error * error;
                
                // Gradient descent: w_jk := w_jk + alpha * d_k * g_j
                for (int j = 0; j < NUM_CLASSES; j++) {
                    outputWeights[j][k] += LEARNING_RATE * error * rbfOuts[j];
                }
            }
        }
        
        totalError /= NUM_CLASSES;
        
        // Check convergence
        if (abs(prevError - totalError) < EPSILON) {
            break;
        }
        prevError = totalError;
    }
    
    return iterations;
}

// ============================================================================
// Add noise to pattern
// ============================================================================
Pattern addNoise(const Pattern& original, int noisePercent) {
    Pattern noisy = original;
    int numPixelsToFlip = (INPUT_SIZE * noisePercent) / 100;
    
    vector<int> indices(INPUT_SIZE);
    for (int i = 0; i < INPUT_SIZE; i++) indices[i] = i;
    shuffle(indices.begin(), indices.end(), rng);
    
    for (int i = 0; i < numPixelsToFlip; i++) {
        // Flip pixel: 0 -> 1, 1 -> 0
        noisy[indices[i]] = (noisy[indices[i]] > 0.5) ? 0.0 : 1.0;
    }
    
    return noisy;
}

// ============================================================================
// Recognize pattern
// ============================================================================
vector<double> recognize(const Pattern& input) {
    vector<double> rbfOuts = computeRBFOutput(input);
    vector<double> outputs = computeOutputLayer(rbfOuts);
    
    // Convert outputs to positive values (softmax-like normalization)
    double sum = 0.0;
    vector<double> expOutputs(NUM_CLASSES);
    
    // Find max for numerical stability
    double maxOutput = *max_element(outputs.begin(), outputs.end());
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        expOutputs[i] = exp(outputs[i] - maxOutput);
        sum += expOutputs[i];
    }
    
    // Normalize to percentages
    vector<double> percentages(NUM_CLASSES);
    if (sum > EPSILON) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            percentages[i] = (expOutputs[i] / sum) * 100.0;
        }
    } else {
        // If sum is zero, uniform distribution
        for (int i = 0; i < NUM_CLASSES; i++) {
            percentages[i] = 100.0 / NUM_CLASSES;
        }
    }
    
    return percentages;
}

// ============================================================================
// Print pattern to console
// ============================================================================
void printPattern(const Pattern& p, const string& title = "") {
    if (!title.empty()) cout << title << ":" << endl;
    
    for (int i = 0; i < IMAGE_SIZE; i++) {
        cout << "|   ";
        for (int j = 0; j < IMAGE_SIZE; j++) {
            int idx = i * IMAGE_SIZE + j;
            cout << (p[idx] > 0.5 ? "#" : ".") << " ";
        }
        cout << "                                  |" << endl;
    }
}

// ============================================================================
// Generate test patterns
// ============================================================================
void generateTestPatterns() {
    cout << "Generating test patterns..." << endl;
    
    int noiseLevels[] = {10, 20, 30, 40, 50};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    for (int c = 0; c < NUM_CLASSES; c++) {
        // Create subfolder for each class
        string classDir = TESTS_DIR + CLASS_NAMES[c] + "/";
        fs::create_directories(classDir);
        
        for (int n = 0; n < numNoiseLevels; n++) {
            int noise = noiseLevels[n];
            string noiseDir = classDir + "noise_" + to_string(noise) + "/";
            fs::create_directories(noiseDir);
            
            for (int t = 0; t < NUM_PATTERNS_PER_CLASS; t++) {
                Pattern noisy = addNoise(patterns[c], noise);
                string filename = noiseDir + "test_" + to_string(t + 1) + ".txt";
                string header = "Class " + CLASS_NAMES[c] + ", noise " + to_string(noise) + "%, test " + to_string(t + 1);
                savePattern(filename, noisy, header);
            }
        }
    }
    
    cout << "Test patterns saved to " << TESTS_DIR << endl;
}

// ============================================================================
// Main function
// ============================================================================
int main() {
    // Initialize random number generator
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    rng.seed(seed);
    
    cout << "========================================================" << endl;
    cout << "    LAB 3: RBF NETWORK" << endl;
    cout << "    Variant 2: Classes N, F, I, P, D" << endl;
    cout << "========================================================" << endl << endl;
    
    // Create directories
    fs::create_directories(PATTERNS_DIR);
    fs::create_directories(TESTS_DIR);
    
    // Load reference patterns
    cout << "1. Loading reference patterns from " << PATTERNS_DIR << "..." << endl;
    if (!initPatterns()) {
        cerr << "Error loading patterns!" << endl;
        return 1;
    }
    
    // Display ideal training patterns
    cout << "\nIdeal training patterns (6x6):" << endl;
    cout << "------------------------------" << endl;
    for (int i = 0; i < NUM_CLASSES; i++) {
        cout << "Class " << (i + 1) << " (" << CLASS_NAMES[i] << "):" << endl;
        for (int row = 0; row < IMAGE_SIZE; row++) {
            cout << "  ";
            for (int col = 0; col < IMAGE_SIZE; col++) {
                int idx = row * IMAGE_SIZE + col;
                cout << (patterns[i][idx] > 0.5 ? "#" : ".") << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    
    // Initialize RBF cells
    cout << "2. Initializing RBF cells..." << endl;
    initRBFNeurons();
    cout << "   Number of RBF cells: " << NUM_CLASSES << endl;
    for (int i = 0; i < NUM_CLASSES; i++) {
        cout << "   Class " << (i + 1) << " (" << CLASS_NAMES[i] << "): sigma = " 
             << fixed << setprecision(3) << rbfNeurons[i].sigma << endl;
    }
    cout << endl;
    
    // Train output layer
    cout << "3. Training output layer using gradient descent..." << endl;
    int trainingSteps = trainOutputLayer();
    cout << "   Training completed in " << trainingSteps << " steps" << endl << endl;
    
    // Generate test patterns
    cout << "4. Generating test patterns with various noise levels..." << endl;
    generateTestPatterns();
    cout << endl;
    
    // Testing
    cout << "5. Recognition testing:" << endl;
    cout << "=======================" << endl << endl;
    
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
                
                // Recognition
                vector<double> percentages = recognize(testPattern);
                
                // Output results
                cout << "+--------------------------------------------------+" << endl;
                cout << "| Recognized pattern (6x6):                        |" << endl;
                cout << "|                                                  |" << endl;
                printPattern(testPattern);
                cout << "|                                                  |" << endl;
                cout << "+--------------------------------------------------+" << endl;
                cout << "| Similarity percentage (RBF output):              |" << endl;
                
                int bestClass = 0;
                double maxPercent = percentages[0];
                for (int i = 0; i < NUM_CLASSES; i++) {
                    if (percentages[i] > maxPercent) {
                        maxPercent = percentages[i];
                        bestClass = i;
                    }
                }
                
                for (int i = 0; i < NUM_CLASSES; i++) {
                    string marker = (i == bestClass) ? "  <-- Recognized as \"" + CLASS_NAMES[i] + "\"" : "";
                    cout << "|   Class " << (i + 1) << " (" << CLASS_NAMES[i] << "): " 
                         << setw(5) << fixed << setprecision(1) << percentages[i] << "%" << marker << endl;
                }
                
                cout << "|                                                  |" << endl;
                cout << "+--------------------------------------------------+" << endl;
                cout << "| Training steps: " << setw(30) << trainingSteps << " |" << endl;
                cout << "+--------------------------------------------------+" << endl << endl;
            }
        }
    }
    
    // Conclusions
    cout << "6. Conclusions:" << endl;
    cout << "===============" << endl;
    cout << "- RBF network successfully trained on 5 classes (N, F, I, P, D)" << endl;
    cout << "- Training completed in " << trainingSteps << " steps" << endl;
    cout << "- Network can recognize noisy patterns" << endl;
    cout << "- Similarity percentage shows correspondence to each class" << endl;
    
    return 0;
}
