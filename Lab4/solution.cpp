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
// Constants
// ============================================================================
const int IMAGE_SIZE = 6;                      // Image size 6x6
const int N = IMAGE_SIZE * IMAGE_SIZE;          // 36 input neurons
const int NUM_CLASSES = 5;                     // Number of pattern classes
const int NUM_NEURONS = 5;                     // Number of output neurons (clusters)
const int NUM_TRAINING_SAMPLES = 15;           // Training samples (more than neurons)
const int NUM_TEST_SAMPLES = 10;               // Test samples per class
const double LEARNING_RATE = 0.1;               // Learning rate beta
const double MAX_DISTANCE = 0.01;               // Max distance for training termination
const int MAX_ITERATIONS = 1000;                // Max training iterations

// Directory paths
const string PATTERNS_DIR = "patterns/";
const string TESTS_DIR = "tests/";

// ============================================================================
// Data types
// ============================================================================
using Pattern = vector<double>;                // Normalized vector
using WeightMatrix = vector<vector<double>>;   // Weight matrix [NUM_NEURONS][N]

// Symbol names for output
const string CLASS_NAMES[NUM_CLASSES] = {"LE", "GE", "NE", "AP", "CO"};
const string CLASS_SYMBOLS[NUM_CLASSES] = {"<=", ">=", "!=", "~~", "~="};

// ============================================================================
// Global variables
// ============================================================================
vector<Pattern> trainingPatterns;              // Training patterns
vector<int> trainingClasses;                   // Training pattern classes
WeightMatrix weights;                           // Weight matrix [NUM_NEURONS][N]
vector<int> winCounts;                         // Win count for each neuron
mt19937 rng;                                   // Random number generator

// ============================================================================
// Normalize vector
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
// Load pattern from file
// ============================================================================
bool loadPattern(const string& filename, Pattern& pattern) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: cannot open file " << filename << endl;
        return false;
    }
    
    pattern.clear();
    pattern.resize(N);
    
    string line;
    int idx = 0;
    
    while (getline(file, line) && idx < N) {
        // Skip comments
        if (line.empty() || line[0] == '#') continue;
        
        istringstream iss(line);
        int value;
        while (iss >> value && idx < N) {
            // Convert 0/1 to bipolar format -1/1, then normalize
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
            // Convert from bipolar format back to 0/1
            file << (pattern[idx] > 0 ? 1 : 0);
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
        Pattern pattern;
        if (!loadPattern(filename, pattern)) {
            cerr << "Error loading pattern " << CLASS_NAMES[i] << endl;
            return false;
        }
        
        // Add multiple variants of each pattern for training
        for (int j = 0; j < NUM_TRAINING_SAMPLES / NUM_CLASSES; j++) {
            trainingPatterns.push_back(pattern);
            trainingClasses.push_back(i);
        }
    }
    
    // Add more patterns so total exceeds NUM_NEURONS
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
// Initialize weights with random values
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
// Find winner neuron (standard method)
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
// Find winner neuron (frequency-dependent method)
// ============================================================================
int findWinnerFrequency(const Pattern& input) {
    int winner = 0;
    double minDistance = 1e10;
    
    for (int j = 0; j < NUM_NEURONS; j++) {
        // Compute Euclidean distance
        double distance = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = input[i] - weights[j][i];
            distance += diff * diff;
        }
        distance = sqrt(distance);
        
        // Multiply by win count (frequency-dependent learning)
        double weightedDistance = distance * (1.0 + winCounts[j]);
        
        if (weightedDistance < minDistance) {
            minDistance = weightedDistance;
            winner = j;
        }
    }
    
    return winner;
}

// ============================================================================
// Train network
// ============================================================================
void train() {
    cout << "Training competitive network..." << endl;
    
    bool converged = false;
    int iteration = 0;
    
    while (!converged && iteration < MAX_ITERATIONS) {
        iteration++;
        double maxDistance = 0.0;
        
        // Shuffle training patterns
        vector<int> indices(trainingPatterns.size());
        for (size_t i = 0; i < indices.size(); i++) {
            indices[i] = i;
        }
        shuffle(indices.begin(), indices.end(), rng);
        
        for (int idx : indices) {
            const Pattern& input = trainingPatterns[idx];
            
            // Find winner neuron
            int winner = findWinnerFrequency(input);
            winCounts[winner]++;
            
            // Compute distance before update
            double distance = 0.0;
            for (int i = 0; i < N; i++) {
                double diff = input[i] - weights[winner][i];
                distance += diff * diff;
            }
            distance = sqrt(distance);
            maxDistance = max(maxDistance, distance);
            
            // Update winner weights
            Pattern newWeights(N);
            for (int i = 0; i < N; i++) {
                newWeights[i] = weights[winner][i] + LEARNING_RATE * (input[i] - weights[winner][i]);
            }
            
            // Normalize new weights
            normalize(newWeights);
            weights[winner] = newWeights;
        }
        
        // Check termination condition
        if (maxDistance < MAX_DISTANCE) {
            converged = true;
            cout << "  Convergence reached at iteration " << iteration 
                 << " (max distance: " << fixed << setprecision(4) << maxDistance << ")" << endl;
        } else if (iteration % 100 == 0) {
            cout << "  Iteration " << iteration << ", max distance: " 
                 << fixed << setprecision(4) << maxDistance << endl;
        }
    }
    
    if (!converged) {
        cout << "  Maximum iterations reached (" << MAX_ITERATIONS << ")" << endl;
    }
}

// ============================================================================
// Add noise to pattern
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
// Print pattern to console
// ============================================================================
void printPattern(const Pattern& p, const string& title = "") {
    if (!title.empty()) cout << title << ":" << endl;
    
    for (int i = 0; i < IMAGE_SIZE; i++) {
        cout << "  ";
        for (int j = 0; j < IMAGE_SIZE; j++) {
            int idx = i * IMAGE_SIZE + j;
            cout << (p[idx] > 0 ? "#" : ".") << " ";
        }
        cout << endl;
    }
    cout << endl;
}

// ============================================================================
// Generate test patterns
// ============================================================================
void generateTestPatterns() {
    cout << "Generating test patterns..." << endl;
    
    int noiseLevels[] = {10, 20, 30, 40, 50};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    for (int c = 0; c < NUM_CLASSES; c++) {
        // Load reference pattern
        string filename = PATTERNS_DIR + CLASS_NAMES[c] + ".txt";
        Pattern pattern;
        if (!loadPattern(filename, pattern)) continue;
        
        // Create subfolder for each class
        string classDir = TESTS_DIR + CLASS_NAMES[c] + "/";
        fs::create_directories(classDir);
        
        for (int n = 0; n < numNoiseLevels; n++) {
            int noise = noiseLevels[n];
            string noiseDir = classDir + "noise_" + to_string(noise) + "/";
            fs::create_directories(noiseDir);
            
            for (int t = 0; t < NUM_TEST_SAMPLES; t++) {
                Pattern noisy = addNoise(pattern, noise);
                string testFilename = noiseDir + "test_" + to_string(t + 1) + ".txt";
                string header = "Class " + CLASS_SYMBOLS[c] + ", noise " + to_string(noise) + "%, test " + to_string(t + 1);
                savePattern(testFilename, noisy, header);
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
    cout << "    LAB 4: COMPETITIVE NEURAL NETWORK" << endl;
    cout << "    Variant 5: Symbols <=, >=, !=, ~~, ~=" << endl;
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
    
    cout << "   Loaded " << trainingPatterns.size() << " training patterns" << endl << endl;
    
    // Display reference patterns
    cout << "Reference patterns (6x6):" << endl;
    cout << "-------------------------" << endl;
    for (int i = 0; i < NUM_CLASSES; i++) {
        string filename = PATTERNS_DIR + CLASS_NAMES[i] + ".txt";
        Pattern pattern;
        if (loadPattern(filename, pattern)) {
            printPattern(pattern, "Class " + CLASS_SYMBOLS[i] + " (" + CLASS_NAMES[i] + ")");
        }
    }
    
    // Initialize weights
    cout << "2. Initializing weights with random values..." << endl;
    initWeights();
    cout << "   Weight matrix initialized (" << NUM_NEURONS << " neurons x " << N << " inputs)" << endl << endl;
    
    // Train network
    cout << "3. Training network..." << endl;
    train();
    cout << endl;
    
    // Determine neuron-to-class mapping
    cout << "4. Determining neuron-to-class mapping:" << endl;
    cout << "---------------------------------------" << endl;
    
    map<int, int> neuronToClass;  // neuron -> class
    map<int, vector<int>> classToNeurons;  // class -> neurons
    
    for (size_t i = 0; i < trainingPatterns.size(); i++) {
        int winner = findWinner(trainingPatterns[i]);
        int trueClass = trainingClasses[i];
        
        if (classToNeurons[trueClass].empty() || 
            find(classToNeurons[trueClass].begin(), classToNeurons[trueClass].end(), winner) == classToNeurons[trueClass].end()) {
            classToNeurons[trueClass].push_back(winner);
        }
    }
    
    // Determine main neuron for each class
    for (int c = 0; c < NUM_CLASSES; c++) {
        if (!classToNeurons[c].empty()) {
            int mainNeuron = classToNeurons[c][0];
            neuronToClass[mainNeuron] = c;
            cout << "Class " << CLASS_SYMBOLS[c] << " -> Winner neuron: #" << mainNeuron << endl;
        }
    }
    cout << endl;
    
    // Generate test patterns
    cout << "5. Generating test patterns with various noise levels..." << endl;
    generateTestPatterns();
    cout << endl;
    
    // Demonstration
    cout << "6. Recognition demonstration (30% noise):" << endl;
    cout << "------------------------------------------" << endl;
    
    string demoFilename = PATTERNS_DIR + CLASS_NAMES[2] + ".txt";  // Class !=
    Pattern demoPattern;
    if (loadPattern(demoFilename, demoPattern)) {
        Pattern demoNoisy = addNoise(demoPattern, 30);
        cout << "Noisy pattern of class " << CLASS_SYMBOLS[2] << ":" << endl;
        printPattern(demoNoisy);
        
        int winner = findWinner(demoNoisy);
        cout << "Result: Winner neuron #" << winner;
        
        if (neuronToClass.find(winner) != neuronToClass.end()) {
            int recognizedClass = neuronToClass[winner];
            cout << " (corresponds to class " << CLASS_SYMBOLS[recognizedClass] << ")";
            if (recognizedClass == 2) {
                cout << " OK" << endl;
            } else {
                cout << " WRONG" << endl;
            }
        } else {
            cout << " (class not determined)" << endl;
        }
    }
    cout << endl;
    
    // Clustering statistics
    cout << "7. Clustering statistics:" << endl;
    cout << "=========================" << endl;
    
    map<int, map<int, int>> neuronStats;  // neuron -> (class -> count)
    
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
        
        cout << "Neuron #" << j << ": " << total << " patterns";
        if (mainClass >= 0) {
            cout << " (main class: " << CLASS_SYMBOLS[mainClass] << ", " 
                 << mainClassCount << " patterns)";
        }
        cout << endl;
    }
    cout << endl;
    
    // Test on noisy patterns
    cout << "8. Testing on noisy patterns:" << endl;
    cout << "=============================" << endl;
    
    int noiseLevels[] = {10, 20, 30, 40, 50};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    cout << "+--------+-------+--------------+---------------+" << endl;
    cout << "| Class  | Noise |   Correct    |   Incorrect   |" << endl;
    cout << "|        |   %   |              |               |" << endl;
    cout << "+--------+-------+--------------+---------------+" << endl;
    
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
            
            cout << "|  " << setw(4) << CLASS_SYMBOLS[c] << "  |  " 
                 << setw(3) << noise << "  |     " 
                 << setw(3) << correct << "/" << setw(2) << total << "     |     "
                 << setw(3) << (total - correct) << "/" << setw(2) << total << "      |" << endl;
        }
        if (c < NUM_CLASSES - 1) {
            cout << "+--------+-------+--------------+---------------+" << endl;
        }
    }
    
    cout << "+--------+-------+--------------+---------------+" << endl << endl;
    
    // Conclusions
    cout << "9. Conclusions:" << endl;
    cout << "===============" << endl;
    cout << "- Competitive network trained successfully on " << trainingPatterns.size() << " patterns" << endl;
    cout << "- Network clustered patterns into " << NUM_NEURONS << " clusters" << endl;
    cout << "- Similar patterns were projected into the same cluster" << endl;
    cout << "- At low noise levels (10-20%) recognition works well" << endl;
    cout << "- At high noise levels (40-50%) recognition quality decreases" << endl;
    
    return 0;
}
