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
const int IMAGE_SIZE = 10;                      // Image size 10x10
const int N = IMAGE_SIZE * IMAGE_SIZE;          // 100 neurons
const int NUM_PATTERNS = 3;                     // Number of reference patterns
const int MAX_ITERATIONS = 1000;                // Max recall iterations
const int TESTS_PER_NOISE_LEVEL = 10;           // Tests per noise level

// Directory paths
const string PATTERNS_DIR = "patterns/";
const string TESTS_DIR = "tests/";

// ============================================================================
// Data types
// ============================================================================
using Pattern = array<int, N>;                  // Bipolar vector {-1, 1}
using WeightMatrix = array<array<int, N>, N>;   // Weight matrix 100x100

// Pattern names for output
const string PATTERN_NAMES[NUM_PATTERNS] = {"D", "N", "X"};

// ============================================================================
// Global variables
// ============================================================================
Pattern patterns[NUM_PATTERNS];     // Reference patterns
WeightMatrix weights;               // Weight matrix
mt19937 rng;                        // Random number generator

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
    
    while (getline(file, line) && idx < N) {
        // Skip comments
        if (line.empty() || line[0] == '#') continue;
        
        istringstream iss(line);
        int value;
        while (iss >> value && idx < N) {
            // Convert 0/1 to bipolar format -1/1
            pattern[idx++] = (value == 1) ? 1 : -1;
        }
    }
    
    file.close();
    return idx == N;
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
            file << (pattern[idx] == 1 ? 1 : 0);
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
    for (int i = 0; i < NUM_PATTERNS; i++) {
        string filename = PATTERNS_DIR + PATTERN_NAMES[i] + ".txt";
        if (!loadPattern(filename, patterns[i])) {
            cerr << "Error loading pattern " << PATTERN_NAMES[i] << endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// Train network using Hebbian learning rule
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
// Synchronous recall
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
// Asynchronous recall
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
    
    return noisy;
}

// ============================================================================
// Compare patterns
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
// Print pattern to console
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
// Calculate similarity percentage
// ============================================================================
double calculateSimilarity(const Pattern& a, const Pattern& b) {
    int matches = 0;
    for (int i = 0; i < N; i++) {
        if (a[i] == b[i]) matches++;
    }
    return (100.0 * matches) / N;
}

// ============================================================================
// Generate test patterns and save to tests/ folder
// ============================================================================
void generateTestPatterns() {
    cout << "Generating test patterns..." << endl;
    
    int noiseLevels[] = {10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    for (int p = 0; p < NUM_PATTERNS; p++) {
        // Create subfolder for each letter
        string letterDir = TESTS_DIR + PATTERN_NAMES[p] + "/";
        fs::create_directories(letterDir);
        
        for (int n = 0; n < numNoiseLevels; n++) {
            int noise = noiseLevels[n];
            string noiseDir = letterDir + "noise_" + to_string(noise) + "/";
            fs::create_directories(noiseDir);
            
            for (int t = 0; t < TESTS_PER_NOISE_LEVEL; t++) {
                Pattern noisy = addNoise(patterns[p], noise);
                string filename = noiseDir + "test_" + to_string(t + 1) + ".txt";
                string header = "Letter " + PATTERN_NAMES[p] + ", noise " + to_string(noise) + "%, test " + to_string(t + 1);
                savePattern(filename, noisy, header);
            }
        }
    }
    
    cout << "Test patterns saved to " << TESTS_DIR << endl;
}

// ============================================================================
// Test result structure
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
// Run tests from files
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
        
        // Synchronous recall
        int iterSync;
        Pattern resultSync = recallSync(noisy, iterSync);
        result.avgIterSync += iterSync;
        result.avgSimilaritySync += calculateSimilarity(resultSync, patterns[patternIdx]);
        
        if (findClosestPattern(resultSync) == patternIdx) {
            result.correctSync++;
        }
        
        // Asynchronous recall
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
// Save results to file
// ============================================================================
void saveResults(const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) return;
    
    int noiseLevels[] = {10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    file << "Letter,Noise%,Sync_Success,Sync_Iter,Sync_Similarity%,Async_Success,Async_Iter,Async_Similarity%" << endl;
    
    for (int p = 0; p < NUM_PATTERNS; p++) {
        for (int n = 0; n < numNoiseLevels; n++) {
            TestResult res = runTestFromFiles(p, noiseLevels[n]);
            
            file << PATTERN_NAMES[p] << ","
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
// Main function
// ============================================================================
int main() {
    // Initialize random number generator
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    rng.seed(seed);
    
    cout << "========================================================" << endl;
    cout << "    LAB 1: HOPFIELD NEURAL NETWORK" << endl;
    cout << "    Variant 5: Letters D, N, X" << endl;
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
    
    // Display reference patterns
    cout << "\nReference patterns (10x10):" << endl;
    cout << "---------------------------" << endl;
    for (int i = 0; i < NUM_PATTERNS; i++) {
        printPattern(patterns[i], "Letter " + PATTERN_NAMES[i]);
    }
    
    // Train network
    cout << "2. Training network using Hebbian rule..." << endl;
    train();
    cout << "   Weight matrix computed (" << N << "x" << N << ")" << endl << endl;
    
    // Generate test patterns
    cout << "3. Generating test patterns with various noise levels..." << endl;
    generateTestPatterns();
    cout << endl;
    
    // Demonstration
    cout << "4. Recognition demonstration (30% noise):" << endl;
    cout << "------------------------------------------" << endl;
    
    Pattern demo = addNoise(patterns[0], 30);
    cout << "Noisy pattern of letter D:" << endl;
    printPattern(demo);
    
    int iterDemo;
    Pattern recovered = recallSync(demo, iterDemo);
    cout << "Recovered pattern (synchronous mode, " << iterDemo << " iterations):" << endl;
    printPattern(recovered);
    
    int recognized = findClosestPattern(recovered);
    cout << "Recognized as: " << PATTERN_NAMES[recognized] << endl;
    cout << "Similarity to reference: " << fixed << setprecision(1) 
         << calculateSimilarity(recovered, patterns[0]) << "%" << endl << endl;
    
    // Mass testing from files
    cout << "5. File-based recognition testing:" << endl;
    cout << "===================================" << endl << endl;
    
    int noiseLevels[] = {10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100};
    int numNoiseLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    
    // Table header
    cout << "+--------+-------+-------------------------+-------------------------+" << endl;
    cout << "| Letter | Noise |    Synchronous Mode     |   Asynchronous Mode     |" << endl;
    cout << "|        |   %   |  Success |  Avg.iter.  |  Success |  Avg.iter.   |" << endl;
    cout << "+--------+-------+----------+-------------+----------+--------------+" << endl;
    
    int totalCorrectSync[12] = {0};
    int totalCorrectAsync[12] = {0};
    int totalTests[12] = {0};
    
    for (int p = 0; p < NUM_PATTERNS; p++) {
        for (int n = 0; n < numNoiseLevels; n++) {
            TestResult res = runTestFromFiles(p, noiseLevels[n]);
            
            totalCorrectSync[n] += res.correctSync;
            totalCorrectAsync[n] += res.correctAsync;
            totalTests[n] += res.total;
            
            cout << "|   " << PATTERN_NAMES[p] << "    |  " 
                 << setw(3) << noiseLevels[n] << "  |  "
                 << setw(2) << res.correctSync << "/" << setw(2) << res.total << "   |   "
                 << setw(6) << fixed << setprecision(1) << res.avgIterSync << "    |  "
                 << setw(2) << res.correctAsync << "/" << setw(2) << res.total << "   |   "
                 << setw(7) << fixed << setprecision(1) << res.avgIterAsync << "    |" << endl;
        }
        if (p < NUM_PATTERNS - 1) {
            cout << "+--------+-------+----------+-------------+----------+--------------+" << endl;
        }
    }
    
    cout << "+--------+-------+----------+-------------+----------+--------------+" << endl;
    
    // Summary statistics
    cout << endl << "6. Summary statistics by noise level:" << endl;
    cout << "======================================" << endl << endl;
    
    cout << "+-------+-----------------+-----------------+" << endl;
    cout << "| Noise |   Synchronous   |  Asynchronous   |" << endl;
    cout << "|   %   |   % success     |   % success     |" << endl;
    cout << "+-------+-----------------+-----------------+" << endl;
    
    for (int n = 0; n < numNoiseLevels; n++) {
        double successSync = totalTests[n] > 0 ? (100.0 * totalCorrectSync[n]) / totalTests[n] : 0;
        double successAsync = totalTests[n] > 0 ? (100.0 * totalCorrectAsync[n]) / totalTests[n] : 0;
        
        cout << "|  " << setw(3) << noiseLevels[n] << "  |      "
             << setw(5) << fixed << setprecision(1) << successSync << "%      |      "
             << setw(5) << fixed << setprecision(1) << successAsync << "%      |" << endl;
    }
    
    cout << "+-------+-----------------+-----------------+" << endl;
    
    // Save results
    cout << endl << "7. Saving results..." << endl;
    saveResults(TESTS_DIR + "results.csv");
    cout << "   Results saved to " << TESTS_DIR << "results.csv" << endl;
    
    // Conclusions
    cout << endl << "8. Conclusions:" << endl;
    cout << "===============" << endl;
    cout << "- Hopfield network successfully trained on 3 patterns (D, N, X)" << endl;
    cout << "- At low noise levels (up to 30%) recognition works well" << endl;
    cout << "- At noise above 40-50% recognition quality drops significantly" << endl;
    cout << "- Asynchronous mode usually requires more iterations" << endl;
    
    return 0;
}
