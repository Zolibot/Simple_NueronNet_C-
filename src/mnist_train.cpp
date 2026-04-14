#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>

#include "NeuralNet.h"
#include "NeuralNetController.h"
#include "MnistReader.h"

// ============================================================================
// Конфигурация
// ============================================================================
static const std::string TRAIN_IMAGES = "../data/train-images-idx3-ubyte";
static const std::string TRAIN_LABELS = "../data/train-labels-idx1-ubyte";
static const std::string TEST_IMAGES = "../data/t10k-images-idx3-ubyte";
static const std::string TEST_LABELS = "../data/t10k-labels-idx1-ubyte";

static const int INPUT_SIZE = 784;
static const int HIDDEN_SIZE = 30;
static const int OUTPUT_SIZE = 10;

static const float LEARNING_RATE = 0.5f;
static const int MAX_EPOCHS = 5;
static const int TRAIN_LIMIT = 60000;
static const int TEST_LIMIT = 0;

static const std::string WEIGHTS_FILE = "../data/mnist_weights.w";

// ============================================================================
// Helpers
// ============================================================================
static void digitToTargets(int digit, float targets[], int outputSize)
{
    for (int i = 0; i < outputSize; i++)
        targets[i] = (i == digit) ? 0.9f : 0.1f;
}

static int predictDigit(const NeuralNetController &brain, int outputLayerIndex)
{
    const auto &layer = brain.getLayer(outputLayerIndex);
    int maxIdx = 0;
    float maxVal = layer[0].output;
    for (size_t i = 1; i < layer.size(); i++)
    {
        if (layer[i].output > maxVal)
        {
            maxVal = layer[i].output;
            maxIdx = static_cast<int>(i);
        }
    }
    return maxIdx;
}

static float evaluateAccuracy(NeuralNetController &brain, const MnistDataset &dataset,
                              int outputLayerIndex, bool verbose = false)
{
    int correct = 0;
    for (size_t i = 0; i < dataset.size(); i++)
    {
        brain.setData(dataset.images[i].pixels.data(),
                      static_cast<int>(dataset.images[i].pixels.size()), 0);
        brain.forwardPass();
        if (predictDigit(brain, outputLayerIndex) == dataset.labels[i])
            correct++;

        if (verbose && i % 1000 == 0)
        {
            std::cout << "  Evaluated: " << (i + 1) << "/" << dataset.size()
                      << " (accuracy: " << (correct * 100.0f / (i + 1)) << "%)" << std::endl;
        }
    }
    return (correct * 100.0f) / dataset.size();
}

// ============================================================================
// Main
// ============================================================================
int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "  MNIST Neural Network Training" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Architecture: " << INPUT_SIZE << " -> " << HIDDEN_SIZE << " -> " << OUTPUT_SIZE << std::endl;
    std::cout << "Learning rate: " << LEARNING_RATE << std::endl;
    std::cout << "Max epochs: " << MAX_EPOCHS << std::endl;
    std::cout << std::endl;

    // Load data
    std::cout << "[1/5] Loading training data..." << std::endl;
    MnistDataset trainData;
    if (!MnistReader::loadDataset(TRAIN_IMAGES, TRAIN_LABELS, trainData, TRAIN_LIMIT))
    {
        std::cerr << "ERROR: Failed to load training data" << std::endl;
        return 1;
    }
    MnistReader::printSummary(trainData);
    std::cout << std::endl;

    // Create network
    std::cout << "[2/5] Creating neural network..." << std::endl;
    NeuralNetController brain(LEARNING_RATE);
    brain.addLayer(INPUT_SIZE);
    brain.addLayer(HIDDEN_SIZE);
    brain.addLayer(OUTPUT_SIZE);
    brain.initialize();
    std::cout << "  Network created: " << INPUT_SIZE << " -> " << HIDDEN_SIZE << " -> " << OUTPUT_SIZE << std::endl;
    std::cout << std::endl;

    // Train
    std::cout << "[3/5] Training..." << std::endl;
    std::clock_t trainStart = std::clock();
    std::vector<float> targets(OUTPUT_SIZE);

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++)
    {
        float totalError = 0.0f;
        int correct = 0;

        for (size_t i = 0; i < trainData.size(); i++)
        {
            brain.setData(trainData.images[i].pixels.data(),
                          static_cast<int>(trainData.images[i].pixels.size()), 0);
            digitToTargets(trainData.labels[i], targets.data(), OUTPUT_SIZE);
            brain.train(targets.data());

            totalError += brain.getError();
            if (predictDigit(brain, 2) == trainData.labels[i])
                correct++;
        }

        float avgError = totalError / trainData.size();
        float epochAccuracy = (correct * 100.0f) / trainData.size();
        float elapsed = (std::clock() - trainStart) / static_cast<float>(CLOCKS_PER_SEC);

        std::cout << "  Epoch " << (epoch + 1) << "/" << MAX_EPOCHS
                  << " | Error: " << avgError
                  << " | Accuracy: " << epochAccuracy << "%"
                  << " | Time: " << elapsed << "s" << std::endl;
    }

    float totalTime = (std::clock() - trainStart) / static_cast<float>(CLOCKS_PER_SEC);
    std::cout << "  Total training time: " << totalTime << "s" << std::endl;
    std::cout << std::endl;

    // Save weights
    std::cout << "[4/5] Saving trained weights..." << std::endl;
    auto weightStrings = brain.saveState();
    std::ofstream wf(WEIGHTS_FILE);
    if (wf.is_open())
    {
        for (const auto &line : weightStrings)
            wf << line << std::endl;
        std::cout << "  Saved " << weightStrings.size() << " lines" << std::endl;
    }
    std::cout << std::endl;

    // Evaluate
    std::cout << "[5/5] Evaluating on test set..." << std::endl;
    MnistDataset testData;
    if (!MnistReader::loadDataset(TEST_IMAGES, TEST_LABELS, testData, TEST_LIMIT))
    {
        std::cerr << "ERROR: Failed to load test data" << std::endl;
        return 1;
    }
    MnistReader::printSummary(testData);
    std::cout << std::endl;

    std::clock_t evalStart = std::clock();
    float testAccuracy = evaluateAccuracy(brain, testData, 2, true);
    float evalTime = (std::clock() - evalStart) / static_cast<float>(CLOCKS_PER_SEC);

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Train samples: " << trainData.size() << std::endl;
    std::cout << "  Test samples:  " << testData.size() << std::endl;
    std::cout << "  Test accuracy: " << testAccuracy << "%" << std::endl;
    std::cout << "  Eval time:     " << evalTime << "s" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\nDone. Press Enter to exit..." << std::endl;
    int t = 0;
    std::cin >> t;
    return 0;
}
