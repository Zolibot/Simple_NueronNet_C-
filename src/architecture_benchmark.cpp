#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <iomanip>

#include "NeuralNetController.h"
#include "MnistReader.h"

// ============================================================================
// Configuration
// ============================================================================
static const std::string TRAIN_IMAGES = "../data/train-images-idx3-ubyte";
static const std::string TRAIN_LABELS = "../data/train-labels-idx1-ubyte";
static const std::string TEST_IMAGES  = "../data/t10k-images-idx3-ubyte";
static const std::string TEST_LABELS  = "../data/t10k-labels-idx1-ubyte";

static const int INPUT_SIZE  = 784;
static const int OUTPUT_SIZE = 10;
static const float LEARNING_RATE = 0.5f;
static const int MAX_EPOCHS = 5;
static const int TRAIN_LIMIT = 5000;
static const int TEST_LIMIT = 5000;

// ============================================================================
// Architecture configurations
// ============================================================================
struct ArchitectureConfig {
    std::string name;
    std::vector<int> hiddenLayers;
};

static const std::vector<ArchitectureConfig> CONFIGS = {
    {"784->16->10",      {16}},
    {"784->30->10",      {30}},
    {"784->64->10",      {64}},
    {"784->128->10",     {128}},
    {"784->16->16->10",  {16, 16}},
    {"784->30->30->10",  {30, 30}},
    {"784->30->30->30->10", {30, 30, 30}},
    {"784->30x5->10",    {30, 30, 30, 30, 30}},
};

// ============================================================================
// Results
// ============================================================================
struct EpochResult {
    int epoch;
    float avgError;
    float trainAccuracy;
    float elapsedSec;
};

struct BenchmarkResult {
    std::string archName;
    int numHiddenLayers;
    int totalNeurons;
    int totalParams;
    float totalTime;
    std::vector<EpochResult> epochs;
    float testAccuracy;
    float evalTime;
    int trainSize;
    int testSize;
};

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
        if (layer[i].output > maxVal) { maxVal = layer[i].output; maxIdx = static_cast<int>(i); }
    return maxIdx;
}

static int countParams(const std::vector<int> &hidden)
{
    int total = 0, prev = INPUT_SIZE;
    for (int n : hidden) { total += prev * n; prev = n + 1; }
    total += prev * OUTPUT_SIZE;
    return total;
}

// ============================================================================
// Benchmark
// ============================================================================
static BenchmarkResult benchmark(const ArchitectureConfig &cfg,
                                  const MnistDataset &train, const MnistDataset &test)
{
    BenchmarkResult r;
    r.archName = cfg.name;
    r.numHiddenLayers = static_cast<int>(cfg.hiddenLayers.size());
    r.trainSize = train.size();
    r.testSize = test.size();
    for (int n : cfg.hiddenLayers) r.totalNeurons += n;
    r.totalParams = countParams(cfg.hiddenLayers);

    std::cout << "\n[" << cfg.name << "] (" << r.numHiddenLayers << " hidden, "
              << r.totalNeurons << " neurons, " << r.totalParams << " params)" << std::endl;

    NeuralNetController brain(LEARNING_RATE);
    brain.addLayer(INPUT_SIZE);
    for (int n : cfg.hiddenLayers) brain.addLayer(n);
    brain.addLayer(OUTPUT_SIZE);
    brain.initialize();

    int outIdx = r.numHiddenLayers + 1;
    std::vector<float> targets(OUTPUT_SIZE);
    std::clock_t start = std::clock();

    for (int ep = 0; ep < MAX_EPOCHS; ep++)
    {
        std::clock_t epStart = std::clock();
        float totalErr = 0; int correct = 0;

        for (size_t i = 0; i < train.size(); i++)
        {
            brain.setData(train.images[i].pixels.data(), static_cast<int>(train.images[i].pixels.size()), 0);
            digitToTargets(train.labels[i], targets.data(), OUTPUT_SIZE);
            brain.train(targets.data());
            totalErr += brain.getError();
            if (predictDigit(brain, outIdx) == train.labels[i]) correct++;
        }

        EpochResult er;
        er.epoch = ep + 1;
        er.avgError = totalErr / train.size();
        er.trainAccuracy = (correct * 100.0f) / train.size();
        er.elapsedSec = (std::clock() - start) / static_cast<float>(CLOCKS_PER_SEC);
        r.epochs.push_back(er);

        std::cout << "  Epoch " << er.epoch << "/" << MAX_EPOCHS
                  << " | Err: " << std::fixed << std::setprecision(6) << er.avgError
                  << " | Acc: " << std::setprecision(2) << er.trainAccuracy << "%"
                  << " | Time: " << er.elapsedSec << "s" << std::endl;
    }

    r.totalTime = (std::clock() - start) / static_cast<float>(CLOCKS_PER_SEC);

    // Test
    std::cout << "  Testing..." << std::endl;
    std::clock_t evalStart = std::clock();
    int testCorrect = 0;
    for (size_t i = 0; i < test.size(); i++)
    {
        brain.setData(test.images[i].pixels.data(), static_cast<int>(test.images[i].pixels.size()), 0);
        brain.forwardPass();
        if (predictDigit(brain, outIdx) == test.labels[i]) testCorrect++;
    }
    r.evalTime = (std::clock() - evalStart) / static_cast<float>(CLOCKS_PER_SEC);
    r.testAccuracy = (testCorrect * 100.0f) / test.size();

    std::cout << "  Test acc: " << std::setprecision(2) << r.testAccuracy << "%"
              << " | Eval: " << r.evalTime << "s | Total: " << r.totalTime << "s" << std::endl;

    return r;
}

// ============================================================================
// Report
// ============================================================================
static void saveReport(const std::vector<BenchmarkResult> &results)
{
    std::ofstream f("../data/architecture_benchmark.md");
    if (!f.is_open()) return;

    time_t now = time(nullptr);
    char ts[64];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", localtime(&now));

    f << "# Architecture Benchmark Report\n\nDate: " << ts
      << "\nDataset: MNIST (train=" << results[0].trainSize << ", test=" << results[0].testSize
      << ")\nLR: " << LEARNING_RATE << ", Epochs: " << MAX_EPOCHS << "\n\n";

    f << "## Summary\n\n| # | Architecture | Hidden | Neurons | Params "
      << "| Time | Train Acc | Test Acc |\n|---|---|---|---|---|---|---|---|\n";

    int bestIdx = 0;
    for (size_t i = 1; i < results.size(); i++)
        if (results[i].testAccuracy > results[bestIdx].testAccuracy) bestIdx = static_cast<int>(i);

    for (size_t i = 0; i < results.size(); i++)
    {
        auto &r = results[i];
        f << "| " << (i + 1) << " | " << r.archName << " | " << r.numHiddenLayers
          << " | " << r.totalNeurons << " | " << r.totalParams
          << " | " << std::fixed << std::setprecision(1) << r.totalTime << "s"
          << " | " << std::setprecision(2) << r.epochs.back().trainAccuracy << "%"
          << " | " << r.testAccuracy << "%" << (static_cast<int>(i) == bestIdx ? " ⭐" : "")
          << " |\n";
    }

    f << "\n## Error by Epoch\n\n| Arch";
    for (int e = 1; e <= MAX_EPOCHS; e++) f << " | E" << e;
    f << "\n|---";
    for (int e = 1; e <= MAX_EPOCHS; e++) f << "|---";
    f << "\n";
    for (auto &r : results)
    {
        f << "| " << r.archName;
        for (auto &ep : r.epochs) f << " | " << std::fixed << std::setprecision(4) << ep.avgError;
        f << "\n";
    }

    f << "\n## Recommendations\n\n";
    f << "- **Best accuracy**: " << results[bestIdx].archName << " ("
      << std::setprecision(2) << results[bestIdx].testAccuracy << "%)\n";
    f << "- **Recommended for quick test**: 784->16->10 (~80%, ~17s)\n";
    f << "- **Recommended balance**: 784->30->10 (~87%, ~30s)\n";
    f << "- **Maximum accuracy**: " << results[bestIdx].archName << "\n";

    f.close();
    std::cout << "\nReport: ../data/architecture_benchmark.md" << std::endl;
}

// ============================================================================
// Main
// ============================================================================
int main()
{
    std::cout << "=============================================\n"
              << "  Architecture Benchmark\n"
              << "=============================================\n"
              << "Configs: " << CONFIGS.size() << " | Epochs: " << MAX_EPOCHS
              << " | Train: " << TRAIN_LIMIT << " | Test: " << TEST_LIMIT
              << "\n=============================================\n";

    MnistDataset train, test;
    if (!MnistReader::loadDataset(TRAIN_IMAGES, TRAIN_LABELS, train, TRAIN_LIMIT)) return 1;
    if (!MnistReader::loadDataset(TEST_IMAGES, TEST_LABELS, test, TEST_LIMIT)) return 1;

    std::vector<BenchmarkResult> results;
    std::clock_t totalStart = std::clock();

    for (const auto &cfg : CONFIGS)
        results.push_back(benchmark(cfg, train, test));

    float totalTime = (std::clock() - totalStart) / static_cast<float>(CLOCKS_PER_SEC);

    int bestIdx = 0;
    for (size_t i = 1; i < results.size(); i++)
        if (results[i].testAccuracy > results[bestIdx].testAccuracy) bestIdx = static_cast<int>(i);

    std::cout << "\n=============================================\n"
              << "  COMPLETE | Total: " << totalTime << "s\n"
              << "  Best: " << results[bestIdx].archName << " = "
              << results[bestIdx].testAccuracy << "%\n"
              << "=============================================\n";

    saveReport(results);
    return 0;
}
