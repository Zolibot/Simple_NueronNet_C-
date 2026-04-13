#pragma once

#include <string>
#include <vector>
#include <cstdint>

/// Represents a single MNIST image (28x28 grayscale, normalized to [0, 1])
struct MnistImage {
    std::vector<float> pixels;  // 784 values
};

/// Complete MNIST dataset: images + labels
struct MnistDataset {
    std::vector<MnistImage> images;
    std::vector<uint8_t> labels;

    /// Returns number of samples
    size_t size() const {
        return images.size();
    }

    /// Clear all data
    void clear() {
        images.clear();
        labels.clear();
    }
};

/// Reads MNIST data from IDX binary files
class MnistReader {
public:
    /// Read images from IDX3 file
    /// Returns number of images loaded
    static size_t readImages(const std::string &filePath, MnistDataset &dataset);

    /// Read labels from IDX1 file
    /// Returns number of labels loaded
    static size_t readLabels(const std::string &filePath, MnistDataset &dataset);

    /// Load a complete dataset (images + labels) from paired files
    /// imageFilePath — path to images IDX3 file
    /// labelFilePath — path to labels IDX1 file
    /// maxSamples — limit number of samples (0 = load all)
    static bool loadDataset(
        const std::string &imageFilePath,
        const std::string &labelFilePath,
        MnistDataset &dataset,
        size_t maxSamples = 0
    );

    /// Print dataset summary to stdout
    static void printSummary(const MnistDataset &dataset);
};
