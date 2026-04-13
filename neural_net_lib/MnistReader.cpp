#include "MnistReader.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>

// Convert 32-bit big-endian integer to host byte order
static uint32_t readBigEndian32(const unsigned char *buf)
{
    return (static_cast<uint32_t>(buf[0]) << 24) |
           (static_cast<uint32_t>(buf[1]) << 16) |
           (static_cast<uint32_t>(buf[2]) <<  8) |
           (static_cast<uint32_t>(buf[3]) <<  0);
}

size_t MnistReader::readImages(const std::string &filePath, MnistDataset &dataset)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "ERROR: Cannot open image file: " << filePath << std::endl;
        return 0;
    }

    // Read header (16 bytes)
    unsigned char header[16];
    file.read(reinterpret_cast<char *>(header), 16);
    if (!file.good())
    {
        std::cerr << "ERROR: Failed to read image file header" << std::endl;
        return 0;
    }

    uint32_t magic = readBigEndian32(header);
    if (magic != 2051)
    {
        std::cerr << "ERROR: Invalid image magic number: " << magic
                  << " (expected 2051)" << std::endl;
        return 0;
    }

    uint32_t numImages = readBigEndian32(header + 4);
    uint32_t numRows   = readBigEndian32(header + 8);
    uint32_t numCols   = readBigEndian32(header + 12);

    size_t pixelsPerImage = numRows * numCols; // should be 784

    std::cout << "  Images file: " << filePath << std::endl;
    std::cout << "  Magic: " << magic
              << ", Count: " << numImages
              << ", Size: " << numRows << "x" << numCols << std::endl;

    // Read pixel data
    std::vector<unsigned char> rawPixels(pixelsPerImage);

    for (uint32_t i = 0; i < numImages; i++)
    {
        file.read(reinterpret_cast<char *>(rawPixels.data()), pixelsPerImage);
        if (!file.good())
        {
            std::cerr << "WARNING: Failed to read image #" << i << std::endl;
            break;
        }

        MnistImage img;
        img.pixels.resize(pixelsPerImage);

        // Normalize: [0, 255] -> [0.0, 1.0]
        for (size_t p = 0; p < pixelsPerImage; p++)
        {
            img.pixels[p] = rawPixels[p] / 255.0f;
        }

        dataset.images.push_back(std::move(img));
    }

    return dataset.images.size();
}

size_t MnistReader::readLabels(const std::string &filePath, MnistDataset &dataset)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "ERROR: Cannot open label file: " << filePath << std::endl;
        return 0;
    }

    // Read header (8 bytes)
    unsigned char header[8];
    file.read(reinterpret_cast<char *>(header), 8);
    if (!file.good())
    {
        std::cerr << "ERROR: Failed to read label file header" << std::endl;
        return 0;
    }

    uint32_t magic = readBigEndian32(header);
    if (magic != 2049)
    {
        std::cerr << "ERROR: Invalid label magic number: " << magic
                  << " (expected 2049)" << std::endl;
        return 0;
    }

    uint32_t numLabels = readBigEndian32(header + 4);

    std::cout << "  Labels file: " << filePath << std::endl;
    std::cout << "  Magic: " << magic
              << ", Count: " << numLabels << std::endl;

    // Read label data
    for (uint32_t i = 0; i < numLabels; i++)
    {
        unsigned char label;
        file.read(reinterpret_cast<char *>(&label), 1);
        if (!file.good())
        {
            std::cerr << "WARNING: Failed to read label #" << i << std::endl;
            break;
        }
        dataset.labels.push_back(label);
    }

    return dataset.labels.size();
}

bool MnistReader::loadDataset(
    const std::string &imageFilePath,
    const std::string &labelFilePath,
    MnistDataset &dataset,
    size_t maxSamples
)
{
    dataset.clear();

    // Load images
    size_t numImages = readImages(imageFilePath, dataset);
    if (numImages == 0)
    {
        std::cerr << "ERROR: No images loaded" << std::endl;
        return false;
    }

    // Load labels
    size_t numLabels = readLabels(labelFilePath, dataset);
    if (numLabels == 0)
    {
        std::cerr << "ERROR: No labels loaded" << std::endl;
        return false;
    }

    // Verify counts match
    if (numImages != numLabels)
    {
        std::cerr << "WARNING: Image count (" << numImages
                  << ") != Label count (" << numLabels << ")" << std::endl;
    }

    // Limit samples if requested
    if (maxSamples > 0 && maxSamples < dataset.size())
    {
        dataset.images.resize(maxSamples);
        dataset.labels.resize(maxSamples);
        std::cout << "  Limited to " << maxSamples << " samples" << std::endl;
    }

    // Final verification: resize to match
    size_t count = std::min(dataset.images.size(), dataset.labels.size());
    dataset.images.resize(count);
    dataset.labels.resize(count);

    std::cout << "  Dataset loaded: " << dataset.size() << " samples" << std::endl;
    return true;
}

void MnistReader::printSummary(const MnistDataset &dataset)
{
    std::cout << "=== Dataset Summary ===" << std::endl;
    std::cout << "  Total samples: " << dataset.size() << std::endl;

    if (dataset.size() > 0)
    {
        std::cout << "  Image size: " << dataset.images[0].pixels.size() << " pixels" << std::endl;

        // Count label distribution
        int labelCount[10] = {0};
        for (size_t i = 0; i < dataset.labels.size(); i++)
        {
            if (dataset.labels[i] < 10)
            {
                labelCount[dataset.labels[i]]++;
            }
        }

        std::cout << "  Label distribution:" << std::endl;
        for (int d = 0; d < 10; d++)
        {
            std::cout << "    Digit " << d << ": " << labelCount[d] << std::endl;
        }
    }
    std::cout << "=======================" << std::endl;
}
