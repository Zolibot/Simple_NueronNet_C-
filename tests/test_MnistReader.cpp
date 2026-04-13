#include "catch_amalgamated.hpp"
#include "MnistReader.h"
#include <string>

// =============================================================================
// TESTS FOR: MnistReader (MnistReader.h / MnistReader.cpp)
// =============================================================================

// Paths relative to build/ directory (where tests executable runs from)
static const std::string DATA_PATH = "../data/";

// -----------------------------------------------------------------------------
// MR1: readImages — Loads correct number of images
// -----------------------------------------------------------------------------
TEST_CASE("MnistReader::readImages loads training images correctly", "[MnistReader]")
{
    MnistDataset dataset;
    std::string path = DATA_PATH + "train-images-idx3-ubyte";

    size_t count = MnistReader::readImages(path, dataset);

    SECTION("loads 60000 images")
    {
        REQUIRE(count == 60000);
    }

    SECTION("dataset has correct size")
    {
        REQUIRE(dataset.images.size() == 60000);
    }
}

// -----------------------------------------------------------------------------
// MR2: readImages — Pixel values are normalized to [0, 1]
// -----------------------------------------------------------------------------
TEST_CASE("MnistReader::readImages normalizes pixels to [0, 1]", "[MnistReader]")
{
    MnistDataset dataset;
    std::string path = DATA_PATH + "train-images-idx3-ubyte";

    MnistReader::readImages(path, dataset);

    // Check first 100 images
    for (size_t i = 0; i < 100 && i < dataset.images.size(); i++)
    {
        REQUIRE(dataset.images[i].pixels.size() == 784);

        for (size_t p = 0; p < 784; p++)
        {
            REQUIRE(dataset.images[i].pixels[p] >= 0.0f);
            REQUIRE(dataset.images[i].pixels[p] <= 1.0f);
        }
    }
}

// -----------------------------------------------------------------------------
// MR3: readLabels — Loads correct number of labels
// -----------------------------------------------------------------------------
TEST_CASE("MnistReader::readLabels loads training labels correctly", "[MnistReader]")
{
    MnistDataset dataset;
    std::string path = DATA_PATH + "train-labels-idx1-ubyte";

    size_t count = MnistReader::readLabels(path, dataset);

    SECTION("loads 60000 labels")
    {
        REQUIRE(count == 60000);
    }

    SECTION("dataset has correct label count")
    {
        REQUIRE(dataset.labels.size() == 60000);
    }
}

// -----------------------------------------------------------------------------
// MR4: readLabels — Label values are 0-9
// -----------------------------------------------------------------------------
TEST_CASE("MnistReader::readLabels produces values in range 0-9", "[MnistReader]")
{
    MnistDataset dataset;
    std::string path = DATA_PATH + "train-labels-idx1-ubyte";

    MnistReader::readLabels(path, dataset);

    for (size_t i = 0; i < dataset.labels.size(); i++)
    {
        REQUIRE(dataset.labels[i] <= 9);
    }
}

// -----------------------------------------------------------------------------
// MR5: loadDataset — Loads paired images + labels
// -----------------------------------------------------------------------------
TEST_CASE("MnistReader::loadDataset loads paired dataset", "[MnistReader]")
{
    MnistDataset dataset;

    bool success = MnistReader::loadDataset(
        DATA_PATH + "train-images-idx3-ubyte",
        DATA_PATH + "train-labels-idx1-ubyte",
        dataset
    );

    SECTION("returns true")
    {
        REQUIRE(success == true);
    }

    SECTION("images and labels counts match")
    {
        REQUIRE(dataset.images.size() == dataset.labels.size());
    }

    SECTION("has 60000 samples")
    {
        REQUIRE(dataset.size() == 60000);
    }
}

// -----------------------------------------------------------------------------
// MR6: loadDataset — Respects maxSamples limit
// -----------------------------------------------------------------------------
TEST_CASE("MnistReader::loadDataset respects maxSamples limit", "[MnistReader]")
{
    MnistDataset dataset;

    bool success = MnistReader::loadDataset(
        DATA_PATH + "train-images-idx3-ubyte",
        DATA_PATH + "train-labels-idx1-ubyte",
        dataset,
        1000  // Load only 1000 samples
    );

    SECTION("returns true")
    {
        REQUIRE(success == true);
    }

    SECTION("has exactly 1000 samples")
    {
        REQUIRE(dataset.size() == 1000);
    }

    SECTION("images and labels match after limit")
    {
        REQUIRE(dataset.images.size() == 1000);
        REQUIRE(dataset.labels.size() == 1000);
    }
}

// -----------------------------------------------------------------------------
// MR7: loadDataset — Loads test set
// -----------------------------------------------------------------------------
TEST_CASE("MnistReader::loadDataset loads test set correctly", "[MnistReader]")
{
    MnistDataset dataset;

    bool success = MnistReader::loadDataset(
        DATA_PATH + "t10k-images-idx3-ubyte",
        DATA_PATH + "t10k-labels-idx1-ubyte",
        dataset
    );

    SECTION("returns true")
    {
        REQUIRE(success == true);
    }

    SECTION("has 10000 samples")
    {
        REQUIRE(dataset.size() == 10000);
    }
}

// -----------------------------------------------------------------------------
// MR8: readImages — Handles non-existent file
// -----------------------------------------------------------------------------
TEST_CASE("MnistReader::readImages handles non-existent file", "[MnistReader]")
{
    MnistDataset dataset;

    size_t count = MnistReader::readImages("nonexistent_file.xyz", dataset);

    REQUIRE(count == 0);
}

// -----------------------------------------------------------------------------
// MR9: readLabels — Handles non-existent file
// -----------------------------------------------------------------------------
TEST_CASE("MnistReader::readLabels handles non-existent file", "[MnistReader]")
{
    MnistDataset dataset;

    size_t count = MnistReader::readLabels("nonexistent_file.xyz", dataset);

    REQUIRE(count == 0);
}

// -----------------------------------------------------------------------------
// MR10: Image pixel size is 784
// -----------------------------------------------------------------------------
TEST_CASE("Each MNIST image has exactly 784 pixels", "[MnistReader]")
{
    MnistDataset dataset;
    MnistReader::loadDataset(
        DATA_PATH + "t10k-images-idx3-ubyte",
        DATA_PATH + "t10k-labels-idx1-ubyte",
        dataset,
        100
    );

    for (size_t i = 0; i < dataset.images.size(); i++)
    {
        REQUIRE(dataset.images[i].pixels.size() == 784);
    }
}

// -----------------------------------------------------------------------------
// MR11: First few labels match known values
// Known: first training labels are [5, 0, 4, 1, 9, 2, 1, 3, 1, 4, ...]
// -----------------------------------------------------------------------------
TEST_CASE("First training labels match known MNIST values", "[MnistReader]")
{
    MnistDataset dataset;
    MnistReader::loadDataset(
        DATA_PATH + "train-images-idx3-ubyte",
        DATA_PATH + "train-labels-idx1-ubyte",
        dataset,
        20
    );

    // First 10 known MNIST training labels
    uint8_t knownLabels[] = {5, 0, 4, 1, 9, 2, 1, 3, 1, 4};

    for (int i = 0; i < 10; i++)
    {
        REQUIRE(dataset.labels[i] == knownLabels[i]);
    }
}
