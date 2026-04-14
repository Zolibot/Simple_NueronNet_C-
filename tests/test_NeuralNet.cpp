#include "catch_amalgamated.hpp"
#include "NeuralNet.h"
#include <cmath>
#include <set>

// =============================================================================
// TESTS FOR: NeuralNet (NeuralNet.h / NeuralNet.cpp)
// =============================================================================

// -----------------------------------------------------------------------------
// N1: NeuralNet constructor
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNet constructor initializes correctly", "[NeuralNet]")
{
    NeuralNet net;

    SECTION("weightRange is 0.5")
    {
        REQUIRE(net.weightRange == 0.5f);
    }
}

// -----------------------------------------------------------------------------
// N2: random returns value in range
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNet::random returns value in range", "[NeuralNet]")
{
    NeuralNet net;

    SECTION("range [-1.0, 1.0]")
    {
        for (int i = 0; i < 100; i++)
        {
            float val = net.random(-1.0f, 1.0f);
            REQUIRE(val >= -1.0f);
            REQUIRE(val <= 1.0f);
        }
    }

    SECTION("range [-0.5, 0.5]")
    {
        for (int i = 0; i < 100; i++)
        {
            float val = net.random(-0.5f, 0.5f);
            REQUIRE(val >= -0.5f);
            REQUIRE(val <= 0.5f);
        }
    }
}

// -----------------------------------------------------------------------------
// N2b: random produces varying values
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNet::random produces varying values", "[NeuralNet]")
{
    NeuralNet net;
    std::set<float> values;
    for (int i = 0; i < 50; i++)
        values.insert(net.random(-10.0f, 10.0f));
    REQUIRE(values.size() > 40);
}

// -----------------------------------------------------------------------------
// N3: sigmoid correct values
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNet::sigmoid computes correctly", "[NeuralNet]")
{
    SECTION("sigmoid(0) = 0.5")
    {
        REQUIRE(NeuralNet::sigmoid(0.0f) == Catch::Approx(0.5f).epsilon(0.001));
    }

    SECTION("sigmoid(1) ≈ 0.731")
    {
        REQUIRE(NeuralNet::sigmoid(1.0f) == Catch::Approx(0.731058f).epsilon(0.001));
    }

    SECTION("sigmoid(-1) ≈ 0.269")
    {
        REQUIRE(NeuralNet::sigmoid(-1.0f) == Catch::Approx(0.268941f).epsilon(0.001));
    }
}

// -----------------------------------------------------------------------------
// N3b: sigmoid derivative
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNet::sigmoidDerivative computes correctly", "[NeuralNet]")
{
    SECTION("derivative at sigmoid(0) = 0.25")
    {
        float s = NeuralNet::sigmoid(0.0f);
        REQUIRE(NeuralNet::sigmoidDerivative(s) == Catch::Approx(0.25f).epsilon(0.001));
    }

    SECTION("derivative at sigmoid(1) ≈ 0.197")
    {
        float s = NeuralNet::sigmoid(1.0f);
        REQUIRE(NeuralNet::sigmoidDerivative(s) == Catch::Approx(0.1966f).epsilon(0.005));
    }
}

// -----------------------------------------------------------------------------
// N4: forwardPass
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNet::forwardPass computes output in (0, 1)", "[NeuralNet]")
{
    NeuralNet net;

    Layer inL = {{0.5f, 0.0f}, {0.3f, 0.0f}, {0.8f, 0.0f}};
    WeightMatrix weN = {{0.1f, -0.2f, 0.3f}, {-0.1f, 0.4f, -0.5f}};
    Layer ouL = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    net.forwardPass(inL, weN, ouL);

    SECTION("output count matches")
    {
        REQUIRE(ouL.size() == 2);
    }

    SECTION("all outputs in (0, 1)")
    {
        for (auto &n : ouL)
        {
            REQUIRE(n.output > 0.0f);
            REQUIRE(n.output < 1.0f);
        }
    }

    SECTION("error not modified")
    {
        for (auto &n : ouL)
            REQUIRE(n.error == 0.0f);
    }
}

// -----------------------------------------------------------------------------
// N4b: forwardPass with skipBiasIndex
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNet::forwardPass skips bias neuron", "[NeuralNet]")
{
    NeuralNet net;

    Layer inL = {{0.5f, 0.0f}, {1.0f, 0.0f}}; // second neuron is bias
    WeightMatrix weN = {{0.1f, 0.2f}, {0.3f, 0.4f}, {0.5f, 0.6f}};
    Layer ouL = {{0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}}; // third is bias

    net.forwardPass(inL, weN, ouL, 2); // skip index 2

    REQUIRE(ouL[2].output == 1.0f); // bias preserved
    REQUIRE(ouL[0].output > 0.0f);
    REQUIRE(ouL[0].output < 1.0f);
}

// -----------------------------------------------------------------------------
// N6: randomizeWeights
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNet::randomizeWeights fills correctly", "[NeuralNet]")
{
    NeuralNet net;

    SECTION("He init (default): values are small for large fanIn")
    {
        // He scale = sqrt(2/fanIn). For fanIn=3: sqrt(2/3) ≈ 0.816
        // Values should be within [-2*scale, +2*scale] ≈ [-1.63, +1.63]
        WeightMatrix w = {{0,0,0},{0,0,0},{0,0,0}}; // fanIn=3, fanOut=3
        net.randomizeWeights(w, INIT_HE);

        float scale = std::sqrt(2.0f / 3.0f);
        for (auto &row : w)
            for (auto &v : row)
            {
                REQUIRE(v >= -2.0f * scale);
                REQUIRE(v <= 2.0f * scale);
            }
    }

    SECTION("Xavier init: values scale with fanIn+fanOut")
    {
        WeightMatrix w(10, std::vector<float>(100, 0.0f)); // fanIn=100, fanOut=10
        net.randomizeWeights(w, INIT_XAVIER);
        float scale = std::sqrt(2.0f / 110.0f); // ≈ 0.135
        for (auto &row : w)
            for (auto &v : row)
            {
                REQUIRE(v >= -2.0f * scale);
                REQUIRE(v <= 2.0f * scale);
            }
    }

    SECTION("Uniform init: within weightRange")
    {
        WeightMatrix w = {{0,0,0},{0,0,0},{0,0,0}};
        net.randomizeWeights(w, INIT_UNIFORM);
        for (auto &row : w)
            for (auto &v : row)
            {
                REQUIRE(v >= -net.weightRange);
                REQUIRE(v <= net.weightRange);
            }
    }
}

// -----------------------------------------------------------------------------
// N8: backwardPass modifies weights
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNet::backwardPass modifies weights", "[NeuralNet]")
{
    NeuralNet net;
    Layer inL = {{0.5f, 0.0f}, {0.3f, 0.0f}};
    WeightMatrix weN = {{0.1f, 0.2f}, {-0.1f, 0.3f}};
    Layer ouL = {{0.7f, 0.5f}};

    auto old = weN;
    float lr = 0.1f;
    net.backwardPass(inL, weN, ouL, lr);

    bool changed = false;
    for (size_t i = 0; i < weN.size() && !changed; i++)
        for (size_t j = 0; j < weN[i].size() && !changed; j++)
            if (weN[i][j] != old[i][j]) changed = true;

    REQUIRE(changed);
}

// -----------------------------------------------------------------------------
// N7: computeError propagates errors
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNet::computeError propagates to hidden layer", "[NeuralNet]")
{
    NeuralNet net;
    Layer inL = {{0.5f, 0.0f}, {0.3f, 0.0f}};
    WeightMatrix weN = {{0.1f, 0.2f}, {-0.1f, 0.3f}};
    Layer ouL = {{0.7f, 0.5f}, {0.3f, -0.2f}};

    net.computeError(inL, weN, ouL, inL);

    bool hasError = false;
    for (auto &n : inL)
        if (n.error != 0.0f) hasError = true;
    REQUIRE(hasError);
}

// -----------------------------------------------------------------------------
// N9: setOutputError
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNet::setOutputError computes correctly", "[NeuralNet]")
{
    Layer ouL = {{0.7f, 0.0f}, {0.3f, 0.0f}, {0.5f, 0.0f}};
    float targets[] = {1.0f, 0.0f, 1.0f};

    NeuralNet::setOutputError(targets, 3, ouL);

    REQUIRE(ouL[0].error == Catch::Approx(0.3f).epsilon(0.001));
    REQUIRE(ouL[1].error == Catch::Approx(-0.3f).epsilon(0.001));
    REQUIRE(ouL[2].error == Catch::Approx(0.5f).epsilon(0.001));
}
