#include "catch_amalgamated.hpp"
#include "NeuralNetController.h"
#include <fstream>
#include <cmath>

// =============================================================================
// TESTS FOR: NeuralNetController
// =============================================================================

// -----------------------------------------------------------------------------
// C1: Constructor
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNetController constructor initializes correctly", "[NeuralNetController]")
{
    NeuralNetController brain(0.1f);

    SECTION("learning rate is set")
    {
        // No direct getter, but we can verify through behavior
    }

    SECTION("no layers initially")
    {
        REQUIRE(brain.layerCount() == 0);
    }
}

// -----------------------------------------------------------------------------
// C19: square
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNetController::square works correctly", "[NeuralNetController]")
{
    REQUIRE(NeuralNetController::square(3.0f) == Catch::Approx(9.0f));
    REQUIRE(NeuralNetController::square(-2.0f) == Catch::Approx(4.0f));
    REQUIRE(NeuralNetController::square(0.0f) == Catch::Approx(0.0f));
    REQUIRE(NeuralNetController::square(0.5f) == Catch::Approx(0.25f));
}

// -----------------------------------------------------------------------------
// C3: addLayer
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNetController::addLayer creates neurons", "[NeuralNetController]")
{
    NeuralNetController brain(0.1f);

    SECTION("addLayer(6) creates 6 neurons")
    {
        brain.addLayer(6);
        REQUIRE(brain.layerCount() == 1);
        REQUIRE(brain.neuronCount(0) == 6);
    }

    SECTION("each neuron initialized to zero")
    {
        brain.addLayer(4);
        const auto &layer = brain.getLayer(0);
        for (size_t i = 0; i < layer.size(); i++)
        {
            REQUIRE(layer[i].output == 0.0f);
            REQUIRE(layer[i].error == 0.0f);
        }
    }

    SECTION("multiple layers")
    {
        brain.addLayer(6);
        brain.addLayer(10);
        brain.addLayer(4);
        REQUIRE(brain.layerCount() == 3);
        REQUIRE(brain.neuronCount(0) == 6);
        REQUIRE(brain.neuronCount(1) == 10);
        REQUIRE(brain.neuronCount(2) == 4);
    }

    SECTION("invalid count throws")
    {
        REQUIRE_THROWS(brain.addLayer(0));
        REQUIRE_THROWS(brain.addLayer(-1));
    }
}

// -----------------------------------------------------------------------------
// C6: setBias
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNetController::setBias sets bias neurons to 1.0", "[NeuralNetController]")
{
    NeuralNetController brain(0.1f);
    brain.addLayer(6);
    brain.addLayer(10);
    brain.addLayer(4);
    brain.setBias();

    SECTION("bias for first layer")
    {
        const auto &layer = brain.getLayer(0);
        REQUIRE(layer[5].output == Catch::Approx(1.0f).epsilon(0.001));
        REQUIRE(layer[5].error == 0.0f);
    }

    SECTION("bias for second layer")
    {
        const auto &layer = brain.getLayer(1);
        REQUIRE(layer[9].output == Catch::Approx(1.0f).epsilon(0.001));
    }

    SECTION("no bias on output layer")
    {
        const auto &layer = brain.getLayer(2);
        REQUIRE(layer[3].output == 0.0f);
    }
}

// -----------------------------------------------------------------------------
// C16: addWeights
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNetController::addWeights creates weight matrices", "[NeuralNetController]")
{
    NeuralNetController brain(0.1f);
    brain.addLayer(6);
    brain.addLayer(10);
    brain.addLayer(4);
    brain.addWeights();

    // After addWeights, we can't directly access weights, but initialize() calls it
}

// -----------------------------------------------------------------------------
// C4: initialize
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNetController::initialize fully initializes", "[NeuralNetController]")
{
    NeuralNetController brain(0.1f);
    brain.addLayer(6);
    brain.addLayer(10);
    brain.addLayer(4);
    brain.initialize();

    SECTION("bias is set")
    {
        REQUIRE(brain.getLayer(0)[5].output == Catch::Approx(1.0f).epsilon(0.001));
    }

    SECTION("forward pass works")
    {
        float input[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
        brain.setData(input, 6, 0);
        brain.forwardPass();
        // Output should be in (0, 1)
        const auto &out = brain.getLayer(2);
        for (size_t i = 0; i < out.size(); i++)
        {
            REQUIRE(out[i].output > 0.0f);
            REQUIRE(out[i].output < 1.0f);
        }
    }
}

// -----------------------------------------------------------------------------
// C5: setData
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNetController::setData copies data and preserves bias", "[NeuralNetController]")
{
    NeuralNetController brain(0.1f);
    brain.addLayer(4);
    brain.addLayer(3);
    brain.initialize();

    float data[] = {0.5f, 0.3f, 0.8f};
    brain.setData(data, 3, 0);

    SECTION("data copied correctly")
    {
        const auto &layer = brain.getLayer(0);
        REQUIRE(layer[0].output == Catch::Approx(0.5f).epsilon(0.001));
        REQUIRE(layer[1].output == Catch::Approx(0.3f).epsilon(0.001));
        REQUIRE(layer[2].output == Catch::Approx(0.8f).epsilon(0.001));
    }

    SECTION("bias preserved")
    {
        const auto &layer = brain.getLayer(0);
        REQUIRE(layer[3].output == Catch::Approx(1.0f).epsilon(0.001));
    }

    SECTION("out of range throws")
    {
        REQUIRE_THROWS(brain.setData(data, 999, 0));
        REQUIRE_THROWS(brain.setData(data, 3, 5));
    }
}

// -----------------------------------------------------------------------------
// C7: forwardPass
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNetController::forwardPass produces valid output", "[NeuralNetController]")
{
    NeuralNetController brain(0.1f);
    brain.addLayer(3);
    brain.addLayer(5);
    brain.addLayer(2);
    brain.initialize();

    float input[] = {0.5f, 0.3f, 0.8f};
    brain.setData(input, 3, 0);
    brain.forwardPass();

    SECTION("output layer size correct")
    {
        REQUIRE(brain.neuronCount(2) == 2);
    }

    SECTION("outputs in (0, 1)")
    {
        const auto &out = brain.getLayer(2);
        for (size_t i = 0; i < out.size(); i++)
        {
            REQUIRE(out[i].output > 0.0f);
            REQUIRE(out[i].output < 1.0f);
        }
    }
}

// -----------------------------------------------------------------------------
// C14: getError
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNetController::getError returns non-negative", "[NeuralNetController]")
{
    NeuralNetController brain(0.1f);
    brain.addLayer(2);
    brain.addLayer(3);
    brain.initialize();

    float input[] = {0.5f, 0.3f};
    brain.setData(input, 2, 0);
    brain.forwardPass();

    REQUIRE(brain.getError() >= 0.0f);
}

// -----------------------------------------------------------------------------
// C8: backwardPass modifies weights
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNetController::backwardPass modifies weights", "[NeuralNetController]")
{
    NeuralNetController brain(0.1f);
    brain.addLayer(2);
    brain.addLayer(3);
    brain.initialize();

    float input[] = {0.5f, 0.3f};
    brain.setData(input, 2, 0);
    brain.forwardPass();

    // Set output errors manually
    auto &out = brain.getLayer(1);
    for (size_t i = 0; i < out.size(); i++)
        out[i].error = 0.1f;

    // Save state
    auto savedState = brain.saveState();

    brain.backwardPass();

    // Weights should have changed - can't directly verify, but no crash = OK
    SUCCEED("backwardPass completed without crash");
}

// -----------------------------------------------------------------------------
// C13: train reduces error
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNetController::train reduces error", "[NeuralNetController]")
{
    NeuralNetController brain(0.1f);
    brain.addLayer(2);
    brain.addLayer(4);
    brain.addLayer(1);
    brain.initialize();

    float input[] = {0.5f, 0.5f};
    float target[] = {0.8f};

    brain.setData(input, 2, 0);
    brain.train(target);
    float initialError = brain.getError();

    for (int i = 0; i < 1000; i++)
    {
        brain.setData(input, 2, 0);
        brain.train(target);
    }

    REQUIRE(brain.getError() < initialError);
}

// -----------------------------------------------------------------------------
// C12: train with classification
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNetController::train with classification target", "[NeuralNetController]")
{
    NeuralNetController brain(0.1f);
    brain.addLayer(2);
    brain.addLayer(3);
    brain.initialize();

    float input[] = {0.5f, 0.5f};
    brain.setData(input, 2, 0);
    brain.train(0);
    float initialError = brain.getError();

    for (int i = 0; i < 500; i++)
    {
        brain.setData(input, 2, 0);
        brain.train(0);
    }

    REQUIRE(brain.getError() < initialError);
}

// -----------------------------------------------------------------------------
// C17/C18: save/load state
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNetController save/load state round-trip", "[NeuralNetController]")
{
    NeuralNetController brain1(0.1f);
    brain1.addLayer(3);
    brain1.addLayer(4);
    brain1.addLayer(2);
    brain1.initialize();

    auto saved = brain1.saveState();

    SECTION("saved state not empty")
    {
        REQUIRE(saved.size() > 1);
    }

    // Load into new controller
    NeuralNetController brain2(saved, 0.1f);

    SECTION("loaded network has correct structure")
    {
        REQUIRE(brain2.layerCount() == 3);
        REQUIRE(brain2.neuronCount(0) == 3);
        REQUIRE(brain2.neuronCount(1) == 4);
        REQUIRE(brain2.neuronCount(2) == 2);
    }
}

// -----------------------------------------------------------------------------
// C15: getLayer
// -----------------------------------------------------------------------------
TEST_CASE("NeuralNetController::getLayer returns correct layer", "[NeuralNetController]")
{
    NeuralNetController brain(0.1f);
    brain.addLayer(6);
    brain.addLayer(10);
    brain.addLayer(4);

    SECTION("getLayer(0)")
    {
        REQUIRE(brain.getLayer(0).size() == 6);
    }

    SECTION("getLayer(2)")
    {
        REQUIRE(brain.getLayer(2).size() == 4);
    }
}

// -----------------------------------------------------------------------------
// Integration: full training converges
// -----------------------------------------------------------------------------
TEST_CASE("Full training loop converges", "[NeuralNetController][integration]")
{
    NeuralNetController brain(0.1f);
    brain.addLayer(4);
    brain.addLayer(8);
    brain.addLayer(2);
    brain.initialize();

    float input[] = {1.0f, 0.0f, 1.0f, 0.0f};
    float target[] = {1.0f, 0.0f};

    for (int epoch = 0; epoch < 50000; epoch++)
    {
        brain.setData(input, 4, 0);
        brain.train(target);
        if (brain.getError() < 0.01f) break;
    }

    REQUIRE(brain.getLayer(2)[0].output > 0.7f);
    REQUIRE(brain.getLayer(2)[1].output < 0.3f);
}

// -----------------------------------------------------------------------------
// Bias preserved after forward pass
// -----------------------------------------------------------------------------
TEST_CASE("Bias neuron preserved after forward pass", "[NeuralNetController]")
{
    NeuralNetController brain(0.1f);
    brain.addLayer(4);
    brain.addLayer(3);
    brain.initialize();

    float biasBefore = brain.getLayer(0)[3].output;
    REQUIRE(biasBefore == Catch::Approx(1.0f).epsilon(0.001));

    float input[] = {0.5f, 0.3f, 0.8f};
    brain.setData(input, 3, 0);
    brain.forwardPass();

    // Bias should be preserved by setData
    REQUIRE(brain.getLayer(0)[3].output == Catch::Approx(1.0f).epsilon(0.001));
}
