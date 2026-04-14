#include "NeuralNet.h"

#include <algorithm>

NeuralNet::NeuralNet()
    : rng_(std::random_device{}())
{
}

float NeuralNet::random(float low, float high)
{
    std::uniform_real_distribution<float> dist(low, high);
    return dist(rng_);
}

float NeuralNet::sigmoid(float x)
{
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

float NeuralNet::sigmoidDerivative(float activatedOutput)
{
    return activatedOutput * (1.0f - activatedOutput);
}

float NeuralNet::relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

float NeuralNet::reluDerivative(float x)
{
    return x > 0.0f ? 1.0f : 0.0f;
}

float NeuralNet::leakyRelu(float x)
{
    return x > 0.0f ? x : 0.01f * x;
}

float NeuralNet::leakyReluDerivative(float x)
{
    return x > 0.0f ? 1.0f : 0.01f;
}

float NeuralNet::activate(float x, ActivationType type)
{
    switch (type)
    {
    case ACTIVATION_RELU:
        return relu(x);
    case ACTIVATION_LEAKY_RELU:
        return leakyRelu(x);
    case ACTIVATION_SIGMOID:
    default:
        return sigmoid(x);
    }
}

float NeuralNet::activateDerivative(float activatedOutput, ActivationType type)
{
    switch (type)
    {
    case ACTIVATION_RELU:
        return reluDerivative(activatedOutput);
    case ACTIVATION_LEAKY_RELU:
        return leakyReluDerivative(activatedOutput);
    case ACTIVATION_SIGMOID:
    default:
        return sigmoidDerivative(activatedOutput);
    }
}

void NeuralNet::forwardPass(const Layer &inL, const WeightMatrix &weN, Layer &ouL, int skipBiasIndex)
{
    const int fanOut = static_cast<int>(weN.size());
    const int fanIn  = static_cast<int>(weN[0].size());

    for (int i = 0; i < fanOut; i++)
    {
        if (i == skipBiasIndex) continue;

        float sum = 0.0f;
        for (int u = 0; u < fanIn; u++)
            sum += inL[u].output * weN[i][u];

        ouL[i].output = sigmoid(sum);
    }
}

void NeuralNet::reversePass(const Layer &inL, const WeightMatrix &weN, Layer &ouL, int skipBiasIndex)
{
    const int fanOut = static_cast<int>(weN.size());
    const int fanIn  = static_cast<int>(weN[0].size());

    for (int i = 0; i < fanIn; i++)
    {
        if (i == skipBiasIndex) continue;

        float sum = 0.0f;
        for (int u = 0; u < fanOut; u++)
            sum += inL[u].output * weN[u][i];

        ouL[i].output = sigmoid(sum);
    }
}

void NeuralNet::randomizeWeights(WeightMatrix &we, WeightInit init)
{
    float scale = 0.5f;

    if (init == INIT_XAVIER || init == INIT_HE)
    {
        int fanIn = static_cast<int>(we[0].size());
        int fanOut = static_cast<int>(we.size());
        if (init == INIT_XAVIER)
            scale = std::sqrt(2.0f / (fanIn + fanOut));
        else
            scale = std::sqrt(2.0f / fanIn);
    }

    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto &row : we)
        for (auto &w : row)
            w = dist(rng_);
}

void NeuralNet::computeError(const Layer &inL, const WeightMatrix &weN, const Layer &ouL, Layer &outInL)
{
    const int fanIn  = static_cast<int>(weN[0].size());
    const int fanOut = static_cast<int>(weN.size());

    // Propagate errors backwards, skip bias neuron
    for (int i = 0; i < fanIn - 1; i++)
    {
        float err = 0.0f;
        for (int u = 0; u < fanOut; u++)
            err += weN[u][i] * ouL[u].error;

        outInL[i].error = err;
        // sigmoidDerivative applied in backwardPass
    }
}

void NeuralNet::backwardPass(const Layer &inL, WeightMatrix &weN, const Layer &ouL, float learningRate)
{
    const int fanOut = static_cast<int>(weN.size());
    const int fanIn  = static_cast<int>(weN[0].size());

    for (int i = 0; i < fanOut; i++)
    {
        float grad = ouL[i].error * sigmoidDerivative(ouL[i].output);
        for (int u = 0; u < fanIn; u++)
            weN[i][u] += learningRate * grad * inL[u].output;
    }
}

void NeuralNet::setOutputError(const float targets[], int len, Layer &ouL)
{
    for (int i = 0; i < len; i++)
        ouL[i].error = targets[i] - ouL[i].output;
}
