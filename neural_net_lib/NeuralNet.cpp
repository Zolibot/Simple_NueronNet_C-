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
    return 1.0f / (1.0f + std::exp(-x));
}

float NeuralNet::sigmoidDerivative(float activatedOutput)
{
    return activatedOutput * (1.0f - activatedOutput);
}

void NeuralNet::forwardPass(const Layer &inL, const WeightMatrix &weN, Layer &ouL, int skipBiasIndex)
{
    for (size_t i = 0; i < weN.size(); i++)
    {
        // Пропускаем bias-нейрон
        if (static_cast<int>(i) == skipBiasIndex)
            continue;

        float sum = 0.0f;
        for (size_t u = 0; u < weN[0].size(); u++)
        {
            sum += inL[u].output * weN[i][u];
        }
        ouL[i].output = sigmoid(sum);
    }
}

void NeuralNet::reversePass(const Layer &inL, const WeightMatrix &weN, Layer &ouL, int skipBiasIndex)
{
    for (size_t i = 0; i < weN[0].size(); i++)
    {
        if (static_cast<int>(i) == skipBiasIndex)
            continue;

        float sum = 0.0f;
        for (size_t u = 0; u < weN.size(); u++)
        {
            sum += inL[u].output * weN[u][i];
        }
        ouL[i].output = sigmoid(sum);
    }
}

void NeuralNet::randomizeWeights(WeightMatrix &we, WeightInit init)
{
    float scale = 0.5f; // default uniform range

    if (init == INIT_XAVIER || init == INIT_HE)
    {
        int fanIn = static_cast<int>(we[0].size());
        int fanOut = static_cast<int>(we.size());

        if (init == INIT_XAVIER)
            scale = std::sqrt(2.0f / (fanIn + fanOut));
        else // INIT_HE
            scale = std::sqrt(2.0f / fanIn);
    }

    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto &row : we)
    {
        for (auto &w : row)
        {
            w = dist(rng_);
        }
    }
}

void NeuralNet::computeError(const Layer &inL, const WeightMatrix &weN, const Layer &ouL, Layer &outInL)
{
    for (size_t i = 0; i < weN[0].size(); i++)
    {
        outInL[i].error = 0.0f;
        for (size_t u = 0; u < weN.size(); u++)
        {
            outInL[i].error += weN[u][i] * ouL[u].error;
        }
        // Умножаем на производную сигмоиды
        outInL[i].error *= sigmoidDerivative(outInL[i].output);
    }
}

void NeuralNet::backwardPass(const Layer &inL, WeightMatrix &weN, const Layer &ouL, float learningRate)
{
    for (size_t i = 0; i < weN.size(); i++)
    {
        float gradient = ouL[i].error * sigmoidDerivative(ouL[i].output);
        for (size_t u = 0; u < weN[0].size(); u++)
        {
            weN[i][u] += learningRate * gradient * inL[u].output;
        }
    }
}

void NeuralNet::setOutputError(const float targets[], int len, Layer &ouL)
{
    for (int i = 0; i < len; i++)
    {
        ouL[i].error = targets[i] - ouL[i].output;
    }
}
