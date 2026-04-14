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

float NeuralNet::tanhActivation(float x)
{
    return std::tanh(x);
}

float NeuralNet::tanhDerivative(float activatedOutput)
{
    return 1.0f - activatedOutput * activatedOutput;
}

float NeuralNet::activate(float x, ActivationType type)
{
    switch (type)
    {
    case ACTIVATION_RELU:
        return relu(x);
    case ACTIVATION_LEAKY_RELU:
        return leakyRelu(x);
    case ACTIVATION_TANH:
        return tanhActivation(x);
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
    case ACTIVATION_TANH:
        return tanhDerivative(activatedOutput);
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

        ouL[i].z = sum;
        ouL[i].output = activate(sum, activationType);
    }
}

void NeuralNet::forwardPassWithDropout(const Layer &inL, const WeightMatrix &weN, Layer &ouL, Layer &dropoutMask, float rate, int skipBiasIndex)
{
    const int fanOut = static_cast<int>(weN.size());
    const int fanIn  = static_cast<int>(weN[0].size());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < fanOut; i++)
    {
        if (i == skipBiasIndex) continue;

        float sum = 0.0f;
        for (int u = 0; u < fanIn; u++)
            sum += inL[u].output * weN[i][u];

        ouL[i].z = sum;
        ouL[i].output = activate(sum, activationType);
        
        dropoutMask[i].output = (dist(rng_) > rate) ? 1.0f / (1.0f - rate) : 0.0f;
        ouL[i].output *= dropoutMask[i].output;
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

        ouL[i].z = sum;
        ouL[i].output = activate(sum, activationType);
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

    for (int i = 0; i < fanIn - 1; i++)
    {
        float err = 0.0f;
        for (int u = 0; u < fanOut; u++)
            err += weN[u][i] * ouL[u].error;

        outInL[i].error = err;
    }
}

void NeuralNet::computeErrorWithDropout(const Layer &inL, const WeightMatrix &weN, const Layer &ouL, Layer &outInL, const Layer &dropoutMask, int skipBiasIndex)
{
    const int fanIn  = static_cast<int>(weN[0].size());
    const int fanOut = static_cast<int>(weN.size());

    for (int i = 0; i < fanIn - 1; i++)
    {
        float err = 0.0f;
        for (int u = 0; u < fanOut; u++)
            err += weN[u][i] * ouL[u].error * dropoutMask[u].output;

        outInL[i].error = err;
    }
}

void NeuralNet::backwardPass(const Layer &inL, WeightMatrix &weN, const Layer &ouL, float learningRate)
{
    const int fanOut = static_cast<int>(weN.size());
    const int fanIn  = static_cast<int>(weN[0].size());

    for (int i = 0; i < fanOut; i++)
    {
        float deriv = activateDerivative(ouL[i].output, activationType);
        float grad = ouL[i].error * deriv;

        // Gradient clipping to prevent exploding gradients
        if (grad > 5.0f) grad = 5.0f;
        if (grad < -5.0f) grad = -5.0f;

        for (int u = 0; u < fanIn; u++)
            weN[i][u] += learningRate * grad * inL[u].output;
    }
}

void NeuralNet::backwardPassAdam(const Layer &inL, WeightMatrix &weN, const Layer &ouL, float learningRate, WeightMatrix &m, WeightMatrix &v, int t)
{
    const int fanOut = static_cast<int>(weN.size());
    const int fanIn  = static_cast<int>(weN[0].size());

    float beta1t = adamBeta1;
    float beta2t = adamBeta2;
    for (int k = 0; k < t; k++) {
        beta1t *= adamBeta1;
        beta2t *= adamBeta2;
    }
    float lrHat = learningRate * std::sqrt(1.0f - beta2t) / (1.0f - beta1t);

    for (int i = 0; i < fanOut; i++)
    {
        float deriv = activateDerivative(ouL[i].output, activationType);
        float grad = ouL[i].error * deriv;

        // Gradient clipping to prevent NaN/exploding gradients
        if (grad > 5.0f) grad = 5.0f;
        if (grad < -5.0f) grad = -5.0f;

        for (int u = 0; u < fanIn; u++)
        {
            float g = grad * inL[u].output;
            m[i][u] = adamBeta1 * m[i][u] + (1.0f - adamBeta1) * g;
            v[i][u] = adamBeta2 * v[i][u] + (1.0f - adamBeta2) * g * g;
            float mHat = m[i][u] / (1.0f - beta1t);
            float vHat = v[i][u] / (1.0f - beta2t);
            weN[i][u] += lrHat * mHat / (std::sqrt(vHat) + adamEpsilon);
        }
    }
}

void NeuralNet::backwardPassMomentum(const Layer &inL, WeightMatrix &weN, const Layer &ouL, float learningRate, WeightMatrix &velocity)
{
    const int fanOut = static_cast<int>(weN.size());
    const int fanIn  = static_cast<int>(weN[0].size());

    for (int i = 0; i < fanOut; i++)
    {
        float deriv = activateDerivative(ouL[i].output, activationType);
        float grad = ouL[i].error * deriv;

        // Gradient clipping
        if (grad > 5.0f) grad = 5.0f;
        if (grad < -5.0f) grad = -5.0f;

        for (int u = 0; u < fanIn; u++)
        {
            float g = learningRate * grad * inL[u].output;
            velocity[i][u] = momentum * velocity[i][u] + g;
            weN[i][u] += velocity[i][u];
        }
    }
}

void NeuralNet::setOutputError(const float targets[], int len, Layer &ouL)
{
    for (int i = 0; i < len; i++)
        ouL[i].error = targets[i] - ouL[i].output;
}
