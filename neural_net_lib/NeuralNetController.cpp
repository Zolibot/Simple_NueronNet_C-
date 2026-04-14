#include "NeuralNetController.h"

#include <algorithm>
#include <stdexcept>

// ============================================================================
// Конструкторы / деструктор
// ============================================================================

NeuralNetController::NeuralNetController(float learningRate)
    : learningRate_(learningRate)
{
}

NeuralNetController::NeuralNetController(const std::vector<std::string> &savedState, float learningRate)
    : learningRate_(learningRate)
{
    loadState(savedState);
}

// ============================================================================
// Публичные методы
// ============================================================================

float NeuralNetController::square(float n)
{
    return n * n;
}

void NeuralNetController::reverseWeights()
{
    for (int i = layerCount() - 2; i >= 0; i--)
    {
        int skipBias = biasIndices_.at(i + 1);
        net_.reversePass(layers_.at(i + 1), weights_.at(i), layers_.at(i), skipBias);
    }
}

void NeuralNetController::train(int targetClass)
{
    forwardPass();
    setOutputError(targetClass);
    computeErrors();
    backwardPass();
}

void NeuralNetController::train(const float targets[])
{
    forwardPass();
    setOutputError(targets);
    computeErrors();
    backwardPass();
}

void NeuralNetController::initialize()
{
    setBias();
    addWeights();
    randomizeWeights();
}

void NeuralNetController::addLayer(int neuronCount)
{
    if (neuronCount <= 0)
    {
        throw std::invalid_argument("neuronCount must be positive");
    }

    Layer layer(neuronCount, {0.0f, 0.0f});
    layers_.push_back(std::move(layer));
    biasIndices_.push_back(-1);
}

void NeuralNetController::addWeights()
{
    if (layerCount() < 2)
        return;

    for (int i = 0; i < hiddenLayerCount(); i++)
    {
        int outputNeurons = neuronCount(i + 1);
        int inputNeurons = neuronCount(i);
        weights_.emplace_back(outputNeurons, std::vector<float>(inputNeurons, 0.0f));
    }
}

void NeuralNetController::setData(const float data[], int len, int layerIndex)
{
    if (layerIndex < 0 || layerIndex >= layerCount())
    {
        throw std::out_of_range("layerIndex out of range");
    }

    auto &layer = layers_.at(layerIndex);
    int biasIdx = biasIndices_.at(layerIndex);

    if (len > static_cast<int>(layer.size()))
    {
        throw std::out_of_range("data length exceeds layer size");
    }

    for (int e = 0; e < len; e++)
    {
        layer[e].output = data[e];
    }

    // Восстанавливаем bias-нейрон
    if (biasIdx >= 0)
    {
        layer[biasIdx].output = 1.0f;
        layer[biasIdx].error = 0.0f;
    }
}

void NeuralNetController::setBias()
{
    for (int i = 0; i < hiddenLayerCount(); i++)
    {
        int biasIdx = neuronCount(i) - 1;
        layers_[i][biasIdx].output = 1.0f;
        layers_[i][biasIdx].error = 0.0f;
        biasIndices_[i] = biasIdx;
    }
}

void NeuralNetController::forwardPass()
{
    for (int i = 0; i < hiddenLayerCount(); i++)
    {
        int skipBias = (i + 1 < layerCount()) ? biasIndices_.at(i + 1) : -1;
        net_.forwardPass(layers_.at(i), weights_.at(i), layers_.at(i + 1), skipBias);
    }
}

void NeuralNetController::backwardPass()
{
    for (int i = layerCount() - 1; i >= 1; i--)
    {
        net_.backwardPass(layers_.at(i - 1), weights_.at(i - 1), layers_.at(i), learningRate_);
    }
}

float NeuralNetController::getError() const
{
    if (layers_.empty())
        return 0.0f;

    const auto &outLayer = layers_.back();
    float error = 0.0f;
    for (const auto &neuron : outLayer)
    {
        error += square(neuron.error);
    }
    return error;
}

void NeuralNetController::computeErrors()
{
    for (int i = hiddenLayerCount() - 1; i >= 0; i--)
    {
        net_.computeError(layers_.at(i), weights_.at(i), layers_.at(i + 1), layers_.at(i));
    }
}

const Layer &NeuralNetController::getLayer(int index) const
{
    return layers_.at(index);
}

Layer &NeuralNetController::getLayer(int index)
{
    return layers_.at(index);
}

void NeuralNetController::setOutputError(int targetClass)
{
    auto &outLayer = layers_.back();
    for (size_t r = 0; r < outLayer.size(); r++)
    {
        if (static_cast<int>(r) == targetClass)
        {
            outLayer[r].error = 1.0f - outLayer[r].output;
        }
        else
        {
            outLayer[r].error = 0.0f - outLayer[r].output;
        }
    }
}

void NeuralNetController::setOutputError(const float targets[])
{
    auto &outLayer = layers_.back();
    for (size_t r = 0; r < outLayer.size(); r++)
    {
        outLayer[r].error = targets[r] - outLayer[r].output;
    }
}

void NeuralNetController::randomizeWeights()
{
    for (auto &wm : weights_)
    {
        net_.randomizeWeights(wm, weightInit_);
    }
}

void NeuralNetController::loadState(const std::vector<std::string> &savedState)
{
    if (savedState.empty())
        return;

    // Первая строка: размеры слоёв
    std::istringstream stream(savedState[0]);
    int size;
    while (stream >> size)
    {
        if (size > 0)
            addLayer(size);
    }

    if (layers_.empty())
        return;

    addWeights();
    setBias();

    // Остальные строки: веса
    int lineIdx = 1;
    for (size_t i = 0; i < weights_.size(); i++)
    {
        for (size_t x = 0; x < weights_[i][0].size(); x++)
        {
            if (lineIdx >= static_cast<int>(savedState.size()))
                break;

            std::istringstream ws(savedState[lineIdx]);
            float w;
            int z = 0;
            while (ws >> w)
            {
                if (z < static_cast<int>(weights_[i].size()))
                {
                    weights_[i][z][x] = w;
                    z++;
                }
            }
            lineIdx++;
        }
    }
}

std::vector<std::string> NeuralNetController::saveState() const
{
    std::vector<std::string> result;

    // Первая строка: размеры слоёв
    std::ostringstream sizes;
    for (const auto &layer : layers_)
        sizes << layer.size() << " ";
    result.push_back(sizes.str());

    // Веса
    std::ostringstream wss;
    for (size_t i = 0; i < weights_.size(); i++)
    {
        for (size_t x = 0; x < weights_[i][0].size(); x++)
        {
            for (size_t u = 0; u < weights_[i].size(); u++)
                wss << weights_[i][u][x] << " ";
            wss << "\n";
        }
    }

    std::string line;
    std::istringstream wsi(wss.str());
    while (std::getline(wsi, line))
    {
        if (!line.empty())
            result.push_back(line);
    }

    return result;
}
