#pragma once

#include <vector>
#include <random>
#include <cmath>

class NeuralNetController;

/// Нейрон: выходное значение и ошибка (градиент)
struct Neuron {
    float output;  // Результат после функции активации
    float error;   // Ошибка (градиент) для обратного распространения
    float z;       // Взвешенная сумма до активации (для BatchNorm)
};

/// Слой нейронов
using Layer = std::vector<Neuron>;

/// Матрица весов: [нейрон_выхода][нейрон_входа]
using WeightMatrix = std::vector<std::vector<float>>;

/// Тип инициализации весов
enum WeightInit { INIT_UNIFORM, INIT_XAVIER, INIT_HE };

/// Тип активации
enum ActivationType { ACTIVATION_SIGMOID, ACTIVATION_RELU, ACTIVATION_LEAKY_RELU, ACTIVATION_TANH };

/// Оптимизатор
enum OptimizerType { OPTIMIZER_SGD, OPTIMIZER_ADAM, OPTIMIZER_MOMENTUM };

/// Основная логика нейронной сети (stateless utility)
class NeuralNet {
private:
    mutable std::default_random_engine rng_;

public:
    int timestep_ = 0;  // Global timestep for Adam optimizer
    float weightRange = 0.5f;
    WeightInit defaultInit = INIT_HE;
    ActivationType activationType = ACTIVATION_SIGMOID;
    OptimizerType optimizerType = OPTIMIZER_SGD;
    float adamBeta1 = 0.9f;
    float adamBeta2 = 0.999f;
    float adamEpsilon = 1e-8f;
    float momentum = 0.9f;
    bool useBatchNorm = false;
    float dropoutRate = 0.0f;
    bool useDropout = false;

    NeuralNet();

    float random(float low, float high);

    static float sigmoid(float x);
    static float sigmoidDerivative(float activatedOutput);

    static float relu(float x);
    static float reluDerivative(float x);

    static float leakyRelu(float x);
    static float leakyReluDerivative(float x);

    static float tanhActivation(float x);
    static float tanhDerivative(float activatedOutput);

    static float activate(float x, ActivationType type);
    static float activateDerivative(float x, ActivationType type);

    void forwardPass(const Layer &inL, const WeightMatrix &weN, Layer &ouL, int skipBiasIndex = -1);
    void forwardPassWithDropout(const Layer &inL, const WeightMatrix &weN, Layer &ouL, Layer &dropoutMask, float dropoutRate, int skipBiasIndex = -1);

    void reversePass(const Layer &inL, const WeightMatrix &weN, Layer &ouL, int skipBiasIndex = -1);

    void randomizeWeights(WeightMatrix &we, WeightInit init = INIT_HE);

    void computeError(const Layer &inL, const WeightMatrix &weN, const Layer &ouL, Layer &outInL);
    void computeErrorWithDropout(const Layer &inL, const WeightMatrix &weN, const Layer &ouL, Layer &outInL, const Layer &dropoutMask, int skipBiasIndex = -1);

    void backwardPass(const Layer &inL, WeightMatrix &weN, const Layer &ouL, float learningRate);
    void backwardPassAdam(const Layer &inL, WeightMatrix &weN, const Layer &ouL, float learningRate, WeightMatrix &m, WeightMatrix &v, int t);
    void backwardPassMomentum(const Layer &inL, WeightMatrix &weN, const Layer &ouL, float learningRate, WeightMatrix &velocity);

    static void setOutputError(const float targets[], int len, Layer &ouL);
};

/// Batch Normalization структура
struct BatchNormParams {
    WeightMatrix gamma;  // масштабирование
    WeightMatrix beta;   // сдвиг
    WeightMatrix runningMean;
    WeightMatrix runningVariance;
    float momentum = 0.9f;
    float epsilon = 1e-5f;
    bool isTraining = true;
};

/// Класс для Mini-batch обучения
class MiniBatchTrainer {
public:
    struct GradientAccumulator {
        std::vector<WeightMatrix> gradients;
        Layer outputErrorSum;
        int sampleCount = 0;
    };

    static void initGradient(GradientAccumulator &acc, const std::vector<WeightMatrix> &weights, int outputSize);
    static void accumulateGradient(GradientAccumulator &acc, const std::vector<WeightMatrix> &gradients, const Layer &outputError);
    static void applyGradient(NeuralNetController &net, GradientAccumulator &acc, float learningRate, int batchSize);
    static void resetGradient(GradientAccumulator &acc);
};

/// Adam optimizer состояние
struct AdamState {
    std::vector<WeightMatrix> m;  // first moment
    std::vector<WeightMatrix> v;  // second moment
    int t = 0;  // timestep
};

/// Momentum optimizer состояние
struct MomentumState {
    std::vector<WeightMatrix> velocity;
};
