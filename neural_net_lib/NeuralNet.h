#pragma once

#include <vector>
#include <random>
#include <cmath>

/// Нейрон: выходное значение и ошибка (градиент)
struct Neuron {
    float output;  // Результат после функции активации
    float error;   // Ошибка (градиент) для обратного распространения
};

/// Слой нейронов
using Layer = std::vector<Neuron>;

/// Матрица весов: [нейрон_выхода][нейрон_входа]
using WeightMatrix = std::vector<std::vector<float>>;

/// Основная логика нейронной сети (stateless utility)
class NeuralNet {
private:
    /// Генератор случайных чисел (один на весь объект)
    mutable std::default_random_engine rng_;

public:
    /// Диапазон инициализации весов [-weightRange, +weightRange]
    float weightRange = 0.5f;

    NeuralNet();

    /// Случайное число в диапазоне [low, high]
    float random(float low, float high);

    /// Сигмоида: σ(x) = 1 / (1 + e^(-x))
    static float sigmoid(float x);

    /// Производная сигмоиды: σ'(x) = σ(x) * (1 - σ(x))
    static float sigmoidDerivative(float activatedOutput);

    /// Прямое распространение (forward pass)
    /// inL  — входной слой
    /// weN  — матрица весов
    /// ouL  — выходной слой (заполняется)
    /// skipBiasIndex — индекс bias-нейрона в ouL (пропускается, -1 = нет bias)
    void forwardPass(const Layer &inL, const WeightMatrix &weN, Layer &ouL, int skipBiasIndex = -1);

    /// Обратное распространение (тестовая функция)
    void reversePass(const Layer &inL, const WeightMatrix &weN, Layer &ouL, int skipBiasIndex = -1);

    /// Инициализация весов случайными значениями
    void randomizeWeights(WeightMatrix &we);

    /// Вычисление ошибок для скрытых слоёв (обратное распространение ошибки)
    void computeError(const Layer &inL, const WeightMatrix &weN, const Layer &ouL, Layer &outInL);

    /// Корректировка весов (градиентный спуск)
    void backwardPass(const Layer &inL, WeightMatrix &weN, const Layer &ouL, float learningRate);

    /// Вычисление ошибки выходного слоя: error[i] = target[i] - output[i]
    static void setOutputError(const float targets[], int len, Layer &ouL);
};
