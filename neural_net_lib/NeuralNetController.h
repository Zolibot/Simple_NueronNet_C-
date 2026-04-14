#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "NeuralNet.h"

/// Контроллер нейронной сети: управление слоями, весами, обучением
class NeuralNetController {
private:
    /// Все слои сети (входной + скрытые + выходной)
    std::vector<Layer> layers_;

    /// Матрицы весов между слоями
    std::vector<WeightMatrix> weights_;

    /// Индексы bias-нейронов в каждом слое (-1 = нет bias)
    std::vector<int> biasIndices_;

    /// Ядро нейронной сети
    NeuralNet net_;

    /// Скорость обучения
    float learningRate_;
    
    /// Learning rate decay
    float learningRateDecay_ = 0.0f;
    
    /// Текущая эпоха для decay
    int currentEpoch_ = 0;

    /// Тип инициализации весов
    WeightInit weightInit_ = INIT_HE;

    /// Optimizer states
    AdamState adamState_;
    MomentumState momentumState_;

public:
    /// Конструктор
    explicit NeuralNetController(float learningRate);

    /// Конструктор с загрузкой из сохранённого состояния
    NeuralNetController(const std::vector<std::string> &savedState, float learningRate);

    /// Деструктор
    ~NeuralNetController() = default;

    // Запрещаем копирование (Rule of Five)
    NeuralNetController(const NeuralNetController &) = delete;
    NeuralNetController &operator=(const NeuralNetController &) = delete;

    // Разрешаем перемещение
    NeuralNetController(NeuralNetController &&) = default;
    NeuralNetController &operator=(NeuralNetController &&) = default;

    /// Квадрат числа
    static float square(float n);

    /// Обратное прохождение весов (тестовая функция)
    void reverseWeights();

    /// Обучение: классификация (target = один нейрон)
    void train(int targetClass);

    /// Обучение: регрессия (target = массив значений)
    void train(const float targets[]);

    /// Инициализация сети: bias + веса + рандомизация
    void initialize();

    /// Установить тип инициализации весов (по умолчанию INIT_HE)
    void setWeightInit(WeightInit init) { weightInit_ = init; }

    /// Установить тип активации
    void setActivation(ActivationType type) { net_.activationType = type; }

    /// Установить оптимизатор
    void setOptimizer(OptimizerType type) { net_.optimizerType = type; }

    /// Установить dropout
    void setDropout(float rate) { net_.useDropout = (rate > 0.0f); net_.dropoutRate = rate; }

    /// Включить batch normalization
    void setBatchNorm(bool enable) { net_.useBatchNorm = enable; }

    /// Установить learning rate decay
    void setLearningRateDecay(float decay) { learningRateDecay_ = decay; }

    /// Обучение mini-batch
    void trainMiniBatch(const std::vector<std::vector<float>> &inputs,
                        const std::vector<std::vector<float>> &targets);

    /// Обучение с Adam
    void trainAdam(const float targets[]);

    /// Добавить слой с указанным количеством нейронов
    void addLayer(int neuronCount);

    /// Создать матрицы весов между слоями
    void addWeights();

    /// Установить входные данные в слой c
    void setData(const float data[], int len, int layerIndex);

    /// Установить bias-нейроны (value = 1.0, error = 0.0)
    void setBias();

    /// Прямое распространение по всей сети
    void forwardPass();

    /// Обратное распространение по всей сети
    void backwardPass();

    /// Получить общую ошибку (сумма квадратов ошибок выходного слоя)
    float getError() const;

    /// Вычислить ошибки для всех слоёв
    void computeErrors();

    /// Получить слой по индексу (const)
    const Layer &getLayer(int index) const;

    /// Получить слой по индексу (не-const)
    Layer &getLayer(int index);

    /// Ошибка выходного слоя: классификация
    void setOutputError(int targetClass);

    /// Ошибка выходного слоя: регрессия
    void setOutputError(const float targets[]);

    /// Рандомизация всех весов
    void randomizeWeights();

    /// Загрузить состояние из строк
    void loadState(const std::vector<std::string> &savedState);

    /// Сохранить состояние в строки
    std::vector<std::string> saveState() const;

    /// Количество слоёв
    int layerCount() const { return static_cast<int>(layers_.size()); }

    /// Количество нейронов в слое
    int neuronCount(int layerIndex) const { return static_cast<int>(layers_.at(layerIndex).size()); }

    /// Количество скрытых слоёв (без выходного)
    int hiddenLayerCount() const { return layerCount() - 1; }
};
