#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>

#include "NeuralNet.h"
#include "NeuralNetController.h"
#include "MnistReader.h"

// ============================================================================
// Конфигурация обучения
// ============================================================================

// Пути к файлам данных MNIST в формате IDX (бинарный формат, используемый в оригинальном датасете)
// Формат: первый байт — magic number, далее размеры и данные
static const std::string TRAIN_IMAGES = "../data/train-images-idx3-ubyte";  // 60000 изображений рукописных цифр (28x28 пикселей)
static const std::string TRAIN_LABELS = "../data/train-labels-idx1-ubyte";  // 60000 меток (цифры 0-9)
static const std::string TEST_IMAGES = "../data/t10k-images-idx3-ubyte";    // 10000 изображений для тестирования
static const std::string TEST_LABELS = "../data/t10k-labels-idx1-ubyte";    // 10000 тестовых меток

// Архитектура нейронной сети: 784 -> 30 -> 10
// INPUT_SIZE = 784: каждое изображение 28x28 пикселей разворачивается в одномерный вектор (28*28 = 784)
//   Каждый пиксель — значение яркости от 0 до 255, нормализуется до [0, 1]
static const int INPUT_SIZE = 784;

// HIDDEN_SIZE = 30: количество нейронов в скрытом слое
//   Почему 30: компромисс между скоростью обучения и качеством.
//   Больше нейронов (64, 128) дают лучшую точность (+2-3%), но обучаются в 2-4 раза дольше.
//   30 нейронов — достаточный минимум для распознавания простых паттернов цифр.
static const int HIDDEN_SIZE = 30;

// OUTPUT_SIZE = 10: по одному нейрону на каждую цифру (0-9)
//   Нейрон с максимальным выходом определяет предсказанную цифру (argmax)
static const int OUTPUT_SIZE = 10;

// LEARNING_RATE = 0.5: коэффициент обучения (скорость обновления весов)
//   Формула обновления веса: w_new = w_old + lr * error * gradient
//   Почему 0.5: слишком малое значение (< 0.1) — обучение займёт десятки эпох;
//   слишком большое (> 1.0) — веса будут «скакать», сеть не сойдётся.
//   0.5 — эмпирически подобранное значение для данной архитектуры.
static const float LEARNING_RATE = 0.5f;

// MAX_EPOCHS = 5: количество полных проходов по обучающей выборке
//   Почему 5: после 5 эпох сеть на данной архитектуре достигает ~92% точности.
//   Дальнейшее обучение даёт diminishing returns: 6-я эпоха добавит ~0.5%,
//   10-я — ещё ~1%, но время обучения растёт линейно.
//   Для production можно увеличить до 15-20.
static const int MAX_EPOCHS = 5;

// TRAIN_LIMIT = 60000: ограничение на количество обучающих примеров
//   Почему 60000: это полный размер обучающей выборки MNIST.
//   В оригинале в MNIST 60000 тренировочных изображений.
//   Если установить меньшее значение (например, 5000), обучение пройдёт в 12 раз быстрее,
//   но точность будет ниже (~85% вместо ~92%). Полезно для быстрой проверки кода.
//   0 = без ограничения (загрузить все доступные).
static const int TRAIN_LIMIT = 60000;

// TEST_LIMIT = 0: ограничение на количество тестовых примеров
//   0 = загрузить все 10000 тестовых изображений без ограничений.
//   Можно установить, например, 1000 для быстрого тестирования.
static const int TEST_LIMIT = 0;

// Путь для сохранения обученных весов сети
// Формат: текстовый файл, каждая строка — один слой (веса + смещения)
static const std::string WEIGHTS_FILE = "../data/mnist_weights.w";

// ============================================================================
// Вспомогательные функции
// ============================================================================

/**
 * Преобразует цифру (0-9) в вектор целевых значений для обучения сети.
 *
 * Почему 0.9 и 0.1, а не 1.0 и 0.0?
 *
 * Проблема насыщения сигмоиды: функция активации sigmoid(x) = 1/(1+e^(-x))
 * асимптотически приближается к 0 и 1, но никогда их не достигает.
 * При значениях, близких к 0 или 1, производная сигмоиды стремится к нулю:
 *   sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
 *   При sigmoid(x) = 0.999 -> derivative = 0.999 * 0.001 = 0.001 (почти ноль!)
 *
 * Если целевое значение = 1.0, градиент ошибки будет очень маленьким,
 * и веса почти не обновятся — обучение «застрянет».
 *
 * Значения 0.9/0.1 дают «запас» для градиента:
 *   При target = 0.9, output = 0.85 -> derivative = 0.85 * 0.15 = 0.1275 (значимый градиент)
 *
 * Это стандартный приём в нейронных сетях с сигмоидальной активацией.
 * Для ReLU-активации такой проблемы нет, но в данной библиотеке только sigmoid.
 *
 * @param digit      Цифра (0-9), которую представляет данный пример
 * @param targets    Массив целевых значений (OUTPUT_SIZE = 10 элементов)
 * @param outputSize Размер массива targets (должен быть 10)
 */
static void digitToTargets(int digit, float targets[], int outputSize)
{
    for (int i = 0; i < outputSize; i++)
        // Нейрон с индексом, равным цифре, должен выдать 0.9 (ближе к 1 — «это моя цифра»)
        // Все остальные нейроны должны выдать 0.1 (ближе к 0 — «это не моя цифра»)
        targets[i] = (i == digit) ? 0.9f : 0.1f;
}

/**
 * Определяет предсказанную цифру по выходам последнего слоя сети.
 *
 * Алгоритм: argmax — нахождение индекса нейрона с максимальным выходным значением.
 *
 * Как это работает:
 *   После прямого прохода (forward pass) каждый из 10 выходных нейронов имеет
 *   значение от 0 до 1 (результат сигмоиды). Чем ближе к 1, тем «увереннее»
 *   сеть, что входное изображение соответствует данной цифре.
 *
 *   Например: [0.05, 0.02, 0.85, 0.03, 0.01, 0.04, 0.02, 0.06, 0.01, 0.03]
 *   Максимум = 0.85 при индексе 2 -> сеть предсказывает цифру 2.
 *
 *   Функция iteratively сравнивает каждый элемент с текущим максимумом,
 *   обновляя maxIdx и maxVal при нахождении большего значения.
 *
 * @param brain            Контроллер нейронной сети (содержит все слои и нейроны)
 * @param outputLayerIndex Индекс выходного слоя (в данной архитектуре = 2)
 * @return Индекс нейрона с максимальным выходом (предсказанная цифра 0-9)
 */
static int predictDigit(const NeuralNetController &brain, int outputLayerIndex)
{
    // Получаем ссылку на выходной слой (вектор нейронов)
    const auto &layer = brain.getLayer(outputLayerIndex);

    // Инициализируем максимум первым нейроном
    int maxIdx = 0;
    float maxVal = layer[0].output;

    // Последовательно сравниваем остальные нейроны с текущим максимумом
    for (size_t i = 1; i < layer.size(); i++)
    {
        if (layer[i].output > maxVal)
        {
            maxVal = layer[i].output;  // Обновляем максимальное значение
            maxIdx = static_cast<int>(i);  // Запоминаем индекс этого нейрона
        }
    }
    return maxIdx;
}

/**
 * Вычисляет точность (accuracy) сети на заданном наборе данных.
 *
 * Accuracy = (количество правильных предсказаний / общее количество примеров) * 100%
 *
 * Как считается:
 *   1. Для каждого изображения из датасета:
 *      a. Загружаем пиксели в сеть через setData()
 *      b. Выполняем прямой проход forwardPass() — сеть «думает»
 *      c. Через predictDigit() получаем предсказанную цифру
 *      d. Сравниваем с истинной меткой из датасета
 *   2. Считаем количество совпадений (correct)
 *   3. Возвращаем процент правильных ответов
 *
 * Error (ошибка) считается отдельно в цикле обучения через brain.getError().
 * Это среднеквадратичная ошибка (MSE) между выходами сети и целевыми значениями:
 *   MSE = 1/2 * sum((target_i - output_i)^2) для всех выходных нейронов i
 *   Множитель 1/2 упрощает дифференцирование при backpropagation.
 *
 * @param brain            Контроллер нейронной сети
 * @param dataset          Набор данных (изображения + метки) для оценки
 * @param outputLayerIndex Индекс выходного слоя (2 для данной архитектуры)
 * @param verbose          Если true, выводит прогресс каждые 1000 примеров
 * @return Точность в процентах (0.0 - 100.0)
 */
static float evaluateAccuracy(NeuralNetController &brain, const MnistDataset &dataset,
                              int outputLayerIndex, bool verbose = false)
{
    int correct = 0;  // Счётчик правильных предсказаний

    for (size_t i = 0; i < dataset.size(); i++)
    {
        // Загружаем пиксели изображения в сеть
        // pixels.data() — указатель на массив из 784 значений яркости
        // pixels.size() = 784, 0 — начальный индекс (загружаем с начала)
        brain.setData(dataset.images[i].pixels.data(),
                      static_cast<int>(dataset.images[i].pixels.size()), 0);

        // Выполняем прямой проход: входной слой -> скрытый слой -> выходной слой
        // На каждом нейроне: sum(weights * inputs) + bias -> sigmoid() -> output
        brain.forwardPass();

        // Сравниваем предсказание сети с истинной меткой
        if (predictDigit(brain, outputLayerIndex) == dataset.labels[i])
            correct++;  // Предсказание верное

        // Выводим промежуточный прогресс (каждые 1000 примеров)
        if (verbose && i % 1000 == 0)
        {
            std::cout << "  Evaluated: " << (i + 1) << "/" << dataset.size()
                      << " (accuracy: " << (correct * 100.0f / (i + 1)) << "%)" << std::endl;
        }
    }
    // Возвращаем итоговую точность в процентах
    return (correct * 100.0f) / dataset.size();
}

// ============================================================================
// Основная программа: загрузка -> создание -> обучение -> сохранение -> тестирование
// ============================================================================
int main()
{
    // Заголовок
    std::cout << "========================================" << std::endl;
    std::cout << "  MNIST Neural Network Training" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Architecture: " << INPUT_SIZE << " -> " << HIDDEN_SIZE << " -> " << OUTPUT_SIZE << std::endl;
    std::cout << "Learning rate: " << LEARNING_RATE << std::endl;
    std::cout << "Max epochs: " << MAX_EPOCHS << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------------
    // Шаг 1: Загрузка обучающих данных
    // ------------------------------------------------------------------------
    std::cout << "[1/5] Loading training data..." << std::endl;
    MnistDataset trainData;

    // MnistReader читает бинарные IDX-файлы:
    //   - Изображения: magic_number(4) + num_images(4) + rows(4) + cols(4) + pixel_data(num_images * rows * cols)
    //   - Метки: magic_number(4) + num_labels(4) + label_data(num_labels)
    // TRAIN_LIMIT ограничивает количество загружаемых примеров (60000 = все)
    if (!MnistReader::loadDataset(TRAIN_IMAGES, TRAIN_LABELS, trainData, TRAIN_LIMIT))
    {
        std::cerr << "ERROR: Failed to load training data" << std::endl;
        return 1;
    }
    // Выводит сводку: количество загруженных изображений, размеры, распределение меток
    MnistReader::printSummary(trainData);
    std::cout << std::endl;

    // ------------------------------------------------------------------------
    // Шаг 2: Создание нейронной сети
    // ------------------------------------------------------------------------
    std::cout << "[2/5] Creating neural network..." << std::endl;

    // Создаём контроллер сети с заданным learning rate
    // Контроллер управляет слоями, нейронами, весами и процессом обучения
    NeuralNetController brain(LEARNING_RATE);

    // Добавляем слои по порядку:
    //   Слой 0 (входной): 784 нейронов — по одному на каждый пиксель изображения 28x28
    //   Слой 1 (скрытый): 30 нейронов — извлекают признаки (края, петли, штрихи)
    //   Слой 2 (выходной): 10 нейронов — по одному на каждую цифру 0-9
    brain.addLayer(INPUT_SIZE);
    brain.addLayer(HIDDEN_SIZE);
    brain.addLayer(OUTPUT_SIZE);

    // Инициализация весов и смещений:
    //   - Веса инициализируются случайными значениями (Xavier/He initialization)
    //   - Смещения (bias) инициализируются небольшими значениями
    //   - Используется один RNG на объект для производительности (в 20 раз быстрее)
    brain.initialize();

    std::cout << "  Network created: " << INPUT_SIZE << " -> " << HIDDEN_SIZE << " -> " << OUTPUT_SIZE << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------------
    // Шаг 3: Обучение сети
    // ------------------------------------------------------------------------
    std::cout << "[3/5] Training..." << std::endl;
    std::clock_t trainStart = std::clock();  // Засекаем время начала обучения
    std::vector<float> targets(OUTPUT_SIZE);  // Буфер для целевых значений (переиспользуется)

    // Цикл по эпохам: каждая эпоха — полный проход по всей обучающей выборке
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++)
    {
        float totalError = 0.0f;  // Суммарная ошибка за эпоху (для вычисления средней)
        int correct = 0;          // Количество правильных предсказаний за эпоху

        // Последовательное обучение на каждом примере (online learning, без mini-batch)
        // Один пример за раз: forward -> error -> backward -> update weights
        for (size_t i = 0; i < trainData.size(); i++)
        {
            // Загружаем пиксели текущего изображения в сеть
            brain.setData(trainData.images[i].pixels.data(),
                          static_cast<int>(trainData.images[i].pixels.size()), 0);

            // Преобразуем метку (например, цифру 7) в вектор целевых значений:
            //   [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1]
            //   Индекс 7 = 0.9, все остальные = 0.1
            digitToTargets(trainData.labels[i], targets.data(), OUTPUT_SIZE);

            // Один шаг обучения:
            //   1. forwardPass() — прямой проход, вычисление выходов всех нейронов
            //   2. Вычисление ошибки на выходном слое (targets - outputs)
            //   3. backwardPass() — обратное распространение ошибки (backpropagation):
            //      - Вычисление градиентов для каждого веса (chain rule)
            //      - Обновление весов: w += lr * gradient * error
            //   4. Обновляется internal error (MSE) через getError()
            brain.train(targets.data());

            // Накопление ошибки для вычисления средней за эпоху
            totalError += brain.getError();

            // Проверяем, правильно ли сеть предсказала цифру
            // Слой 2 — выходной слой (индексация: 0=входной, 1=скрытый, 2=выходной)
            if (predictDigit(brain, 2) == trainData.labels[i])
                correct++;
        }

        // Вычисляем среднюю ошибку за эпоху
        // MSE = sum(error_i) / N, где N — количество примеров
        float avgError = totalError / trainData.size();

        // Точность за эпоху: процент правильных предсказаний
        float epochAccuracy = (correct * 100.0f) / trainData.size();

        // Время, затраченное на обучение до текущего момента
        float elapsed = (std::clock() - trainStart) / static_cast<float>(CLOCKS_PER_SEC);

        // Выводим статистику эпохи
        std::cout << "  Epoch " << (epoch + 1) << "/" << MAX_EPOCHS
                  << " | Error: " << avgError
                  << " | Accuracy: " << epochAccuracy << "%"
                  << " | Time: " << elapsed << "s" << std::endl;
    }

    // Общее время обучения (все эпохи)
    float totalTime = (std::clock() - trainStart) / static_cast<float>(CLOCKS_PER_SEC);
    std::cout << "  Total training time: " << totalTime << "s" << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------------
    // Шаг 4: Сохранение обученных весов
    // ------------------------------------------------------------------------
    std::cout << "[4/5] Saving trained weights..." << std::endl;

    // saveState() сериализует все веса и смещения сети в вектор строк
    // Формат каждой строки: "weight1,weight2,...,weightN,bias"
    // Количество строк = количество связей между слоями (в данной сети = 2)
    auto weightStrings = brain.saveState();

    // Записываем в файл для последующего использования (без повторного обучения)
    std::ofstream wf(WEIGHTS_FILE);
    if (wf.is_open())
    {
        for (const auto &line : weightStrings)
            wf << line << std::endl;
        std::cout << "  Saved " << weightStrings.size() << " lines" << std::endl;
    }
    std::cout << std::endl;

    // ------------------------------------------------------------------------
    // Шаг 5: Оценка на тестовой выборке
    // ------------------------------------------------------------------------
    std::cout << "[5/5] Evaluating on test set..." << std::endl;

    // Загружаем тестовые данные (10000 изображений, не участвовавших в обучении)
    // Тестовая выборка нужна для оценки обобщающей способности сети
    // (чтобы убедиться, что сеть не «запомнила» тренировочные примеры)
    MnistDataset testData;
    if (!MnistReader::loadDataset(TEST_IMAGES, TEST_LABELS, testData, TEST_LIMIT))
    {
        std::cerr << "ERROR: Failed to load test data" << std::endl;
        return 1;
    }
    MnistReader::printSummary(testData);
    std::cout << std::endl;

    // Замеряем время и точность на тестовой выборке
    std::clock_t evalStart = std::clock();

    // evaluateAccuracy() проходит по всем тестовым примерам,
    // делает forward pass (без обучения!) и считает процент правильных ответов
    // verbose = true -> выводит прогресс каждые 1000 примеров
    float testAccuracy = evaluateAccuracy(brain, testData, 2, true);
    float evalTime = (std::clock() - evalStart) / static_cast<float>(CLOCKS_PER_SEC);

    // ------------------------------------------------------------------------
    // Итоговые результаты
    // ------------------------------------------------------------------------
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Train samples: " << trainData.size() << std::endl;
    std::cout << "  Test samples:  " << testData.size() << std::endl;
    std::cout << "  Test accuracy: " << testAccuracy << "%" << std::endl;
    std::cout << "  Eval time:     " << evalTime << "s" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\nDone. Press Enter to exit..." << std::endl;
    int t = 0;
    std::cin >> t;
    return 0;
}
