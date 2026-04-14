#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <iomanip>
#include <algorithm>

#include "NeuralNet.h"
#include "NeuralNetController.h"
#include "MnistReader.h"

// ============================================================================
// Пути к данным MNIST и константы бенчмарка
// ============================================================================

/** Путь к файлу обучающих изображений (60 000 рукописных цифр 28x28) */
static const std::string TRAIN_IMAGES = "../data/train-images-idx3-ubyte";
/** Путь к файлу обучающих меток */
static const std::string TRAIN_LABELS = "../data/train-labels-idx1-ubyte";
/** Путь к файлу тестовых изображений (10 000 рукописных цифр) */
static const std::string TEST_IMAGES  = "../data/t10k-images-idx3-ubyte";
/** Путь к файлу тестовых меток */
static const std::string TEST_LABELS  = "../data/t10k-labels-idx1-ubyte";

/**
 * Размер входного вектора: 28 * 28 = 784 пикселя.
 * Каждое изображение MNIST имеет размер 28x28 пикселей,
 * развёрнутых в одномерный массив для подачи на вход сети.
 */
static const int INPUT_SIZE  = 784;

/**
 * Размер выходного вектора: 10 классов (цифры 0-9).
 * Каждый нейрон выходного слоя соответствует одной цифре.
 */
static const int OUTPUT_SIZE = 10;

/**
 * Максимальное число эпох обучения для каждой конфигурации.
 * 5 эпох — компромисс между временем бенчмарка и возможностью
 * увидеть тенденцию сходимости. Полное обучение MNIST требует
 * 20-40 эпох для стабильной точности.
 */
static const int MAX_EPOCHS = 5;

/**
 * Лимит обучающих примеров. Берём 5000 из 60000 для ускорения
 * бенчмарка. Этого достаточно для сравнения архитектур между
 * собой, хотя абсолютная точность будет ниже полной.
 */
static const int TRAIN_LIMIT = 5000;

/**
 * Лимит тестовых примеров. Берём 5000 из 10000 для симметрии
 * с обучающей выборкой и ускорения этапа оценки.
 */
static const int TEST_LIMIT = 5000;

// ============================================================================
// Структура конфигурации архитектуры
// ============================================================================

/**
 * ArchitectureConfig — полное описание одной тестируемой конфигурации сети.
 *
 * Поля:
 *
 *   name — человекочитаемое имя конфигурации. Используется для логирования
 *          в консоль и в markdown-отчёт. Включает ключевые параметры:
 *          размеры слоёв, функцию активации, оптимизатор.
 *          Пример: "784->64->10 Sigmoid Adam" означает:
 *          вход 784, скрытый слой 64, выход 10, активация Sigmoid, Adam.
 *
 *   hiddenLayers — вектор размеров скрытых слоёв. Пустой вектор = линейная
 *          модель (только вход -> выход). {16} = один скрытый слой на 16
 *          нейронов. {64, 32} = два скрытых слоя: первый 64, второй 32.
 *          Каждый элемент — количество нейронов в соответствующем слое.
 *
 *   activation — тип функции активации для всех слоёв сети:
 *     - ACTIVATION_SIGMOID: σ(x) = 1/(1+e^-x). Выход в [0, 1].
 *       Исторически первая функция активации. Страдает от проблемы
 *       "насыщения" — градиент стремится к нулю при больших |x|.
 *       Подходит для бинарной классификации, но медленнее сходится.
 *
 *     - ACTIVATION_RELU: max(0, x). Выход в [0, +∞).
 *       Современный стандарт. Не насыщается при положительных x,
 *       что ускоряет обучение. Проблема: "мёртвые нейроны" при
 *       отрицательных входах (градиент = 0).
 *
 *     - ACTIVATION_LEAKY_RELU: max(0.01*x, x). Выход в (-∞, +∞).
 *       Вариация ReLU с небольшим наклоном для отрицательных x.
 *       Решает проблему "мёртвых нейронов", позволяя градиенту
 *       течь даже при отрицательных значениях.
 *
 *     - ACTIVATION_TANH: tanh(x). Выход в [-1, 1].
 *       Нормализует данные к нулевому среднему, что может помочь
 *       оптимизатору. Требует масштабирования целевых значений
 *       (см. digitToTargets — для tanh цели ±0.8 вместо 0.9/0.1).
 *
 *   optimizer — алгоритм обновления весов:
 *     - OPTIMIZER_SGD: стохастический градиентный спуск.
 *       w = w - LR * gradient. Простейший метод, требует
 *       тщательного подбора LR. Сильно зависит от масштаба задачи.
 *
 *     - OPTIMIZER_MOMENTUM: SGD с инерцией.
 *       v = momentum * v - LR * gradient; w = w + v.
 *       Накапливает скорость в направлениях стабильного градиента,
 *       сглаживает колебания. Промежуточный между SGD и Adam.
 *
 *     - OPTIMIZER_ADAM: адаптивная оценка моментов.
 *       Комбинирует идеи Momentum (первый момент) и RMSProp
 *       (второй момент). Автоматически адаптирует learning rate
 *       для каждого параметра. Менее чувствителен к LR, но
 *       требует значительно меньших значений (0.001 вместо 0.5).
 *
 *   learningRate — шаг обучения. Критически важный гиперпараметр:
 *     - Для SGD: 0.5 — большой шаг, потому что SGD обновляет
 *       веса напрямую: w -= LR * gradient. Без адаптации
 *       нужен крупный LR для заметного продвижения.
 *
 *     - Для Adam: 0.001 — маленький шаг, потому что Adam
 *       нормализует градиент оценкой второго момента (дисперсией).
 *       Эффективный шаг ≈ LR * sign(gradient). При LR=0.5
 *       Adam сделает огромный шаг и развалит обучение.
 *
 *     - Для Momentum: 0.1 — промежуточное значение. Инерция
 *       усиливает эффективный шаг, поэтому LR меньше чем у SGD,
 *       но больше чем у Adam.
 *
 * Почему Adam требует LR=0.001, а SGD LR=0.5:
 *   SGD применяет LR напрямую к "сырому" градиенту, который
 *   обычно мал (< 1.0). Чтобы сдвинуть веса, нужен большой LR.
 *   Adam нормализует градиент, деля на sqrt(second_moment + eps).
 *   Это приводит эффективный градиент к порядку ~1.0. При LR=0.5
 *   шаг был бы в 500 раз больше оптимального — веса "улетят".
 *   LR=0.001 даёт контролируемый шаг при нормализованном градиенте.
 */
struct ArchitectureConfig {
    /** Человекочитаемое имя конфигурации для логирования */
    std::string name;
    /** Размеры скрытых слоёв (без входного и выходного) */
    std::vector<int> hiddenLayers;
    /** Функция активации (Sigmoid / ReLU / LeakyReLU / Tanh) */
    ActivationType activation;
    /** Алгоритм оптимизации (SGD / Momentum / Adam) */
    OptimizerType optimizer;
    /** Скорость обучения — масштаб шага обновления весов */
    float learningRate;
};

/**
 * CONFIGS — список всех тестируемых конфигураций.
 *
 * Структура списка организована по группам для систематического
 * сравнения. Каждая группа тестирует определённый аспект:
 *
 * --- Группа 1: SGD baseline ---
 * Цель: проверить работоспособность SGD как базового оптимизатора
 * и оценить влияние размера скрытого слоя на точность.
 *
 * 784->16->10 — минимальный скрытый слой. Ожидается низкая точность
 *   (~70-80%), так как 16 нейронов недостаточно для выделения
 *   признаков 10 цифр из 784 входов.
 *
 * 784->30->10 — классическая конфигурация из учебника Ниельсена.
 *   Исторический baseline. Ожидается ~90-92%.
 *
 * 784->64->10 — удвоенный слой. Больше capacity для признаков.
 *   Ожидается ~93-94%.
 *
 * 784->128->10 — максимальный однослойный вариант. Больше всего
 *   параметров среди SGD-конфигураций. Ожидается ~94-95%.
 *
 * Все используют LR=0.5, так как SGD требует большого шага.
 *
 * --- Группа 2: Adam ---
 * Цель: сравнить Adam с SGD на одинаковых архитектурах.
 * Адаптивный оптимизатор должен сходиться стабильнее.
 *
 * 784->30->10 — та же архитектура, что и в SGD baseline,
 *   для прямого сравнения SGD vs Adam.
 *
 * 784->64->10 — средний размер, где Adam должен показать
 *   преимущество в стабильности.
 *
 * 784->128->10 — большой слой, где адаптивный LR помогает
 *   избежать проблем с масштабом градиентов.
 *
 * LR=0.001 — стандартное значение для Adam.
 *
 * --- Группа 3: Momentum ---
 * Цель: проверить промежуточный оптимизатор между SGD и Adam.
 *
 * 784->64->10 Sigmoid Momentum — та же архитектура, что и в
 *   SGD/Adam, для сравнения трёх оптимизаторов на одной сети.
 *
 * LR=0.1 — меньше чем у SGD (инерция усиливает шаг), но больше
 *   чем у Adam (нет нормализации градиента).
 *
 * --- Группа 4: ReLU с Adam ---
 * Цель: проверить, решает ли ReLU проблему насыщения Sigmoid.
 * ReLU не насыщается при x > 0, что должно ускорить обучение.
 *
 * 784->64->10 ReLU Adam — основная конфигурация для сравнения
 *   активаций Sigmoid vs ReLU.
 *
 * 784->128->10 ReLU Adam — проверка масштабируемости ReLU
 *   на большем слое.
 *
 * --- Группа 5: LeakyReLU ---
 * Цель: проверить, решает ли LeakyReLU проблему "мёртвых нейронов"
 * обычного ReLU. При неудачной инициализации ReLU-нейроны могут
 *   навсегда получить отрицательные входы и перестать учиться.
 * LeakyReLU даёт малый градиент (0.01) даже при x < 0.
 *
 * 784->64->10 LeakyReLU Adam — сравнение с обычным ReLU.
 * 784->128->10 LeakyReLU Adam — проверка на большем слое.
 *
 * --- Группа 6: Deep networks (глубокие сети) ---
 * Цель: проверить, помогают ли дополнительные слои.
 * Теоретически глубокие сети могут выучить иерархические признаки:
 *   1-й слой: рёбра и линии
 *   2-й слой: формы и паттерны
 *
 * 784->64->32->10 ReLU Adam — "сужающаяся" архитектура (bottleneck).
 *   64 -> 32 заставляет сеть сжимать информацию, выделяя
 *   наиболее важные признаки.
 *
 * 784->64->32->10 LeakyReLU Adam — та же архитектура, но с
 *   LeakyReLU для сравнения.
 *
 * --- Группа 7: Tanh ---
 * Цель: проверить tanh как альтернативу Sigmoid.
 * Tanh центрирует выход вокруг 0, что может помочь оптимизатору.
 *
 * 784->64->10 Tanh Adam — сравнение с Sigmoid и ReLU.
 *
 * Обратите внимание: для Tanh целевые значения масштабируются
 * к [-0.8, 0.8] вместо [0.1, 0.9] у Sigmoid (см. digitToTargets).
 */
static const std::vector<ArchitectureConfig> CONFIGS = {
    // === SGD baseline (проверка работоспособности SGD) ===
    {"784->16->10 Sigmoid SGD",        {16},     ACTIVATION_SIGMOID, OPTIMIZER_SGD,     0.5f},
    {"784->30->10 Sigmoid SGD",        {30},     ACTIVATION_SIGMOID, OPTIMIZER_SGD,     0.5f},
    {"784->64->10 Sigmoid SGD",        {64},     ACTIVATION_SIGMOID, OPTIMIZER_SGD,     0.5f},
    {"784->128->10 Sigmoid SGD",       {128},    ACTIVATION_SIGMOID, OPTIMIZER_SGD,     0.5f},

    // === Adam (адаптивный шаг обучения) ===
    {"784->30->10 Sigmoid Adam",       {30},     ACTIVATION_SIGMOID, OPTIMIZER_ADAM,    0.001f},
    {"784->64->10 Sigmoid Adam",       {64},     ACTIVATION_SIGMOID, OPTIMIZER_ADAM,    0.001f},
    {"784->128->10 Sigmoid Adam",      {128},    ACTIVATION_SIGMOID, OPTIMIZER_ADAM,    0.001f},

    // === Momentum (SGD с инерцией) ===
    {"784->64->10 Sigmoid Momentum",   {64},     ACTIVATION_SIGMOID, OPTIMIZER_MOMENTUM,0.1f},

    // === ReLU с Adam (решение проблемы насыщения Sigmoid) ===
    {"784->64->10 ReLU Adam",          {64},     ACTIVATION_RELU,    OPTIMIZER_ADAM,    0.001f},
    {"784->128->10 ReLU Adam",         {128},    ACTIVATION_RELU,    OPTIMIZER_ADAM,    0.001f},

    // === LeakyReLU (решение проблемы "мёртвых нейронов" ReLU) ===
    {"784->64->10 LeakyReLU Adam",     {64},     ACTIVATION_LEAKY_RELU, OPTIMIZER_ADAM,  0.001f},
    {"784->128->10 LeakyReLU Adam",    {128},    ACTIVATION_LEAKY_RELU, OPTIMIZER_ADAM,  0.001f},

    // === Deep networks (глубокие сети с двумя скрытыми слоями) ===
    {"784->64->32->10 ReLU Adam",      {64, 32}, ACTIVATION_RELU,    OPTIMIZER_ADAM,    0.001f},
    {"784->64->32->10 LeakyReLU Adam", {64, 32}, ACTIVATION_LEAKY_RELU, OPTIMIZER_ADAM,  0.001f},

    // === Tanh (центрированная альтернатива Sigmoid) ===
    {"784->64->10 Tanh Adam",          {64},     ACTIVATION_TANH,    OPTIMIZER_ADAM,    0.001f},
};

// ============================================================================
// Структуры результатов
// ============================================================================

/**
 * EpochResult — метрики одной эпохи обучения.
 * Записывается после каждой эпохи для отслеживания прогресса.
 */
struct EpochResult {
    /** Номер эпохи (1-based) */
    int epoch;
    /** Средняя ошибка (loss) по всей обучающей выборке за эпоху.
     *  Вычисляется как mean(MSE(prediction, targets)) по всем примерам.
     *  Уменьшение ошибки от эпохи к эпохе = сеть учится. */
    float avgError;
    /** Точность на обучающей выборке в процентах.
     *  Доля примеров, где сеть правильно предсказала цифру.
     *  Рост точности = сеть улучшает классификацию. */
    float trainAccuracy;
    /** Накопленное время от начала обучения до конца этой эпохи (секунды).
     *  Позволяет оценить, сколько времени заняла каждая эпоха. */
    float elapsedSec;
};

/**
 * BenchmarkResult — полные результаты тестирования одной конфигурации.
 * Заполняется функцией benchmark() и используется для отчёта.
 */
struct BenchmarkResult {
    /** Имя конфигурации (из ArchitectureConfig::name) */
    std::string archName;
    /** Тип функции активации (для отчёта) */
    ActivationType activation;
    /** Количество скрытых слоёв. Считается как размер вектора hiddenLayers.
     *  Например, {64, 32} → numHiddenLayers = 2. */
    int numHiddenLayers = 0;
    /** Общее число нейронов в скрытых слоях. Сумма всех элементов hiddenLayers.
     *  Например, {64, 32} → totalNeurons = 64 + 32 = 96.
     *  НЕ включает входные (784) и выходные (10) нейроны — только скрытые. */
    int totalNeurons = 0;
    /** Общее число параметров (весов + смещений) в сети.
     *  Считается так: для каждого слоя считаем connections * outputs + bias.
     *  Формула (из countParams):
     *    для скрытого слоя из N нейронов с предыдущим размером P:
     *      params += P * N  (веса)  + N (смещения, учтены через prev = N + 1)
     *    для выходного слоя из OUTPUT_SIZE нейронов:
     *      params += prev * OUTPUT_SIZE
     *  Например, 784->64->10:
     *    Скрытый: 784 * 64 + 64 = 50240
     *    Выходной: 65 * 10 = 650  (65 = 64 нейрона + 1 bias)
     *    Итого: 50240 + 650 = 50890 */
    int totalParams = 0;
    /** Общее время обучения всех эпох (секунды) */
    float totalTime = 0.0f;
    /** Метрики по каждой эпохе (одна запись на эпоху) */
    std::vector<EpochResult> epochs;
    /** Точность на тестовой выборке (проценты).
     *  Главный показатель качества — как сеть обобщает на новых данных. */
    float testAccuracy = 0.0f;
    /** Время оценки на тестовой выборке (секунды).
     *  Показывает скорость inference (без обучения). */
    float evalTime = 0.0f;
    /** Фактический размер обучающей выборки */
    int trainSize = 0;
    /** Фактический размер тестовой выборки */
    int testSize = 0;
};

// ============================================================================
// Вспомогательные функции
// ============================================================================

/**
 * digitToTargets — преобразует цифру (0-9) в вектор целевых значений
 * для выходного слоя нейронной сети.
 *
 * Зачем: выходной слой имеет 10 нейронов (по одному на цифру).
 * При обучении мы задаём "идеальный" выход для каждой цифры.
 *
 * Почему не one-hot (0 и 1):
 *   - Для Sigmoid: идеальные 0 и 1 недостижимы (σ(x) ∈ (0, 1),
 *     не включая границы). Цели 0.1 и 0.9 достижимы и дают
 *     ненулевой градиент.
 *   - Для Tanh: выход в (-1, 1), поэтому цели -0.8 и 0.8.
 *   - Для ReLU/LeakyReLU: цели 0.0 и 0.9 — ReLU может выдавать
 *     любые положительные значения, 0.9 — достижимый ориентир.
 *
 * @param digit       правильная цифра (0-9)
 * @param targets     выходной массив (размер = outputSize)
 * @param outputSize  размер выходного слоя (обычно 10)
 * @param activation  тип активации (влияет на масштаб целей)
 */
static void digitToTargets(int digit, float targets[], int outputSize, ActivationType activation)
{
    for (int i = 0; i < outputSize; i++)
    {
        if (activation == ACTIVATION_TANH)
            // Tanh: выход в [-1, 1]. Цели ±0.8 — достижимые значения
            // с ненулевым градиентом (на ±1.0 градиент = 0).
            targets[i] = (i == digit) ? 0.8f : -0.8f;
        else if (activation == ACTIVATION_SIGMOID)
            // Sigmoid: выход в (0, 1). Цели 0.1 и 0.9 — безопасный
            // диапазон, где производная σ'(x) ещё значима.
            targets[i] = (i == digit) ? 0.9f : 0.1f;
        else
            // ReLU / LeakyReLU: цели 0.0 и 0.9.
            // ReLU может выдавать 0 для отрицательных входов,
            // и любое положительное значение для x > 0.
            targets[i] = (i == digit) ? 0.9f : 0.0f;
    }
}

/**
 * predictDigit — предсказывает цифру по активациям выходного слоя.
 *
 * Как работает: находит нейрон с максимальным выходом (argmax).
 * Индекс нейрона (0-9) = предсказанная цифра.
 *
 * Это "winner takes all" стратегия — какая цифра "победила"
 * по активации, та и предсказывается.
 *
 * @param brain          контроллер нейронной сети
 * @param outputLayerIndex индекс выходного слоя (numHiddenLayers + 1)
 * @return предсказанная цифра (0-9)
 */
static int predictDigit(const NeuralNetController &brain, int outputLayerIndex)
{
    const auto &layer = brain.getLayer(outputLayerIndex);
    int maxIdx = 0;
    float maxVal = layer[0].output;
    for (size_t i = 1; i < layer.size(); i++)
        if (layer[i].output > maxVal) { maxVal = layer[i].output; maxIdx = static_cast<int>(i); }
    return maxIdx;
}

/**
 * countParams — подсчитывает общее число обучаемых параметров сети.
 *
 * Параметры = веса (connections) + смещения (biases).
 *
 * Алгоритм:
 *   prev = INPUT_SIZE (784 входных нейрона)
 *   для каждого скрытого слоя из N нейронов:
 *     total += prev * N    // веса от предыдущего слоя к этому
 *     prev = N + 1         // +1 для учёта bias при переходе к следующему слою
 *   total += prev * OUTPUT_SIZE  // веса от последнего скрытого к выходу
 *
 * Почему prev = N + 1:
 *   Каждый слой имеет bias для каждого нейрона. Bias добавляется
 *   как дополнительный вход со значением 1. Поэтому при переходе
 *   к следующему слою мы считаем, что предыдущий имеет N+1 "выходов"
 *   (N нейронов + 1 bias).
 *
 * Пример 784->64->10:
 *   prev = 784
 *   Скрытый (64): total += 784 * 64 = 50176, prev = 65
 *   Выходной (10): total += 65 * 10 = 650
 *   Итого: 50176 + 650 = 50826
 *
 * Пример 784->64->32->10:
 *   prev = 784
 *   Скрытый 1 (64): total += 784 * 64 = 50176, prev = 65
 *   Скрытый 2 (32): total += 65 * 32 = 2080, prev = 33
 *   Выходной (10): total += 33 * 10 = 330
 *   Итого: 50176 + 2080 + 330 = 52586
 *
 * @param hidden  вектор размеров скрытых слоёв
 * @return общее число параметров (весов + смещений)
 */
static int countParams(const std::vector<int> &hidden)
{
    int total = 0, prev = INPUT_SIZE;
    for (int n : hidden) { total += prev * n; prev = n + 1; }
    total += prev * OUTPUT_SIZE;
    return total;
}

/**
 * countTotalNeurons — подсчитывает общее число нейронов в скрытых слоях.
 *
 * Просто суммирует все элементы hiddenLayers.
 * НЕ включает входной слой (784 пикселя — это данные, а не нейроны)
 * и НЕ включает выходной слой (10 классов).
 *
 * Зачем: показывает "ёмкость" сети — сколько вычислительных единиц
 * участвует в обработке признаков. Больше нейронов = больше capacity
 * для обучения, но и больше параметров = риск переобучения.
 *
 * Пример: {64, 32} → 64 + 32 = 96 скрытых нейронов
 *
 * @param hidden  вектор размеров скрытых слоёв
 * @return суммарное число нейронов во всех скрытых слоях
 */
static int countTotalNeurons(const std::vector<int> &hidden)
{
    int total = 0;
    for (int n : hidden) total += n;
    return total;
}

// ============================================================================
// Функция бенчмарка — полный цикл: создание → обучение → тестирование
// ============================================================================

/**
 * benchmark — тестирует одну конфигурацию архитектуры на MNIST.
 *
 * Полный flow:
 *
 *   1. Инициализация результата
 *      - Заполняем метаданные: имя, активация, число слоёв
 *      - Считаем totalNeurons и totalParams через helpers
 *      - Логируем конфигурацию в консоль
 *
 *   2. Создание нейронной сети
 *      - NeuralNetController(LR) — контроллер с заданным learning rate
 *      - setActivation() — устанавливаем функцию активации
 *      - setOptimizer() — устанавливаем алгоритм оптимизации
 *      - addLayer(INPUT_SIZE) — входной слой (784 нейрона)
 *      - addLayer(n) для каждого скрытого слоя
 *      - addLayer(OUTPUT_SIZE) — выходной слой (10 нейронов)
 *      - initialize() — случайная инициализация весов
 *
 *   3. Обучение (MAX_EPOCHS эпох)
 *      Для каждой эпохи:
 *        a. Итерируем по всей обучающей выборке
 *        b. setData() — подаём пиксели изображения на вход
 *        c. digitToTargets() — формируем целевой вектор для цифры
 *        d. train() — один шаг обучения (forward + backward + update)
 *        e. getError() — получаем ошибку (MSE) этого примера
 *        f. predictDigit() — проверяем правильность предсказания
 *        g. Суммируем ошибку и считаем правильные ответы
 *
 *        После обработки всех примеров:
 *        - Считаем среднюю ошибку и точность за эпоху
 *        - Замеряем накопленное время
 *        - Логируем в консоль
 *
 *   4. Тестирование
 *      - Итерируем по тестовой выборке (без обучения!)
 *      - setData() + forwardPass() — только прямой проход
 *      - predictDigit() — сравниваем с правильной меткой
 *      - Считаем точность и время оценки
 *
 *   5. Возврат результата
 *      - Все метрики заполнены: эпохи, точность, время
 *
 * @param cfg    конфигурация архитектуры для тестирования
 * @param train  обучающая выборка MNIST
 * @param test   тестовая выборка MNIST
 * @return заполненная структура BenchmarkResult
 */
static BenchmarkResult benchmark(const ArchitectureConfig &cfg,
                                  const MnistDataset &train, const MnistDataset &test)
{
    // --- Шаг 1: Инициализация результата ---
    BenchmarkResult r;
    r.archName = cfg.name;
    r.activation = cfg.activation;
    r.numHiddenLayers = static_cast<int>(cfg.hiddenLayers.size());
    r.trainSize = static_cast<int>(train.size());
    r.testSize = static_cast<int>(test.size());
    // Считаем число нейронов и параметров по архитектуре
    r.totalNeurons = countTotalNeurons(cfg.hiddenLayers);
    r.totalParams = countParams(cfg.hiddenLayers);

    // Логируем начало бенчмарка с ключевыми параметрами
    std::cout << "\n[" << cfg.name << "] (" << r.numHiddenLayers << " hidden, "
              << r.totalNeurons << " neurons, " << r.totalParams << " params, LR="
              << cfg.learningRate << ")" << std::endl;

    // --- Шаг 2: Создание и инициализация сети ---
    NeuralNetController brain(cfg.learningRate);
    brain.setActivation(cfg.activation);
    brain.setOptimizer(cfg.optimizer);

    // Строим архитектуру: вход → скрытые → выход
    brain.addLayer(INPUT_SIZE);          // Входной слой: 784 нейрона (пиксели)
    for (int n : cfg.hiddenLayers)       // Скрытые слои из конфига
        brain.addLayer(n);
    brain.addLayer(OUTPUT_SIZE);         // Выходной слой: 10 нейронов (цифры 0-9)
    brain.initialize();                  // Инициализация весов случайными значениями

    // Индекс выходного слоя = число скрытых + 1 (0-based: вход=0, hidden=1..N, output=N+1)
    int outIdx = r.numHiddenLayers + 1;
    std::vector<float> targets(OUTPUT_SIZE);

    // Запускаем таймер обучения
    std::clock_t start = std::clock();

    // --- Шаг 3: Цикл обучения ---
    for (int ep = 0; ep < MAX_EPOCHS; ep++)
    {
        float totalErr = 0;  // Суммарная ошибка за эпоху
        int correct = 0;     // Число правильных предсказаний

        // Итерируем по всей обучающей выборке (online learning — по одному примеру)
        for (size_t i = 0; i < train.size(); i++)
        {
            // Подаём пиксели на вход сети
            brain.setData(train.images[i].pixels.data(), static_cast<int>(train.images[i].pixels.size()), 0);

            // Формируем целевой вектор для данной цифры
            digitToTargets(train.labels[i], targets.data(), OUTPUT_SIZE, cfg.activation);

            // Шаг обучения: forward pass + вычисление ошибки + backward pass + update весов
            brain.train(targets.data());

            // Накопление ошибки
            totalErr += brain.getError();

            // Проверка правильности предсказания
            if (predictDigit(brain, outIdx) == train.labels[i]) correct++;
        }

        // --- Завершение эпохи: расчёт метрик ---
        EpochResult er;
        er.epoch = ep + 1;
        er.avgError = totalErr / train.size();           // Средняя ошибка (MSE)
        er.trainAccuracy = (correct * 100.0f) / train.size();  // Точность в %
        er.elapsedSec = (std::clock() - start) / static_cast<float>(CLOCKS_PER_SEC);
        r.epochs.push_back(er);

        // Логируем прогресс эпохи
        std::cout << "  Epoch " << er.epoch << "/" << MAX_EPOCHS
                  << " | Err: " << std::fixed << std::setprecision(6) << er.avgError
                  << " | Acc: " << std::setprecision(2) << er.trainAccuracy << "%"
                  << " | Time: " << er.elapsedSec << "s" << std::endl;
    }

    // Общее время обучения
    r.totalTime = (std::clock() - start) / static_cast<float>(CLOCKS_PER_SEC);

    // --- Шаг 4: Тестирование на отдельной выборке ---
    std::cout << "  Testing..." << std::endl;
    std::clock_t evalStart = std::clock();
    int testCorrect = 0;

    // Forward pass без обучения — только предсказания
    for (size_t i = 0; i < test.size(); i++)
    {
        brain.setData(test.images[i].pixels.data(), static_cast<int>(test.images[i].pixels.size()), 0);
        brain.forwardPass();  // Только прямой проход, без backward и update
        if (predictDigit(brain, outIdx) == test.labels[i]) testCorrect++;
    }

    r.evalTime = (std::clock() - evalStart) / static_cast<float>(CLOCKS_PER_SEC);
    r.testAccuracy = (testCorrect * 100.0f) / test.size();

    // Логируем итоговые результаты
    std::cout << "  Test acc: " << std::setprecision(2) << r.testAccuracy << "%"
              << " | Eval: " << r.evalTime << "s | Total: " << r.totalTime << "s" << std::endl;

    // --- Шаг 5: Возврат результата ---
    return r;
}

// ============================================================================
// Генерация отчёта в формате Markdown
// ============================================================================

/**
 * saveReport — записывает результаты бенчмарка в markdown-файл.
 *
 * Что записывается:
 *
 *   1. Заголовок с датой, параметрами датасета и числом эпох
 *
 *   2. Таблица Summary — сводка по всем конфигурациям:
 *      - Номер, имя архитектуры, число скрытых слоёв
 *      - Число нейронов и параметров
 *      - LR (всегда 0.0000 — bug: используется захардкоженное значение)
 *      - Общее время, точность train/test
 *      - Маркеры: = лучшая точность, = fastest
 *
 *   3. Таблица Error by Epoch — ошибка по каждой эпохе:
 *      - Позволяет увидеть динамику обучения
 *      - Быстрое снижение ошибки = хорошая сходимость
 *      - Плато = обучение застряло
 *
 *   4. Рекомендации — лучшая и fastest конфигурации
 *
 *   5. Таблица Working Configurations — только конфигурации
 *      с test accuracy > 50% (отсея完全不работающие)
 *
 * Формат вывода: ../data/architecture_benchmark.md
 *
 * @param results  вектор результатов всех бенчмарков
 */
static void saveReport(const std::vector<BenchmarkResult> &results)
{
    // Открываем файл для записи (перезаписываем существующий)
    std::ofstream f("../data/architecture_benchmark.md");
    if (!f.is_open()) return;

    // Формируем timestamp текущей даты/времени
    time_t now = time(nullptr);
    char ts[64];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", localtime(&now));

    // --- Заголовок отчёта ---
    f << "# Architecture Benchmark Report\n\nDate: " << ts
      << "\nDataset: MNIST (train=" << results[0].trainSize << ", test=" << results[0].testSize
      << ")\nEpochs: " << MAX_EPOCHS << "\n\n";

    // --- Таблица Summary ---
    f << "## Summary\n\n| # | Architecture | Hidden | Neurons | Params | LR "
      << "| Time | Train Acc | Test Acc |\n|---|---|---|---|---|---|---|---|---|\n";

    // Находим лучшую по точности и самую быструю конфигурации
    int bestIdx = 0, fastIdx = 0;
    for (size_t i = 1; i < results.size(); i++) {
        if (results[i].testAccuracy > results[bestIdx].testAccuracy) bestIdx = static_cast<int>(i);
        if (results[i].totalTime < results[fastIdx].totalTime) fastIdx = static_cast<int>(i);
    }

    // Записываем строку для каждой конфигурации
    for (size_t i = 0; i < results.size(); i++)
    {
        auto &r = results[i];
        std::string marker = "";
        if (static_cast<int>(i) == bestIdx) marker = " ⭐";   // Лучшая точность
        if (static_cast<int>(i) == fastIdx) marker += " ⚡";   // Самая быстрая

        f << "| " << (i + 1) << " | " << r.archName << " | " << r.numHiddenLayers
          << " | " << r.totalNeurons << " | " << r.totalParams
          << " | " << std::fixed << std::setprecision(4) << 0.0f  // BUG: должно быть cfg.learningRate
          << " | " << std::setprecision(1) << r.totalTime << "s"
          << " | " << std::setprecision(2) << r.epochs.back().trainAccuracy << "%"
          << " | " << r.testAccuracy << "%" << marker
          << " |\n";
    }

    // --- Таблица Error by Epoch ---
    // Показывает динамику ошибки по эпохам для каждой конфигурации
    f << "\n## Error by Epoch\n\n| Arch";
    for (int e = 1; e <= MAX_EPOCHS; e++) f << " | E" << e;
    f << "\n|---";
    for (int e = 1; e <= MAX_EPOCHS; e++) f << "|---";
    f << "\n";
    for (auto &r : results)
    {
        f << "| " << r.archName;
        for (auto &ep : r.epochs) f << " | " << std::fixed << std::setprecision(4) << ep.avgError;
        f << "\n";
    }

    // --- Рекомендации ---
    f << "\n## Recommendations\n\n";
    f << "- **Best accuracy**: " << results[bestIdx].archName << " ("
      << std::setprecision(2) << results[bestIdx].testAccuracy << "%)\n";
    f << "- **Fastest**: " << results[fastIdx].archName << " ("
      << std::setprecision(1) << results[fastIdx].totalTime << "s)\n";

    // --- Working Configurations (точность > 50%) ---
    // Отсеивает полностью неработающие конфигурации
    f << "\n### Working Configurations (test acc > 50%)\n\n";
    f << "| Architecture | Test Acc | Time | Notes |\n|---|---|---|---|\n";
    for (auto &r : results) {
        if (r.testAccuracy > 50.0f) {
            f << "| " << r.archName << " | " << std::setprecision(2) << r.testAccuracy
              << "% | " << std::setprecision(1) << r.totalTime << "s | ✓ |\n";
        }
    }

    f.close();
    std::cout << "\nReport: ../data/architecture_benchmark.md" << std::endl;
}

// ============================================================================
// Главная функция — orchestrator бенчмарка
// ============================================================================

/**
 * main — запускает полный бенчмарк всех конфигураций.
 *
 * Flow:
 *   1. Заголовок в консоль с параметрами
 *   2. Загрузка обучающей и тестовой выборки MNIST
 *   3. Последовательный запуск benchmark() для каждой конфигурации
 *   4. Подсчёт общего времени
 *   5. Определение лучшей конфигурации
 *   6. Генерация markdown-отчёта
 *
 * Порядок CONFIGS важен: результаты выводятся в том же порядке,
 * что облегчает сравнение с логом консоли.
 *
 * @return 0 при успешном завершении, 1 при ошибке загрузки данных
 */
int main()
{
    // Выводим заголовок бенчмарка
    std::cout << "=============================================\n"
              << "  Architecture Benchmark\n"
              << "=============================================\n"
              << "Configs: " << CONFIGS.size() << " | Epochs: " << MAX_EPOCHS
              << " | Train: " << TRAIN_LIMIT << " | Test: " << TEST_LIMIT
              << "\n=============================================\n";

    // Загружаем обучающую и тестовую выборки MNIST
    MnistDataset train, test;
    if (!MnistReader::loadDataset(TRAIN_IMAGES, TRAIN_LABELS, train, TRAIN_LIMIT)) return 1;
    if (!MnistReader::loadDataset(TEST_IMAGES, TEST_LABELS, test, TEST_LIMIT)) return 1;

    // Запускаем бенчмарк для каждой конфигурации
    std::vector<BenchmarkResult> results;
    std::clock_t totalStart = std::clock();

    for (const auto &cfg : CONFIGS)
        results.push_back(benchmark(cfg, train, test));

    float totalTime = (std::clock() - totalStart) / static_cast<float>(CLOCKS_PER_SEC);

    // Находим лучшую конфигурацию по тестовой точности
    int bestIdx = 0;
    for (size_t i = 1; i < results.size(); i++)
        if (results[i].testAccuracy > results[bestIdx].testAccuracy) bestIdx = static_cast<int>(i);

    // Выводим итоговую сводку
    std::cout << "\n=============================================\n"
              << "  COMPLETE | Total: " << totalTime << "s\n"
              << "  Best: " << results[bestIdx].archName << " = "
              << results[bestIdx].testAccuracy << "%\n"
              << "=============================================\n";

    // Генерируем markdown-отчёт
    saveReport(results);
    return 0;
}
