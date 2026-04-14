# Simple_NeuronNet_C++

Простая реализация нейронной сети прямого распространения на C++17 (только STL).

![githab](https://raw.githubusercontent.com/Zolibot/Interview_of_a_real_fighter/main/bender.gif)

[![License](https://img.shields.io/github/license/Zolibot/Simple_NueronNet_C-)](LICENSE.md)
[![Powered by](https://img.shields.io/badge/Powered%20by-C%2B%2B17-green)](https://isocpp.org/)

---

## Оглавление

- [Описание](#описание)
- [Быстрый старт](#быстрый-старт)
- [Архитектура](#архитектура)
- [Обучение на MNIST](#обучение-на-mnist)
- [Результаты benchmark](#результаты-benchmark)
- [Оптимизации производительности](docs/PERFORMANCE.md)
- [API Reference](#api-reference)
- [Тесты](#тесты)
- [Сборка](#сборка)
- [Автор](#автор)

---

## Описание

Нейронная сеть с полным связыванием (fully connected), обучаемая алгоритмом обратного
распространения ошибки (backpropagation). Реализована только с использованием стандартной
библиотеки C++ — без NumPy, TensorFlow или других ML-фреймворков.

**Возможности:**
- Произвольное количество слоёв и нейронов
- Функция активации: сигмоида
- Инициализация весов: uniform, Xavier, He
- Сохранение/загрузка состояния сети
- Чтение датасета MNIST (IDX формат)
- 37 unit-тестов (Catch2)

**Ограничения:**
- Только сигмоида (ReLU/Tanh не реализованы)
- Один семпл за раз (без mini-batch)
- Нет регуляризации (dropout, L2)
- Оптимизирована для сетей до 3 скрытых слоёв

---

## Быстрый старт

```bash
# Клонировать репозиторий
git clone https://github.com/Zolibot/Simple_NueronNet_C-.git
cd Simple_NueronNet_C-

# Собрать
cmake -S . -B build/
cd build/
make -j$(nproc)

# Запустить обучение на MNIST
LD_LIBRARY_PATH=./neural_net_lib ./mnist_train

# Запустить unit-тесты
LD_LIBRARY_PATH=./neural_net_lib ./tests/tests
```

---

## Архитектура

### Структура сети

```
Входной слой → Скрытый слой(и) → Выходной слой
    (784)          (30)             (10)
```

- **Входной слой:** 784 нейрона (28×28 пикселей MNIST)
- **Скрытый слой:** настраиваемое количество нейронов (по умолчанию 30)
- **Выходной слой:** 10 нейронов (цифры 0–9)
- **Bias-нейрон:** последний нейрон каждого слоя (кроме выходного), значение = 1.0

### Структура данных

```cpp
struct Neuron {
    float output;  // Результат после сигмоиды
    float error;   // Ошибка для backpropagation
};

using Layer = std::vector<Neuron>;
using WeightMatrix = std::vector<std::vector<float>>;  // [выход][вход]
```

### Алгоритм обучения

1. **Forward pass:** вход → скрытый → выход (сигмоида на каждом нейроне)
2. **Вычисление ошибки выхода:** `error[i] = target[i] - output[i]`
3. **Обратное распространение ошибки:** от выхода к входу через веса
4. **Обновление весов:** `weight += lr × error × derivative × input`

```
for each epoch:
    for each sample:
        forward_pass()
        compute_output_error()
        backpropagate_errors()
        update_weights()
```

---

## Обучение на MNIST

### Подготовка данных

Файлы MNIST (формат IDX) должны быть в директории `data/`:

```
data/
├── train-images-idx3-ubyte   # 60,000 изображений
├── train-labels-idx1-ubyte   # 60,000 лейблов
├── t10k-images-idx3-ubyte    # 10,000 тестовых изображений
└── t10k-labels-idx1-ubyte    # 10,000 тестовых лейблов
```

Скачать можно с [официального сайта MNIST](http://yann.lecun.com/exdb/mnist/) или:

```bash
cd data/
curl -sL "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz" | gunzip > train-images-idx3-ubyte
curl -sL "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz" | gunzip > train-labels-idx1-ubyte
curl -sL "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz" | gunzip > t10k-images-idx3-ubyte
curl -sL "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz" | gunzip > t10k-labels-idx1-ubyte
```

### Запуск обучения

```bash
cd build
LD_LIBRARY_PATH=./neural_net_lib ./mnist_train
```

**Параметры по умолчанию:**
| Параметр | Значение |
|----------|----------|
| Архитектура | 784 → 30 → 10 |
| Скорость обучения | 0.5 |
| Инициализация весов | He |
| Сэмплов для обучения | 5,000 |
| Эпох | 5 |
| Функция активации | Сигмоида |

### Настройка параметров

Измените константы в `src/mnist_train.cpp`:

```cpp
static const int HIDDEN_SIZE = 64;        // Больше нейронов → выше точность
static const float LEARNING_RATE = 0.5f;  // 0.1–1.0
static const int MAX_EPOCHS = 10;         // Больше эпох → лучше, но медленнее
static const int TRAIN_LIMIT = 10000;     // Больше данных → лучше обобщение
```

---

## Результаты benchmark

### Точность по архитектурам (5000 train, 5 эпох)

| Архитектура | Время | Train Acc | Test Acc | Рекомендация |
|-------------|-------|-----------|----------|-------------|
| 784→16→10 | 6с | 87.8% | 81.0% | Быстрый тест |
| **784→30→10** | **8с** | **91.9%** | **88.7%** | **Баланс ⭐** |
| 784→64→10 | 14с | 93.8% | 88.1% | Больше точности |
| 784→128→10 | 28с | 94.8% | 88.7% | Максимум |
| 784→16→16→10 | 8с | 71.0% | 67.2% | ❌ Не рекомендуется |
| 784→30×3→10 | 10с | 19.0% | 14.3% | ❌ Затухание градиента |

### Влияние оптимизаций

| Оптимизация | Время | Точность | Статус |
|-------------|-------|----------|--------|
| Без оптимизаций (baseline) | 40с | 89.1% | — |
| + He инициализация | **8с** | 88.7% | ✅ |
| Flat weight storage | 4с | 9.8% ❌ | ❌ Откачено |

### Ошибка по эпохам (784→30→10)

| Эпоха | Ошибка | Train Acc |
|-------|--------|-----------|
| 1 | 0.197 | 82.5% |
| 2 | 0.122 | 89.4% |
| 3 | 0.106 | 90.6% |
| 4 | 0.095 | 91.7% |
| 5 | 0.090 | 91.9% |

---

## API Reference

### NeuralNet (ядро)

```cpp
class NeuralNet {
public:
    WeightInit defaultInit = INIT_HE;  // Тип инициализации

    // Активация
    static float sigmoid(float x);
    static float sigmoidDerivative(float output);

    // Прямое распространение
    void forwardPass(const Layer &in, const WeightMatrix &weights, Layer &out, int skipBias = -1);

    // Обратное распространение
    void backwardPass(const Layer &in, WeightMatrix &weights, const Layer &out, float lr);

    // Вычисление ошибок
    void computeError(const Layer &in, const WeightMatrix &weights, const Layer &out, Layer &inError);

    // Инициализация весов
    void randomizeWeights(WeightMatrix &we, WeightInit init = INIT_HE);
};
```

### NeuralNetController (контроллер)

```cpp
class NeuralNetController {
public:
    NeuralNetController(float learningRate);

    // Настройка
    void setWeightInit(WeightInit init);  // INIT_UNIFORM, INIT_XAVIER, INIT_HE
    void addLayer(int neurons);

    // Инициализация
    void initialize();  // bias + weights + randomize

    // Обучение
    void train(const float targets[]);  // targets[10] — one-hot-like [0.1, 0.1, 0.9, ...]

    // Прогноз
    void forwardPass();
    int predict() const;  // argmax output layer
    float getError() const;

    // Сохранение/загрузка
    std::vector<std::string> saveState() const;
    void loadState(const std::vector<std::string> &state);
};
```

### MnistReader

```cpp
struct MnistDataset {
    std::vector<MnistImage> images;  // pixels: vector<float> [0, 1]
    std::vector<uint8_t> labels;      // 0–9
};

class MnistReader {
public:
    static bool loadDataset(const string &images, const string &labels,
                            MnistDataset &data, size_t maxSamples = 0);
    static void printSummary(const MnistDataset &data);
};
```

---

## Тесты

```bash
cd build
LD_LIBRARY_PATH=./neural_net_lib ./tests/tests
```

**Результат:**
```
All tests passed (2518 assertions in 27 test cases)
```

### Покрытие тестами

| Компонент | Тестов | Assert |
|-----------|--------|--------|
| NeuralNet | 10 | 518 |
| NeuralNetController | 17 | 2000 |
| MnistReader | 10 | 215000+ |

---

## Сборка

### Требования
- **C++17** компилятор (clang++, g++)
- **CMake** 3.20.5+
- **STL** только — никаких внешних зависимостей

### Команды

```bash
# Стандартная сборка
cmake -S . -B build/
cd build/
make -j$(nproc)

# С конкретным компилятором
CXX=/usr/bin/clang++ cmake -S . -B build/
cd build/
make -j$(nproc)
```

### Цели сборки

| Цель | Описание |
|------|----------|
| `main` | Демо на простых данных (6→10→4) |
| `mnist_train` | Обучение на MNIST |
| `architecture_benchmark` | Сравнение 8 архитектур |
| `tests` | Unit-тесты |

---

## Оптимизации производительности

Подробная документация: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)

### Кратко

| Оптимизация | Ускорение | Влияние на точность |
|-------------|-----------|-------------------|
| He/Xavier инициализация | **5x** | +2% (epoch 1) |
| Neuron struct вместо vector | marginal | 0% |
| Один RNG на объект | 20x (инициализация) | 0% |
| Const-correctness | marginal | 0% |

---

## Автор

- [Александр Андреевич (Zolibot)](https://github.com/Zolibot) — original developer
- Рефакторинг и оптимизации — Qwen Code

## Лицензия

[MIT License](LICENSE.md)
