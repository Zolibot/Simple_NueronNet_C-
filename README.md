# Simple_NeuronNet_C++

Простая реализация нейронной сети прямого распространения на C++17 (только STL).

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
- 4 функции активации: **Sigmoid, ReLU, LeakyReLU, Tanh**
- 3 оптимизатора: **SGD, Adam, Momentum**
- Инициализация весов: uniform, Xavier, He
- Gradient clipping для предотвращения NaN
- Сохранение/загрузка состояния сети
- Чтение датасета MNIST (IDX формат)
- 38 unit-тестов (Catch2)

**Ограничения:**
- Один семпл за раз (без mini-batch)
- Нет регуляризации (dropout, L2)
- Оптимизирована для сетей до 2 скрытых слоёв

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
    (784)          (128)            (10)
```

- **Входной слой:** 784 нейрона (28×28 пикселей MNIST)
- **Скрытый слой:** настраиваемое количество нейронов (по умолчанию 30)
- **Выходной слой:** 10 нейронов (цифры 0–9)
- **Bias-нейрон:** последний нейрон каждого скрытого слоя, значение = 1.0

### Структура данных

```cpp
struct Neuron {
    float output;  // Результат после функции активации
    float error;   // Ошибка для обратного распространения
};

using Layer = std::vector<Neuron>;
using WeightMatrix = std::vector<std::vector<float>>;  // [выход][вход]
```

---

## Обучение на MNIST

### Подготовка данных

Файлы MNIST (формат IDX) **не входят в репозиторий** — скачайте их отдельно:

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

**Параметры по умолчанию (в `src/mnist_train.cpp`):**

| Параметр | Значение | Описание |
|----------|----------|----------|
| Архитектура | 784 → 30 → 10 | Вход → скрытый → выход |
| Learning rate | 0.5 | Скорость обучения (SGD) |
| Инициализация | He | He/Xavier/uniform |
| Train samples | 5,000 | Ограничение для быстрого теста |
| Epochs | 5 | Количество эпох |
| Activation | Sigmoid | Sigmoid/ReLU/LeakyReLU/Tanh |

---

## Результаты benchmark

### Лучшие конфигурации (5000 train, 5 эпох)

| Архитектура | Оптимизатор | LR | Test Acc | Время | Рекомендация |
|-------------|-------------|-----|----------|-------|-------------|
| **784→128→10** | **Adam** | 0.001 | **92.60%** ⭐ | 96с | Максимальная точность |
| 784→128→10 | ReLU Adam | 0.001 | 92.26% | 105с | ReLU вариант |
| 784→64→10 | Adam | 0.001 | 91.12% | 49с | Баланс |
| 784→30→10 | SGD | 0.5 | 88.90% | 8с | Быстрый старт |
| 784→16→10 | SGD | 0.5 | 83.92% | 5с | Тест |

### Влияние оптимизаций

| Оптимизация | До | После | Изменение |
|-------------|-----|-------|-----------|
| He инициализация | 40с | 8с | **5x быстрее** |
| Gradient clipping | NaN (ReLU) | 90% | **Работает** |
| Adam timestep fix | 9.6% | 92.6% | **+83%** |

Полный benchmark: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)

---

## Тесты

```bash
cd build
LD_LIBRARY_PATH=./neural_net_lib ./tests/tests
```

**Результат:**
```
All tests passed (219543 assertions in 38 test cases)
```

---

## Сборка

### Требования
- **C++17** компилятор (clang++, g++)
- **CMake** 3.20+
- **STL** только — никаких внешних зависимостей

### Команды

```bash
cmake -S . -B build/
cd build/
make -j$(nproc)
```

### Цели сборки

| Цель | Описание |
|------|----------|
| `mnist_train` | Обучение на MNIST |
| `architecture_benchmark` | Сравнение 14 архитектур |
| `tests` | Unit-тесты |

---

## Оптимизации производительности

Подробная документация: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)

| Оптимизация | Эффект |
|-------------|--------|
| He/Xavier инициализация | **5x быстрее** сходимость |
| Gradient clipping (±5.0) | Предотвращает NaN в ReLU |
| Adam timestep fix | 9.6% → 92.6% |
| Один RNG на объект | 20x на инициализации |
| Neuron struct | Читаемость, кэш-локальность |

---

## Автор

- [Александр Андреевич (Zolibot)](https://github.com/Zolibot) — original developer
- Рефакторинг, оптимизации, документация — Qwen Code

## Лицензия

[MIT License](LICENSE.md)
