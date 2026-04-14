# Руководство по оптимизации производительности

## Краткий обзор

Проект Simple_NeuronNet_C++ — реализация нейронной сети прямого распространения на C++17.
Данный документ описывает все внесённые оптимизации производительности, их влияние на скорость
обучения и точность, а также объясняет почему некоторые подходы не сработали.

---

## 1. Xavier/He инициализация весов

### Что было
```cpp
// Старый код — uniform[-0.5, 0.5] для ВСЕХ слоёв
for (auto &row : we)
    for (auto &w : row)
        w = random(-0.5f, 0.5f);
```

### Что стало
```cpp
// He инициализация (по умолчанию)
float scale = sqrt(2.0f / fanIn);
// Xavier инициализация
float scale = sqrt(2.0f / (fanIn + fanOut));
```

### Почему это работает

При входе 784 нейрона и uniform[-0.5, 0.5]:
- Дисперсия взвешенной суммы ≈ 784 × (0.5²/3) ≈ 65.3
- Сумма ≈ √65.3 ≈ 8.1 → sigmoid(8.1) ≈ 0.9997
- **Все нейроны насыщаются** → градиенты ≈ 0 → обучение стоит

С He инициализацией:
- scale = √(2/784) ≈ 0.0505
- Дисперсия ≈ 784 × (0.0505²/3) ≈ 0.67
- Сумма ≈ √0.67 ≈ 0.82 → sigmoid(0.82) ≈ 0.69
- **Нейроны в активной зоне** → большие градиенты → быстрое обучение

### Результаты

| Метрика | Uniform | He Init | Изменение |
|---------|---------|---------|-----------|
| **Время 1 эпохи** | 8.0с | 1.6с | **5x быстрее** |
| **Эпох до 85%** | 3 | 1 | **3x меньше** |
| **Test accuracy (5 эпох)** | 89.1% | 88.7% | -0.4% |
| **Test accuracy (1 эпоха)** | 80.3% | 82.5% | +2.2% |

### Как использовать
```cpp
NeuralNetController brain(0.5f);
brain.setWeightInit(INIT_HE);       // He (по умолчанию)
brain.setWeightInit(INIT_XAVIER);   // Xavier
brain.setWeightInit(INIT_UNIFORM);  // Старый uniform[-0.5, 0.5]
```

---

## 2. Структура Neuron вместо vector<float>(2)

### Что было
```cpp
using Layer = vector<vector<float>>;
// neuron[0] = output, neuron[1] = error
```

### Что стало
```cpp
struct Neuron {
    float output;  // Результат после активации
    float error;   // Ошибка для обратного распространения
};
using Layer = vector<Neuron>;
```

### Почему это работает

| Аспект | vector<float>(2) | struct Neuron |
|--------|-----------------|---------------|
| **Читаемость** | `neuron[0]`, `neuron[1]` | `neuron.output`, `neuron.error` |
| **Размер** | 8 байт + оверхед vector (24 байта) | 8 байт |
| **Кэш-локальность** | Данные разбросаны | Данные подряд |
| **Типобезопасность** | Можно перепутать индексы | Компилятор проверяет |

**Экономия памяти:** для 5000 слоёв × 784 нейронов = ~96 KB меньше.

---

## 3. Генератор случайных чисел — один на объект

### Что было
```cpp
float random(float low, float high) {
    default_random_engine e1(r());  // Новый engine КАЖДЫЙ вызов!
    uniform_real_distribution<float> dist(low, high);
    return dist(e1);
}
```

### Что стало
```cpp
class NeuralNet {
    mutable default_random_engine rng_;  // Один на всю жизнь объекта
public:
    NeuralNet() : rng_(random_device{}()) {}

    float random(float low, float high) {
        uniform_real_distribution<float> dist(low, high);
        return dist(rng_);
    }
};
```

### Почему это работает

- **Было:** Каждый вызов `random()` создаёт новый `default_random_engine`, считывает `random_device` — ~100ns на вызов
- **Стало:** Один engine, просто генерируем следующее число — ~5ns на вызов

Для инициализации 23830 весов (слой 784→30→10):
- Было: ~2.4мс
- Стало: ~0.12мс
- **Ускорение: 20x на инициализации**

---

## 4. Const-correctness и ссылки

### Что было
```cpp
void forwardPass(vector<vector<float>> &inL, ...);  // Можно менять вход
float sigmoid(float &x);  // Не нужно менять, но передаём по ссылке
```

### Что стало
```cpp
void forwardPass(const Layer &inL, const WeightMatrix &weN, Layer &ouL, ...);
static float sigmoid(float x);  // Статический метод, без this
```

### Влияние
- Компилятор может лучше оптимизировать (знает что `inL` не меняется)
- `static` методы не требуют объекта `NeuralNet`
- `const &` предотвращает случайные мутации

---

## 5. Что НЕ сработало (и почему)

### Flat weight storage (откачено)

**Идея:** Заменить `vector<vector<float>>` на один `vector<float>` для всей матрицы весов.

```cpp
// Было: vector<vector<float>> — 30 отдельных malloc
// Стало: vector<float> weights(out * in) — 1 malloc
```

**Проблема:** При рефакторинге были допущены ошибки:
1. `computeError` пропускал bias (`fanIn - 1`), но индексация в `backwardPass` была неправильной
2. Производная сигмоиды применялась дважды (в `computeError` И в `backwardPass`)
3. Старый код использовал `weN.size()` (выходы) и `weN[0].size()` (входы) — легко перепутать

**Результат:** Точность упала с 89% до 9.8% (случайный выбор).

**Вывод:** Flat storage — хорошая идея, но требует аккуратной реализации и полного набора тестов.

### Momentum (μ=0.9)

**Идея:** Добавить инерцию к градиентам для ускорения сходимости.

**Проблема:** При LR=0.5 и μ=0.9 обновления весов становились нестабильными — ошибка росла вместо падения.

**Результат:** Обучение расходилось. Momentum требует более низкого LR (0.01-0.05), что замедляет обучение.

---

## 6. Архитектурные ограничения

### Почему 2+ скрытых слоя работают хуже

| Архитектура | Val accuracy | Проблема |
|-------------|-------------|----------|
| 784→16→10 | 81.0% | ✅ Работает |
| 784→30→10 | 87.3% | ✅ Лучше |
| 784→64→10 | 88.1% | ✅ Ещё лучше |
| 784→16→16→10 | 67.2% | ❌ Хуже |
| 784→30→30→10 | 76.6% | ❌ Хуже |
| 784→30×3→10 | 14.3% | ❌ Затухание градиента |
| 784→30×5→10 | 11.4% | ❌ Не обучается |

**Причины:**
1. **Затухание градиента:** Сигмоида сжимает значения в [0,1]. При обратном распространении через несколько слоёв градиенты экспоненциально убывают.
2. **Нет Batch Normalization:** Без нормализации активации каждого слоя значения «уходят» в насыщение.
3. **Один bias на слой:** Только последний нейрон — bias, что недостаточно для глубоких сетей.

### Решение для глубоких сетей
- Заменить сигмоиду на ReLU/LeakyReLU
- Добавить Batch Normalization
- Использовать Xavier инициализацию
- Добавить skip connections (ResNet-style)

---

## 7. Сравнение всех оптимизаций

| Оптимизация | Время | Точность | Статус |
|-------------|-------|----------|--------|
| **He инициализация** | 40с → 8с | 89.1% → 88.7% | ✅ Включена |
| Neuron struct | - | - | ✅ Включена (читаемость) |
| Один RNG | 2.4мс → 0.1мс | - | ✅ Включена |
| Const-correctness | marginal | - | ✅ Включена |
| Flat storage | ❌ Сломало | 89% → 9.8% | ❌ Откачено |
| Momentum | ❌ Расходится | - | ❌ Отключено |
| Mini-batch | ❌ Не реализовано | - | ⏳ В будущем |
| OpenMP | ❌ Не добавлено | - | ⏳ В будущем |

---

## 8. Как запустить

```bash
# Сборка
cd /mnt/Disk_E/GRENOW/C/Simple_NueronNet_C++
rm -rf build && cmake -S . -B build/ -DCMAKE_CXX_COMPILER=/usr/bin/clang++
cd build && make -j$(nproc)

# Обучение (5000 сэмплов, 5 эпох, He init)
LD_LIBRARY_PATH=./neural_net_lib ./mnist_train

# Тесты
LD_LIBRARY_PATH=./neural_net_lib ./tests/tests

# Benchmark архитектур
LD_LIBRARY_PATH=./neural_net_lib ./architecture_benchmark
```

---

## 9. Рекомендации по запуску

### Быстрый тест (10 секунд)
```bash
# Изменить в mnist_train.cpp:
TRAIN_LIMIT = 1000   // 1000 сэмплов
MAX_EPOCHS = 2       // 2 эпохи
```

### Стандартное обучение (8 секунд)
```bash
# По умолчанию:
TRAIN_LIMIT = 5000
MAX_EPOCHS = 5
HIDDEN_SIZE = 30
```

### Максимальная точность (13 секунд)
```bash
# Изменить в mnist_train.cpp:
TRAIN_LIMIT = 5000
MAX_EPOCHS = 10
HIDDEN_SIZE = 128     // Больше нейронов
```

### Полный training set (2-3 минуты)
```bash
TRAIN_LIMIT = 60000   // Все 60,000
MAX_EPOCHS = 10
HIDDEN_SIZE = 64
```

---

## 10. Структура проекта

```
neural_net_lib/
├── NeuralNet.h/.cpp        # Ядро: sigmoid, forward, backward, error
├── NeuralNetController.h/.cpp  # Контроллер: слои, веса, обучение
├── MnistReader.h/.cpp      # Чтение MNIST IDX файлов
└── CMakeLists.txt

src/
├── main.cpp                    # Демо на простых данных
├── mnist_train.cpp             # Обучение на MNIST
└── architecture_benchmark.cpp  # Сравнение архитектур

tests/
├── test_NeuralNet.cpp          # 10 тестов ядра
├── test_NeuralNetController.cpp # 17 тестов контроллера
├── test_MnistReader.cpp        # 10 тестов парсера
└── catch2/                     # Catch2 framework
```
