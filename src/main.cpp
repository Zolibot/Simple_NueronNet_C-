#include <iostream>
#include <vector>

#include "NeuralNetController.h"

using namespace std;

// Демонстрация работы нейронной сети
int main()
{
    cout << "Neural Network Demo" << endl;
    cout << "===================" << endl << endl;

    // Создаём сеть: 6 входов -> 10 скрытых -> 4 выхода
    NeuralNetController brain(0.1f);
    brain.addLayer(6);
    brain.addLayer(10);
    brain.addLayer(4);
    brain.initialize();

    cout << "Network: 6 -> 10 -> 4" << endl;
    cout << "Layers: " << brain.layerCount() << endl;
    cout << endl;

    // Обучаем на простом паттерне
    float input[] = {0, 0.3f, 1, 0.1f, 1, 0};
    float target[] = {1, 0, 1, 0};

    cout << "Training..." << endl;
    int iterations = 0;
    clock_t start = clock();

    do
    {
        brain.setData(input, 6, 0);
        brain.train(target);

        if (iterations % 10000 == 0)
            cout << "  Iteration: " << iterations << ", Error: " << brain.getError() << endl;

        iterations++;
    } while (brain.getError() > 0.0016f && iterations < 500000);

    clock_t end = clock();

    cout << endl;
    cout << "Results after " << iterations << " iterations:" << endl;
    cout << "  Error: " << brain.getError() << endl;
    cout << "  Time: " << (end - start) / static_cast<double>(CLOCKS_PER_SEC) << "s" << endl;
    cout << endl;

    // Вывод результатов
    cout << "Input: ";
    for (int i = 0; i < 6; i++) cout << input[i] << " ";
    cout << endl;

    cout << "Target: ";
    for (int i = 0; i < 4; i++) cout << target[i] << " ";
    cout << endl;

    cout << "Output: ";
    const auto &out = brain.getLayer(2);
    for (size_t i = 0; i < out.size(); i++)
        cout << out[i].output << " ";
    cout << endl;

    // Сохраняем и загружаем
    auto saved = brain.saveState();
    cout << endl << "Saved state: " << saved.size() << " lines" << endl;

    NeuralNetController brain2(saved, 0.1f);
    cout << "Loaded network layers: " << brain2.layerCount() << endl;

    cout << endl << "Done. Press Enter to exit..." << endl;
    int t = 0;
    cin >> t;
    return 0;
}
