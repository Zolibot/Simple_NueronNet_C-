# AGENTS.md

## Build & Run

```bash
cd /path/to/repo
rm -rf build
CXX=/usr/lib/llvm-19.1/bin/clang++ cmake -S . -B build/ -DCMAKE_BUILD_TYPE=Release
cd build && make -j$(nproc)
```

All executables require `LD_LIBRARY_PATH=./neural_net_lib` at runtime because the neural net library is a shared library.

## Optimized Build

The following flags provide **~10% speedup**:
- Release mode with -O3 -march=native -flto
- Use clang++-19: `CXX=/usr/lib/llvm-19.1/bin/clang++`

## Executables

| Target | Purpose |
|--------|---------|
| `main` | Demo with simple data (6→10→4) |
| `mnist_train` | Train on MNIST |
| `architecture_benchmark` | Compare architectures |
| `tests` | Unit tests (Catch2) |

## Running

```bash
cd build
LD_LIBRARY_PATH=./neural_net_lib ./main
LD_LIBRARY_PATH=./neural_net_lib ./mnist_train
LD_LIBRARY_PATH=./neural_net_lib ./tests/tests
```

Or via CTest: `ctest --output-on-failure`

## Architecture

- Library: `neural_net_lib/` (shared library `libSimpleNeuralNetLib.so`)
- Sources: `NeuralNet.cpp`, `NeuralNetController.cpp`, `MnistReader.cpp`
- Headers: `NeuralNet.h`, `NeuralNetController.h`, `MnistReader.h`
- Tests: `tests/` with Catch2 (amalgamated, included in repo)

## Notes

- Pure C++17, no external dependencies beyond STL
- MNIST data must be in `data/` directory in IDX format
- Main architecture: 784→30→10 (input→hidden→output)
- Only sigmoid activation (no ReLU/Tanh)
- No mini-batch support (one sample at a time)