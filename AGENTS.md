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

## Testing

```bash
cd build
LD_LIBRARY_PATH=./neural_net_lib ./tests/tests
```

Result:
```
All tests passed (2518 assertions in 27 test cases)
```

## Performance

### Architecture Benchmarks (784→30→10)
| Architecture | Time | Accuracy |
|--------------|------|----------|
| 784→30→10 | 8s | 91.9% |
| 784→64→10 | 14s | 93.8% |
| 784→128→10 | 28s | 94.8% |

### Optimizations
- He/Xavier initialization: ~5x speedup, +2% accuracy (epoch 1)
- One RNG per object: 20x speedup (initialization)
- Const-correctness: marginal improvement

## Key Commands

- Build: `cmake -S . -B build/ && cd build && make -j$(nproc)`
- Run tests: `cd build && LD_LIBRARY_PATH=./neural_net_lib ./tests/tests`
- Run demo: `cd build && LD_LIBRARY_PATH=./neural_net_lib ./main`
- Run MNIST train: `cd build && LD_LIBRARY_PATH=./neural_net_lib ./mnist_train`
- Run benchmark: `cd build && LD_LIBRARY_PATH=./neural_net_lib ./architecture_benchmark`
