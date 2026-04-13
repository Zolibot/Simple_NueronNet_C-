cmake_minimum_required(VERSION 3.20.5)
project(SimpleNeuralNet)

enable_testing()

set(SOURCE_EXE src/main.cpp)
include_directories(neural_net_lib)
add_executable(main ${SOURCE_EXE})
add_subdirectory(neural_net_lib)
target_link_libraries(main SimpleNeuralNetLib)

# MNIST training executable
add_executable(mnist_train src/mnist_train.cpp)
target_link_libraries(mnist_train SimpleNeuralNetLib)

# Architecture benchmark executable
add_executable(architecture_benchmark src/architecture_benchmark.cpp)
target_link_libraries(architecture_benchmark SimpleNeuralNetLib)

add_subdirectory(tests)
