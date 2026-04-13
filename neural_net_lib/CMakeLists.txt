cmake_minimum_required(VERSION 3.20.5)
project(SimpleNeuralNetLib)
set(SOURCE_LIB NeuralNet.cpp NeuralNetController.cpp MnistReader.cpp)
add_library(SimpleNeuralNetLib SHARED ${SOURCE_LIB})
