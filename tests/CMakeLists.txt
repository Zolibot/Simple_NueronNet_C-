cmake_minimum_required(VERSION 3.20.5)

add_executable(tests
    test_main.cpp
    test_NeuralNet.cpp
    test_NeuralNetController.cpp
    test_MnistReader.cpp
    catch2/catch_amalgamated.cpp
)

target_include_directories(tests PRIVATE
    ${CMAKE_SOURCE_DIR}/neural_net_lib
    ${CMAKE_CURRENT_SOURCE_DIR}/catch2
)

target_link_libraries(tests PRIVATE SimpleNeuralNetLib)

add_test(NAME NeuralNetTests COMMAND tests)
