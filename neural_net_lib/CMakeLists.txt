cmake_minimum_required(VERSION 3.20.5)
project(SimpleNueronNetLib)
set(SOURCE_LIB NueralNetController.cpp NueralNetFloat.cpp)
add_library(SimpleNueronNetLib SHARED ${SOURCE_LIB})