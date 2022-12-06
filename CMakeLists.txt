cmake_minimum_required(VERSION 3.0)
project(openANN)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++14")

set(MNIST_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
set(MNIST_DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/data)
set(MNIST_FOUND TRUE)

add_executable(
    openANN
    src/main.cpp
    src/Neuron.cpp
    src/Matrix.cpp
    src/Layer.cpp
    src/utils/Math.cpp    
    src/neural_network/NeuralNetwork.cpp
    src/neural_network/setErrors.cpp
    src/neural_network/feedForward.cpp
    src/neural_network/backPropagation.cpp
    src/neural_network/train.cpp
)
target_compile_definitions(openANN PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")


