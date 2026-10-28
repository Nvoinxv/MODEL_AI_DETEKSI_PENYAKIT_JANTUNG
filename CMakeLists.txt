cmake_minimum_required(VERSION 3.14)
project(NEURAL_NETWORK_KLASIFIKASI_DETAK_JANTUNG_CPP VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME} 
    main.cpp
    src/READ_CSV.cpp
    src/PREPROCESSING_DATA.cpp
    src/utils_program.cpp
    src/SPLIT_TRAINING_TEST_DATA.cpp
    src/Binary_Cross_Entropy.cpp
    src/Dense_Layer.cpp
    src/RELU_layer.cpp
    src/Sigmoid_function.cpp
    src/SGD.cpp
    src/Neural_Network.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)