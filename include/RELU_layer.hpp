#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP

#include <cmath>
#include <vector>
#include "Sigmoid_function.hpp"

class RELU_LAYER_AKTIVASI
{
private:
    std::vector<float> x_data;

    std::vector<float> x_data_grad;

    float hasil_perhitungan;

    std::vector<float> y_data_grad;

    AKTIVASI_SIGMOID aktivasi_sigmoid;

public:
    RELU_LAYER_AKTIVASI();

    std::vector<float> RELU_FORWARD(
        std::vector<float> &x_data);

    std::vector<float> RELU_BACKWARD(
        std::vector<float> &x_data_grad);

    const std::vector<float> &mendapatkan_x_grad()
    {
        return x_data_grad;
    }
};

#endif