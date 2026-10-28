#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP

#include <vector>

class RELU_LAYER_AKTIVASI
{
private:
    std::vector<float> x_data;

    float hasil_perhitungan;

    std::vector<float> hasil_perhitungan_grad;

public:
    RELU_LAYER_AKTIVASI();

    std::vector<float> RELU_FORWARD(
        std::vector<float> &x_data);

    std::vector<float> RELU_BACKWARD(
        std::vector<float> &x_data_grad);
};

#endif