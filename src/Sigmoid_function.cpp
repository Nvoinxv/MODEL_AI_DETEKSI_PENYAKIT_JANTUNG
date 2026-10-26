#include "Sigmoid_function.hpp"

AKTIVASI_SIGMOID::AKTIVASI_SIGMOID() : x_data(),
                                       y_pred_grad(0.0f),
                                       y_actual_grad(0.0f),
                                       hasil_perhitungan_sigmoid(0.0f),
                                       hasil_grad_sigmoid(),
                                       EULAR(2.718281828459045235360287471352),
                                       hasil_forward_sigmoid()
{
}

std::vector<float> AKTIVASI_SIGMOID::SIGMOID_FORWARD(
    std::vector<float> &x_data)
{
    int n = x_data.size();

    for (int i = 0; i < n; ++i)
    {
        hasil_perhitungan_sigmoid = (1.0f /
                                     (1.0f + std::pow(EULAR, -x_data[i])));
    }

    hasil_forward_sigmoid.push_back(hasil_perhitungan_sigmoid);

    return hasil_forward_sigmoid;
}

std::vector<float> AKTIVASI_SIGMOID::SIGMOID_BACKWARD(
    float &y_pred_grad,
    float &y_actual_grad)
{
    hasil_perhitungan_sigmoid = y_pred_grad - y_actual_grad;

    hasil_grad_sigmoid.push_back(hasil_perhitungan_sigmoid);

    return hasil_grad_sigmoid;
}