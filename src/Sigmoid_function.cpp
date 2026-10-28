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
    int neuron_total,
    std::vector<float> &x_data)
{
    hasil_forward_sigmoid.clear();

    hasil_forward_sigmoid.reserve(x_data.size());

    for (size_t i = 0; i < x_data.size(); ++i)
    {
        hasil_perhitungan_sigmoid =
            1.0f / (1.0f + std::exp(-x_data[i]));

        hasil_forward_sigmoid.push_back(
            hasil_perhitungan_sigmoid);
    }

    return hasil_forward_sigmoid;
}

std::vector<float> AKTIVASI_SIGMOID::SIGMOID_BACKWARD(
    std::vector<float> &y_pred,
    std::vector<float> &y_pred_grad)
{
    hasil_grad_sigmoid.clear();

    if (y_pred.size() != y_pred_grad.size())
    {
        return hasil_grad_sigmoid;
    }

    for (size_t i = 0; i < y_pred.size(); ++i)
    {
        hasil_perhitungan_sigmoid =
            y_pred_grad[i] *
            y_pred[i] *
            (1.0f - y_pred[i]);

        hasil_grad_sigmoid.push_back(
            hasil_perhitungan_sigmoid);
    }

    return hasil_grad_sigmoid;
}
