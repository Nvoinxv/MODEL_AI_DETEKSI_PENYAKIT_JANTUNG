#include "RELU_layer.hpp"

RELU_LAYER_AKTIVASI::RELU_LAYER_AKTIVASI() : x_data(),
                                             hasil_perhitungan(0.0f),
                                             y_data_grad(),
                                             hasil_perhitungan_grad()
{
}

std::vector<float> RELU_LAYER_AKTIVASI::RELU_FORWARD(
    std::vector<float> &x_data)
{
    this->x_data = x_data;

    std::vector<float> hasil_forward_relu;

    size_t n = x_data.size();

    for (size_t i = 0; i < n; ++i)
    {
        if (x_data[i] > 0.0f)
        {
            hasil_perhitungan = x_data[i];
        }

        else
        {
            hasil_perhitungan = 0.0f;
        }

        hasil_forward_relu.push_back(hasil_perhitungan);
    }

    return hasil_forward_relu;
}

std::vector<float> RELU_LAYER_AKTIVASI::RELU_BACKWARD(
    std::vector<float> &x_data_grad)
{
    hasil_perhitungan_grad.clear();

    size_t n = this->x_data.size();

    if (x_data_grad.size() != n)
    {
        return hasil_perhitungan_grad;
    }

    for (size_t i = 0; i < n; ++i)
    {
        if (this->x_data[i] > 0.0f)
        {
            hasil_perhitungan = x_data_grad[i];
        }

        else
        {
            hasil_perhitungan = 0.0f;
        }

        hasil_perhitungan_grad.push_back(hasil_perhitungan);
    }

    return hasil_perhitungan_grad;
}