#include "RELU_layer.hpp"

RELU_LAYER_AKTIVASI::RELU_LAYER_AKTIVASI() : x_data(),
                                             x_data_grad(),
                                             hasil_perhitungan(0.0f),
                                             hasil_perhitungan_grad()
{
}

std::vector<float> RELU_LAYER_AKTIVASI::RELU_FORWARD(
    std::vector<float> &x_data)
{
    size_t n = x_data.size();

    for (size_t i = 0; i < n; ++i)
    {
        if (x_data[i] > 0)
        {
            hasil_perhitungan = x_data[i];
        }

        else if (x_data[i] < 0)
        {
            x_data[i] = 0;
        }

        x_data.push_back(hasil_perhitungan);
    }

    return x_data;
}

std::vector<std::vector<float>> RELU_LAYER_AKTIVASI::RELU_BACKWARD(
    std::vector<float> &x_data,
    std::vector<float> &x_data_grad)
{
    int n = x_data.size();

    for (int i = 0; i < n; ++i)
    {
        if (x_data[i] > 0)
        {
            x_data[i] = 1.0f;
        }

        else if (x_data[i] <= 0.0f)
        {
            x_data[i] = 0.0f;
        }

        hasil_perhitungan = x_data_grad[i] * x_data[i];
    }

    hasil_perhitungan_grad.push_back(hasil_perhitungan);

    return hasil_perhitungan_grad;
}