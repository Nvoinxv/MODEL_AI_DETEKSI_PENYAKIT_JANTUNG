#include "Dense_Layer.hpp"

DENSE_LAYER_NN::DENSE_LAYER_NN() : hasil_output_grad(),
                                   a_distribusi(0.0f),
                                   perhitungan_neuron(0.0f),
                                   bobot_grad(),
                                   bias_grad(0.0f),
                                   bias(0.0f),
                                   n_input(0),
                                   n_output(1),
                                   a(),
                                   neuron_total(0.0f)
{
}

std::vector<float> DENSE_LAYER_NN::DENSE_FORWARD_LAYER(
    int &neuorn_total,
    std::vector<std::vector<float>> &x_data)
{
    std::vector<float> hasil_perhitungan_dense_forward;

    if (x_data.empty() || x_data[0].empty())
    {
        return hasil_perhitungan_dense_forward;
    }

    n_input = x_data[0].size();

    for (size_t i = 0; i < x_data.size(); ++i)
    {
        if (x_data[i].size() != n_input)
        {
            return hasil_perhitungan_dense_forward;
        }
    }

    n_output = 1;

    if (a.bobot.size() != n_output ||
        a.bobot[0].size() != n_input)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        float jumlah_input = static_cast<float>(n_input);
        float jumlah_output = static_cast<float>(n_output);

        a_distribusi = inisialisasi_bobot(
            jumlah_input,
            jumlah_output);

        std::uniform_real_distribution<float> dist(
            -a_distribusi,
            a_distribusi);

        a.bobot.assign(
            n_output,
            std::vector<float>(n_input));

        for (size_t i = 0; i < n_input; ++i)
        {
            a.bobot[0][i] = dist(gen);
        }

        a.bias = 0.0f;
        bias = a.bias;
    }

    for (int i = 0; i < a.bobot.size(); ++i)
    {
        for (int j = 0; j < a.bobot[i].size(); ++j)
        {
            a.bobot[i][j] = a.bobot[i][j] * neuron_total;
        }

        a.bias = neuron_total;
    }

    for (size_t i = 0; i < x_data.size(); ++i)
    {
        float perhitungan_dense = 0.0f;

        for (size_t j = 0; j < n_input; ++j)
        {
            perhitungan_dense +=
                x_data[i][j] * a.bobot[0][j];
        }

        perhitungan_dense += a.bias;

        hasil_perhitungan_dense_forward.push_back(
            perhitungan_dense);
    }

    return hasil_perhitungan_dense_forward;
}

std::vector<std::vector<float>> DENSE_LAYER_NN::DENSE_BACKWARD_LAYER(
    std::vector<std::vector<float>> &x_data,
    std::vector<float> &x_data_grad)
{
    hasil_output_grad.clear();
    bobot_grad.clear();

    bias_grad = 0.0f;

    if (x_data.empty() ||
        x_data_grad.size() != x_data.size() ||
        a.bobot.size() != 1)
    {
        return hasil_output_grad;
    }

    n_input = x_data[0].size();

    if (n_input == 0 || a.bobot[0].size() != n_input)
    {
        return hasil_output_grad;
    }

    for (size_t i = 0; i < x_data.size(); ++i)
    {
        if (x_data[i].size() != n_input)
        {
            hasil_output_grad.clear();
            return hasil_output_grad;
        }
    }

    bobot_grad.assign(n_input, 0.0f);

    hasil_output_grad.assign(
        x_data.size(),
        std::vector<float>(n_input, 0.0f));

    for (size_t i = 0; i < x_data.size(); ++i)
    {
        for (size_t j = 0; j < n_input; ++j)
        {
            bobot_grad[j] +=
                x_data[i][j] * x_data_grad[i];

            hasil_output_grad[i][j] =
                a.bobot[0][j] * x_data_grad[i];
        }

        bias_grad += x_data_grad[i];
    }

    float jumlah_data = static_cast<float>(x_data.size());

    for (size_t j = 0; j < n_input; ++j)
    {
        bobot_grad[j] /= jumlah_data;
    }

    bias_grad /= jumlah_data;

    return hasil_output_grad;
}