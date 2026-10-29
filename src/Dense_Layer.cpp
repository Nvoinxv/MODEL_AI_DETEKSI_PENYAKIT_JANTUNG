#include "Dense_Layer.hpp"

DENSE_LAYER_NN::DENSE_LAYER_NN() : hasil_output_grad(),
                                   a_distribusi(0.0f),
                                   perhitungan_neuron(0.0f),
                                   bobot_grad(),
                                   bias_grad(0.0f),
                                   bias(0.0f),
                                   n_input(0),
                                   n_output(1),
                                   neuron_total(0.0f),
                                   bce_loss(),
                                   a()
{
}

std::vector<float> DENSE_LAYER_NN::DENSE_FORWARD_LAYER(
    int neuorn_total,
    std::vector<std::vector<float>> &x_data)
{
    std::vector<float> hasil_perhitungan_dense_forward;

    if (x_data.empty() ||
        x_data[0].empty() ||
        neuorn_total <= 0)
    {
        return hasil_perhitungan_dense_forward;
    }

    n_input = x_data[0].size();
    n_output = static_cast<size_t>(neuorn_total);

    for (size_t i = 0; i < x_data.size(); ++i)
    {
        if (x_data[i].size() != n_input)
        {
            throw std::invalid_argument(
                "Jumlah fitur setiap sampel harus sama.");
        }
    }

    if (a.bobot.size() != n_output ||
        a.bias.size() != n_output ||
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

        a.bias.assign(n_output, 0.0f);

        for (size_t i = 0; i < n_output; ++i)
        {
            for (size_t j = 0; j < n_input; ++j)
            {
                a.bobot[i][j] = dist(gen);
            }
        }
    }

    hasil_perhitungan_dense_forward.reserve(
        x_data.size() * n_output);

    for (size_t i = 0; i < x_data.size(); ++i)
    {
        for (size_t j = 0; j < n_output; ++j)
        {
            float perhitungan_dense = 0.0f;

            for (size_t k = 0; k < n_input; ++k)
            {
                perhitungan_dense +=
                    x_data[i][k] * a.bobot[j][k];
            }

            perhitungan_dense += a.bias[j];

            hasil_perhitungan_dense_forward.push_back(
                perhitungan_dense);
        }
    }

    return hasil_perhitungan_dense_forward;
}

std::vector<std::vector<float>> DENSE_LAYER_NN::DENSE_BACKWARD_LAYER(
    std::vector<std::vector<float>> &x_data,
    std::vector<float> &x_data_grad)
{
    if (x_data.empty() ||
        x_data[0].empty() ||
        a.bobot.empty() ||
        a.bobot[0].empty())
    {
        throw std::invalid_argument(
            "Input dan bobot dense tidak boleh kosong.");
    }

    size_t jumlah_data = x_data.size();
    size_t jumlah_input = x_data[0].size();
    size_t jumlah_output = a.bobot.size();

    if (x_data_grad.size() != jumlah_data * jumlah_output)
    {
        throw std::invalid_argument(
            "Ukuran gradien output tidak sesuai dengan output dense.");
    }

    if (a.bobot.size() != jumlah_output)
    {
        throw std::invalid_argument(
            "Ukuran bobot tidak sesuai dengan jumlah neuron output.");
    }

    for (const auto &baris : x_data)
    {
        if (baris.size() != jumlah_input)
        {
            throw std::invalid_argument(
                "Jumlah fitur setiap sampel harus sama.");
        }
    }

    for (const auto &baris : a.bobot)
    {
        if (baris.size() != jumlah_input)
        {
            throw std::invalid_argument(
                "Ukuran setiap baris bobot tidak sesuai dengan jumlah input.");
        }
    }

    bobot_grad.assign(
        jumlah_input * jumlah_output,
        0.0f);

    bias_grad = 0.0f;

    std::vector<std::vector<float>> grad_input(
        jumlah_data,
        std::vector<float>(jumlah_input, 0.0f));

    for (size_t i = 0;
         i < jumlah_data;
         ++i)
    {
        for (size_t j = 0;
             j < jumlah_output;
             ++j)
        {
            float grad_output =
                x_data_grad[i * jumlah_output + j];

            bias_grad += grad_output;

            for (size_t k = 0;
                 k < jumlah_input;
                 ++k)
            {
                bobot_grad[j * jumlah_input + k] +=
                    x_data[i][k] * grad_output;

                grad_input[i][k] +=
                    grad_output * a.bobot[j][k];
            }
        }
    }

    return grad_input;
}