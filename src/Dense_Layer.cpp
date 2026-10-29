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

    (void)neuorn_total;

    if (x_data.empty() || x_data[0].empty())
    {
        return hasil_perhitungan_dense_forward;
    }

    n_input = x_data[0].size();
    n_output = 1;

    for (size_t i = 0; i < x_data.size(); ++i)
    {
        if (x_data[i].size() != n_input)
        {
            return hasil_perhitungan_dense_forward;
        }
    }

    if (a.bobot.size() != n_output ||
        (a.bobot.size() == n_output &&
         a.bobot[0].size() != n_input))
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

    hasil_perhitungan_dense_forward.reserve(x_data.size());

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
    size_t jumlah_output = a.bobot[0].size();

    if (x_data_grad.size() != jumlah_data * jumlah_output)
    {
        throw std::invalid_argument(
            "Ukuran gradien output tidak sesuai dengan output dense.");
    }

    if (a.bobot.size() != jumlah_input)
    {
        throw std::invalid_argument(
            "Ukuran bobot tidak sesuai dengan jumlah input.");
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
        if (baris.size() != jumlah_output)
        {
            throw std::invalid_argument(
                "Ukuran setiap baris bobot harus sama.");
        }
    }

    bobot_grad.assign(jumlah_input * jumlah_output, 0.0f);
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
                bobot_grad[k * jumlah_output + j] +=
                    x_data[i][k] * grad_output;

                grad_input[i][k] +=
                    grad_output * a.bobot[k][j];
            }
        }
    }

    return grad_input;
}
