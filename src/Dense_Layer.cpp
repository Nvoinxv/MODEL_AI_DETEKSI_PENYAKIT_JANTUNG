#include "Dense_Layer.hpp"

DENSE_LAYER_NN::DENSE_LAYER_NN() : hasil_output_grad(),
                                   a_distribusi(0.0f),
                                   n_input(0.0f),
                                   n_output(0.0f),
                                   bias(0.0f)
{
}

std::vector<float> DENSE_LAYER_NN::DENSE_FORWARD_LAYER(
    std::vector<std::vector<float>> &x_data)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(0.0f, 10.0f);

    float a = dist(gen);
    float negatif_a = -a;

    std::vector<std::vector<float>> bobot;

    std::vector<float> hasil_perhitungan_dense_forward;

    float perhitungan_dense = 0.0f;

    for (int i = 0;
         i < bobot.size();
         ++i)
    {

        for (int j = 0;
             j < bobot[i].size();
             ++j)
        {
            bobot[i][j] = inisialisasi_bobot(negatif_a, a);
        }
    }

    for (int i = 0;
         i < x_data.size();
         ++i)
    {

        for (int j = 0;
             j < x_data[i].size();
             ++j)
        {
            perhitungan_dense += (x_data[i][j] *
                                  bobot[i][j]) +
                                 bias;
        }
    }

    hasil_perhitungan_dense_forward.push_back(perhitungan_dense);

    return hasil_perhitungan_dense_forward;
}

std::vector<float> DENSE_LAYER_NN::DENSE_BACKWARD_LAYER(
    std::vector<std::vector<float>> &x_data)
{
    float perhitungan_backward = 0.0f;

    for (int i = 0;
         i < x_data.size();
         ++i)
    {
        for (int j = 0;
             j < x_data[i].size();
             ++j)
        {
            perhitungan_backward += (y_pred_grad - y_actual_grad) * x_data[i][j];
        }
    }

    bias_grad = (y_pred_grad - y_actual_grad);

    bobot_grad.push_back(perhitungan_backward);

    return bobot_grad;
}
