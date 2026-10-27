#include "SGD.hpp"

Sthocastic_Gradient_Descent::Sthocastic_Gradient_Descent()
{
}

void Sthocastic_Gradient_Descent::update_parameter(
    DENSE_LAYER_NN &dense_layer,
    float &learning_rate)
{
    std::vector<std::vector<float>> &nilai_bobot = dense_layer.mendapatkan_bobot();

    float &nilai_bias = dense_layer.mendapatkan_bias();

    const std::vector<float> &nilai_bobot_grad =
        dense_layer.mendapatkan_nilai_bobot_grad();

    const float &nilai_bias_grad =
        dense_layer.mendapatkan_nilai_bias_grad();

    if (nilai_bobot.empty() ||
        nilai_bobot[0].size() != nilai_bobot_grad.size())
    {
        return;
    }

    for (size_t i = 0;
         i < nilai_bobot[0].size();
         ++i)
    {
        nilai_bobot[0][i] -=
            learning_rate * nilai_bobot_grad[i];
    }

    nilai_bias -=
        learning_rate * nilai_bias_grad;
}