#include "SGD.hpp"

Sthocastic_Gradient_Descent::Sthocastic_Gradient_Descent() : utils_cpp()
{
}

void Sthocastic_Gradient_Descent::update_parameter(
    DENSE_LAYER_NN &dense_layer,
    float &learning_rate)
{
    std::vector<std::vector<float>> &nilai_bobot = dense_layer.mendapatkan_bobot();

    // std::cout << "DEBUG PADA NILAI BOBOT FORWARD: ";
    // utils_cpp.print_vektor_2d(nilai_bobot);

    std::vector<float> &nilai_bias = dense_layer.mendapatkan_bias();

    // std::cout << "DEBUG PADA NILAI BIAS FORWARD: "
    //          << nilai_bias << "\n";

    const std::vector<float> &nilai_bobot_grad =
        dense_layer.mendapatkan_nilai_bobot_grad();

    // std::cout << "DEBUG PADA NILAI BOBOT GRAD: ";
    // utils_cpp.print_vektor_1d_float_const(nilai_bobot_grad);

    const float &nilai_bias_grad =
        dense_layer.mendapatkan_nilai_bias_grad();

    // std::cout << "DEBUG PADA NILAI BIAS GRAD: "
    // << nilai_bias_grad << "\n";

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

    for (size_t i = 0;
         i < nilai_bias.size(); ++i)
    {
        nilai_bias[i] -= learning_rate * nilai_bias_grad;
    }
}