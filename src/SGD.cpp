#include "SGD.hpp"

Sthocastic_Gradient_Descent::Sthocastic_Gradient_Descent() : nilai_bobot_grad(),
                                                             nilai_bias_grad()
{
}

std::vector<float> Sthocastic_Gradient_Descent::update_parameter(
    std::vector<float> &nilai_bobot,
    float &nilai_bias,
    float &learning_rate)
{
    for (int i = 0;
         i < nilai_bobot.size();
         ++i)
    {
        nilai_bobot[i] - (learning_rate * nilai_bobot_grad[i]);
    }

    float perhitungan_bias_baru = 0.0f;
    perhitungan_bias_baru = nilai_bias - (learning_rate * nilai_bias_grad);

    std::vector<float> menghasilkan_nilai;
}