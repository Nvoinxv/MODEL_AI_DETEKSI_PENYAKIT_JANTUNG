#ifndef SGD_OPTIMISASI_HPP
#define SGD_OPTIMISASI_HPP

#include "Dense_Layer.hpp"

class Sthocastic_Gradient_Descent
{
private:
    DENSE_LAYER_NN dense_layer;

    std::vector<float> nilai_bobot_grad = dense_layer.mendapatkan_nilai_bobot_grad();
    float nilai_bias_grad = dense_layer.mendapatkan_nilai_bias_grad();

public:
    Sthocastic_Gradient_Descent();

    std::vector<float> update_parameter(
        std::vector<float> &nilai_bobot,
        float &nilai_bias,
        std::vector<float> &nilai_bobot_grad,
        float &nilai_bias_grad,
        float &learning_rate);
};

#endif