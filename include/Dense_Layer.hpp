#ifndef DENSE_LAYER_HPP
#define DENSE_LAYEER_HPP

#include "RELU_layer.hpp"
#include "SPLIT_TRAINING_TEST_DATA.hpp"
#include "Sigmoid_function.hpp"
#include "Binary_Cross_Entropy.hpp"

#include <random>
#include <cmath>

struct DENSE_PARAMETER
{
    std::vector<std::vector<float>> bobot;
    float bias = 0.0f;
};

class DENSE_LAYER_NN
{
private:
    RELU_LAYER_AKTIVASI relu;

    std::vector<float> hasil_output_grad = relu.mendapatkan_x_grad();

    float a_distribusi;
    float perhitungan_neuron;

    std::vector<float> bobot_grad;
    float bias_grad;

    float bias = 0.0f;

    float n_input;
    float n_output;

    BCE_PERHITUNGAN bce_loss;

    float y_actual_grad = bce_loss.mendapatkan_y_actual_backward();
    float y_pred_grad = bce_loss.mendapatkan_y_pred_backward();

    DENSE_PARAMETER a;

    float inisialisasi_bobot(
        float &n_input,
        float &n_out)
    {
        for (int i = 0;
             i < a.bobot.size();
             ++i)
        {

            for (int j = 0;
                 j < a.bobot[i].size();
                 ++j)
            {
                n_input = a.bobot[i][j];
            }
        }

        perhitungan_neuron = (6 / (n_input + n_out));

        a_distribusi = std::sqrt(perhitungan_neuron);

        return a_distribusi;
    }

public:
    DENSE_LAYER_NN();

    std::vector<float> DENSE_FORWARD_LAYER(
        std::vector<std::vector<float>> &x_data);

    std::vector<float> DENSE_BACKWARD_LAYER(
        std::vector<std::vector<float>> &x_data);

    const std::vector<float> &mendapatkan_nilai_bobot_grad() const
    {
        return bobot_grad;
    }

    const float &mendapatkan_nilai_bias_grad() const
    {
        return bias_grad;
    }
};

#endif