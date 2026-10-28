#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "RELU_layer.hpp"
#include "SPLIT_TRAINING_TEST_DATA.hpp"
#include "Sigmoid_function.hpp"
#include "Binary_Cross_Entropy.hpp"

#include <random>
#include <cmath>
#include <vector>
#include <cstddef>

struct DENSE_PARAMETER
{
    std::vector<std::vector<float>> bobot;
    float bias = 0.0f;
};

class DENSE_LAYER_NN
{
private:
    RELU_LAYER_AKTIVASI relu;

    std::vector<std::vector<float>> hasil_output_grad;

    float a_distribusi;
    float perhitungan_neuron;

    std::vector<float> bobot_grad;
    float bias_grad;

    float bias = 0.0f;

    size_t n_input;
    size_t n_output;

    float neuron_total;

    BCE_PERHITUNGAN bce_loss;

    DENSE_PARAMETER a;

    float inisialisasi_bobot(
        float &n_input,
        float &n_out)
    {
        perhitungan_neuron = 6.0f / (n_input + n_out);

        a_distribusi = std::sqrt(perhitungan_neuron);

        return a_distribusi;
    }

public:
    DENSE_LAYER_NN();

    std::vector<float> DENSE_FORWARD_LAYER(
        int neuorn_total,
        std::vector<std::vector<float>> &x_data);

    std::vector<std::vector<float>> DENSE_BACKWARD_LAYER(
        std::vector<std::vector<float>> &x_data,
        std::vector<float> &x_data_grad);

    const std::vector<float> &mendapatkan_nilai_bobot_grad() const
    {
        return bobot_grad;
    }

    const float &mendapatkan_nilai_bias_grad() const
    {
        return bias_grad;
    }

    std::vector<std::vector<float>> &mendapatkan_bobot()
    {
        return a.bobot;
    }

    float &mendapatkan_bias()
    {
        return a.bias;
    }
};

#endif
