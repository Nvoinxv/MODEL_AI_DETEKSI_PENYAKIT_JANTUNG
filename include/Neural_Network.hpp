#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "Binary_Cross_Entropy.hpp"
#include "Dense_Layer.hpp"
#include "RELU_layer.hpp"
#include "Sigmoid_function.hpp"
#include "SPLIT_TRAINING_TEST_DATA.hpp"
#include "utils_program.hpp"
#include "SGD.hpp"

class neural_network_cpp
{
private:
    // PEMANGGILAN KELAS
    BCE_PERHITUNGAN bce_loss;
    DENSE_LAYER_NN dense_layer;
    RELU_LAYER_AKTIVASI relu_layer;
    AKTIVASI_SIGMOID sigmoid_layer;
    Sthocastic_Gradient_Descent SGD_OPTIM;

    TIPE_DATA_TRAIN variabel_train_test;
    TRAIN_TEST_DATA_X_Y train_test;

    // MEMPERSIAPKAN VARIABEL NEURON FORWARD
    std::vector<float> hasil_dense;
    std::vector<float> hasil_dense_2;
    std::vector<float> hasil_dense_3;

    std::vector<float> hasil_relu;
    std::vector<float> hasil_relu_2;
    std::vector<float> hasil_relu_3;

    std::vector<float> y_pred;

    // MEMPERSIAPKAN VARIABEL BACKWARD
    std::vector<float> grad_y_pred;
    std::vector<float> grad_sigmoid;

    std::vector<float> grad_relu;

    float hasil_loss;

    // PEMBAGIAN PELATIHAN DATA
    std::vector<std::vector<float>> x_data;
    std::vector<float> y_data;

    float ukuran_pelatihan = 0.8;
    float ukuran_test_data = 0.2;

    TIPE_DATA_TRAIN pembagian_pelatihan = train_test.fit(x_data,
                                                         y_data,
                                                         ukuran_pelatihan,
                                                         ukuran_test_data);

    std::vector<std::vector<float>> x_train = pembagian_pelatihan.x_train;
    std::vector<std::vector<float>> x_train_layer_2;
    std::vector<std::vector<float>> x_train_layer_3;

    std::vector<std::vector<float>> x_test = pembagian_pelatihan.x_test;

    std::vector<float> y_train = pembagian_pelatihan.y_train;
    std::vector<float> y_test = pembagian_pelatihan.y_test;

public:
    neural_network_cpp();

    void forward_nn();
    void backward_nn();

    void pelatihan_model_ai(
        std::vector<std::vector<float>> &x_train,
        std::vector<float> &y_train,
        int &epoch);

    void evaluasi_model_ai(std::vector<float> &y_train,
                           std::vector<float> &y_test);

    void pembersihan_memori_ai();
};

#endif