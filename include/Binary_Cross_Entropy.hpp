#ifndef BINARY_CROSS_ENTROPY_HPP
#define BINARY_CROSS_ENTROPY_HPP

#include "SPLIT_TRAINING_TEST_DATA.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <utils_program.hpp>

class BCE_PERHITUNGAN
{
private:
    TIPE_DATA_TRAIN hasil_perhitungan_tipe_data;

    double eular = 2.71828182846;
    std::vector<float> y_pred;
    std::vector<float> y_actual;

    float y_pred_grad;
    float y_actual_grad;

    float hasil_perhitungan;

    UTILS_CPP utils_cpp;

public:
    BCE_PERHITUNGAN();
    float BCE_FORWARD(std::vector<float> &y_pred,
                      std::vector<float> &y_actual);

    std::vector<float> BCE_BACKWARD(
        std::vector<float> &y_pred,
        std::vector<float> &y_actual);

    const float &mendapatkan_y_actual_backward() const
    {
        return y_actual_grad;
    }

    const float &mendapatkan_y_pred_backward() const
    {
        return y_pred_grad;
    }

    const float &mendapatkan_grad_output() const
    {
        return hasil_perhitungan;
    }
};

#endif