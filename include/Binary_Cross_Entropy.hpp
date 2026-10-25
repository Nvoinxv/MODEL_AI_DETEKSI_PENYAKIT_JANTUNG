#ifndef BINARY_CROSS_ENTROPY_HPP
#define BINARY_CROSS_ENTROPY_HPP

#include "SPLIT_TRAINING_TEST_DATA.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

class BCE_PERHITUNGAN
{
private:
    TIPE_DATA_TRAIN hasil_perhitungan;

    double eular = 2.71828182846;
    std::vector<float> y_pred;
    std::vector<float> y_actual;

public:
    BCE_PERHITUNGAN();
    float BCE_FORWARD(std::vector<float> &y_pred,
                      std::vector<float> &y_actual);

    float BCE_BACKWARD(
        std::vector<float> &y_pred,
        std::vector<float> &y_actual);

    const std::vector<float> &mendapatkan_y_actual_backward() const
    {
        return y_actual;
    }

    const std::vector<float> &mendapatkan_y_pred_backward() const
    {
        return y_pred;
    }
};

#endif