#ifndef SIGMOID_FUNCTION_HPP
#define SIGMOID_FUNCTION_HPP

#include "SPLIT_TRAINING_TEST_DATA.hpp"
#include <cmath>

class AKTIVASI_SIGMOID
{
private:
    std::vector<float> x_data;

    float y_pred_grad;
    float y_actual_grad;

    std::vector<float> hasil_grad_sigmoid;
    std::vector<float> hasil_forward_sigmoid;

    float hasil_perhitungan_sigmoid;

    const float EULAR = 2.718281828459045235360287471352;

public:
    AKTIVASI_SIGMOID();

    std::vector<float> SIGMOID_FORWARD(std::vector<float> &x_data);

    std::vector<float> SIGMOID_BACKWARD(float &y_actual_grad,
                                        float &y_pred_grad);

    /*
    const float &mendapatkan_nilai_sigmoid_grad()
    {
        return hasil_perhitungan_sigmoid;
    }
    */

    const std::vector<float> &mendapatkan_sigmoid_grad()
    {
        return hasil_grad_sigmoid;
    }
};

#endif