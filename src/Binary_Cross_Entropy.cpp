#include "Binary_Cross_Entropy.hpp"

BCE_PERHITUNGAN::BCE_PERHITUNGAN() : hasil_perhitungan_tipe_data(),
                                     y_actual(),
                                     eular(2.71828182846),
                                     hasil_perhitungan(0.0f),
                                     y_pred(),
                                     y_actual_grad(0.0f),
                                     y_pred_grad(0.0f)
{
}

float BCE_PERHITUNGAN::BCE_FORWARD(
    std::vector<float> &y_pred,
    std::vector<float> &y_actual)
{
    const float epsilon = 1e-7f;

    for (size_t i = 0;
         i < y_pred.size();
         ++i)
    {
        float p = std::clamp(
            y_pred[i],
            epsilon,
            1.0f - epsilon);

        float y = y_actual[i];

        hasil_perhitungan += y * std::log(p) +
                             (1.0f - y) *
                                 std::log(1.0f - p);
    }

    return hasil_perhitungan / static_cast<float>(
                                   y_pred.size());
}

std::vector<float> BCE_PERHITUNGAN::BCE_BACKWARD(
    std::vector<float> &y_pred,
    std::vector<float> &y_actual)
{
    const float epsilon = 1e-7f;

    std::vector<float> grad_y_pred;
    grad_y_pred.reserve(y_pred.size());

    for (size_t i = 0;
         i < y_pred.size();
         ++i)
    {
        float p = std::clamp(
            y_pred[i],
            epsilon,
            1.0f - epsilon);

        float y = y_actual[i];

        hasil_perhitungan = (p - y) /
                            (p * (1.0f - p));

        grad_y_pred.push_back(
            hasil_perhitungan /
            static_cast<float>(y_pred.size()));
    }

    return grad_y_pred;
}