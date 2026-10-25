#include "Binary_Cross_Entropy.hpp"

BCE_PERHITUNGAN::BCE_PERHITUNGAN() : hasil_perhitungan(),
                                     y_actual(),
                                     eular(2.71828182846),
                                     y_pred()
{
}

float BCE_PERHITUNGAN::BCE_FORWARD(
    std::vector<float> &y_pred,
    std::vector<float> &y_actual)
{
    float hasi_perhitungan_float = 0.0f;
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

        hasi_perhitungan_float += y * std::log(p) +
                                  (1.0f - y) *
                                      std::log(1.0f - p);
    }

    return hasi_perhitungan_float / static_cast<float>(
                                        y_pred.size());
}

float BCE_PERHITUNGAN::BCE_BACKWARD(
    std::vector<float> &y_pred,
    std::vector<float> &y_actual)
{
    float hasil_perhitungan_float;

    for (int i = 0;
         i < hasil_perhitungan.y_train.size();
         ++i)
    {
        hasil_perhitungan_float = (hasil_perhitungan.y_train[i] - hasil_perhitungan.y_test[i]) / hasil_perhitungan.y_train[i] * (1.0f - hasil_perhitungan.y_train[i]);
    }

    return hasil_perhitungan_float;
}