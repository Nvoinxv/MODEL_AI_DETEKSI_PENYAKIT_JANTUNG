#include "Binary_Cross_Entropy.hpp"

BCE_PERHITUNGAN::BCE_PERHITUNGAN() : hasil_perhitungan_tipe_data(),
                                     y_actual(),
                                     eular(2.71828182846),
                                     hasil_perhitungan(0.0f),
                                     y_pred()
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

float BCE_PERHITUNGAN::BCE_BACKWARD(
    std::vector<float> &y_pred,
    std::vector<float> &y_actual)
{

    for (int i = 0;
         i < hasil_perhitungan_tipe_data.y_train.size();
         ++i)
    {
        hasil_perhitungan = (hasil_perhitungan_tipe_data.y_train[i] -
                             hasil_perhitungan_tipe_data.y_test[i]) /
                            hasil_perhitungan_tipe_data.y_train[i] *
                            (1.0f - hasil_perhitungan_tipe_data.y_train[i]);
    }

    return hasil_perhitungan;
}