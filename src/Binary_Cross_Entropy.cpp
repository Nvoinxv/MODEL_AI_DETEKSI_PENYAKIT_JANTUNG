#include "Binary_Cross_Entropy.hpp"

BCE_PERHITUNGAN::BCE_PERHITUNGAN() : hasil_perhitungan_tipe_data(),
                                     y_actual(),
                                     eular(2.71828182846),
                                     hasil_perhitungan(0.0f),
                                     y_pred(),
                                     y_actual_grad(0.0f),
                                     y_pred_grad(0.0f),
                                     utils_cpp()
{
}

float BCE_PERHITUNGAN::BCE_FORWARD(
    std::vector<float> &y_pred,
    std::vector<float> &y_actual)
{
    const float epsilon = 1e-7f;

    // MEMBERIKAN SAMPLE UNTUK DEBUGGING
    std::cout << "UKURAN Y_PRED: "
              << y_pred.size() << "\n";

    std::cout << "UKURAN Y_ACTUAL: "
              << y_actual.size() << "\n";

    if (y_pred.size() != y_actual.size() ||
        y_pred.empty())
    {
        throw std::invalid_argument(
            "Ukuran y_pred dan y_actual harus sama dan tidak boleh kosong.");
    }

    size_t jumlah_debug = std::min(
        size_t{16},
        y_pred.size());

    std::cout << "DEBUG SAMPLE Y_PRED:\n";

    for (size_t i = 0; i < jumlah_debug; ++i)
    {
        std::cout << "INDEKS " << i
                  << ": " << y_pred[i] << "\n";
    }

    std::cout << "DEBUG SAMPLE Y_ACTUAL:\n";

    for (size_t i = 0; i < jumlah_debug; ++i)
    {
        std::cout << "INDEKS " << i
                  << ": " << y_actual[i] << "\n";
    }

    hasil_perhitungan = 0.0f;

    for (size_t i = 0;
         i < y_pred.size();
         ++i)
    {
        float p = std::clamp(
            y_pred[i],
            epsilon,
            1.0f - epsilon);

        float y = y_actual[i];

        hasil_perhitungan -= y * std::log(p) +
                             (1.0f - y) *
                                 std::log(1.0f - p);
    }

    std::cout << "DEBUG HASIL PERHITUNGAN PADA LOSS: "
              << hasil_perhitungan << "\n";

    return hasil_perhitungan / static_cast<float>(
                                   y_pred.size());
}

std::vector<float> BCE_PERHITUNGAN::BCE_BACKWARD(
    std::vector<float> &y_pred,
    std::vector<float> &y_actual)
{
    const float epsilon = 1e-7f;

    if (y_pred.size() != y_actual.size() ||
        y_pred.empty())
    {
        throw std::invalid_argument(
            "Ukuran y_pred dan y_actual harus sama dan tidak boleh kosong.");
    }

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

        float grad = (p - y) /
                     (p * (1.0f - p));

        grad_y_pred.push_back(
            grad /
            static_cast<float>(y_pred.size()));
    }

    return grad_y_pred;
}