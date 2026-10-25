#include "SPLIT_TRAINING_TEST_DATA.hpp"

TRAIN_TEST_DATA_X_Y::TRAIN_TEST_DATA_X_Y() : ukuran_train_data(0.0f),
                                             ukuran_test_data(0.0f),
                                             jumlah_train_data(0.0f),
                                             jumlah_test_data(0.0f)

{
}

TIPE_DATA_TRAIN TRAIN_TEST_DATA_X_Y::fit(std::vector<std::vector<float>> &x_data,
                                         std::vector<size_t> &y_data,
                                         float &ukuran_train_data,
                                         float &ukuran_test_data)
{
    int jumlah_data = x_data.size();

    jumlah_train_data = static_cast<int>(jumlah_data * ukuran_train_data);
    jumlah_test_data = jumlah_data - ukuran_test_data;

    TIPE_DATA_TRAIN hasil;

    for (size_t i = 0;
         i < jumlah_train_data;
         ++i)
    {
        hasil.x_train.push_back(x_data[i]);
        hasil.y_train.push_back(
            static_cast<int>(y_data[i]));
    }

    for (size_t i = jumlah_train_data;
         i < jumlah_data;
         ++i)
    {
        hasil.x_test.push_back(x_data[i]);
        hasil.y_test.push_back(
            static_cast<int>(y_data[i]));
    }

    return hasil;
}