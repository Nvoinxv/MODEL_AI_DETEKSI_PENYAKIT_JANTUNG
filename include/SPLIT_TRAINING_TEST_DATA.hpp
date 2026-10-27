#ifndef SPLIT_TRAINING_TEST_DATA_HPP
#define SPLIT_TRAINING_TEST_DATA_HPP

#include <vector>
#include <iostream>

struct TIPE_DATA_TRAIN
{
    std::vector<std::vector<float>> x_train;
    std::vector<std::vector<float>> x_test;

    std::vector<float> y_train;
    std::vector<float> y_test;
};
class TRAIN_TEST_DATA_X_Y
{
private:
    float ukuran_train_data;
    float ukuran_test_data;

    float jumlah_train_data;
    float jumlah_test_data;

public:
    TRAIN_TEST_DATA_X_Y();

    TIPE_DATA_TRAIN fit(std::vector<std::vector<float>> &x_data,
                        std::vector<float> &y_data,
                        float &ukuran_train_data,
                        float &ukuran_test_data);
};

#endif