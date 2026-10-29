#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <vector>
#include <random>
#include "SPLIT_TRAINING_TEST_DATA.hpp"

class UTILS_CPP
{
public:
    UTILS_CPP();

    void print_vektor_2d(std::vector<std::vector<float>> &data);
    std::vector<std::vector<float>> data_random_float_2d(
        std::vector<std::vector<float>> &data,
        int &n);

    std::vector<float> data_random_int_1d(std::vector<float> &data, int &n);
    void print_vektor_1d_float(std::vector<float> &data);

    void print_vektor_struct_data(TIPE_DATA_TRAIN &a);

    void print_vektor_1d_float_const(const std::vector<float> &data);
};

#endif