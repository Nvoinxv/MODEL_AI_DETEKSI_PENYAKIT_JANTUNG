#ifndef PREPROCESSING_DATA_HPP
#define PREPROCESSING_DATA_HPP

#include <vector>
#include <iostream>
#include <stdexcept>
#include <bits/stdc++.h>
#include <algorithm>

class melakukan_preprocessing_data
{
private:
    std::vector<std::vector<float>> x_data;

public:
    melakukan_preprocessing_data();

    float perhitungan_median(std::vector<float> &data);
    void menghapus_outlier(std::vector<std::vector<float>> &x_data);
    void scaling_min_max(std::vector<std::vector<float>> &x_data);
};

#endif