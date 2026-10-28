#include "utils_program.hpp"

UTILS_CPP::UTILS_CPP()
{
}

void UTILS_CPP::print_vektor_2d(std::vector<std::vector<float>> &data)
{
    int n = data.size();

    for (int i = 0; i < n; ++i)
    {

        for (int j = 0; j < data[i].size(); ++j)
        {
            std::cout << data[i][j] << " ";
        }

        std::cout << "\n";
    }

    std::cout << "\n";
}

std::vector<std::vector<float>> UTILS_CPP::data_random_float_2d(
    std::vector<std::vector<float>> &data,
    int &n)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    data.clear();

    for (int i = 0; i < n; ++i)
    {
        std::vector<float> data_random;

        for (int j = 0; j < n; ++j)
        {
            data_random.push_back(dist(gen));
        }

        data.push_back(data_random);
    }

    return data;
}

std::vector<float> UTILS_CPP::data_random_int_1d(
    std::vector<float> &data,
    int &n)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 1);

    data.clear();
    for (int i = 0; i < n; ++i)
    {
        data.push_back(dist(gen));
    }

    return data;
}

void UTILS_CPP::print_vektor_1d_float(
    std::vector<float> &data)
{
    int n = data.size();

    for (int i = 0; i < n; ++i)
    {
        std::cout << data[i] << " ";
    }

    std::cout << "\n";
}

void UTILS_CPP::print_vektor_struct_data(TIPE_DATA_TRAIN &a)
{
    // BAGIAN X TRAIN
    for (int i = 0;
         i < a.x_train.size();
         ++i)
    {
        for (int j = 0;
             j < a.x_train[i].size();
             ++j)
        {
            std::cout << a.x_train[i][j] << " ";
        }

        std::cout << "\n";
    }

    // BAGIAN X TEST
    for (int i = 0;
         i < a.x_test.size();
         ++i)
    {

        for (int j = 0;
             j < a.x_test[i].size();
             ++j)
        {
            std::cout << a.x_test[i][j] << " ";
        }

        std::cout << "\n";
    }

    // BAGIAN Y TRAIN
    for (int i = 0;
         i < a.y_train.size(); ++i)
    {
        std::cout << a.y_train[i] << " ";
    }
    std::cout << "\n";

    // BAGIAN Y TEST
    for (int i = 0;
         i < a.y_test.size();
         ++i)
    {
        std::cout << a.y_test[i] << " ";
    }

    std::cout << "\n";
}