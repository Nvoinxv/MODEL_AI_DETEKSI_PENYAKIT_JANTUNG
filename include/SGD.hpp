#ifndef SGD_OPTIMISASI_HPP
#define SGD_OPTIMISASI_HPP

#include "Dense_Layer.hpp"
#include "utils_program.hpp"
#include <iomanip>

class Sthocastic_Gradient_Descent
{
private:
    UTILS_CPP utils_cpp;

public:
    Sthocastic_Gradient_Descent();

    void update_parameter(
        DENSE_LAYER_NN &dense_layer,
        float &learning_rate);
};

#endif