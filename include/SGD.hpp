#ifndef SGD_OPTIMISASI_HPP
#define SGD_OPTIMISASI_HPP

#include "Dense_Layer.hpp"

class Sthocastic_Gradient_Descent
{
public:
    Sthocastic_Gradient_Descent();

    void update_parameter(
        DENSE_LAYER_NN &dense_layer,
        float &learning_rate);
};

#endif