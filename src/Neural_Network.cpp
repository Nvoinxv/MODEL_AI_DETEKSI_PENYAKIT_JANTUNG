#include "Neural_Network.hpp"

neural_network_cpp::neural_network_cpp() : bce_loss(),
                                           dense_layer(),
                                           relu_layer(),
                                           sigmoid_layer(),
                                           SGD_OPTIM(),
                                           variabel_train_test(),
                                           x_train(),
                                           x_test(),
                                           y_train(),
                                           y_test(),
                                           hasil_dense(),
                                           hasil_relu(),
                                           y_pred(),
                                           grad_y_pred(),
                                           grad_sigmoid(),
                                           grad_relu(),
                                           hasil_loss(0.0f)
{
}

void neural_network_cpp::forward_nn()
{
    hasil_dense = dense_layer.DENSE_FORWARD_LAYER(x_train);

    hasil_relu = relu_layer.RELU_FORWARD(hasil_dense);

    y_pred = sigmoid_layer.SIGMOID_FORWARD(hasil_relu);

    hasil_loss = bce_loss.BCE_FORWARD(y_pred, y_train);
}

void neural_network_cpp::backward_nn()
{
    grad_y_pred = bce_loss.BCE_BACKWARD(y_pred, y_train);

    grad_sigmoid = sigmoid_layer.SIGMOID_BACKWARD(
        y_pred,
        grad_y_pred);

    grad_relu = relu_layer.RELU_BACKWARD(grad_sigmoid);

    dense_layer.DENSE_BACKWARD_LAYER(
        x_train,
        grad_relu);
}