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

void neural_network_cpp::pelatihan_model_ai(
    std::vector<std::vector<float>> &x_train,
    std::vector<float> &y_train,
    int &epoch)
{
    std::cout << "MEMULAI PELATIHAN MODEL AI\n";

    for (int i = 0; i < epoch; ++i)
    {
        forward_nn();
        backward_nn();

        std::cout << "EPOCH "
                  << (i + 1) << ":\n";

        std::cout << "NILAI LOSS: "
                  << hasil_loss << "\n";
    }

    std::cout << "PELATIHAN MODEL AI SELESAI\n";
}

void neural_network_cpp::evaluasi_model_ai(std::vector<float> &y_train,
                                           std::vector<float> &y_test)
{
    std::cout << "MEMULAI EVALUASI MODEL AI\n";

    std::cout << "HASIL NILAI LOSS MODEL AI: "
              << hasil_loss << "\n";

    float hasil_akurasi = 0.0f;

    for (int i = 0;
         i < y_train.size();
         ++i)
    {
        hasil_akurasi = (y_test[i]) / (y_train[i]);
    }

    float konversi_persen = hasil_akurasi * 100;

    std::cout << "HASIL AKURASI MODEL: "
              << konversi_persen
              << "%" << "\n";

    if (konversi_persen >= 70)
    {
        std::cout << "AKURASI HASIL PELATIHAN BAGUS\n";
    }

    else
    {
        std::cout << "AKURASI HASIL PELATIHAN KURANG BAGUS\n";
    }

    std::cout << "20 SAMPLE DATA KOMPERASI\n";

    for (int i = 0;
         i < 20;
         ++i)
    {
        std::cout << "HASIL Y_ACUTAL: "
                  << y_train[i]
                  << " | " << "HASIL Y_PRED: "
                  << y_test[i]
                  << "\n";
    }
}

void neural_network_cpp::pembersihan_memori_ai()
{
    std::cout << "MELAKUKAN PEMBERSIHAN MEMORI MODEL AI\n";
    hasil_dense.clear();
    hasil_relu.clear();
    y_pred.clear();

    grad_y_pred.clear();
    grad_sigmoid.clear();
    grad_relu.clear();

    x_train.clear();
    x_test.clear();

    y_train.clear();
    y_test.clear();

    std::cout << "PEMBERSIHAN SUDAH SELESAI\n";
}