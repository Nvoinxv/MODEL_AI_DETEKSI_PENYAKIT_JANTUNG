#include "Neural_Network.hpp"

neural_network_cpp::neural_network_cpp() : bce_loss(),
                                           dense_layer(),
                                           dense_layer_2(),
                                           dense_layer_3(),
                                           relu_layer(),
                                           relu_layer_2(),
                                           relu_layer_3(),
                                           dense_output_layer(),
                                           sigmoid_layer(),
                                           SGD_OPTIM(),
                                           variabel_train_test(),
                                           utils_cpp(),
                                           x_train(),
                                           x_train_layer_2(),
                                           x_train_layer_3(),
                                           x_train_output(),
                                           x_test(),
                                           x_test_layar_2(),
                                           x_test_layar_3(),
                                           x_test_output(),
                                           y_train(),
                                           y_test(),
                                           hasil_dense(),
                                           hasil_dense_2(),
                                           hasil_dense_3(),
                                           hasil_dense_output_1d(),
                                           hasil_relu(),
                                           hasil_relu_2(),
                                           hasil_relu_3(),
                                           y_pred(),
                                           y_pred_test(),
                                           hasil_dense_test(),
                                           hasil_dense_test_2(),
                                           hasil_dense_test_3(),
                                           hasil_output_dense_test(),
                                           hasil_relu_test(),
                                           hasil_relu_test_2(),
                                           hasil_relu_test_3(),
                                           grad_y_pred(),
                                           grad_sigmoid(),
                                           grad_relu(),
                                           grad_relu_2(),
                                           grad_relu_3(),
                                           hasil_loss(0.0f),
                                           learning_rate(0.01f)
{
}

std::vector<std::vector<float>> mengubah_1d_ke_2d(
    const std::vector<float> &data,
    size_t jumlah_neuron)
{
    std::vector<std::vector<float>> hasil;

    if (jumlah_neuron == 0 ||
        data.size() % jumlah_neuron != 0)
    {
        throw std::invalid_argument(
            "Ukuran data tidak sesuai dengan jumlah neuron.");
    }

    size_t jumlah_data = data.size() / jumlah_neuron;

    hasil.reserve(jumlah_data);

    for (size_t i = 0; i < data.size(); i += jumlah_neuron)
    {
        hasil.emplace_back(
            data.begin() + i,
            data.begin() + i + jumlah_neuron);
    }

    return hasil;
}

std::vector<float> mengubah_2d_ke_1d(
    const std::vector<std::vector<float>> &data)
{
    std::vector<float> hasil;

    for (const auto &baris : data)
    {
        hasil.insert(
            hasil.end(),
            baris.begin(),
            baris.end());
    }

    return hasil;
}

void neural_network_cpp::forward_nn()
{
    // BAGIAN PERTAMA DENGAN NEURON LAYER 128
    hasil_dense = dense_layer.DENSE_FORWARD_LAYER(128, x_train);
    hasil_relu = relu_layer.RELU_FORWARD(hasil_dense);

    // BAGIAN KEDUA DENGAN NEURON LAYER 64
    x_train_layer_2 = mengubah_1d_ke_2d(hasil_relu, 128);

    hasil_dense_2 = dense_layer_2.DENSE_FORWARD_LAYER(
        64,
        x_train_layer_2);

    hasil_relu_2 = relu_layer_2.RELU_FORWARD(hasil_dense_2);

    // BAGIAN KETIGA DENGAN NEURON LAYER 32
    x_train_layer_3 = mengubah_1d_ke_2d(hasil_relu_2, 64);

    hasil_dense_3 = dense_layer_3.DENSE_FORWARD_LAYER(
        32,
        x_train_layer_3);

    hasil_relu_3 = relu_layer_3.RELU_FORWARD(hasil_dense_3);

    // BAGIAN AKHIR DENGAN OUTPUT LAYER
    x_train_output = mengubah_1d_ke_2d(hasil_relu_3, 32);

    hasil_dense_output = dense_output_layer.DENSE_FORWARD_LAYER(
        1,
        x_train_output);

    // FUNGSI AKTIVASI SIGMOID
    y_pred = sigmoid_layer.SIGMOID_FORWARD(
        1,
        hasil_dense_output);

    // PERHITUNGAN LOSS
    hasil_loss = bce_loss.BCE_FORWARD(y_pred, y_train);
}

void neural_network_cpp::forward_nn_test()
{
    hasil_dense_test = dense_layer.DENSE_FORWARD_LAYER(128, x_test);
    hasil_relu_test = relu_layer.RELU_FORWARD(hasil_dense_test);

    x_test_layar_2 = mengubah_1d_ke_2d(hasil_relu_test, 128);

    hasil_dense_test_2 = dense_layer_2.DENSE_FORWARD_LAYER(64, x_test_layar_2);
    hasil_relu_test_2 = relu_layer_2.RELU_FORWARD(hasil_dense_test_2);

    x_test_layar_3 = mengubah_1d_ke_2d(hasil_relu_2, 64);

    hasil_dense_test_3 = dense_layer_3.DENSE_FORWARD_LAYER(32, x_test_layar_3);
    hasil_relu_test_3 = relu_layer_3.RELU_FORWARD(hasil_dense_test_3);

    x_test_output = mengubah_1d_ke_2d(hasil_relu_test_3, 1);

    hasil_output_dense_test = dense_output_layer.DENSE_FORWARD_LAYER(
        1,
        x_test_output);

    y_pred_test = sigmoid_layer.SIGMOID_FORWARD(1, hasil_output_dense_test);

    hasil_loss = bce_loss.BCE_FORWARD(y_pred_test, y_test);
}

void neural_network_cpp::backward_nn()
{
    // BACKWARD BINARY CROSS ENTROPY
    grad_y_pred = bce_loss.BCE_BACKWARD(
        y_pred,
        y_train);

    // BACKWARD SIGMOID
    grad_sigmoid = sigmoid_layer.SIGMOID_BACKWARD(
        y_pred,
        grad_y_pred);

    // BACKWARD DENSE OUTPUT
    std::vector<std::vector<float>> grad_input_output =
        dense_output_layer.DENSE_BACKWARD_LAYER(
            x_train_output,
            grad_sigmoid);

    std::vector<float> grad_i_o_3 = mengubah_2d_ke_1d(grad_input_output);

    // BACKWARD RELU 3
    grad_relu_3 = relu_layer_3.RELU_BACKWARD(
        grad_i_o_3);

    // BACKWARD DENSE 3
    std::vector<std::vector<float>> grad_input_dense_3 =
        dense_layer_3.DENSE_BACKWARD_LAYER(
            x_train_layer_3,
            grad_relu_3);

    std::vector<float> grad_i_d_3 = mengubah_2d_ke_1d(grad_input_dense_3);

    // BACKWARD RELU 2
    grad_relu_2 = relu_layer_2.RELU_BACKWARD(
        grad_i_d_3);

    // BACKWARD DENSE 2
    std::vector<std::vector<float>> grad_input_dense_2 =
        dense_layer_2.DENSE_BACKWARD_LAYER(
            x_train_layer_2,
            grad_relu_2);

    std::vector<float> grad_i_d_2 = mengubah_2d_ke_1d(grad_input_dense_2);

    // BACKWARD RELU 1
    grad_relu = relu_layer.RELU_BACKWARD(
        grad_i_d_2);

    // BACKWARD DENSE 1
    std::vector<std::vector<float>> grad_input_dense_1 =
        dense_layer.DENSE_BACKWARD_LAYER(
            x_train,
            grad_relu);

    // OPTIMISASI UPDATE BOBOT DAN BIAS
    SGD_OPTIM.update_parameter(
        dense_output_layer,
        learning_rate);

    SGD_OPTIM.update_parameter(
        dense_layer_3,
        learning_rate);

    SGD_OPTIM.update_parameter(
        dense_layer_2,
        learning_rate);

    SGD_OPTIM.update_parameter(
        dense_layer,
        learning_rate);
}

void neural_network_cpp::pelatihan_model_ai(
    std::vector<std::vector<float>> &x_train,
    std::vector<float> &y_train,
    int &epoch)
{
    this->x_train = x_train;
    this->y_train = y_train;

    std::cout << "UKURAN X TRAIN DALAM MODEL: "
              << this->x_train.size() << "\n";

    std::cout << "UKURAN Y TRAIN DALAM MODEL: "
              << this->y_train.size() << "\n";

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

void neural_network_cpp::evaluasi_model_ai(
    const std::vector<float> &y_pred_test,
    const std::vector<float> &y_test)
{
    std::cout << "MEMULAI EVALUASI MODEL AI\n";

    // std::cout << "UKURAN Y_PRED: ";
    //  utils_cpp.print_vektor_1d_float_const(y_pred);

    // std::cout << "UKURAN Y_ACTUAL: ";
    // utils_cpp.print_vektor_1d_float_const(y_pred_test);

    std::cout << "UKURAN Y_PRED: "
              << y_pred_test.size() << "\n";

    std::cout << "UKURAN Y_TES: "
              << y_test.size() << "\n";

    if (y_pred_test.size() != y_test.size())
    {
        throw std::invalid_argument(
            "Jumlah prediksi dan label testing tidak sama.");
    }

    if (y_test.empty())
    {
        throw std::invalid_argument(
            "Data testing kosong.");
    }

    std::cout << "JUMLAH DATA TESTING: "
              << y_test.size() << "\n";

    std::size_t jumlah_prediksi_benar = 0;

    for (std::size_t i = 0; i < y_test.size(); ++i)
    {
        int prediksi = y_pred_test[i] >= 0.5f ? 1 : 0;
        int aktual = y_test[i] >= 0.5f ? 1 : 0;

        if (prediksi == aktual)
        {
            ++jumlah_prediksi_benar;
        }
    }

    float hasil_akurasi =
        static_cast<float>(jumlah_prediksi_benar) /
        static_cast<float>(y_test.size());

    float konversi_persen = hasil_akurasi * 100.0f;

    std::cout << "HASIL AKURASI MODEL: "
              << konversi_persen << "%\n";

    if (konversi_persen >= 70.0f)
    {
        std::cout << "AKURASI HASIL EVALUASI BAGUS\n";
    }
    else
    {
        std::cout << "AKURASI HASIL EVALUASI KURANG BAGUS\n";
    }

    std::cout << "20 SAMPLE DATA KOMPARASI\n";

    std::size_t jumlah_sample =
        std::min<std::size_t>(20, y_test.size());

    for (std::size_t i = 0; i < jumlah_sample; ++i)
    {
        int prediksi = y_pred_test[i] >= 0.5f ? 1 : 0;

        std::cout << "HASIL Y_ACTUAL: "
                  << y_test[i]
                  << " | HASIL Y_PRED: "
                  << y_pred_test[i]
                  << " | HASIL KLASIFIKASI: "
                  << prediksi << "\n";
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