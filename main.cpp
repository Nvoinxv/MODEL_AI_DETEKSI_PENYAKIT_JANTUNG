#include "include/READ_CSV.hpp"
#include "include/PREPROCESSING_DATA.hpp"
#include "include/utils_program.hpp"
#include "include/SPLIT_TRAINING_TEST_DATA.hpp"
#include "include/Binary_Cross_Entropy.hpp"

int main()
{
    std::ifstream File_CSV("/home/nvoinxv/Documents/NEURAL_NETWORK_KLASIFIKASI_DETAK_JANTUNG_CPP/data_klasifikasi_penyakit_jantung.csv");

    MEMBACA_FILE_CSV_CPP read_file_csv;
    UTILS_CPP utils_cpp;

    std::vector<std::vector<float>> x_data;
    std::vector<size_t> y_data;

    read_file_csv.membaca_file_csv(File_CSV, x_data, y_data);

    // OPSIONAL JIKA MAU COBA
    // read_file_csv.menampilkan_data_csv(x_data, y_data);
    // read_file_csv.menampilkan_tipe_tipe_data_csv(x_data, y_data);

    melakukan_preprocessing_data preprocess_data;
    preprocess_data.menghapus_outlier(x_data);

    std::cout << "HASIL DATA SETELAH MELAKUKAN MIN MAX SCALING\n";
    preprocess_data.scaling_min_max(x_data);
    utils_cpp.print_vektor_2d(x_data);

    std::cout << "UKURAN PADA DATA X: "
              << x_data.size() << "\n";

    TIPE_DATA_TRAIN train_test_data;

    float ukuran_pelatihan = 0.8;
    float ukuran_test = 0.2;

    TRAIN_TEST_DATA_X_Y split_train_test;

    train_test_data = split_train_test.fit(
        x_data, y_data,
        ukuran_pelatihan, ukuran_test);

    std::cout << "SAMPLE UKURAN DATA X TRAIN: "
              << train_test_data.x_train.size() << "\n";

    std::cout << "SAMPLE UKURAN DATA X TEST: "
              << train_test_data.x_test.size() << "\n";

    std::cout << "SAMPLE UKURAN DATA Y TRAIN: "
              << train_test_data.y_train.size() << "\n";

    std::cout << "SAMPLE UKURAN DATA Y TEST: "
              << train_test_data.y_test.size() << "\n";

    // TESTING DUMMY DATA
    // MENCOBA PERHITUNGAN LOSS
    std::vector<float> data_mentah_asli;
    std::vector<float> data_mentah_pred;

    int n = 10;

    utils_cpp.data_random_int_1d(data_mentah_asli, n);
    utils_cpp.data_random_int_1d(data_mentah_pred, n);
    BCE_PERHITUNGAN bce;

    /*
    TESTING SUPPORT TIPE DATA STRUCT
    TIPE_DATA_TRAIN forward_bce = bce.BCE_FORWARD(
    train_test_data.y_train,
    train_test_data.y_test);
    */

    std::cout << "TESTING PADA DATA FORWARD BCE\n";
    float forward_bce_test = bce.BCE_FORWARD(data_mentah_asli, data_mentah_pred);

    std::cout << "HASIL LOSS TESTING FORWARD: "
              << forward_bce_test << "\n";

    return 0;
}