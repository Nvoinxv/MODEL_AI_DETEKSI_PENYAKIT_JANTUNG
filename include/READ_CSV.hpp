#ifndef READ_CSV_HPP
#define READ_CSV_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>

class MEMBACA_FILE_CSV_CPP
{
private:
    // BUTUH 4 VEKTOR UNTUK MENANGANI FEATURESNYA
    std::vector<std::vector<float>> x_data;

    // UNTUK TARGTE CUKUP 1 VEKTOR
    std::vector<float> y_data;

    std::vector<std::string> mendapatkan_teks_data;

    std::string baris;
    std::string kolom;

    float umur;
    float tekanan_darah;
    float kolestrol;
    float detak_jantung;
    float st_depresi;
    float penyakit_jantung;

public:
    MEMBACA_FILE_CSV_CPP();

    void membaca_file_csv(std::ifstream &file_csv,
                          std::vector<std::vector<float>> &x_data,
                          std::vector<float> &y_data);

    void menampilkan_data_csv(std::vector<std::vector<float>> &x_data,
                              std::vector<float> &y_data);

    void menampilkan_tipe_tipe_data_csv(std::vector<std::vector<float>> &x_data,
                                        std::vector<float> &y_data);
};

#endif
