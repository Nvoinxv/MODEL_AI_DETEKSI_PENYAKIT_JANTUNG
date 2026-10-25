#include "READ_CSV.hpp"

MEMBACA_FILE_CSV_CPP::MEMBACA_FILE_CSV_CPP() : x_data(), y_data(), mendapatkan_teks_data(),
                                               baris(), kolom(),
                                               umur(0.0f),
                                               tekanan_darah(0.0f),
                                               kolestrol(0.0f),
                                               detak_jantung(0.0f),
                                               st_depresi(0.0f),
                                               penyakit_jantung(0)
{
}

void MEMBACA_FILE_CSV_CPP::membaca_file_csv(
    std::ifstream &data_csv,
    std::vector<std::vector<float>> &x_data,
    std::vector<size_t> &y_data)
{
    if (!std::getline(data_csv, baris))
    {
        throw std::invalid_argument("TIDAK TERDETEKSI FILE CSV!");
    }

    while (std::getline(data_csv, baris))
    {
        std::stringstream pemisah_baris(baris);

        mendapatkan_teks_data.clear();

        while (std::getline(pemisah_baris, kolom, ','))
        {
            mendapatkan_teks_data.push_back(kolom);
        }

        try
        {
            // MELAKUKAN KONVERSI DATA
            umur = std::stof(mendapatkan_teks_data[0]);
            tekanan_darah = std::stof(mendapatkan_teks_data[1]);
            kolestrol = std::stof(mendapatkan_teks_data[2]);
            detak_jantung = std::stof(mendapatkan_teks_data[3]);
            st_depresi = std::stof(mendapatkan_teks_data[4]);
            penyakit_jantung = std::stoi(mendapatkan_teks_data[5]);
        }

        catch (std::exception &e)
        {
            std::cout << "GAGAL MELAKUKAN KONVERSI PADA BARIS: "
                      << baris << "\n"
                      << e.what() << "\n";

            continue;
        }

        // MASUKKAN KE DALAM VARIABEL X DAN Y
        x_data.push_back({umur,
                          tekanan_darah,
                          kolestrol,
                          st_depresi});

        y_data.push_back({penyakit_jantung});
    }
}

void MEMBACA_FILE_CSV_CPP::menampilkan_data_csv(
    std::vector<std::vector<float>> &x_data,
    std::vector<size_t> &y_data)
{
    std::cout << "MENAMPILKAN DATA CSV PER INDEKS\n";

    size_t n = x_data.size();

    // Memastikan data berubah
    /*
        for (size_t i = 0; i < x_data.size(); ++i)
    {

        std::cout << "indeks ke " << i << "\n";

        for (float &data_test : x_data[i])
        {
            std::cout << data_test << " ";
        }

        std::cout << "\n";
    }*/

    for (size_t i = 0; i < n; ++i)
    {
        std::cout << "DATA X INDEKS KE " << i << ": " << "\n";

        for (size_t j = 0;
             j < x_data[i].size();
             ++j)
        {
            std::cout << x_data[i][j] << " ";
        }

        std::cout << "\n";

        std::cout << "DATA Y INDEKS KE "
                  << i << ":" << "\n";

        std::cout << y_data[i] << " "
                  << "\n";
    }
}

void MEMBACA_FILE_CSV_CPP::menampilkan_tipe_tipe_data_csv(
    std::vector<std::vector<float>> &x_data,
    std::vector<size_t> &y_data)
{
    std::cout << "MENAMPILKAN KONVERSI DATA PADA SETIAP CSV\n";

    std::cout << "DATA X KOLOM 0: " << typeid(x_data[0]).name() << "\n";
    std::cout << "DATA X KOLOM 1: " << typeid(x_data[1]).name() << "\n";
    std::cout << "DATA X KOLOM 2: " << typeid(x_data[2]).name() << "\n";
    std::cout << "DATA X KOLOM 3: " << typeid(x_data[3]).name() << "\n";
    std::cout << "DATA Y: " << typeid(y_data).name() << "\n";
}