#include "PREPROCESSING_DATA.hpp"

melakukan_preprocessing_data::melakukan_preprocessing_data() : x_data()
{
}

float melakukan_preprocessing_data::perhitungan_median(std::vector<float> &data)
{
    int n = x_data.size();

    int tengah = n / 2;

    if (n % 2 == 0)
    {
        std::cout << "PERHITUNGAN MEDIAN GENAP\n";

        return {
            data[tengah] + (data[tengah] + 1.0f) / 2.0f};
    }

    else
    {
        std::cout << "PERHITUNGAN MEDIAN GANJIL\n";

        return {
            data[tengah]};
    }
}

void melakukan_preprocessing_data::menghapus_outlier(std::vector<std::vector<float>> &x_data)
{
    std::cout << "SEBELUM MELAKUKAN PENGHAPUSAN OUTLIER: "
              << x_data.size() << "\n";

    size_t jumlah_fitur_data = x_data[0].size();

    std::vector<float> Q1_GABUNGAN(jumlah_fitur_data);
    std::vector<float> Q3_GABUNGAN(jumlah_fitur_data);

    for (int i = 0; i < jumlah_fitur_data; ++i)
    {
        std::vector<float> data_kolom;
        for (const auto &baris_data : x_data)
        {

            data_kolom.push_back(baris_data[i]);
        }
        std::sort(data_kolom.begin(), data_kolom.end());

        int n = data_kolom.size();
        int tengah = n / 2;

        std::vector<float> Low(
            data_kolom.begin(),
            data_kolom.begin() + tengah);

        std::vector<float> High(
            data_kolom.begin() + tengah + (n % 2),
            data_kolom.end());

        float Q1 = perhitungan_median(Low);
        float Q3 = perhitungan_median(High);

        Q1_GABUNGAN[i] = Q1;
        Q3_GABUNGAN[i] = Q3;
    }

    std::cout << "MENAMPILKAN HASIL PERHITUNGAN Q3 DAN Q1\n";

    std::cout << "Q1 KOLOM 0: " << Q1_GABUNGAN[0] << "\n";
    std::cout << "Q1 KOLOM 1: " << Q1_GABUNGAN[1] << "\n";
    std::cout << "Q1 KOLOM 2: " << Q1_GABUNGAN[2] << "\n";
    std::cout << "Q1 KOLOM 3: " << Q1_GABUNGAN[3] << "\n";

    std::cout << "Q3 KOLOM 0: " << Q3_GABUNGAN[0] << "\n";
    std::cout << "Q3 KOLOM 1: " << Q3_GABUNGAN[1] << "\n";
    std::cout << "Q3 KOLOM 2: " << Q3_GABUNGAN[2] << "\n";
    std::cout << "Q3 KOLOM 3: " << Q3_GABUNGAN[3] << "\n";

    std::vector<float> IQR(jumlah_fitur_data);

    std::vector<float> bagian_bawah(jumlah_fitur_data);
    std::vector<float> bagian_atas(jumlah_fitur_data);

    for (int i = 0; i < jumlah_fitur_data; ++i)
    {
        IQR[i] = Q3_GABUNGAN[i] - Q1_GABUNGAN[i];

        // PERHITUNGAN BOUND LOW DAN HIGH
        bagian_bawah[i] = Q1_GABUNGAN[i] - (1.5f * IQR[i]);
        bagian_atas[i] = Q3_GABUNGAN[i] + (1.5f * IQR[i]);
    }

    std::cout << "MENAMPILKAN HASIL PERHITUNGAN IQR\n";

    std::cout << "IQR KOLOM 0: " << IQR[0] << "\n";
    std::cout << "IQR KOLOM 1: " << IQR[1] << "\n";
    std::cout << "IQR KOLOM 2: " << IQR[2] << "\n";
    std::cout << "IQR KOLOM 3: " << IQR[3] << "\n";

    std::cout << "MENAMPILKAN HASIL PERHITUNGAN BAGIAN BAWAH\n";

    std::cout << "BAGIAN BAWAH KOLOM 0: " << bagian_bawah[0] << "\n";
    std::cout << "BAGIAN BAWAH KOLOM 1: " << bagian_bawah[1] << "\n";
    std::cout << "BAGIAN BAWAH KOLOM 2: " << bagian_bawah[2] << "\n";
    std::cout << "BAGIAN BAHWA KOLOM 3: " << bagian_bawah[3] << "\n";

    std::cout << "MENAMPILKAN HASIL PERHITUNGAN BAGIAN ATAS\n";

    std::cout << "BAGIAN ATAS KOLOM 0: " << bagian_atas[0] << "\n";
    std::cout << "BAGIAN ATAS KOLOM 1: " << bagian_atas[1] << "\n";
    std::cout << "BAGIAN ATAS KOLOM 2: " << bagian_atas[2] << "\n";
    std::cout << "BAGIAN ATAS KOLOM 3: " << bagian_atas[3] << "\n";

    x_data.erase(
        std::remove_if(
            x_data.begin(), x_data.end(), [&](const std::vector<float> &data)
            { return data[0] < bagian_bawah[0] ||
                     data[0] > bagian_atas[0] ||

                     data[1] < bagian_bawah[1] ||
                     data[1] > bagian_atas[1] ||

                     data[2] < bagian_bawah[2] ||
                     data[2] > bagian_atas[2] ||

                     data[3] < bagian_bawah[3] ||
                     data[3] > bagian_atas[3]; }),
        x_data.end());

    std::cout << "SESUDAH MELAKUKAN MENGHAPUS OUTLIER: "
              << x_data.size() << "\n";
}

void melakukan_preprocessing_data::scaling_min_max(
    std::vector<std::vector<float>> &x_data)
{
    size_t n = x_data[0].size();

    for (size_t i = 0; i < n; ++i)
    {
        float nilai_max = x_data[0][i];
        float nilai_min = x_data[0][i];

        for (size_t j = 1; j < x_data.size(); ++j)
        {
            nilai_max = std::max(nilai_max, x_data[j][i]);
            nilai_min = std::min(nilai_min, x_data[j][i]);
        }

        if (nilai_max == nilai_min)
        {
            for (size_t k = 0; k < x_data.size(); ++k)
            {
                x_data[k][i] = 0.0f;
            }

            continue;
        }

        for (size_t k = 0; k < x_data.size(); ++k)
        {
            x_data[k][i] =
                (x_data[k][i] - nilai_min) /
                (nilai_max - nilai_min);
        }
    }
}