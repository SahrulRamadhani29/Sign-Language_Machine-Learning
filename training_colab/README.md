# Training CNN LatihIsyarat di Google Colab

Folder ini adalah paket training mandiri. Ia tidak mengatur struktur proyek
Flutter. Hasil yang nanti dibutuhkan Flutter adalah `model_float32.tflite`,
`labels.json`, dan `model_metadata.json`.

[Buka notebook di Google Colab](https://colab.research.google.com/github/SahrulRamadhani29/Sign-Language_Machine-Learning/blob/main/training_colab/LatihIsyarat_Training_Colab.ipynb)

## Cara termudah

1. Push repositori ini ke GitHub.
2. Klik tautan **Buka notebook di Google Colab** atau upload file
   `LatihIsyarat_Training_Colab.ipynb` ke Colab.
3. Pada menu Colab pilih **Runtime > Change runtime type > T4 GPU** jika tersedia.
4. Jalankan **Runtime > Run all**.
5. Izinkan akses Google Drive ketika diminta.

Notebook akan melakukan semuanya secara berurutan:

1. Memasang dependensi ringan yang belum tersedia.
2. Mengunduh dataset resmi
   `datamunge/sign-language-mnist` melalui `kagglehub`.
3. Memvalidasi label, ukuran 28x28, rentang piksel, dan duplikat identik.
4. Membagi training bawaan menjadi training 80% dan validation 20% dengan
   stratifikasi dan seed tetap.
5. Melatih kandidat CNN dari bobot acak.
6. Menyimpan checkpoint dengan `val_loss` terbaik pada setiap kandidat.
7. Memilih kandidat terbaik hanya berdasarkan validation loss.
8. Mengevaluasi test bawaan satu kali setelah pemilihan selesai.
9. Mengekspor model TFLite float32 dan float16 serta membandingkannya dengan
    prediksi Keras.
10. Menyimpan laporan, hash, metadata, grafik, dan model ke Google Drive.

## Baseline atau perbandingan

- Pada notebook, set `QUICK_BASELINE_ONLY = True` untuk melatih satu baseline.
  Gunakan ini untuk memastikan seluruh pipeline berjalan.
- Set `QUICK_BASELINE_ONLY = False` untuk melatih tiga kandidat terkontrol.
  Model dengan validation loss terendah otomatis menjadi `best_model.keras`.
- `config_baseline.json` dan `config_compare.json` menyediakan konfigurasi yang
  sama jika pipeline dijalankan melalui terminal.

Notebook membandingkan tiga kandidat secara default. Untuk tes pertama, ubah
`QUICK_BASELINE_ONLY` menjadi `True`.

## Arti "model terbaik"

Ada dua tingkat pemilihan:

1. Dalam satu eksperimen, `ModelCheckpoint` menyimpan epoch dengan validation
   loss terendah, bukan otomatis epoch terakhir.
2. Jika beberapa eksperimen dijalankan, pipeline memilih checkpoint dengan
   validation loss terendah di antara semua kandidat.

Test set tidak digunakan untuk memilih epoch, learning rate, dropout, atau
augmentasi. Ini mencegah hasil test menjadi terlalu optimistis.

Model terbaik di dataset juga belum tentu terbaik pada kamera HP. Setelah
integrasi Flutter, model masih harus diuji pada gambar kamera terpisah,
termasuk ROI kosong dan kondisi tanpa tangan.

## Folder hasil di Google Drive

Secara default hasil disimpan di:

```text
MyDrive/LatihIsyarat_training_outputs/asl24_YYYYMMDD_HHMMSS_utc/
```

Isi pentingnya:

```text
best_model.keras
model_float32.tflite
model_float16.tflite
labels.json
model_metadata.json
evaluation.json
experiment_comparison.csv
training_history.csv
training_curves.png
confusion_matrix.png
dataset_samples.png
model_card.md
dataset_report.json
tflite_parity.json
artifact_manifest.csv
experiments/
```

Gunakan `model_float32.tflite` sebagai model integrasi pertama. Pakai float16
hanya jika hasil `tflite_parity.json` baik dan pengujian perangkat menunjukkan
manfaat ukuran atau kecepatan.

## Kontrak input untuk Flutter

- Bentuk tensor: `[1, 28, 28, 1]`.
- Dtype: `float32`.
- Nilai sebelum masuk model: `0..255`.
- Warna: grayscale.
- Normalisasi `1/255` sudah berada di dalam model.
- Flutter tidak boleh membagi nilai dengan 255 lagi.
- Urutan label wajib dibaca dari `labels.json`.
- Output indeks 9 adalah K, bukan J.

Formula grayscale yang dicatat dalam metadata:

```text
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

Orientasi, mirror kamera depan, crop ROI, dan perilaku resize tetap harus diuji
dengan gambar referensi yang sama di Python dan Flutter.

## Jika download Kaggle gagal

Dataset yang benar:

https://www.kaggle.com/datasets/datamunge/sign-language-mnist

Unduh ZIP secara manual, ekstrak, lalu pastikan tersedia:

```text
sign_mnist_train.csv
sign_mnist_test.csv
```

Upload folder CSV ke Google Drive, kemudian isi `DATASET_DIR` pada notebook
dengan path folder tersebut. Jangan memasukkan CSV ke Git karena ukurannya
besar dan tidak diperlukan oleh aplikasi.

## Menjalankan pemeriksaan lokal

Pemeriksaan kontrak yang tidak membutuhkan TensorFlow:

```bash
python -m unittest training_colab/test_pipeline.py -v
```

Training melalui terminal Colab atau mesin yang memiliki TensorFlow:

```bash
python training_colab/training_pipeline.py \
  --config training_colab/config_baseline.json \
  --output-dir /content/drive/MyDrive/LatihIsyarat_training_outputs
```

## Catatan klaim hasil

Jangan menulis angka accuracy, ukuran model, atau kecepatan sebelum notebook
benar-benar selesai dijalankan. `evaluation.json` adalah sumber angka evaluasi
dataset. Performa kamera dan latency perangkat harus dilaporkan terpisah.
