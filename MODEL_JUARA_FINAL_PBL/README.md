# Model Juara Final PBL

Folder ini berisi model dengan accuracy tertinggi dari seluruh eksperimen
LatihIsyarat yang tersedia pada 10 September 2026.

## Hasil utama

- Model: ensemble CNN `wide_3_5mb` setelah full-data fine-tune.
- Anggota ensemble: seed 42, 123, dan 2026.
- Training data: 27.455 gambar Sign Language MNIST.
- Test data: 7.172 gambar Sign Language MNIST.
- Test accuracy: 99,302844% (7.122 benar, 50 salah).
- Macro precision: 99,341579%.
- Macro recall: 99,354557%.
- Macro-F1: 99,332522%.
- Ukuran TFLite: 5.523.980 byte / 5,268 MiB.
- TFLite label agreement: 100% pada 512 sampel parity.
- SHA-256 model: `80127bd9f692750047447500578bc8ecc3cdca466eee14037a55d53dc7ea79a9`.

Accuracy tersebut berasal dari official test set Sign Language MNIST. Angka
ini bukan accuracy kamera HP dan harus dilaporkan dengan konteks tersebut.

## File untuk Flutter

Salin tiga file dalam `flutter_assets/` ke proyek Flutter:

```text
flutter_assets/
|-- model_float16.tflite
|-- labels.json
`-- model_metadata.json
```

Contoh deklarasi `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/ml/model_float16.tflite
    - assets/ml/labels.json
    - assets/ml/model_metadata.json
```

## Kontrak input dan output

- Input tensor: `[1, 28, 28, 1]`.
- Input dtype: `float32`.
- Warna: grayscale.
- Rentang nilai sebelum model: `0..255`.
- Normalisasi `1/255` sudah berada di dalam model.
- Aplikasi tidak boleh membagi nilai piksel dengan 255 lagi.
- Output tensor: `[1, 24]`, dtype `float32`.
- Urutan label wajib dibaca dari `labels.json`.
- Huruf J dan Z tidak didukung karena membutuhkan pengenalan gerakan.

Walaupun bobot model disimpan sebagai float16, input dan output model tetap
float32. Kamera perlu melakukan crop tangan, resize 28x28, grayscale, dan
menangani orientasi serta mirror kamera depan dengan benar.

## Catatan implementasi

Model adalah ensemble tiga CNN di dalam satu file TFLite. Ukurannya aman untuk
on-device, tetapi komputasinya sekitar tiga kali model tunggal. Sebelum dipakai
sebagai model produksi, ukur latency, RAM, suhu perangkat, dan kestabilan
prediksi pada HP target.

Kesalahan test paling banyak adalah H menjadi G dan T menjadi X. Gunakan
confidence threshold dan voting beberapa frame agar aplikasi tidak langsung
mengubah hasil karena satu frame yang tidak stabil. Model 24 kelas ini juga
belum memiliki kelas `no_hand`, sehingga aplikasi harus menolak frame tanpa
tangan melalui deteksi tangan atau threshold tambahan.

## Laporan pendukung

Folder `reports/` berisi:

- `evaluation.json`: metrik lengkap per huruf.
- `confusion_matrix.png`: visualisasi kesalahan klasifikasi.
- `test_predictions.csv`: seluruh prediksi pada 7.172 test images.
- `tflite_parity.json`: perbandingan Keras dengan TFLite.
- `MODEL_CARD_SOURCE.md`: kartu model dari eksperimen sumber.
- `artifact_manifest_source.csv`: manifest artefak sumber.

Model ini sebelumnya dipilih dari hasil eksperimen di:

```text
LatihIsyarat_experimental_outputs/
wide_3seed_full_data_refit_v1/
MODEL_FINAL/
```

Folder eksperimen mentah tersebut telah dihapus setelah model juara, metadata,
laporan evaluasi, dan checksum disalin serta diverifikasi ke bundle ini.
