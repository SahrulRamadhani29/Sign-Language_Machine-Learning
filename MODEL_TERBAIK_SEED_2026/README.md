# Model Terbaik LatihIsyarat - Seed 2026

Folder ini adalah salinan terpisah dari model baseline seed 2026 yang dipilih
berdasarkan validation loss terendah dalam perbandingan seed 42, 123, dan 2026.
Hasil eksperimen asli tetap berada di `LatihIsyarat_training_outputs/`.

## Hasil utama

- Test accuracy: 96.695482%
- Macro-F1: 96.537414%
- Validation accuracy: 100%
- Validation loss: 0.000091
- Best epoch: 29 dari 30
- TFLite float32 label agreement: 100% pada 512 sampel parity

## File untuk Flutter

Gunakan file dalam `flutter_assets/`:

```text
model_float32.tflite
labels.json
model_metadata.json
```

`model_float16.tflite` adalah kandidat lebih kecil. Gunakan hanya setelah
latency dan hasilnya dibandingkan pada perangkat Android sasaran.

Kontrak input model float32:

```text
shape  : [1, 28, 28, 1]
dtype  : float32
range  : 0-255
color  : grayscale
output : [1, 24]
```

Normalisasi 1/255 sudah berada di dalam model. Aplikasi Flutter tidak boleh
membagi input dengan 255 lagi. Urutan label wajib dibaca dari `labels.json`.

## Model Keras

`keras/best_model.keras` digunakan untuk evaluasi Python, konversi ulang, atau
training lanjutan. File ini tidak perlu dimasukkan ke aplikasi Flutter.

## Laporan

Folder `reports/` menyimpan konfigurasi training, hasil evaluasi, kurva
training, confusion matrix, model card, dan laporan parity Keras-TFLite.

Model ini terbaik pada evaluasi Sign Language MNIST. Performa kamera HP masih
harus diuji terpisah sebelum model dinyatakan final untuk aplikasi.

