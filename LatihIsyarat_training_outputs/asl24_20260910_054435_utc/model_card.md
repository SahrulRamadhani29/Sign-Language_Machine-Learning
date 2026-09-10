# Model Card - LatihIsyarat ASL 24

Dibuat: 2026-09-10T05:46:56+00:00

## Ringkasan

CNN klasifikasi 24 huruf statis ASL (tanpa J dan Z), dilatih dari bobot acak
menggunakan Sign Language MNIST. Model menerima tensor grayscale 28x28x1
float32 bernilai 0-255. Normalisasi 1/255 berada di dalam model.

## Model terpilih

- Eksperimen: `baseline`
- Epoch terbaik: 29
- Validation loss terbaik: 0.000091
- Validation accuracy pada epoch tersebut: 1.000000
- Aturan pemilihan: validation loss terendah, tanpa melihat test set.

## Evaluasi test bawaan

- Accuracy: 0.966955
- Macro precision: 0.962672
- Macro recall: 0.970499
- Macro F1: 0.965374

## Artefak mobile

- Float32 SHA-256: `7cd7cd33676da8962a8072a8e4898478b7c7d0b56b782d6c5a1ee57840ca1ed9`
- Float32 size: 414240 bytes
- Float16 SHA-256: `b11391e44990e37ddc03d4b0d7fa0b3cc7957753513393cd81232b7b028a3a76`
- Float16 size: 210408 bytes

## Dataset

- Sumber: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
- Training bawaan: 27455 baris
- Test bawaan: 7172 baris
- J dan Z tidak tersedia pada dataset dan tidak didukung model.

## Batasan penting

- Hasil test dataset tidak sama dengan performa kamera HP.
- Model tertutup 24 kelas dapat memberi skor tinggi pada gambar tanpa tangan.
- Model belum mengenali gerakan, kalimat, BISINDO, atau SIBI.
- Orientasi, mirror, ROI, dan preprocessing Flutter wajib diverifikasi.
- Lakukan pengujian kamera nyata sebelum membuat klaim penggunaan.
