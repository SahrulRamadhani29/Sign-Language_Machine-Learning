# Model Card - LatihIsyarat ASL 24

Dibuat: 2026-09-10T05:21:03+00:00

## Ringkasan

CNN klasifikasi 24 huruf statis ASL (tanpa J dan Z), dilatih dari bobot acak
menggunakan Sign Language MNIST. Model menerima tensor grayscale 28x28x1
float32 bernilai 0-255. Normalisasi 1/255 berada di dalam model.

## Model terpilih

- Eksperimen: `baseline`
- Epoch terbaik: 29
- Validation loss terbaik: 0.000730
- Validation accuracy pada epoch tersebut: 1.000000
- Aturan pemilihan: validation loss terendah, tanpa melihat test set.

## Evaluasi test bawaan

- Accuracy: 0.959565
- Macro precision: 0.958894
- Macro recall: 0.961174
- Macro F1: 0.958624

## Artefak mobile

- Float32 SHA-256: `e0e033191ff05abdab9fcb567009df94a8f1118fba00c17460097b83bfaa72c7`
- Float32 size: 414240 bytes
- Float16 SHA-256: `594a2ac65580106c5ccfc7b01bcd5fd6b31d7970158cd23dbfac475a477f38cf`
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
