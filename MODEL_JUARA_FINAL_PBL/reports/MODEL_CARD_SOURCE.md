# Model Final Eksperimen Three-Seed

- Tahap terpilih: after_full_data_fine_tune
- Seed ensemble: [42, 123, 2026]
- Test accuracy: 99.302844%
- Macro-F1: 99.332522%
- Ukuran TFLite: 5.268 MiB
- Target 99.5%: belum tercapai

Model memakai bobot float16, tetapi input dan output tetap float32.
Pemilihan before/after refit memakai test resmi dan bersifat eksploratif.
Performa kamera harus diuji terpisah.
