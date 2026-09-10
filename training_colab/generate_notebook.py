"""Generate the standalone Colab notebook from training_pipeline.py."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "training_pipeline.py"
NOTEBOOK_PATH = ROOT / "LatihIsyarat_Training_Colab.ipynb"
COMPARE_NOTEBOOK_PATH = ROOT / "LatihIsyarat_Compare_Seeds_Colab.ipynb"


def main() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    source = source.split('\nif __name__ == "__main__":', maxsplit=1)[0]

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {
            "name": "LatihIsyarat_Training_Colab.ipynb",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }

    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Training CNN LatihIsyarat - ASL 24 Huruf

Notebook ini siap dijalankan sendiri di Google Colab. Ia akan mengunduh dataset
`datamunge/sign-language-mnist`, melatih CNN dari bobot acak, memilih model
berdasarkan **validation loss**, mengevaluasi test set, dan mengekspor TFLite.

Sebelum mulai, pilih **Runtime > Change runtime type > T4 GPU** jika tersedia.
Dataset, checkpoint, model, dan laporan disimpan permanen ke Google Drive."""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. Pasang dependensi ringan

TensorFlow bawaan Colab dipakai apa adanya agar tidak merusak kompatibilitas
runtime. Versi aktual akan direkam di `environment.json`."""
        ),
        nbf.v4.new_code_cell(
            "%pip install -q kagglehub seaborn scikit-learn"
        ),
        nbf.v4.new_markdown_cell("## 2. Periksa runtime dan GPU"),
        nbf.v4.new_code_cell(
            """import sys
import tensorflow as tf

print("Python     :", sys.version)
print("TensorFlow :", tf.__version__)
print("GPU        :", tf.config.list_physical_devices("GPU"))
if not tf.config.list_physical_devices("GPU"):
    print("PERINGATAN: GPU tidak aktif. Training tetap bisa berjalan, tetapi lebih lambat.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 3. Hubungkan Google Drive

Dataset diunduh sekali lalu dua CSV yang diperlukan disimpan ke Drive. Checkpoint
juga ditulis langsung ke Drive agar tidak hilang jika runtime Colab terputus."""
        ),
        nbf.v4.new_code_cell(
            """from google.colab import drive

drive.mount("/content/drive")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. Definisi pipeline

Sel ini berasal dari `training_pipeline.py`. Tidak perlu diedit."""
        ),
        nbf.v4.new_code_cell(source),
        nbf.v4.new_markdown_cell(
            """## 5. Konfigurasi training

- `DATASET_DIR` adalah cache permanen di Drive. Run pertama mengunduh dataset;
  run berikutnya langsung menggunakan CSV yang sudah tersimpan.
- Set `QUICK_BASELINE_ONLY = True` untuk tes alur satu model.
- Set `False` untuk membandingkan tiga kandidat dan memilih validation loss
  terendah. Test set tidak ikut memilih model."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path

OUTPUT_BASE_DIR = Path("/content/drive/MyDrive/LatihIsyarat_training_outputs")
DATASET_DIR = Path("/content/drive/MyDrive/LatihIsyarat_datasets/sign-language-mnist")
QUICK_BASELINE_ONLY = False

TRAINING_CONFIG = TrainingConfig(
    seed=42,
    validation_fraction=0.20,
    batch_size=64,
    max_epochs=30,
    early_stopping_patience=5,
    reduce_lr_patience=2,
    reduce_lr_factor=0.5,
    min_learning_rate=1e-6,
    tflite_parity_samples=512,
    export_float16=True,
)

EXPERIMENTS = (
    (DEFAULT_EXPERIMENTS[0],)
    if QUICK_BASELINE_ONLY
    else DEFAULT_EXPERIMENTS
)

print("Output       :", OUTPUT_BASE_DIR)
print("Cache dataset:", DATASET_DIR)
print("Eksperimen   :", [experiment.name for experiment in EXPERIMENTS])"""
        ),
        nbf.v4.new_markdown_cell(
            """## 6. Jalankan training

Waktu bergantung pada GPU Colab. Jangan hentikan sel ketika file checkpoint
sedang ditulis. Setiap run menggunakan folder baru agar hasil lama tidak
tertimpa."""
        ),
        nbf.v4.new_code_cell(
            """RUN_DIR = run_training_pipeline(
    output_base_dir=OUTPUT_BASE_DIR,
    dataset_dir=DATASET_DIR,
    config=TRAINING_CONFIG,
    experiments=EXPERIMENTS,
)

print("\\nSemua hasil tersimpan di:", RUN_DIR)"""
        ),
        nbf.v4.new_markdown_cell("## 7. Lihat ringkasan hasil"),
        nbf.v4.new_code_cell(
            """from IPython.display import display

comparison = pd.read_csv(RUN_DIR / "experiment_comparison.csv")
evaluation = json.loads((RUN_DIR / "evaluation.json").read_text())
parity = json.loads((RUN_DIR / "tflite_parity.json").read_text())

display(comparison)
print("Model terpilih :", comparison.iloc[0]["name"])
print("Test accuracy  :", evaluation["accuracy"])
print("Test macro F1  :", evaluation["macro_f1"])
print("TFLite parity  :", parity)
print("Model Flutter  :", RUN_DIR / "model_float32.tflite")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Jika download Kaggle gagal

Unduh dataset dari:
https://www.kaggle.com/datasets/datamunge/sign-language-mnist

Jika download otomatis tetap gagal, ekstrak dan upload `sign_mnist_train.csv`
serta `sign_mnist_test.csv` ke folder cache berikut:

```python
DATASET_DIR = "/content/drive/MyDrive/LatihIsyarat_datasets/sign-language-mnist"
```

Jangan menilai performa kamera hanya dari test accuracy dataset. Model terpilih
masih harus diuji dengan kamera HP, orientasi yang benar, ROI kosong, dan
kondisi tanpa tangan."""
        ),
    ]

    nbf.validate(notebook)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook dibuat: {NOTEBOOK_PATH}")

    compare_notebook = nbf.v4.new_notebook()
    compare_notebook["metadata"] = {
        "colab": {
            "name": "LatihIsyarat_Compare_Seeds_Colab.ipynb",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    compare_notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Perbandingan Model LatihIsyarat Multi-Seed

Notebook ini hanya membaca hasil training seed 42, 123, dan 2026 dari Google
Drive. Notebook **tidak melakukan training dan tidak mengunduh dataset**.

Hasil yang dihitung: rata-rata, standar deviasi, rentang metrik, kesamaan
prediksi antarmodel, metrik per huruf, ukuran model, dan rekomendasi model
berdasarkan validation loss."""
        ),
        nbf.v4.new_markdown_cell("## 1. Pasang dependensi ringan"),
        nbf.v4.new_code_cell(
            "%pip install -q seaborn scikit-learn"
        ),
        nbf.v4.new_markdown_cell("## 2. Hubungkan Google Drive"),
        nbf.v4.new_code_cell(
            """from google.colab import drive

drive.mount("/content/drive")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 3. Definisi fungsi perbandingan

Sel ini berasal dari `training_pipeline.py` dan tidak perlu diedit."""
        ),
        nbf.v4.new_code_cell(source),
        nbf.v4.new_markdown_cell(
            """## 4. Folder hasil tiga seed

Path berikut sudah diisi berdasarkan run yang telah selesai. Ubah hanya jika
folder di Google Drive dipindahkan atau namanya berbeda."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path

SEED_RUNS = {
    42: Path(
        "/content/drive/MyDrive/LatihIsyarat_training_outputs/"
        "asl24_20260910_052717_utc"
    ),
    123: Path(
        "/content/drive/MyDrive/LatihIsyarat_training_outputs/"
        "asl24_20260910_054136_utc"
    ),
    2026: Path(
        "/content/drive/MyDrive/LatihIsyarat_training_outputs/"
        "asl24_20260910_054435_utc"
    ),
}

COMPARISON_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/LatihIsyarat_training_outputs/"
    "comparison_seed_42_123_2026"
)

for seed, run_dir in SEED_RUNS.items():
    print(f"Seed {seed}: {run_dir}")
print("Output perbandingan:", COMPARISON_OUTPUT_DIR)"""
        ),
        nbf.v4.new_markdown_cell("## 5. Bandingkan ketiga model"),
        nbf.v4.new_code_cell(
            """MULTI_SEED_RESULT = compare_seed_runs(
    SEED_RUNS,
    output_dir=COMPARISON_OUTPUT_DIR,
)

print("Perbandingan selesai.")"""
        ),
        nbf.v4.new_markdown_cell("## 6. Tampilkan hasil"),
        nbf.v4.new_code_cell(
            """from IPython.display import display

print("PERBANDINGAN UTAMA")
display(MULTI_SEED_RESULT["comparison"].round(6))

print("STATISTIK ANTAR-SEED")
display(MULTI_SEED_RESULT["statistics"].round(6))

print("KESEPAKATAN PREDIKSI ANTARMODEL")
display(MULTI_SEED_RESULT["pairwise"].round(6))

print("LIMA HURUF DENGAN F1 TERENDAH PER SEED")
worst_per_seed = (
    MULTI_SEED_RESULT["per_class"]
    .sort_values(["seed", "f1_score"])
    .groupby("seed", as_index=False)
    .head(5)
)
display(worst_per_seed.round(6))

print("Seed rekomendasi :", MULTI_SEED_RESULT["recommended_seed"])
print("Model rekomendasi:", MULTI_SEED_RESULT["recommended_model_path"])
print("Laporan tersimpan:", MULTI_SEED_RESULT["output_dir"])"""
        ),
        nbf.v4.new_markdown_cell(
            """## Hasil yang tersimpan di Drive

```text
MyDrive/LatihIsyarat_training_outputs/comparison_seed_42_123_2026/
├── multi_seed_comparison.csv
├── multi_seed_statistics.csv
├── pairwise_prediction_agreement.csv
├── multi_seed_per_class_metrics.csv
├── multi_seed_comparison.png
└── multi_seed_summary.json
```

Pemilihan model menggunakan validation loss, bukan test accuracy. Metrik test
tetap dilaporkan sebagai rata-rata dan variasi antar-seed agar tidak memilih
hasil acak yang kebetulan paling tinggi."""
        ),
    ]
    nbf.validate(compare_notebook)
    nbf.write(compare_notebook, COMPARE_NOTEBOOK_PATH)
    print(f"Notebook dibuat: {COMPARE_NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
