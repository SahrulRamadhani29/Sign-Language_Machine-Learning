"""Generate the standalone Colab notebook from training_pipeline.py."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "training_pipeline.py"
NOTEBOOK_PATH = ROOT / "LatihIsyarat_Training_Colab.ipynb"


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
Hasil permanen disimpan ke Google Drive; dataset hanya berada di cache Colab."""
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

Checkpoint disimpan langsung ke Drive sehingga tidak hilang jika runtime Colab
terputus setelah suatu epoch selesai."""
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

- Biarkan `DATASET_DIR = None` agar dataset Kaggle diunduh otomatis.
- Set `QUICK_BASELINE_ONLY = True` untuk tes alur satu model.
- Set `False` untuk membandingkan tiga kandidat dan memilih validation loss
  terendah. Test set tidak ikut memilih model."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path

OUTPUT_BASE_DIR = Path("/content/drive/MyDrive/LatihIsyarat_training_outputs")
DATASET_DIR = None
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
print("Dataset      :", DATASET_DIR or "download otomatis dari Kaggle")
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

Ekstrak dan upload `sign_mnist_train.csv` serta `sign_mnist_test.csv` ke satu
folder Google Drive. Kemudian ubah `DATASET_DIR` pada konfigurasi, misalnya:

```python
DATASET_DIR = "/content/drive/MyDrive/datasets/sign-language-mnist"
```

Jangan menilai performa kamera hanya dari test accuracy dataset. Model terpilih
masih harus diuji dengan kamera HP, orientasi yang benar, ROI kosong, dan
kondisi tanpa tangan."""
        ),
    ]

    nbf.validate(notebook)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook dibuat: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()

