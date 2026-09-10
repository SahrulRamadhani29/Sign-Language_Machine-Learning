"""Generate the Colab notebook for the high-accuracy experiment."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "LatihIsyarat_Experimental_99_5_Colab.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {
            "name": "LatihIsyarat_Experimental_99_5_Colab.ipynb",
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
            """# Eksperimen CNN LatihIsyarat Target 99,5%

Notebook ini mencoba tiga CNN custom berkapasitas lebih tinggi, semuanya
dilatih dari bobot acak tanpa model pretrained.

**Target 99,5% adalah sasaran eksperimen, bukan jaminan.** Test set tidak
digunakan untuk training, early stopping, atau memilih kandidat. Model stabil
seed 2026 tetap aman dan tidak akan ditimpa.

Sebelum mulai, pilih **Runtime > Change runtime type > T4 GPU**."""
        ),
        nbf.v4.new_markdown_cell("## 1. Pasang dependensi"),
        nbf.v4.new_code_cell(
            "%pip install -q kagglehub seaborn scikit-learn"
        ),
        nbf.v4.new_markdown_cell("## 2. Hubungkan Google Drive"),
        nbf.v4.new_code_cell(
            """from google.colab import drive

drive.mount("/content/drive")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 3. Ambil source eksperimen terbaru

Notebook mengambil kode dari repositori GitHub proyek. Tidak ada dataset atau
model lama yang dihapus."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import subprocess
import sys
import tempfile

REPOSITORY_URL = "https://github.com/SahrulRamadhani29/Sign-Language_Machine-Learning.git"
REPOSITORY_DIR = Path(tempfile.mkdtemp(prefix="latihisyarat_repo_")) / "repository"

subprocess.run(
    ["git", "clone", "--depth", "1", REPOSITORY_URL, str(REPOSITORY_DIR)],
    check=True,
)
sys.path.insert(0, str(REPOSITORY_DIR))

from training_colab.experimental_high_accuracy import (
    DEFAULT_HIGH_ACCURACY_CANDIDATES,
    HighAccuracyConfig,
    build_high_accuracy_model,
    run_high_accuracy_experiment,
)

print("Source eksperimen siap:", REPOSITORY_DIR)"""
        ),
        nbf.v4.new_markdown_cell("## 4. Periksa TensorFlow dan GPU"),
        nbf.v4.new_code_cell(
            """import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
if not tf.config.list_physical_devices("GPU"):
    print("PERINGATAN: GPU tidak aktif; eksperimen besar akan sangat lambat.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 5. Konfigurasi eksperimen

Ketiga kandidat memakai split dan seed yang sama agar perbandingannya adil.
Output menggunakan folder baru di Drive."""
        ),
        nbf.v4.new_code_cell(
            """DATASET_DIR = Path(
    "/content/drive/MyDrive/LatihIsyarat_datasets/sign-language-mnist"
)
OUTPUT_BASE_DIR = Path(
    "/content/drive/MyDrive/LatihIsyarat_experimental_outputs"
)
STABLE_REFERENCE_MODEL = Path(
    "/content/drive/MyDrive/LatihIsyarat_training_outputs/"
    "asl24_20260910_054435_utc/model_float32.tflite"
)

EXPERIMENT_CONFIG = HighAccuracyConfig(
    seed=2026,
    validation_fraction=0.20,
    batch_size=64,
    max_epochs=80,
    early_stopping_patience=12,
    reduce_lr_patience=3,
    reduce_lr_factor=0.5,
    min_learning_rate=1e-7,
    target_test_accuracy=0.995,
    tflite_parity_samples=512,
    export_float16=True,
)

CANDIDATES = DEFAULT_HIGH_ACCURACY_CANDIDATES

print("Dataset        :", DATASET_DIR)
print("Output         :", OUTPUT_BASE_DIR)
print("Model stabil   :", STABLE_REFERENCE_MODEL)
print("Target         :", EXPERIMENT_CONFIG.target_test_accuracy * 100, "%")
print("Kandidat       :", [candidate.name for candidate in CANDIDATES])"""
        ),
        nbf.v4.new_markdown_cell(
            """## 6. Lihat perkiraan kapasitas dan ukuran bobot

Ukuran TFLite aktual baru diketahui setelah konversi. Nilai berikut hanya
perkiraan bobot float32 mentah dan belum termasuk metadata/operator."""
        ),
        nbf.v4.new_code_cell(
            """import pandas as pd
from IPython.display import display

capacity_rows = []

for candidate in CANDIDATES:
    tf.keras.backend.clear_session()
    model = build_high_accuracy_model(candidate)
    parameters = model.count_params()
    capacity_rows.append({
        "candidate": candidate.name,
        "parameters": parameters,
        "raw_float32_mib": parameters * 4 / 1024 / 1024,
        "raw_float16_mib": parameters * 2 / 1024 / 1024,
    })

capacity = pd.DataFrame(capacity_rows)
display(capacity.round(3))
tf.keras.backend.clear_session()"""
        ),
        nbf.v4.new_markdown_cell(
            """## 7. Jalankan turnamen training

Bagian ini melatih tiga kandidat berurutan. ModelCheckpoint menyimpan epoch
validation loss terbaik. Early stopping dapat menghentikan training sebelum 80
epoch. Jangan menutup runtime ketika checkpoint sedang ditulis."""
        ),
        nbf.v4.new_code_cell(
            """EXPERIMENT_RUN_DIR = run_high_accuracy_experiment(
    output_base_dir=OUTPUT_BASE_DIR,
    dataset_dir=DATASET_DIR,
    reference_model_path=STABLE_REFERENCE_MODEL,
    config=EXPERIMENT_CONFIG,
    candidates=CANDIDATES,
)

print("Eksperimen selesai:", EXPERIMENT_RUN_DIR)"""
        ),
        nbf.v4.new_markdown_cell("## 8. Tampilkan hasil akhir"),
        nbf.v4.new_code_cell(
            """from IPython.display import display
import json

candidate_comparison = pd.read_csv(
    EXPERIMENT_RUN_DIR / "candidate_comparison.csv"
)
evaluation = json.loads(
    (EXPERIMENT_RUN_DIR / "evaluation.json").read_text()
)
target_status = json.loads(
    (EXPERIMENT_RUN_DIR / "target_status.json").read_text()
)
metadata = json.loads(
    (EXPERIMENT_RUN_DIR / "experimental_model_metadata.json").read_text()
)

display(candidate_comparison)
print("Test accuracy :", evaluation["accuracy"] * 100, "%")
print("Macro-F1      :", evaluation["macro_f1"] * 100, "%")
print("Target 99.5%  :", target_status["target_achieved"])
print(
    "TFLite float32:",
    EXPERIMENT_RUN_DIR / "experimental_model_float32.tflite",
)
print(
    "Ukuran float32:",
    metadata["tflite_models"][0]["size_bytes"] / 1024 / 1024,
    "MiB",
)
print("Semua hasil   :", EXPERIMENT_RUN_DIR)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Aturan keputusan

- Jika target tercapai dan parity TFLite baik, model tetap harus diuji di HP.
- Jika target tidak tercapai, hasil tersebut tetap valid; jangan menambah
  eksperimen hanya dengan melihat test set yang sama.
- Bandingkan ukuran, latency, dan performa kamera terhadap model stabil seed
  2026 sebelum mengganti model utama.
- Validation/test accuracy tinggi tidak menyelesaikan prediksi palsu ketika
  kamera tidak melihat tangan."""
        ),
    ]
    nbf.validate(notebook)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook dibuat: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
