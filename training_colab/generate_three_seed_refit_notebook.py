"""Generate the Colab notebook for the three-seed full-data refit."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "LatihIsyarat_ThreeSeed_FullData_Refit_Colab.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {"name": NOTEBOOK_PATH.name, "provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Eksperimen Three-Seed + Full-Data Fine-Tune

Notebook ini melanjutkan hasil `large_under_10mb_multiseed_v1` tanpa mengulang
15 training sebelumnya.

1. Mempertahankan model pemenang dua seed sebagai baseline aman.
2. Menggabungkan checkpoint `wide_3_5mb` seed 42, 123, dan 2026.
3. Mengevaluasi ensemble tiga seed sebelum fine-tune.
4. Fine-tune setiap anggota selama 8 epoch pada seluruh 27.455 training rows.
5. Mengevaluasi ensemble setelah fine-tune.
6. Mengekspor tahap terbaik sebagai TFLite float16 di bawah 10 MiB.

Target official test accuracy tetap **>=99,5%**, tetapi tidak dijamin. Karena
ketiga tahap dibandingkan memakai official test, hasil ini bersifat eksperimen,
bukan benchmark test yang masih benar-benar untouched."""
        ),
        nbf.v4.new_markdown_cell("## 1. Pasang dependensi"),
        nbf.v4.new_code_cell("%pip install -q kagglehub seaborn scikit-learn"),
        nbf.v4.new_markdown_cell("## 2. Hubungkan Google Drive"),
        nbf.v4.new_code_cell(
            """from google.colab import drive

drive.mount("/content/drive")"""
        ),
        nbf.v4.new_markdown_cell("## 3. Ambil kode terbaru"),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import subprocess
import sys
import tempfile

REPOSITORY_URL = "https://github.com/SahrulRamadhani29/Sign-Language_Machine-Learning.git"
REPOSITORY_DIR = Path(tempfile.mkdtemp(prefix="latihisyarat_refit_")) / "repository"
subprocess.run(
    ["git", "clone", "--depth", "1", REPOSITORY_URL, str(REPOSITORY_DIR)],
    check=True,
)
sys.path.insert(0, str(REPOSITORY_DIR))

from training_colab.experimental_three_seed_refit import (
    ThreeSeedRefitConfig,
    run_three_seed_refit_experiment,
)

print("Source siap:", REPOSITORY_DIR)"""
        ),
        nbf.v4.new_markdown_cell("## 4. Periksa GPU"),
        nbf.v4.new_code_cell(
            """import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
if not tf.config.list_physical_devices("GPU"):
    raise RuntimeError("GPU belum aktif. Pilih runtime T4 GPU.")"""
        ),
        nbf.v4.new_markdown_cell("## 5. Konfigurasi"),
        nbf.v4.new_code_cell(
            """SOURCE_RUN_DIR = Path(
    "/content/drive/MyDrive/LatihIsyarat_experimental_outputs/"
    "large_under_10mb_multiseed_v1"
)
DATASET_DIR = Path(
    "/content/drive/MyDrive/LatihIsyarat_datasets/sign-language-mnist"
)
OUTPUT_BASE_DIR = Path(
    "/content/drive/MyDrive/LatihIsyarat_experimental_outputs"
)
RUN_NAME = "wide_3seed_full_data_refit_v1"

CONFIG = ThreeSeedRefitConfig(
    candidate_name="wide_3_5mb",
    seeds=(42, 123, 2026),
    batch_size=64,
    fine_tune_epochs=8,
    fine_tune_learning_rate=5e-5,
    final_learning_rate_ratio=0.05,
    target_test_accuracy=0.995,
    max_tflite_size_mib=10.0,
    parity_samples=512,
)

print("Source checkpoint:", SOURCE_RUN_DIR)
print("Output           :", OUTPUT_BASE_DIR / RUN_NAME)
print("Fine-tune epoch  :", CONFIG.fine_tune_epochs)
print("Seed ensemble    :", CONFIG.seeds)"""
        ),
        nbf.v4.new_markdown_cell(
            """## 6. Jalankan eksperimen

Perkiraan waktu sekitar **10-30 menit pada T4**, tergantung runtime. Jika
terputus, jalankan ulang dengan `RUN_NAME` sama; anggota fine-tune yang sudah
selesai otomatis dilewati."""
        ),
        nbf.v4.new_code_cell(
            """EXPERIMENT_DIR = run_three_seed_refit_experiment(
    source_run_dir=SOURCE_RUN_DIR,
    dataset_dir=DATASET_DIR,
    output_base_dir=OUTPUT_BASE_DIR,
    run_name=RUN_NAME,
    config=CONFIG,
)

print("Eksperimen selesai:", EXPERIMENT_DIR)"""
        ),
        nbf.v4.new_markdown_cell("## 7. Tampilkan hasil akhir"),
        nbf.v4.new_code_cell(
            """import json
import pandas as pd

comparison = pd.read_csv(
    EXPERIMENT_DIR / "PERBANDINGAN_SEBELUM_SESUDAH.csv"
)
final_result = json.loads(
    (EXPERIMENT_DIR / "HASIL_AKHIR.json").read_text()
)

display(comparison.round(6))
print()
print("Tahap terpilih:", final_result["selected_stage"])
print(
    "Model final    :",
    EXPERIMENT_DIR / "MODEL_FINAL" / "model_float16.tflite",
)
print(
    "ZIP download   :",
    EXPERIMENT_DIR / "MODEL_FINAL_SIAP_DOWNLOAD.zip",
)
print("Semua hasil   :", EXPERIMENT_DIR)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Untuk Flutter

Download `MODEL_FINAL_SIAP_DOWNLOAD.zip`. Model menggunakan bobot float16 agar
ensemble tiga seed tetap di bawah 10 MiB. Tensor input dan output tetap
`float32`, jadi kontrak preprocessing Flutter tidak berubah:

- input `[1, 28, 28, 1]`
- grayscale `0..255`
- jangan membagi 255 lagi
- baca urutan kelas dari `labels.json`"""
        ),
    ]
    nbf.validate(notebook)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook dibuat: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
