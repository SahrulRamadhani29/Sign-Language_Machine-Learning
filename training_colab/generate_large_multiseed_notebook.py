"""Generate the unified Colab notebook for the under-10-MiB experiment."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "LatihIsyarat_Experimental_Large_MultiSeed_Colab.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {
            "name": NOTEBOOK_PATH.name,
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
            """# Eksperimen Besar Multi-Seed - Maksimal 10 MiB

Satu notebook ini menjalankan **5 approach x 3 seed = 15 training**. Kandidat
berukuran sekitar 2,25 sampai 8,9 MiB. Model pemenang otomatis ditempatkan di
folder `MODEL_PEMENANG` dan dibuatkan satu ZIP siap download.

Target eksperimen adalah **official test accuracy >=99,5%**, tetapi target
tersebut tidak dijamin. Kandidat dipilih dari rata-rata validation loss
antar-seed; official test baru digunakan sesudah pemilihan.

Model stabil dan eksperimen 2,25 MiB sebelumnya tidak ditimpa. Aktifkan
**Runtime > Change runtime type > T4 GPU** sebelum memilih Run all."""
        ),
        nbf.v4.new_markdown_cell("## 1. Pasang dependensi"),
        nbf.v4.new_code_cell("%pip install -q kagglehub seaborn scikit-learn"),
        nbf.v4.new_markdown_cell("## 2. Hubungkan Google Drive"),
        nbf.v4.new_code_cell(
            """from google.colab import drive

drive.mount("/content/drive")"""
        ),
        nbf.v4.new_markdown_cell("## 3. Ambil source terbaru dari GitHub"),
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

from training_colab.experimental_large_multiseed import (
    DEFAULT_LARGE_CANDIDATES,
    LargeMultiSeedConfig,
    inspect_candidate_capacities,
    run_large_multiseed_experiment,
)

print("Source siap:", REPOSITORY_DIR)"""
        ),
        nbf.v4.new_markdown_cell("## 4. Pastikan GPU aktif"),
        nbf.v4.new_code_cell(
            """import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
if not tf.config.list_physical_devices("GPU"):
    raise RuntimeError("GPU belum aktif. Ubah runtime Colab ke T4 GPU.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 5. Konfigurasi tunggal

`RUN_NAME` sengaja tetap. Jika runtime Colab terputus, jalankan ulang notebook
dengan nama yang sama; checkpoint yang lengkap akan dilewati. Ganti nama ini
hanya ketika ingin memulai eksperimen baru dari nol."""
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
RUN_NAME = "large_under_10mb_multiseed_v1"

CONFIG = LargeMultiSeedConfig(
    seeds=(42, 123, 2026),
    split_seed=2026,
    validation_fraction=0.20,
    batch_size=64,
    max_epochs=70,
    early_stopping_patience=10,
    reduce_lr_patience=3,
    target_test_accuracy=0.995,
    max_float32_tflite_mib=10.0,
    tflite_parity_samples=512,
    export_float16=True,
)
CANDIDATES = DEFAULT_LARGE_CANDIDATES

print("Dataset       :", DATASET_DIR)
print("Output        :", OUTPUT_BASE_DIR / RUN_NAME)
print("Seed          :", CONFIG.seeds)
print("Approach      :", [item.name for item in CANDIDATES])
print("Total training:", len(CONFIG.seeds) * len(CANDIDATES))"""
        ),
        nbf.v4.new_markdown_cell("## 6. Cek kapasitas sebelum training"),
        nbf.v4.new_code_cell(
            """from IPython.display import display

capacity = inspect_candidate_capacities(CANDIDATES)
display(capacity.round(3))

if capacity["estimated_float32_mib"].max() >= CONFIG.max_float32_tflite_mib:
    raise RuntimeError("Ada kandidat yang diperkirakan melewati 10 MiB.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 7. Jalankan seluruh eksperimen

Cell ini menjalankan semuanya. Durasi sangat bergantung pada GPU dan early
stopping; siapkan kira-kira **2-5 jam pada T4**. Proses aman dilanjutkan ulang
karena checkpoint kandidat-seed yang sudah lengkap tidak dilatih kembali.

Jika kandidat 2,25 MiB menang, tiga seed dapat digabung menjadi satu ensemble
sekitar 6,75 MiB. Kandidat 3,5 MiB dapat memakai dua seed sekitar 7 MiB.
Kandidat yang lebih besar memakai satu checkpoint agar tetap di bawah 10 MiB."""
        ),
        nbf.v4.new_code_cell(
            """EXPERIMENT_DIR = run_large_multiseed_experiment(
    output_base_dir=OUTPUT_BASE_DIR,
    dataset_dir=DATASET_DIR,
    reference_model_path=STABLE_REFERENCE_MODEL,
    run_name=RUN_NAME,
    config=CONFIG,
    candidates=CANDIDATES,
    resume=True,
)

print("Eksperimen selesai:", EXPERIMENT_DIR)"""
        ),
        nbf.v4.new_markdown_cell("## 8. Tampilkan hasil yang penting saja"),
        nbf.v4.new_code_cell(
            """import json
import pandas as pd

WINNER_DIR = EXPERIMENT_DIR / "MODEL_PEMENANG"
comparison = pd.read_csv(EXPERIMENT_DIR / "PERBANDINGAN_UTAMA.csv")
evaluation = json.loads((WINNER_DIR / "evaluation.json").read_text())
status = json.loads((WINNER_DIR / "target_status.json").read_text())
selection = json.loads((WINNER_DIR / "selected_model.json").read_text())

display(comparison.round(6))
print()
print("Kandidat         :", selection["deployment"]["candidate"])
print("Mode             :", selection["deployment"]["mode"])
print("Seed anggota     :", selection["deployment"]["member_seeds"])
print("Test accuracy    :", evaluation["accuracy"] * 100, "%")
print("Macro-F1         :", evaluation["macro_f1"] * 100, "%")
print("Target 99.5%     :", status["target_achieved"])
print("Ukuran float32   :", status["actual_float32_tflite_mib"], "MiB")
print("Lolos batas 10 MB:", status["within_size_limit"])
print()
print("MODEL PEMENANG   :", WINNER_DIR / "model_float32.tflite")
print("ZIP DOWNLOAD     :", EXPERIMENT_DIR / "MODEL_PEMENANG_SIAP_DOWNLOAD.zip")
print("SEMUA HASIL      :", EXPERIMENT_DIR)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Yang dipakai di Flutter

Ambil `MODEL_PEMENANG_SIAP_DOWNLOAD.zip`, lalu gunakan:

- `model_float32.tflite`
- `labels.json`
- `model_metadata.json`

Tetap lakukan pengujian kamera, pencahayaan, mirror, crop tangan, latency, RAM,
dan suhu perangkat. Accuracy dataset dan accuracy kamera adalah dua hasil yang
berbeda."""
        ),
    ]
    nbf.validate(notebook)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook dibuat: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
