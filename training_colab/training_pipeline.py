"""End-to-end training pipeline for the LatihIsyarat ASL classifier.

The module is intentionally kept in one file so the same implementation can be
embedded in the standalone Google Colab notebook. TensorFlow is imported lazily
so dataset contract tests can run on machines without TensorFlow installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


CLASS_NAMES = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")
ORIGINAL_LABELS = tuple(i for i in range(26) if i not in (9, 25))
LABEL_TO_INDEX = {label: index for index, label in enumerate(ORIGINAL_LABELS)}
INDEX_TO_ORIGINAL_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
EXPECTED_PIXEL_COUNT = 28 * 28
DATASET_SLUG = "datamunge/sign-language-mnist"


@dataclass(frozen=True)
class ExperimentConfig:
    """One model configuration selected only with validation metrics."""

    name: str
    learning_rate: float = 0.001
    dropout: float = 0.3
    use_mild_augmentation: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Shared training and export settings."""

    seed: int = 42
    validation_fraction: float = 0.2
    batch_size: int = 64
    max_epochs: int = 30
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    tflite_parity_samples: int = 512
    export_float16: bool = True


DEFAULT_EXPERIMENTS = (
    ExperimentConfig(
        name="baseline",
        learning_rate=0.001,
        dropout=0.3,
        use_mild_augmentation=False,
    ),
    ExperimentConfig(
        name="mild_augmentation",
        learning_rate=0.001,
        dropout=0.3,
        use_mild_augmentation=True,
    ),
    ExperimentConfig(
        name="mild_augmentation_dropout_040",
        learning_rate=0.0005,
        dropout=0.4,
        use_mild_augmentation=True,
    ),
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_name(prefix: str = "asl24") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    return f"{prefix}_{timestamp}"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow tidak tersedia. Jalankan pipeline ini di Google Colab "
            "atau install TensorFlow yang kompatibel dengan Python Anda."
        ) from exc
    return tf


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf = _require_tensorflow()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Some TensorFlow/device combinations do not expose deterministic ops.
        pass


def find_dataset_csvs(dataset_dir: str | Path) -> tuple[Path, Path]:
    root = Path(dataset_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {root}")

    train_matches = sorted(root.rglob("sign_mnist_train.csv"))
    test_matches = sorted(root.rglob("sign_mnist_test.csv"))
    if not train_matches or not test_matches:
        raise FileNotFoundError(
            "CSV dataset tidak lengkap. Diperlukan sign_mnist_train.csv dan "
            f"sign_mnist_test.csv di bawah {root}."
        )
    return train_matches[0], test_matches[0]


def download_dataset(dataset_slug: str = DATASET_SLUG) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Paket kagglehub belum tersedia. Jalankan `%pip install -q kagglehub` "
            "di Colab lalu ulangi sel ini."
        ) from exc

    print(f"Mengunduh dataset Kaggle: {dataset_slug}")
    downloaded_path = Path(kagglehub.dataset_download(dataset_slug))
    print(f"Dataset tersedia di: {downloaded_path}")
    return downloaded_path


def cache_dataset(
    dataset_dir: str | Path,
    dataset_slug: str = DATASET_SLUG,
) -> tuple[Path, Path]:
    """Reuse Drive CSVs or download once and copy only the required files."""

    cache_root = Path(dataset_dir).expanduser().resolve()
    try:
        train_csv, test_csv = find_dataset_csvs(cache_root)
        print(f"Dataset ditemukan di cache permanen: {cache_root}")
        print("Download Kaggle dilewati.")
        return train_csv, test_csv
    except FileNotFoundError:
        pass

    print(f"Cache dataset belum lengkap: {cache_root}")
    downloaded_root = download_dataset(dataset_slug)
    source_train, source_test = find_dataset_csvs(downloaded_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    for source_path in (source_train, source_test):
        destination = cache_root / source_path.name
        partial_destination = destination.with_suffix(destination.suffix + ".partial")
        print(f"Menyimpan permanen ke Google Drive: {destination}")
        shutil.copy2(source_path, partial_destination)
        partial_destination.replace(destination)

    save_json(
        cache_root / "dataset_source.json",
        {
            "dataset_slug": dataset_slug,
            "source_url": f"https://www.kaggle.com/datasets/{dataset_slug}",
            "cached_at_utc": utc_now_iso(),
            "files": ["sign_mnist_train.csv", "sign_mnist_test.csv"],
            "note": "Only the two required CSV files are persisted.",
        },
    )
    print("Dataset berhasil disimpan permanen. Run berikutnya tidak perlu download.")
    return find_dataset_csvs(cache_root)


def resolve_dataset_paths(
    dataset_dir: str | Path | None = None,
    dataset_slug: str = DATASET_SLUG,
) -> tuple[Path, Path]:
    if dataset_dir:
        return cache_dataset(dataset_dir, dataset_slug)
    return find_dataset_csvs(download_dataset(dataset_slug))


def remap_original_labels(original_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(original_labels, dtype=np.int64)
    invalid = sorted(set(labels.tolist()) - set(ORIGINAL_LABELS))
    if invalid:
        raise ValueError(
            "Dataset memuat label di luar kontrak 24 kelas ASL: "
            f"{invalid}. Label J=9 dan Z=25 tidak boleh ada."
        )

    lookup = np.full(26, -1, dtype=np.int64)
    for original, mapped in LABEL_TO_INDEX.items():
        lookup[original] = mapped
    return lookup[labels]


def load_sign_mnist_csv(path: str | Path, split_name: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV {split_name} tidak ditemukan: {csv_path}")

    frame = pd.read_csv(csv_path)
    if "label" not in frame.columns:
        raise ValueError(f"CSV {split_name} tidak mempunyai kolom 'label'.")
    if len(frame.columns) != EXPECTED_PIXEL_COUNT + 1:
        raise ValueError(
            f"CSV {split_name} mempunyai {len(frame.columns)} kolom; "
            f"seharusnya 785 (1 label + {EXPECTED_PIXEL_COUNT} piksel)."
        )
    if frame.isnull().to_numpy().any():
        raise ValueError(f"CSV {split_name} mengandung nilai kosong.")

    original_y = frame["label"].to_numpy(dtype=np.int64, copy=True)
    pixel_frame = frame.drop(columns=["label"])
    pixel_values = pixel_frame.to_numpy(dtype=np.float32, copy=True)

    if not np.isfinite(pixel_values).all():
        raise ValueError(f"CSV {split_name} mengandung nilai piksel non-finite.")
    pixel_min = float(pixel_values.min())
    pixel_max = float(pixel_values.max())
    if pixel_min < 0 or pixel_max > 255:
        raise ValueError(
            f"Rentang piksel {split_name} adalah {pixel_min}..{pixel_max}; "
            "kontrak mengharuskan 0..255."
        )
    if not np.all(pixel_values == np.floor(pixel_values)):
        raise ValueError(f"CSV {split_name} mengandung piksel non-integer.")

    y = remap_original_labels(original_y)
    x = pixel_values.reshape((-1, 28, 28, 1)).astype(np.float32, copy=False)
    original_counts = Counter(int(value) for value in original_y)
    mapped_counts = Counter(int(value) for value in y)

    missing_original_labels = sorted(set(ORIGINAL_LABELS) - set(original_counts))
    if missing_original_labels:
        raise ValueError(
            f"CSV {split_name} kehilangan kelas asli: {missing_original_labels}."
        )

    report = {
        "path": str(csv_path),
        "sha256": sha256_file(csv_path),
        "rows": int(len(frame)),
        "pixel_columns": EXPECTED_PIXEL_COUNT,
        "pixel_min": pixel_min,
        "pixel_max": pixel_max,
        "original_label_counts": {
            str(label): int(original_counts[label]) for label in ORIGINAL_LABELS
        },
        "mapped_label_counts": {
            CLASS_NAMES[index]: int(mapped_counts[index])
            for index in range(len(CLASS_NAMES))
        },
    }
    return x, y, report


def stratified_train_validation_split(
    x: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction harus berada di antara 0 dan 1.")
    return train_test_split(
        x,
        y,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )


def _image_fingerprints(images: np.ndarray) -> list[str]:
    flattened = np.asarray(images, dtype=np.uint8).reshape((len(images), -1))
    return [
        hashlib.blake2b(row.tobytes(), digest_size=16).hexdigest()
        for row in flattened
    ]


def audit_exact_duplicates(
    x_train_builtin: np.ndarray,
    y_train_builtin: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    print("Memeriksa duplikat gambar identik (bukan kemiripan visual)...")
    train_hashes = _image_fingerprints(x_train_builtin)
    test_hashes = _image_fingerprints(x_test)
    train_counts = Counter(train_hashes)
    test_counts = Counter(test_hashes)

    labels_by_hash: dict[str, set[int]] = defaultdict(set)
    for fingerprint, label in zip(train_hashes, y_train_builtin):
        labels_by_hash[fingerprint].add(int(label))
    for fingerprint, label in zip(test_hashes, y_test):
        labels_by_hash[fingerprint].add(int(label))

    conflict_hashes = [
        fingerprint
        for fingerprint, labels in labels_by_hash.items()
        if len(labels) > 1
    ]
    overlap = set(train_counts).intersection(test_counts)
    return {
        "method": "BLAKE2b fingerprint of exact uint8 pixel bytes",
        "limitation": (
            "Audit hanya mendeteksi gambar identik. Identitas sumber dan turunan "
            "augmentasi yang mirip tidak tersedia dari CSV."
        ),
        "train_duplicate_rows_beyond_first": int(
            sum(count - 1 for count in train_counts.values() if count > 1)
        ),
        "test_duplicate_rows_beyond_first": int(
            sum(count - 1 for count in test_counts.values() if count > 1)
        ),
        "cross_split_unique_image_overlap": int(len(overlap)),
        "cross_split_train_rows_in_overlap": int(
            sum(train_counts[fingerprint] for fingerprint in overlap)
        ),
        "cross_split_test_rows_in_overlap": int(
            sum(test_counts[fingerprint] for fingerprint in overlap)
        ),
        "identical_images_with_conflicting_labels": int(len(conflict_hashes)),
    }


def _class_distribution(y: np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in y)
    return {
        CLASS_NAMES[index]: int(counts[index])
        for index in range(len(CLASS_NAMES))
    }


def build_model(learning_rate: float, dropout: float):
    tf = _require_tensorflow()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1), name="image_0_255"),
            tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_0_1"),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_scores"),
        ],
        name="latihisyarat_asl24_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_mild_augmenter(seed: int):
    tf = _require_tensorflow()
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
                fill_mode="nearest",
                seed=seed,
            ),
            tf.keras.layers.RandomRotation(
                factor=0.03,
                fill_mode="nearest",
                seed=seed + 1,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.05, 0.05),
                width_factor=(-0.05, 0.05),
                fill_mode="nearest",
                seed=seed + 2,
            ),
        ],
        name="mild_training_only_augmentation",
    )


def make_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    training: bool,
    seed: int,
    use_mild_augmentation: bool = False,
):
    tf = _require_tensorflow()
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(x),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.batch(batch_size)

    if training and use_mild_augmentation:
        augmenter = build_mild_augmenter(seed)

        def augment_batch(images, labels):
            return augmenter(images, training=True), labels

        dataset = dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.prefetch(tf.data.AUTOTUNE)


def plot_training_history(history_frame: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    epochs = np.arange(1, len(history_frame) + 1)

    axes[0].plot(epochs, history_frame["loss"], label="training")
    axes[0].plot(epochs, history_frame["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history_frame["accuracy"], label="training")
    axes[1].plot(epochs, history_frame["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_dataset_samples(
    images: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(4, 6, figsize=(11, 8))
    for class_index, axis in enumerate(axes.flat):
        candidates = np.flatnonzero(labels == class_index)
        selected_index = int(rng.choice(candidates))
        axis.imshow(images[selected_index, :, :, 0], cmap="gray", vmin=0, vmax=255)
        axis.set_title(CLASS_NAMES[class_index])
        axis.axis("off")
    figure.suptitle("Satu Sampel Training per Kelas", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train_experiment(
    experiment: ExperimentConfig,
    config: TrainingConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    experiment_dir: Path,
) -> dict[str, Any]:
    tf = _require_tensorflow()
    tf.keras.backend.clear_session()
    set_reproducibility(config.seed)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        learning_rate=experiment.learning_rate,
        dropout=experiment.dropout,
    )
    checkpoint_path = experiment_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stopping_patience,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_learning_rate,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(experiment_dir / "epoch_log.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    train_dataset = make_tf_dataset(
        x_train,
        y_train,
        batch_size=config.batch_size,
        training=True,
        seed=config.seed,
        use_mild_augmentation=experiment.use_mild_augmentation,
    )
    validation_dataset = make_tf_dataset(
        x_validation,
        y_validation,
        batch_size=config.batch_size,
        training=False,
        seed=config.seed,
    )

    print(f"\n=== Eksperimen: {experiment.name} ===")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.max_epochs,
        callbacks=callbacks,
        verbose=1,
    )
    history_frame = pd.DataFrame(history.history)
    history_frame.index = np.arange(1, len(history_frame) + 1)
    history_frame.index.name = "epoch"
    history_frame.to_csv(experiment_dir / "training_history.csv")
    plot_training_history(history_frame, experiment_dir / "training_curves.png")

    best_row_index = int(history_frame["val_loss"].to_numpy().argmin())
    summary = {
        "experiment": asdict(experiment),
        "epochs_completed": int(len(history_frame)),
        "best_epoch": best_row_index + 1,
        "best_val_loss": float(history_frame.iloc[best_row_index]["val_loss"]),
        "best_val_accuracy": float(history_frame.iloc[best_row_index]["val_accuracy"]),
        "checkpoint_path": str(checkpoint_path),
    }
    save_json(experiment_dir / "experiment_summary.json", summary)
    return summary


def select_best_experiment(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Tidak ada hasil eksperimen untuk dipilih.")
    # Test set is deliberately excluded from model selection.
    return min(
        results,
        key=lambda item: (item["best_val_loss"], -item["best_val_accuracy"]),
    )


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=False,
        ax=axis,
    )
    axis.set_xlabel("Prediksi")
    axis.set_ylabel("Label benar")
    axis.set_title("Confusion Matrix - Test Bawaan Sign Language MNIST")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate_selected_model(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    run_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    probabilities = model.predict(x_test, batch_size=batch_size, verbose=1)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    top_two_margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]

    report = classification_report(
        y_test,
        predicted,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_test,
        predicted,
        labels=np.arange(len(CLASS_NAMES)),
    )
    plot_confusion_matrix(matrix, run_dir / "confusion_matrix.png")

    predictions_frame = pd.DataFrame(
        {
            "true_index": y_test,
            "true_label": [CLASS_NAMES[index] for index in y_test],
            "predicted_index": predicted,
            "predicted_label": [CLASS_NAMES[index] for index in predicted],
            "confidence": confidence,
            "top_two_margin": top_two_margin,
            "correct": predicted == y_test,
        }
    )
    predictions_frame.to_csv(run_dir / "test_predictions.csv", index=False)

    evaluation = {
        "evaluated_at_utc": utc_now_iso(),
        "selection_rule": (
            "Eksperimen dipilih dengan validation loss terendah; test bawaan "
            "baru dievaluasi setelah pemilihan."
        ),
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_precision": float(
            precision_score(y_test, predicted, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, predicted, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_test, predicted, average="macro", zero_division=0)
        ),
        "classification_report": report,
        "limitations": [
            "Metrik ini hanya berasal dari test bawaan Sign Language MNIST.",
            "Metrik ini bukan bukti performa kamera HP atau kondisi tanpa tangan.",
            "Dataset dapat memuat turunan augmentasi dari sumber yang terbatas.",
        ],
    }
    save_json(run_dir / "evaluation.json", evaluation)
    return evaluation, probabilities


def export_tflite(model, output_path: Path, float16: bool = False) -> dict[str, Any]:
    tf = _require_tensorflow()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()
    output_path.write_bytes(tflite_bytes)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "size_bytes": int(output_path.stat().st_size),
        "quantization": "float16_weights" if float16 else "float32",
    }


def _tensor_detail_for_json(detail: dict[str, Any]) -> dict[str, Any]:
    quantization_parameters = detail.get("quantization_parameters", {})
    return {
        "name": detail.get("name"),
        "shape": np.asarray(detail.get("shape", [])).astype(int).tolist(),
        "shape_signature": np.asarray(
            detail.get("shape_signature", detail.get("shape", []))
        ).astype(int).tolist(),
        "dtype": np.dtype(detail["dtype"]).name,
        "quantization": list(detail.get("quantization", (0.0, 0))),
        "scales": np.asarray(
            quantization_parameters.get("scales", [])
        ).astype(float).tolist(),
        "zero_points": np.asarray(
            quantization_parameters.get("zero_points", [])
        ).astype(int).tolist(),
    }


def compare_keras_and_tflite(
    model,
    tflite_path: Path,
    x_samples: np.ndarray,
    variant_name: str,
) -> dict[str, Any]:
    tf = _require_tensorflow()
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    requested_shape = [len(x_samples), 28, 28, 1]
    interpreter.resize_tensor_input(input_detail["index"], requested_shape, strict=False)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    tflite_input = x_samples.astype(input_detail["dtype"], copy=False)
    interpreter.set_tensor(input_detail["index"], tflite_input)
    interpreter.invoke()
    tflite_probabilities = interpreter.get_tensor(output_detail["index"])
    keras_probabilities = model.predict(x_samples, verbose=0)

    keras_labels = keras_probabilities.argmax(axis=1)
    tflite_labels = tflite_probabilities.argmax(axis=1)
    absolute_difference = np.abs(keras_probabilities - tflite_probabilities)
    return {
        "variant": variant_name,
        "samples": int(len(x_samples)),
        "label_agreement": float(np.mean(keras_labels == tflite_labels)),
        "maximum_absolute_score_difference": float(absolute_difference.max()),
        "mean_absolute_score_difference": float(absolute_difference.mean()),
        "input_tensor": _tensor_detail_for_json(input_detail),
        "output_tensor": _tensor_detail_for_json(output_detail),
    }


def save_environment_report(run_dir: Path) -> dict[str, Any]:
    tf = _require_tensorflow()
    report = {
        "created_at_utc": utc_now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
    }
    save_json(run_dir / "environment.json", report)
    return report


def write_model_card(
    run_dir: Path,
    selected: dict[str, Any],
    evaluation: dict[str, Any],
    float32_export: dict[str, Any],
    float16_export: dict[str, Any] | None,
    dataset_report: dict[str, Any],
) -> None:
    lines = [
        "# Model Card - LatihIsyarat ASL 24",
        "",
        f"Dibuat: {utc_now_iso()}",
        "",
        "## Ringkasan",
        "",
        "CNN klasifikasi 24 huruf statis ASL (tanpa J dan Z), dilatih dari bobot acak",
        "menggunakan Sign Language MNIST. Model menerima tensor grayscale 28x28x1",
        "float32 bernilai 0-255. Normalisasi 1/255 berada di dalam model.",
        "",
        "## Model terpilih",
        "",
        f"- Eksperimen: `{selected['experiment']['name']}`",
        f"- Epoch terbaik: {selected['best_epoch']}",
        f"- Validation loss terbaik: {selected['best_val_loss']:.6f}",
        f"- Validation accuracy pada epoch tersebut: {selected['best_val_accuracy']:.6f}",
        "- Aturan pemilihan: validation loss terendah, tanpa melihat test set.",
        "",
        "## Evaluasi test bawaan",
        "",
        f"- Accuracy: {evaluation['accuracy']:.6f}",
        f"- Macro precision: {evaluation['macro_precision']:.6f}",
        f"- Macro recall: {evaluation['macro_recall']:.6f}",
        f"- Macro F1: {evaluation['macro_f1']:.6f}",
        "",
        "## Artefak mobile",
        "",
        f"- Float32 SHA-256: `{float32_export['sha256']}`",
        f"- Float32 size: {float32_export['size_bytes']} bytes",
    ]
    if float16_export:
        lines.extend(
            [
                f"- Float16 SHA-256: `{float16_export['sha256']}`",
                f"- Float16 size: {float16_export['size_bytes']} bytes",
            ]
        )
    lines.extend(
        [
            "",
            "## Dataset",
            "",
            f"- Sumber: https://www.kaggle.com/datasets/{DATASET_SLUG}",
            f"- Training bawaan: {dataset_report['train_builtin']['rows']} baris",
            f"- Test bawaan: {dataset_report['test_builtin']['rows']} baris",
            "- J dan Z tidak tersedia pada dataset dan tidak didukung model.",
            "",
            "## Batasan penting",
            "",
            "- Hasil test dataset tidak sama dengan performa kamera HP.",
            "- Model tertutup 24 kelas dapat memberi skor tinggi pada gambar tanpa tangan.",
            "- Model belum mengenali gerakan, kalimat, BISINDO, atau SIBI.",
            "- Orientasi, mirror, ROI, dan preprocessing Flutter wajib diverifikasi.",
            "- Lakukan pengujian kamera nyata sebelum membuat klaim penggunaan.",
            "",
        ]
    )
    (run_dir / "model_card.md").write_text("\n".join(lines), encoding="utf-8")


def write_run_manifest(run_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.csv":
            rows.append(
                {
                    "relative_path": path.relative_to(run_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    with (run_dir / "artifact_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_training_pipeline(
    output_base_dir: str | Path,
    dataset_dir: str | Path | None = None,
    run_name: str | None = None,
    config: TrainingConfig | None = None,
    experiments: Sequence[ExperimentConfig] | None = None,
) -> Path:
    """Train candidates, select by validation loss, evaluate once, and export."""

    tf = _require_tensorflow()
    config = config or TrainingConfig()
    experiments = tuple(experiments or DEFAULT_EXPERIMENTS)
    if not experiments:
        raise ValueError("Minimal satu eksperimen harus tersedia.")
    names = [experiment.name for experiment in experiments]
    if len(names) != len(set(names)):
        raise ValueError("Nama eksperimen harus unik.")

    set_reproducibility(config.seed)
    run_dir = Path(output_base_dir).expanduser() / (run_name or make_run_name())
    if run_dir.exists():
        raise FileExistsError(
            f"Folder run sudah ada: {run_dir}. Gunakan run_name baru agar hasil lama aman."
        )
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {[device.name for device in tf.config.list_physical_devices('GPU')]}")
    print(f"Output permanen: {run_dir}")
    environment = save_environment_report(run_dir)
    save_json(
        run_dir / "run_config.json",
        {
            "training": asdict(config),
            "experiments": [asdict(experiment) for experiment in experiments],
            "selection_metric": "minimum validation loss",
            "test_used_for_selection": False,
        },
    )

    train_csv, test_csv = resolve_dataset_paths(dataset_dir)
    print(f"Training CSV: {train_csv}")
    print(f"Test CSV: {test_csv}")
    x_train_builtin, y_train_builtin, train_report = load_sign_mnist_csv(
        train_csv, "training bawaan"
    )
    x_test, y_test, test_report = load_sign_mnist_csv(test_csv, "test bawaan")
    duplicate_report = audit_exact_duplicates(
        x_train_builtin,
        y_train_builtin,
        x_test,
        y_test,
    )

    x_train, x_validation, y_train, y_validation = stratified_train_validation_split(
        x_train_builtin,
        y_train_builtin,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    dataset_report = {
        "source": f"https://www.kaggle.com/datasets/{DATASET_SLUG}",
        "train_builtin": train_report,
        "test_builtin": test_report,
        "split": {
            "seed": config.seed,
            "validation_fraction": config.validation_fraction,
            "training_rows": int(len(y_train)),
            "validation_rows": int(len(y_validation)),
            "test_rows": int(len(y_test)),
            "training_distribution": _class_distribution(y_train),
            "validation_distribution": _class_distribution(y_validation),
            "test_distribution": _class_distribution(y_test),
        },
        "exact_duplicate_audit": duplicate_report,
    }
    save_json(run_dir / "dataset_report.json", dataset_report)
    plot_dataset_samples(
        x_train,
        y_train,
        run_dir / "dataset_samples.png",
        seed=config.seed,
    )
    print(
        "Split: "
        f"train={len(y_train)}, validation={len(y_validation)}, test={len(y_test)}"
    )

    experiment_results = []
    for experiment in experiments:
        result = train_experiment(
            experiment=experiment,
            config=config,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            experiment_dir=run_dir / "experiments" / experiment.name,
        )
        experiment_results.append(result)

    comparison = pd.DataFrame(
        [
            {
                "name": item["experiment"]["name"],
                "learning_rate": item["experiment"]["learning_rate"],
                "dropout": item["experiment"]["dropout"],
                "use_mild_augmentation": item["experiment"]["use_mild_augmentation"],
                "epochs_completed": item["epochs_completed"],
                "best_epoch": item["best_epoch"],
                "best_val_loss": item["best_val_loss"],
                "best_val_accuracy": item["best_val_accuracy"],
            }
            for item in experiment_results
        ]
    ).sort_values(["best_val_loss", "best_val_accuracy"], ascending=[True, False])
    comparison.to_csv(run_dir / "experiment_comparison.csv", index=False)

    selected = select_best_experiment(experiment_results)
    selected_checkpoint = Path(selected["checkpoint_path"])
    best_keras_path = run_dir / "best_model.keras"
    shutil.copy2(selected_checkpoint, best_keras_path)
    selected_experiment_dir = selected_checkpoint.parent
    shutil.copy2(
        selected_experiment_dir / "training_history.csv",
        run_dir / "training_history.csv",
    )
    shutil.copy2(
        selected_experiment_dir / "training_curves.png",
        run_dir / "training_curves.png",
    )
    save_json(run_dir / "selected_experiment.json", selected)
    print(
        f"Model terpilih: {selected['experiment']['name']} "
        f"(val_loss={selected['best_val_loss']:.6f})"
    )

    best_model = tf.keras.models.load_model(best_keras_path)
    evaluation, _ = evaluate_selected_model(
        best_model,
        x_test,
        y_test,
        batch_size=config.batch_size,
        run_dir=run_dir,
    )

    labels_payload = {
        "schema_version": 1,
        "labels": list(CLASS_NAMES),
        "output_index_to_label": {
            str(index): label for index, label in enumerate(CLASS_NAMES)
        },
        "output_index_to_original_dataset_label": {
            str(index): INDEX_TO_ORIGINAL_LABEL[index]
            for index in range(len(CLASS_NAMES))
        },
        "unsupported_dynamic_letters": ["J", "Z"],
    }
    save_json(run_dir / "labels.json", labels_payload)

    float32_path = run_dir / "model_float32.tflite"
    float32_export = export_tflite(best_model, float32_path, float16=False)
    float16_export = None
    if config.export_float16:
        float16_path = run_dir / "model_float16.tflite"
        float16_export = export_tflite(best_model, float16_path, float16=True)

    sample_count = min(config.tflite_parity_samples, len(x_test))
    rng = np.random.default_rng(config.seed)
    sample_indices = rng.choice(len(x_test), size=sample_count, replace=False)
    parity_reports = [
        compare_keras_and_tflite(
            best_model,
            float32_path,
            x_test[sample_indices],
            "float32",
        )
    ]
    if float16_export:
        parity_reports.append(
            compare_keras_and_tflite(
                best_model,
                Path(float16_export["path"]),
                x_test[sample_indices],
                "float16_weights",
            )
        )
    save_json(run_dir / "tflite_parity.json", parity_reports)

    model_metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now_iso(),
        "model_name": "LatihIsyarat ASL 24 CNN",
        "model_version": run_dir.name,
        "selected_experiment": selected,
        "keras_model": {
            "path": best_keras_path.name,
            "sha256": sha256_file(best_keras_path),
            "size_bytes": best_keras_path.stat().st_size,
        },
        "tflite_models": [
            export
            for export in (float32_export, float16_export)
            if export is not None
        ],
        "input_contract": {
            "shape": [1, 28, 28, 1],
            "dtype": "float32",
            "value_range_before_model": [0.0, 255.0],
            "color_space": "grayscale",
            "normalization": "Rescaling(1/255) inside the model",
            "mobile_must_divide_by_255": False,
            "resize": "bilinear to 28x28; verify Python and Flutter parity",
            "grayscale_formula": "0.299*R + 0.587*G + 0.114*B",
            "orientation_and_mirror": (
                "Must be established with reference images before camera release"
            ),
        },
        "output_contract": {
            "shape": [1, 24],
            "dtype": "float32",
            "activation": "softmax",
            "labels": list(CLASS_NAMES),
        },
        "initial_app_policy_not_calibrated": {
            "score_threshold": 0.8,
            "top_two_margin": 0.15,
            "history_size": 5,
            "minimum_matching_predictions": 3,
            "warning": (
                "These are starting values from the concept, not validated results."
            ),
        },
        "dataset": {
            "source": dataset_report["source"],
            "train_csv_sha256": train_report["sha256"],
            "test_csv_sha256": test_report["sha256"],
        },
        "test_metrics": {
            key: evaluation[key]
            for key in (
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
            )
        },
        "runtime": environment,
        "known_limitations": evaluation["limitations"],
    }
    save_json(run_dir / "model_metadata.json", model_metadata)
    write_model_card(
        run_dir,
        selected,
        evaluation,
        float32_export,
        float16_export,
        dataset_report,
    )
    write_run_manifest(run_dir)

    print("\nTraining selesai.")
    print(f"Model Keras terbaik : {best_keras_path}")
    print(f"Model TFLite utama  : {float32_path}")
    print(f"Evaluasi            : {run_dir / 'evaluation.json'}")
    print(f"Model card           : {run_dir / 'model_card.md'}")
    return run_dir


def load_json_config(path: str | Path) -> tuple[TrainingConfig, tuple[ExperimentConfig, ...]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = TrainingConfig(**payload.get("training", {}))
    experiments = tuple(
        ExperimentConfig(**experiment)
        for experiment in payload.get("experiments", [])
    )
    return config, experiments or DEFAULT_EXPERIMENTS


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder induk untuk semua run, sebaiknya berada di Google Drive.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Folder CSV lokal. Jika kosong, dataset diunduh melalui kagglehub.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config", default=None, help="Path konfigurasi JSON.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.config:
        config, experiments = load_json_config(args.config)
    else:
        config, experiments = TrainingConfig(), DEFAULT_EXPERIMENTS
    return run_training_pipeline(
        output_base_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        run_name=args.run_name,
        config=config,
        experiments=experiments,
    )


if __name__ == "__main__":
    main()
