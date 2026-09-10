"""Experimental high-capacity CNN tournament for Sign Language MNIST.

This experiment aims for >=99.5% test accuracy without training on the test
set. The target is aspirational, not guaranteed. The stable seed-2026 baseline
is never overwritten because all outputs use a separate directory.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from training_colab import training_pipeline as base
except ModuleNotFoundError:
    import training_pipeline as base


@dataclass(frozen=True)
class HighAccuracyCandidate:
    name: str
    filters: tuple[int, int, int]
    dense_units: int
    learning_rate: float
    block_dropouts: tuple[float, float, float]
    dense_dropout: float
    weight_decay: float = 1e-5


@dataclass(frozen=True)
class HighAccuracyConfig:
    seed: int = 2026
    validation_fraction: float = 0.2
    batch_size: int = 64
    max_epochs: int = 80
    early_stopping_patience: int = 12
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-7
    target_test_accuracy: float = 0.995
    tflite_parity_samples: int = 512
    export_float16: bool = True


DEFAULT_HIGH_ACCURACY_CANDIDATES = (
    HighAccuracyCandidate(
        name="spatial_bn_medium",
        filters=(32, 64, 128),
        dense_units=256,
        learning_rate=0.001,
        block_dropouts=(0.05, 0.10, 0.15),
        dense_dropout=0.25,
    ),
    HighAccuracyCandidate(
        name="spatial_bn_large",
        filters=(64, 128, 256),
        dense_units=512,
        learning_rate=0.001,
        block_dropouts=(0.05, 0.10, 0.15),
        dense_dropout=0.25,
    ),
    HighAccuracyCandidate(
        name="spatial_bn_large_low_dropout",
        filters=(64, 128, 256),
        dense_units=512,
        learning_rate=0.0005,
        block_dropouts=(0.00, 0.05, 0.10),
        dense_dropout=0.15,
    ),
)


def build_high_accuracy_model(candidate: HighAccuracyCandidate):
    tf = base._require_tensorflow()
    regularizer = tf.keras.regularizers.L2(candidate.weight_decay)
    inputs = tf.keras.layers.Input(shape=(28, 28, 1), name="image_0_255")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(inputs)

    for block_index, (filters, dropout) in enumerate(
        zip(candidate.filters, candidate.block_dropouts),
        start=1,
    ):
        for convolution_index in (1, 2):
            x = tf.keras.layers.Conv2D(
                filters,
                kernel_size=3,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
                name=f"block{block_index}_conv{convolution_index}",
            )(x)
            x = tf.keras.layers.BatchNormalization(
                name=f"block{block_index}_bn{convolution_index}"
            )(x)
            x = tf.keras.layers.Activation(
                "relu",
                name=f"block{block_index}_relu{convolution_index}",
            )(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=2,
            name=f"block{block_index}_pool",
        )(x)
        if dropout > 0:
            x = tf.keras.layers.SpatialDropout2D(
                dropout,
                name=f"block{block_index}_dropout",
            )(x)

    # Flatten intentionally preserves spatial information that GAP discards.
    x = tf.keras.layers.Flatten(name="spatial_flatten")(x)
    x = tf.keras.layers.Dense(
        candidate.dense_units,
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizer,
        name="dense_features",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="dense_bn")(x)
    x = tf.keras.layers.Activation("relu", name="dense_relu")(x)
    x = tf.keras.layers.Dropout(
        candidate.dense_dropout,
        name="dense_dropout",
    )(x)
    outputs = tf.keras.layers.Dense(
        len(base.CLASS_NAMES),
        activation="softmax",
        dtype="float32",
        name="class_scores",
    )(x)
    model = tf.keras.Model(inputs, outputs, name=f"latihisyarat_{candidate.name}")

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def _write_model_summary(model, path: Path) -> None:
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    path.write_text(buffer.getvalue(), encoding="utf-8")


def train_high_accuracy_candidate(
    candidate: HighAccuracyCandidate,
    config: HighAccuracyConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    candidate_dir: Path,
) -> dict[str, Any]:
    tf = base._require_tensorflow()
    tf.keras.backend.clear_session()
    base.set_reproducibility(config.seed)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    model = build_high_accuracy_model(candidate)
    _write_model_summary(model, candidate_dir / "model_summary.txt")
    checkpoint_path = candidate_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
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
        tf.keras.callbacks.CSVLogger(candidate_dir / "epoch_log.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    train_dataset = base.make_tf_dataset(
        x_train,
        y_train,
        batch_size=config.batch_size,
        training=True,
        seed=config.seed,
        use_mild_augmentation=False,
    )
    validation_dataset = base.make_tf_dataset(
        x_validation,
        y_validation,
        batch_size=config.batch_size,
        training=False,
        seed=config.seed,
    )

    print(f"\n=== Kandidat eksperimental: {candidate.name} ===")
    print(f"Parameter model: {model.count_params():,}")
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
    history_frame.to_csv(candidate_dir / "training_history.csv")
    base.plot_training_history(
        history_frame,
        candidate_dir / "training_curves.png",
    )

    best_row_index = int(history_frame["val_loss"].to_numpy().argmin())
    result = {
        "candidate": asdict(candidate),
        "parameter_count": int(model.count_params()),
        "epochs_completed": int(len(history_frame)),
        "best_epoch": best_row_index + 1,
        "best_val_loss": float(history_frame.iloc[best_row_index]["val_loss"]),
        "best_val_accuracy": float(
            history_frame.iloc[best_row_index]["val_accuracy"]
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": int(checkpoint_path.stat().st_size),
    }
    base.save_json(candidate_dir / "candidate_summary.json", result)
    return result


def select_high_accuracy_candidate(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not results:
        raise ValueError("Tidak ada kandidat eksperimental untuk dipilih.")
    return min(
        results,
        key=lambda item: (item["best_val_loss"], -item["best_val_accuracy"]),
    )


def _write_experimental_model_card(
    run_dir: Path,
    selected: dict[str, Any],
    evaluation: dict[str, Any],
    target_accuracy: float,
    float32_export: dict[str, Any],
    reference_comparison: dict[str, Any] | None,
) -> None:
    achieved = evaluation["accuracy"] >= target_accuracy
    lines = [
        "# Experimental High-Accuracy Model Card",
        "",
        "This artifact is experimental and does not replace the stable seed-2026 model.",
        "",
        "## Selection",
        "",
        f"- Candidate: `{selected['candidate']['name']}`",
        f"- Parameters: {selected['parameter_count']:,}",
        f"- Best epoch: {selected['best_epoch']}",
        f"- Best validation loss: {selected['best_val_loss']:.8f}",
        f"- Best validation accuracy: {selected['best_val_accuracy']:.6f}",
        "- Candidate selection used validation loss, not test accuracy.",
        "",
        "## Test result",
        "",
        f"- Accuracy: {evaluation['accuracy']:.6f}",
        f"- Macro-F1: {evaluation['macro_f1']:.6f}",
        f"- Target: {target_accuracy:.4f}",
        f"- Target achieved: {achieved}",
        "",
        "## Mobile artifact",
        "",
        f"- Float32 size: {float32_export['size_bytes']} bytes",
        f"- Float32 SHA-256: `{float32_export['sha256']}`",
    ]
    if reference_comparison:
        lines.extend(
            [
                "",
                "## Stable baseline comparison",
                "",
                f"- Accuracy delta: {reference_comparison['accuracy_delta_points']:.4f} points",
                f"- Size multiplier: {reference_comparison['size_multiplier']:.3f}x",
            ]
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- A high Sign Language MNIST score does not prove camera performance.",
            "- Do not repeatedly tune candidates using the test result.",
            "- Evaluate latency, RAM, and camera data before replacing the stable model.",
            "",
        ]
    )
    (run_dir / "experimental_model_card.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_high_accuracy_experiment(
    output_base_dir: str | Path,
    dataset_dir: str | Path,
    reference_model_path: str | Path | None = None,
    run_name: str | None = None,
    config: HighAccuracyConfig | None = None,
    candidates: Sequence[HighAccuracyCandidate] | None = None,
) -> Path:
    """Run a predeclared tournament, evaluate once, and measure model size."""

    tf = base._require_tensorflow()
    config = config or HighAccuracyConfig()
    candidates = tuple(candidates or DEFAULT_HIGH_ACCURACY_CANDIDATES)
    if not candidates:
        raise ValueError("Minimal satu kandidat eksperimental diperlukan.")
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("Nama kandidat eksperimental harus unik.")

    base.set_reproducibility(config.seed)
    run_dir = Path(output_base_dir).expanduser() / (
        run_name or base.make_run_name("high_accuracy_experiment")
    )
    if run_dir.exists():
        raise FileExistsError(f"Folder eksperimen sudah ada: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {[device.name for device in tf.config.list_physical_devices('GPU')]}")
    print(f"Output eksperimental: {run_dir}")
    print(f"Target test accuracy: {config.target_test_accuracy * 100:.2f}%")
    environment = base.save_environment_report(run_dir)
    base.save_json(
        run_dir / "experimental_config.json",
        {
            "warning": (
                "Target is aspirational. Test data is not used for training, "
                "early stopping, or candidate selection."
            ),
            "training": asdict(config),
            "candidates": [asdict(candidate) for candidate in candidates],
            "selection_metric": "minimum validation loss",
        },
    )

    train_csv, test_csv = base.resolve_dataset_paths(dataset_dir)
    x_train_builtin, y_train_builtin, train_report = base.load_sign_mnist_csv(
        train_csv, "training bawaan"
    )
    x_test, y_test, test_report = base.load_sign_mnist_csv(
        test_csv, "test bawaan"
    )
    duplicate_report = base.audit_exact_duplicates(
        x_train_builtin,
        y_train_builtin,
        x_test,
        y_test,
    )
    x_train, x_validation, y_train, y_validation = (
        base.stratified_train_validation_split(
            x_train_builtin,
            y_train_builtin,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
        )
    )
    dataset_report = {
        "source": f"https://www.kaggle.com/datasets/{base.DATASET_SLUG}",
        "train_builtin": train_report,
        "test_builtin": test_report,
        "split": {
            "seed": config.seed,
            "validation_fraction": config.validation_fraction,
            "training_rows": int(len(y_train)),
            "validation_rows": int(len(y_validation)),
            "test_rows": int(len(y_test)),
        },
        "exact_duplicate_audit": duplicate_report,
    }
    base.save_json(run_dir / "dataset_report.json", dataset_report)
    base.plot_dataset_samples(
        x_train,
        y_train,
        run_dir / "dataset_samples.png",
        seed=config.seed,
    )

    results = []
    for candidate in candidates:
        results.append(
            train_high_accuracy_candidate(
                candidate,
                config,
                x_train,
                y_train,
                x_validation,
                y_validation,
                run_dir / "candidates" / candidate.name,
            )
        )

    comparison_frame = pd.DataFrame(
        [
            {
                "name": result["candidate"]["name"],
                "filters": "-".join(
                    str(value) for value in result["candidate"]["filters"]
                ),
                "dense_units": result["candidate"]["dense_units"],
                "learning_rate": result["candidate"]["learning_rate"],
                "dense_dropout": result["candidate"]["dense_dropout"],
                "parameter_count": result["parameter_count"],
                "epochs_completed": result["epochs_completed"],
                "best_epoch": result["best_epoch"],
                "best_val_loss": result["best_val_loss"],
                "best_val_accuracy": result["best_val_accuracy"],
                "checkpoint_size_bytes": result["checkpoint_size_bytes"],
            }
            for result in results
        ]
    ).sort_values(["best_val_loss", "best_val_accuracy"], ascending=[True, False])
    comparison_frame.to_csv(run_dir / "candidate_comparison.csv", index=False)

    selected = select_high_accuracy_candidate(results)
    selected_checkpoint = Path(selected["checkpoint_path"])
    best_model_path = run_dir / "experimental_best_model.keras"
    shutil.copy2(selected_checkpoint, best_model_path)
    shutil.copy2(
        selected_checkpoint.parent / "training_history.csv",
        run_dir / "training_history.csv",
    )
    shutil.copy2(
        selected_checkpoint.parent / "training_curves.png",
        run_dir / "training_curves.png",
    )
    base.save_json(run_dir / "selected_candidate.json", selected)

    best_model = tf.keras.models.load_model(best_model_path)
    evaluation, _ = base.evaluate_selected_model(
        best_model,
        x_test,
        y_test,
        batch_size=config.batch_size,
        run_dir=run_dir,
    )
    target_status = {
        "target_test_accuracy": config.target_test_accuracy,
        "actual_test_accuracy": evaluation["accuracy"],
        "target_achieved": evaluation["accuracy"] >= config.target_test_accuracy,
        "warning": (
            "Do not add more candidates based only on this test result. "
            "That would turn the test set into a tuning set."
        ),
    }
    base.save_json(run_dir / "target_status.json", target_status)

    labels_payload = {
        "schema_version": 1,
        "labels": list(base.CLASS_NAMES),
        "output_index_to_label": {
            str(index): label for index, label in enumerate(base.CLASS_NAMES)
        },
        "unsupported_dynamic_letters": ["J", "Z"],
    }
    base.save_json(run_dir / "labels.json", labels_payload)

    float32_path = run_dir / "experimental_model_float32.tflite"
    float32_export = base.export_tflite(best_model, float32_path, float16=False)
    float16_export = None
    if config.export_float16:
        float16_path = run_dir / "experimental_model_float16.tflite"
        float16_export = base.export_tflite(best_model, float16_path, float16=True)

    sample_count = min(config.tflite_parity_samples, len(x_test))
    rng = np.random.default_rng(config.seed)
    sample_indices = rng.choice(len(x_test), size=sample_count, replace=False)
    parity_reports = [
        base.compare_keras_and_tflite(
            best_model,
            float32_path,
            x_test[sample_indices],
            "float32",
        )
    ]
    if float16_export:
        parity_reports.append(
            base.compare_keras_and_tflite(
                best_model,
                Path(float16_export["path"]),
                x_test[sample_indices],
                "float16_weights",
            )
        )
    base.save_json(run_dir / "tflite_parity.json", parity_reports)

    reference_comparison = None
    if reference_model_path:
        reference_path = Path(reference_model_path).expanduser()
        reference_evaluation_path = reference_path.parent / "evaluation.json"
        if reference_path.exists() and reference_evaluation_path.exists():
            reference_evaluation = json.loads(reference_evaluation_path.read_text())
            reference_comparison = {
                "reference_model_path": str(reference_path),
                "reference_accuracy": reference_evaluation["accuracy"],
                "experimental_accuracy": evaluation["accuracy"],
                "accuracy_delta_points": float(
                    (evaluation["accuracy"] - reference_evaluation["accuracy"])
                    * 100
                ),
                "reference_size_bytes": int(reference_path.stat().st_size),
                "experimental_size_bytes": int(float32_path.stat().st_size),
                "size_multiplier": float(
                    float32_path.stat().st_size / reference_path.stat().st_size
                ),
            }
            base.save_json(
                run_dir / "stable_baseline_comparison.json",
                reference_comparison,
            )

    metadata = {
        "schema_version": 1,
        "experimental": True,
        "created_at_utc": base.utc_now_iso(),
        "model_name": "LatihIsyarat Experimental High-Accuracy CNN",
        "model_version": run_dir.name,
        "selected_candidate": selected,
        "target_status": target_status,
        "input_contract": {
            "shape": [1, 28, 28, 1],
            "dtype": "float32",
            "value_range_before_model": [0.0, 255.0],
            "color_space": "grayscale",
            "normalization": "Rescaling(1/255) inside the model",
            "mobile_must_divide_by_255": False,
        },
        "output_contract": {
            "shape": [1, 24],
            "dtype": "float32",
            "labels": list(base.CLASS_NAMES),
        },
        "keras_model": {
            "path": best_model_path.name,
            "sha256": base.sha256_file(best_model_path),
            "size_bytes": best_model_path.stat().st_size,
            "parameter_count": selected["parameter_count"],
        },
        "tflite_models": [
            item for item in (float32_export, float16_export) if item is not None
        ],
        "test_metrics": {
            key: evaluation[key]
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        },
        "runtime": environment,
        "stable_baseline_comparison": reference_comparison,
        "warning": (
            "Experimental dataset score does not establish camera accuracy."
        ),
    }
    base.save_json(run_dir / "experimental_model_metadata.json", metadata)
    _write_experimental_model_card(
        run_dir,
        selected,
        evaluation,
        config.target_test_accuracy,
        float32_export,
        reference_comparison,
    )
    base.write_run_manifest(run_dir)

    print("\n=== HASIL EKSPERIMEN ===")
    print(f"Kandidat terpilih : {selected['candidate']['name']}")
    print(f"Parameter         : {selected['parameter_count']:,}")
    print(f"Test accuracy     : {evaluation['accuracy'] * 100:.4f}%")
    print(f"Macro-F1          : {evaluation['macro_f1'] * 100:.4f}%")
    print(f"Target 99.5%      : {'TERCAPAI' if target_status['target_achieved'] else 'BELUM TERCAPAI'}")
    print(
        f"TFLite float32    : {float32_export['size_bytes'] / 1024 / 1024:.3f} MiB"
    )
    if float16_export:
        print(
            f"TFLite float16    : {float16_export['size_bytes'] / 1024 / 1024:.3f} MiB"
        )
    if reference_comparison:
        print(
            "Selisih baseline  : "
            f"{reference_comparison['accuracy_delta_points']:+.4f} poin, "
            f"ukuran {reference_comparison['size_multiplier']:.2f}x"
        )
    print(f"Output            : {run_dir}")
    return run_dir


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--reference-model", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_high_accuracy_experiment(
        output_base_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        reference_model_path=args.reference_model,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
