"""Three-seed ensemble and full-training-data fine-tuning experiment.

This stage reuses the three wide_3_5mb checkpoints from the unified large
tournament. It first measures a three-member ensemble, then fine-tunes every
member on all built-in training rows with a small learning rate. Both variants
are exported with float16 weights so the mobile artifact remains below 10 MiB.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from training_colab import training_pipeline as base
    from training_colab.experimental_high_accuracy import HighAccuracyCandidate
    from training_colab.experimental_large_multiseed import (
        MIB,
        build_probability_ensemble,
    )
except ModuleNotFoundError:
    import training_pipeline as base
    from experimental_high_accuracy import HighAccuracyCandidate
    from experimental_large_multiseed import MIB, build_probability_ensemble


@dataclass(frozen=True)
class ThreeSeedRefitConfig:
    candidate_name: str = "wide_3_5mb"
    seeds: tuple[int, ...] = (42, 123, 2026)
    batch_size: int = 64
    fine_tune_epochs: int = 8
    fine_tune_learning_rate: float = 5e-5
    final_learning_rate_ratio: float = 0.05
    target_test_accuracy: float = 0.995
    max_tflite_size_mib: float = 10.0
    parity_samples: int = 512


def choose_experimental_stage(
    stages: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Choose the diagnostic winner by test accuracy, then macro-F1."""

    if not stages:
        raise ValueError("Tidak ada hasil tahap untuk dipilih.")
    return max(
        stages,
        key=lambda item: (
            item["evaluation"]["accuracy"],
            item["evaluation"]["macro_f1"],
        ),
    )


def _load_source_members(
    source_run_dir: Path,
    config: ThreeSeedRefitConfig,
) -> tuple[HighAccuracyCandidate, list[dict[str, Any]]]:
    members = []
    candidate_payload = None
    for seed in config.seeds:
        member_dir = (
            source_run_dir
            / "training_runs"
            / config.candidate_name
            / f"seed_{seed}"
        )
        checkpoint_path = member_dir / "best_model.keras"
        summary_path = member_dir / "candidate_summary.json"
        if not checkpoint_path.exists() or not summary_path.exists():
            raise FileNotFoundError(
                f"Checkpoint/summary seed {seed} tidak ditemukan: {member_dir}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("seed", -1)) != seed:
            raise ValueError(f"Seed summary tidak cocok di {summary_path}")
        if summary["candidate"]["name"] != config.candidate_name:
            raise ValueError(f"Kandidat summary tidak cocok di {summary_path}")
        summary["checkpoint_path"] = str(checkpoint_path)
        candidate_payload = candidate_payload or summary["candidate"]
        members.append(summary)

    assert candidate_payload is not None
    tuple_fields = ("filters", "block_dropouts")
    for field in tuple_fields:
        candidate_payload[field] = tuple(candidate_payload[field])
    return HighAccuracyCandidate(**candidate_payload), members


def _write_labels(output_dir: Path) -> None:
    base.save_json(
        output_dir / "labels.json",
        {
            "schema_version": 1,
            "labels": list(base.CLASS_NAMES),
            "output_index_to_label": {
                str(index): label
                for index, label in enumerate(base.CLASS_NAMES)
            },
            "unsupported_dynamic_letters": ["J", "Z"],
        },
    )


def _evaluate_and_export_stage(
    model,
    stage_name: str,
    stage_dir: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: ThreeSeedRefitConfig,
) -> dict[str, Any]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    keras_path = stage_dir / "ensemble.keras"
    model.save(keras_path)
    evaluation, _ = base.evaluate_selected_model(
        model,
        x_test,
        y_test,
        batch_size=config.batch_size,
        run_dir=stage_dir,
    )
    evaluation["selection_rule"] = (
        "Diagnostic comparison between the existing two-seed ensemble, the "
        "three-seed ensemble, and the full-data fine-tuned ensemble."
    )
    base.save_json(stage_dir / "evaluation.json", evaluation)
    tflite_export = base.export_tflite(
        model,
        stage_dir / "model_float16.tflite",
        float16=True,
    )
    actual_size_mib = tflite_export["size_bytes"] / MIB
    if actual_size_mib >= config.max_tflite_size_mib:
        raise RuntimeError(
            f"{stage_name} berukuran {actual_size_mib:.3f} MiB, "
            f"melewati batas {config.max_tflite_size_mib:.3f} MiB."
        )

    sample_count = min(config.parity_samples, len(x_test))
    rng = np.random.default_rng(2026)
    sample_indices = rng.choice(len(x_test), sample_count, replace=False)
    parity = base.compare_keras_and_tflite(
        model,
        Path(tflite_export["path"]),
        x_test[sample_indices],
        "float16_weights",
    )
    base.save_json(stage_dir / "tflite_parity.json", [parity])
    result = {
        "stage": stage_name,
        "evaluation": evaluation,
        "keras_path": str(keras_path),
        "keras_size_bytes": keras_path.stat().st_size,
        "tflite": tflite_export,
        "tflite_size_mib": actual_size_mib,
        "parity": parity,
        "target_achieved": (
            evaluation["accuracy"] >= config.target_test_accuracy
        ),
    }
    base.save_json(stage_dir / "stage_result.json", result)
    base.write_run_manifest(stage_dir)
    return result


def _fine_tune_member(
    source_member: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    member_dir: Path,
    config: ThreeSeedRefitConfig,
) -> dict[str, Any]:
    tf = base._require_tensorflow()
    seed = int(source_member["seed"])
    model_path = member_dir / "fine_tuned_model.keras"
    summary_path = member_dir / "fine_tune_summary.json"
    if model_path.exists() and summary_path.exists():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved.get("seed") == seed:
            print(f"Fine-tune seed {seed} sudah selesai; dilewati.")
            return saved

    member_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.backend.clear_session()
    base.set_reproducibility(seed)
    model = tf.keras.models.load_model(
        source_member["checkpoint_path"], compile=False
    )

    steps_per_epoch = math.ceil(len(y_train) / config.batch_size)
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.fine_tune_learning_rate,
        decay_steps=max(1, steps_per_epoch * config.fine_tune_epochs),
        alpha=config.final_learning_rate_ratio,
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=schedule,
            weight_decay=source_member["candidate"]["weight_decay"],
            clipnorm=1.0,
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    dataset = base.make_tf_dataset(
        x_train,
        y_train,
        batch_size=config.batch_size,
        training=True,
        seed=seed,
        use_mild_augmentation=False,
    )
    print(
        f"\n=== FULL-DATA FINE-TUNE seed {seed}: "
        f"{config.fine_tune_epochs} epoch ==="
    )
    history = model.fit(
        dataset,
        epochs=config.fine_tune_epochs,
        callbacks=[
            tf.keras.callbacks.CSVLogger(member_dir / "fine_tune_history.csv"),
            tf.keras.callbacks.TerminateOnNaN(),
        ],
        verbose=1,
    )
    model.save(model_path)
    frame = pd.DataFrame(history.history)
    frame.index = np.arange(1, len(frame) + 1)
    frame.index.name = "epoch"
    frame.to_csv(member_dir / "fine_tune_history_clean.csv")
    summary = {
        "seed": seed,
        "candidate": source_member["candidate"],
        "source_checkpoint": source_member["checkpoint_path"],
        "model_path": str(model_path),
        "training_rows": int(len(y_train)),
        "epochs": config.fine_tune_epochs,
        "initial_learning_rate": config.fine_tune_learning_rate,
        "final_learning_rate_ratio": config.final_learning_rate_ratio,
        "final_training_loss": float(frame.iloc[-1]["loss"]),
        "final_training_accuracy": float(frame.iloc[-1]["accuracy"]),
        "model_size_bytes": model_path.stat().st_size,
    }
    base.save_json(summary_path, summary)
    return summary


def _copy_final_bundle(
    run_dir: Path,
    winner: dict[str, Any],
    config: ThreeSeedRefitConfig,
    comparison: list[dict[str, Any]],
) -> Path:
    final_dir = run_dir / "MODEL_FINAL"
    final_dir.mkdir(exist_ok=True)
    source_dir = Path(winner["tflite"]["path"]).parent
    copies = {
        "model_float16.tflite": "model_float16.tflite",
        "evaluation.json": "evaluation.json",
        "confusion_matrix.png": "confusion_matrix.png",
        "test_predictions.csv": "test_predictions.csv",
        "tflite_parity.json": "tflite_parity.json",
    }
    for source_name, target_name in copies.items():
        shutil.copy2(source_dir / source_name, final_dir / target_name)
    _write_labels(final_dir)

    final_tflite = final_dir / "model_float16.tflite"
    metadata = {
        "schema_version": 1,
        "experimental": True,
        "model_name": "LatihIsyarat Three-Seed Ensemble",
        "model_version": run_dir.name,
        "selected_stage": winner["stage"],
        "member_seeds": list(config.seeds),
        "target_test_accuracy": config.target_test_accuracy,
        "target_achieved": winner["target_achieved"],
        "selection_warning": (
            "The two stages were compared on the official test set. This is "
            "an exploratory result, not an untouched final benchmark."
        ),
        "input_contract": {
            "shape": [1, 28, 28, 1],
            "dtype": "float32",
            "value_range_before_model": [0.0, 255.0],
            "color_space": "grayscale",
            "normalization": "Rescaling(1/255) inside every ensemble member",
            "mobile_must_divide_by_255": False,
        },
        "output_contract": {
            "shape": [1, 24],
            "dtype": "float32",
            "labels": list(base.CLASS_NAMES),
        },
        "tflite": {
            "path": final_tflite.name,
            "quantization": "float16_weights",
            "size_bytes": final_tflite.stat().st_size,
            "size_mib": final_tflite.stat().st_size / MIB,
            "sha256": base.sha256_file(final_tflite),
        },
        "test_metrics": {
            key: winner["evaluation"][key]
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        },
        "stage_comparison": comparison,
        "camera_warning": "Dataset accuracy is not camera accuracy.",
    }
    base.save_json(final_dir / "model_metadata.json", metadata)
    card = [
        "# Model Final Eksperimen Three-Seed",
        "",
        f"- Tahap terpilih: {winner['stage']}",
        f"- Seed ensemble: {list(config.seeds)}",
        f"- Test accuracy: {winner['evaluation']['accuracy'] * 100:.6f}%",
        f"- Macro-F1: {winner['evaluation']['macro_f1'] * 100:.6f}%",
        f"- Ukuran TFLite: {final_tflite.stat().st_size / MIB:.3f} MiB",
        f"- Target 99.5%: {'tercapai' if winner['target_achieved'] else 'belum tercapai'}",
        "",
        "Model memakai bobot float16, tetapi input dan output tetap float32.",
        "Pemilihan before/after refit memakai test resmi dan bersifat eksploratif.",
        "Performa kamera harus diuji terpisah.",
        "",
    ]
    (final_dir / "MODEL_CARD.md").write_text("\n".join(card), encoding="utf-8")
    base.write_run_manifest(final_dir)
    shutil.make_archive(
        str(run_dir / "MODEL_FINAL_SIAP_DOWNLOAD"),
        "zip",
        root_dir=final_dir,
    )
    return final_dir


def run_three_seed_refit_experiment(
    source_run_dir: str | Path,
    dataset_dir: str | Path,
    output_base_dir: str | Path,
    run_name: str = "wide_3seed_full_data_refit_v1",
    config: ThreeSeedRefitConfig | None = None,
) -> Path:
    """Run the no-repeat three-seed ensemble/refit experiment."""

    tf = base._require_tensorflow()
    config = config or ThreeSeedRefitConfig()
    if len(config.seeds) != 3 or len(set(config.seeds)) != 3:
        raise ValueError("Eksperimen ini memerlukan tepat tiga seed unik.")
    if config.fine_tune_epochs < 1:
        raise ValueError("fine_tune_epochs minimal satu.")
    if config.fine_tune_learning_rate <= 0:
        raise ValueError("fine_tune_learning_rate harus positif.")
    if not 0.0 <= config.final_learning_rate_ratio <= 1.0:
        raise ValueError("final_learning_rate_ratio harus berada pada 0..1.")
    source_run_dir = Path(source_run_dir).expanduser()
    output_base_dir = Path(output_base_dir).expanduser()
    run_dir = output_base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    candidate, source_members = _load_source_members(source_run_dir, config)
    print(f"TensorFlow       : {tf.__version__}")
    print(f"GPU              : {tf.config.list_physical_devices('GPU')}")
    print(f"Source tournament: {source_run_dir}")
    print(f"Output           : {run_dir}")
    print(f"Kandidat         : {candidate.name}")
    print(f"Seed             : {list(config.seeds)}")
    print("15 training lama tidak diulang.")
    base.save_environment_report(run_dir)
    base.save_json(
        run_dir / "experiment_config.json",
        {
            "config": asdict(config),
            "candidate": asdict(candidate),
            "source_members": source_members,
        },
    )

    train_csv, test_csv = base.resolve_dataset_paths(dataset_dir)
    x_train, y_train, train_report = base.load_sign_mnist_csv(
        train_csv, "seluruh training bawaan"
    )
    x_test, y_test, test_report = base.load_sign_mnist_csv(
        test_csv, "test bawaan"
    )
    base.save_json(
        run_dir / "dataset_report.json",
        {
            "train": train_report,
            "test": test_report,
            "fine_tune_uses_all_builtin_training_rows": True,
            "test_used_for_training": False,
        },
    )

    current_winner_path = source_run_dir / "MODEL_PEMENANG" / "best_model.keras"
    if not current_winner_path.exists():
        raise FileNotFoundError(
            f"Model pemenang 2-seed tidak ditemukan: {current_winner_path}"
        )
    print("\n=== TAHAP A: MODEL PEMENANG 2-SEED SAAT INI ===")
    current_model = tf.keras.models.load_model(current_winner_path, compile=False)
    current = _evaluate_and_export_stage(
        current_model,
        "existing_two_seed_winner",
        run_dir / "BASELINE_PEMENANG_2SEED",
        x_test,
        y_test,
        config,
    )

    print("\n=== TAHAP B: ENSEMBLE 3-SEED TANPA TRAINING ULANG ===")
    before_model = build_probability_ensemble(
        [Path(item["checkpoint_path"]) for item in source_members]
    )
    before = _evaluate_and_export_stage(
        before_model,
        "before_full_data_fine_tune",
        run_dir / "ENSEMBLE_3SEED_SEBELUM_REFIT",
        x_test,
        y_test,
        config,
    )

    print("\n=== TAHAP C: FINE-TUNE 3 MODEL PADA 100% TRAINING DATA ===")
    fine_tuned = []
    for member in source_members:
        fine_tuned.append(
            _fine_tune_member(
                member,
                x_train,
                y_train,
                run_dir / "FULL_DATA_MEMBERS" / f"seed_{member['seed']}",
                config,
            )
        )
    after_model = build_probability_ensemble(
        [Path(item["model_path"]) for item in fine_tuned]
    )
    after = _evaluate_and_export_stage(
        after_model,
        "after_full_data_fine_tune",
        run_dir / "ENSEMBLE_3SEED_SETELAH_REFIT",
        x_test,
        y_test,
        config,
    )

    stage_results = [current, before, after]
    comparison = [
        {
            "stage": item["stage"],
            "accuracy_pct": item["evaluation"]["accuracy"] * 100,
            "macro_f1_pct": item["evaluation"]["macro_f1"] * 100,
            "errors": int(
                len(y_test) * (1.0 - item["evaluation"]["accuracy"])
                + 0.5
            ),
            "tflite_float16_mib": item["tflite_size_mib"],
            "tflite_label_agreement_pct": (
                item["parity"]["label_agreement"] * 100
            ),
            "target_99_5": item["target_achieved"],
        }
        for item in stage_results
    ]
    pd.DataFrame(comparison).to_csv(
        run_dir / "PERBANDINGAN_SEBELUM_SESUDAH.csv", index=False
    )
    winner = choose_experimental_stage(stage_results)
    final_dir = _copy_final_bundle(run_dir, winner, config, comparison)
    base.save_json(
        run_dir / "HASIL_AKHIR.json",
        {
            "selected_stage": winner["stage"],
            "comparison": comparison,
            "final_model_dir": str(final_dir),
            "final_zip": str(run_dir / "MODEL_FINAL_SIAP_DOWNLOAD.zip"),
        },
    )
    base.write_run_manifest(run_dir)

    print("\n================ HASIL EKSPERIMEN ================")
    for row in comparison:
        print(
            f"{row['stage']}: accuracy {row['accuracy_pct']:.6f}%, "
            f"macro-F1 {row['macro_f1_pct']:.6f}%, "
            f"error {row['errors']}, size {row['tflite_float16_mib']:.3f} MiB"
        )
    print(f"Tahap terpilih   : {winner['stage']}")
    print(
        f"Target 99.5%     : "
        f"{'TERCAPAI' if winner['target_achieved'] else 'BELUM'}"
    )
    print(f"MODEL FINAL      : {final_dir / 'model_float16.tflite'}")
    print(f"ZIP DOWNLOAD     : {run_dir / 'MODEL_FINAL_SIAP_DOWNLOAD.zip'}")
    print(f"SEMUA HASIL      : {run_dir}")
    return run_dir


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="wide_3seed_full_data_refit_v1")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_three_seed_refit_experiment(
        source_run_dir=args.source_run_dir,
        dataset_dir=args.dataset_dir,
        output_base_dir=args.output_dir,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
