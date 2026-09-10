"""Unified multi-seed tournament for larger Sign Language MNIST CNNs.

The tournament keeps every training run in one experiment directory and
places the selected mobile artifacts in a clearly named MODEL_PEMENANG folder.
Candidate and seed selection uses validation metrics only. The official test
set is evaluated once after selection.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from training_colab import training_pipeline as base
    from training_colab.experimental_high_accuracy import (
        HighAccuracyCandidate,
        HighAccuracyConfig,
        build_high_accuracy_model,
        train_high_accuracy_candidate,
    )
except ModuleNotFoundError:
    import training_pipeline as base
    from experimental_high_accuracy import (
        HighAccuracyCandidate,
        HighAccuracyConfig,
        build_high_accuracy_model,
        train_high_accuracy_candidate,
    )


MIB = 1024 * 1024


@dataclass(frozen=True)
class LargeMultiSeedConfig:
    seeds: tuple[int, ...] = (42, 123, 2026)
    split_seed: int = 2026
    validation_fraction: float = 0.20
    batch_size: int = 64
    max_epochs: int = 70
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-7
    target_test_accuracy: float = 0.995
    max_float32_tflite_mib: float = 10.0
    tflite_parity_samples: int = 512
    export_float16: bool = True


# The first candidate is retained because three independently trained copies
# can form one roughly 6.75 MiB ensemble. The others fill the space up to the
# requested 10 MiB on-device ceiling as larger single-model alternatives.
DEFAULT_LARGE_CANDIDATES = (
    HighAccuracyCandidate(
        name="ensemble_ready_2_25mb",
        filters=(32, 64, 128),
        dense_units=256,
        learning_rate=0.001,
        block_dropouts=(0.05, 0.10, 0.15),
        dense_dropout=0.25,
        weight_decay=1e-5,
    ),
    HighAccuracyCandidate(
        name="wide_3_5mb",
        filters=(40, 80, 160),
        dense_units=320,
        learning_rate=0.0008,
        block_dropouts=(0.03, 0.08, 0.12),
        dense_dropout=0.20,
        weight_decay=1e-5,
    ),
    HighAccuracyCandidate(
        name="balanced_5mb",
        filters=(48, 96, 192),
        dense_units=384,
        learning_rate=0.0007,
        block_dropouts=(0.02, 0.06, 0.10),
        dense_dropout=0.18,
        weight_decay=8e-6,
    ),
    HighAccuracyCandidate(
        name="low_dropout_6_8mb",
        filters=(56, 112, 224),
        dense_units=448,
        learning_rate=0.0006,
        block_dropouts=(0.00, 0.04, 0.08),
        dense_dropout=0.15,
        weight_decay=8e-6,
    ),
    HighAccuracyCandidate(
        name="max_capacity_8_9mb",
        filters=(64, 128, 256),
        dense_units=512,
        learning_rate=0.0005,
        block_dropouts=(0.00, 0.03, 0.06),
        dense_dropout=0.12,
        weight_decay=5e-6,
    ),
)


def inspect_candidate_capacities(
    candidates: Sequence[HighAccuracyCandidate],
) -> pd.DataFrame:
    """Build each model once and return its parameter/size estimate."""

    tf = base._require_tensorflow()
    rows = []
    for candidate in candidates:
        tf.keras.backend.clear_session()
        model = build_high_accuracy_model(candidate)
        parameters = int(model.count_params())
        rows.append(
            {
                "candidate": candidate.name,
                "parameters": parameters,
                "estimated_float32_mib": parameters * 4 / MIB,
                "estimated_float16_mib": parameters * 2 / MIB,
            }
        )
    tf.keras.backend.clear_session()
    return pd.DataFrame(rows)


def _validate_inputs(
    config: LargeMultiSeedConfig,
    candidates: Sequence[HighAccuracyCandidate],
    capacities: pd.DataFrame,
) -> None:
    if len(config.seeds) < 2:
        raise ValueError("Eksperimen multi-seed memerlukan minimal dua seed.")
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("Daftar seed tidak boleh mengandung duplikat.")
    if not candidates:
        raise ValueError("Minimal satu kandidat model diperlukan.")
    if len({item.name for item in candidates}) != len(candidates):
        raise ValueError("Nama kandidat model harus unik.")

    # A small reserve covers FlatBuffer metadata and operators.
    safe_raw_limit = config.max_float32_tflite_mib - 0.15
    too_large = capacities[
        capacities["estimated_float32_mib"] > safe_raw_limit
    ]
    if not too_large.empty:
        names = ", ".join(too_large["candidate"].tolist())
        raise ValueError(
            f"Perkiraan kandidat melewati batas aman "
            f"{config.max_float32_tflite_mib:.1f} MiB: {names}"
        )


def _training_config(
    config: LargeMultiSeedConfig,
    seed: int,
) -> HighAccuracyConfig:
    return HighAccuracyConfig(
        seed=seed,
        validation_fraction=config.validation_fraction,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        early_stopping_patience=config.early_stopping_patience,
        reduce_lr_patience=config.reduce_lr_patience,
        reduce_lr_factor=config.reduce_lr_factor,
        min_learning_rate=config.min_learning_rate,
        target_test_accuracy=config.target_test_accuracy,
        tflite_parity_samples=config.tflite_parity_samples,
        export_float16=config.export_float16,
    )


def summarize_multi_seed_results(
    results: Sequence[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for result in results:
        candidate = result["candidate"]
        rows.append(
            {
                "candidate": candidate["name"],
                "seed": int(result["seed"]),
                "parameters": int(result["parameter_count"]),
                "estimated_float32_mib": (
                    int(result["parameter_count"]) * 4 / MIB
                ),
                "best_epoch": int(result["best_epoch"]),
                "epochs_completed": int(result["epochs_completed"]),
                "best_val_loss": float(result["best_val_loss"]),
                "best_val_accuracy_pct": (
                    float(result["best_val_accuracy"]) * 100
                ),
                "checkpoint_path": result["checkpoint_path"],
            }
        )
    detailed = pd.DataFrame(rows).sort_values(
        ["candidate", "seed"], ignore_index=True
    )
    aggregate = (
        detailed.groupby("candidate", as_index=False)
        .agg(
            parameters=("parameters", "first"),
            estimated_float32_mib=("estimated_float32_mib", "first"),
            seeds_completed=("seed", "count"),
            mean_val_loss=("best_val_loss", "mean"),
            std_val_loss=("best_val_loss", lambda values: values.std(ddof=0)),
            mean_val_accuracy_pct=("best_val_accuracy_pct", "mean"),
            std_val_accuracy_pct=(
                "best_val_accuracy_pct",
                lambda values: values.std(ddof=0),
            ),
        )
        .sort_values(
            ["mean_val_loss", "std_val_loss", "mean_val_accuracy_pct"],
            ascending=[True, True, False],
            ignore_index=True,
        )
    )
    return detailed, aggregate


def select_multi_seed_winner(
    results: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Select architecture by seed mean, then its best validation checkpoint."""

    detailed, aggregate = summarize_multi_seed_results(results)
    winning_candidate = str(aggregate.iloc[0]["candidate"])
    eligible = [
        result
        for result in results
        if result["candidate"]["name"] == winning_candidate
    ]
    selected = min(
        eligible,
        key=lambda item: (item["best_val_loss"], -item["best_val_accuracy"]),
    )
    return selected, detailed, aggregate


def select_deployment_members(
    results: Sequence[dict[str, Any]],
    winning_candidate: str,
    max_float32_tflite_mib: float,
) -> list[dict[str, Any]]:
    """Use as many best validation seeds as fit under the size ceiling."""

    eligible = sorted(
        (
            result
            for result in results
            if result["candidate"]["name"] == winning_candidate
        ),
        key=lambda item: (item["best_val_loss"], -item["best_val_accuracy"]),
    )
    if not eligible:
        raise ValueError(f"Kandidat pemenang tidak ditemukan: {winning_candidate}")
    single_estimated_mib = eligible[0]["parameter_count"] * 4 / MIB
    safe_limit_mib = max_float32_tflite_mib - 0.15
    member_limit = max(1, int(safe_limit_mib // single_estimated_mib))
    return eligible[: min(member_limit, len(eligible))]


def build_probability_ensemble(
    checkpoint_paths: Sequence[Path],
):
    """Wrap seed models as one Keras/TFLite model by averaging probabilities."""

    tf = base._require_tensorflow()
    if not checkpoint_paths:
        raise ValueError("Ensemble memerlukan minimal satu checkpoint.")
    members = [tf.keras.models.load_model(path) for path in checkpoint_paths]
    if len(members) == 1:
        return members[0]

    for member in members:
        member.trainable = False
    inputs = tf.keras.layers.Input(shape=(28, 28, 1), name="image_0_255")
    outputs = [member(inputs, training=False) for member in members]
    averaged = tf.keras.layers.Average(name="average_seed_probabilities")(outputs)
    return tf.keras.Model(inputs, averaged, name="latihisyarat_seed_ensemble")


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


def run_large_multiseed_experiment(
    output_base_dir: str | Path,
    dataset_dir: str | Path,
    reference_model_path: str | Path | None = None,
    run_name: str | None = None,
    config: LargeMultiSeedConfig | None = None,
    candidates: Sequence[HighAccuracyCandidate] | None = None,
    resume: bool = True,
) -> Path:
    """Train all candidate/seed pairs and export one consolidated winner."""

    tf = base._require_tensorflow()
    config = config or LargeMultiSeedConfig()
    candidates = tuple(candidates or DEFAULT_LARGE_CANDIDATES)
    capacities = inspect_candidate_capacities(candidates)
    _validate_inputs(config, candidates, capacities)

    run_dir = Path(output_base_dir).expanduser() / (
        run_name or base.make_run_name("large_under_10mb_multiseed")
    )
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Folder eksperimen sudah ada: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=resume)
    winner_dir = run_dir / "MODEL_PEMENANG"
    training_runs_dir = run_dir / "training_runs"

    print(f"TensorFlow       : {tf.__version__}")
    print(f"GPU              : {tf.config.list_physical_devices('GPU')}")
    print(f"Output tunggal   : {run_dir}")
    print(f"Kandidat         : {len(candidates)}")
    print(f"Seed             : {list(config.seeds)}")
    print(f"Total training   : {len(candidates) * len(config.seeds)}")
    print(f"Batas TFLite     : < {config.max_float32_tflite_mib:.1f} MiB")
    print(f"Target test      : {config.target_test_accuracy * 100:.2f}%")

    environment = base.save_environment_report(run_dir)
    capacities.to_csv(run_dir / "PERKIRAAN_UKURAN_MODEL.csv", index=False)
    base.save_json(
        run_dir / "KONFIGURASI_EKSPERIMEN.json",
        {
            "training": asdict(config),
            "candidates": [asdict(item) for item in candidates],
            "total_training_runs": len(candidates) * len(config.seeds),
            "selection": (
                "minimum mean validation loss across seeds; then minimum "
                "validation loss within the winning candidate"
            ),
            "test_policy": "official test evaluated once after selection",
        },
    )

    train_csv, test_csv = base.resolve_dataset_paths(dataset_dir)
    x_builtin, y_builtin, train_report = base.load_sign_mnist_csv(
        train_csv, "training bawaan"
    )
    x_test, y_test, test_report = base.load_sign_mnist_csv(
        test_csv, "test bawaan"
    )
    duplicate_report = base.audit_exact_duplicates(
        x_builtin, y_builtin, x_test, y_test
    )
    x_train, x_validation, y_train, y_validation = (
        base.stratified_train_validation_split(
            x_builtin,
            y_builtin,
            validation_fraction=config.validation_fraction,
            seed=config.split_seed,
        )
    )
    base.save_json(
        run_dir / "dataset_report.json",
        {
            "source": f"https://www.kaggle.com/datasets/{base.DATASET_SLUG}",
            "train_builtin": train_report,
            "test_builtin": test_report,
            "split": {
                "seed": config.split_seed,
                "validation_fraction": config.validation_fraction,
                "training_rows": len(y_train),
                "validation_rows": len(y_validation),
                "test_rows": len(y_test),
            },
            "exact_duplicate_audit": duplicate_report,
        },
    )
    base.plot_dataset_samples(
        x_train, y_train, run_dir / "dataset_samples.png", config.split_seed
    )

    results: list[dict[str, Any]] = []
    total_runs = len(candidates) * len(config.seeds)
    run_index = 0
    for candidate in candidates:
        for seed in config.seeds:
            run_index += 1
            print(
                f"\n##### TRAINING {run_index}/{total_runs}: "
                f"{candidate.name}, seed {seed} #####"
            )
            candidate_dir = training_runs_dir / candidate.name / f"seed_{seed}"
            checkpoint_path = candidate_dir / "best_model.keras"
            summary_path = candidate_dir / "candidate_summary.json"
            if resume and checkpoint_path.exists() and summary_path.exists():
                saved_result = json.loads(summary_path.read_text(encoding="utf-8"))
                if saved_result.get("seed") == seed:
                    print("Checkpoint lengkap ditemukan; training dilewati.")
                    results.append(saved_result)
                    continue

            # A unique graph name lets checkpoints from several seeds coexist
            # as nested members inside one ensemble model.
            graph_candidate = replace(
                candidate,
                name=f"{candidate.name}_seed_{seed}",
            )
            result = train_high_accuracy_candidate(
                candidate=graph_candidate,
                config=_training_config(config, seed),
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                y_validation=y_validation,
                candidate_dir=candidate_dir,
            )
            result["graph_candidate_name"] = graph_candidate.name
            result["candidate"] = asdict(candidate)
            result["seed"] = seed
            base.save_json(
                Path(result["checkpoint_path"]).parent / "candidate_summary.json",
                result,
            )
            results.append(result)

    selected, detailed, aggregate = select_multi_seed_winner(results)
    detailed.to_csv(run_dir / "SEMUA_HASIL_TRAINING.csv", index=False)
    aggregate.to_csv(run_dir / "PERBANDINGAN_UTAMA.csv", index=False)

    winning_candidate = selected["candidate"]["name"]
    deployment_members = select_deployment_members(
        results,
        winning_candidate=winning_candidate,
        max_float32_tflite_mib=config.max_float32_tflite_mib,
    )
    deployment = {
        "candidate": winning_candidate,
        "mode": "ensemble" if len(deployment_members) > 1 else "single",
        "member_count": len(deployment_members),
        "member_seeds": [item["seed"] for item in deployment_members],
        "members": deployment_members,
        "estimated_float32_mib": sum(
            item["parameter_count"] * 4 / MIB for item in deployment_members
        ),
    }

    winner_dir.mkdir(exist_ok=True)
    best_model_path = winner_dir / "best_model.keras"
    best_model = build_probability_ensemble(
        [Path(item["checkpoint_path"]) for item in deployment_members]
    )
    best_model.save(best_model_path)
    shutil.copy2(
        Path(selected["checkpoint_path"]).parent / "training_history.csv",
        winner_dir / "training_history.csv",
    )
    shutil.copy2(
        Path(selected["checkpoint_path"]).parent / "training_curves.png",
        winner_dir / "training_curves.png",
    )
    base.save_json(
        winner_dir / "selected_model.json",
        {"architecture_selection": selected, "deployment": deployment},
    )
    _write_labels(winner_dir)

    evaluation, _ = base.evaluate_selected_model(
        best_model,
        x_test,
        y_test,
        batch_size=config.batch_size,
        run_dir=winner_dir,
    )
    float32_export = base.export_tflite(
        best_model,
        winner_dir / "model_float32.tflite",
        float16=False,
    )
    float16_export = None
    if config.export_float16:
        float16_export = base.export_tflite(
            best_model,
            winner_dir / "model_float16.tflite",
            float16=True,
        )

    actual_mib = float32_export["size_bytes"] / MIB
    within_size_limit = actual_mib < config.max_float32_tflite_mib
    sample_count = min(config.tflite_parity_samples, len(x_test))
    rng = np.random.default_rng(config.split_seed)
    sample_indices = rng.choice(len(x_test), sample_count, replace=False)
    parity = [
        base.compare_keras_and_tflite(
            best_model,
            Path(float32_export["path"]),
            x_test[sample_indices],
            "float32",
        )
    ]
    if float16_export:
        parity.append(
            base.compare_keras_and_tflite(
                best_model,
                Path(float16_export["path"]),
                x_test[sample_indices],
                "float16_weights",
            )
        )
    base.save_json(winner_dir / "tflite_parity.json", parity)

    reference_comparison = None
    if reference_model_path:
        reference_path = Path(reference_model_path).expanduser()
        reference_evaluation_path = reference_path.parent / "evaluation.json"
        if reference_path.exists() and reference_evaluation_path.exists():
            reference_evaluation = json.loads(
                reference_evaluation_path.read_text(encoding="utf-8")
            )
            reference_comparison = {
                "reference_model": str(reference_path),
                "reference_accuracy": reference_evaluation["accuracy"],
                "winner_accuracy": evaluation["accuracy"],
                "accuracy_delta_points": (
                    evaluation["accuracy"] - reference_evaluation["accuracy"]
                )
                * 100,
                "reference_size_mib": reference_path.stat().st_size / MIB,
                "winner_size_mib": actual_mib,
            }

    target_status = {
        "target_test_accuracy": config.target_test_accuracy,
        "actual_test_accuracy": evaluation["accuracy"],
        "target_achieved": evaluation["accuracy"] >= config.target_test_accuracy,
        "max_float32_tflite_mib": config.max_float32_tflite_mib,
        "actual_float32_tflite_mib": actual_mib,
        "within_size_limit": within_size_limit,
    }
    base.save_json(winner_dir / "target_status.json", target_status)
    metadata = {
        "schema_version": 1,
        "experimental": True,
        "created_at_utc": base.utc_now_iso(),
        "model_name": "LatihIsyarat Large Multi-Seed Winner",
        "model_version": run_dir.name,
        "selected_model": selected,
        "deployment": deployment,
        "selection_rule": (
            "candidate with minimum mean validation loss across seeds, then "
            "its checkpoint with minimum validation loss"
        ),
        "seeds": list(config.seeds),
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
            "size_bytes": best_model_path.stat().st_size,
            "sha256": base.sha256_file(best_model_path),
            "parameter_count": int(best_model.count_params()),
        },
        "tflite_models": [
            item for item in (float32_export, float16_export) if item is not None
        ],
        "test_metrics": {
            key: evaluation[key]
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        },
        "reference_comparison": reference_comparison,
        "runtime": environment,
        "warning": "Dataset test accuracy is not camera accuracy.",
    }
    base.save_json(winner_dir / "model_metadata.json", metadata)

    card = [
        "# Model Pemenang Eksperimen Besar",
        "",
        f"- Kandidat: `{selected['candidate']['name']}`",
        f"- Mode deployment: {deployment['mode']}",
        f"- Seed anggota: {deployment['member_seeds']}",
        f"- Parameter total: {best_model.count_params():,}",
        f"- Test accuracy: {evaluation['accuracy'] * 100:.6f}%",
        f"- Macro-F1: {evaluation['macro_f1'] * 100:.6f}%",
        f"- TFLite float32: {actual_mib:.3f} MiB",
        f"- Target 99.5%: {'tercapai' if target_status['target_achieved'] else 'belum tercapai'}",
        f"- Batas <10 MiB: {'lolos' if within_size_limit else 'tidak lolos'}",
        "",
        "Arsitektur dipilih memakai rata-rata validation loss antar-seed.",
        "Official test set hanya dievaluasi setelah model dipilih.",
        "Performa kamera harus diuji terpisah pada aplikasi Flutter.",
        "",
    ]
    (winner_dir / "MODEL_CARD.md").write_text("\n".join(card), encoding="utf-8")

    base.write_run_manifest(winner_dir)
    shutil.make_archive(
        str(run_dir / "MODEL_PEMENANG_SIAP_DOWNLOAD"),
        "zip",
        root_dir=winner_dir,
    )
    base.write_run_manifest(run_dir)

    print("\n================ HASIL AKHIR ================")
    print(f"Kandidat pemenang : {selected['candidate']['name']}")
    print(f"Mode deployment   : {deployment['mode']}")
    print(f"Seed anggota      : {deployment['member_seeds']}")
    print(f"Test accuracy     : {evaluation['accuracy'] * 100:.6f}%")
    print(f"Macro-F1          : {evaluation['macro_f1'] * 100:.6f}%")
    print(f"Target 99.5%      : {'TERCAPAI' if target_status['target_achieved'] else 'BELUM'}")
    print(f"TFLite float32    : {actual_mib:.3f} MiB")
    print(f"Batas <10 MiB     : {'LOLOS' if within_size_limit else 'TIDAK LOLOS'}")
    print(f"MODEL PEMENANG    : {winner_dir}")
    print(f"ZIP SIAP DOWNLOAD : {run_dir / 'MODEL_PEMENANG_SIAP_DOWNLOAD.zip'}")
    print(f"SEMUA HASIL       : {run_dir}")
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
    return run_large_multiseed_experiment(
        output_base_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        reference_model_path=args.reference_model,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
