"""Fast contract tests that do not require TensorFlow or the Kaggle dataset."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from training_colab.training_pipeline import (
    CLASS_NAMES,
    LABEL_TO_INDEX,
    ORIGINAL_LABELS,
    ExperimentConfig,
    compare_seed_runs,
    load_sign_mnist_csv,
    remap_original_labels,
    resolve_dataset_paths,
    select_best_experiment,
    stratified_train_validation_split,
)
from training_colab.experimental_high_accuracy import (
    HighAccuracyCandidate,
    select_high_accuracy_candidate,
)
from training_colab.experimental_large_multiseed import (
    select_deployment_members,
    select_multi_seed_winner,
)
from training_colab.experimental_three_seed_refit import (
    choose_experimental_stage,
)


class LabelContractTest(unittest.TestCase):
    def test_24_class_order_skips_j_and_z(self) -> None:
        self.assertEqual(len(CLASS_NAMES), 24)
        self.assertEqual(CLASS_NAMES[8], "I")
        self.assertEqual(CLASS_NAMES[9], "K")
        self.assertNotIn("J", CLASS_NAMES)
        self.assertNotIn("Z", CLASS_NAMES)

    def test_original_labels_are_remapped_contiguously(self) -> None:
        original = np.asarray(ORIGINAL_LABELS)
        mapped = remap_original_labels(original)
        np.testing.assert_array_equal(mapped, np.arange(24))
        self.assertEqual(LABEL_TO_INDEX[10], 9)

    def test_j_and_z_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            remap_original_labels(np.asarray([0, 9, 25]))


class DatasetContractTest(unittest.TestCase):
    def test_existing_drive_cache_skips_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            train_path = root / "sign_mnist_train.csv"
            test_path = root / "sign_mnist_test.csv"
            train_path.touch()
            test_path.touch()
            with patch(
                "training_colab.training_pipeline.download_dataset",
                side_effect=AssertionError("download seharusnya dilewati"),
            ):
                resolved_train, resolved_test = resolve_dataset_paths(root)
        self.assertEqual(resolved_train.name, "sign_mnist_train.csv")
        self.assertEqual(resolved_test.name, "sign_mnist_test.csv")

    def test_csv_load_and_stratified_split(self) -> None:
        rows_per_class = 5
        labels = np.repeat(np.asarray(ORIGINAL_LABELS), rows_per_class)
        pixels = np.zeros((len(labels), 784), dtype=np.uint8)
        for index in range(len(labels)):
            pixels[index, index % 784] = index % 256

        columns = [f"pixel{index}" for index in range(1, 785)]
        frame = pd.DataFrame(pixels, columns=columns)
        frame.insert(0, "label", labels)

        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "sign_mnist_train.csv"
            frame.to_csv(csv_path, index=False)
            x, y, report = load_sign_mnist_csv(csv_path, "test fixture")

        self.assertEqual(x.shape, (120, 28, 28, 1))
        self.assertEqual(y.shape, (120,))
        self.assertEqual(report["rows"], 120)
        x_train, x_validation, y_train, y_validation = (
            stratified_train_validation_split(x, y, 0.2, 42)
        )
        self.assertEqual(len(x_train), 96)
        self.assertEqual(len(x_validation), 24)
        self.assertEqual(set(y_train), set(range(24)))
        self.assertEqual(set(y_validation), set(range(24)))


class SelectionContractTest(unittest.TestCase):
    def test_selection_uses_validation_loss(self) -> None:
        results = [
            {
                "experiment": {"name": "higher_accuracy"},
                "best_val_loss": 0.20,
                "best_val_accuracy": 0.99,
            },
            {
                "experiment": {"name": "lower_loss"},
                "best_val_loss": 0.10,
                "best_val_accuracy": 0.98,
            },
        ]
        selected = select_best_experiment(results)
        self.assertEqual(selected["experiment"]["name"], "lower_loss")

    def test_experiment_defaults_are_safe(self) -> None:
        experiment = ExperimentConfig(name="baseline")
        self.assertFalse(experiment.use_mild_augmentation)
        self.assertEqual(experiment.dropout, 0.3)


class MultiSeedComparisonTest(unittest.TestCase):
    def _write_fake_run(
        self,
        root: Path,
        seed: int,
        validation_loss: float,
        predicted: list[int],
    ) -> Path:
        run_dir = root / f"seed_{seed}"
        run_dir.mkdir()
        true_labels = [0, 1, 2, 3]
        correct = np.asarray(predicted) == np.asarray(true_labels)
        report = {
            label: {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 1,
            }
            for label in CLASS_NAMES
        }
        payloads = {
            "run_config.json": {"training": {"seed": seed}},
            "selected_experiment.json": {
                "experiment": {"name": "baseline"},
                "best_epoch": 10,
                "best_val_loss": validation_loss,
                "best_val_accuracy": 1.0,
            },
            "evaluation.json": {
                "accuracy": float(correct.mean()),
                "macro_precision": 0.9,
                "macro_recall": 0.9,
                "macro_f1": 0.9,
                "classification_report": report,
            },
            "tflite_parity.json": [
                {
                    "variant": "float32",
                    "label_agreement": 1.0,
                    "maximum_absolute_score_difference": 1e-6,
                }
            ],
        }
        for filename, payload in payloads.items():
            (run_dir / filename).write_text(
                json.dumps(payload), encoding="utf-8"
            )
        pd.DataFrame(
            {
                "true_index": true_labels,
                "predicted_index": predicted,
                "correct": correct,
            }
        ).to_csv(run_dir / "test_predictions.csv", index=False)
        (run_dir / "model_float32.tflite").write_bytes(b"fake-tflite")
        return run_dir

    def test_compare_seed_runs_selects_by_validation_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            run_42 = self._write_fake_run(root, 42, 0.2, [0, 1, 2, 0])
            run_123 = self._write_fake_run(root, 123, 0.1, [0, 1, 0, 3])
            output_dir = root / "comparison"
            result = compare_seed_runs(
                {42: run_42, 123: run_123},
                output_dir=output_dir,
            )
            self.assertEqual(result["recommended_seed"], 123)
            self.assertEqual(len(result["pairwise"]), 1)
            self.assertTrue((output_dir / "multi_seed_summary.json").exists())
            self.assertTrue((output_dir / "multi_seed_comparison.png").exists())


class HighAccuracyExperimentTest(unittest.TestCase):
    def test_candidate_selection_uses_validation_loss(self) -> None:
        candidate_a = HighAccuracyCandidate(
            name="a",
            filters=(32, 64, 128),
            dense_units=256,
            learning_rate=0.001,
            block_dropouts=(0.0, 0.1, 0.2),
            dense_dropout=0.2,
        )
        candidate_b = HighAccuracyCandidate(
            name="b",
            filters=(64, 128, 256),
            dense_units=512,
            learning_rate=0.0005,
            block_dropouts=(0.0, 0.05, 0.1),
            dense_dropout=0.15,
        )
        selected = select_high_accuracy_candidate(
            [
                {
                    "candidate": {"name": candidate_a.name},
                    "best_val_loss": 0.01,
                    "best_val_accuracy": 1.0,
                },
                {
                    "candidate": {"name": candidate_b.name},
                    "best_val_loss": 0.005,
                    "best_val_accuracy": 0.999,
                },
            ]
        )
        self.assertEqual(selected["candidate"]["name"], "b")


class LargeMultiSeedExperimentTest(unittest.TestCase):
    @staticmethod
    def _result(
        candidate: str,
        seed: int,
        parameters: int,
        val_loss: float,
        val_accuracy: float,
    ) -> dict:
        return {
            "candidate": {"name": candidate},
            "seed": seed,
            "parameter_count": parameters,
            "best_epoch": 10,
            "epochs_completed": 15,
            "best_val_loss": val_loss,
            "best_val_accuracy": val_accuracy,
            "checkpoint_path": f"/{candidate}/seed_{seed}/best_model.keras",
        }

    def test_architecture_selection_uses_mean_validation_loss(self) -> None:
        results = [
            self._result("stable", 42, 500_000, 0.02, 0.99),
            self._result("stable", 123, 500_000, 0.02, 0.99),
            self._result("lucky_once", 42, 500_000, 0.001, 1.0),
            self._result("lucky_once", 123, 500_000, 0.10, 0.98),
        ]
        selected, _, aggregate = select_multi_seed_winner(results)
        self.assertEqual(selected["candidate"]["name"], "stable")
        self.assertEqual(aggregate.iloc[0]["candidate"], "stable")

    def test_ensemble_member_count_respects_size_limit(self) -> None:
        parameters = 600_000  # Roughly 2.29 MiB per float32 member.
        results = [
            self._result("small", seed, parameters, 0.01 + index, 0.99)
            for index, seed in enumerate((42, 123, 2026))
        ]
        members = select_deployment_members(results, "small", 10.0)
        self.assertEqual(len(members), 3)
        self.assertEqual(members[0]["seed"], 42)

    def test_large_candidate_stays_single_model(self) -> None:
        parameters = 2_340_952  # Roughly 8.93 MiB per float32 member.
        results = [
            self._result("large", seed, parameters, 0.01, 0.99)
            for seed in (42, 123, 2026)
        ]
        members = select_deployment_members(results, "large", 10.0)
        self.assertEqual(len(members), 1)

    def test_generated_notebook_code_cells_have_valid_python(self) -> None:
        notebook_path = (
            Path(__file__).parent
            / "LatihIsyarat_Experimental_Large_MultiSeed_Colab.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            python_lines = [
                line
                for line in source.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            ]
            compile(
                "\n".join(python_lines),
                f"notebook_cell_{index}",
                "exec",
            )


class ThreeSeedRefitExperimentTest(unittest.TestCase):
    def test_stage_selection_prefers_test_accuracy_then_macro_f1(self) -> None:
        selected = choose_experimental_stage(
            [
                {
                    "stage": "before",
                    "evaluation": {"accuracy": 0.990, "macro_f1": 0.995},
                },
                {
                    "stage": "after",
                    "evaluation": {"accuracy": 0.991, "macro_f1": 0.990},
                },
            ]
        )
        self.assertEqual(selected["stage"], "after")

    def test_refit_notebook_code_cells_have_valid_python(self) -> None:
        notebook_path = (
            Path(__file__).parent
            / "LatihIsyarat_ThreeSeed_FullData_Refit_Colab.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            python_lines = [
                line
                for line in source.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            ]
            compile(
                "\n".join(python_lines),
                f"refit_notebook_cell_{index}",
                "exec",
            )


if __name__ == "__main__":
    unittest.main()
