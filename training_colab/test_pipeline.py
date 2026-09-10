"""Fast contract tests that do not require TensorFlow or the Kaggle dataset."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from training_colab.training_pipeline import (
    CLASS_NAMES,
    LABEL_TO_INDEX,
    ORIGINAL_LABELS,
    ExperimentConfig,
    load_sign_mnist_csv,
    remap_original_labels,
    select_best_experiment,
    stratified_train_validation_split,
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


if __name__ == "__main__":
    unittest.main()
