import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.utils.training_artifacts import (
    EpochMetrics,
    metrics_to_dataframe,
    save_training_plots,
    write_metrics_csv,
)


class TestTrainingArtifacts(unittest.TestCase):
    def test_metrics_csv_and_plots_are_written(self):
        history = [
            EpochMetrics(
                epoch=1,
                train_loss=1.2,
                val_loss=1.0,
                val_cer=0.3,
                learning_rate=2e-4,
                epoch_time_sec=12.5,
                augment_enabled=False,
                blank_logit_bias=2.0,
                grad_accum_steps=4,
                amp_enabled=True,
            ),
            EpochMetrics(
                epoch=2,
                train_loss=0.8,
                val_loss=0.7,
                val_cer=0.2,
                learning_rate=2e-4,
                epoch_time_sec=11.8,
                augment_enabled=True,
                blank_logit_bias=2.0,
                grad_accum_steps=4,
                amp_enabled=True,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = write_metrics_csv(history, Path(tmpdir) / "metrics.csv")
            saved_plots = save_training_plots(history, tmpdir)

            dataframe = pd.read_csv(csv_path)
            self.assertEqual(list(dataframe["epoch"]), [1, 2])
            self.assertEqual(len(saved_plots), 3)
            for path in saved_plots:
                self.assertTrue(Path(path).exists())

    def test_metrics_dataframe_adds_cer_percent(self):
        dataframe = metrics_to_dataframe(
            [
                EpochMetrics(
                    epoch=1,
                    train_loss=1.0,
                    val_loss=0.9,
                    val_cer=0.25,
                    learning_rate=1e-3,
                    epoch_time_sec=5.0,
                    augment_enabled=False,
                    blank_logit_bias=2.0,
                    grad_accum_steps=1,
                    amp_enabled=False,
                )
            ]
        )
        self.assertAlmostEqual(float(dataframe.loc[0, "val_cer_percent"]), 25.0)


if __name__ == "__main__":
    unittest.main()
